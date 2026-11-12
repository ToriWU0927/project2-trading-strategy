"""Part A - features, labels and the modelling panel (A6, A7, A9, A10).

    python src/features.py      # reads data/interim/clean.parquet, writes data/processed/panel.parquet
                                # and docs/data_dictionary.md

Timing (config.json["timing"]):
    features on row t use information up to the close of day t only
    the position decided at close t is filled at open t+1 and held to open t+2
    label  y_ret_t   = Open_{t+2} / Open_{t+1} - 1
           y_exret_t = y_ret_t - rf_daily_t   (rf_daily = T-bill BEY x days held / 365)

Features are NOT standardised here; B fits any scaling inside each training fold.

Interface for B / C / D:
    build_panel(cfg)        -> DataFrame (features + labels + prices)
    get_feature_groups()    -> {"G1a": [...], ..., "M1": [...], "M2": [...], "M3": [...]}
    period_of(index, cfg)   -> Series of "warmup" / "dev" / "test" / "after"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import ROOT, load_config

GROUPS = {
    "G1a": "trend / momentum (SPY)",
    "G1b": "short-term reversal (SPY)",
    "G1c": "volatility / volume (SPY)",
    "G2": "VIX sentiment",
    "G3": "rates / credit (FRED)",
}

# name -> dict(group, definition, unit, window, source, rationale)
FEATURE_META: dict[str, dict] = {}


def _meta(name, group, definition, unit, window, source, rationale):
    FEATURE_META[name] = dict(group=group, definition=definition, unit=unit,
                              window=window, source=source, rationale=rationale)


# every non-feature column of the panel
OTHER_META: dict[str, dict] = {
    "period": dict(definition="'warmup' / 'dev' / 'embargo' / 'test' from config.json['splits']; "
                              "'embargo' = last dev rows whose label uses test-period opens",
                   unit="label", known_at="-", use="row filter for B / C"),
    "y_ret": dict(definition="Open_(t+2) / Open_(t+1) - 1 (SPY adjusted)", unit="decimal",
                  known_at="open of t+2", use="C: strategy and buy-and-hold return"),
    "rf_annual": dict(definition="DTB3 (value dated < t) converted from discount basis to "
                                 "bond-equivalent yield: 365 d / (360 - 91 d)",
                      unit="decimal per year", known_at="close of t",
                      use="rate behind rf_daily"),
    "rf_daily": dict(definition="rf_annual x calendar days from open t+1 to open t+2 / 365 "
                                "(3 days over a weekend); NaN on the last rows like the label",
                     unit="decimal per holding period", known_at="close of t "
                     "(trading calendar is published in advance)",
                     use="C: cash return; risk-free for y_exret"),
    "y_exret": dict(definition="y_ret - rf_daily", unit="decimal",
                    known_at="open of t+2", use="B: main regression label"),
    "y_up": dict(definition="1 if y_exret > 0 else 0 (NaN if y_exret is NaN)", unit="0/1",
                 known_at="open of t+2", use="B: optional classification label"),
    "px_open": dict(definition="SPY open, split + dividend adjusted", unit="USD (adjusted)",
                    known_at="open of t", use="C: prices / checks"),
    "px_close": dict(definition="SPY close, split + dividend adjusted", unit="USD (adjusted)",
                     known_at="close of t", use="C: prices / checks"),
    "ret_cc": dict(definition="px_close_t / px_close_(t-1) - 1 (same values as feature rev_ret_1)",
                   unit="decimal",
                   known_at="close of t", use="C: e.g. volatility-scaled benchmark"),
}


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder RSI (0-100); exponential smoothing is causal."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(clean: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    c, o, v = clean["spy_close"], clean["spy_open"], clean["spy_volume"]
    logret = np.log(c).diff()
    X = pd.DataFrame(index=clean.index)

    # G1a - trend / time-series momentum
    for w in f["mom_return_windows"]:
        X[f"mom_ret_{w}"] = c / c.shift(w) - 1
        _meta(f"mom_ret_{w}", "G1a", f"Close_t / Close_(t-{w}) - 1", "decimal", w,
              "SPY adj close", "time-series momentum: past trend tends to persist")
    for w in f["mom_ma_windows"]:
        X[f"mom_ma_{w}"] = c / c.rolling(w).mean() - 1
        _meta(f"mom_ma_{w}", "G1a", f"Close_t / MA{w}(Close)_t - 1", "decimal", w,
              "SPY adj close", "distance from a slow moving average = trend strength")

    # G1b - short-term reversal
    for w in f["rev_return_windows"]:
        X[f"rev_ret_{w}"] = c / c.shift(w) - 1
        _meta(f"rev_ret_{w}", "G1b", f"Close_t / Close_(t-{w}) - 1", "decimal", w,
              "SPY adj close", "short-horizon overreaction tends to reverse")
    w = f["rev_ma_window"]
    X[f"rev_ma_{w}"] = c / c.rolling(w).mean() - 1
    _meta(f"rev_ma_{w}", "G1b", f"Close_t / MA{w}(Close)_t - 1", "decimal", w,
          "SPY adj close", "stretch from a fast moving average, mean reversion")
    w = f["rsi_window"]
    X[f"rev_rsi_{w}"] = _rsi(c, w)
    _meta(f"rev_rsi_{w}", "G1b", f"Wilder RSI({w}) of Close", "0-100", w,
          "SPY adj close", "overbought / oversold oscillator")
    X["rev_overnight"] = o / c.shift(1) - 1
    _meta("rev_overnight", "G1b", "Open_t / Close_(t-1) - 1", "decimal", 1,
          "SPY adj open/close", "overnight gap; overnight and intraday returns behave differently")

    # G1c - volatility and volume
    for w in f["vol_windows"]:
        X[f"vol_rv_{w}"] = logret.rolling(w).std() * np.sqrt(252)
        _meta(f"vol_rv_{w}", "G1c", f"std of daily log returns over {w} days x sqrt(252)",
              "decimal, annualised", w, "SPY adj close",
              "risk level; volatility-managed portfolios (Moreira & Muir 2017)")
    w = f["volume_ma_window"]
    X["vol_volume"] = np.log(v / v.rolling(w).mean())
    _meta("vol_volume", "G1c", f"log(Volume_t / MA{w}(Volume)_t)", "log ratio", w,
          "SPY volume", "abnormal trading activity / attention")

    # G2 - VIX sentiment
    vix = clean["vix_close"]
    X["vix_level"] = vix
    _meta("vix_level", "G2", "VIX close_t", "index points", 1, "Yahoo ^VIX",
          "option-implied fear; high VIX = high expected risk premium")
    w = f["vix_change_window"]
    X[f"vix_chg_{w}"] = np.log(vix / vix.shift(w))
    _meta(f"vix_chg_{w}", "G2", f"log(VIX_t / VIX_(t-{w}))", "log ratio", w, "Yahoo ^VIX",
          "change in fear")
    rv_w = f["vol_windows"][0]
    X["vix_vrp"] = vix / 100 - X[f"vol_rv_{rv_w}"]
    _meta("vix_vrp", "G2", f"VIX_t/100 - vol_rv_{rv_w}_t", "decimal, annualised vol", rv_w,
          "Yahoo ^VIX, SPY", "variance risk premium proxy: implied minus realised volatility")
    w = f["vix_rank_window"]
    X[f"vix_rank_{w}"] = vix.rolling(w).rank(pct=True)
    _meta(f"vix_rank_{w}", "G2", f"percentile of VIX_t within the last {w} days", "(0, 1]", w,
          "Yahoo ^VIX", "VIX relative to its own recent history (regime)")

    # G3 - rates and credit (FRED, value dated < t, see data.fred_asof)
    w = f["mac_change_window"]
    names = {"DTB3": ("tbill", "3-month T-bill rate", "short rate / monetary stance"),
             "T10Y3M": ("term", "10y minus 3m Treasury spread",
                        "term spread predicts returns (Goyal & Welch 2008)"),
             "BAA10Y": ("credit", "Moody's Baa minus 10y Treasury",
                        "credit spread = default risk premium")}
    for sid, (short, desc, why) in names.items():
        X[f"mac_{short}"] = clean[sid]
        _meta(f"mac_{short}", "G3", f"{desc}, latest value dated before t", "percent", 1,
              f"FRED {sid}", why)
        X[f"mac_{short}_chg_{w}"] = clean[sid] - clean[sid].shift(w)
        _meta(f"mac_{short}_chg_{w}", "G3", f"{sid}_t - {sid}_(t-{w})", "percentage points", w,
              f"FRED {sid}", f"recent change in the {short} rate / spread")
    return X


def build_labels(clean: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h = cfg["timing"]["label_horizon_days"]
    o, c = clean["spy_open"], clean["spy_close"]
    L = pd.DataFrame(index=clean.index)
    L["y_ret"] = o.shift(-(1 + h)) / o.shift(-1) - 1
    # DTB3 is quoted on a discount basis; convert to a bond-equivalent yield and
    # accrue it over the calendar days actually held (open t+1 -> open t+1+h)
    d = clean["DTB3"] / 100
    L["rf_annual"] = 365 * d / (360 - 91 * d)
    dates = pd.Series(clean.index, index=clean.index)
    days = (dates.shift(-(1 + h)) - dates.shift(-1)).dt.days
    L["rf_daily"] = L["rf_annual"] * days / 365
    L["y_exret"] = L["y_ret"] - L["rf_daily"]
    L["y_up"] = (L["y_exret"] > 0).astype(float).where(L["y_exret"].notna())
    # prices for C (adjusted, total-return)
    L["px_open"] = o
    L["px_close"] = c
    L["ret_cc"] = c.pct_change()
    return L


def get_feature_groups(columns=None) -> dict[str, list[str]]:
    if not FEATURE_META:
        build_features(_demo_clean(), load_config())
    cols = list(FEATURE_META) if columns is None else [c for c in columns if c in FEATURE_META]
    g = {k: [c for c in cols if FEATURE_META[c]["group"] == k] for k in GROUPS}
    g["M1"] = g["G1a"] + g["G1b"] + g["G1c"]
    g["M2"] = g["M1"] + g["G2"]
    g["M3"] = g["M2"] + g["G3"]
    return g


def _demo_clean() -> pd.DataFrame:
    """Tiny dummy frame, only used to populate FEATURE_META without reading data."""
    idx = pd.bdate_range("2000-01-03", periods=300)
    cols = ["spy_open", "spy_close", "spy_volume", "vix_close", "DTB3", "T10Y3M", "BAA10Y"]
    return pd.DataFrame(1.0, index=idx, columns=cols)


def period_of(index: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    """'warmup' / 'dev' / 'embargo' / 'test' / 'after' for every row.

    The last `embargo_days` development rows are labelled 'embargo': their
    labels use opens from the test period, so validation must not score them.
    They are still fine as training rows once the test period has started.
    """
    s = cfg["splits"]
    h = cfg["timing"]["label_horizon_days"]
    if s["embargo_days"] < h + 1:
        raise ValueError(f"splits.embargo_days must be >= label_horizon_days + 1 = {h + 1}")
    p = pd.Series("after", index=index)
    p[index <= pd.Timestamp(s["test_end"])] = "test"
    p[index < pd.Timestamp(s["test_start"])] = "dev"
    p[index < pd.Timestamp(s["dev_start"])] = "warmup"
    dev_pos = np.flatnonzero((p == "dev").to_numpy())
    p.iloc[dev_pos[-s["embargo_days"]:]] = "embargo"
    return p


def build_panel(cfg: dict | None = None, clean: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    if clean is None:
        clean = pd.read_parquet(ROOT / cfg["data"]["interim_dir"] / "clean.parquet")
    X = build_features(clean, cfg)
    L = build_labels(clean, cfg)
    panel = pd.concat([X, L], axis=1)
    first_valid = X.dropna().index[0]          # end of the warm-up window
    panel = panel.loc[first_valid:]
    panel.insert(0, "period", period_of(panel.index, cfg))
    panel.index.name = "date"
    return panel


def write_feature_table(path: Path) -> None:
    rows = [{"column": k, **m} for k, m in FEATURE_META.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_data_dictionary(panel: pd.DataFrame, path: Path) -> None:
    """docs/data_dictionary.md, generated so it never drifts from the code."""
    lines = [
        "# Data dictionary - `data/processed/panel.parquet`",
        "",
        "Generated by `python src/features.py`; do not edit by hand.",
        "",
        f"- Index: `date` (SPY trading day t), {len(panel)} rows, "
        f"{panel.index[0].date()} -> {panel.index[-1].date()}",
        "- Every feature on row t is known at the **close of day t**; the trade is filled at "
        "the **open of t+1** and held to the **open of t+2**.",
        "- Features are unscaled. FRED inputs use the latest value dated strictly before t.",
        "",
        "## Features",
        "",
        "| column | group | definition | unit | lookback (days) | source | rationale |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, m in FEATURE_META.items():
        lines.append(f"| `{k}` | {m['group']} | {m['definition']} | {m['unit']} | {m['window']} "
                     f"| {m['source']} | {m['rationale']} |")
    lines += ["", "Groups: " + "; ".join(f"**{k}** = {v}" for k, v in GROUPS.items()) + ".",
              "Ablation sets: **M1** = G1a+G1b+G1c, **M2** = M1+G2, **M3** = M2+G3.", "",
              "## Labels, prices and helper columns", "",
              "| column | definition | unit | known at | used by |", "|---|---|---|---|---|"]
    for k, m in OTHER_META.items():
        lines.append(f"| `{k}` | {m['definition']} | {m['unit']} | {m['known_at']} | {m['use']} |")
    nan_label = int(panel["y_ret"].isna().sum())
    lines += ["", "## Missing values", "",
              f"- Features: none (rows before the end of the "
              f"{max(int(m['window']) for m in FEATURE_META.values())}-day warm-up are dropped).",
              f"- `y_ret`, `y_exret`, `y_up`: last {nan_label} rows are NaN (no future open yet).",
              ""]
    path.write_text("\n".join(lines))


def main() -> pd.DataFrame:
    cfg = load_config()
    panel = build_panel(cfg)
    out = ROOT / cfg["data"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "panel.parquet")
    write_feature_table(out / "feature_table.csv")
    groups = get_feature_groups(panel.columns)
    (out / "feature_groups.json").write_text(json.dumps(groups, indent=2))
    write_data_dictionary(panel, ROOT / "docs" / "data_dictionary.md")
    feats = groups["M3"]
    n_nan = int(panel.loc[panel["period"] != "warmup", feats].isna().sum().sum())
    if n_nan:  # e.g. a refreshed download with a long FRED gap
        raise ValueError(f"{n_nan} missing feature values after warm-up; check clean_report.json")
    print(f"panel -> {(out / 'panel.parquet').relative_to(ROOT)}  {panel.shape[0]} rows x "
          f"{panel.shape[1]} cols; {len(feats)} features; "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}")
    print("rows by period:", panel["period"].value_counts().to_dict())
    print("feature NaN after warm-up:", int(panel[feats].isna().sum().sum()),
          "| label NaN:", int(panel["y_ret"].isna().sum()), "(last rows, no future open)")
    return panel


if __name__ == "__main__":
    main()
