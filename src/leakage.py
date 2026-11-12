"""Part A - leakage tests and the time-alignment example (A8).

    python src/leakage.py

Every test re-runs the whole Part A pipeline (raw inputs -> clean -> features
-> labels) on modified raw inputs, so the FRED publication lag is tested too.

  1. Truncation  : cut every raw input at date t (FRED keeps the value dated t
                   itself). Features and rf on row t must equal the full-sample ones
                   (NaN on one side only counts as a difference).
                   Negative control: the same test must FAIL when a shift(-1) or a
                   centred-window feature is added.
  1b. Adjustment : rebuild adjusted prices from raw close + dividends known at t
                   only; features on row t must not change (vendor adjustment uses
                   later dividends, but features are scale-free ratios).
  2. Future noise: scramble every raw input after t (and FRED dated >= t).
                   Features on rows <= t must not move; labels on rows <= t-2 must
                   not move; the label on row t must move (test is sensitive).
  3. One open    : change only Open_s. Exactly the labels of rows s-2 and s-1
                   change, and among features only rev_overnight on row s.

Writes results/partA/leakage_checks.json and docs/alignment_example.md.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from data import ROOT, build_clean, load_config, load_inputs
from features import FEATURE_META, build_features, build_labels

TOL = 1e-12
SPECIAL_DATES = ["1994-02-01", "2008-10-13", "2015-01-02", "2020-03-16",
                 "2025-10-14", "2026-04-06"]


def run(inputs: dict, cfg: dict, extra=None):
    """Whole pipeline on `inputs`; `extra(clean)` may add test-only feature columns."""
    clean, _ = build_clean(cfg, inputs)
    X = build_features(clean, cfg)
    if extra is not None:
        X = X.join(extra(clean))
    return X, build_labels(clean, cfg), clean


def truncate(inputs: dict, t: pd.Timestamp) -> dict:
    return {"spy": inputs["spy"].loc[:t], "spy_raw": inputs["spy_raw"].loc[:t],
            "vix": inputs["vix"].loc[:t],
            "fred": {k: v.loc[:t] for k, v in inputs["fred"].items()}}


def scramble_after(inputs: dict, t: pd.Timestamp, rng) -> dict:
    out = {k: v.copy() for k, v in inputs.items() if k != "fred"}
    for k in ("spy", "spy_raw", "vix"):
        df = out[k]
        m = df.index > t
        for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
            if c in df.columns:
                df[c] = df[c].astype(float)
                df.loc[m, c] = df.loc[m, c] * rng.uniform(0.5, 1.5, m.sum())
    out["fred"] = {}
    for k, s in inputs["fred"].items():
        s = s.copy()
        m = s.index >= t                       # includes the value dated t itself
        s[m] = s[m] + rng.normal(0, 1, m.sum())
        out["fred"][k] = s
    return out


def max_diff(a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series) -> float:
    """Largest absolute difference; NaN on exactly one side counts as infinite.

    A leak through shift(-1) shows up as NaN on the truncated side, so a plain
    nanmax would silently ignore it.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0:
        return 0.0
    if (np.isnan(a) != np.isnan(b)).any():
        return float("inf")
    d = np.abs(a - b)
    return float(np.nanmax(d)) if (~np.isnan(d)).any() else 0.0


def test_truncation(inputs, cfg, dates, extra=None) -> dict:
    X, L, _ = run(inputs, cfg, extra)
    worst, worst_rf = 0.0, 0.0
    for t in dates:
        Xt, Lt, _ = run(truncate(inputs, t), cfg, extra)
        worst = max(worst, max_diff(Xt.loc[t], X.loc[t]))
        # rf_annual is the rate known at t; rf_daily also needs the next two
        # trading dates, which come from the (publicly known) exchange calendar
        worst_rf = max(worst_rf, max_diff(Lt.loc[[t], "rf_annual"], L.loc[[t], "rf_annual"]))
    return {"dates_tested": len(dates), "max_abs_diff_features": worst,
            "max_abs_diff_rf_annual": worst_rf,
            "passed": bool(worst <= TOL and worst_rf <= TOL)}


LEAKY = {
    "shift(-1): Close_(t+1) / Close_t - 1":
        lambda c: c["spy_close"].shift(-1) / c["spy_close"] - 1,
    "centred MA20: Close_t / centred MA20 - 1 (uses 9 future closes)":
        lambda c: c["spy_close"] / c["spy_close"].rolling(20, center=True).mean() - 1,
}


def test_negative_control(inputs, cfg, dates) -> dict:
    """test_truncation itself must FAIL once a deliberately leaky feature is added."""
    out = {}
    for name, f in LEAKY.items():
        r = test_truncation(inputs, cfg, dates, extra=lambda c, f=f: f(c).rename("leak"))
        out[name] = {"truncation_test_passed": r["passed"],
                     "max_abs_diff_features": r["max_abs_diff_features"]}
    out["passed"] = bool(all(not v["truncation_test_passed"] for v in out.values()))
    return out


def adjust_with_dividends_upto(raw: pd.DataFrame) -> pd.DataFrame:
    """Yahoo-style back-adjustment using only the dividends present in `raw`."""
    from checks import future_product
    p, d = raw["Close"], raw["Dividends"].fillna(0.0)
    g = pd.Series(np.where(d > 0, 1.0 - d / p.shift(1), 1.0), index=p.index).fillna(1.0)
    f = future_product(g)
    adj = raw[["Open", "High", "Low", "Close"]].mul(f, axis=0)
    adj["Volume"] = raw["Volume"]
    return adj


def test_adjustment_lookahead(inputs, cfg, X, dates, tol: float = 2e-3) -> dict:
    """Vendor prices are back-adjusted with dividends paid AFTER t. Rebuild the
    adjusted series from raw close + dividends known at t only and check the
    features on row t do not change (beyond float32 storage noise)."""
    worst = pd.Series(0.0, index=X.columns)
    for t in dates:
        inp = truncate(inputs, t)
        inp["spy"] = adjust_with_dividends_upto(inp["spy_raw"])
        Xt, _, _ = run(inp, cfg)
        worst = np.maximum(worst, (Xt.loc[t] - X.loc[t]).abs())
    return {"dates_tested": len(dates), "tolerance": tol,
            "max_abs_diff_any_feature": float(worst.max()),
            "worst_feature": str(worst.idxmax()),
            "max_abs_diff_excluding_rsi": float(worst.drop("rev_rsi_14").max()),
            "passed": bool(worst.max() <= tol)}


def test_future_noise(inputs, cfg, X, L, dates, rng) -> dict:
    res = {"dates_tested": len(dates), "max_abs_diff_features_upto_t": 0.0,
           "max_abs_diff_labels_upto_t_minus_2": 0.0, "label_t_changed_every_time": True}
    for t in dates:
        Xn, Ln, _ = run(scramble_after(inputs, t, rng), cfg)
        i = X.index.get_loc(t)
        res["max_abs_diff_features_upto_t"] = max(res["max_abs_diff_features_upto_t"],
                                                  max_diff(Xn.loc[:t], X.loc[:t]))
        safe = X.index[i - 2]
        res["max_abs_diff_labels_upto_t_minus_2"] = max(
            res["max_abs_diff_labels_upto_t_minus_2"],
            max_diff(Ln.loc[:safe, ["y_ret", "y_exret", "rf_daily"]],
                     L.loc[:safe, ["y_ret", "y_exret", "rf_daily"]]))
        if abs(Ln.loc[t, "y_ret"] - L.loc[t, "y_ret"]) <= TOL:
            res["label_t_changed_every_time"] = False
    res["passed"] = bool(res["max_abs_diff_features_upto_t"] <= TOL
                         and res["max_abs_diff_labels_upto_t_minus_2"] <= TOL
                         and res["label_t_changed_every_time"])
    return res


def test_one_open(inputs, cfg, X, L, dates) -> dict:
    res = {"dates_tested": len(dates), "label_rows_changed_always_s-2_s-1": True,
           "feature_changes_only_rev_overnight_on_s": True}
    idx = X.index
    for s in dates:
        mod = dict(inputs)
        mod["spy"] = inputs["spy"].copy()
        mod["spy"].loc[s, "Open"] *= 1.01
        Xn, Ln, _ = run(mod, cfg)
        changed = Ln.index[(Ln["y_ret"] - L["y_ret"]).abs() > TOL]
        i = idx.get_loc(s)
        if list(changed) != [idx[i - 2], idx[i - 1]]:
            res["label_rows_changed_always_s-2_s-1"] = False
        dX = (Xn - X).abs() > TOL
        cells = [(r, c) for r, c in zip(*np.nonzero(dX.to_numpy()))]
        if [(idx[r], X.columns[c]) for r, c in cells] != [(s, "rev_overnight")]:
            res["feature_changes_only_rev_overnight_on_s"] = False
    res["passed"] = bool(res["label_rows_changed_always_s-2_s-1"]
                         and res["feature_changes_only_rev_overnight_on_s"])
    return res


# ---------------------------------------------------------------- alignment example

def _inputs_used(col: str, clean: pd.DataFrame, i: int) -> str:
    idx = clean.index

    def d(k):
        return str(idx[i + k].date())

    def src(sid, k=0):
        return str(pd.Timestamp(clean[f"{sid}_src_date"].iloc[i + k]).date())

    w = int(re.findall(r"_(\d+)$", col)[0]) if re.search(r"_(\d+)$", col) else None
    fred = {"tbill": "DTB3", "term": "T10Y3M", "credit": "BAA10Y"}
    if col.startswith(("mom_ret_", "rev_ret_")):
        return f"SPY close {d(-w)} and {d(0)}"
    if col.startswith(("mom_ma_", "rev_ma_")):
        return f"SPY closes {d(-w + 1)} ... {d(0)} ({w} closes)"
    if col.startswith("rev_rsi_"):
        return f"all SPY closes up to {d(0)} (Wilder smoothing)"
    if col == "rev_overnight":
        return f"SPY close {d(-1)}, SPY open {d(0)}"
    if col.startswith("vol_rv_"):
        return f"SPY closes {d(-w)} ... {d(0)}"
    if col == "vol_volume":
        return f"SPY volume {d(-19)} ... {d(0)}"
    if col == "vix_level":
        return f"VIX close {d(0)}"
    if col.startswith("vix_chg_"):
        return f"VIX close {d(-w)} and {d(0)}"
    if col == "vix_vrp":
        return f"VIX close {d(0)}; SPY closes {d(-20)} ... {d(0)}"
    if col.startswith("vix_rank_"):
        return f"VIX closes {d(-w + 1)} ... {d(0)}"
    if col.startswith("mac_"):
        sid = fred[col.split("_")[1]]
        if "_chg_" in col:
            return f"FRED {sid} dated {src(sid, -w)} and {src(sid)}"
        return f"FRED {sid} dated {src(sid)}"
    if col == "y_ret":
        return f"SPY open {d(1)} and {d(2)}"
    if col == "rf_daily":
        return (f"FRED DTB3 dated {src('DTB3')}; calendar days {d(1)} -> {d(2)} "
                f"(exchange calendar, known in advance)")
    if col == "y_exret":
        return f"SPY open {d(1)} and {d(2)}; FRED DTB3 dated {src('DTB3')}"
    if col == "px_open":
        return f"SPY open {d(0)}"
    if col == "px_close":
        return f"SPY close {d(0)}"
    if col == "ret_cc":
        return f"SPY close {d(-1)} and {d(0)}"
    return ""


def alignment_table(X, L, clean, t: pd.Timestamp) -> pd.DataFrame:
    i = clean.index.get_loc(t)
    rows = []
    for col in list(X.columns) + ["y_ret", "rf_daily", "y_exret", "px_open", "px_close", "ret_cc"]:
        v = X.loc[t, col] if col in X.columns else L.loc[t, col]
        kind = "feature" if col in X.columns else (
            "label" if col.startswith("y_") else "helper")
        rows.append({"column": col, "kind": kind,
                     "group": FEATURE_META.get(col, {}).get("group", "-"),
                     "value": float(v), "inputs used": _inputs_used(col, clean, i),
                     "all inputs <= t ?": "no (future, by design)" if kind == "label"
                     else "yes"})
    return pd.DataFrame(rows)


def write_alignment_md(tables: dict[str, pd.DataFrame], clean: pd.DataFrame, path) -> None:
    lines = ["# Time-alignment example", "",
             "Generated by `python src/leakage.py`. For a given decision date t the table "
             "lists every panel column and the exact input observations it is built from.",
             "Decision at the close of t, trade at the open of t+1, exit at the open of t+2.", ""]
    for t, tab in tables.items():
        ts = pd.Timestamp(t)
        i = clean.index.get_loc(ts)
        lines += [f"## t = {t}", "",
                  f"Previous trading day {clean.index[i - 1].date()}, "
                  f"next trading days {clean.index[i + 1].date()} (trade) and "
                  f"{clean.index[i + 2].date()} (exit).", "",
                  "| column | kind | group | value | inputs used | all inputs <= t ? |",
                  "|---|---|---|---|---|---|"]
        for r in tab.itertuples(index=False):
            lines.append(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]:.6g} | {r[4]} | {r[5]} |")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> dict:
    cfg = load_config()
    rng = np.random.default_rng(5205)
    inputs = load_inputs(cfg)
    X, L, clean = run(inputs, cfg)
    valid = X.dropna().index
    valid = valid[valid <= clean.index[-3]]   # need two future opens for label tests
    pick = list(rng.choice(valid, 30, replace=False)) + [pd.Timestamp(d) for d in SPECIAL_DATES]
    pick = sorted(set(pd.Timestamp(d) for d in pick))
    few = pick[::3]                            # the full-pipeline noise tests are slower

    res = {
        "truncation": test_truncation(inputs, cfg, pick),
        "truncation_negative_control": test_negative_control(inputs, cfg, few),
        "adjustment_lookahead": test_adjustment_lookahead(inputs, cfg, X, pick),
        "future_noise": test_future_noise(inputs, cfg, X, L, few, rng),
        "one_open": test_one_open(inputs, cfg, X, L, few),
        "embargo_note": ("row t's label uses Open_(t+1) and Open_(t+2). At decision date t the "
                         "latest trainable row is t-2, and a training fold that ends right before "
                         "validation day v must drop row v-1 (its label uses Open_(v+1)). "
                         f"config.splits.embargo_days = {cfg['splits']['embargo_days']} "
                         f"(must be >= label_horizon_days + 1 = "
                         f"{cfg['timing']['label_horizon_days'] + 1})."),
    }
    res["all_passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    out = ROOT / cfg["data"]["results_dir"] / "partA"
    out.mkdir(parents=True, exist_ok=True)
    (out / "leakage_checks.json").write_text(json.dumps(res, indent=2))

    tables = {t: alignment_table(X, L, clean, pd.Timestamp(t))
              for t in ("2020-03-16", "2025-10-14")}
    write_alignment_md(tables, clean, ROOT / "docs" / "alignment_example.md")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
