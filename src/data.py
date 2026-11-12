"""Part A - raw data download and cache.

Downloads every series listed in config.json["data"] and stores one parquet
file per series under data/raw/, plus a manifest describing what was fetched.

    python src/data.py            # download only what is missing
    python src/data.py --refresh  # re-download everything

Sources (lawful, free, no API key):
    Yahoo Finance via yfinance: SPY, ^VIX (prices), AAPL (split check only)
    FRED CSV endpoint:          DTB3, T10Y3M, BAA10Y
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# yfinance 1.7 still calls inplace=... internally, which pandas 3.1 flags; harmless.
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

ROOT = Path(__file__).resolve().parents[1]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
RETRIES = 3


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.json") as f:
        return json.load(f)


def raw_dir(cfg: dict) -> Path:
    d = ROOT / cfg["data"]["raw_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(ticker: str) -> str:
    """'^VIX' -> 'VIX' so it can be used in a file name."""
    return ticker.replace("^", "").replace("=", "_")


def _retry(fn, what: str):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # network errors, empty responses
            if attempt == RETRIES:
                raise RuntimeError(f"failed to download {what}: {e}") from e
            wait = 2 ** attempt
            print(f"  {what}: attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------- Yahoo

def _yf_download(ticker: str, start: str, end: str, adjusted: bool) -> pd.DataFrame:
    """One ticker from Yahoo. `end` is exclusive.

    adjusted=True : Open/High/Low/Close already split- and dividend-adjusted.
    adjusted=False: raw Open/High/Low/Close + vendor 'Adj Close' + Dividends
                    + Stock Splits, kept for the adjustment check (Appendix B).
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=adjusted,
                     actions=not adjusted, progress=False,
                     multi_level_index=False, threads=False)
    if df is None or df.empty:
        raise ValueError("empty response")
    if isinstance(df.columns, pd.MultiIndex):  # older yfinance versions
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.sort_index()


def fetch_yf(ticker: str, start: str, end: str, out: Path, refresh: bool) -> list[Path]:
    paths = []
    for kind, adjusted in (("adj", True), ("raw", False)):
        path = out / f"yf_{safe_name(ticker)}_{kind}.parquet"
        if path.exists() and not refresh:
            print(f"  cached  {path.name}")
        else:
            df = _retry(lambda: _yf_download(ticker, start, end, adjusted),
                        f"{ticker} ({kind})")
            df.to_parquet(path)
            print(f"  saved   {path.name}  {len(df)} rows "
                  f"{df.index[0].date()} -> {df.index[-1].date()}")
        paths.append(path)
    return paths


# ---------------------------------------------------------------- FRED

def _fred_download(series: str, start: str, end: str) -> pd.DataFrame:
    """One FRED series as a float column; FRED marks holidays with '.'."""
    r = requests.get(FRED_URL, params={"id": series, "cosd": start, "coed": end},
                     timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values=".")
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df[series] = pd.to_numeric(df[series], errors="coerce")
    if df.empty:
        raise ValueError("empty response")
    return df[[series]]


def fetch_fred(series: str, start: str, end: str, out: Path, refresh: bool) -> Path:
    path = out / f"fred_{series}.parquet"
    if path.exists() and not refresh:
        print(f"  cached  {path.name}")
    else:
        df = _retry(lambda: _fred_download(series, start, end), series)
        df.to_parquet(path)
        print(f"  saved   {path.name}  {len(df)} rows "
              f"{df.index[0].date()} -> {df.index[-1].date()}")
    return path


# ---------------------------------------------------------------- manifest

def describe(path: Path) -> dict:
    df = pd.read_parquet(path)
    return {
        "file": path.name,
        "saved_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                                .isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "columns": list(map(str, df.columns)),
        "missing_by_column": {str(c): int(df[c].isna().sum()) for c in df.columns},
    }


def write_manifest(cfg: dict, paths: list[Path], out: Path) -> Path:
    manifest = {
        "manifest_written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_window": [cfg["data"]["start"], cfg["data"]["end"]],
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "yfinance": yf.__version__,
            "requests": requests.__version__,
        },
        "files": [describe(p) for p in paths],
    }
    path = out / "_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def download_all(cfg: dict, refresh: bool = False) -> list[Path]:
    d = cfg["data"]
    out = raw_dir(cfg)
    paths: list[Path] = []

    print("Yahoo Finance")
    for t in d["yf_tickers"]:
        paths += fetch_yf(t, d["start"], d["end"], out, refresh)
    for t, (s, e) in d["verification_tickers"].items():
        paths += fetch_yf(t, s, e, out, refresh)

    print("FRED")
    for s in d["fred_series"]:
        paths.append(fetch_fred(s, d["start"], d["end"], out, refresh))

    m = write_manifest(cfg, paths, out)
    print(f"manifest -> {m.relative_to(ROOT)}")
    return paths


def load_raw(cfg: dict, name: str, kind: str | None = None) -> pd.DataFrame:
    """Read a cached raw file, e.g. load_raw(cfg, 'SPY', 'adj') or load_raw(cfg, 'DTB3')."""
    out = raw_dir(cfg)
    if kind is None:
        df = pd.read_parquet(out / f"fred_{name}.parquet")
    else:
        df = pd.read_parquet(out / f"yf_{safe_name(name)}_{kind}.parquet")
    # pandas 3 keeps whatever resolution each source had (ms / us); unify it
    df.index = df.index.astype("datetime64[ns]")
    return df


# ---------------------------------------------------------------- cleaning (A4)

def fred_asof(series: pd.Series, calendar: pd.DatetimeIndex, lag: int,
              ffill_limit: int) -> pd.DataFrame:
    """Value of a FRED series that is usable at the close of each trading day.

    For day t we take the latest valid observation dated strictly before the
    (lag-1)-th previous trading day, i.e. with lag=1 the latest value dated < t.
    FRED usually publishes day d's value on d+1, so this is conservative.
    Holidays ('.') are skipped, which acts as a forward fill. If the value used
    is more than `ffill_limit` business days older than a normal 1-day lag, it
    is set to NaN instead of being carried further.
    """
    obs = series.dropna().rename("value").rename_axis("src_date").reset_index()
    key = pd.Series(calendar, name="date")
    left = pd.DataFrame({"date": key, "key": key.shift(lag - 1)}).dropna(subset=["key"])
    m = pd.merge_asof(left, obs, left_on="key", right_on="src_date",
                      direction="backward", allow_exact_matches=False)
    stale = np.full(len(m), np.nan)
    ok = m["src_date"].notna()
    stale[ok.to_numpy()] = np.busday_count(
        m.loc[ok, "src_date"].to_numpy().astype("datetime64[D]"),
        m.loc[ok, "date"].to_numpy().astype("datetime64[D]")) - lag
    m["stale_bdays"] = stale
    m.loc[m["stale_bdays"] > ffill_limit, "value"] = np.nan
    return m.set_index("date").reindex(calendar)[["value", "src_date", "stale_bdays"]]


def load_inputs(cfg: dict) -> dict:
    """All raw inputs of the cleaning step, kept in memory (used by the leakage tests)."""
    d = cfg["data"]
    return {"spy": load_raw(cfg, d["trade_asset"], "adj"),
            "spy_raw": load_raw(cfg, d["trade_asset"], "raw"),
            "vix": load_raw(cfg, "^VIX", "adj"),
            "fred": {s: load_raw(cfg, s)[s] for s in d["fred_series"]}}


def build_clean(cfg: dict, inputs: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One table on the SPY trading calendar with every input series.

    Columns
      spy_open/high/low/close/volume : split+dividend adjusted (Yahoo auto_adjust)
      spy_close_raw, spy_dividend    : unadjusted close and cash dividend (ex-date)
      vix_close                      : ^VIX close of day t
      <FRED id>, <FRED id>_src_date  : value usable at close t and the date it is from
    No feature lags are applied here except the FRED publication lag.
    """
    d, tm = cfg["data"], cfg["timing"]
    inputs = inputs or load_inputs(cfg)
    spy, spy_raw, vix = inputs["spy"], inputs["spy_raw"], inputs["vix"]
    cal = spy.index
    report: dict = {"calendar": {"source": f"{d['trade_asset']} trading days (Yahoo)",
                                 "first": str(cal[0].date()), "last": str(cal[-1].date()),
                                 "days": int(len(cal))}}

    df = pd.DataFrame(index=cal)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"spy_{c.lower()}"] = spy[c].astype(float)
    df["spy_close_raw"] = spy_raw["Close"].reindex(cal).astype(float)
    df["spy_dividend"] = spy_raw["Dividends"].reindex(cal).fillna(0.0).astype(float)

    extra_vix = vix.index[(vix.index >= cal[0]) & ~vix.index.isin(cal)]
    df["vix_close"] = vix["Close"].reindex(cal).astype(float)
    report["vix_rows_dropped_not_spy_days"] = [str(x.date()) for x in extra_vix]

    report["fred"] = {}
    for s in d["fred_series"]:
        a = fred_asof(inputs["fred"][s], cal, tm["fred_lag_days"], tm["fred_ffill_limit_days"])
        df[s] = a["value"]
        df[f"{s}_src_date"] = a["src_date"]
        report["fred"][s] = {
            "days_using_older_value_than_normal_lag": int((a["stale_bdays"] > 0).sum()),
            "max_extra_stale_bdays": float(a["stale_bdays"].max()),
            "nan_after_asof": int(a["value"].isna().sum()),
        }

    o, h, l, c = (df[f"spy_{x}"] for x in ("open", "high", "low", "close"))
    tol = 1e-4
    report["sanity"] = {
        "duplicate_dates": int(cal.duplicated().sum()),
        "nonpositive_prices": int((df[["spy_open", "spy_high", "spy_low", "spy_close"]] <= 0).sum().sum()),
        "zero_volume_days": int((df["spy_volume"] == 0).sum()),
        "ohlc_violations": int(((h < l) | (o > h * (1 + tol)) | (o < l * (1 - tol))
                                | (c > h * (1 + tol)) | (c < l * (1 - tol))).sum()),
    }
    r = c.pct_change()
    thr = cfg["cleaning"]["extreme_return_flag"]
    report["extreme_close_returns_kept"] = {str(k.date()): round(float(v), 4)
                                            for k, v in r[r.abs() > thr].items()}
    report["nan_by_column"] = {k: int(v) for k, v in df.isna().sum().items()}
    return df, report


def save_clean(cfg: dict) -> Path:
    df, report = build_clean(cfg)
    out = ROOT / cfg["data"]["interim_dir"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "clean.parquet")
    (out / "clean_report.json").write_text(json.dumps(report, indent=2))
    print(f"clean -> {(out / 'clean.parquet').relative_to(ROOT)}  {len(df)} rows; "
          f"report -> {(out / 'clean_report.json').relative_to(ROOT)}")
    return out / "clean.parquet"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["download", "clean", "all"], default="all")
    ap.add_argument("--refresh", action="store_true", help="re-download cached files")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage in ("download", "all"):
        download_all(cfg, refresh=args.refresh)
    if args.stage in ("clean", "all"):
        save_clean(cfg)
