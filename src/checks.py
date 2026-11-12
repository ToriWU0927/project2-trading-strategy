"""Part A - adjustment checks (A5, project Appendix B.4).

    python src/checks.py

Writes results/partA/adjustment_checks.json and results/partA/fig_adjustment_checks.png.

What is checked
  1. Dividend back-adjustment: rebuild SPY's total-return price A_t from the
     unadjusted close and cash dividends, compare daily returns with Yahoo's
     Adj Close (return continuity) and check one ex-dividend date by hand.
  2. Split adjustment: AAPL 4-for-1 on 2020-08-31, checked against the two
     published traded closes around the split (Yahoo's "raw" Close turns out to
     be split-adjusted already) and against Yahoo's separate Adj Close column.
  3. Adjusted OHLC: auto_adjust=True Close equals the vendor Adj Close, and the
     adjusted Open uses the same factor as the Close (we trade at the open).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ROOT, load_config, load_raw

# Published AAPL closes around the 4:1 split (traded prices, USD): the last day
# before the split and the first day after it.
AAPL_CLOSE_2020_08_28 = 499.23
AAPL_CLOSE_2020_08_31 = 129.04


def future_product(g: pd.Series) -> pd.Series:
    """prod_{k>t} g_k for every t (today's factor excluded)."""
    rev = g.iloc[::-1].cumprod().iloc[::-1]
    return rev.shift(-1).fillna(1.0)


def back_adjust(price: pd.Series, g: pd.Series) -> pd.Series:
    """A_t = P_t * prod_{k>t} g_k  (Appendix B.2)."""
    return price * future_product(g)


def check_dividends(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    p, d = raw["Close"], raw["Dividends"].fillna(0.0)
    p_prev = p.shift(1)
    g_appendix = pd.Series(np.where(d > 0, 1.0 / (1.0 + d / p_prev), 1.0), index=p.index)
    g_yahoo = pd.Series(np.where(d > 0, 1.0 - d / p_prev, 1.0), index=p.index)
    vendor_ret = raw["Adj Close"].pct_change()

    out: dict = {"n_dividends": int((d > 0).sum())}
    series = {}
    for name, g in (("appendix_1/(1+D/P)", g_appendix), ("yahoo_1-D/P", g_yahoo)):
        a = back_adjust(p, g.fillna(1.0))
        diff = (a.pct_change() - vendor_ret).abs().dropna()
        ex = diff[d.reindex(diff.index) > 0]
        out[name] = {
            "mean_abs_return_diff_all_days": float(diff.mean()),
            "max_abs_return_diff_all_days": float(diff.max()),
            "mean_abs_return_diff_ex_dates": float(ex.mean()),
            "max_abs_return_diff_non_ex_dates": float(diff[d.reindex(diff.index) == 0].max()),
        }
        series[name] = a

    # one representative ex-dividend date: the most recent one
    t = d[d > 0].index[-1]
    i = p.index.get_loc(t)
    t_prev = p.index[i - 1]
    tr_by_hand = (p[t] + d[t]) / p[t_prev] - 1
    out["example_ex_date"] = {
        "date": str(t.date()), "prev_close": float(p[t_prev]), "close": float(p[t]),
        "dividend": float(d[t]),
        "price_return": float(p[t] / p[t_prev] - 1),
        "total_return_by_hand_(P+D)/P_prev-1": float(tr_by_hand),
        "vendor_adj_close_return": float(vendor_ret[t]),
        "abs_diff_vendor_vs_by_hand": float(abs(vendor_ret[t] - tr_by_hand)),
    }
    frame = pd.DataFrame({"raw_close": p, "vendor_adj_close": raw["Adj Close"],
                          "manual_adj_appendix": series["appendix_1/(1+D/P)"]})
    return out, frame


def check_split(raw: pd.DataFrame, split_date: str = "2020-08-31") -> tuple[dict, pd.DataFrame]:
    """Split check against published prices, not against our own arithmetic.

    1. Anchor: Yahoo's 'raw' Close on the day before the split times the split
       ratio must equal the published (traded) close -> Yahoo's Close is already
       split-adjusted.
    2. Split day: the return implied by the two published traded closes,
       (1 + r_traded) * S - 1, must equal Yahoo's split-adjusted return.
    3. Vendor consistency: on days without a dividend, Close and Adj Close
       (two separately published columns) must give the same daily return,
       including the split day.
    """
    s = raw["Stock Splits"].fillna(0.0).replace(0.0, 1.0)
    yahoo_close = raw["Close"]
    t = pd.Timestamp(split_date)
    t_prev = raw.index[raw.index.get_loc(t) - 1]
    S = float(s[t])
    published_traded_ret = AAPL_CLOSE_2020_08_31 / AAPL_CLOSE_2020_08_28 - 1
    implied_adjusted_ret = (1 + published_traded_ret) * S - 1
    yahoo_adjusted_ret = float(yahoo_close[t] / yahoo_close[t_prev] - 1)
    no_div = raw["Dividends"].fillna(0.0) == 0
    r_close, r_adj = yahoo_close.pct_change(), raw["Adj Close"].pct_change()
    out = {
        "split_date": split_date,
        "split_ratio": S,
        "published_close_prev_day": AAPL_CLOSE_2020_08_28,
        "published_close_split_day": AAPL_CLOSE_2020_08_31,
        "yahoo_close_prev_day": float(yahoo_close[t_prev]),
        "yahoo_raw_close_is_split_adjusted": bool(
            abs(yahoo_close[t_prev] * S - AAPL_CLOSE_2020_08_28) < 0.01),
        "published_traded_return_on_split_day": published_traded_ret,
        "implied_split_adjusted_return": implied_adjusted_ret,
        "yahoo_split_adjusted_return": yahoo_adjusted_ret,
        "abs_diff_implied_vs_yahoo": abs(implied_adjusted_ret - yahoo_adjusted_ret),
        "max_abs_return_diff_close_vs_adjclose_no_dividend_days":
            float((r_close - r_adj)[no_div].abs().max()),
    }
    traded = yahoo_close / future_product(1.0 / s)   # undo the split, for the figure only
    frame = pd.DataFrame({"traded_price": traded, "split_adjusted": yahoo_close})
    return out, frame


def check_adjusted_ohlc(adj: pd.DataFrame, raw: pd.DataFrame) -> dict:
    raw = raw.reindex(adj.index)
    factor = raw["Adj Close"] / raw["Close"]
    return {
        "max_rel_diff_auto_adjust_close_vs_adj_close":
            float(((adj["Close"] - raw["Adj Close"]) / raw["Adj Close"]).abs().max()),
        "max_rel_diff_adj_open_vs_raw_open_times_factor":
            float(((adj["Open"] - raw["Open"] * factor) / adj["Open"]).abs().max()),
    }


def plot(div_frame: pd.DataFrame, split_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    div_frame[["raw_close", "vendor_adj_close"]].plot(ax=ax[0], lw=1)
    ax[0].set_title("SPY: price-only vs dividend-adjusted close")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("")
    w = split_frame.loc["2020-08-14":"2020-09-15"]
    w.plot(ax=ax[1], marker=".", lw=1)
    ax[1].axvline(pd.Timestamp("2020-08-31"), color="grey", ls="--", lw=0.8)
    ax[1].set_title("AAPL 4:1 split (2020-08-31): traded vs split-adjusted")
    ax[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    cfg = load_config()
    asset = cfg["data"]["trade_asset"]
    out_dir = ROOT / cfg["data"]["results_dir"] / "partA"
    out_dir.mkdir(parents=True, exist_ok=True)

    spy_raw, spy_adj = load_raw(cfg, asset, "raw"), load_raw(cfg, asset, "adj")
    aapl_raw = load_raw(cfg, "AAPL", "raw")

    div, div_frame = check_dividends(spy_raw)
    split, split_frame = check_split(aapl_raw)
    res = {"dividends_" + asset: div, "split_AAPL": split,
           "adjusted_ohlc_" + asset: check_adjusted_ohlc(spy_adj, spy_raw)}
    (out_dir / "adjustment_checks.json").write_text(json.dumps(res, indent=2))
    plot(div_frame, split_frame, out_dir / "fig_adjustment_checks.png")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
