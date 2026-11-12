"""Part A - descriptive statistics for the report (A11).

    python src/eda.py

Everything that touches labels or feature distributions uses the DEVELOPMENT
period only (config.splits), so nothing here peeks at the test period. The one
exception is the input time-series figure, which only plots raw inputs and
shades the test period for orientation.

Outputs in results/partA/:
    coverage.csv, feature_stats_dev.csv, label_stats_dev.csv, ic_dev.csv,
    group_summary_dev.csv, top_correlations_dev.csv, eda_summary.md,
    ic_subperiods_dev.csv, subperiod_checks_dev.csv, stambaugh_simulation.csv,
    open_quality_by_year.csv,
    fig_inputs_timeseries.png, fig_corr_heatmap_dev.png, fig_univariate_ic_dev.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm

from data import ROOT, load_config
from features import GROUPS, get_feature_groups

# reference palette (dataviz skill), light mode
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"
BLUE, RED = "#2a78d6", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, NEUTRAL, RED])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 9,
})


def load():
    cfg = load_config()
    panel = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "panel.parquet")
    clean = pd.read_parquet(ROOT / cfg["data"]["interim_dir"] / "clean.parquet")
    return cfg, panel, clean


def coverage(cfg, clean) -> pd.DataFrame:
    man = json.loads((ROOT / cfg["data"]["raw_dir"] / "_manifest.json").read_text())
    src = {"SPY": "Yahoo Finance (yfinance)", "VIX": "Yahoo Finance (yfinance)",
           "AAPL": "Yahoo Finance (yfinance)", "DTB3": "FRED", "T10Y3M": "FRED",
           "BAA10Y": "FRED"}
    rows = []
    for f in man["files"]:
        name = f["file"].replace("yf_", "").replace("fred_", "").replace(".parquet", "")
        key = name.split("_")[0]
        rows.append({"file": f["file"], "series": key, "source": src.get(key, ""),
                     "first_date": f["first_date"], "last_date": f["last_date"],
                     "rows_raw": f["rows"],
                     "missing_raw": sum(f["missing_by_column"].values())})
    return pd.DataFrame(rows)


def feature_stats(X: pd.DataFrame) -> pd.DataFrame:
    q = X.quantile([0.01, 0.99]).T
    return pd.DataFrame({"mean": X.mean(), "std": X.std(), "skew": X.skew(),
                         "excess_kurtosis": X.kurt(), "min": X.min(), "p01": q[0.01],
                         "p99": q[0.99], "max": X.max()})


def label_stats(y: pd.Series, r: pd.Series) -> pd.DataFrame:
    s = {
        "observations": int(y.count()),
        "mean_daily_excess": y.mean(),
        "annualised_mean_excess": y.mean() * 252,
        "annualised_vol": y.std() * np.sqrt(252),
        "buy_and_hold_sharpe": y.mean() / y.std() * np.sqrt(252),
        "skew": y.skew(), "excess_kurtosis": y.kurt(),
        "share_positive": float((y > 0).mean()),
        "autocorr_lag1": y.autocorr(1),
        "min": y.min(), "max": y.max(),
        "raw_return_annualised_mean": r.mean() * 252,
    }
    return pd.DataFrame({"value": s})


def univariate_ic(X: pd.DataFrame, y: pd.Series, block: int = 63, reps: int = 1000,
                  seed: int = 5205) -> pd.DataFrame:
    """Correlation of each feature on row t with next-day excess return y_t.

    Pooled Pearson and Spearman over the whole development period, with a 95%
    moving-block bootstrap CI for the Spearman IC (blocks of `block` days keep
    the serial dependence of the slow features).

    Stability is shown on two halves of the development period, NOT year by
    year: with ~250 days per year, a persistent feature's within-year IC is
    biased towards negative values (Stambaugh 1999 small-sample bias). Under
    simulated iid returns, mom_ret_120 still shows a mean yearly IC of about
    -0.07 and is positive in only ~12% of years, so yearly ICs would mislead.
    """
    df = X.join(y.rename("y")).dropna()
    R = df.rank().to_numpy()
    R = (R - R.mean(0)) / R.std(0)
    n, k = R.shape[0], R.shape[1] - 1
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block, size=(reps, int(np.ceil(n / block))))
    boot = np.empty((reps, k))
    for b in range(reps):
        idx = (starts[b][:, None] + np.arange(block)).ravel()[:n]
        Rb = R[idx]
        Rb = (Rb - Rb.mean(0)) / Rb.std(0)
        boot[b] = (Rb[:, :k] * Rb[:, [k]]).mean(0)
    mid = df.index[len(df) // 2]
    first, second = df.loc[:mid], df.loc[mid:].iloc[1:]
    feats = df.columns.drop("y")
    out = pd.DataFrame({
        "pearson_ic": df[feats].corrwith(df["y"]),
        "spearman_ic": df[feats].corrwith(df["y"], method="spearman"),
        "ci95_low": np.percentile(boot, 2.5, axis=0),
        "ci95_high": np.percentile(boot, 97.5, axis=0),
        "spearman_ic_first_half": first[feats].corrwith(first["y"], method="spearman"),
        "spearman_ic_second_half": second[feats].corrwith(second["y"], method="spearman"),
    }, index=feats)
    out["ci_excludes_zero"] = (out["ci95_low"] > 0) | (out["ci95_high"] < 0)
    # 24 features tested at 5% -> about one false positive expected by chance;
    # Bonferroni with the bootstrap standard error as a stricter check
    out["boot_se"] = boot.std(axis=0)
    z_crit = norm.ppf(1 - 0.025 / k)
    out["significant_bonferroni"] = (out["spearman_ic"].abs() / out["boot_se"]) > z_crit
    out["same_sign_both_halves"] = (np.sign(out["spearman_ic_first_half"])
                                    == np.sign(out["spearman_ic_second_half"]))
    out.attrs.update(block=block, reps=reps, split=str(mid.date()), z_bonferroni=z_crit)
    return out


def group_summary(X: pd.DataFrame, ic: pd.DataFrame, groups: dict) -> pd.DataFrame:
    C = X.corr().abs()
    rows = []
    for g in GROUPS:
        cols = groups[g]
        other = [c for c in X.columns if c not in cols]
        inner = C.loc[cols, cols].to_numpy()[np.triu_indices(len(cols), 1)]
        rows.append({"group": g, "description": GROUPS[g], "n_features": len(cols),
                     "mean_abs_spearman_ic": ic.loc[cols, "spearman_ic"].abs().mean(),
                     "max_abs_spearman_ic": ic.loc[cols, "spearman_ic"].abs().max(),
                     "best_feature": ic.loc[cols, "spearman_ic"].abs().idxmax(),
                     "mean_abs_corr_within": inner.mean() if inner.size else np.nan,
                     "mean_abs_corr_to_other_groups": C.loc[cols, other].to_numpy().mean()})
    return pd.DataFrame(rows).set_index("group")


def subperiod_ic(X: pd.DataFrame, y: pd.Series, subperiods: dict) -> pd.DataFrame:
    """Spearman IC in development sub-periods (US quotes went decimal in 2001)."""
    out = {}
    for name, (a, b) in subperiods.items():
        Xa, ya = X.loc[a:b], y.loc[a:b]
        out[name] = Xa.corrwith(ya, method="spearman")
    out = pd.DataFrame(out)
    out.index.name = "feature"
    return out


def subperiod_checks(dev: pd.DataFrame, y_cc: pd.Series, subperiods: dict,
                     focus: list[str]) -> pd.DataFrame:
    """Is the reversal signal an artefact of coarse early opens?

    Compares, per sub-period, the open-to-open label with a close-to-close
    label over the same next-day window (Close_(t+2)/Close_(t+1) - 1).
    """
    rows = {}
    for name, (a, b) in subperiods.items():
        x, ycc = dev.loc[a:b], y_cc.loc[a:b]
        r = {"rows": len(x),
             "autocorr_lag1_y_ret": x["y_ret"].autocorr(),
             "corr_ret_cc_t_vs_y_ret": x["ret_cc"].corr(x["y_ret"])}
        for f in focus:
            r[f"ic_{f}_open_to_open"] = x[f].corr(x["y_exret"], method="spearman")
            r[f"ic_{f}_close_to_close"] = x[f].corr(ycc, method="spearman")
        rows[name] = r
    out = pd.DataFrame(rows)
    out.index.name = "metric"
    return out


def stambaugh_sim(dev: pd.DataFrame, windows: list[int], reps: int, subperiods: dict,
                  seed: int) -> pd.DataFrame:
    """IC of trailing-return features under iid returns (true IC = 0).

    Shows the small-sample bias of persistent predictors: pooled over the whole
    period vs computed year by year vs per multi-year sub-period.
    """
    rng = np.random.default_rng(seed)
    warm = max(windows)
    idx = pd.bdate_range(end=dev.index[-1], periods=len(dev) + warm + 1)
    mu, sd = dev["y_ret"].mean(), dev["y_ret"].std()
    res = {w: {"pooled": [], "yearly_mean": [], "years_positive": [], "subperiod_mean": []}
           for w in windows}
    for _ in range(reps):
        r = pd.Series(rng.normal(mu, sd, len(idx)), index=idx)
        c = (1 + r).cumprod()
        y = r.shift(-1)
        for w in windows:
            df = pd.DataFrame({"x": c / c.shift(w) - 1, "y": y}).dropna()
            yr = df.groupby(df.index.year).apply(lambda g: g["x"].corr(g["y"], method="spearman"))
            sub = [df.loc[a:b, "x"].corr(df.loc[a:b, "y"], method="spearman")
                   for a, b in subperiods.values()]
            res[w]["pooled"].append(df["x"].corr(df["y"], method="spearman"))
            res[w]["yearly_mean"].append(yr.mean())
            res[w]["years_positive"].append((yr > 0).mean())
            res[w]["subperiod_mean"].append(np.mean(sub))
    out = pd.DataFrame({f"trailing_return_{w}d": {
        "reps": reps,
        "mean_pooled_ic": np.mean(v["pooled"]),
        "mean_yearly_ic": np.mean(v["yearly_mean"]),
        "share_of_years_with_positive_ic": np.mean(v["years_positive"]),
        "mean_subperiod_ic": np.mean(v["subperiod_mean"]),
    } for w, v in res.items()})
    out.index.name = "metric"
    return out


def open_quality(clean: pd.DataFrame) -> pd.DataFrame:
    """How often the SPY open looks stale or sits on the day's extreme, by year.

    Before decimalisation (2001) SPY quoted in 1/8 or 1/16 dollar steps, so
    early opens are coarse; this matters because the label is open-to-open.
    """
    o, h, l, c = (clean[f"spy_{x}"] for x in ("open", "high", "low", "close"))
    prev = c.shift(1)
    df = pd.DataFrame({
        "open_equals_prev_close": np.isclose(o, prev, rtol=1e-6),
        "open_equals_prev_open": np.isclose(o, o.shift(1), rtol=1e-6),
        "open_equals_high_or_low": np.isclose(o, h, rtol=1e-6) | np.isclose(o, l, rtol=1e-6),
        "mean_abs_overnight_return": (o / prev - 1).abs(),
    }, index=clean.index)
    out = df.groupby(df.index.year).mean()
    out.index.name = "year"
    return out


def top_correlations(X: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    C = X.corr()
    iu = np.triu_indices(len(C), 1)
    pairs = pd.DataFrame({"feature_a": C.index[iu[0]], "feature_b": C.columns[iu[1]],
                          "corr": C.to_numpy()[iu]})
    return pairs.reindex(pairs["corr"].abs().sort_values(ascending=False).index).head(k)


# ---------------------------------------------------------------- figures

def _group_spans(cols, groups):
    spans, start = [], 0
    for g in GROUPS:
        n = len([c for c in cols if c in groups[g]])
        spans.append((g, start, start + n))
        start += n
    return spans


def fig_inputs(clean: pd.DataFrame, cfg: dict, path) -> None:
    s = cfg["splits"]
    series = [("SPY adjusted close (log scale, USD)", clean["spy_close"], True),
              ("VIX (index points)", clean["vix_close"], False),
              ("Term spread T10Y3M (%)", clean["T10Y3M"], False),
              ("Credit spread BAA10Y (%)", clean["BAA10Y"], False)]
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 8.5), sharex=True)
    for ax, (title, x, log) in zip(axes, series):
        ax.axvspan(pd.Timestamp(s["test_start"]), pd.Timestamp(s["test_end"]),
                   color=NEUTRAL, lw=0, zorder=0)
        ax.plot(x.index, x, color=BLUE, lw=1.0)
        if log:
            ax.set_yscale("log")
            ax.set_yticks([25, 50, 100, 200, 400, 800], ["25", "50", "100", "200", "400", "800"])
            ax.minorticks_off()
        if "T10Y3M" in title:
            ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(title)
        ax.margins(x=0)
    axes[0].text(pd.Timestamp(s["test_start"]), 1.02, "  test period (not used for any statistic)",
                 transform=axes[0].get_xaxis_transform(), color=INK2, fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_corr(X: pd.DataFrame, groups: dict, path, period: str) -> None:
    C = X.corr().to_numpy()
    n = len(X.columns)
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), X.columns, rotation=90)
    ax.set_yticks(range(n), X.columns)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for g, a, b in _group_spans(X.columns, groups):
        if a:
            ax.axhline(a - 0.5, color=SURFACE, lw=2)
            ax.axvline(a - 0.5, color=SURFACE, lw=2)
        ax.text(n - 0.3, (a + b - 1) / 2, f" {g}", va="center", ha="left", color=INK2, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.08)
    cb.outline.set_visible(False)
    cb.set_label("Pearson correlation", color=INK2)
    ax.set_title(f"Feature correlations, development period {period}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_ic(ic: pd.DataFrame, groups: dict, path, period: str) -> None:
    cols = list(ic.index)
    y = np.arange(len(cols))[::-1]
    v = ic["spearman_ic"].to_numpy()
    err = np.vstack([v - ic["ci95_low"].to_numpy(), ic["ci95_high"].to_numpy() - v])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(y, v, height=0.6, color=[RED if x > 0 else BLUE for x in v], zorder=2)
    ax.errorbar(v, y, xerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    ax.set_yticks(y, cols)
    ax.grid(axis="y", visible=False)
    for g, a, b in _group_spans(cols, groups):
        if a:
            ax.axhline(len(cols) - a - 0.5, color=GRID, lw=1)
        ax.text(1.0, len(cols) - (a + b - 1) / 2 - 1, g, transform=ax.get_yaxis_transform(),
                ha="left", va="center", color=INK2, fontsize=9)
    ax.set_xlabel(f"Spearman IC with next-day excess return "
                  f"(95% CI, {ic.attrs['block']}-day block bootstrap)")
    ax.set_title(f"Univariate predictive correlation, development period {period}")
    ax.text(0.99, 0.01, "red = positive, blue = negative", transform=ax.transAxes,
            ha="right", va="bottom", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- summary

def _md_table(df: pd.DataFrame, fmt: str = "{:.4g}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [fmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in row]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg, panel, clean = load()
    out = ROOT / cfg["data"]["results_dir"] / "partA"
    out.mkdir(parents=True, exist_ok=True)
    groups = get_feature_groups(panel.columns)
    feats = groups["M3"]
    dev = panel[panel["period"] == "dev"]
    X, y = dev[feats], dev["y_exret"]

    cov = coverage(cfg, clean)
    fs = feature_stats(X)
    ls = label_stats(y, dev["y_ret"])
    a = cfg["analysis"]
    ic = univariate_ic(X, y, a["bootstrap_block_days"], a["bootstrap_reps"], a["seed"])
    gs = group_summary(X, ic, groups)
    tc = top_correlations(X)
    sp = subperiod_ic(X, y, a["subperiods"])
    y_cc = (panel["px_close"].shift(-2) / panel["px_close"].shift(-1) - 1).loc[dev.index]
    sc = subperiod_checks(dev, y_cc, a["subperiods"], a["subperiod_focus_features"])
    st = stambaugh_sim(dev, a["stambaugh_windows"], a["stambaugh_reps"], a["subperiods"], a["seed"])
    oq = open_quality(clean)
    sp.to_csv(out / "ic_subperiods_dev.csv")
    sc.to_csv(out / "subperiod_checks_dev.csv")
    st.to_csv(out / "stambaugh_simulation.csv")
    oq.to_csv(out / "open_quality_by_year.csv")

    cov.to_csv(out / "coverage.csv", index=False)
    fs.to_csv(out / "feature_stats_dev.csv")
    ls.to_csv(out / "label_stats_dev.csv")
    ic.to_csv(out / "ic_dev.csv")
    gs.to_csv(out / "group_summary_dev.csv")
    tc.to_csv(out / "top_correlations_dev.csv", index=False)

    fig_inputs(clean, cfg, out / "fig_inputs_timeseries.png")
    period = f"{dev.index[0].year}-{dev.index[-1].year}"
    fig_corr(X, groups, out / "fig_corr_heatmap_dev.png", period)
    fig_ic(ic, groups, out / "fig_univariate_ic_dev.png", period)

    fs.index.name = ic.index.name = "feature"
    md = [
        "# Part A - descriptive statistics (development period only)", "",
        f"Development period: {dev.index[0].date()} -> {dev.index[-1].date()}, {len(dev)} rows. "
        "Generated by `python src/eda.py`.", "",
        "## Data coverage", "", _md_table(cov.set_index("file")), "",
        "## Label (y_exret, next-day open-to-open excess return)", "",
        _md_table(ls), "",
        "## Feature statistics", "", _md_table(fs), "",
        "## Univariate IC with y_exret", "", _md_table(ic), "",
        "## Feature groups", "", _md_table(gs), "",
        "## Most correlated feature pairs", "", _md_table(tc.set_index("feature_a")), "",
        "## Spearman IC by development sub-period", "", _md_table(sp), "",
        "## Sub-period checks: open-to-open vs close-to-close label", "", _md_table(sc), "",
        "## Stambaugh simulation (iid returns, true IC = 0)", "", _md_table(st), "",
        "## SPY open-price quality by year (no labels used)", "",
        _md_table(oq.loc[:a["open_quality_table_until_year"]]), "",
    ]
    (out / "eda_summary.md").write_text("\n".join(md))
    pd.set_option("display.width", 200)
    print(ls.round(4))
    print(ic[["spearman_ic", "ci95_low", "ci95_high", "spearman_ic_first_half",
              "spearman_ic_second_half", "ci_excludes_zero", "same_sign_both_halves"]].round(4))
    print(gs.round(3))
    print(tc.round(3))


if __name__ == "__main__":
    main()
