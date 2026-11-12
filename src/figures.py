"""Assignment-suggested figures (PDF section 4.9):
 (i)  fig_oos_r2.png  : out-of-sample R^2 of predictions (vs expanding historical mean)
 (iii) fig_yearly.png : net return and Sharpe ratio by calendar year (main cost scenario)
Run AFTER run_partC.py.  Usage: python src/figures.py --period test   (or val)
"""
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(__file__))
from backtest import ANN, load_parquet
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))["partC"]
ap = argparse.ArgumentParser(); ap.add_argument("--period", choices=["val", "test"], required=True)
period = ap.parse_args().period
OUT = os.path.join(ROOT, cfg["output_dir"], period)
BP = cfg["costs"]["main_bp"]

# ------------------------------------------------------------ (i) OOS R^2
if period == "test":
    pred = load_parquet(os.path.join(ROOT, cfg["inputs"]["test_predictions"]))
    y_col, base = "y_exret", "mean"
    models = ["ridge_M1", "ridge_M2", "ridge_M3", "elasticnet_M1", "elasticnet_M2", "elasticnet_M3",
              "lasso_M1", "lasso_M2", "lasso_M3", "ar1"]
    main = "ridge_M1"
else:
    pred = load_parquet(os.path.join(ROOT, cfg["inputs"]["val_predictions"]))
    y_col, base = "y_true", "train_mean"
    models = ["y_hat"]; main = "y_hat"
pred = pred[pred[y_col].notna()]
y = pred[y_col]
den = ((y - pred[base]) ** 2).sum()
r2 = pd.Series({m: 1 - ((y - pred[m]) ** 2).sum() / den for m in models})
r2_year = pd.Series({yr: 1 - ((g[y_col] - g[main]) ** 2).sum() / ((g[y_col] - g[base]) ** 2).sum()
                     for yr, g in pred.groupby(pred.index.year)})
# N4: two correct aggregations. "oos_r2" pools all rows (1 - total SSE / total benchmark SSE, as Part B does for test);
# on validation Part B selected models on the equal-weight mean of the 10 yearly fold R2 (0.2407% for the champion),
# which differs from the pooled value (0.2694%). Both are exported; the pooled one is plotted and labelled as such.
r2_tab = pd.DataFrame({"oos_r2": r2, "aggregation": "pooled over all rows"})
if period == "val":
    r2_tab["oos_r2_fold_mean"] = r2_year.mean()
    r2_tab["note"] = "Part B model selection uses oos_r2_fold_mean (equal-weight mean of 10 yearly folds); oos_r2 is pooled"
r2_tab.to_csv(os.path.join(OUT, "oos_r2_by_model.csv"), index_label="model")
r2_year.rename("oos_r2_main").to_csv(os.path.join(OUT, "oos_r2_by_year.csv"), index_label="year")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.1, 1.4, 1.1]})
lab = [m.replace("champion", "") for m in r2.index]
cols = ["#1f4e9c" if m == main else ("#c0392b" if v < 0 else "#7aa6e0") for m, v in r2.items()]
ax[0].barh(lab, r2.values * 100, color=cols); ax[0].axvline(0, color="k", lw=.7)
ax[0].invert_yaxis(); ax[0].set_xlabel("OOS R² (%) vs historical mean")
ax[0].set_title("(a) whole period, pooled R²" + (" — dark = champion (ridge_M1)" if period == "test"
                 else f"\n(Part B selection uses fold mean {r2_year.mean()*100:.3f}%)"), fontsize=9)
for i, v in enumerate(r2.values):
    ax[0].text(v * 100 if v >= 0 else v * 100, i, f" {v*100:.3f} ", va="center", fontsize=7,
               ha="left", color="k" if v >= 0 else "white")
for m in (["ridge_M1", "ridge_M2", "ridge_M3", "ar1"] if period == "test" else ["y_hat"]):
    d = ((y - pred[base]) ** 2 - (y - pred[m]) ** 2).cumsum()
    ax[1].plot(d.index, d.values * 1e4, lw=2 if m == main else 1, label=m if period == "test" else "champion")
ax[1].axhline(0, color="k", lw=.7); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("(b) cumulative squared-error advantage over historical mean\n(rising = model forecasting better)", fontsize=9)
ax[1].set_ylabel("Σ[(y−mean)² − (y−model)²] ×10⁴")
ax[2].bar(r2_year.index.astype(str), r2_year.values * 100,
          color=["#1f4e9c" if v >= 0 else "#c0392b" for v in r2_year.values])
ax[2].axhline(0, color="k", lw=.7); ax[2].tick_params(axis="x", rotation=60, labelsize=7)
ax[2].set_ylabel("OOS R² (%)"); ax[2].set_title("(c) champion OOS R² by calendar year", fontsize=9)
for a in ax: a.grid(alpha=.3)
fig.suptitle(f"{period}: out-of-sample R² of predictions", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_oos_r2.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------ (iii) yearly return & Sharpe
wide = pd.read_csv(os.path.join(OUT, "ledger_all_strategies.csv"), index_col=0, parse_dates=True)
rf = pd.read_csv(os.path.join(OUT, "ledger_main.csv"), index_col=0, parse_dates=True)["rf_daily"]
S = ["main: cost-aware champion", "MV hist-mean", "target-vol SPY", "buy & hold SPY"]
COL = {"main: cost-aware champion": "#1f4e9c", "MV hist-mean": "#e3a869", "target-vol SPY": "#8e44ad",
       "buy & hold SPY": "#222222"}
yr = wide.index.year
ret = pd.DataFrame({s: (1 + wide[f"{s}|r_net"]).groupby(yr).prod() - 1 for s in S})
shp = pd.DataFrame({s: (wide[f"{s}|r_net"] - rf).groupby(yr).apply(lambda e: e.mean() / e.std(ddof=1) * np.sqrt(ANN))
                    for s in S})
pd.concat({"net_return": ret, "sharpe": shp}, axis=1).to_csv(os.path.join(OUT, "yearly_return_sharpe.csv"), index_label="year")

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
x = np.arange(len(ret)); wdt = 0.8 / len(S)
for i, s in enumerate(S):
    ax[0].bar(x + (i - (len(S) - 1) / 2) * wdt, ret[s] * 100, wdt, color=COL[s], label=s)
    ax[1].bar(x + (i - (len(S) - 1) / 2) * wdt, shp[s], wdt, color=COL[s])
for a in ax: a.axhline(0, color="k", lw=.7); a.grid(alpha=.3, axis="y")
ax[0].set_ylabel("net return in year (%)"); ax[1].set_ylabel("Sharpe within year")
ax[1].set_xticks(x); ax[1].set_xticklabels([str(i) for i in ret.index])
ax[0].legend(fontsize=8, frameon=False, ncol=4)
last = ret.index[-1]; n_last = (yr == last).sum()
ax[0].set_title(f"{period}: return and Sharpe by calendar year (net of {BP}bp cost, gamma={cfg['position']['gamma_main']})"
                + (f"; {last} is partial ({n_last} days)" if n_last < 240 else ""), fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_yearly.png"), dpi=150); plt.close(fig)
print("OOS R2:", r2.round(5).to_dict()); print(ret.round(3).to_string()); print(shp.round(2).to_string())
