"""Break-even transaction-cost analysis (post-freeze EVALUATION, no rule/parameter change).

For every pre-specified rule in config.json["partC"], re-run the backtest on a fine
cost grid (0-10bp, step 0.1bp) and find the highest one-way cost at which the
strategy still matches a benchmark (buy & hold, MV hist-mean, cash) on
annualised excess return, Sharpe and certainty equivalent (gamma=3).
The cost-aware rule uses c inside the rule, so its curve need not be monotone;
we report the first crossing from below 0bp upward.

Usage (from the repository root):  python src/breakeven.py --period test   (or val)
"""
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(__file__))
from backtest import aligned_index, calib_vol, ANN, load_parquet, w_mv, w_cost_aware, w_sign, run, metrics, paired_bootstrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))["partC"]
P = lambda k: os.path.join(ROOT, cfg["inputs"][k])
ap = argparse.ArgumentParser(); ap.add_argument("--period", choices=["val", "test"], required=True)
period = ap.parse_args().period
OUT = os.path.join(ROOT, cfg["output_dir"], period); os.makedirs(OUT, exist_ok=True)

panel = load_parquet(P("panel"))
if period == "test":
    pred = load_parquet(P("test_predictions")); mu_main_s, mu_mean_s = pred["champion"], pred["mean"]
else:
    pred = load_parquet(P("val_predictions")); mu_main_s, mu_mean_s = pred["y_hat"], pred["train_mean"]
idx, _dropped = aligned_index(panel, pred, period, [c for c in pred.columns if c not in ("y_true", "y_exret")])
f = panel.loc[idx]
y, rf = f["y_ret"].values, f["rf_daily"].values
sigma2 = (f["vol_rv_20"] ** 2 / ANN).values
mu_main, mu_mean = mu_main_s.loc[idx].values, mu_mean_s.loc[idx].values
sigma_target = calib_vol(panel, idx[0]).median()
G, W0 = cfg["position"]["gamma_main"], cfg["accounting"]["initial_weight"]

def strategies(c):
    return {
        "main: cost-aware champion": w_cost_aware(mu_main, sigma2, G, c, W0),
        "MV champion": w_mv(mu_main, sigma2, G),
        "sign champion": w_sign(mu_main),
        "MV hist-mean": w_mv(mu_mean, sigma2, G),
        "buy & hold SPY": np.ones(len(idx)),
        "MA200 trend": (f["mom_ma_200"].values > 0).astype(float),
        "target-vol SPY": np.minimum(sigma_target / f["vol_rv_20"].values, 1.0),
    }

GRID = np.round(np.arange(0, 10.0001, 0.1), 1)            # bp
rows = []
for bp in GRID:
    c = bp / 1e4
    for name, w in strategies(c).items():
        m = metrics(run(w, y, rf, c, W0, idx), 3, name)
        rows.append({"strategy": name, "cost_bp": bp, "ann_excess": m.ann_excess, "sharpe": m.sharpe,
                     "CE_ann": m.CE_ann, "ann_turnover": m.ann_turnover})
curve = pd.DataFrame(rows)
curve.to_csv(os.path.join(OUT, "breakeven_curves.csv"), index=False)

def crossing(diff, grid):
    """Highest cost before diff (strategy - benchmark) first turns negative, linear interpolation."""
    if diff[0] < 0: return "worse even at 0bp"
    for i in range(1, len(diff)):
        if diff[i] < 0:
            x0, x1, d0, d1 = grid[i - 1], grid[i], diff[i - 1], diff[i]
            return round(x0 + (x1 - x0) * d0 / (d0 - d1), 2)
    return f"> {grid[-1]:.0f}bp"

piv = {k: curve.pivot(index="cost_bp", columns="strategy", values=k) for k in ["ann_excess", "sharpe", "CE_ann"]}
be = []
for s in ["main: cost-aware champion", "MV champion", "sign champion", "MV hist-mean", "MA200 trend", "target-vol SPY"]:
    r = {"strategy": s,
         "sharpe_at_0bp": piv["sharpe"].loc[0.0, s], "sharpe_at_1bp": piv["sharpe"].loc[1.0, s],
         "sharpe_at_2bp": piv["sharpe"].loc[2.0, s], "sharpe_at_5bp": piv["sharpe"].loc[5.0, s],
         "ann_turnover_at_5bp": curve[(curve.strategy == s) & (curve.cost_bp == 5.0)].ann_turnover.iloc[0]}
    for bench in ["buy & hold SPY", "MV hist-mean"]:
        if bench == s: continue
        for k in ["ann_excess", "sharpe", "CE_ann"]:
            r[f"breakeven_bp_vs_{bench}_{k}"] = crossing((piv[k][s] - piv[k][bench]).values, GRID)
    r["breakeven_bp_vs_cash_ann_excess"] = crossing(piv["ann_excess"][s].values, GRID)
    be.append(r)
be = pd.DataFrame(be)
be.to_csv(os.path.join(OUT, "breakeven_table.csv"), index=False)

# bootstrap at low cost levels (is any advantage distinguishable from noise?)
bs = []
for bp in [0.0, 1.0, 2.0]:
    c = bp / 1e4; st = strategies(c)
    led = {n: run(w, y, rf, c, W0, idx) for n, w in st.items()}
    for a, b in [("sign champion", "buy & hold SPY"), ("MV champion", "buy & hold SPY"),
                 ("sign champion", "MV hist-mean"), ("MV champion", "MV hist-mean"),
                 ("main: cost-aware champion", "buy & hold SPY")]:
        d = paired_bootstrap(led[a].r_net_ex.values, led[b].r_net_ex.values, 63, 2000, 5205, 3)
        d = d[d.stat == "sharpe_diff"]; d.insert(0, "cost_bp", bp); d.insert(1, "A", a); d.insert(2, "B", b)
        bs.append(d)
bs = pd.concat(bs); bs.to_csv(os.path.join(OUT, "breakeven_bootstrap_sharpe.csv"), index=False)

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"main: cost-aware champion": "#1f4e9c", "MV champion": "#7aa6e0", "sign champion": "#c0392b",
       "MV hist-mean": "#e3a869", "buy & hold SPY": "#222222", "MA200 trend": "#2e8b57", "target-vol SPY": "#8e44ad"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for s in COL:
    ax[0].plot(GRID, piv["sharpe"][s], color=COL[s], lw=2.2 if s.startswith("main") else 1.3,
               ls="--" if s == "buy & hold SPY" else "-", label=s)
    ax[1].plot(GRID, piv["ann_excess"][s] * 100, color=COL[s], lw=2.2 if s.startswith("main") else 1.3,
               ls="--" if s == "buy & hold SPY" else "-")
for a in ax:
    a.axvspan(0, 1, color="#2e8b57", alpha=.08)
    a.axvline(cfg["costs"]["main_bp"], color="grey", lw=.8, ls=":"); a.grid(alpha=.3); a.set_xlabel("one-way cost (bp per unit turnover)")
ax[1].text(0.1, 0.97, "shaded 0-1bp; SPY quoted\nspread ~0.3bp", transform=ax[1].get_xaxis_transform(), va="top", fontsize=7, color="#2e8b57")
ax[0].set_ylabel("Sharpe (excess, net)"); ax[1].set_ylabel("annualised excess return, net (%)")
ax[0].legend(fontsize=7, frameon=False, loc="lower left"); ax[0].set_title(f"{period}: performance vs cost (dotted = main {cfg['costs']['main_bp']}bp)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_breakeven.png"), dpi=150); plt.close(fig)

pd.set_option("display.width", 250)
print(be.T.to_string()); print(bs.round(3).to_string(index=False))
