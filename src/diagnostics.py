"""Post-hoc EXPLANATORY diagnostics for the frozen test run (no rule/parameter changes).
1) Which branch of the cost-aware rule produced each exposure cut (mechanical label, NOT a causal split
   between forecast and volatility; the vol-change columns are descriptive only)
2) How long does the champion forecast carry information? corr(mu_t, y_exret_{t+k}).
3) How concentrated is the test OOS R^2 in a few days?"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from backtest import load_parquet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))["partC"]
L = pd.read_csv(os.path.join(ROOT, cfg["output_dir"], "test", "ledger_main.csv"), index_col=0, parse_dates=True)
g, c = cfg["position"]["gamma_main"], cfg["costs"]["main_bp"] / 1e4
U = (L.mu_hat + c) / (g * L.sigma2_daily)
cut = L.w < L.w_prev_target - 1e-12
add = L.w > L.w_prev_target + 1e-12
why = np.where(L.mu_hat < -c, "forecast < -c (exit on bearish forecast)",
      np.where(L.mu_hat < 0, "forecast in [-c,0), band upper bound below previous target",
               "forecast >= 0, risk-adjusted band upper bound (mu+c)/(gamma*sigma2) below previous target"))
vol_up = (L.vol_rv_20 > L.vol_rv_20.shift(1)).values
cuts = pd.DataFrame({"reason": why[cut], "size": (L.w_prev_target - L.w)[cut], "vol_up": vol_up[cut]})
out = cuts.groupby("reason").agg(n_days=("size", "count"), total_reduction=("size", "sum"),
                                 n_days_vol_rose_vs_prev_day=("vol_up", "sum"))
out["n_days_vol_not_higher_than_prev_day"] = out.n_days - out.n_days_vol_rose_vs_prev_day
out["share_of_reduction"] = out.total_reduction / out.total_reduction.sum()
print(out.round(3)); out.to_csv(os.path.join(ROOT, cfg["output_dir"], "test", "diag_exposure_cuts.csv"))
print("increases: n", int(add.sum()), "mean vol_rv_20 on cut days", L.vol_rv_20[cut].mean().round(3),
      "| all days", L.vol_rv_20.mean().round(3))
rows = []
for k in range(0, 6):
    fut = L.y_exret.shift(-k)
    m = fut.notna()
    rows.append({"k_days_ahead": k, "corr(mu_t, y_exret_t+k)": np.corrcoef(L.mu_hat[m], fut[m])[0, 1]})
dec = pd.DataFrame(rows); print(dec.round(4).to_string(index=False))
dec.to_csv(os.path.join(ROOT, cfg["output_dir"], "test", "diag_signal_decay.csv"), index=False)
# stickiness: lagged positions
wl = pd.Series(np.where(L.mu_hat > 0, 1.0, 0.0), L.index)
for lag in [0, 1, 2, 5]:
    w = wl.shift(lag).fillna(0)
    print(f"sign position lagged {lag}d: gross timing cov*252 = {(np.mean((w-w.mean())*(L.y_exret-L.y_exret.mean()))*252):.4f}")

# 3) How concentrated is the test OOS R^2? (champion vs expanding historical mean)
P = load_parquet(os.path.join(ROOT, cfg["inputs"]["test_predictions"])); P = P[P.y_exret.notna()]
panel_idx = load_parquet(os.path.join(ROOT, cfg["inputs"]["panel"])).index
gain = (P.y_exret - P["mean"]) ** 2 - (P.y_exret - P.champion) ** 2
top = gain.sort_values(ascending=False)
rows = []
for k in [0, 1, 3, 5, 10]:
    q = P.drop(top.index[:k])
    pi = panel_idx; loc = pi.get_indexer(top.index[:k])
    rows.append({"days_removed": k, "removed_decision_dates": ";".join(str(d.date()) for d in top.index[:k]),
                 "return_windows_open_to_open": ";".join(f"{pi[i+1].date()}->{pi[i+2].date()}" for i in loc),
                 "share_of_total_gain": top.iloc[:k].sum() / gain.sum() if k else 0.0,
                 "oos_r2_after_removal": 1 - ((q.y_exret - q.champion) ** 2).sum() / ((q.y_exret - q["mean"]) ** 2).sum()})
conc = pd.DataFrame(rows); print(conc.round(4).to_string(index=False))
conc.to_csv(os.path.join(ROOT, cfg["output_dir"], "test", "diag_oos_r2_concentration.csv"), index=False)
