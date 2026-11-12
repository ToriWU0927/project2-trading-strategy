"""CAPM regression of each strategy's daily net excess return on SPY's excess return
(post-freeze EVALUATION, no rule/parameter change).

    r_strategy,t - rf_t = alpha + beta * (r_SPY,t - rf_t) + e_t

SPY's excess return is the same open-to-open holding-period return the strategies earn
(buy & hold, no cost), so alpha = return not explained by average SPY exposure.
Standard errors: Newey-West (HAC), lag = floor(4*(T/100)^(2/9)), Bartlett kernel.
Alpha and its interval are annualised x252. Run after run_partC.py (reads its ledgers).

Usage (from the repository root):  python src/capm.py --period test   (or val)
"""
import argparse, json, math, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))["partC"]
ap = argparse.ArgumentParser(); ap.add_argument("--period", choices=["val", "test"], required=True)
period = ap.parse_args().period
OUT = os.path.join(ROOT, cfg["output_dir"], period)
ANN = 252

wide = pd.read_csv(os.path.join(OUT, "ledger_all_strategies.csv"), index_col=0, parse_dates=True)
main = pd.read_csv(os.path.join(OUT, "ledger_main.csv"), index_col=0, parse_dates=True)
rf, mkt = main["rf_daily"], main["y_ret"] - main["rf_daily"]

def newey_west(y, X, lag):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return b, np.sqrt(np.diag(V)), r2

T = len(mkt); LAG = int(math.floor(4 * (T / 100) ** (2 / 9)))
X = np.column_stack([np.ones(T), mkt.values])
names = [c.split("|")[0] for c in wide.columns if c.endswith("|r_net")]
rows = []
for n in names:
    if n in ("cash", "buy & hold SPY"):
        continue
    y = (wide[f"{n}|r_net"] - rf).values
    b, se, r2 = newey_west(y, X, LAG)
    t_a = b[0] / se[0]
    p_a = math.erfc(abs(t_a) / math.sqrt(2))      # two-sided, normal approximation
    rows.append({"strategy": n, "alpha_ann": b[0] * ANN, "alpha_ci_lo_95": (b[0] - 1.96 * se[0]) * ANN,
                 "alpha_ci_hi_95": (b[0] + 1.96 * se[0]) * ANN, "alpha_t": t_a, "alpha_p": p_a,
                 "beta": b[1], "beta_se": se[1], "r2": r2, "n_days": T, "nw_lag": LAG})
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "capm_alpha.csv"), index=False)
pd.set_option("display.width", 200)
print(f"{period}: T={T}, Newey-West lag={LAG}")
print(tab[["strategy", "alpha_ann", "alpha_ci_lo_95", "alpha_ci_hi_95", "alpha_t", "alpha_p", "beta", "r2"]].round(4).to_string(index=False))
