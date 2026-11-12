"""Hand-checkable accounting tests for the Part C engine. Run: python src/test_accounting.py"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from backtest import run, w_mv, w_cost_aware, w_sign, load_parquet, metrics, calib_vol, aligned_index, selection_mismatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))["partC"]
panel = load_parquet(os.path.join(ROOT, cfg["inputs"]["panel"]))
results = []

def check(name, ok, detail=""):
    results.append({"test": name, "pass": bool(ok), "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

# --- 1. toy example computed by hand
y = np.array([0.01, -0.02, 0.03]); rf = np.array([0.0001] * 3)
led = run([0, 1, 0], y, rf, c=0.001)
exp_net = np.array([0.0001, -0.02 - 0.001, 0.0001 - 0.001])
check("toy: cash->SPY->cash costs and returns", np.allclose(led.r_net, exp_net), led.r_net.round(6).tolist())
check("toy: identity r_net - rf = w*y_exret - cost",
      np.allclose(led.r_net_ex, led.w * (y - rf) - led.cost))

# --- 2. drift: constant 50% weight still trades after price moves
led = run([0.5, 0.5], [0.10, 0.0], [0.0, 0.0], c=0.0, w0=0.5)
exp_drift = 0.5 * 1.1 / 1.05
check("drift: w_drift after +10% = 0.55/1.05", np.isclose(led.w_drift_before_trade.iloc[1], exp_drift),
      round(led.w_drift_before_trade.iloc[1], 6))
check("drift: exact turnover > 0 while target diff = 0",
      led.turnover.iloc[1] > 0 and led.turnover_approx.iloc[1] == 0)

# --- 3. real data: cash, buy & hold, NAV vs price ratio
t = panel[(panel.period == "test") & panel.y_ret.notna()]
led = run(np.zeros(len(t)), t.y_ret, t.rf_daily, c=0.0005, index=t.index)
check("cash: r_net == rf_daily and no cost", np.allclose(led.r_net, t.rf_daily) and led.cost.sum() == 0)
led = run(np.ones(len(t)), t.y_ret, t.rf_daily, c=0.0, index=t.index)
nxt = panel.px_open.shift(-1); nxt2 = panel.px_open.shift(-2)
ratio = nxt2.loc[t.index[-1]] / nxt.loc[t.index[0]]
check("buy&hold NAV == Open(last+2)/Open(first+1)", np.isclose(led.nav.iloc[-1], ratio, rtol=1e-10),
      f"{led.nav.iloc[-1]:.6f} vs {ratio:.6f}")
led5 = run(np.ones(len(t)), t.y_ret, t.rf_daily, c=0.0005, index=t.index)
check("buy&hold: only entry cost (turnover sum = 1)", np.isclose(led5.turnover.sum(), 1.0)
      and np.isclose(led5.nav.iloc[-1], ratio * (1 - 0.0005) * (1 + 0) / 1, rtol=1e-3))
check("full SPY drift weight stays exactly 1", np.allclose(led5.w_drift_before_trade.iloc[1:], 1.0))

# --- 4. cost-aware rule equals brute-force argmax of its objective
rng = np.random.default_rng(0)
grid = np.linspace(0, 1, 100001)
ok = True
for _ in range(300):
    mu = rng.normal(3e-4, 6e-4); s2 = (rng.uniform(0.08, 0.6) ** 2) / 252
    c = rng.choice([0, 2e-4, 5e-4, 1e-3]); prev = rng.uniform(0, 1); g = rng.choice([1, 3, 5])
    obj = grid * mu - g / 2 * grid ** 2 * s2 - c * np.abs(grid - prev)
    brute = grid[obj.argmax()]
    mine = w_cost_aware([mu], [s2], g, c, w0=prev)[0]
    val_mine = mine * mu - g / 2 * mine ** 2 * s2 - c * abs(mine - prev)
    ok &= val_mine >= obj.max() - 1e-12
check("cost-aware rule = argmax of single-period objective (300 random cases)", ok)
check("cost-aware with c=0 equals pure MV",
      np.allclose(w_cost_aware([1e-4, -1e-4, 5e-4], [1e-4] * 3, 3, 0.0), w_mv([1e-4, -1e-4, 5e-4], [1e-4] * 3, 3)))

# --- 5. timing: weight pairs with same-row y_ret = Open(t+2)/Open(t+1)-1
check("timing: y_ret_t == Open(t+2)/Open(t+1)-1 on test rows",
      np.allclose(t.y_ret, (nxt2 / nxt - 1).loc[t.index]))

# --- 6. R1: tilt rule with zero incremental signal (hand-checkable)
# half-width of the no-trade band = c / (gamma * sigma^2) = 1e-4 / (3 * 0.1440675^2 / 252) = 0.4047
s2b = 0.1440675252213504 ** 2 / 252
hw = 1e-4 / (3 * s2b)
for name, (lo, hi), expect in [("T1", (0.5, 1.0), 1 - hw), ("T2", (0.5, 1.5), 1 - hw), ("T3", (0.75, 1.25), 0.75)]:
    d = w_cost_aware(np.zeros(20), np.full(20, s2b), 3, 1e-4, w0=-1.0, lo=lo - 1, hi=hi - 1)
    check(f"R1 zero signal from cash: {name} weight stays at {expect:.6f} (not buy & hold)",
          np.allclose(1 + d, expect), round(1 + d[-1], 6))
d = w_cost_aware(np.zeros(20), np.full(20, s2b), 3, 1e-4, w0=0.0, lo=-0.5, hi=0.0)
check("R1 zero signal starting fully invested: weight stays at 1", np.allclose(1 + d, 1.0))

# --- 7. R2: calibration data end strictly before the first decision date of each period
for per, start in [("val", pd.Timestamp("2005-01-03")), ("test", pd.Timestamp("2015-01-02"))]:
    cv = calib_vol(panel, start)
    check(f"R2 {per}: calibration window {cv.index.min().date()}..{cv.index.max().date()} ends before {start.date()}",
          cv.index.max() < start, f"median={cv.median():.4f}")

# --- 8. N1: forecast/panel alignment is refused when dates are missing, duplicated, unsorted or unexpected
tp = load_parquet(os.path.join(ROOT, cfg["inputs"]["test_predictions"]))
cols = [c for c in tp.columns if c != "y_exret"]
idx_ok, drop_ok = aligned_index(panel, tp, "test", cols)
check("N1 original test forecasts accepted; only the 2 trailing unlabelled rows dropped",
      len(idx_ok) == len(tp) - 2 and drop_ok == [str(d.date()) for d in tp.index[-2:]], drop_ok)
mid = pd.Timestamp("2016-12-27")
bad_inputs = {"middle date removed": tp.drop(mid),
              "duplicated date": pd.concat([tp, tp.loc[[mid]]]).sort_index(),
              "unsorted dates": tp.iloc[::-1],
              "unexpected extra date": pd.concat([tp, tp.iloc[[0]].set_axis([pd.Timestamp("2014-12-29")])]).sort_index(),
              "non-finite forecast": tp.assign(champion=tp.champion.where(tp.index != mid, np.nan))}
for name, bad in bad_inputs.items():
    try:
        aligned_index(panel, bad, "test", cols); refused, msg = False, "accepted"
    except ValueError as e:
        refused, msg = True, str(e)[:80]
    check(f"N1 refuses test forecasts with {name}", refused, msg)
vp = load_parquet(os.path.join(ROOT, cfg["inputs"]["val_predictions"]))
check("N1 original validation forecasts accepted, nothing dropped",
      aligned_index(panel, vp, "val", ["y_hat", "train_mean"])[1] == [])

# --- 9. N5: a stored selection is rejected when gamma, cost, a variant definition or a validation number differs
rec = {"cost_bp": 1.0, "gamma": 3, "sigma_bar": 0.15427294877110057, "selected_variant": "T1_under",   # self-contained
       "variants": {"T1_under": [0.5, 1.0], "T2_sym": [0.5, 1.5], "T3_small": [0.75, 1.25]},     # (no results/ needed)
       "validation_CE": {"T1_under": 0.027353524516806033, "T2_sym": 0.004855719491233146, "T3_small": 0.016989860855871314}}
same = {k: rec[k] for k in ("cost_bp", "gamma", "sigma_bar", "variants", "selected_variant", "validation_CE")}
check("N5 identical selection record accepted", selection_mismatches(rec, same) == [])
for name, change in {"gamma 5": {"gamma": 5}, "cost 5bp": {"cost_bp": 5.0},
                     "different variant bounds": {"variants": {**rec["variants"], "T1_under": [0.4, 1.0]}},
                     "different validation CE": {"validation_CE": {**rec["validation_CE"], "T1_under": 0.03}},
                     "different selected variant": {"selected_variant": "T3_small"},
                     "different calibrated sigma": {"sigma_bar": rec["sigma_bar"] * 1.001}}.items():
    check(f"N5 selection record rejected for {name}", len(selection_mismatches(rec, {**same, **change})) > 0)
fp = {"data_sha256": {"panel": "a" * 64, "val_predictions": "b" * 64}, "impl_sha256": "c" * 64, "cost_bp": 1.0, "gamma": 3}
check("N5 validation check rejected when an input data file changed",
      len(selection_mismatches(fp, {**fp, "data_sha256": {"panel": "d" * 64, "val_predictions": "b" * 64}})) > 0)
check("N5 validation check rejected when strategy code changed", len(selection_mismatches(fp, {**fp, "impl_sha256": "e" * 64})) > 0)

OUT_DIR = os.path.join(ROOT, cfg["output_dir"]); os.makedirs(OUT_DIR, exist_ok=True)   # N2: clean checkout has no results/
pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "accounting_tests.csv"), index=False)
print(f"{sum(r['pass'] for r in results)}/{len(results)} passed")
sys.exit(0 if all(r["pass"] for r in results) else 1)
