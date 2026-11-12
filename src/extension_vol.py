"""POST-HOC EXTENSION: do not cut exposure just because volatility is high.

Motivation (from diagnostics, see docs/extension_vol_protocol.md): in the highest-
volatility quintile the champion's direction signal separates future returns most
(test: +14.6bp vs -10.8bp/day; validation: +4.3bp vs -4.3bp), yet the main rule
holds its lowest weights there because sigma2 = vol_rv_20^2/252 is in the
denominator.

Pre-declared variants (all: cost-aware rule, gamma=3, main cost from config):
  A_current : sigma_t = vol_rv_20_t                       (frozen main rule)
  B_fixed   : sigma_t = median(vol_rv_20 | dev)            (constant; weight driven by forecast only)
  C_capped  : sigma_t = min(vol_rv_20_t, q75(vol_rv_20 | dev))  (no further cut above the cap)
Selection rule (pre-declared, VALIDATION ONLY): highest net CE (gamma=3); tie -> Sharpe.
The test period is run once, only after the selection file exists.

Usage (from the repository root):
  python src/extension_vol.py --period val    # selects the variant, writes ext_vol_selection.json
  python src/extension_vol.py --period test   # one-shot evaluation of all variants (selected one flagged)
"""
import argparse, json, os, sys, warnings, datetime, hashlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(__file__))
from backtest import file_sha256, impl_sha256, selection_mismatches, aligned_index, calib_vol, realised_window, ANN, load_parquet, w_cost_aware, w_mv, run, metrics, paired_bootstrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))["partC"]
P = lambda k: os.path.join(ROOT, cfg["inputs"][k])
ap = argparse.ArgumentParser(); ap.add_argument("--period", choices=["val", "test"], required=True)
period = ap.parse_args().period
OUT = os.path.join(ROOT, cfg["output_dir"], period); os.makedirs(OUT, exist_ok=True)
SEL = os.path.join(ROOT, cfg["output_dir"], "ext_vol_selection.json")
if period == "test" and not os.path.exists(SEL):
    sys.exit("Run --period val first: the variant must be selected on validation before the test run.")

panel = load_parquet(P("panel"))
if period == "test":
    pred = load_parquet(P("test_predictions")); mu_s, mean_s = pred["champion"], pred["mean"]
else:
    pred = load_parquet(P("val_predictions")); mu_s, mean_s = pred["y_hat"], pred["train_mean"]
idx, _dropped = aligned_index(panel, pred, period, [c for c in pred.columns if c not in ("y_true", "y_exret")])
f = panel.loc[idx]
y, rf, y_ex = f["y_ret"].values, f["rf_daily"].values, f["y_exret"].values
vol = f["vol_rv_20"].values
mu, mu_mean = mu_s.loc[idx].values, mean_s.loc[idx].values
dev_vol = calib_vol(panel, idx[0])
SIG_FIXED, SIG_CAP = dev_vol.median(), dev_vol.quantile(0.75)
G, W0, C = cfg["position"]["gamma_main"], cfg["accounting"]["initial_weight"], cfg["costs"]["main_bp"] / 1e4

SIGMA = {"A_current": vol, "B_fixed": np.full(len(idx), SIG_FIXED), "C_capped": np.minimum(vol, SIG_CAP)}

# ---------------------------------------------------------------- evidence table
L = pd.DataFrame({"vol": vol, "mu": mu, "y": y_ex}, index=idx)
L["vol_q"] = pd.qcut(L.vol, 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
w_main = w_cost_aware(mu, SIGMA["A_current"] ** 2 / ANN, G, C, W0)
L["w_main"] = w_main
gq = L.groupby("vol_q", observed=True)
ev = pd.DataFrame({
    "n_days": gq.size(), "avg_vol": gq.vol.mean(),
    "avg_forecast_bp": gq.mu.mean() * 1e4, "avg_realised_exret_bp": gq.y.mean() * 1e4,
    "realised_bp_when_forecast_up": pd.Series({k: d.y[d.mu > 0].mean() * 1e4 for k, d in gq}),
    "realised_bp_when_forecast_down": pd.Series({k: d.y[d.mu <= 0].mean() * 1e4 for k, d in gq}),
    "direction_hit_rate": pd.Series({k: ((d.mu > 0) == (d.y > 0)).mean() for k, d in gq}),
    "avg_weight_main_rule": gq.w_main.mean()})
ev["spread_up_minus_down_bp"] = ev.realised_bp_when_forecast_up - ev.realised_bp_when_forecast_down
ev.to_csv(os.path.join(OUT, "diag_vol_regime.csv"), index_label="vol_quintile")

# ---------------------------------------------------------------- variants
rows, leds = [], {}
def add(name, w, kind):
    led = run(w, y, rf, C, W0, idx); leds[name] = led
    m = metrics(led, 3, name).astype(object)   # object: holds the text column "kind" (pandas-4 safe)
    wv = led["w"].values
    m["exposure_part"] = wv.mean() * y_ex.mean() * ANN
    m["timing_part"] = np.mean((wv - wv.mean()) * (y_ex - y_ex.mean())) * ANN
    m["avg_weight_top_vol_quintile"] = wv[(L.vol_q == "Q5 high").values].mean()
    m["kind"] = kind; rows.append(m)
for v, s in SIGMA.items():
    add(f"{v} | champion", w_cost_aware(mu, s ** 2 / ANN, G, C, W0), "strategy")
    add(f"{v} | hist-mean (same rule)", w_cost_aware(mu_mean, s ** 2 / ANN, G, C, W0), "benchmark: same rule, no model")
add("buy & hold SPY", np.ones(len(idx)), "benchmark")
add("target-vol SPY", np.minimum(SIG_FIXED / vol, 1.0), "benchmark")
tab = pd.DataFrame(rows).infer_objects()
tab.to_csv(os.path.join(OUT, "ext_vol_performance.csv"), index_label="strategy")

# ---------------------------------------------------------------- selection (validation only)
_cw = calib_vol(panel, idx[0])
json.dump({"period": period, "calibration_window": [str(_cw.index.min().date()), str(_cw.index.max().date())],
           "first_decision_date": str(idx[0].date()), "ends_before_first_decision": bool(_cw.index.max() < idx[0]),
           "sigma_fixed": SIG_FIXED, "sigma_cap": SIG_CAP}, open(os.path.join(OUT, os.path.basename(SEL).replace("selection", "calibration")), "w"), indent=2)
cand = tab.loc[[f"{v} | champion" for v in SIGMA]]
CURRENT = {"cost_bp": C * 1e4, "gamma": G, "sigma_fixed": SIG_FIXED, "sigma_cap": SIG_CAP,
           "selected_variant": cand.sort_values(["CE_ann", "sharpe"], ascending=False).index[0].split(" | ")[0],
           "validation_CE": cand.CE_ann.to_dict(), "validation_sharpe": cand.sharpe.to_dict()}
TEST_KEYS = ["cost_bp", "gamma"]   # sigma differs by design on test (calibrated on data before 2015)

# ---- N5: a stored selection is used only if the current code/data/config reproduce it exactly
CHECK = SEL.replace("_selection.json", "_selection_check.json")
FP = {"data_sha256": {k: file_sha256(P(k)) for k in ("panel", "val_predictions")},
      "impl_sha256": impl_sha256(), "cost_bp": C * 1e4, "gamma": G}
if period == "val" and os.path.exists(SEL):
    _rec = json.load(open(SEL))
    _bad = selection_mismatches(_rec, CURRENT)
    if _bad:
        sys.exit("Stored validation selection is NOT reproduced by the current code/data/config; "
                 "do not overwrite it silently - review, log and re-select explicitly:\n  " + "\n  ".join(_bad))
    json.dump({"checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
               "selection_file": os.path.basename(SEL), "reproduced": True, "checked_keys": sorted(CURRENT), **FP},
              open(CHECK, "w"), indent=2)
    print("Selection already recorded and reproduced exactly by the current code/data; not overwritten:", _rec["selected_variant"])
elif period == "test":
    _rec = json.load(open(SEL))
    if not os.path.exists(CHECK):
        sys.exit("Run --period val first: it verifies that the stored selection is reproduced by the current code/data.")
    _chk = json.load(open(CHECK))
    _bad = selection_mismatches(_chk, FP) + selection_mismatches(_rec, {k: CURRENT[k] for k in TEST_KEYS})
    if _bad:
        sys.exit("Validation check is stale for the current code/data/config - rerun --period val:\n  " + "\n  ".join(_bad))
if period == "val" and not os.path.exists(SEL):
    best = CURRENT["selected_variant"]
    sel = {"selected_variant": best, "criterion": "validation net CE (gamma=3), tie -> Sharpe",
           "cost_bp": C * 1e4, "gamma": G, "sigma_fixed": SIG_FIXED, "sigma_cap": SIG_CAP,
           "validation_CE": cand.CE_ann.to_dict(), "validation_sharpe": cand.sharpe.to_dict(),
           "selected_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
           "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    json.dump(sel, open(SEL, "w"), indent=2)
    print("SELECTED on validation:", best)
    json.dump({"checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
               "selection_file": os.path.basename(SEL), "reproduced": True, "checked_keys": sorted(CURRENT), **FP},
              open(CHECK, "w"), indent=2)
sel = json.load(open(SEL))["selected_variant"]

# ---------------------------------------------------------------- bootstrap: selected vs current / benchmarks
bs = []
pairs = []
for v in SIGMA:
    pairs += [(f"{v} | champion", "A_current | champion"), (f"{v} | champion", "buy & hold SPY"),
              (f"{v} | champion", f"{v} | hist-mean (same rule)"), (f"{v} | champion", "target-vol SPY")]
for a, b in pairs:
    if a == b: continue
    d = paired_bootstrap(leds[a].r_net_ex.values, leds[b].r_net_ex.values, 63, 2000, 5205, 3)
    d.insert(0, "A", a); d.insert(1, "B", b); d.insert(2, "selected_on_validation", a.startswith(sel)); bs.append(d)
pd.concat(bs).to_csv(os.path.join(OUT, "ext_vol_bootstrap.csv"), index=False)

# yearly / events for selected vs current
yrs = idx.year
yr = pd.DataFrame({n: (1 + leds[n].r_net).groupby(yrs).prod() - 1
                   for n in ["A_current | champion", "B_fixed | champion", "C_capped | champion", "buy & hold SPY"]})
yr.to_csv(os.path.join(OUT, "ext_vol_yearly.csv"), index_label="year")
if period == "test":
    ev_rows = []
    for e, win in cfg["events"].items():
        if e.startswith("_"): continue
        msk = (idx >= win[0]) & (idx <= win[1])
        ex0, re1 = realised_window(panel.index, idx[msk])
        r = {"event": e, "decision_start": win[0], "decision_end": win[1], "returns_from_open": ex0, "returns_to_open": re1}
        for n in ["A_current | champion", "B_fixed | champion", "C_capped | champion", "buy & hold SPY"]:
            r[n] = (1 + leds[n].r_net[msk]).prod() - 1; r[n + " avg w"] = leds[n].w[msk].mean()
        ev_rows.append(r)
    pd.DataFrame(ev_rows).to_csv(os.path.join(OUT, "ext_vol_events.csv"), index=False)

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1.5, 1]})
xq = np.arange(5)
ax[0].bar(xq - .2, ev.realised_bp_when_forecast_up, .4, color="#1f4e9c", label="model says UP")
ax[0].bar(xq + .2, ev.realised_bp_when_forecast_down, .4, color="#c0392b", label="model says DOWN")
ax2 = ax[0].twinx(); ax2.plot(xq, ev.avg_weight_main_rule, "k-o", lw=1.2, label="main-rule avg weight")
ax2.set_ylim(0, 1); ax2.set_ylabel("avg SPY weight (current rule)")
ax[0].set_xticks(xq); ax[0].set_xticklabels(ev.index, fontsize=8); ax[0].axhline(0, color="k", lw=.6)
ax[0].set_ylabel("realised next-period excess return (bp/day)")
ax[0].set_title(f"(a) {period}: realised return by volatility quintile when the\nmodel says up / down, and the current rule's average weight", fontsize=9)
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[0].legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
COL = {"A_current | champion": "#7aa6e0", "B_fixed | champion": "#1f4e9c", "C_capped | champion": "#2e8b57",
       "buy & hold SPY": "#222222"}
for n, c in COL.items():
    ax[1].plot(idx, leds[n].nav, color=c, lw=2.2 if n.startswith(sel) else 1.1,
               ls="--" if n == "buy & hold SPY" else "-", label=n + ("  ← selected on validation" if n.startswith(sel) else ""))
ax[1].set_yscale("log"); ax[1].legend(fontsize=7, frameon=False); ax[1].grid(alpha=.3)
ax[1].set_title(f"(b) {period}: NAV, net of {C*1e4:g}bp, gamma={G}", fontsize=9)
names = list(COL)
ax[2].bar(range(len(names)), [tab.loc[n, "sharpe"] for n in names], color=[COL[n] for n in names])
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels([n.split(" |")[0] for n in names], fontsize=8, rotation=15)
ax[2].set_ylabel("Sharpe (net)"); ax[2].grid(alpha=.3, axis="y"); ax[2].set_title("(c) Sharpe", fontsize=9)
for i, n in enumerate(names): ax[2].text(i, tab.loc[n, "sharpe"], f"{tab.loc[n, 'sharpe']:.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ext_vol.png"), dpi=150); plt.close(fig)

pd.set_option("display.width", 250)
print(ev.round(3).to_string())
print(tab[["CAGR", "ann_excess", "ann_excess_vol", "sharpe", "max_drawdown", "ann_turnover", "avg_weight",
           "avg_weight_top_vol_quintile", "CE_ann", "exposure_part", "timing_part"]].round(4).to_string())
print(pd.read_csv(os.path.join(OUT, "ext_vol_bootstrap.csv")).round(4).to_string(index=False))
