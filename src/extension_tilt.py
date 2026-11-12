"""POST-HOC EXTENSION: buy-and-hold base + model tilt.  Protocol: docs/extension_tilt_protocol.md

w_t = 1 + d_t,  d_t = cost-aware projection of (mu_t - mubar_t)/(gamma*sigma_bar^2) onto [lo-1, hi-1].
Revision R1 (2026-11-09): with a zero incremental signal the rule does NOT reach buy & hold
(starting from cash, the no-trade band stops it at 0.595 for T1/T2 and 0.75 for T3). Each variant
therefore gets a same-rule zero-signal control (mu - mubar = 0, everything else identical);
(tilt - control) measures the model's incremental value, (tilt - buy & hold) the investment value.

Usage (from the repository root):
  python src/extension_tilt.py --period val    # selects the variant (written once to results/ext_tilt_selection.json)
  python src/extension_tilt.py --period test   # one-shot evaluation, all variants reported
"""
import argparse, json, os, sys, warnings, datetime, hashlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.dirname(__file__))
from backtest import file_sha256, impl_sha256, selection_mismatches, aligned_index, calib_vol, realised_window, ANN, load_parquet, w_cost_aware, run, metrics, paired_bootstrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))["partC"]
P = lambda k: os.path.join(ROOT, cfg["inputs"][k])
ap = argparse.ArgumentParser(); ap.add_argument("--period", choices=["val", "test"], required=True)
period = ap.parse_args().period
OUT = os.path.join(ROOT, cfg["output_dir"], period); os.makedirs(OUT, exist_ok=True)
SEL = os.path.join(ROOT, cfg["output_dir"], "ext_tilt_selection.json")
if period == "test" and not os.path.exists(SEL):
    sys.exit("Run --period val first: the variant must be selected on validation before the test run.")

panel = load_parquet(P("panel"))
if period == "test":
    pred = load_parquet(P("test_predictions")); mu_s, mean_s = pred["champion"], pred["mean"]
else:
    pred = load_parquet(P("val_predictions")); mu_s, mean_s = pred["y_hat"], pred["train_mean"]
idx, _dropped = aligned_index(panel, pred, period, [c for c in pred.columns if c not in ("y_true", "y_exret")])
f = panel.loc[idx]
y, rf, y_ex, vol = f["y_ret"].values, f["rf_daily"].values, f["y_exret"].values, f["vol_rv_20"].values
mu, mubar = mu_s.loc[idx].values, mean_s.loc[idx].values
SIG = calib_vol(panel, idx[0]).median()
G, W0, C = cfg["position"]["gamma_main"], cfg["accounting"]["initial_weight"], cfg["costs"]["main_bp"] / 1e4
s2 = np.full(len(idx), SIG ** 2 / ANN)
VARIANTS = {"T1_under": (0.5, 1.0), "T2_sym": (0.5, 1.5), "T3_small": (0.75, 1.25)}

def tilt(lo, hi, signal):
    d = w_cost_aware(signal, s2, G, C, w0=W0 - 1.0, lo=lo - 1.0, hi=hi - 1.0)
    return 1.0 + d
ZERO = lambda v: f"{v} zero-signal control"

rows, leds = [], {}
def add(name, w):
    led = run(w, y, rf, C, W0, idx); leds[name] = led
    m = metrics(led, 3, name); wv = led.w.values
    m["exposure_part"] = wv.mean() * y_ex.mean() * ANN
    m["timing_part"] = np.mean((wv - wv.mean()) * (y_ex - y_ex.mean())) * ANN
    m["share_days_w_above_1"] = (wv > 1 + 1e-9).mean(); m["share_days_w_below_1"] = (wv < 1 - 1e-9).mean()
    m["min_w"], m["max_w"] = wv.min(), wv.max(); rows.append(m)
for v, (lo, hi) in VARIANTS.items():
    add(v, tilt(lo, hi, mu - mubar))
for v, (lo, hi) in VARIANTS.items():
    add(ZERO(v), tilt(lo, hi, np.zeros(len(idx))))
add("buy & hold SPY", np.ones(len(idx)))
add("main: cost-aware champion", w_cost_aware(mu, vol ** 2 / ANN, G, C, W0))
add("target-vol SPY", np.minimum(SIG / vol, 1.0))
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(OUT, "ext_tilt_performance.csv"), index_label="strategy")
BH = "buy & hold SPY"

_cw = calib_vol(panel, idx[0])
json.dump({"period": period, "calibration_window": [str(_cw.index.min().date()), str(_cw.index.max().date())],
           "first_decision_date": str(idx[0].date()), "ends_before_first_decision": bool(_cw.index.max() < idx[0]),
           "sigma_bar": SIG}, open(os.path.join(OUT, os.path.basename(SEL).replace("selection", "calibration")), "w"), indent=2)
cand = tab.loc[list(VARIANTS)]
CURRENT = {"cost_bp": C * 1e4, "gamma": G, "sigma_bar": SIG,
           "variants": {k: list(v) for k, v in VARIANTS.items()},
           "selected_variant": cand.sort_values(["CE_ann", "sharpe"], ascending=False).index[0],
           "validation_CE": cand.CE_ann.to_dict(), "validation_sharpe": cand.sharpe.to_dict()}
TEST_KEYS = ["cost_bp", "gamma", "variants"]

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
    json.dump({"selected_variant": best, "criterion": "validation net CE (gamma=3), tie -> Sharpe",
               "cost_bp": C * 1e4, "gamma": G, "sigma_bar": SIG, "variants": VARIANTS,
               "validation_CE": cand.CE_ann.to_dict(), "validation_sharpe": cand.sharpe.to_dict(),
               "selected_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
               "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()},
              open(SEL, "w"), indent=2)
    print("SELECTED on validation:", best)
    json.dump({"checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
               "selection_file": os.path.basename(SEL), "reproduced": True, "checked_keys": sorted(CURRENT), **FP},
              open(CHECK, "w"), indent=2)
sel = json.load(open(SEL))["selected_variant"]

bs = []
for v in VARIANTS:
    for b in [ZERO(v), BH, "main: cost-aware champion", "target-vol SPY"]:
        d = paired_bootstrap(leds[v].r_net_ex.values, leds[b].r_net_ex.values, 63, 2000, 5205, 3)
        d.insert(0, "A", v); d.insert(1, "B", b); d.insert(2, "selected_on_validation", v == sel); bs.append(d)
pd.concat(bs).to_csv(os.path.join(OUT, "ext_tilt_bootstrap.csv"), index=False)

names = list(VARIANTS) + [ZERO(v) for v in VARIANTS] + [BH, "main: cost-aware champion"]
yr = pd.DataFrame({n: (1 + leds[n].r_net).groupby(idx.year).prod() - 1 for n in names})
yr.to_csv(os.path.join(OUT, "ext_tilt_yearly.csv"), index_label="year")
if period == "test":
    ev = []
    for e, win in cfg["events"].items():
        if e.startswith("_"): continue
        m_ = (idx >= win[0]) & (idx <= win[1]); ex0, re1 = realised_window(panel.index, idx[m_])
        r = {"event": e, "decision_start": win[0], "decision_end": win[1], "returns_from_open": ex0, "returns_to_open": re1}
        for n in names: r[n] = (1 + leds[n].r_net[m_]).prod() - 1
        for v in VARIANTS: r[v + " avg w"] = leds[v].w[m_].mean()
        ev.append(r)
    pd.DataFrame(ev).to_csv(os.path.join(OUT, "ext_tilt_events.csv"), index=False)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"T1_under": "#7aa6e0", "T2_sym": "#1f4e9c", "T3_small": "#2e8b57", ZERO(sel): "#999999", BH: "#222222", "main: cost-aware champion": "#b5651d"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.5, 1.2, 1]})
for n, c in COL.items():
    lab = n + ("  ← selected on validation" if n == sel else "")
    ax[0].plot(idx, leds[n].nav, color=c, lw=2.2 if n == sel else 1.1, ls="--" if n == BH else (":" if n == ZERO(sel) else "-"), label=lab)
ax[0].set_yscale("log"); ax[0].legend(fontsize=7, frameon=False); ax[0].grid(alpha=.3)
ax[0].set_title(f"(a) {period}: NAV, net of {C*1e4:g}bp, gamma={G}", fontsize=9)
ax[1].plot(idx, leds[sel].w, color=COL[sel], lw=.5)
ax[1].plot(idx, pd.Series(leds[sel].w.values, idx).rolling(63).mean(), color="k", lw=1.2, label="63-day average")
ax[1].axhline(1, color="grey", ls="--", lw=.8); ax[1].legend(fontsize=7, loc="lower right", framealpha=.9); ax[1].grid(alpha=.3)
ax[1].set_ylabel("SPY weight"); ax[1].set_title(f"(b) weight of selected variant {sel}", fontsize=9)
ks = list(COL)
ax[2].bar(range(len(ks)), [tab.loc[k, "sharpe"] for k in ks], color=[COL[k] for k in ks])
ax[2].set_xticks(range(len(ks))); ax[2].set_xticklabels(["T1", "T2", "T3", f"{sel[:2]} zero", "buy&hold", "main"], fontsize=8)
for i, k in enumerate(ks): ax[2].text(i, tab.loc[k, "sharpe"], f"{tab.loc[k, 'sharpe']:.2f}", ha="center", va="bottom", fontsize=8)
ax[2].set_ylabel("Sharpe (net)"); ax[2].grid(alpha=.3, axis="y"); ax[2].set_title("(c) Sharpe", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ext_tilt.png"), dpi=150); plt.close(fig)

pd.set_option("display.width", 250)
print(tab[["CAGR", "ann_excess", "ann_excess_vol", "sharpe", "max_drawdown", "ann_turnover", "avg_weight", "min_w", "max_w",
           "share_days_w_above_1", "share_days_w_below_1", "CE_ann", "exposure_part", "timing_part"]].round(4).to_string())
b = pd.read_csv(os.path.join(OUT, "ext_tilt_bootstrap.csv")); print(b[b.stat != "ann_excess_diff"].round(3).to_string(index=False))
