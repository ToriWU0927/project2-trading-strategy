"""Part C: turn Part B forecasts into SPY/cash trades and evaluate.

Usage (from the repository root):
    python src/test_accounting.py          # accounting checks
    python src/run_partC.py --period val   # development check on validation forecasts (2005-2014)
    python src/run_partC.py --period test  # test evaluation with the frozen rules (deterministic, 2015-2026)
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from backtest import (aligned_index, calib_vol, realised_window, ANN, load_parquet, w_mv, w_cost_aware, w_sign, run, metrics, paired_bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(ROOT, "config.json")))["partC"]
P = lambda k: os.path.join(ROOT, cfg["inputs"][k])

ap = argparse.ArgumentParser(); ap.add_argument("--period", choices=["val", "test"], required=True)
period = ap.parse_args().period
OUT = os.path.join(ROOT, cfg["output_dir"], period); os.makedirs(OUT, exist_ok=True)

panel = load_parquet(P("panel"))
if period == "test":
    pred = load_parquet(P("test_predictions"))
    mu_main_full, mu_mean_full = pred[cfg["signal"]["main"]], pred[cfg["signal"]["mean_benchmark_test"]]
else:
    pred = load_parquet(P("val_predictions"))
    mu_main_full, mu_mean_full = pred[cfg["signal"]["val_column"]], pred[cfg["signal"]["mean_benchmark_val"]]

# rows with a realised holding-period return only (never fill missing labels with 0)
idx, dropped = aligned_index(panel, pred, period, [c for c in pred.columns if c not in ("y_true", "y_exret")])
f = panel.loc[idx]
y, rf = f["y_ret"].values, f["rf_daily"].values
y_ex = f["y_exret"].values
sigma2 = (f["vol_rv_20"] ** 2 / ANN).values
mu_main, mu_mean = mu_main_full.loc[idx].values, mu_mean_full.loc[idx].values
sigma_target = calib_vol(panel, idx[0]).median()

pos = cfg["position"]; G0 = pos["gamma_main"]; C_LIST = [b / 1e4 for b in cfg["costs"]["scenarios_bp"]]
C0 = cfg["costs"]["main_bp"] / 1e4; W0 = cfg["accounting"]["initial_weight"]
assert not np.isnan(mu_main).any() and not np.isnan(sigma2).any()


def strategies(c, g=G0):
    """All position paths for cost scenario c and risk aversion g. Names are stable keys."""
    return {
        "main: cost-aware champion": w_cost_aware(mu_main, sigma2, g, c, W0),
        "MV champion": w_mv(mu_main, sigma2, g),
        "sign champion": w_sign(mu_main),
        "cost-aware hist-mean": w_cost_aware(mu_mean, sigma2, g, c, W0),
        "MV hist-mean": w_mv(mu_mean, sigma2, g),
        "buy & hold SPY": np.ones(len(idx)),
        "cash": np.zeros(len(idx)),
        "MA200 trend": (f["mom_ma_200"].values > 0).astype(float),
        "target-vol SPY": np.minimum(sigma_target / f["vol_rv_20"].values, 1.0),
    }


def decompose(led):
    """gross excess = mean(w)*mean(y_ex) [exposure] + cov(w, y_ex) [timing]; net = gross - cost."""
    w = led["w"].values
    return {"exposure_part": w.mean() * y_ex.mean() * ANN,
            "timing_part": np.mean((w - w.mean()) * (y_ex - y_ex.mean())) * ANN,
            "cost_part": -led["cost"].mean() * ANN}


def table(c, g=G0):
    rows, leds = [], {}
    for name, w in strategies(c, g).items():
        led = run(w, y, rf, c, W0, idx); leds[name] = led
        m = metrics(led, cfg["metrics"]["certainty_equivalent_gamma"], name)
        for k, v in decompose(led).items(): m[k] = v
        rows.append(m)
    return pd.DataFrame(rows), leds


# ------------------------------------------------------------ 1. main table
main_tab, leds = table(C0)
main_tab.insert(0, "cost_bp", cfg["costs"]["main_bp"]); main_tab.insert(1, "gamma", G0)
main_tab.to_csv(os.path.join(OUT, "performance_main.csv"), index_label="strategy")

# vol-matched buy & hold (ex-post, descriptive)
s_main = leds["main: cost-aware champion"]["r_net_ex"].std(ddof=1)
bh = leds["buy & hold SPY"]; k = s_main / bh["r_net_ex"].std(ddof=1)
vm = run(np.full(len(idx), k), y, rf, 0.0, k, idx)  # constant weight k, entry cost ignored (ex-post scaling)
vm_m = metrics(vm, 3, "buy & hold scaled to main vol (ex-post)"); vm_m["scale_k"] = k

# ------------------------------------------------------------ 2. cost scenarios
cost_rows = []
for c in C_LIST:
    t_, _ = table(c); t_.insert(0, "cost_bp", round(c * 1e4, 1)); cost_rows.append(t_)
cost_tab = pd.concat(cost_rows); cost_tab.to_csv(os.path.join(OUT, "performance_by_cost.csv"), index_label="strategy")

# ------------------------------------------------------------ 3. gamma sensitivity (pre-specified)
gam_rows = []
for g in [G0] + pos["gamma_sensitivity"]:
    for c in C_LIST:
        st = strategies(c, g)
        for name in ["main: cost-aware champion", "MV champion", "cost-aware hist-mean", "MV hist-mean"]:
            led = run(st[name], y, rf, c, W0, idx)
            m = metrics(led, 3, name); m["gamma"] = g; m["cost_bp"] = round(c * 1e4, 1); gam_rows.append(m)
gam_tab = pd.DataFrame(gam_rows).sort_values(["gamma", "cost_bp"])
gam_tab.to_csv(os.path.join(OUT, "performance_by_gamma.csv"), index_label="strategy")

# ------------------------------------------------------------ 4. yearly & events (main cost, gamma)
KEY = ["main: cost-aware champion", "MV champion", "cost-aware hist-mean", "MV hist-mean",
       "buy & hold SPY", "MA200 trend", "target-vol SPY", "cash"]     # strategies drawn in the NAV figure
EXPORT = KEY + [n for n in leds if n not in KEY]                    # every strategy goes into the CSV exports (N6)
yr = pd.DataFrame({n: (1 + leds[n]["r_net"]).groupby(idx.year).prod() - 1 for n in EXPORT})
yr["main avg weight"] = leds[KEY[0]]["w"].groupby(idx.year).mean()
yr["main turnover"] = leds[KEY[0]]["turnover"].groupby(idx.year).sum()
yr["main n_days"] = leds[KEY[0]]["w"].groupby(idx.year).size()
yr.to_csv(os.path.join(OUT, "yearly_returns.csv"), index_label="year")

ev_rows = []
if period == "test":
    for ev, win in cfg["events"].items():
        if ev.startswith("_"): continue
        a, b = win
        msk = (idx >= a) & (idx <= b)
        ex0, re1 = realised_window(panel.index, idx[msk])
        r = {"event": ev, "decision_start": a, "decision_end": b, "returns_from_open": ex0, "returns_to_open": re1,
             "n_days": int(msk.sum())}
        for n in EXPORT:
            r[n] = (1 + leds[n]["r_net"][msk]).prod() - 1
        r["main avg weight"] = leds[KEY[0]]["w"][msk].mean()
        r["MV champion avg weight"] = leds["MV champion"]["w"][msk].mean()
        ev_rows.append(r)
    pd.DataFrame(ev_rows).to_csv(os.path.join(OUT, "event_windows.csv"), index=False)

# ------------------------------------------------------------ 5. ablation: every B forecast, same rules
abl_rows = []
if period == "test":
    yex_full = pred["y_exret"].loc[idx].values
    den = ((yex_full - mu_mean) ** 2).sum()
    for col in cfg["signal"]["ablation_columns"]:
        mu = pred[col].loc[idx].values
        r2 = 1 - ((yex_full - mu) ** 2).sum() / den
        for rule, w in [("cost-aware", w_cost_aware(mu, sigma2, G0, C0, W0)), ("MV", w_mv(mu, sigma2, G0)),
                        ("sign", w_sign(mu))]:
            led = run(w, y, rf, C0, W0, idx)
            m = metrics(led, 3, col).astype(object); m["rule"] = rule; m["oos_r2"] = r2; abl_rows.append(m)
    pd.DataFrame(abl_rows).infer_objects().to_csv(os.path.join(OUT, "ablation_models.csv"), index_label="model")

# ------------------------------------------------------------ 6. bootstrap
bcfg = cfg["metrics"]["bootstrap"]
pairs = [("main: cost-aware champion", "buy & hold SPY"), ("main: cost-aware champion", "MV hist-mean"),
         ("main: cost-aware champion", "cost-aware hist-mean"),
         ("main: cost-aware champion", "target-vol SPY"),
         ("main: cost-aware champion", "MV champion"), ("MV champion", "MV hist-mean"),
         ("MV champion", "buy & hold SPY")]
bs = []
for a, b in pairs:
    d = paired_bootstrap(leds[a]["r_net_ex"].values, leds[b]["r_net_ex"].values,
                         bcfg["block_days"], bcfg["reps"], bcfg["seed"], 3)
    d.insert(0, "A", a); d.insert(1, "B", b); bs.append(d)
bs = pd.concat(bs); bs.to_csv(os.path.join(OUT, "bootstrap_differences.csv"), index=False)

# ------------------------------------------------------------ 7. ledger
pi = panel.index; loc = pi.get_indexer(idx)
main_led = leds["main: cost-aware champion"].copy()
main_led.insert(0, "exec_date_open", pi[loc + 1]); main_led.insert(1, "realize_date_open", pi[loc + 2])
main_led.insert(2, "mu_hat", mu_main); main_led.insert(3, "mu_hist_mean", mu_mean)
main_led.insert(4, "sigma2_daily", sigma2); main_led.insert(5, "vol_rv_20", f["vol_rv_20"].values)
main_led.insert(6, "y_exret", y_ex)
main_led.to_csv(os.path.join(OUT, "ledger_main.csv"), index_label="decision_date")
wide = pd.concat({n: leds[n][["w", "turnover", "cost", "r_net", "nav"]] for n in EXPORT}, axis=1)
wide.columns = [f"{a}|{b}" for a, b in wide.columns]
wide.to_csv(os.path.join(OUT, "ledger_all_strategies.csv"), index_label="decision_date")

# ------------------------------------------------------------ 8. figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {"main: cost-aware champion": "#1f4e9c", "MV champion": "#7aa6e0", "cost-aware hist-mean": "#b5651d",
       "MV hist-mean": "#e3a869", "buy & hold SPY": "#222222", "MA200 trend": "#2e8b57",
       "target-vol SPY": "#8e44ad", "cash": "#999999"}
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
for n in KEY:
    ax[0].plot(idx, leds[n]["nav"], label=n, color=COL[n], lw=2.2 if n == KEY[0] else 1.1)
    ax[1].plot(idx, leds[n]["drawdown"], color=COL[n], lw=1.6 if n == KEY[0] else 0.8)
ax[0].set_yscale("log"); ax[0].set_ylabel("NAV (log)"); ax[0].legend(fontsize=8, ncol=2, frameon=False)
ax[0].set_title(f"{period}: net of {cfg['costs']['main_bp']}bp cost, gamma={G0}"); ax[1].set_ylabel("drawdown")
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_nav_drawdown.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(idx, leds["MV champion"]["w"], color=COL["MV champion"], lw=.4, label="MV champion (daily)")
ax[0].plot(idx, leds[KEY[0]]["w"], color=COL[KEY[0]], lw=1.2, label="main: cost-aware champion")
ax[0].set_ylabel("SPY weight"); ax[0].legend(fontsize=8, frameon=False)
ax[1].plot(idx, pd.Series(mu_main, idx) * 1e4, lw=.4, color="#555", label="champion mu_hat (bp)")
ax[1].plot(idx, pd.Series(mu_mean, idx) * 1e4, lw=1.2, color=COL["MV hist-mean"], label="hist-mean (bp)")
ax[1].axhline(0, color="k", lw=.6)
for c in [C0]: ax[1].axhline(c * 1e4, color="r", lw=.6, ls="--"); ax[1].axhline(-c * 1e4, color="r", lw=.6, ls="--", label=f"±{c*1e4:.0f}bp")
ax[1].set_ylabel("forecast (bp/day)"); ax[1].legend(fontsize=8, frameon=False)
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_positions_signal.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in ["main: cost-aware champion", "MV champion", "sign champion", "cost-aware hist-mean", "MV hist-mean",
          "buy & hold SPY", "MA200 trend", "target-vol SPY"]:
    s = cost_tab.loc[n]
    ax[0].plot(s["cost_bp"], s["sharpe"], marker="o", label=n, color=COL.get(n, "#c0392b"))
    ax[1].plot(s["cost_bp"], s["CE_ann"] * 100, marker="o", color=COL.get(n, "#c0392b"))
ax[0].set_xlabel("cost (bp per unit one-way turnover)"); ax[0].set_ylabel("Sharpe (excess)")
ax[1].set_xlabel("cost (bp)"); ax[1].set_ylabel("CE, gamma=3 (% p.a.)"); ax[0].legend(fontsize=7, frameon=False)
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cost_sensitivity.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------ 9. summary
meta = {"period": period, "first_decision": str(idx[0].date()), "last_decision": str(idx[-1].date()),
        "n_periods": len(idx), "dropped_rows_without_label": dropped, "sigma_target": sigma_target,
        "calibration_window_for_sigma_target": [str(calib_vol(panel, idx[0]).index.min().date()),
                                                str(calib_vol(panel, idx[0]).index.max().date())],
        "calibration_ends_before_first_decision": bool(calib_vol(panel, idx[0]).index.max() < idx[0]),
        "turnover_exact_vs_approx_main": [leds[KEY[0]]["turnover"].sum(), leds[KEY[0]]["turnover_approx"].sum()],
        "vol_matched_buy_hold": vm_m.to_dict()}
json.dump(meta, open(os.path.join(OUT, "run_meta.json"), "w"), indent=2, default=float)

fmt = main_tab.set_index(main_tab.index)
show = ["CAGR", "ann_excess", "ann_excess_vol", "sharpe", "max_drawdown", "max_dd_duration_days", "ann_turnover",
        "avg_weight", "ann_cost_drag", "CE_ann", "exposure_part", "timing_part", "cost_part"]
pct = {"CAGR", "ann_excess", "ann_excess_vol", "max_drawdown", "ann_cost_drag", "CE_ann", "exposure_part", "timing_part", "cost_part"}
def md(df, cols):
    h = "| strategy | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for n, r in df.iterrows():
        h += f"| {n} | " + " | ".join(f"{r[c]*100:.2f}%" if c in pct else f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return h
with open(os.path.join(OUT, "summary.md"), "w") as fh:
    fh.write(f"# Part C {period} summary (auto-generated)\n\n{meta['first_decision']} → {meta['last_decision']}, "
             f"{len(idx)} holding periods; dropped (no label): {dropped}\n\n"
             f"Main: cost {cfg['costs']['main_bp']}bp, gamma={G0}, w_-1={W0}. sigma_target={sigma_target:.4f}\n\n")
    fh.write(md(main_tab, show))
    fh.write(f"\nEx-post vol-matched buy & hold (k={k:.3f}): ann_excess={vm_m['ann_excess']*100:.2f}%, "
             f"sharpe={vm_m['sharpe']:.2f}, maxDD={vm_m['max_drawdown']*100:.2f}%\n")
    fh.write("\n## Paired block bootstrap (63d, 2000 reps), A − B\n\n| A | B | stat | point | 95% lo | 95% hi | share≤0 |\n|---|---|---|---|---|---|---|\n")
    for _, r in bs.iterrows():
        fh.write(f"| {r.A} | {r.B} | {r.stat} | {r.point:.4f} | {r.ci_lo_95:.4f} | {r.ci_hi_95:.4f} | {r.share_draws_le_0:.3f} |\n")
print(main_tab[["CAGR", "ann_excess", "ann_excess_vol", "sharpe", "max_drawdown", "ann_turnover", "avg_weight", "ann_cost_drag", "CE_ann", "timing_part"]].round(4).to_string())
print("saved to", OUT)
