"""Part B entry point: CV -> selection -> robustness -> walk-forward -> outputs.

Run from the repository root:
    python src/run_partB.py
"""

import json
from pathlib import Path

import pandas as pd

import model as M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "partB"


def _params_for(row: pd.Series, grids: dict) -> dict:
    fam = row["family"]
    return {k: row[k] for k in grids[fam].keys()}


def main():
    cfg = M.load_config()
    splits = M.load_splits(cfg)
    panel = M.load_panel(cfg)
    groups = M.load_groups(cfg)

    m = cfg["model"]
    label, ar_feature = m["label"], m["ar_feature"]
    clip, seed = m["clip_sigma"], m["seed"]
    families, feature_sets, grids = m["families"], m["feature_sets"], m["grids"]
    val_years, embargo = splits["cv_val_years"], splits["embargo_days"]

    dev = panel[panel.period == "dev"].dropna(subset=[label])
    print(f"[B] dev rows = {len(dev)} ; val years = {val_years} ; embargo = {embargo}")

    # ---- 1. cross-validation on the full dev period ------------------------- #
    cv_df = M.run_cv(dev, groups, families, feature_sets, grids, val_years, embargo, label, clip)
    selected, champion = M.select_best(cv_df)
    ar1_r2 = M.baseline_cv(dev, label, ar_feature, val_years, embargo)

    n_grid = sum(len(M.expand_grid(grids[f])) for f in families)
    print(f"[B] tried {n_grid} grid points x {len(feature_sets)} feature sets "
          f"= {len(cv_df)} configs, each over {len(val_years)} folds")
    print(f"[B] AR(1) mean val OOS R2 = {ar1_r2:.4f}")
    print("[B] selected (best alpha per family x feature set):")
    print(selected.to_string(index=False))
    champ_name = f"{champion['family']}_{champion['feature_set']}"
    print(f"[B] CHAMPION = {champ_name}  (mean val OOS R2 = {champion['mean_oos_r2']:.4f})")

    # ---- 2. robustness: same CV on dev restricted to >= min_date ------------ #
    min_date = pd.Timestamp(cfg["robustness"]["min_date"])
    dev01 = dev[dev.index >= min_date]
    cv01 = M.run_cv(dev01, groups, families, feature_sets, grids, val_years, embargo, label, clip)
    sel01, champ01 = M.select_best(cv01)
    champ01_name = f"{champ01['family']}_{champ01['feature_set']}"
    print(f"[B] robustness ({min_date.date()}+, {len(dev01)} rows): "
          f"champion = {champ01_name}, mean val OOS R2 = {champ01['mean_oos_r2']:.4f}")

    # ---- 2b. RFF + Ridge (Kelly-style complexity exploration) -------------- #
    rff_cfg = cfg["rff"]
    rff_feats = groups[rff_cfg["base_feature_set"]]
    rff_df = M.run_rff_cv(dev, rff_feats, rff_cfg["n_components"], rff_cfg["alpha"],
                          val_years, embargo, label, clip, rff_cfg["gamma"], seed)
    rff_best = rff_df.loc[rff_df.groupby("n_components")["mean_oos_r2"].idxmax()]
    print("[B] RFF + Ridge (base = M3): best mean val OOS R2 per n_components:")
    print(rff_best.to_string(index=False))

    # ---- 2c. label winsorization sensitivity -------------------------------- #
    wcfg = cfg["winsor"]
    cv_w = M.run_cv(dev, groups, families, feature_sets, grids, val_years, embargo,
                    label, clip, winsor=(wcfg["lo"], wcfg["hi"]))
    sel_w, champ_w = M.select_best(cv_w)
    champ_w_name = f"{champ_w['family']}_{champ_w['feature_set']}"
    print(f"[B] winsor({wcfg['lo']}/{wcfg['hi']}): champion = {champ_w_name}, "
          f"mean val OOS R2 = {champ_w['mean_oos_r2']:.4f}  "
          f"(main: {champ_name} {champion['mean_oos_r2']:.4f})")

    # ---- 3. walk-forward on the test period (hyperparams frozen) ------------ #
    test_idx = panel[panel.period == "test"].index
    preds = pd.DataFrame(index=test_idx)
    preds["y_exret"] = panel.loc[test_idx, label]
    preds["mean"] = M.mean_walkforward(panel, label, embargo)
    preds["ar1"] = M.ar1_walkforward(panel, label, ar_feature, embargo)

    coef_paths = {}
    for _i, row in selected.iterrows():
        fam, fs = row["family"], row["feature_set"]
        col = f"{fam}_{fs}"
        p, coefs = M.model_walkforward(panel, groups[fs], fam, _params_for(row, grids),
                                       label, embargo, clip)
        preds[col] = p
        coef_paths[col] = pd.DataFrame.from_dict(coefs, orient="index", columns=groups[fs])

    preds["champion"] = preds[champ_name]

    # test OOS R2 of every forecast column vs the expanding mean
    def _test_r2(col):
        mm = preds.dropna(subset=["y_exret"])
        return M.oos_r2(mm["y_exret"].to_numpy(), mm[col].to_numpy(),
                        mm["mean"].to_numpy())

    test_r2s = {c: _test_r2(c) for c in preds.columns if c not in ("y_exret", "mean")}
    print("[B] test OOS R2 (vs expanding mean):")
    for c, v in test_r2s.items():
        print(f"    {c:<20s} {v:+.4f}")

    # ---- 4. save outputs ---------------------------------------------------- #
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["paths"]["predictions"]).parent.mkdir(parents=True, exist_ok=True)

    cv_df.to_csv(OUT / "cv_results.csv", index=False)
    selected.to_csv(OUT / "selection.csv", index=False)
    cv01.to_csv(OUT / "cv_results_2001plus.csv", index=False)
    sel01.to_csv(OUT / "selection_2001plus.csv", index=False)
    (OUT / "baselines.json").write_text(
        json.dumps({"mean_oos_r2": 0.0, "ar1_mean_oos_r2": ar1_r2}, indent=2))
    (OUT / "test_oos_r2.json").write_text(json.dumps(test_r2s, indent=2))
    (OUT / "champion.json").write_text(
        json.dumps({"name": champ_name, **champion.to_dict()}, indent=2))
    (OUT / "robustness.json").write_text(json.dumps({
        "min_date": str(min_date.date()),
        "full_dev_champion": {"name": champ_name, "mean_oos_r2": champion["mean_oos_r2"]},
        "restricted_champion": {"name": champ01_name, "mean_oos_r2": champ01["mean_oos_r2"]},
    }, indent=2))
    rff_df.to_csv(OUT / "rff_cv.csv", index=False)
    cv_w.to_csv(OUT / "cv_results_winsor.csv", index=False)
    sel_w.to_csv(OUT / "selection_winsor.csv", index=False)
    (OUT / "winsor.json").write_text(json.dumps({
        "lo": wcfg["lo"], "hi": wcfg["hi"],
        "main_champion": {"name": champ_name, "mean_oos_r2": champion["mean_oos_r2"]},
        "winsor_champion": {"name": champ_w_name, "mean_oos_r2": champ_w["mean_oos_r2"]},
    }, indent=2))

    preds.to_parquet(ROOT / cfg["paths"]["predictions"])
    coef_paths[champ_name].to_csv(OUT / "coef_path_champion.csv")

    # feature selection frequency across monthly refits (answers "does LASSO drop G3?")
    for fam in ("lasso", "elasticnet"):
        col = f"{fam}_M3"
        if col in coef_paths:
            (coef_paths[col] != 0).mean(axis=0).rename("selection_frequency").to_csv(
                OUT / f"selection_freq_{col}.csv")

    # champion validation predictions (for C to check the flow / val OOS R2)
    vp = M.val_predictions(dev, groups[champion["feature_set"]], champion["family"],
                           _params_for(champion, grids), label, val_years, embargo, clip)
    vp.to_parquet(OUT / "val_predictions_champion.parquet")

    _write_summary(cfg, splits, cv_df, selected, champion, champ_name, ar1_r2,
                   champ01_name, champ01, test_r2s, len(cv_df), n_grid,
                   rff_best, champ_w_name, champ_w)

    print(f"[B] done. outputs -> {OUT}")
    print(f"[B] predictions -> {ROOT / cfg['paths']['predictions']}")


def _write_summary(cfg, splits, cv_df, selected, champion, champ_name, ar1_r2,
                   champ01_name, champ01, test_r2s, n_configs, n_grid,
                   rff_best, champ_w_name, champ_w):
    lines = []
    a = lines.append
    a("# Part B — model & validation summary")
    a("")
    a("## Setup (all pre-declared)")
    a(f"- label: `{cfg['model']['label']}` (open-to-open next-day SPY excess return)")
    a(f"- features: M1 (14) / M2 (18) / M3 (24), unscaled")
    a(f"- scaling: per-fold StandardScaler + clip to ±{cfg['model']['clip_sigma']}σ")
    a(f"- baselines: historical (expanding) mean ; AR(1) via `{cfg['model']['ar_feature']}` lag")
    a(f"- models: {', '.join(cfg['model']['families'])}")
    a(f"- grid: {n_grid} hyperparameter points per feature set → {n_configs} configs total")
    a(f"- validation: expanding window, one year per fold {splits['cv_val_years']}, "
      f"embargo = {splits['embargo_days']} rows")
    a(f"- selection rule: {cfg['model']['selection_rule']}")
    a(f"- test: freeze hyperparameters, monthly walk-forward refit, {splits['test_start']} → "
      f"{splits['test_end']}")
    a(f"- seed: {cfg['model']['seed']}")
    a("")
    a("## Baselines (mean validation OOS R2)")
    a(f"- historical mean: 0.0 by construction")
    a(f"- AR(1) (ret_cc lag): {ar1_r2:.4f}")
    a("")
    a("## Selection (best alpha per family × feature set, by mean validation OOS R2)")
    a("")
    a("```")
    a(selected.to_string(index=False))
    a("```")
    a("")
    a(f"### Champion: **{champ_name}**  (mean val OOS R2 = {champion['mean_oos_r2']:.4f})")
    a("")
    a("## Robustness (same CV, dev restricted to 2001+)")
    a(f"- full-dev champion: {champ_name} ({champion['mean_oos_r2']:.4f})")
    a(f"- 2001+ champion: {champ01_name} ({champ01['mean_oos_r2']:.4f})")
    a("")
    a("## Label winsorization sensitivity (1%/99% clipped per train fold)")
    a(f"- main champion: {champ_name} ({champion['mean_oos_r2']:.4f})")
    a(f"- winsorized champion: {champ_w_name} ({champ_w['mean_oos_r2']:.4f})")
    a("")
    a("## RFF + Ridge complexity exploration (Kelly et al. 2024 style, base = M3)")
    a("")
    a("```")
    a(rff_best.to_string(index=False))
    a("```")
    a("")
    a("## Test-period out-of-sample R2 (vs expanding mean)")
    a("")
    a("```")
    for c, v in test_r2s.items():
        a(f"{c:<20s} {v:+.4f}")
    a("```")
    a("")
    a("Files: `cv_results.csv` (full grid), `selection.csv`, `coef_path_champion.csv`, "
      "`selection_freq_lasso_M3.csv` / `_elasticnet_M3.csv`, `val_predictions_champion.parquet`, "
      "`data/predictions.parquet`.")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
