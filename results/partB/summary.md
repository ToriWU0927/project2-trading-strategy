# Part B — model & validation summary

## Setup (all pre-declared)
- label: `y_exret` (open-to-open next-day SPY excess return)
- features: M1 (14) / M2 (18) / M3 (24), unscaled
- scaling: per-fold StandardScaler + clip to ±5.0σ
- baselines: historical (expanding) mean ; AR(1) via `ret_cc` lag
- models: ridge, lasso, elasticnet
- grid: 28 hyperparameter points per feature set → 84 configs total
- validation: expanding window, one year per fold [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014], embargo = 2 rows
- selection rule: max mean validation OOS R2 (vs expanding train mean)
- test: freeze hyperparameters, monthly walk-forward refit, 2015-01-01 → 2026-10-30
- seed: 5205

## Baselines (mean validation OOS R2)
- historical mean: 0.0 by construction
- AR(1) (ret_cc lag): -0.0025

## Selection (best alpha per family × feature set, by mean validation OOS R2)

```
    family feature_set     alpha  mean_oos_r2  l1_ratio
elasticnet          M1     0.001     0.002022       0.5
elasticnet          M2     0.001     0.002022       0.5
elasticnet          M3     0.001     0.002022       0.5
     lasso          M1     0.010     0.000000       NaN
     lasso          M2     0.010     0.000000       NaN
     lasso          M3     0.010     0.000000       NaN
     ridge          M1 10000.000     0.002407       NaN
     ridge          M2 10000.000     0.001883       NaN
     ridge          M3 10000.000     0.001160       NaN
```

### Champion: **ridge_M1**  (mean val OOS R2 = 0.0024)

## Robustness (same CV, dev restricted to 2001+)
- full-dev champion: ridge_M1 (0.0024)
- 2001+ champion: ridge_M1 (0.0009)

## Label winsorization sensitivity (1%/99% clipped per train fold)
- main champion: ridge_M1 (0.0024)
- winsorized champion: ridge_M1 (0.0024)

## RFF + Ridge complexity exploration (Kelly et al. 2024 style, base = M3)

```
 n_components   alpha  mean_oos_r2
           24  1000.0     0.000613
           96 10000.0    -0.000055
          384 10000.0    -0.000016
```

## Test-period out-of-sample R2 (vs expanding mean)

```
ar1                  -0.0024
elasticnet_M1        +0.0024
elasticnet_M2        +0.0024
elasticnet_M3        +0.0024
lasso_M1             +0.0000
lasso_M2             +0.0000
lasso_M3             +0.0000
ridge_M1             +0.0027
ridge_M2             +0.0026
ridge_M3             +0.0029
champion             +0.0027
```

Files: `cv_results.csv` (full grid), `selection.csv`, `coef_path_champion.csv`, `selection_freq_lasso_M3.csv` / `_elasticnet_M3.csv`, `val_predictions_champion.parquet`, `data/predictions.parquet`.