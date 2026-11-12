# DSA5205 Project 2 — Daily SPY market-timing strategy

Predict the next-day excess return of SPY with shrinkage linear models and turn the forecast into a
long-only SPY/cash position with a cost-aware mean-variance rule.

**Timing convention (used everywhere):** row *t* is a decision date. Features use information up to the
close of *t*; the trade executes at the open of *t+1*; the label is
`y_exret_t = Open_{t+2} / Open_{t+1} − 1 − rf_daily_t`.

| Part | What it does | Code | Outputs |
|---|---|---|---|
| A — data & features | download SPY/VIX (Yahoo) and DTB3/T10Y3M/BAA10Y (FRED), clean, check price adjustments, build 24 features + label, leakage tests, descriptive statistics | `src/data.py`, `checks.py`, `features.py`, `leakage.py`, `eda.py`, `run_partA.py` | `data/` (not committed), `results/partA/` |
| B — models & validation | Ridge / LASSO / Elastic Net on feature sets M1/M2/M3, expanding-window CV 2005–2014 (10 yearly folds, 2-row embargo), monthly walk-forward on the test period 2015–2026 | `src/model.py`, `run_partB.py` | `results/partB/` |
| C — trading & evaluation | cost-aware MV rule (γ = 3, 1 bp main cost), benchmarks, cost/γ sensitivity, bootstrap, break-even costs, CAPM alpha, two exploratory extensions | `src/backtest.py`, `run_partC.py`, `figures.py`, `breakeven.py`, `diagnostics.py`, `capm.py`, `extension_vol.py`, `extension_tilt.py`, `test_accounting.py`, `mini_parquet.py` | `results/partC/` |

`src/run_all.py` runs all three parts in order.

## Requirements

- **Python 3.13** (tested with 3.13.9 on macOS). Exact package versions are pinned in `requirements.txt`
  (yfinance 1.7.0, pandas 3.1.0, numpy 2.5.4, pyarrow 26.0.1, requests 2.34.2, matplotlib 3.11.2,
  scikit-learn 1.9.1, scipy 1.18.1).
- Internet access on the first run. **Data is downloaded at run time** from Yahoo Finance (via `yfinance`)
  and FRED (public CSV endpoint). **No API key is needed.** Downloaded data is cached in `data/` and is
  **not committed** to the repository (see `.gitignore`).

## Install and run

macOS / Linux:

```bash
git clone https://github.com/ToriWU0927/project2-trading-strategy.git
cd project2-trading-strategy
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_all.py
```

Windows (PowerShell): same steps, with the venv's interpreter at `.venv\Scripts\python.exe`:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\run_all.py
```

Always run from the repository root. Options:

```bash
.venv/bin/python src/run_all.py --refresh      # force a fresh download of the raw data
.venv/bin/python src/run_all.py --parts B C    # re-run only Parts B and C (needs Part A's data/ from an earlier run)
```

Each part can also be run on its own: `python src/run_partA.py`, `python src/run_partB.py`, and the Part C
scripts in the order listed in `src/run_all.py`.

**Expected runtime** (Apple-silicon laptop, measured on a fresh clone): `pip install` about 20–60 s;
first full run about 60 s including the data download (Part A ≈ 35 s, Part B ≈ 12 s, Part C ≈ 15 s);
later runs about 40 s using the data cache.

**Randomness:** every random step uses seed **5205** (Part A block bootstrap and Stambaugh simulation in
`config.json → analysis.seed`; Part B random Fourier features in `partB.model.seed`; Part C paired block
bootstrap in `partC.metrics.bootstrap.seed`). Ridge/LASSO/Elastic Net fits are deterministic.

**Self-checks:** the run stops with an error if a Part A adjustment check, leakage test or missing-value
check fails, if any of the 33 Part C accounting tests fails (`results/partC/accounting_tests.csv`), or if a
stored validation selection of a Part C extension is not reproduced exactly (tolerance 1e-10) by the current
code and data.

**Data vintage and exact reproducibility:** the sample is fixed to 1993-01-01 … 2026-11-01 in
`config.json`; `data/raw/_manifest.json` records what was downloaded. Cached files are reused unless
`--refresh` is given. Yahoo recomputes its dividend-adjusted prices on every request: a download on
2026-11-12 differed from the members' original download by at most 1.4·10⁻⁶ (relative) in adjusted SPY
prices (FRED and VIX identical). With such a download every reported number agrees to 4 decimals (main
table, champion, OOS R², selected extension variants), and the last digits of some statistics differ.
The committed `results/` come from the members' original download.

Because the Part C extensions check their stored validation selection to 1e-10, the selection files
(`results/partC/ext_{vol,tilt}_selection.json` and `_check.json`) are **not committed**: a fresh run
selects each extension's variant on the validation period (same rule, validation data only) before the test
run. The members' original records are in `results/partC/recorded_selection/`; the fresh run selects the
same variants (`A_current`, `T1_under`).

## Configuration

All settings are in one file, `config.json`:

| Section | Used by | Content |
|---|---|---|
| `data`, `timing`, `cleaning`, `features`, `analysis` | Part A | tickers, sample dates, FRED lag, feature windows, EDA settings |
| `splits` | Parts A, B | dev 1994-02-01 … 2014-12-31, CV validation years 2005–2014, `embargo_days = 2`, test 2015-01-01 … 2026-10-30 |
| `partB` | Part B | label, feature sets, hyper-parameter grids, clip ±5σ, seed, robustness / RFF / winsor settings |
| `partC` | Part C | signal columns, γ = 3 (sensitivity 1, 5), cost scenarios 0/1/2/5/10 bp (main 1 bp), benchmarks, bootstrap, event windows |

## Outputs

### Part A — `results/partA/` (from `run_partA.py`)

| File | Script | Content |
|---|---|---|
| `fig_inputs_timeseries.png` | `eda.py` | raw inputs over time (report Fig. 1) |
| `fig_adjustment_checks.png` | `checks.py` | dividend / split adjustment checks (report Fig. 2) |
| `fig_univariate_ic_dev.png` | `eda.py` | univariate information coefficients, dev period (report Fig. 3) |
| `fig_corr_heatmap_dev.png` | `eda.py` | feature correlation heatmap, dev period (report Fig. 4) |
| `adjustment_checks.json`, `leakage_checks.json` | `checks.py`, `leakage.py` | check results |
| `ic_dev.csv`, `ic_subperiods_dev.csv`, `feature_stats_dev.csv`, `label_stats_dev.csv`, `group_summary_dev.csv`, `top_correlations_dev.csv`, `subperiod_checks_dev.csv`, `stambaugh_simulation.csv`, `open_quality_by_year.csv`, `coverage.csv`, `eda_summary.md` | `eda.py` | descriptive statistics |

Part A also writes (not committed) `data/raw/*.parquet`, `data/interim/clean.parquet`,
`data/processed/panel.parquet` (8,245 rows × 33 columns), `feature_groups.json`, `feature_table.csv`, and
regenerates `docs/data_dictionary.md` and `docs/alignment_example.md`.

### Part B — `results/partB/` (from `run_partB.py`)

| File | Content |
|---|---|
| `predictions.parquet` | test-period forecasts of every model (input to Part C) |
| `val_predictions_champion.parquet` | out-of-sample validation forecasts of the champion (input to Part C) |
| `cv_results.csv`, `selection.csv`, `champion.json` | 72 configurations × 10 folds; best α per family × feature set; champion = **ridge_M1, α = 10,000** |
| `baselines.json`, `test_oos_r2.json` | AR(1) baseline; test OOS R² of each model vs the expanding mean |
| `coef_path_champion.csv`, `selection_freq_{lasso,elasticnet}_M3.csv` | monthly refit coefficients; LASSO/EN selection frequency |
| `robustness.json`, `cv_results_2001plus.csv`, `selection_2001plus.csv` | same CV on 2001+ |
| `rff_cv.csv`; `winsor.json`, `cv_results_winsor.csv`, `selection_winsor.csv` | random-Fourier-feature exploration; label winsorisation sensitivity |
| `summary.md` | one-page summary |

### Part C — `results/partC/{val,test}/` (from the Part C scripts; `val` = 2005–2014 validation forecasts, `test` = 2015–2026)

| File | Script | Content |
|---|---|---|
| `performance_main.csv`, `summary.md` | `run_partC.py` | main performance table (1 bp, γ = 3) for the strategy and all benchmarks |
| `performance_by_cost.csv`, `performance_by_gamma.csv` | `run_partC.py` | cost and γ sensitivity |
| `bootstrap_differences.csv` | `run_partC.py` | paired circular block bootstrap (63 days, 2,000 reps) of Sharpe / CE / excess-return differences |
| `yearly_returns.csv`, `event_windows.csv` (test), `ablation_models.csv` (test) | `run_partC.py` | yearly returns, pre-specified market events, every Part B model through the same rules |
| `ledger_main.csv`, `ledger_all_strategies.csv`, `run_meta.json` | `run_partC.py` | daily ledgers and run metadata |
| `fig_nav_drawdown.png`, `fig_positions_signal.png`, `fig_cost_sensitivity.png` | `run_partC.py` | NAV & drawdown; weights and forecast; Sharpe/CE vs cost |
| `fig_oos_r2.png`, `oos_r2_by_model.csv`, `oos_r2_by_year.csv` | `figures.py` | out-of-sample R² |
| `fig_yearly.png`, `yearly_return_sharpe.csv` | `figures.py` | return and Sharpe by calendar year |
| `fig_breakeven.png`, `breakeven_table.csv`, `breakeven_curves.csv`, `breakeven_bootstrap_sharpe.csv` | `breakeven.py` | break-even transaction costs |
| `diag_exposure_cuts.csv`, `diag_signal_decay.csv`, `diag_oos_r2_concentration.csv` (test) | `diagnostics.py` | explanatory diagnostics |
| `capm_alpha.csv` | `capm.py` | CAPM α/β with Newey–West standard errors |
| `fig_ext_vol.png`, `ext_vol_*.csv/json`, `diag_vol_regime.csv` | `extension_vol.py` | exploratory extension 1: fixed / capped risk estimate |
| `fig_ext_tilt.png`, `ext_tilt_*.csv/json` | `extension_tilt.py` | exploratory extension 2: buy-and-hold base + model tilt |

`results/partC/` also holds `accounting_tests.csv` (33 hand-checkable tests), the members' original
validation selections of the two extensions (`recorded_selection/`), `freeze_log.json`, and `archive_main5bp/` (outputs of the original 5 bp main specification, kept for
transparency; see `docs/important_changes.md`).

## Repository layout

```
README.md  requirements.txt  .gitignore  config.json
src/        all code (Parts A, B, C) + run_all.py
results/    partA/, partB/, partC/ — committed outputs of the last full run
docs/       data dictionary, alignment example, Part A process notes, Part B/C notes, change log
notebooks/  (empty — all analysis is in scripts)
data/       created at run time, not committed
```

The notes in `docs/partB_README.md`, `docs/partC_README.md`, `docs/handoff_*.md` were written by each
member before the repositories were merged; paths there (`partA/`, `partB/`, `partC/`,
`config_partB.json`, `config_partC.json`) now correspond to `config.json → partB / partC` and
`results/partA|partB|partC/` here.
