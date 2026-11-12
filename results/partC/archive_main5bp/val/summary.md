# Part C val summary (auto-generated)

2005-01-03 → 2014-12-29, 2515 holding periods; dropped (no label): []

Main: cost 5bp, gamma=3, w_-1=0.0. sigma_target=0.1441

| strategy | CAGR | ann_excess | ann_vol | sharpe | max_drawdown | max_dd_duration_days | ann_turnover | avg_weight | ann_cost_drag | CE_ann | exposure_part | timing_part | cost_part |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main: cost-aware champion | 9.19% | 8.32% | 13.99% | 0.59 | -32.14% | 789.00 | 7.04 | 0.71 | 0.35% | 5.38% | 5.67% | 3.00% | -0.35% |
| MV champion | 6.26% | 5.58% | 13.81% | 0.40 | -25.04% | 683.00 | 58.23 | 0.55 | 2.91% | 2.72% | 4.43% | 4.06% | -2.91% |
| sign champion | 4.25% | 4.34% | 18.03% | 0.24 | -44.89% | 1439.00 | 56.61 | 0.69 | 2.83% | -0.53% | 5.52% | 1.66% | -2.83% |
| cost-aware hist-mean | 1.46% | 0.00% | 0.00% | nan | 0.00% | 0.00 | 0.00 | 0.00 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% |
| MV hist-mean | 7.36% | 6.24% | 10.82% | 0.58 | -23.81% | 843.00 | 3.81 | 0.79 | 0.19% | 4.49% | 6.34% | 0.09% | -0.19% |
| buy & hold SPY | 7.78% | 8.02% | 19.83% | 0.40 | -55.42% | 1222.00 | 0.10 | 1.00 | 0.01% | 2.12% | 8.02% | 0.00% | -0.01% |
| cash | 1.46% | 0.00% | 0.00% | nan | 0.00% | 0.00 | 0.00 | 0.00 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% |
| MA200 trend | 6.08% | 5.06% | 10.97% | 0.46 | -24.66% | 744.00 | 7.31 | 0.77 | 0.37% | 3.25% | 6.20% | -0.78% | -0.37% |
| target-vol SPY | 8.34% | 7.35% | 12.54% | 0.59 | -32.75% | 808.00 | 2.87 | 0.86 | 0.14% | 4.99% | 6.93% | 0.56% | -0.14% |

Ex-post vol-matched buy & hold (k=0.705): ann_excess=5.66%, sharpe=0.40, maxDD=-41.87%

## Paired block bootstrap (63d, 2000 reps), A − B

| A | B | stat | point | 95% lo | 95% hi | share≤0 |
|---|---|---|---|---|---|---|
| main: cost-aware champion | buy & hold SPY | ann_excess_diff | 0.0030 | -0.0584 | 0.0688 | 0.475 |
| main: cost-aware champion | buy & hold SPY | sharpe_diff | 0.1905 | -0.1477 | 0.4727 | 0.142 |
| main: cost-aware champion | buy & hold SPY | ce_diff | 0.0327 | -0.0374 | 0.1174 | 0.213 |
| main: cost-aware champion | MV hist-mean | ann_excess_diff | 0.0208 | -0.0168 | 0.0637 | 0.151 |
| main: cost-aware champion | MV hist-mean | sharpe_diff | 0.0179 | -0.3000 | 0.3860 | 0.423 |
| main: cost-aware champion | MV hist-mean | ce_diff | 0.0090 | -0.0312 | 0.0508 | 0.331 |
| main: cost-aware champion | target-vol SPY | ann_excess_diff | 0.0097 | -0.0268 | 0.0484 | 0.296 |
| main: cost-aware champion | target-vol SPY | sharpe_diff | 0.0089 | -0.2594 | 0.3013 | 0.421 |
| main: cost-aware champion | target-vol SPY | ce_diff | 0.0040 | -0.0314 | 0.0406 | 0.411 |
| main: cost-aware champion | MV champion | ann_excess_diff | 0.0274 | -0.0088 | 0.0649 | 0.070 |
| main: cost-aware champion | MV champion | sharpe_diff | 0.1909 | -0.0684 | 0.4979 | 0.080 |
| main: cost-aware champion | MV champion | ce_diff | 0.0267 | -0.0101 | 0.0642 | 0.076 |
| MV champion | MV hist-mean | ann_excess_diff | -0.0067 | -0.0425 | 0.0348 | 0.643 |
| MV champion | MV hist-mean | sharpe_diff | -0.1730 | -0.5233 | 0.1980 | 0.810 |
| MV champion | MV hist-mean | ce_diff | -0.0177 | -0.0550 | 0.0233 | 0.808 |
| MV champion | buy & hold SPY | ann_excess_diff | -0.0244 | -0.0951 | 0.0522 | 0.747 |
| MV champion | buy & hold SPY | sharpe_diff | -0.0004 | -0.4200 | 0.3231 | 0.512 |
| MV champion | buy & hold SPY | ce_diff | 0.0060 | -0.0753 | 0.0976 | 0.478 |
