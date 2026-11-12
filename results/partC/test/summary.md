# Part C test summary (auto-generated)

2015-01-02 → 2026-10-28, 2973 holding periods; dropped (no label): ['2026-10-29', '2026-10-30']

Main: cost 1bp, gamma=3, w_-1=0.0. sigma_target=0.1441

| strategy | CAGR | ann_excess | ann_excess_vol | sharpe | max_drawdown | max_dd_duration_days | ann_turnover | avg_weight | ann_cost_drag | CE_ann | exposure_part | timing_part | cost_part |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main: cost-aware champion | 9.91% | 8.19% | 13.42% | 0.61 | -20.42% | 330.00 | 31.28 | 0.66 | 0.31% | 5.49% | 7.98% | 0.53% | -0.31% |
| MV champion | 10.03% | 8.30% | 13.41% | 0.62 | -21.46% | 350.00 | 53.36 | 0.65 | 0.53% | 5.60% | 7.79% | 1.04% | -0.53% |
| sign champion | 14.42% | 12.54% | 15.61% | 0.80 | -32.21% | 360.00 | 50.09 | 0.76 | 0.50% | 8.88% | 9.21% | 3.83% | -0.50% |
| cost-aware hist-mean | 10.99% | 9.06% | 12.56% | 0.72 | -19.10% | 390.00 | 1.45 | 0.88 | 0.01% | 6.70% | 10.67% | -1.60% | -0.01% |
| MV hist-mean | 11.11% | 9.15% | 12.39% | 0.74 | -18.91% | 390.00 | 3.08 | 0.88 | 0.03% | 6.85% | 10.65% | -1.47% | -0.03% |
| buy & hold SPY | 13.65% | 12.06% | 16.84% | 0.72 | -32.05% | 488.00 | 0.08 | 1.00 | 0.00% | 7.81% | 12.07% | 0.00% | -0.00% |
| cash | 2.18% | 0.00% | 0.00% | nan | -0.00% | 12.00 | 0.00 | 0.00 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% |
| MA200 trend | 9.63% | 7.75% | 11.93% | 0.65 | -19.85% | 506.00 | 5.34 | 0.83 | 0.05% | 5.61% | 10.02% | -2.22% | -0.05% |
| target-vol SPY | 11.55% | 9.58% | 12.62% | 0.76 | -19.15% | 384.00 | 2.58 | 0.89 | 0.03% | 7.19% | 10.75% | -1.14% | -0.03% |

Ex-post vol-matched buy & hold (k=0.797): ann_excess=9.62%, sharpe=0.72, maxDD=-26.26%

## Paired block bootstrap (63d, 2000 reps), A − B

| A | B | stat | point | 95% lo | 95% hi | share≤0 |
|---|---|---|---|---|---|---|
| main: cost-aware champion | buy & hold SPY | ann_excess_diff | -0.0387 | -0.0824 | 0.0009 | 0.970 |
| main: cost-aware champion | buy & hold SPY | sharpe_diff | -0.1060 | -0.3605 | 0.1408 | 0.800 |
| main: cost-aware champion | buy & hold SPY | ce_diff | -0.0232 | -0.0643 | 0.0166 | 0.869 |
| main: cost-aware champion | MV hist-mean | ann_excess_diff | -0.0095 | -0.0383 | 0.0222 | 0.739 |
| main: cost-aware champion | MV hist-mean | sharpe_diff | -0.1282 | -0.3674 | 0.1271 | 0.840 |
| main: cost-aware champion | MV hist-mean | ce_diff | -0.0136 | -0.0440 | 0.0174 | 0.812 |
| main: cost-aware champion | cost-aware hist-mean | ann_excess_diff | -0.0087 | -0.0396 | 0.0226 | 0.706 |
| main: cost-aware champion | cost-aware hist-mean | sharpe_diff | -0.1115 | -0.3435 | 0.1397 | 0.806 |
| main: cost-aware champion | cost-aware hist-mean | ce_diff | -0.0121 | -0.0421 | 0.0185 | 0.785 |
| main: cost-aware champion | target-vol SPY | ann_excess_diff | -0.0138 | -0.0410 | 0.0159 | 0.830 |
| main: cost-aware champion | target-vol SPY | sharpe_diff | -0.1483 | -0.3709 | 0.0913 | 0.888 |
| main: cost-aware champion | target-vol SPY | ce_diff | -0.0170 | -0.0457 | 0.0129 | 0.879 |
| main: cost-aware champion | MV champion | ann_excess_diff | -0.0011 | -0.0120 | 0.0102 | 0.581 |
| main: cost-aware champion | MV champion | sharpe_diff | -0.0085 | -0.0903 | 0.0799 | 0.588 |
| main: cost-aware champion | MV champion | ce_diff | -0.0011 | -0.0121 | 0.0102 | 0.585 |
| MV champion | MV hist-mean | ann_excess_diff | -0.0085 | -0.0385 | 0.0238 | 0.723 |
| MV champion | MV hist-mean | sharpe_diff | -0.1197 | -0.3574 | 0.1404 | 0.824 |
| MV champion | MV hist-mean | ce_diff | -0.0124 | -0.0433 | 0.0192 | 0.801 |
| MV champion | buy & hold SPY | ann_excess_diff | -0.0377 | -0.0793 | 0.0017 | 0.970 |
| MV champion | buy & hold SPY | sharpe_diff | -0.0975 | -0.3438 | 0.1477 | 0.788 |
| MV champion | buy & hold SPY | ce_diff | -0.0221 | -0.0622 | 0.0173 | 0.865 |
