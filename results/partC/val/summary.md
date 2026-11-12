# Part C val summary (auto-generated)

2005-01-03 → 2014-12-29, 2515 holding periods; dropped (no label): []

Main: cost 1bp, gamma=3, w_-1=0.0. sigma_target=0.1543

| strategy | CAGR | ann_excess | ann_excess_vol | sharpe | max_drawdown | max_dd_duration_days | ann_turnover | avg_weight | ann_cost_drag | CE_ann | exposure_part | timing_part | cost_part |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main: cost-aware champion | 8.01% | 7.21% | 13.83% | 0.52 | -24.40% | 639.00 | 33.03 | 0.57 | 0.33% | 4.35% | 4.57% | 2.97% | -0.33% |
| MV champion | 8.76% | 7.91% | 13.81% | 0.57 | -24.43% | 662.00 | 58.23 | 0.55 | 0.58% | 5.05% | 4.43% | 4.06% | -0.58% |
| sign champion | 6.64% | 6.61% | 18.02% | 0.37 | -43.59% | 1234.00 | 56.61 | 0.69 | 0.57% | 1.74% | 5.52% | 1.66% | -0.57% |
| cost-aware hist-mean | 6.92% | 5.84% | 10.93% | 0.53 | -23.85% | 1386.00 | 1.33 | 0.78 | 0.01% | 4.05% | 6.25% | -0.40% | -0.01% |
| MV hist-mean | 7.53% | 6.40% | 10.82% | 0.59 | -23.49% | 841.00 | 3.81 | 0.79 | 0.04% | 4.64% | 6.34% | 0.09% | -0.04% |
| buy & hold SPY | 7.79% | 8.02% | 19.83% | 0.40 | -55.42% | 1222.00 | 0.10 | 1.00 | 0.00% | 2.12% | 8.02% | 0.00% | -0.00% |
| cash | 1.46% | 0.00% | 0.00% | nan | 0.00% | 0.00 | 0.00 | 0.00 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% |
| MA200 trend | 6.39% | 5.35% | 10.96% | 0.49 | -24.29% | 725.00 | 7.31 | 0.77 | 0.07% | 3.55% | 6.20% | -0.78% | -0.07% |
| target-vol SPY | 8.65% | 7.70% | 13.02% | 0.59 | -34.24% | 805.00 | 2.48 | 0.88 | 0.02% | 5.16% | 7.08% | 0.64% | -0.02% |

Ex-post vol-matched buy & hold (k=0.697): ann_excess=5.59%, sharpe=0.40, maxDD=-41.45%

## Paired block bootstrap (63d, 2000 reps), A − B

| A | B | stat | point | 95% lo | 95% hi | share≤0 |
|---|---|---|---|---|---|---|
| main: cost-aware champion | buy & hold SPY | ann_excess_diff | -0.0081 | -0.0796 | 0.0702 | 0.600 |
| main: cost-aware champion | buy & hold SPY | sharpe_diff | 0.1173 | -0.2979 | 0.4327 | 0.283 |
| main: cost-aware champion | buy & hold SPY | ce_diff | 0.0222 | -0.0577 | 0.1134 | 0.330 |
| main: cost-aware champion | MV hist-mean | ann_excess_diff | 0.0082 | -0.0321 | 0.0504 | 0.349 |
| main: cost-aware champion | MV hist-mean | sharpe_diff | -0.0692 | -0.4106 | 0.3187 | 0.618 |
| main: cost-aware champion | MV hist-mean | ce_diff | -0.0029 | -0.0416 | 0.0389 | 0.553 |
| main: cost-aware champion | cost-aware hist-mean | ann_excess_diff | 0.0138 | -0.0275 | 0.0580 | 0.258 |
| main: cost-aware champion | cost-aware hist-mean | sharpe_diff | -0.0124 | -0.3587 | 0.3640 | 0.501 |
| main: cost-aware champion | cost-aware hist-mean | ce_diff | 0.0030 | -0.0380 | 0.0453 | 0.450 |
| main: cost-aware champion | target-vol SPY | ann_excess_diff | -0.0049 | -0.0508 | 0.0421 | 0.588 |
| main: cost-aware champion | target-vol SPY | sharpe_diff | -0.0697 | -0.3889 | 0.2691 | 0.631 |
| main: cost-aware champion | target-vol SPY | ce_diff | -0.0081 | -0.0509 | 0.0356 | 0.644 |
| main: cost-aware champion | MV champion | ann_excess_diff | -0.0069 | -0.0252 | 0.0114 | 0.778 |
| main: cost-aware champion | MV champion | sharpe_diff | -0.0510 | -0.1962 | 0.0831 | 0.778 |
| main: cost-aware champion | MV champion | ce_diff | -0.0070 | -0.0253 | 0.0114 | 0.777 |
| MV champion | MV hist-mean | ann_excess_diff | 0.0151 | -0.0212 | 0.0563 | 0.229 |
| MV champion | MV hist-mean | sharpe_diff | -0.0182 | -0.3571 | 0.3535 | 0.504 |
| MV champion | MV hist-mean | ce_diff | 0.0041 | -0.0328 | 0.0454 | 0.415 |
| MV champion | buy & hold SPY | ann_excess_diff | -0.0012 | -0.0721 | 0.0755 | 0.527 |
| MV champion | buy & hold SPY | sharpe_diff | 0.1683 | -0.2292 | 0.4725 | 0.194 |
| MV champion | buy & hold SPY | ce_diff | 0.0293 | -0.0522 | 0.1213 | 0.268 |
