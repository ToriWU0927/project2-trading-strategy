# Part C test summary (auto-generated)

2015-01-02 → 2026-10-28, 2973 holding periods; dropped (no label): ['2026-10-29', '2026-10-30']

Main: cost 5bp, gamma=3, w_-1=0.0. sigma_target=0.1441

| strategy | CAGR | ann_excess | ann_vol | sharpe | max_drawdown | max_dd_duration_days | ann_turnover | avg_weight | ann_cost_drag | CE_ann | exposure_part | timing_part | cost_part |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main: cost-aware champion | 9.03% | 7.44% | 13.78% | 0.54 | -22.85% | 356.00 | 6.15 | 0.75 | 0.31% | 4.59% | 9.09% | -1.34% | -0.31% |
| MV champion | 7.70% | 6.16% | 13.41% | 0.46 | -22.62% | 372.00 | 53.36 | 0.65 | 2.67% | 3.46% | 7.79% | 1.04% | -2.67% |
| sign champion | 12.15% | 10.53% | 15.61% | 0.67 | -32.32% | 379.00 | 50.09 | 0.76 | 2.50% | 6.88% | 9.21% | 3.83% | -2.50% |
| cost-aware hist-mean | 2.18% | 0.00% | 0.00% | nan | -0.00% | 12.00 | 0.00 | 0.00 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% |
| MV hist-mean | 10.97% | 9.02% | 12.39% | 0.73 | -19.14% | 390.00 | 3.08 | 0.88 | 0.15% | 6.72% | 10.65% | -1.47% | -0.15% |
| buy & hold SPY | 13.65% | 12.06% | 16.84% | 0.72 | -32.05% | 488.00 | 0.08 | 1.00 | 0.00% | 7.81% | 12.07% | 0.00% | -0.00% |
| cash | 2.18% | 0.00% | 0.00% | nan | -0.00% | 12.00 | 0.00 | 0.00 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% |
| MA200 trend | 9.39% | 7.53% | 11.93% | 0.63 | -20.04% | 512.00 | 5.34 | 0.83 | 0.27% | 5.40% | 10.02% | -2.22% | -0.27% |
| target-vol SPY | 11.44% | 9.47% | 12.62% | 0.75 | -19.30% | 384.00 | 2.58 | 0.89 | 0.13% | 7.08% | 10.75% | -1.14% | -0.13% |

Ex-post vol-matched buy & hold (k=0.818): ann_excess=9.87%, sharpe=0.72, maxDD=-26.88%

## Paired block bootstrap (63d, 2000 reps), A − B

| A | B | stat | point | 95% lo | 95% hi | share≤0 |
|---|---|---|---|---|---|---|
| main: cost-aware champion | buy & hold SPY | ann_excess_diff | -0.0462 | -0.0872 | -0.0077 | 0.992 |
| main: cost-aware champion | buy & hold SPY | sharpe_diff | -0.1760 | -0.3994 | 0.0450 | 0.947 |
| main: cost-aware champion | buy & hold SPY | ce_diff | -0.0321 | -0.0708 | 0.0044 | 0.957 |
| main: cost-aware champion | MV hist-mean | ann_excess_diff | -0.0158 | -0.0482 | 0.0131 | 0.854 |
| main: cost-aware champion | MV hist-mean | sharpe_diff | -0.1884 | -0.4461 | 0.0564 | 0.934 |
| main: cost-aware champion | MV hist-mean | ce_diff | -0.0213 | -0.0552 | 0.0084 | 0.913 |
| main: cost-aware champion | target-vol SPY | ann_excess_diff | -0.0203 | -0.0512 | 0.0073 | 0.920 |
| main: cost-aware champion | target-vol SPY | sharpe_diff | -0.2103 | -0.4408 | 0.0177 | 0.965 |
| main: cost-aware champion | target-vol SPY | ce_diff | -0.0249 | -0.0563 | 0.0033 | 0.959 |
| main: cost-aware champion | MV champion | ann_excess_diff | 0.0128 | -0.0163 | 0.0413 | 0.196 |
| main: cost-aware champion | MV champion | sharpe_diff | 0.0806 | -0.1397 | 0.2943 | 0.226 |
| main: cost-aware champion | MV champion | ce_diff | 0.0113 | -0.0177 | 0.0398 | 0.223 |
| MV champion | MV hist-mean | ann_excess_diff | -0.0286 | -0.0592 | 0.0033 | 0.962 |
| MV champion | MV hist-mean | sharpe_diff | -0.2690 | -0.5130 | -0.0072 | 0.978 |
| MV champion | MV hist-mean | ce_diff | -0.0326 | -0.0638 | -0.0008 | 0.977 |
| MV champion | buy & hold SPY | ann_excess_diff | -0.0590 | -0.1007 | -0.0190 | 0.999 |
| MV champion | buy & hold SPY | sharpe_diff | -0.2565 | -0.5038 | -0.0251 | 0.984 |
| MV champion | buy & hold SPY | ce_diff | -0.0434 | -0.0836 | -0.0044 | 0.986 |
