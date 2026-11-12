# Part B：模型与验证（member B）

把 A 的 `panel.parquet` 喂给带 shrinkage 的线性模型，预测 SPY 次日超额收益，并用
expanding-window 交叉验证选模，最后在测试期做每月 walk-forward，产出交给 C 的预测文件。

## 运行

```bash
cd partB
python src/run_partB.py        # 约 15 秒（2026-11-12 实测），无网络依赖
```

输入（来自 Part A，路径在 `config_partB.json`）：
- `../partA/data/processed/panel.parquet`
- `../partA/data/processed/feature_groups.json`
- `../partA/config.json`（只读 `splits` 段：验证年份、embargo、测试期）

## 输出

| 文件 | 内容 |
|---|---|
| `data/predictions.parquet` | **交给 C**。index=date（测试期），列：`y_exret`、`mean`、`ar1`、`ridge_M1`…`elasticnet_M3`、`champion` |
| `results/partB/cv_results.csv` | 全部 84 个配置（28 个超参点 × 3 个特征集）的验证期平均 OOS R² |
| `results/partB/selection.csv` | 每个 (family × feature set) 选出的最优超参 + 冠军 |
| `results/partB/champion.json` | 冠军的 family / feature set / 超参 / OOS R² |
| `results/partB/test_oos_r2.json` | 各模型测试期 OOS R²（vs expanding mean） |
| `results/partB/coef_path_champion.csv` | 冠军每月重估的系数（标准化后），用于解释 |
| `results/partB/selection_freq_{lasso,elasticnet}_M3.csv` | M3 上各特征在每月重估中的入选频率（回答"LASSO 是否把 G3 压到 0"） |
| `results/partB/val_predictions_champion.parquet` | 冠军在验证期的 OOS 预测，供 C 核对流程 |
| `results/partB/robustness.json` / `cv_results_2001plus.csv` | 2001+ 稳健性检验 |
| `results/partB/rff_cv.csv` | RFF + Ridge 复杂度探索（Kelly 风格，负结果：日频下复杂度无益） |
| `results/partB/cv_results_winsor.csv` / `selection_winsor.csv` / `winsor.json` | 标签 1%/99% 截尾的敏感性检验（结果稳健） |
| `results/partB/summary.md` | 自动生成的一页摘要，可直接搬进报告 |

## 已声明的协议（写进报告 §4.4–§4.7）

- **经济逻辑**：日频收益的可预测部分很弱、分散在大量高度共线的特征上 → 用 shrinkage。
- **缩放**：每个 fold 内 fit `StandardScaler` + ±5σ 截断（`ClipScaler`），不泄露。
- **基线**：历史均值（算 OOS R² 的分母）、AR(1) 用 `ret_cc` 滞后（⚠️ 不能用 `y_exret.shift(1)`）。
- **主模型**：Ridge / LASSO / ElasticNet，小网格见 `config_partB.json["model"]["grids"]`。
- **验证**：expanding window，2005–2014 每年一折，共 10 折，embargo=2 行。
- **选模**：对每个 (family, feature set) 取验证期平均 OOS R² 最大的超参；全局冠军取其中最大者。
- **测试**：超参冻结，每月第一个交易日用 `period != warmup` 且早于当月的数据重估系数（去最后 embargo 行），预测当月每一天。测试集只跑一次。
- **稳健性**：同一套 CV 在「2001 年以后」的开发期数据上重跑，检验早期粗报价是否主导结果。
