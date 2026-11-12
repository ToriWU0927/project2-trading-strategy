# 整合与审阅变更记录（2026-11-12）

把 A/B/C 三份代码合并成一个仓库时做的所有改动，以及代码审阅发现、但**没有改**的问题。
最后一列写的是报告需要核对或修改的地方。建模、数据和回测逻辑都没有改。

## 1. 整合改动（只动路径和配置读取，不影响任何数字）

| 改动 | 文件 |
|---|---|
| 三份配置合并成一个 `config.json`：Part A 原有各段不变，新增 `partB`（原 `config_partB.json`）和 `partC`（原 `config_partC.json`）两段，只改了其中的路径 | `config.json` |
| B 从 `config.json["partB"]` 读取配置 | `src/model.py` |
| C 的 8 个脚本从 `config.json["partC"]` 读取配置 | `src/run_partC.py`、`extension_vol.py`、`extension_tilt.py`、`breakeven.py`、`figures.py`、`capm.py`、`diagnostics.py`、`test_accounting.py` |
| B 的预测文件从 `partB/data/predictions.parquet` 移到 `results/partB/predictions.parquet`；C 的输出从 `partC/results/` 移到 `results/partC/` | 配置中的路径 |
| `diagnostics.py` 里写死的 `results/test/` 改为 `cfg["output_dir"]/test` | `src/diagnostics.py` |
| 新增统一入口 `src/run_all.py`（A → B → C，C 的步骤顺序与原 `partC/run_all.py` 相同） | `src/run_all.py` |
| C 两个扩展的选择文件（`ext_{vol,tilt}_selection*.json`）不再入库（见第 4 节）；组员的原始记录另存于 `results/partC/recorded_selection/` | `.gitignore` |

验证：用原始缓存数据跑整合后的代码，Part A 和 Part C 的全部 CSV/JSON 与组员原版完全一致；Part B 只有浮点噪声（最大 8e-6，出现在一个未被选中的 LASSO 网格点）。

## 2. 代码修复（只修复不改变结果的问题）

| 问题 | 修复 | 对结果的影响 | 报告 |
|---|---|---|---|
| `run_partC.py:143` 和 `extension_vol.py:78` 往数值型 Series 里写字符串（`rule` / `kind`），pandas 3.1 会给出 Pandas4Warning，到 pandas 4 会直接报错 | 先 `.astype(object)` 再写入，生成表格后用 `.infer_objects()` 还原成数值列 | 无：`ablation_models.csv`、`ext_vol_performance.csv`（val/test）与修复前逐字节相同，其他输出也不变 | 无需改动 |

## 3. 文档数字更正（直接改了，报告里如有相同数字也要一起改）

依据是 `results/partC/` 中的 CSV，按四舍五入保留到文档原有的位数。原文档的偏差像是二次舍入造成的（例如 2.647 → 2.65 → 2.7）。

| 文档位置 | 原文 | 更正为 | 依据（精确值） |
|---|---|---|---|
| `docs/partC_README.md` 自助法表，"MV champion − MV 历史均值"的年化超额差 | −0.9% | **−0.8%** | `test/bootstrap_differences.csv`：−0.849% |
| `docs/partC_README.md` CAPM 表，主策略验证期 α；`docs/handoff_for_D.md` §3 CAPM | +2.7% | **+2.6%** | `val/capm_alpha.csv`：2.647% |
| `docs/partC_README.md` CAPM 表，MV champion 验证期 α 及区间 | +3.4% [−1.2%, 7.9%] | **+3.3% [−1.2%, 7.8%]** | 3.346% [−1.154%, 7.847%] |
| `docs/partC_README.md` CAPM 表，目标波动率测试期 α 区间下限 | −1.7% | **−1.6%** | `test/capm_alpha.csv`：−1.646% |
| `docs/partC_README.md` 扩展 2 自助法表，测试期 T1 对零信号对照的 CE 差；`docs/handoff_for_D.md` §5 | +1.1% | **+1.2%** | `test/ext_tilt_bootstrap.csv`：1.152% |
| `docs/partB_README.md` 输出表 | 全部 72 个配置 | **84 个配置（28 个超参点 × 3 个特征集）** | `results/partB/cv_results.csv` 共 84 行；Ridge 10 + LASSO 6 + EN 12 = 28 |
| `docs/partB_README.md` 运行时间 | 约 1–2 分钟 | **约 15 秒** | 2026-11-12 实测 |

其余数字都逐条核对过，与结果一致：C 的主表、自助法、盈亏平衡成本、γ 敏感性、模型消融、诊断、两个扩展的表格，以及 `important_changes.md` 和 `handoff_checklist.md` 中引用的数字。

## 4. 复现性：Yahoo 复权价每次下载都会略有不同

- 2026-11-12 重新下载时，SPY 复权价与组员当时下载的数据最多相差 1.4e-6（相对差）；FRED 和 VIX 完全一致。
- 用新下载的数据跑，所有报告数字到 4 位小数都一致（主表、冠军模型、OOS R²、扩展选出的变体），只有部分统计量的末位不同。
- C 的扩展会以 1e-10 的容差核对已记录的验证期选择，因此用新数据跑会中止。处理方式（已与 A 确认）：选择文件不入库，新运行只用验证期数据、按同一规则重新选择，选出的变体与原来相同（`A_current`、`T1_under`）。
- **报告**：Reproducibility 一节应说明"数据在运行时下载；Yahoo 复权价存在约 1e-6 的修订，数字末位可能不同"。

## 5. 审阅发现但没有改的问题（改了会改变方法或结果，交给报告说明）

| # | 事实（代码位置） | 报告需要写清楚的地方 |
|---|---|---|
| 1 | B 的截尾敏感性检验（`model.py` `run_cv(winsor=...)`）用训练折的 1%/99% 分位数截尾，**验证集标签也一起截尾**。所以 winsor 的 R²（0.2438%）是在截尾后的 y 上算的，不能和主结果的 0.2407% 直接比 | 如果报告写"截尾后结果稳健"，应注明比较的是两个口径下各自选出的冠军相同（都是 ridge_M1），而不是 R² 可以直接比较 |
| 2 | 验证期 R² 是 10 个年度折 R² 的等权平均（0.2407%），合并全部验证行计算则是 0.2694%；测试期 R² 是合并计算（0.2699%） | 报告引用 R² 时要注明是哪种口径 |
| 3 | LASSO 选中的 α=0.01 把系数全部压成 0，预测等于训练集均值，测试期 R² 正好为 0。Elastic Net（α=0.001，l1_ratio=0.5）在 M1/M2/M3 上结果完全相同，月度重估时只选中 `rev_ret_5`（100%）和 `rev_ret_1`（35%） | 结论应写"L1 惩罚退化为历史均值或只保留短期反转"，不要写"LASSO 选出了若干特征" |
| 4 | 并列时选模用 `idxmax` 取第一个：LASSO 有多个 α 的 R² 都为 0，选中 0.01 只是因为它排在第一个 | 可以一句带过 |
| 5 | 成本感知规则的不交易区间以上一期**目标**权重 w_{t−1} 为基准，而记账时按**漂移后**的权重算换手和成本（`backtest.run`）；成本直接从当期收益中扣除，不单独记录扣费后的持仓 | 方法部分要写明这两种权重各用在哪里（作业的已知待办项："w_{t−1} vs drifted w"） |
| 6 | 扩展 2 的 T2/T3 允许权重最高到 1.5，借款按无风险利率计，没有借贷利差 | 注明这个假设偏乐观 |
| 7 | 主成本在看过测试结果之后从 5bp 改成了 1bp；两个扩展的想法来自测试期的诊断（`docs/important_changes.md`） | 必须披露 |
| 8 | C 的文档列为"必需"的 `docs/protocol_partC.md`、`docs/extension_vol_protocol.md`、`docs/extension_tilt_protocol.md` 不在组员交来的文件里，本机也找不到 | 需要向 C 要这三份文件；拿不到的话，报告和 README 都不要引用它们 |
| 9 | Walk-forward 每月重估时去掉最后 2 行（`embargo_days=2`）。按标签的定义（用到 Open_{t+2}）严格来说去掉 1 行就够了，2 行偏保守。CV 的 `dev` 集不含 `period=="embargo"` 的 2 行（`run_partB.py` 只取 `period == "dev"`），这 2 行只在测试期 walk-forward 时作为训练数据使用 | 这是作业的已知待办项："CV 是否排除了 2 个 embargo 行"——答案是排除了 |
