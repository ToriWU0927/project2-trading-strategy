# Part A 数据交付说明（给 B 和 C）

> 这份文档说明你们会拿到哪些数据、怎么读、每一列在你们的工作里怎么用。
> 数据是怎么来的、怎么清洗和核验的，见 `partA_process.md`。
> 每一列的完整定义见 `data_dictionary.md`。
> 最后更新：2026-11-05（rf 改为等价收益率 × 实际天数；补充多重比较、Pearson/Spearman 差异、Stambaugh 模拟结果；修正若干数字）

---

## 1. 一句话

你们只需要一个文件：**`data/processed/panel.parquet`**。它是一张日频表，每一行是一个 SPY 交易日 t，包含：

- **24 个特征**：都只用截至 t 日收盘的信息，**没有缩放**。
- **标签**：t+1 开盘买入、t+2 开盘卖出的 SPY 超额收益。
- **回测用的价格和无风险利率。**

| 项目 | 内容 |
|---|---|
| 区间 | 1994-01-27 → 2026-10-30，共 8,245 行 |
| 开发期 `dev` | 1994-02-01 → 2014-12-29，5,265 行（用于训练和验证） |
| `embargo` | 2 行（2014-12-30、12-31）：它们的标签用到了 2015 年的开盘价，**不能用于验证打分**；进入测试期后可以作为训练数据 |
| 测试期 `test` | 2015-01-02 → 2026-10-30，2,975 行（**只在最后跑一次**） |
| `warmup` | 3 行（1994-01-27 至 01-31），**不要用** |

---

## 2. 时间约定（三个人必须一致）

```
          第 t 日                     第 t+1 日              第 t+2 日
 收盘 Close_t ─► 决策（只用 ≤ t 的信息） ─► 开盘 Open_{t+1} 买入 ─► 开盘 Open_{t+2} 卖出 / 调仓
                                              └────────── 第 t 行的标签 ──────────┘
```

- 第 t 行的**特征**在 t 日收盘时就已知。
- 第 t 行的**标签**要到 t+2 日开盘才知道。
- 第 t 行的**仓位** `w_t` 在 t 日收盘后决定，赚的是第 t 行的 `y_ret`。

---

## 3. 怎么读

```python
import json
import pandas as pd

panel  = pd.read_parquet("partA/data/processed/panel.parquet")
groups = json.load(open("partA/data/processed/feature_groups.json"))
cfg    = json.load(open("partA/config.json"))          # splits 等设置

FEATS = groups["M3"]                                   # 全部 24 个特征
dev   = panel[panel["period"] == "dev"]
test  = panel[panel["period"] == "test"]
```

也可以直接调用 Part A 的函数：`from features import build_panel, get_feature_groups, period_of`。

---

## 4. 列一览

### 4.1 特征（24 个，未缩放）

| 组 | 列 | 含义 |
|---|---|---|
| **G1a 趋势/动量** | `mom_ret_20`、`mom_ret_60`、`mom_ret_120`、`mom_ret_252` | 过去 w 日收益 |
| | `mom_ma_50`、`mom_ma_200` | 收盘价相对 50/200 日均线的偏离 |
| **G1b 短期反转** | `rev_ret_1`、`rev_ret_5` | 过去 1/5 日收益 |
| | `rev_ma_20` | 相对 20 日均线的偏离 |
| | `rev_rsi_14` | RSI（0–100） |
| | `rev_overnight` | 隔夜跳空 `Open_t/Close_{t−1} − 1` |
| **G1c 波动/成交量** | `vol_rv_20`、`vol_rv_60` | 年化已实现波动率 |
| | `vol_volume` | `log(成交量 / 20 日均量)` |
| **G2 VIX** | `vix_level` | VIX 收盘（点） |
| | `vix_chg_20` | VIX 的 20 日对数变化 |
| | `vix_vrp` | `VIX/100 − vol_rv_20`（方差风险溢价代理） |
| | `vix_rank_252` | VIX 在过去一年中的分位 |
| **G3 利率/信用** | `mac_tbill`、`mac_term`、`mac_credit` | 3 个月 T-bill、10y−3m 期限利差、Baa−10y 信用利差（%） |
| | `mac_tbill_chg_20`、`mac_term_chg_20`、`mac_credit_chg_20` | 各自的 20 日变化（百分点） |

**消融用的特征组合**（`feature_groups.json`）：

| 名称 | 组成 | 特征数 |
|---|---|---|
| **M1** | G1a + G1b + G1c（纯价格信号，选题 1） | 14 |
| **M2** | M1 + G2 | 18 |
| **M3** | M2 + G3（加入宏观择时扩展，选题 2） | 24 |

### 4.2 标签、价格和辅助列

| 列 | 定义 | 主要使用者 |
|---|---|---|
| `y_exret` | `y_ret − rf_daily` ← **主标签** | B |
| `y_ret` | `Open_{t+2}/Open_{t+1} − 1` | C（策略收益、buy-and-hold） |
| `rf_annual` | DTB3（第 t 日可用的值）从贴现率换算成等价收益率：`365d/(360−91d)` | C（参考） |
| `rf_daily` | `rf_annual × 持有的日历天数 / 365`（t+1 开盘到 t+2 开盘，跨周末是 3 天）；最后 2 行和标签一样是 NaN | C（现金收益） |
| `y_up` | `y_exret > 0` 时为 1，否则为 0（可选的分类标签） | B |
| `px_open`、`px_close` | SPY 复权开盘价和收盘价（总收益口径） | C |
| `ret_cc` | close-to-close 收益（第 t 日收盘已知；数值和特征 `rev_ret_1` 相同，作为特征时不要重复放入） | B（AR 基线）、C |
| `period` | `warmup` / `dev` / `embargo` / `test` | B、C |

最后 2 行的 `y_ret`、`y_exret`、`y_up` 是 NaN（还没有未来的开盘价）。

---

## 5. 给 B：模型与验证

### 5.1 标签与基线
- **主标签**：`y_exret`。开发期日均 3.2 bp，日标准差 1.23%，噪声非常大。所以预期 OOS R² 只会在 0 附近，这很正常。
- **历史均值基线**（算 OOS R² 用）：每个训练窗口里 `y_exret` 的均值。
- **AR(1) 基线**：⚠️ **不能用 `y_exret.shift(1)`**。第 t−1 行的标签要用到 `Open_{t+1}`，在第 t 日收盘时还不知道，用了就是泄露。
  - 正确做法一：用 `ret_cc`（第 t 日的 close-to-close 收益）作为 AR 的滞后项。
  - 正确做法二：用 `y_exret.shift(2)`。
  - 开发期里 `ret_cc` 和 `y_ret` 的相关系数是 −0.07。

### 5.2 缩放与异常值
- 特征**没有缩放**。请在每个训练折内拟合 `StandardScaler`，建议再在 ±5σ 处截断，用 sklearn `Pipeline` 包起来，保证只用训练数据拟合。
- `vol_volume` 在 1994–95 年有极端值（当时 SPY 每天只成交几千股），截断就能处理。
- **早年开盘价比较"粗"**：2001 年美股十进制改革之前，SPY 的报价单位是 1/8 或 1/16 美元（约 0.1–0.3%）。1993–94 年有 9–14% 的天数开盘价等于前一天收盘价（1995 年为 5%），1993–95 年有 22–40% 的天数开盘价恰好落在当天的最高价或最低价上；2002 年以后大多数年份都在 2% 以下。标签是 open-to-open，所以早年的标签带有额外噪声，也会放大短期反转（见 §5.4）。是否要做"只用 2001 年以后数据训练"的稳健性检验，由 B 决定。逐年统计见 `results/partA/open_quality_by_year.csv`。
- **个别开盘价可能有误**：2001 年以后仍有 28 天的开盘价和前一天开盘价完全相同。例如 2025-10-15 和 10-16 的开盘价相同，导致 2025-10-14 这一行的 `y_ret` ≈ 0。这可能是 Yahoo 的数据问题，但没有外部数据可以核实。影响很小，这里只是告知。

### 5.3 交叉验证与 embargo（config 里的日期，**需要你确认**）
- **开发期内的 expanding window CV**：验证年份 2005–2014，每年一折，共 10 折，每折约 252 天。
- **embargo**：第 t 行的标签用到 `Open_{t+1}` 和 `Open_{t+2}`。所以训练集必须在验证集第一天 v 之前至少去掉 1 行，config 里设成 2 行（`embargo_days = 2`），多留一行余量。
- **开发期和测试期之间**同样需要隔开：开发期最后 2 行已经标成 `embargo`，用 `period == "dev"` 筛选时会自动排除，所以 2014 年那一折的验证不会碰到 2015 年的价格。

```python
s = cfg["splits"]
dev = panel[panel.period == "dev"].dropna(subset=["y_exret"])   # 已自动排除 embargo 行

def expanding_folds(dev, val_years=s["cv_val_years"], embargo=s["embargo_days"]):
    for yr in val_years:
        val = dev[dev.index.year == yr]
        train = dev[dev.index < val.index[0]].iloc[:-embargo]
        yield yr, train, val
```

- **测试期 walk-forward**：超参数冻结后，每月第一个交易日 d 用 `period != "warmup"` 且早于 d 的数据重新估计系数，再去掉最后 `embargo` 行，然后预测这个月的每一天。

```python
usable = panel[panel.period != "warmup"]    # dev + embargo + test，连续的历史

i = usable.index.get_loc(d)                 # d = 当月第一个交易日
train = usable.iloc[: i - s["embargo_days"]].dropna(subset=["y_exret"])
```

### 5.4 对模型选择有用的数据事实（只基于开发期）

| 事实 | 对模型的意义 |
|---|---|
| **特征之间高度共线**：`mom_ret_120`~`mom_ma_200` 相关系数 0.94，`vol_rv_20`~`vix_level` 0.89，`mom_ret_20`~`mom_ma_50` 0.86 | OLS 系数会不稳定，这是用 Ridge 和 LASSO 的直接理由 |
| **最强的信号是短期反转**：`rev_ret_5` 的 IC 为 −0.066，`rev_ma_20` 为 −0.047，`rev_rsi_14` 为 −0.046。全开发期的 95% 置信区间都不含 0，前后两个半段的符号一致 | 经济逻辑：日频上存在过度反应后的修正 |
| **VIX 组 4 个特征的 IC 都为正，置信区间都不含 0**（IC 在 +0.026 到 +0.042 之间） | 恐慌越高，风险溢价补偿越高 |
| **信号随时间减弱**：`rev_ret_5` 的 IC 在 1994–2000、2001–2007、2008–2014 三段分别为 −0.085、−0.077、−0.046；`vix_rank_252` 分别为 +0.063、+0.042、+0.015。把标签换成 close-to-close 后，反转依然存在，说明它不只是开盘价报价精度造成的假象 | 测试期（2015 年以后）的信号很可能更弱，OOS 结果偏弱时可以用这一点来解释。也可以考虑让训练窗口更多地反映近期数据 |
| **20 日和 50 日"动量"指标的 IC 是负的**，120 日和 252 日的接近 0 | 日频上看不到时间序列动量；动量和反转信号可能部分抵消 |
| **利率/信用组（G3）全部不显著**，前后两段符号还会翻转 | 预期 M3 相对 M2 提升很小，正好可以展示 LASSO 会把这些系数压到 0 |
| **持续性强的特征在短窗口里有 Stambaugh 偏差**：用 iid 收益模拟 200 次（真实 IC = 0），按年计算时 `mom_ret_120` 的 IC 平均被压到 −0.065，只有 12% 的年份为正；即使是约 7 年的分段，也有约 −0.01 的偏差（`results/partA/stambaugh_simulation.csv`） | 支持用 expanding window，不要用很短的滚动窗口；分段比较慢变量时也要考虑这个偏差 |
| **多重比较**：24 个特征在 5% 水平下检验，大约会有 1 个是偶然显著的。95% 置信区间不含 0 的有 11 个；用更严格的 Bonferroni 标准，仍然显著的有 5 个：`rev_ret_5`、`rev_ma_20`、`rev_rsi_14`、`mom_ret_20`、`vix_rank_252` | 结论应该以这 5 个为主，其他的只作参考 |
| **Pearson 和 Spearman 差别较大**：比如 `rev_ret_1` 的 Pearson IC 是 −0.071，Spearman 只有 −0.026，差异主要来自 2008 年的极端日 | Ridge 用平方损失，更接近 Pearson 的口径，容易被极端日主导。建议做一次标签截尾（比如 1%/99%）的敏感性检验 |

完整的 IC 表和置信区间见 `results/partA/ic_dev.csv`，分时期 IC 见 `ic_subperiods_dev.csv`，图见 `fig_univariate_ic_dev.png`，相关性热图见 `fig_corr_heatmap_dev.png`。
⚠️ 这些 IC **只用来解释经济逻辑**。特征组要固定为预先声明的 M1/M2/M3，不能根据 IC 去挑特征。

### 5.5 B 交给 C 的预测文件（建议格式）
`predictions.parquet`：index 是 `date`（和 panel 一致），列为 `y_hat`；如果做消融，可以有多列，比如 `M1_ridge`、`M3_lasso`。只需要包含测试期的行（以及验证期的行，供 C 检查流程）。

---

## 6. 给 C：交易与评估

### 6.1 收益公式（和标签的时点完全对齐）
仓位 `w_t` 在第 t 日收盘后决定，在 t+1 开盘成交：

```
r_strat_t = w_t · y_ret_t + (1 − w_t) · rf_daily_t − cost · |w_t − w_{t−1}|
```

- 换手发生在 t+1 开盘，成本记在第 t 行。
- 测试期第一天需要一个初始仓位 `w_{−1}`，常见的做法是设为 0（从现金开始，第一天就要付建仓成本），也可以设为 1。由 C 决定，并在报告里写明。
- 年化用 252。
- 策略收益和所有基准都用同一个 `y_ret`，所以不会出现时点错位。

### 6.2 基准都可以直接从 panel 构造
| 基准 | 仓位 |
|---|---|
| SPY buy-and-hold | `w ≡ 1` |
| 现金 | `w ≡ 0`（收益为 `rf_daily`） |
| 波动率管理 SPY（Moreira & Muir 2017） | `w_t = min(σ_target / vol_rv_20_t, w_max)`，`vol_rv_20` 在第 t 日收盘已知 |
| 与策略同波动的 SPY（Kelly et al. 2024 的比较方式） | 把 buy-and-hold 的收益按常数缩放到和策略相同的波动，再比较两者的收益或信息比率 |
| 200 日均线趋势规则 | `w_t = 1 if mom_ma_200_t > 0 else 0`（开发期约 74% 的时间持有） |
| 60/40 等 | 不在本数据范围内 |

### 6.3 参考数字（开发期 1994–2014，只用于对照，不是测试结果）
| 指标 | 值 |
|---|---|
| buy-and-hold 年化超额收益 | 8.0% |
| 年化波动 | 19.6% |
| 夏普比率 | 0.41 |
| 超额峰度 | 9.7（厚尾，最大回撤会很深） |
| 上涨天数占比 | 53.2% |
| 标签一阶自相关 | −0.079 |

### 6.4 测试期的分阶段分析
可以按以下几个阶段切分（日期是常用的近似区间，C 可以自行调整）：
- 2015–2016：油价和中国市场波动
- 2018 年第四季度：加息引发的回调
- **2020-02-19 → 2020-03-23**：新冠暴跌
- **2022-01-03 → 2022-10-12**：加息熊市
- 2025-04：关税冲击（2025-04-09 单日 +10.5%）

### 6.5 解释性分析可以用的分组
`feature_groups.json` 里的 G1a–G3 可以直接用于：
- 按组分解贡献：系数 × 特征，再按组加总
- 按组统计 LASSO 的入选频率

---

## 7. 常见问题

| 问题 | 回答 |
|---|---|
| 为什么不切成 train/val/test 三个文件？ | walk-forward 需要连续的历史；划分日期只放在 `config.json` 一处，改起来只改一处 |
| 特征会不会用到未来信息？ | 不会。截断测试、复权前视测试、未来扰动测试、单点扰动测试和阴性对照全部通过（见 `results/partA/leakage_checks.json`）；逐列的时间对齐见 `docs/alignment_example.md` |
| FRED 的数据在 t 日真的能拿到吗？ | 第 t 行只用**日期早于 t** 的 FRED 值。按我们的了解，美联储 H.15 一般在 t 日美东时间下午 4:15 左右发布 t−1 日的数据，正好在美股收盘之后，所以这个值在 t 日收盘后、t+1 开盘前可以拿到，**刚好可得**，并不是额外多滞后了一天。因为成交在 t+1 开盘，所以不构成泄露。发布时间没有用 ALFRED 逐日核实，BAA10Y（来自 Moody's）的发布时间也不确定 |
| 价格有没有处理分红？ | 有。`px_*` 和 `y_ret` 都是总收益口径（拆股 + 分红调整），并且和手工复权核对过 |
| 想加特征怎么办？ | 告诉 A。所有窗口都在 `config.json["features"]` 里，新增特征也要先通过防泄露测试 |
| 数据要更新怎么办？ | 运行 `python src/run_partA.py --refresh`（大约 30 秒） |
