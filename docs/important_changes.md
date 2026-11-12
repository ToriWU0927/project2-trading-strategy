# Part C: Material Changes and Evaluation Disclosure

**Submission version — 9 November 2026**

This note records changes that affect the trading design, reported results, or interpretation of the evidence. Stages describe the sequence of decisions rather than exact execution timestamps. Routine formatting changes and deterministic reruns are omitted.

## Changes affecting the analysis

| Stage | Decision or correction | Effect on results and interpretation |
|---|---|---|
| Before the first test-period strategy evaluation | Specify the cost-aware, single-period mean–variance rule with risk aversion γ = 3, long-only weights between 0 and 1, and an initial one-way transaction cost of 5 basis points per unit of turnover. Use Part B's validation-selected Ridge–M1 forecast. | Establishes the initial trading specification. Part B's test-period prediction R² had already been disclosed, so the design was not registered before all test information was observed. |
| After the initial test-period strategy results were inspected | Revise the main presentation cost from 5bp to 1bp, informed by public SPY liquidity information; retain 5bp and other cost scenarios as sensitivity analyses. | This is a post-test revision of an assumed execution cost, not a measured historical opening-auction cost. Because cost enters the position rule, the revision changes both trading costs and positions. Broad performance rankings remain similar, but statistical significance changes. Original 5bp results are retained in `results/archive_main5bp/`. |
| After inspecting test-period diagnostics | Add two extensions: fixed/capped risk estimates and a buy-and-hold-based allocation tilt. Select each extension's variant using validation-period certainty-equivalent return. | These hypotheses were motivated by test-period observations. Validation-based variant selection does not restore an independent test set. Both extensions are exploratory and do not replace the original main rule. |
| First methodological review | Correct risk calibration for validation: estimate fixed volatility parameters using 1994–2004 data, before validation begins. Test-period calibration continues to use development-period data available before 2015. | Changes some validation-period benchmark and extension results. Re-evaluation retains `A_current` and `T1_under` as the selected variants. The test-period main strategy's positions and returns are unchanged by this correction. |
| First methodological review | Add a same-rule zero-incremental-signal control for each allocation-tilt variant. Separate net-return volatility from excess-return volatility, clarify holding-period dates, and qualify statistical interpretations. | With transaction costs and a cash initial state, a zero incremental signal does not necessarily imply full investment in SPY. The new controls correct that comparison. These changes add controls and clarify metrics without changing the original main strategy's positions or returns. |
| Second implementation review | Add forecast-date and finite-value checks, automatic output-directory creation, and consistency checks for stored extension selections. Export both pooled and mean-fold validation R², include the sign strategy in detailed ledgers, and use neutral labels for position reductions. | Improves input integrity, reproducibility, and traceability. Existing main-strategy results remain unchanged. The submitted accounting and validation checks report 33 passes in `results/accounting_tests.csv`. |

## Scope of the reported evidence

The main presented cost is **1bp per unit of one-way turnover** (1bp = 0.01%); the original specification used 5bp. Cost sensitivity is reported in `results/test/performance_by_cost.csv`. At 1bp, the main strategy's test-period CAGR is approximately 9.91% and its annualized excess-return Sharpe ratio is 0.61, compared with 13.65% and 0.72 for SPY buy-and-hold.

The reported bootstrap intervals describe sampling uncertainty conditional on the stated analysis. They do not account for all research choices made after observing test-period information. A statistically insignificant difference is not evidence of equivalence, and the exploratory extensions are not independent confirmation of out-of-sample performance.

The final numerical outputs are in `results/test/` and `results/val/`; the corresponding configuration is `config_partC.json`. Historical outputs are distinguished from the current results to avoid mixing specifications. This note summarizes material changes and does not claim that the revised 1bp specification or the exploratory extensions were specified before test results were observed.
