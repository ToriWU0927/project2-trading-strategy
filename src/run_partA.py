"""Part A end to end: download -> clean -> adjustment checks -> panel -> leakage tests -> stats.

    python src/run_partA.py [--refresh]

D can call main() from the team's run_all.py.
"""

from __future__ import annotations

import argparse
import time

import checks
import data
import eda
import features
import leakage


def main(refresh: bool = False) -> None:
    t0 = time.time()
    cfg = data.load_config()
    print("== A3 download");  data.download_all(cfg, refresh=refresh)
    print("== A4 clean");     data.save_clean(cfg)
    print("== A5 adjustment checks")
    adj = checks.main()
    ok = (adj["dividends_SPY"]["yahoo_1-D/P"]["max_abs_return_diff_all_days"] < 1e-5
          and adj["split_AAPL"]["yahoo_raw_close_is_split_adjusted"]
          and adj["split_AAPL"]["abs_diff_implied_vs_yahoo"] < 1e-3
          and adj["split_AAPL"]["max_abs_return_diff_close_vs_adjclose_no_dividend_days"] < 1e-5
          and max(adj["adjusted_ohlc_SPY"].values()) < 1e-9)
    if not ok:
        raise SystemExit("adjustment checks FAILED - see results/partA/adjustment_checks.json")
    print("== A6-A7, A9-A10 panel + data dictionary"); features.main()
    print("== A8 leakage tests")
    res = leakage.main()
    if not res["all_passed"]:
        raise SystemExit("leakage tests FAILED - see results/partA/leakage_checks.json")
    print("== A11 descriptive statistics"); eda.main()
    print(f"Part A done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the whole Part A pipeline")
    ap.add_argument("--refresh", action="store_true", help="re-download raw data")
    main(ap.parse_args().refresh)
