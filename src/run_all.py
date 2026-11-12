"""Run the whole project end to end: Part A (data, features, checks) -> Part B (models) -> Part C (trading).

Usage (from the repository root):
    python src/run_all.py              # full pipeline (downloads data on the first run, then uses the cache)
    python src/run_all.py --refresh    # force a fresh download of the raw data
    python src/run_all.py --parts B C  # re-run only some parts (A's panel must already exist)

Each step runs as its own process from the repository root and the script stops at the first failure.
Part C's order matters: run_partC writes the ledgers that figures/diagnostics/capm read, and each
extension's --period val records (or re-verifies) its validation selection before --period test.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = {
    "A": [["src/run_partA.py"]],
    "B": [["src/run_partB.py"]],
    "C": [
        ["src/test_accounting.py"],
        ["src/run_partC.py", "--period", "val"],
        ["src/run_partC.py", "--period", "test"],
        ["src/figures.py", "--period", "val"],
        ["src/figures.py", "--period", "test"],
        ["src/breakeven.py", "--period", "val"],
        ["src/breakeven.py", "--period", "test"],
        ["src/diagnostics.py"],
        ["src/capm.py", "--period", "val"],
        ["src/capm.py", "--period", "test"],
        ["src/extension_vol.py", "--period", "val"],
        ["src/extension_vol.py", "--period", "test"],
        ["src/extension_tilt.py", "--period", "val"],
        ["src/extension_tilt.py", "--period", "test"],
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download raw data in Part A")
    ap.add_argument("--parts", nargs="+", choices=list(STEPS), default=list(STEPS),
                    help="which parts to run (default: A B C)")
    args = ap.parse_args()

    t0 = time.time()
    for part in args.parts:
        for step in STEPS[part]:
            cmd = [sys.executable] + step + (["--refresh"] if step[0] == "src/run_partA.py" and args.refresh else [])
            print(f"\n===== [{part}] python {' '.join(step)}", flush=True)
            t = time.time()
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode != 0:
                sys.exit(f"Stopped: python {' '.join(step)} failed (exit code {r.returncode}).")
            print(f"----- ok in {time.time() - t:.1f}s", flush=True)
    print(f"\nAll steps done in {time.time() - t0:.0f}s. Outputs in results/partA, results/partB, results/partC.")


if __name__ == "__main__":
    main()
