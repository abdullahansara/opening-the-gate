"""One command: smoke experiment -> analysis -> verification (reproduces the smoke test end to end).

usage: python scripts/run_smoke.py
"""
import _bootstrap
import subprocess
import sys

R = _bootstrap.ROOT


def sh(*a):
    print("$", " ".join(a)); r = subprocess.run([sys.executable, *a], cwd=R)
    if r.returncode:
        sys.exit(r.returncode)


if __name__ == "__main__":
    sh("scripts/run_experiment.py", "configs/smoke.json", "--workers", "4")
    sh("scripts/analyze.py", "results/smoke")
    sh("scripts/verify.py", "results/smoke")
