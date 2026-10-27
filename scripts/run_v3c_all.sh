#!/bin/bash
cd "$(dirname "$0")/.."
run() { for i in 1 2 3; do PYTHONIOENCODING=utf-8 python scripts/run_experiment.py "configs/$1.json" --workers 8 && return 0; echo "retry $1 ($i)"; done; echo "FAILED $1"; return 1; }
run v3c_store; run v3c_anti
echo ALL_DONE
