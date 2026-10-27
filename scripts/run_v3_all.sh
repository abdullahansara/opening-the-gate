#!/bin/bash
# runs every Simulation 3b configuration in order; each step retried up to 3 times (re-running resumes completed chunks)
cd "$(dirname "$0")/.."
run() { for i in 1 2 3; do PYTHONIOENCODING=utf-8 python scripts/run_experiment.py "configs/$1.json" --workers 8 && return 0; echo "retry $1 ($i)"; done; echo "FAILED $1"; return 1; }
run v3a && run v3b
for t in $(cat configs/v3_cells.txt); do run "v3g_$t" && run "v3h_$t"; done
echo ALL_DONE
