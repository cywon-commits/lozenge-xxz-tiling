#!/bin/bash
# usage (from data/stage4):  ../../scripts/run_jobs.sh jobs_b1.txt [workers]
# each line of the job file: <name> <k> <Delta> <beta> <h> <ntherm> <nmeas> <seed> <out.jsonl>
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
xargs -P "${2:-2}" -L 1 python3 "$(dirname "$0")/s4_qmc.py" < "$1" > "${1%.txt}.log" 2>&1
