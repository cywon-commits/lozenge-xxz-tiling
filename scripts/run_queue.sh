#!/bin/bash
# usage (from data/stage4): ../../scripts/run_queue.sh <workers> <jobs1.txt> [jobs2.txt ...]   - runs the job files one after another
w=$1; shift
for f in "$@"; do "$(dirname "$0")/run_jobs.sh" "$f" "$w"; done
