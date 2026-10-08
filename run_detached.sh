#!/usr/bin/env bash
# Launch the benchmark in its own session so it survives the launching shell
# (and the agent session). Output goes to real_run.log; the running PID is
# written by the Python script to results/<model>/run.pid.
#
# Usage:
#   LIMIT=1000 ./run_detached.sh
#   LIMIT=1000 ./run_detached.sh --ctx 65536
# Stop with:
#   pkill -f benchmark_mmlu_reasoning.py; pkill -f llama-server
set -euo pipefail
cd "$(dirname "$0")"

LOG="${LOG:-real_run.log}"
: > "$LOG"
export PYTHONUNBUFFERED=1

# setsid puts the job in a new session, detached from our process group, so it
# is not killed when this script (or the agent turn) exits.
setsid bash -c './run_reasoning.sh "$@"' _ "$@" >>"$LOG" 2>&1 </dev/null &
disown 2>/dev/null || true
sleep 3

echo "Launched detached benchmark."
echo "  log:  $(pwd)/$LOG"
echo "  pid:  $(cat results/*/run.pid 2>/dev/null | tail -1 || echo 'starting...')"
echo "  stop: pkill -f \"[b]enchmark_mmlu_reasoning.py\"; pkill -f \"[l]lama-server\""
