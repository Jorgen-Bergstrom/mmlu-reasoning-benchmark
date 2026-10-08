#!/usr/bin/env bash
# Run the paired reasoning vs non-reasoning MMLU benchmark.
#
# Defaults to a smoke test sized for a quick sanity check. Override with env
# vars, e.g.:
#   LIMIT=1000 ./run_reasoning.sh
#   EFFORT=medium CTX=65536 LIMIT=200 ./run_reasoning.sh
#   ARMS=on LIMIT=50 ./run_reasoning.sh
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-qwen3.8-27B-IQ3_S}"
LIMIT="${LIMIT:-20}"
CTX="${CTX:-131072}"
EFFORT="${EFFORT:-xhigh}"
ARMS="${ARMS:-both}"

args=(--model "$MODEL" --ctx "$CTX" --reasoning-effort "$EFFORT" --arms "$ARMS")
[[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")

exec python3 -u benchmark_mmlu_reasoning.py "${args[@]}" "$@"
