#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
REASONING_ROOT="${REASONING_ROOT:-$PROJECT_ROOT/data/reasoning}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/minecraft_reasoning}"
REQUIRE_REASONING="${REQUIRE_REASONING:-0}"

mkdir -p "$OUT_ROOT"

merge_stage() {
  local stage="$1"
  local args=()
  if [[ "$REQUIRE_REASONING" == "1" ]]; then
    args+=(--require-reasoning)
  fi
  python -m training.reasoning_annotation merge \
    --manifest "$DATA_ROOT/train_stage${stage}.jsonl" \
    --reasoning-jsonl "$REASONING_ROOT/reasoning_stage${stage}.jsonl" \
    --output "$OUT_ROOT/train_stage${stage}.jsonl" \
    "${args[@]}"
}

merge_stage 1
merge_stage 2
merge_stage 3
