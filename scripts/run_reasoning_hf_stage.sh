#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/reasoning}"
STAGE="${STAGE:-3}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-1000}"
START="${START:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MODEL_SAFE="${MODEL//\//_}"

mkdir -p "$OUT_ROOT"

MAX_FRAMES=4
EXTRA_ARGS=()
if [[ "$STAGE" == "1" ]]; then
  MAX_FRAMES=20
  EXTRA_ARGS+=(--tess-stage1-index "$DATA_ROOT/tess_stage1_parquet_index.json")
fi

python -m training.reasoning_hf_teacher \
  --manifest "$DATA_ROOT/${SPLIT}_stage${STAGE}.jsonl" \
  --image-root "$DATA_ROOT" \
  --stage "$STAGE" \
  --max-frames "$MAX_FRAMES" \
  --model "$MODEL" \
  --torch-dtype "$TORCH_DTYPE" \
  --output "$OUT_ROOT/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}.jsonl" \
  --start "$START" \
  --limit "$LIMIT" \
  --resume \
  "${EXTRA_ARGS[@]}"
