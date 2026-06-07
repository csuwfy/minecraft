#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/reasoning/openai_compatible_multi_vlm.example.json}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/reasoning}"
LIMIT="${LIMIT:-0}"
START="${START:-0}"
SPLIT="${SPLIT:-train}"

mkdir -p "$OUT_ROOT"

annotate_stage() {
  local stage="$1"
  local max_frames="$2"
  local max_images="$max_frames"
  local output="$OUT_ROOT/reasoning_stage${stage}.jsonl"
  if [[ "$SPLIT" != "train" ]]; then
    output="$OUT_ROOT/reasoning_stage${stage}_${SPLIT}.jsonl"
  fi
  max_images="${MAX_IMAGES_OVERRIDE:-$max_images}"
  local extra_args=()
  if [[ "$stage" == "1" ]]; then
    extra_args+=(--tess-stage1-index "$DATA_ROOT/tess_stage1_parquet_index.json")
  fi
  python -m training.reasoning_annotation annotate \
    --manifest "$DATA_ROOT/${SPLIT}_stage${stage}.jsonl" \
    --image-root "$DATA_ROOT" \
    --stage "$stage" \
    --max-frames "$max_frames" \
    --max-images "$max_images" \
    --config "$CONFIG" \
    --output "$output" \
    --start "$START" \
    --limit "$LIMIT" \
    --resume \
    "${extra_args[@]}"
}

annotate_stage 1 20
annotate_stage 2 4
annotate_stage 3 4
