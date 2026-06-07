#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/reasoning}"
MERGED_ROOT="${MERGED_ROOT:-$PROJECT_ROOT/data/minecraft_reasoning}"
STAGE="${STAGE:-3}"

if [[ "${REASONING_JSONL:-}" == "" ]]; then
  echo "Set REASONING_JSONL to one or more teacher JSONL files." >&2
  echo "Example: REASONING_JSONL=\"data/reasoning/qwen3b.jsonl data/reasoning/qwen7b.jsonl\" $0" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT" "$MERGED_ROOT"

read -r -a REASONING_FILES <<< "$REASONING_JSONL"
CURATED="$OUT_ROOT/reasoning_stage${STAGE}_curated.jsonl"
REJECTED="$OUT_ROOT/reasoning_stage${STAGE}_rejected.jsonl"
MERGED="$MERGED_ROOT/train_stage${STAGE}.jsonl"
VAL_CURATED="$OUT_ROOT/reasoning_stage${STAGE}_val_curated.jsonl"
VAL_REJECTED="$OUT_ROOT/reasoning_stage${STAGE}_val_rejected.jsonl"
VAL_MERGED="$MERGED_ROOT/val_stage${STAGE}.jsonl"

python -m training.reasoning_annotation curate \
  --reasoning-jsonl "${REASONING_FILES[@]}" \
  --output "$CURATED" \
  --rejected-output "$REJECTED" \
  --preferred-model Qwen2.5-VL-7B-Instruct \
  --preferred-model Qwen2.5-VL-3B-Instruct \
  --preferred-model InternVL3-2B \
  --preferred-model SmolVLM2-2.2B-Instruct \
  --examples 10

python -m training.reasoning_annotation merge \
  --manifest "$DATA_ROOT/train_stage${STAGE}.jsonl" \
  --reasoning-jsonl "$CURATED" \
  --output "$MERGED" \
  --require-reasoning

if [[ "${VAL_REASONING_JSONL:-}" != "" ]]; then
  read -r -a VAL_REASONING_FILES <<< "$VAL_REASONING_JSONL"
  python -m training.reasoning_annotation curate \
    --reasoning-jsonl "${VAL_REASONING_FILES[@]}" \
    --output "$VAL_CURATED" \
    --rejected-output "$VAL_REJECTED" \
    --preferred-model Qwen2.5-VL-7B-Instruct \
    --preferred-model Qwen2.5-VL-3B-Instruct \
    --preferred-model InternVL3-2B \
    --preferred-model SmolVLM2-2.2B-Instruct \
    --examples 10

  python -m training.reasoning_annotation merge \
    --manifest "$DATA_ROOT/val_stage${STAGE}.jsonl" \
    --reasoning-jsonl "$VAL_CURATED" \
    --output "$VAL_MERGED" \
    --require-reasoning
else
  echo "val_manifest_not_written_without_VAL_REASONING_JSONL=$MERGED_ROOT/val_stage${STAGE}.jsonl" >&2
fi

echo "curated_reasoning=$CURATED"
echo "rejected_reasoning=$REJECTED"
echo "merged_manifest=$MERGED"
if [[ "${VAL_REASONING_JSONL:-}" != "" ]]; then
  echo "val_curated_reasoning=$VAL_CURATED"
  echo "val_rejected_reasoning=$VAL_REJECTED"
  echo "val_merged_manifest=$VAL_MERGED"
fi
