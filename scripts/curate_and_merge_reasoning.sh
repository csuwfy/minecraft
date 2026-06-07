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

if [[ -e "$DATA_ROOT/val_stage${STAGE}.jsonl" ]]; then
  cp "$DATA_ROOT/val_stage${STAGE}.jsonl" "$MERGED_ROOT/val_stage${STAGE}.jsonl"
fi

echo "curated_reasoning=$CURATED"
echo "rejected_reasoning=$REJECTED"
echo "merged_manifest=$MERGED"
