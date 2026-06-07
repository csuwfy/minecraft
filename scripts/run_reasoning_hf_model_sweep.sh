#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/reasoning}"
STAGE="${STAGE:-3}"
LIMIT="${LIMIT:-200}"
START="${START:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"

# Keep this list small for quality sweeps. Run full annotation only after audit.
MODELS="${MODELS:-Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-7B-Instruct HuggingFaceTB/SmolVLM2-2.2B-Instruct OpenGVLab/InternVL3-2B}"

mkdir -p "$OUT_ROOT"

MAX_FRAMES=4
EXTRA_ARGS=()
if [[ "$STAGE" == "1" ]]; then
  MAX_FRAMES=20
  EXTRA_ARGS+=(--tess-stage1-index "$DATA_ROOT/tess_stage1_parquet_index.json")
fi

for MODEL in $MODELS; do
  MODEL_SAFE="${MODEL//\//_}"
  OUTPUT="$OUT_ROOT/reasoning_stage${STAGE}_${MODEL_SAFE}_limit${LIMIT}_start${START}.jsonl"
  echo "reasoning_sweep_model=$MODEL output=$OUTPUT"
  set +e
  python -m training.reasoning_hf_teacher \
      --manifest "$DATA_ROOT/train_stage${STAGE}.jsonl" \
      --image-root "$DATA_ROOT" \
      --stage "$STAGE" \
      --max-frames "$MAX_FRAMES" \
      --model "$MODEL" \
      --torch-dtype "$TORCH_DTYPE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --output "$OUTPUT" \
      --start "$START" \
      --limit "$LIMIT" \
      --resume \
      "${EXTRA_ARGS[@]}"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "reasoning_sweep_failed model=$MODEL status=$status" >&2
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      exit "$status"
    fi
    continue
  fi

  python -m training.reasoning_annotation audit \
    --reasoning-jsonl "$OUTPUT" \
    --drop-risky-sentences \
    --examples 5
done
