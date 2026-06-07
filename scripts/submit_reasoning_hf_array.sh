#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/reasoning}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
STAGE="${STAGE:?Set STAGE to 1, 2, or 3.}"
SPLIT="${SPLIT:-train}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TOTAL_RECORDS="${TOTAL_RECORDS:?Set TOTAL_RECORDS to the manifest line count.}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
QUEUE="${QUEUE:-v1_gpu72}"
GPU_TYPE="${GPU_TYPE:-L40S}"
NCPUS="${NCPUS:-8}"
MEM="${MEM:-96gb}"
WALLTIME="${WALLTIME:-24:00:00}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}"
TEMPERATURE="${TEMPERATURE:-0.0}"
QSUB="${QSUB:-qsub}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

MODEL_SAFE="${MODEL//\//_}"
MAX_FRAMES=4
MAX_IMAGES=4
EXTRA_ARGS=""
if [[ "$STAGE" == "1" ]]; then
  MAX_FRAMES=20
  MAX_IMAGES=20
  EXTRA_ARGS="--tess-stage1-index $DATA_ROOT/tess_stage1_parquet_index.json"
fi
MAX_IMAGES="${MAX_IMAGES_OVERRIDE:-$MAX_IMAGES}"

if [[ "$TOTAL_RECORDS" -le 0 || "$CHUNK_SIZE" -le 0 ]]; then
  echo "TOTAL_RECORDS and CHUNK_SIZE must be positive." >&2
  exit 2
fi

CHUNK_COUNT=$(( (TOTAL_RECORDS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
LAST_INDEX=$(( CHUNK_COUNT - 1 ))
SCRIPT="$OUT_ROOT/pbs_reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_array.pbs"
ARRAY_DIRECTIVE=""
if [[ "$CHUNK_COUNT" -gt 1 ]]; then
  ARRAY_DIRECTIVE="#PBS -J 0-${LAST_INDEX}%${MAX_CONCURRENT}"
fi

cat > "$SCRIPT" <<PBS
#!/bin/bash
#PBS -N rsn_s${STAGE}_${SPLIT}_${MODEL_SAFE}
$ARRAY_DIRECTIVE
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}:ngpus=1:gpu_type=${GPU_TYPE}
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -o ${LOG_ROOT}/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_array.log
#PBS -e ${LOG_ROOT}/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_array.err
set -euo pipefail
cd "$PROJECT_ROOT"
export HF_HOME="\${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

index="\${PBS_ARRAY_INDEX:-0}"
start=\$((index * ${CHUNK_SIZE}))
remaining=\$(( ${TOTAL_RECORDS} - start ))
if [[ "\$remaining" -le 0 ]]; then
  echo "skip_empty_chunk index=\$index start=\$start total=${TOTAL_RECORDS}"
  exit 0
fi
limit=${CHUNK_SIZE}
if [[ "\$remaining" -lt "\$limit" ]]; then
  limit="\$remaining"
fi
chunk_id=\$(printf "%09d" "\$start")
output="$OUT_ROOT/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_start\${chunk_id}_limit\${limit}.jsonl"

python -m training.reasoning_hf_teacher \\
  --manifest "$DATA_ROOT/${SPLIT}_stage${STAGE}.jsonl" \\
  --image-root "$DATA_ROOT" \\
  --stage "$STAGE" \\
  --max-frames "$MAX_FRAMES" \\
  --max-images "$MAX_IMAGES" \\
  --model "$MODEL" \\
  --torch-dtype "$TORCH_DTYPE" \\
  --max-new-tokens "$MAX_NEW_TOKENS" \\
  --temperature "$TEMPERATURE" \\
  --output "\$output" \\
  --start "\$start" \\
  --limit "\$limit" \\
  --resume \\
  $EXTRA_ARGS
python -m training.reasoning_annotation audit \\
  --reasoning-jsonl "\$output" \\
  --drop-risky-sentences \\
  --examples 3
PBS

"$QSUB" "$SCRIPT"
echo "submitted_reasoning_array stage=$STAGE split=$SPLIT model=$MODEL chunks=$CHUNK_COUNT chunk_size=$CHUNK_SIZE max_concurrent=$MAX_CONCURRENT script=$SCRIPT"
