#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/reasoning}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
HF_HOME_DIR="${HF_HOME_DIR:-${HF_HOME:-$PROJECT_ROOT/cache/huggingface}}"
STAGE="${STAGE:?Set STAGE to 1, 2, or 3.}"
SPLIT="${SPLIT:-train}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TOTAL_RECORDS="${TOTAL_RECORDS:?Set TOTAL_RECORDS to the manifest line count.}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
START_OFFSET="${START_OFFSET:-0}"
RECORD_LIMIT="${RECORD_LIMIT:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
QUEUE="${QUEUE:-v1_gpu72}"
GPU_TYPE="${GPU_TYPE:-L40S}"
NCPUS="${NCPUS:-8}"
MEM="${MEM:-96gb}"
WALLTIME="${WALLTIME:-24:00:00}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}"
TEMPERATURE="${TEMPERATURE:-0.0}"
QSUB="${QSUB:-qsub}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${ENV:-}" ]]; then
    PYTHON_BIN="$ENV/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

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
EXTRA_ARGS_LINE=""
if [[ "$EXTRA_ARGS" != "" ]]; then
  EXTRA_ARGS_LINE="  $EXTRA_ARGS"
fi

if [[ "$TOTAL_RECORDS" -le 0 || "$CHUNK_SIZE" -le 0 || "$START_OFFSET" -lt 0 || "$RECORD_LIMIT" -lt 0 ]]; then
  echo "TOTAL_RECORDS and CHUNK_SIZE must be positive; START_OFFSET and RECORD_LIMIT must be non-negative." >&2
  exit 2
fi
if [[ "$START_OFFSET" -ge "$TOTAL_RECORDS" ]]; then
  echo "START_OFFSET must be smaller than TOTAL_RECORDS." >&2
  exit 2
fi

RUN_RECORDS=$(( TOTAL_RECORDS - START_OFFSET ))
if [[ "$RECORD_LIMIT" -gt 0 && "$RECORD_LIMIT" -lt "$RUN_RECORDS" ]]; then
  RUN_RECORDS="$RECORD_LIMIT"
fi
CHUNK_COUNT=$(( (RUN_RECORDS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
LAST_INDEX=$(( CHUNK_COUNT - 1 ))
RANGE_TAG="start$(printf "%09d" "$START_OFFSET")_records${RUN_RECORDS}"
SCRIPT="$OUT_ROOT/pbs_reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_${RANGE_TAG}_array.pbs"
ARRAY_DIRECTIVE=""
if [[ "$CHUNK_COUNT" -gt 1 ]]; then
  ARRAY_DIRECTIVE="#PBS -J 0-${LAST_INDEX}%${MAX_CONCURRENT}"
fi

cat > "$SCRIPT" <<PBS
#!/bin/bash
#PBS -N rsn_s${STAGE}_${SPLIT}_${RANGE_TAG}
$ARRAY_DIRECTIVE
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}:ngpus=1:gpu_type=${GPU_TYPE}
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -o ${LOG_ROOT}/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_${RANGE_TAG}_array.log
#PBS -e ${LOG_ROOT}/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_${RANGE_TAG}_array.err
set -euo pipefail
cd "$PROJECT_ROOT"
export HF_HOME="\${HF_HOME:-$HF_HOME_DIR}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="$PYTHON_BIN"
if [[ ! -x "\$PYTHON_BIN" ]]; then
  if ! command -v "\$PYTHON_BIN" >/dev/null 2>&1; then
    echo "PYTHON_BIN is not executable and not on PATH: \$PYTHON_BIN" >&2
    exit 127
  fi
fi

index="\${PBS_ARRAY_INDEX:-0}"
start=\$(( ${START_OFFSET} + index * ${CHUNK_SIZE} ))
batch_end=$(( ${START_OFFSET} + ${RUN_RECORDS} ))
remaining=\$(( batch_end - start ))
if [[ "\$remaining" -le 0 ]]; then
  echo "skip_empty_chunk index=\$index start=\$start batch_end=\$batch_end total=${TOTAL_RECORDS}"
  exit 0
fi
limit=${CHUNK_SIZE}
if [[ "\$remaining" -lt "\$limit" ]]; then
  limit="\$remaining"
fi
chunk_id=\$(printf "%09d" "\$start")
output="$OUT_ROOT/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_start\${chunk_id}_limit\${limit}.jsonl"

"\$PYTHON_BIN" -m training.reasoning_hf_teacher \\
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
  --resume${EXTRA_ARGS_LINE:+ \\
$EXTRA_ARGS_LINE}
"\$PYTHON_BIN" -m training.reasoning_annotation audit \\
  --reasoning-jsonl "\$output" \\
  --drop-risky-sentences \\
  --examples 3
PBS

"$QSUB" "$SCRIPT"
echo "submitted_reasoning_array stage=$STAGE split=$SPLIT model=$MODEL start_offset=$START_OFFSET run_records=$RUN_RECORDS chunks=$CHUNK_COUNT chunk_size=$CHUNK_SIZE max_concurrent=$MAX_CONCURRENT script=$SCRIPT"
