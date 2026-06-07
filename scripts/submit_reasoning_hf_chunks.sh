#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/reasoning}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
HF_HOME_DIR="${HF_HOME_DIR:-${HF_HOME:-$PROJECT_ROOT/cache/huggingface}}"
STAGE="${STAGE:-3}"
SPLIT="${SPLIT:-train}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
TOTAL_RECORDS="${TOTAL_RECORDS:?Set TOTAL_RECORDS to the manifest line count.}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
START_OFFSET="${START_OFFSET:-0}"
RECORD_LIMIT="${RECORD_LIMIT:-0}"
MAX_JOBS="${MAX_JOBS:-0}"
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

submitted=0
if [[ "$TOTAL_RECORDS" -le 0 || "$CHUNK_SIZE" -le 0 || "$START_OFFSET" -lt 0 || "$RECORD_LIMIT" -lt 0 ]]; then
  echo "TOTAL_RECORDS and CHUNK_SIZE must be positive; START_OFFSET and RECORD_LIMIT must be non-negative." >&2
  exit 2
fi
if [[ "$START_OFFSET" -ge "$TOTAL_RECORDS" ]]; then
  echo "START_OFFSET must be smaller than TOTAL_RECORDS." >&2
  exit 2
fi

end="$TOTAL_RECORDS"
if [[ "$RECORD_LIMIT" -gt 0 && $((START_OFFSET + RECORD_LIMIT)) -lt "$end" ]]; then
  end=$((START_OFFSET + RECORD_LIMIT))
fi

for ((start=START_OFFSET; start<end; start+=CHUNK_SIZE)); do
  if [[ "$MAX_JOBS" -gt 0 && "$submitted" -ge "$MAX_JOBS" ]]; then
    break
  fi
  limit="$CHUNK_SIZE"
  remaining=$((end - start))
  if [[ "$remaining" -lt "$limit" ]]; then
    limit="$remaining"
  fi
  chunk_id=$(printf "%09d" "$start")
  output="$OUT_ROOT/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_start${chunk_id}_limit${limit}.jsonl"
  script="$OUT_ROOT/pbs_reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_start${chunk_id}.pbs"
  cat > "$script" <<PBS
#!/bin/bash
#PBS -N rsn_s${STAGE}_${SPLIT}_${chunk_id}
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}:ngpus=1:gpu_type=${GPU_TYPE}
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -o ${LOG_ROOT}/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_start${chunk_id}.log
#PBS -e ${LOG_ROOT}/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_start${chunk_id}.err
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
  --output "$output" \\
  --start "$start" \\
  --limit "$limit" \\
  --resume${EXTRA_ARGS_LINE:+ \\
$EXTRA_ARGS_LINE}
"\$PYTHON_BIN" -m training.reasoning_annotation audit \\
  --reasoning-jsonl "$output" \\
  --drop-risky-sentences \\
  --examples 3
PBS
  "$QSUB" "$script"
  submitted=$((submitted + 1))
done

echo "submitted_reasoning_chunks=$submitted model=$MODEL stage=$STAGE split=$SPLIT start_offset=$START_OFFSET end=$end chunk_size=$CHUNK_SIZE"
