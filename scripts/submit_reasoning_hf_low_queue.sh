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
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
START_OFFSET="${START_OFFSET:-0}"
RECORD_LIMIT="${RECORD_LIMIT:-0}"
WORKERS="${WORKERS:-10}"
QUEUE="${QUEUE:-v1_gpu72}"
GPU_TYPE="${GPU_TYPE:-L40S}"
NCPUS="${NCPUS:-8}"
MEM="${MEM:-96gb}"
WALLTIME="${WALLTIME:-72:00:00}"
BATCH_SIZE="${BATCH_SIZE:-1}"
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

if [[ "$TOTAL_RECORDS" -le 0 || "$CHUNK_SIZE" -le 0 || "$START_OFFSET" -lt 0 || "$RECORD_LIMIT" -lt 0 ]]; then
  echo "TOTAL_RECORDS and CHUNK_SIZE must be positive; START_OFFSET and RECORD_LIMIT must be non-negative." >&2
  exit 2
fi
if [[ "$WORKERS" -le 0 || "$WORKERS" -gt 10 ]]; then
  echo "WORKERS must be between 1 and 10." >&2
  exit 2
fi
if [[ "$BATCH_SIZE" -le 0 ]]; then
  echo "BATCH_SIZE must be positive." >&2
  exit 2
fi
if [[ "$START_OFFSET" -ge "$TOTAL_RECORDS" ]]; then
  echo "START_OFFSET must be smaller than TOTAL_RECORDS." >&2
  exit 2
fi
if [[ $((START_OFFSET % CHUNK_SIZE)) -ne 0 ]]; then
  echo "START_OFFSET must align to CHUNK_SIZE so chunk names preserve coverage." >&2
  exit 2
fi

RUN_RECORDS=$(( TOTAL_RECORDS - START_OFFSET ))
if [[ "$RECORD_LIMIT" -gt 0 && "$RECORD_LIMIT" -lt "$RUN_RECORDS" ]]; then
  RUN_RECORDS="$RECORD_LIMIT"
fi
if [[ $((RUN_RECORDS % CHUNK_SIZE)) -ne 0 && $((START_OFFSET + RUN_RECORDS)) -ne "$TOTAL_RECORDS" ]]; then
  echo "RECORD_LIMIT must align to CHUNK_SIZE unless it reaches TOTAL_RECORDS." >&2
  exit 2
fi

MODEL_SAFE="${MODEL//\//_}"
MAX_FRAMES=4
MAX_IMAGES=4
EXTRA_ARGS=()
if [[ "$STAGE" == "1" ]]; then
  MAX_FRAMES=20
  MAX_IMAGES=20
  EXTRA_ARGS+=(--tess-stage1-index "$DATA_ROOT/tess_stage1_parquet_index.json")
fi
MAX_IMAGES="${MAX_IMAGES_OVERRIDE:-$MAX_IMAGES}"
EXTRA_ARGS_LINE=""
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  EXTRA_ARGS_LINE="  ${EXTRA_ARGS[0]} \"${EXTRA_ARGS[1]}\""
fi

CHUNK_COUNT=$(( (RUN_RECORDS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
if [[ "$WORKERS" -gt "$CHUNK_COUNT" ]]; then
  WORKERS="$CHUNK_COUNT"
fi

submitted=0
for ((worker=0; worker<WORKERS; worker++)); do
  worker_chunk_start=$(( worker * CHUNK_COUNT / WORKERS ))
  worker_chunk_end=$(( (worker + 1) * CHUNK_COUNT / WORKERS ))
  if [[ "$worker_chunk_start" -ge "$worker_chunk_end" ]]; then
    continue
  fi
  worker_start=$(( START_OFFSET + worker_chunk_start * CHUNK_SIZE ))
  worker_end=$(( START_OFFSET + worker_chunk_end * CHUNK_SIZE ))
  total_end=$(( START_OFFSET + RUN_RECORDS ))
  if [[ "$worker_end" -gt "$total_end" ]]; then
    worker_end="$total_end"
  fi
  worker_records=$(( worker_end - worker_start ))
  if [[ "$worker_records" -le 0 ]]; then
    continue
  fi
  worker_tag="start$(printf "%09d" "$worker_start")_records${worker_records}"
  script="$OUT_ROOT/pbs_reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_${worker_tag}_worker.pbs"
  cat > "$script" <<PBS
#!/bin/bash
#PBS -N rsn_s${STAGE}_${SPLIT}_w${worker}
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}:ngpus=1:gpu_type=${GPU_TYPE}
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -o ${LOG_ROOT}/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_${worker_tag}_worker.log
#PBS -e ${LOG_ROOT}/reasoning_stage${STAGE}_${SPLIT}_${MODEL_SAFE}_${worker_tag}_worker.err
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
  --split "$SPLIT" \\
  --max-frames "$MAX_FRAMES" \\
  --max-images "$MAX_IMAGES" \\
  --model "$MODEL" \\
  --torch-dtype "$TORCH_DTYPE" \\
  --max-new-tokens "$MAX_NEW_TOKENS" \\
  --temperature "$TEMPERATURE" \\
  --chunk-output-dir "$OUT_ROOT" \\
  --chunk-size "$CHUNK_SIZE" \\
  --start "$worker_start" \\
  --limit "$worker_records" \\
  --batch-size "$BATCH_SIZE" \\
  --resume \\
  --report-every 100${EXTRA_ARGS_LINE:+ \\
$EXTRA_ARGS_LINE}
PBS
  "$QSUB" "$script"
  submitted=$((submitted + 1))
done

echo "submitted_low_queue_workers=$submitted model=$MODEL stage=$STAGE split=$SPLIT start_offset=$START_OFFSET run_records=$RUN_RECORDS chunk_size=$CHUNK_SIZE workers=$WORKERS batch_size=$BATCH_SIZE"
