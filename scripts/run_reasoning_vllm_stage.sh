#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/reasoning}"
STAGE="${STAGE:-3}"
LIMIT="${LIMIT:-1000}"
START="${START:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
API_KEY="${VLLM_API_KEY:-dummy-key}"
MODEL_SAFE="${MODEL//\//_}"
MAX_IMAGES=4
LIMIT_MM_IMAGE=4
if [[ "$STAGE" == "1" ]]; then
  MAX_IMAGES=20
  LIMIT_MM_IMAGE=20
fi
MAX_IMAGES="${MAX_IMAGES_OVERRIDE:-$MAX_IMAGES}"
LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE_OVERRIDE:-$LIMIT_MM_IMAGE}"

mkdir -p "$OUT_ROOT"

CONFIG="$OUT_ROOT/vllm_stage${STAGE}_config.json"
cat > "$CONFIG" <<EOF
{
  "max_images": $MAX_IMAGES,
  "max_image_side": 448,
  "jpeg_quality": 85,
  "sleep_seconds": 0.0,
  "providers": [
    {
      "name": "local_vllm_stage${STAGE}",
      "model": "$MODEL",
      "api_url": "http://127.0.0.1:$PORT/v1",
      "api_key": "$API_KEY",
      "temperature": 0.2,
      "max_tokens": 320,
      "timeout": 180
    }
  ]
}
EOF

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt image="$LIMIT_MM_IMAGE" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

python - <<PY
import time
import requests
url = "http://127.0.0.1:$PORT/v1/models"
headers = {"Authorization": "Bearer $API_KEY"}
for attempt in range(180):
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            print("vllm_ready", r.text[:200], flush=True)
            break
    except Exception:
        pass
    time.sleep(5)
else:
    raise SystemExit("vLLM server did not become ready")
PY

MAX_FRAMES=4
EXTRA_ARGS=()
if [[ "$STAGE" == "1" ]]; then
  MAX_FRAMES=20
  EXTRA_ARGS+=(--tess-stage1-index "$DATA_ROOT/tess_stage1_parquet_index.json")
fi

python -m training.reasoning_annotation annotate \
  --manifest "$DATA_ROOT/train_stage${STAGE}.jsonl" \
  --image-root "$DATA_ROOT" \
  --stage "$STAGE" \
  --max-frames "$MAX_FRAMES" \
  --config "$CONFIG" \
  --output "$OUT_ROOT/reasoning_stage${STAGE}_${MODEL_SAFE}.jsonl" \
  --start "$START" \
  --limit "$LIMIT" \
  --resume \
  "${EXTRA_ARGS[@]}"
