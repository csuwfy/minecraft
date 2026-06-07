#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATASETS_DIR="${DATASETS_DIR:-$PROJECT_ROOT/datasets}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/minecraft_full}"

STAGE1_TRAIN_COUNT="${STAGE1_TRAIN_COUNT:-100000000}"
STAGE1_VAL_COUNT="${STAGE1_VAL_COUNT:-10000}"
STAGE2_TRAIN_COUNT="${STAGE2_TRAIN_COUNT:-100000000}"
STAGE2_VAL_COUNT="${STAGE2_VAL_COUNT:-10000}"
STAGE3_TRAIN_COUNT="${STAGE3_TRAIN_COUNT:-100000000}"
STAGE3_VAL_COUNT="${STAGE3_VAL_COUNT:-10000}"

mkdir -p "$DATA_ROOT"

python -m training.convert_tess_stage1 \
  --data-dir "$DATASETS_DIR/tess_stage1" \
  --output-root "$DATA_ROOT" \
  --train-count "$STAGE1_TRAIN_COUNT" \
  --val-count "$STAGE1_VAL_COUNT" \
  --val-every 20 \
  --max-frames 20 \
  --task "Play Minecraft." \
  --reference-parquet-frames

python -m training.convert_tess_stage2 \
  --data-dir "$DATASETS_DIR/tess_stage2" \
  --output-root "$DATA_ROOT" \
  --train-count "$STAGE2_TRAIN_COUNT" \
  --val-count "$STAGE2_VAL_COUNT" \
  --max-frames 4

python -m training.convert_optimus_mgoa_stage3 \
  --root "$DATASETS_DIR/optimus2_mgoa" \
  --output-root "$DATA_ROOT" \
  --train-count "$STAGE3_TRAIN_COUNT" \
  --val-count "$STAGE3_VAL_COUNT" \
  --frame-stride 4 \
  --max-window-frames 4 \
  --skip-existing-frames

python -m training.validate_manifest \
  --manifest "$DATA_ROOT/train_stage1.jsonl" \
  --image-root "$DATA_ROOT" \
  --stage 1 \
  --max-frames 20 \
  --tess-stage1-index "$DATA_ROOT/tess_stage1_parquet_index.json" \
  --limit 8 \
  --load-images

python -m training.validate_manifest \
  --manifest "$DATA_ROOT/train_stage2.jsonl" \
  --image-root "$DATA_ROOT" \
  --stage 2 \
  --max-frames 4 \
  --limit 8 \
  --load-images

python -m training.validate_manifest \
  --manifest "$DATA_ROOT/train_stage3.jsonl" \
  --image-root "$DATA_ROOT" \
  --stage 3 \
  --max-frames 4 \
  --limit 8 \
  --load-images

wc -l "$DATA_ROOT"/train_stage*.jsonl "$DATA_ROOT"/val_stage*.jsonl
