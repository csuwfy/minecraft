#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATASETS_DIR="${DATASETS_DIR:-$PROJECT_ROOT/datasets}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
DOWNLOAD_CRAFTJARVIS="${DOWNLOAD_CRAFTJARVIS:-0}"

mkdir -p "$DATASETS_DIR" "$HF_HOME"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

python -m pip install -q "huggingface_hub[hf_transfer]>=0.24.0" hf_transfer

huggingface-cli download TESS-Computer/minecraft-vla-stage1 \
  --repo-type dataset \
  --local-dir "$DATASETS_DIR/tess_stage1"

huggingface-cli download TESS-Computer/minecraft-vla-stage2 \
  --repo-type dataset \
  --local-dir "$DATASETS_DIR/tess_stage2"

huggingface-cli download iLearn-Lab/Optimus-2-MGOA \
  action.tar.gz task_description_map.json \
  --repo-type dataset \
  --local-dir "$DATASETS_DIR/optimus2_mgoa"

for part in aa ab ac ad ae af ag ah ai aj ak al am; do
  huggingface-cli download iLearn-Lab/Optimus-2-MGOA \
    "video.tar.gz.part.${part}" \
    --repo-type dataset \
    --local-dir "$DATASETS_DIR/optimus2_mgoa"
done

(
  cd "$DATASETS_DIR/optimus2_mgoa"
  cat video.tar.gz.part.* > video.tar.gz
  tar -tzf video.tar.gz >/dev/null
)

if [ "$DOWNLOAD_CRAFTJARVIS" = "1" ]; then
  huggingface-cli download CraftJarvis/minecraft-vla-sft \
    --repo-type dataset \
    --local-dir "$DATASETS_DIR/craftjarvis_minecraft_vla_sft"
fi

du -sh "$DATASETS_DIR" "$DATASETS_DIR"/* || true
