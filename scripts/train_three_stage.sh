#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TESS_STAGE1_IMAGE_CACHE_GROUPS="${TESS_STAGE1_IMAGE_CACHE_GROUPS:-2}"

run_stage() {
  local config_path="$1"
  if [ -n "$DEEPSPEED_CONFIG" ]; then
    accelerate launch --num_processes "$NUM_PROCESSES" -m training.train_sft \
      --config "$config_path" \
      --set "training.deepspeed=$DEEPSPEED_CONFIG"
  else
    python -m training.train_sft --config "$config_path"
  fi
}

if [ -n "$DEEPSPEED_CONFIG" ]; then
  python -m pip show deepspeed >/dev/null
fi

run_stage configs/training/paper_qwen25vl3b_stage1_full_sft.json
run_stage configs/training/paper_qwen25vl3b_stage2_full_sft.json
run_stage configs/training/paper_qwen25vl3b_stage3_full_sft.json
