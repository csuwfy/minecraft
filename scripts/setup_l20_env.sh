#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-combatvla-train}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if command -v conda >/dev/null 2>&1; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  python3 -m venv ".venv-$ENV_NAME"
  # shellcheck disable=SC1091
  source ".venv-$ENV_NAME/bin/activate"
fi

python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
python -m pip install -r requirements-training.txt

echo "Environment ready. Activate it before training:"
if command -v conda >/dev/null 2>&1; then
  echo "  conda activate $ENV_NAME"
else
  echo "  source .venv-$ENV_NAME/bin/activate"
fi
