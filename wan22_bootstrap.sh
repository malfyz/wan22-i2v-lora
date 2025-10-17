#!/usr/bin/env bash
set -euo pipefail

echo "[BOOTSTRAP] Python: $(python --version)"
echo "[BOOTSTRAP] CUDA: $(python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
)"

echo "[BOOTSTRAP] Listing /workspace:"
ls -la /workspace

APP_PATH="/workspace/app.py"
if [[ ! -f "$APP_PATH" ]]; then
  echo "[BOOTSTRAP][FATAL] $APP_PATH not found in container. Exiting."
  exit 1
fi

# Optional: Hugging Face cache path (keeps things tidy)
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

# Gradio defaults (can be overridden via env)
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

echo "[BOOTSTRAP] Starting Gradio app: $APP_PATH"
exec python -u "$APP_PATH" \
  --server-name "$GRADIO_SERVER_NAME" \
  --server-port "$GRADIO_SERVER_PORT"
