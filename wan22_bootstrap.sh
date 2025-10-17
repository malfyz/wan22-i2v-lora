#!/usr/bin/env bash
set -euo pipefail

echo "[BOOTSTRAP] Python: $(python --version)"
python - <<'PY'
import torch
print("[BOOTSTRAP] torch:", torch.__version__)
print("[BOOTSTRAP] cuda available:", torch.cuda.is_available())
print("[BOOTSTRAP] device count:", torch.cuda.device_count())
PY

echo "[BOOTSTRAP] PATH: $PATH"
echo "[BOOTSTRAP] which python: $(which python)"
echo "[BOOTSTRAP] pip site info:"
python - <<'PY'
import pip, site
print("pip file:", pip.__file__)
print("site-packages:", site.getsitepackages())
PY

echo "[BOOTSTRAP] Listing /workspace:"
ls -la /workspace || true

# Minimal runtime guard: ensure gradio is importable; if not, install once.
python - <<'PY' 2>/dev/null || (echo "[BOOTSTRAP] gradio missing at runtime; installing now..." && python -m pip install --no-cache-dir "gradio==4.45.0")
import gradio
print("[BOOTSTRAP] gradio version:", gradio.__version__)
PY

APP_PATH="/workspace/app.py"
if [[ ! -f "$APP_PATH" ]]; then
  echo "[BOOTSTRAP][FATAL] $APP_PATH not found. Exiting."
  exit 1
fi

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

echo "[BOOTSTRAP] Starting Gradio app: $APP_PATH"
exec python -u "$APP_PATH" \
  --server-name "$GRADIO_SERVER_NAME" \
  --server-port "$GRADIO_SERVER_PORT"
