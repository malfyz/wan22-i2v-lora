#!/usr/bin/env bash
# wan22_bootstrap.sh — HEADLESS setup only (no Gradio)
set -euo pipefail

# --------- Basics / diagnostics ----------
WORKDIR="/workspace"
MODELS_DIR="$WORKDIR/models"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
mkdir -p "$WORKDIR" "$MODELS_DIR" "$HF_HOME"

echo "[BOOTSTRAP] Python: $(python -V 2>&1)"
python - <<'PY' || true
import torch, sys
print("[BOOTSTRAP] CUDA: torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

echo "[BOOTSTRAP] Listing /workspace:"
ls -la "$WORKDIR" | sed -n '1,120p' || true

# --------- Hugging Face CLI check ----------
if ! command -v hf >/dev/null 2>&1; then
  echo "[BOOTSTRAP] Installing huggingface_hub CLI (hf)…"
  python -m pip install --no-cache-dir --prefer-binary "huggingface_hub>=0.23.0" || true
fi

# Helper: tolerant download via hf CLI
dl () {
  # dl <repo> <path_in_repo> <local_dir>
  local repo="$1"; shift
  local path="$1"; shift
  local out="$1"; shift
  mkdir -p "$out"
  local base="$(basename "$path")"
  if [ ! -f "$out/$base" ]; then
    echo "  - fetching $repo/$path → $out"
    # prefer token if provided
    if [ -n "${HF_TOKEN:-}" ]; then
      hf download "$repo" "$path" --local-dir "$out" --token "$HF_TOKEN" --resume || true
    else
      hf download "$repo" "$path" --local-dir "$out" --resume || true
    fi
  else
    echo "  - exists: $out/$base"
  fi
}

echo "[WAN22] Downloading/validating required models…"

# Repackaged WAN 2.2 assets from Comfy-Org (public)
REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Text encoder
# (This is the encoder you showed earlier in your snippet)
dl "$REPO" "text_encoders/models_t5_umt5-xxl-enc-bf16_fully_uncensored.safetensors" "$MODELS_DIR/text_encoders"

# VAE
dl "$REPO" "vae/wan_2.1_vae.safetensors" "$MODELS_DIR/vae"

# I2V diffusers (fp16) — these arrive under split_files; move them to models/diffusion_models
TMP_SPLIT="$WORKDIR/split_files/diffusion_models"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$WORKDIR"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "$WORKDIR"
mkdir -p "$MODELS_DIR/diffusion_models"
if [ -d "$TMP_SPLIT" ]; then
  mv -f "$TMP_SPLIT/"*.safetensors "$MODELS_DIR/diffusion_models/" 2>/dev/null || true
fi

# --------- Dataset scaffolding (if you rely on local folders) ----------
mkdir -p \
  "$WORKDIR/datasets/character_images" \
  "$WORKDIR/datasets/val" \
  "$WORKDIR/outputs" \
  "$WORKDIR/cache" \
  "$WORKDIR/scripts" \
  "$WORKDIR/configs"

echo "[BOOTSTRAP] Models present:"
find "$MODELS_DIR" -maxdepth 3 -type f -name "*.safetensors" -printf "  %p\n" 2>/dev/null || true

echo "[BOOTSTRAP] Done. Not launching any UI."
echo "[BOOTSTRAP] Next:"
echo "  - SSH in and run your training scripts, e.g.:"
echo "      bash /workspace/scripts/train_i2v_high.sh"
echo "      bash /workspace/scripts/train_i2v_low.sh"
echo "  - Or run python entrypoints as needed."

# Keep container alive for SSH / interactive work
tail -f /dev/null
