#!/usr/bin/env bash
set -euo pipefail

echo "[WAN22] Bootstrap starting…"
echo "[WAN22] WORKDIR=${WORKDIR:-/workspace}"
echo "[WAN22] MODELS_DIR=${MODELS_DIR:-/workspace/models}"
echo "[WAN22] DATASETS_DIR=${DATASETS_DIR:-/workspace/datasets}"
echo "[WAN22] OUTPUTS_DIR=${OUTPUTS_DIR:-/workspace/outputs}"
echo "[WAN22] CACHE_DIR=${CACHE_DIR:-/workspace/cache}"

# Ensure expected dirs
mkdir -p "${MODELS_DIR}/diffusion_models" \
         "${MODELS_DIR}/text_encoders" \
         "${MODELS_DIR}/vae" \
         "${DATASETS_DIR}" \
         "${OUTPUTS_DIR}" \
         "${CACHE_DIR}"

echo "[WAN22] Checking Hugging Face CLI…"
if command -v hf >/dev/null 2>&1; then
  HF_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN="huggingface-cli"
else
  echo "[WAN22] Installing huggingface_hub CLI…"
  python -m pip install --no-cache-dir "huggingface_hub[cli]==0.25.2"
  if command -v hf >/dev/null 2>&1; then HF_BIN="hf"; else HF_BIN="huggingface-cli"; fi
fi
echo "[WAN22] Using HF CLI: ${HF_BIN}"

# Optional: login with token (if provided via env HF_TOKEN)
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[WAN22] Logging into Hugging Face with provided token (hidden)…"
  # both CLIs support non-interactive login
  ${HF_BIN} login --token "${HF_TOKEN}" --add-to-git-credential no || true
else
  echo "[WAN22] HF_TOKEN not set — will download only public files or those that allow anonymous access."
fi

echo "[WAN22] Downloading/validating required models…"

# Helper: download a file if not present
#   dl <repo> <path_in_repo> <local_dir>
dl() {
  local repo="$1"; shift
  local path="$1"; shift
  local out="$1"; shift
  mkdir -p "$out"
  local base="$(basename "$path")"
  if [ ! -f "$out/$base" ]; then
    echo "  - fetching $repo/$path → $out"
    if [ -n "${HF_TOKEN:-}" ]; then
      ${HF_BIN} download "$repo" "$path" --local-dir "$out" --token "$HF_TOKEN" --resume
    else
      ${HF_BIN} download "$repo" "$path" --local-dir "$out" --resume
    fi
  else
    echo "  - exists: $out/$base"
  fi
}

# === WAN 2.2 I2V: Comfy-Org repackaged assets (public) ===
REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Text encoder (correct current filename)
dl "$REPO" "text_encoders/models_t5_umt5-xxl-enc-bf16_fully_uncensored.safetensors" "${MODELS_DIR}/text_encoders"

# VAE
dl "$REPO" "vae/wan_2.1_vae.safetensors" "${MODELS_DIR}/vae"

# I2V diffusers (fp16) — these are provided via split_files layout in the repo
TMP_SPLIT="${WORKDIR:-/workspace}/split_files/diffusion_models"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "${WORKDIR:-/workspace}"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "${WORKDIR:-/workspace}"
mkdir -p "${MODELS_DIR}/diffusion_models"
if [ -d "$TMP_SPLIT" ]; then
  echo "  - moving split_files -> ${MODELS_DIR}/diffusion_models"
  mv -f "$TMP_SPLIT/"*.safetensors "${MODELS_DIR}/diffusion_models/" 2>/dev/null || true
fi

# === (Optional) Your dataset fetch/validate goes here ===
# Example patterns (uncomment/adjust to your real sources):
#
# echo "[WAN22] Dataset setup…"
# # If hosted on HF Datasets:
# # dl "username/dataset-repo" "data/train.zip" "${DATASETS_DIR}/character_images"
# #
# # If you have a direct URL (use aria2 for speed/retries):
# # if [ ! -f "${DATASETS_DIR}/character_images/train.zip" ]; then
# #   aria2c -x 8 -s 8 -o "${DATASETS_DIR}/character_images/train.zip" "https://example.com/train.zip"
# #   unzip -o "${DATASETS_DIR}/character_images/train.zip" -d "${DATASETS_DIR}/character_images"
# # fi

echo "[WAN22] Model & dataset bootstrap complete."

# You can put any light-weight index/checks here (e.g., count files, md5, etc.)
echo "[WAN22] Summary:"
echo "  - text encoders: $(ls -1 ${MODELS_DIR}/text_encoders | wc -l || true)"
echo "  - vae:           $(ls -1 ${MODELS_DIR}/vae | wc -l || true)"
echo "  - diffusers:     $(ls -1 ${MODELS_DIR}/diffusion_models | wc -l || true)"

# DO NOT launch any UI here. The Docker ENTRYPOINT will keep the container alive
# so you can SSH in and start training manually.
echo "[WAN22] Bootstrap finished."
