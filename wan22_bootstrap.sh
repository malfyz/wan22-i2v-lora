#!/usr/bin/env bash
set -euo pipefail

echo "[WAN22] Bootstrap starting…"

# --- Paths (can be overridden by env) ---
WORKDIR="${WORKDIR:-/workspace}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
DATASETS_DIR="${DATASETS_DIR:-/workspace/datasets}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/workspace/outputs}"
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"

mkdir -p "${MODELS_DIR}/diffusion_models" \
         "${MODELS_DIR}/text_encoders" \
         "${MODELS_DIR}/vae" \
         "${DATASETS_DIR}/character_images" \
         "${DATASETS_DIR}/val" \
         "${OUTPUTS_DIR}" \
         "${CACHE_DIR}" \
         "${WORKDIR}/scripts" \
         "${WORKDIR}/configs"

# Normalize CRLF if edited on Windows
sed -i 's/\r$//' "$0" || true

# --- Hugging Face CLI ---
if command -v hf >/dev/null 2>&1; then
  HF_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN="huggingface-cli"
else
  echo "[WAN22] Installing huggingface_hub CLI…"
  python -m pip install --no-cache-dir 'huggingface_hub[cli]==0.25.2'
  if command -v hf >/dev/null 2>&1; then HF_BIN="hf"; else HF_BIN="huggingface-cli"; fi
fi
echo "[WAN22] Using HF CLI: ${HF_BIN}"

# Optional: login with token
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[WAN22] Logging into Hugging Face with token (hidden)…"
  ${HF_BIN} login --token "${HF_TOKEN}" --add-to-git-credential no || true
else
  echo "[WAN22] HF_TOKEN not set — public downloads only."
fi

# --- Install Musubi-Tuner at runtime (inside the pod) ---
if [ ! -d /opt/musubi-tuner ]; then
  echo "[WAN22] Installing Musubi-Tuner (runtime)…"
  git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner
  # bitsandbytes & other heavy libs are installed here on the pod, not in CI
  python -m pip install --no-cache-dir -e /opt/musubi-tuner
else
  echo "[WAN22] Musubi-Tuner already present."
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

# === WAN 2.2 I2V: Comfy-Org repackaged assets ===
REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Correct text encoder for WAN 2.2 I2V
dl "$REPO" "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "${MODELS_DIR}/text_encoders"

# VAE required by 14B I2V
dl "$REPO" "vae/wan_2.1_vae.safetensors" "${MODELS_DIR}/vae"

# I2V diffusers (fp16) — under split_files; then move them
TMP_SPLIT="${WORKDIR}/split_files/diffusion_models"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "${WORKDIR}"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "${WORKDIR}"
mkdir -p "${MODELS_DIR}/diffusion_models"
if [ -d "$TMP_SPLIT" ]; then
  echo "  - moving split_files -> ${MODELS_DIR}/diffusion_models"
  mv -f "$TMP_SPLIT/"*.safetensors "${MODELS_DIR}/diffusion_models/" 2>/dev/null || true
fi

# --- Accelerate config (single GPU, bf16) ---
ACC_CONF="${HOME}/.cache/huggingface/accelerate/default_config.yaml"
if [ ! -f "$ACC_CONF" ]; then
  mkdir -p "$(dirname "$ACC_CONF")"
  cat > "$ACC_CONF" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: NO
gpu_ids: '0'
mixed_precision: bf16
num_machines: 1
num_processes: 1
main_training_function: main
YAML
fi

# --- Minimal dataset config (edit as needed) ---
CONF_JSON="$WORKDIR/configs/dataset_i2v.json"
if [ ! -f "$CONF_JSON" ]; then
  cat > "$CONF_JSON" <<'JSON'
{
  "name": "wan22_i2v_char",
  "type": "i2v",
  "resolution": [1280, 720],
  "frames_per_sample": 32,
  "bucket_resolutions": [[1280,720],[1440,810],[960,544]],
  "train_data": [
    { "path": "/workspace/datasets/character_images", "caption_ext": ".txt", "shuffle": true, "repeat": 1 }
  ],
  "val_images": ["/workspace/datasets/val/portrait1.png"]
}
JSON
fi

# --- Create training scripts (High-Noise / Low-Noise) using Musubi-Tuner ---
HIGH_SH="$WORKDIR/scripts/train_i2v_high.sh"
LOW_SH="$WORKDIR/scripts/train_i2v_low.sh"

cat > "$HIGH_SH" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models
OUT=/workspace/outputs/i2v_high
CACHE=/workspace/cache/i2v_high
CONF=/workspace/configs/dataset_i2v.json
mkdir -p "$OUT" "$CACHE"

python /opt/musubi-tuner/wan_cache_latents.py \
  --model_path "$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  --vae_path "$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$CACHE" \
  --i2v

python /opt/musubi-tuner/wan_cache_text_encoder_outputs.py \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$CACHE"

accelerate launch /opt/musubi-tuner/wan_train_network.py \
  --model_path "$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  --vae_path "$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$OUT" \
  --network_module lora \
  --rank 32 \
  --learning_rate 1e-4 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 2 \
  --mixed_precision bf16 \
  --i2v \
  --cache_latents_dir "$CACHE" \
  --cache_text_encoder_outputs_dir "$CACHE" \
  --enable_xformers \
  --fp8_llm \
  --blocks_to_swap 16

echo "HN LoRA saved to $OUT"
BASH
chmod +x "$HIGH_SH"

cat > "$LOW_SH" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models
OUT=/workspace/outputs/i2v_low
CACHE=/workspace/cache/i2v_low
CONF=/workspace/configs/dataset_i2v.json
mkdir -p "$OUT" "$CACHE"

python /opt/musubi-tuner/wan_cache_latents.py \
  --model_path "$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  --vae_path "$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$CACHE" \
  --i2v

python /opt/musubi-tuner/wan_cache_text_encoder_outputs.py \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$CACHE"

accelerate launch /opt/musubi-tuner/wan_train_network.py \
  --model_path "$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  --vae_path "$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$OUT" \
  --network_module lora \
  --rank 32 \
  --learning_rate 5e-5 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 2 \
  --mixed_precision bf16 \
  --i2v \
  --cache_latents_dir "$CACHE" \
  --cache_text_encoder_outputs_dir "$CACHE" \
  --enable_xformers \
  --fp8_llm \
  --blocks_to_swap 16

echo "LN LoRA saved to $OUT"
BASH
chmod +x "$LOW_SH"

echo "[WAN22] Model & dataset bootstrap complete."
echo "[WAN22] Summary:"
echo "  - text encoders: $(ls -1 ${MODELS_DIR}/text_encoders | wc -l || true)"
echo "  - vae:           $(ls -1 ${MODELS_DIR}/vae | wc -l || true)"
echo "  - diffusers:     $(ls -1 ${MODELS_DIR}/diffusion_models | wc -l || true)"
echo "[WAN22] Training scripts:"
echo "  - bash /workspace/scripts/train_i2v_high.sh"
echo "  - bash /workspace/scripts/train_i2v_low.sh"

echo "[WAN22] Bootstrap finished (no UI)."
