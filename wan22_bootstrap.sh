#!/usr/bin/env bash
# wan22_bootstrap.sh — Runtime setup for WAN 2.2 I2V training (manual start)

set -euo pipefail

echo "[BOOTSTRAP] Starting WAN22 runtime setup..."

# ---------- Paths ----------
WORKDIR="${WORKDIR:-/workspace}"
MODELS_DIR="$WORKDIR/models"
DATASETS_DIR="$WORKDIR/datasets"
OUT_DIR="$WORKDIR/outputs"
CACHE_DIR="$WORKDIR/cache"

mkdir -p \
  "$MODELS_DIR/diffusion_models" \
  "$MODELS_DIR/text_encoders" \
  "$MODELS_DIR/vae" \
  "$DATASETS_DIR/character_images" \
  "$WORKDIR/scripts" \
  "$WORKDIR/configs" \
  "$OUT_DIR" \
  "$CACHE_DIR"

# ---------- HuggingFace CLI ----------
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[BOOTSTRAP] Installing huggingface_hub CLI..."
  python -m pip install --no-cache-dir "huggingface_hub[cli]==0.25.2"
fi
if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi
echo "[BOOTSTRAP] Using HF CLI: $HF"

# ---------- Helper download ----------
dl () {
  local repo="$1"; shift
  local rel="$1"; shift
  local out="$1"; shift

  mkdir -p "$out"
  local base="$(basename "$rel")"
  if [ -f "$out/$base" ]; then
    echo "  - exists: $out/$base"
  else
    echo "  - fetching $repo/$rel -> $out"
    if [ -n "${HF_TOKEN:-}" ]; then
      $HF download "$repo" "$rel" --local-dir "$out" --token "$HF_TOKEN" --resume
    else
      $HF download "$repo" "$rel" --local-dir "$out" --resume
    fi
  fi
}

REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
echo "[WAN22] Downloading WAN 2.2 components..."

# ---------- TEXT ENCODER (FP16) ----------
dl "$REPO" "split_files/text_encoders/umt5_xxl_fp16.safetensors" "$MODELS_DIR/text_encoders"
if [ -f "$MODELS_DIR/text_encoders/split_files/text_encoders/umt5_xxl_fp16.safetensors" ]; then
  mv -f "$MODELS_DIR/text_encoders/split_files/text_encoders/umt5_xxl_fp16.safetensors" \
        "$MODELS_DIR/text_encoders/"
  rm -rf "$MODELS_DIR/text_encoders/split_files" || true
fi
echo "[WAN22] FP16 text encoder ready."

# ---------- VAE ----------
dl "$REPO" "vae/wan_2.1_vae.safetensors" "$MODELS_DIR/vae"

# ---------- DIFFUSION MODELS ----------
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "$MODELS_DIR/diffusion_models"

echo "[WAN22] All model files present:"
find "$MODELS_DIR" -type f -name "*.safetensors" -printf "  %p\n" || true

# ---------- Musubi-Tuner ----------
if [ ! -d /opt/musubi-tuner ]; then
  echo "[BOOTSTRAP] Installing Musubi-Tuner..."
  git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner
  python -m pip install --no-cache-dir -e /opt/musubi-tuner
else
  echo "[BOOTSTRAP] Musubi-Tuner already installed."
fi

# ---------- Accelerate config ----------
ACC="${HOME}/.cache/huggingface/accelerate/default_config.yaml"
mkdir -p "$(dirname "$ACC")"
cat > "$ACC" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: NO
gpu_ids: '0'
mixed_precision: bf16
num_machines: 1
num_processes: 1
main_training_function: main
YAML

# ---------- Dataset config scaffold ----------
CONF="$WORKDIR/configs/dataset_i2v.json"
if [ ! -f "$CONF" ]; then
cat > "$CONF" <<'JSON'
{
  "name": "wan22_i2v_char",
  "type": "i2v",
  "resolution": [1280, 720],
  "frames_per_sample": 32,
  "bucket_resolutions": [[1280,720],[1440,810],[960,544]],
  "train_data": [
    { "path": "/workspace/datasets/character_images", "caption_ext": ".txt", "shuffle": true, "repeat": 1 }
  ],
  "val_images": []
}
JSON
fi

# ---------- Training scripts ----------
echo "[BOOTSTRAP] Creating training scripts..."

cat > "$WORKDIR/scripts/train_i2v_high.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models
OUT=/workspace/outputs/i2v_high
CACHE=/workspace/cache/i2v_high
CONF=/workspace/configs/dataset_i2v.json
mkdir -p "$OUT" "$CACHE"

/opt/conda/bin/python /opt/musubi-tuner/wan_cache_latents.py \
  --model_path "$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --vae_path "$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$CACHE" \
  --i2v

/opt/conda/bin/python /opt/musubi-tuner/wan_cache_text_encoder_outputs.py \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$CACHE"

accelerate launch /opt/musubi-tuner/wan_train_network.py \
  --model_path "$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --vae_path "$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$OUT" \
  --network_module lora \
  --rank 32 \
  --learning_rate 1e-4 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 30 \
  --mixed_precision bf16 \
  --i2v \
  --cache_latents_dir "$CACHE" \
  --cache_text_encoder_outputs_dir "$CACHE" \
  --enable_xformers
BASH
chmod +x "$WORKDIR/scripts/train_i2v_high.sh"

cat > "$WORKDIR/scripts/train_i2v_low.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models
OUT=/workspace/outputs/i2v_low
CACHE=/workspace/cache/i2v_low
CONF=/workspace/configs/dataset_i2v.json
mkdir -p "$OUT" "$CACHE"

/opt/conda/bin/python /opt/musubi-tuner/wan_cache_latents.py \
  --model_path "$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --vae_path "$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$CACHE" \
  --i2v

/opt/conda/bin/python /opt/musubi-tuner/wan_cache_text_encoder_outputs.py \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --dataset_config "$CONF" \
  --output_dir "$CACHE"

accelerate launch /opt/musubi-tuner/wan_train_network.py \
  --model_path "$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --text_encoder_path "$M/text_encoders/umt5_xxl_fp16*

mixed_precision bf16 \
  --i2v \
  --cache_latents_dir "$CACHE" \
  --cache_text_encoder_outputs_dir "$CACHE" \
  --enable_xformers
BASH
chmod +x "$WORKDIR/scripts/train_i2v_low.sh"

echo "[BOOTSTRAP] Training scripts ready."
echo "  -> bash /workspace/scripts/train_i2v_high.sh"
echo "  -> bash /workspace/scripts/train_i2v_low.sh"

# ---------- KEEP CONTAINER ALIVE ----------
echo "[BOOTSTRAP] Setup complete. Container will remain alive for SSH access."
tail -f /dev/null

