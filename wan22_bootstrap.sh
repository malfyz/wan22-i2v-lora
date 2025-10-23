#!/usr/bin/env bash
# WAN 2.2 I2V LoRA bootstrap (RunPod-ready, FINAL fixed)
# 40 epochs, save every 10, keep last 3
# captions enabled, correct T5 .pth, no broken dl_one()

set -u -o pipefail
echo "[BOOTSTRAP] start $(date -Iseconds)"

WORKDIR="${WORKDIR:-/workspace}"
MODELS_DIR="$WORKDIR/models"
DATASETS_DIR="$WORKDIR/datasets"
OUT_DIR="$WORKDIR/outputs"
CACHE_DIR="$WORKDIR/cache"
SCRIPTS_DIR="$WORKDIR/scripts"
CONFIGS_DIR="$WORKDIR/configs"
LOGS_DIR="$WORKDIR/logs"

mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/vae" "$MODELS_DIR/text_encoders" \
         "$DATASETS_DIR/character_images" "$OUT_DIR" "$CACHE_DIR" \
         "$SCRIPTS_DIR" "$CONFIGS_DIR" "$LOGS_DIR"

# dataset zip
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://rsvp.ninja/wan_lora.zip}"

# model URLs (direct curl, no HF CLI bullshit)
WAN22_HIGH="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
WAN22_LOW="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
WAN21_VAE="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
T5_PTH="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"

dl_if_missing() {
  local url="$1" dest="$2"
  if [ ! -f "$dest" ]; then
    echo "  downloading $(basename "$dest")"
    curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$dest" "$url"
  else
    echo "  exists: $dest"
  fi
}

# models
echo "[MODELS] downloading..."
dl_if_missing "$WAN22_HIGH" "$MODELS_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
dl_if_missing "$WAN22_LOW"  "$MODELS_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
dl_if_missing "$WAN21_VAE"  "$MODELS_DIR/vae/wan_2.1_vae.safetensors"
dl_if_missing "$T5_PTH"     "$MODELS_DIR/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"

# system deps
apt-get update -y && apt-get install -y --no-install-recommends git curl unzip rsync

# musubi
if [ ! -d /opt/musubi-tuner ]; then
  git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner
fi
python -m pip install --upgrade pip wheel setuptools
python -m pip install --no-cache-dir -e /opt/musubi-tuner

# ACCELERATE CONFIG (fp16)
mkdir -p "$HOME/.cache/huggingface/accelerate"
cat > "$HOME/.cache/huggingface/accelerate/default_config.yaml" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: NO
gpu_ids: '0'
mixed_precision: fp16
num_machines: 1
num_processes: 1
main_training_function: main
YAML

# dataset unpack
rm -rf /tmp/dsunpack "$DATASETS_DIR/character_images"
mkdir -p /tmp/dsunpack "$DATASETS_DIR/character_images"
curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o /tmp/ds.zip "$DATASET_ZIP_URL"
unzip -oq /tmp/ds.zip -d /tmp/dsunpack
rsync -a --delete /tmp/dsunpack/ "$DATASETS_DIR/character_images"/

# config TOML
cat > "$CONFIGS_DIR/dataset_i2v.toml" <<'TOML'
[general]
resolution = [1280, 720]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
caption_extension = ".txt"

[[datasets]]
image_directory = "/workspace/datasets/character_images"
cache_directory = "/workspace/cache/i2v"
num_repeats = 1
TOML

# TRAIN SCRIPTS
cat > "$SCRIPTS_DIR/train_i2v_high.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/workspace/models
CONF=/workspace/configs/dataset_i2v.toml
LOG=/workspace/logs/train_high.log

python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" --vae "$M/vae/wan_2.1_vae.safetensors" \
  --i2v --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"

python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda --batch_size 4 | tee -a "$LOG"

accelerate launch -m musubi_tuner.wan_train_network \
  --task i2v-A14B \
  --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --network_module networks.lora_wan \
  --network_dim 32 --network_alpha 32 \
  --learning_rate 1e-4 \
  --optimizer_type AdamW --optimizer_args weight_decay=0.01 betas=0.9,0.95 eps=1e-8 \
  --max_grad_norm 1.0 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 40 \
  --save_every_n_epochs 10 \
  --save_last_n_epochs 3 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --offload_inactive_dit \
  --output_dir /workspace/outputs/i2v_high --output_name i2v_high \
  --sdpa | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_high.sh"

cat > "$SCRIPTS_DIR/train_i2v_low.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/workspace/models
CONF=/workspace/configs/dataset_i2v.toml
LOG=/workspace/logs/train_low.log

python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" --vae "$M/vae/wan_2.1_vae.safetensors" \
  --i2v --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"

python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda --batch_size 4 | tee -a "$LOG"

accelerate launch -m musubi_tuner.wan_train_network \
  --task i2v-A14B \
  --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --network_module networks.lora_wan \
  --network_dim 32 --network_alpha 32 \
  --learning_rate 5e-5 \
  --optimizer_type AdamW --optimizer_args weight_decay=0.01 betas=0.9,0.95 eps=1e-8 \
  --max_grad_norm 1.0 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 40 \
  --save_every_n_epochs 10 \
  --save_last_n_epochs 3 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --offload_inactive_dit \
  --output_dir /workspace/outputs/i2v_low --output_name i2v_low \
  --sdpa | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_low.sh"

echo "[DONE] bootstrap ready. run:"
echo "bash /workspace/scripts/train_i2v_high.sh"

