#!/usr/bin/env bash
set -euo pipefail

echo "[BOOTSTRAP] WAN 2.1 T2V LoRA start $(date -Iseconds)"

WORKDIR="/workspace"
MODELS="$WORKDIR/models"
DATA="$WORKDIR/datasets"
CACHE="$WORKDIR/cache"
CONF="$WORKDIR/configs"
OUT="$WORKDIR/outputs"
SCRIPTS="$WORKDIR/scripts"
LOGS="$WORKDIR/logs"

# Create directory structure, including diffusion_models for training scripts
mkdir -p \
  "$MODELS"/{dit,vae,t5,diffusion_models} \
  "$DATA" \
  "$CACHE/t2v" \
  "$CONF" \
  "$OUT" \
  "$SCRIPTS" \
  "$LOGS"

# -----------------------------
# Install musubi-tuner + deps
# -----------------------------
pip install -U pip wheel setuptools
pip install "huggingface_hub[cli]==0.25.2"
pip install git+https://github.com/kohya-ss/musubi-tuner.git
pip install hf_transfer==0.1.8

export HF_HUB_ENABLE_HF_TRANSFER=1
HF="huggingface-cli"

# -----------------------------
# Download WAN 2.1 T2V models
# -----------------------------
echo "[MODELS] Downloading WAN 2.1 T2V models..."

# DiT (T2V 14B)
$HF download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors \
  --local-dir "$MODELS/dit" \
  --local-dir-use-symlinks=False

# VAE
$HF download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/vae/wan_2.1_vae.safetensors \
  --local-dir "$MODELS/vae" \
  --local-dir-use-symlinks=False

# T5 Encoder
$HF download Wan-AI/Wan2.1-I2V-14B-720P \
  models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir "$MODELS/t5" \
  --local-dir-use-symlinks=False

# -----------------------------
# Fix directory mismatch for DiT
# (what bit you before)
#
# Actual download:
#   /workspace/models/dit/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors
#
# Some scripts (and your working cmd) expect:
#   /workspace/models/diffusion_models/wan2.1_t2v_14B_fp16.safetensors
# -----------------------------
mkdir -p "$MODELS/diffusion_models"

DIT_SRC="$MODELS/dit/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors"
DIT_DST="$MODELS/diffusion_models/wan2.1_t2v_14B_fp16.safetensors"

if [ -f "$DIT_SRC" ]; then
  ln -sf "$DIT_SRC" "$DIT_DST"
  echo "[MODELS] Symlinked DiT to $DIT_DST"
else
  echo "[ERROR] DiT model file not found at $DIT_SRC after download." >&2
  exit 1
fi

echo "[MODELS] Done."

# -----------------------------
# Dataset config (T2V)
# -----------------------------
cat > "$CONF/dataset_t2v.toml" <<'TOML'
[general]
# WAN 2.1 T2V native-ish res
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
# Musubi expects flat folder: images + .txt captions
image_directory = "/workspace/datasets"
cache_directory = "/workspace/cache/t2v"
num_repeats = 1
TOML

# -----------------------------
# Train script (cache + train)
# -----------------------------
cat > "$SCRIPTS/train_t2v.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

CONF="/workspace/configs/dataset_t2v.toml"
M="/workspace/models"
LOG="/workspace/logs/train.log"

echo "[CACHE] Latents..."
python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --device cuda \
  --batch_size 2 \
  --num_workers 1 | tee "$LOG"

echo "[CACHE] Text Encoder outputs..."
python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/t5/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda \
  --batch_size 4 | tee -a "$LOG"

echo "[TRAIN] Launching..."
accelerate launch -m musubi_tuner.wan_train_network \
  --task t2v-14B \
  --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.1_t2v_14B_fp16.safetensors" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --t5 "$M/t5/models_t5_umt5-xxl-enc-bf16.pth" \
  --network_module networks.lora_wan \
  --network_dim 32 \
  --network_alpha 32 \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 20 \
  --save_every_n_epochs 5 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --offload_inactive_dit \
  --network_args caption_dropout=0.05 \
  --output_dir "/workspace/outputs/t2v_char" \
  --output_name "t2v_char" \
  --sdpa | tee -a "$LOG"

echo "[TRAIN] Done."
BASH

chmod +x "$SCRIPTS/train_t2v.sh"

echo "[BOOTSTRAP] Completed."
