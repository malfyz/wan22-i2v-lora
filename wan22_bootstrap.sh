#!/usr/bin/env bash
# WAN 2.2 I2V LoRA bootstrap (RunPod-ready, final tuned version)
# - High noise = 40 epochs @ 1e-4 LR
# - Low noise  = 30 epochs @ 5e-5 LR
# - Save every 10 epochs
# - Correct T5 (.pth)
# - Correct VAE (WAN 2.1)
# - SDPA (stable)
# - No safety copy
# - Minimal deps, no debug output

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
mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae} \
         "$DATASETS_DIR/character_images" "$OUT_DIR" "$CACHE_DIR" \
         "$SCRIPTS_DIR" "$CONFIGS_DIR" "$LOGS_DIR"

DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://rsvp.ninja/wan_lora.zip}"

REPACK_REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN21_REPO="Wan-AI/Wan2.1-I2V-14B-720P"

FAILED=(); SUCCEEDED=()
ok(){ SUCCEEDED+=("$1"); echo "[OK]   $1"; }
ko(){ FAILED+=("$1");   echo "[FAIL] $1"; }

retry(){ local t="$1"; shift; local s="$1"; shift
  if [ "${1:-}" = "--" ]; then shift; fi
  local i=1; while :; do "$@" && return 0; [ $i -ge "$t" ] && return 1
    sleep $((s*i)); i=$((i+1)); done; }

# system packages
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive retry 3 5 -- apt-get update -y
  retry 3 5 -- apt-get install -y --no-install-recommends git curl unzip rsync
fi
ok sys_pkgs

# python tooling
retry 3 5 -- python -m pip install --upgrade pip wheel setuptools && ok pip || ko pip

# HF CLI
if command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli; else
  retry 3 5 -- python -m pip install "huggingface_hub[cli]==0.25.2" && HF=huggingface-cli || HF=""
fi
[ -n "${HF:-}" ] && export HF_HUB_ENABLE_HF_TRANSFER=1

# musubi
if [ ! -d /opt/musubi-tuner ]; then
  git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner
fi
python -m pip install --no-cache-dir -e /opt/musubi-tuner && ok musubi || ko musubi

# helper
dl_one(){ local repo="$1" rel="$2" out="$3"
  mkdir -p "$out"
  local base="$(basename "$rel")"
  local target="$out/$base"
  [ -f "$target" ] && return 0
  $HF download "$repo" "$rel" --local-dir "$out" --resume ${HF_TOKEN:+--token "$HF_TOKEN"} || return 1
  [ -f "$target" ] || mv "$(find "$out" -type f -name "$base" | head -n1)" "$target" 2>/dev/null || true
  [ -f "$target" ]
}

# models
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models"
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "$MODELS_DIR/diffusion_models"
dl_one "$REPACK_REPO" "split_files/vae/wan_2.1_vae.safetensors" "$MODELS_DIR/vae"
dl_one "$WAN21_REPO"  "models_t5_umt5-xxl-enc-bf16.pth" "$MODELS_DIR/text_encoders"
ok models

# accelerate config (fp16)
ACC="$HOME/.cache/huggingface/accelerate/default_config.yaml"
mkdir -p "$(dirname "$ACC")"
cat > "$ACC" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: NO
gpu_ids: '0'
mixed_precision: fp16
num_processes: 1
YAML

# dataset
TMP=/tmp/wan_lora.$$
UNPACK=/tmp/dsunpack
rm -rf "$UNPACK" "$DATASETS_DIR/character_images"
mkdir -p "$UNPACK" "$DATASETS_DIR/character_images"
curl -fsSL "$DATASET_ZIP_URL" -o "$TMP"
(unzip -oq "$TMP" -d "$UNPACK" || tar -xof "$TMP" -C "$UNPACK")
rsync -a --delete "$UNPACK"/ "$DATASETS_DIR/character_images"/
ok dataset

# dataset toml
CONF="$CONFIGS_DIR/dataset_i2v.toml"
cat > "$CONF" <<'TOML'
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

export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HIGH script
cat > "$SCRIPTS_DIR/train_i2v_high.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models
CONF=/workspace/configs/dataset_i2v.toml
LOG=/workspace/logs/train_high.log

python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --i2v --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"

python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
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
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --offload_inactive_dit \
  --output_dir /workspace/outputs/i2v_high --output_name i2v_high \
  --sdpa | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_high.sh"

# LOW script
cat > "$SCRIPTS_DIR/train_i2v_low.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models
CONF=/workspace/configs/dataset_i2v.toml
LOG=/workspace/logs/train_low.log

python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --i2v --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"

python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda \
  --batch_size 4 | tee -a "$LOG"

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
  --max_train_epochs 30 \
  --save_every_n_epochs 10 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --offload_inactive_dit \
  --output_dir /workspace/outputs/i2v_low --output_name i2v_low \
  --sdpa | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_low.sh"

echo "[DONE] bootstrap complete"
