#!/usr/bin/env bash
# WAN 2.2 I2V LoRA bootstrap (RunPod-ready)
# - Installs musubi-tuner + hf_transfer (fast HF pulls)
# - Downloads WAN 2.2 I2V high/low, WAN 2.1 VAE, and correct T5 (.pth)
# - Imports dataset zip (expects .txt captions) without nested folder mess
# - Writes HIGH/LOW training scripts: SDPA, fp16, LoRA rank/alpha=64, 50 epochs, save every 10
# - No background keep-alive; exits cleanly

set -euo pipefail
echo "[BOOTSTRAP] start $(date -Iseconds)"

# ---------- layout ----------
WORKDIR="${WORKDIR:-/workspace}"
MODELS_DIR="$WORKDIR/models"
DATASETS_DIR="$WORKDIR/datasets"
OUT_DIR="$WORKDIR/outputs"
CACHE_DIR="$WORKDIR/cache"
SCRIPTS_DIR="$WORKDIR/scripts"
CONFIGS_DIR="$WORKDIR/configs"
LOGS_DIR="$WORKDIR/logs"

mkdir -p \
  "$MODELS_DIR"/{diffusion_models,text_encoders,vae} \
  "$DATASETS_DIR/character_images" \
  "$OUT_DIR" "$CACHE_DIR" "$SCRIPTS_DIR" "$CONFIGS_DIR" "$LOGS_DIR"

# dataset zip (override with env if needed)
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://rsvp.ninja/wan_lora.zip}"

# HF model repos
REPACK_REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN21_REPO="Wan-AI/Wan2.1-I2V-14B-720P"

FAILED=(); SUCCEEDED=()
ok(){ SUCCEEDED+=("$1"); echo "[OK]   $1"; }
ko(){ FAILED+=("$1");   echo "[FAIL] $1"; }

retry(){ # retry <times> <sleep_base> -- <cmd...>
  local t="$1"; shift; local s="$1"; shift
  if [ "$#" -gt 0 ] && [ "${1:-}" = "--" ]; then shift; fi
  local i=1
  while :; do "$@" && return 0; [ $i -ge "$t" ] && return 1
    echo "  retry $i/$t: $*"; sleep $((s*i)); i=$((i+1)); done; }

# ---------- system basics ----------
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  retry 3 5 -- apt-get update -y
  retry 3 5 -- apt-get install -y --no-install-recommends git curl unzip rsync
  ok sys_pkgs || ko sys_pkgs
else
  ok sys_pkgs
fi

retry 3 5 -- python -m pip install -U pip wheel setuptools
ok pip || ko pip

# ---------- HF CLI + fast transfer ----------
if command -v hf >/dev/null 2>&1; then
  HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=huggingface-cli
else
  retry 3 5 -- python -m pip install --no-cache-dir "huggingface_hub[cli]==0.25.2"
  HF=huggingface-cli
fi
# Fast transport
retry 3 5 -- python -m pip install --no-cache-dir hf_transfer==0.1.8
export HF_HUB_ENABLE_HF_TRANSFER=1
ok hf_cli || ko hf_cli

# ---------- musubi-tuner ----------
if [ ! -d /opt/musubi-tuner ]; then
  retry 3 5 -- git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner
fi
retry 3 5 -- python -m pip install --no-cache-dir -e /opt/musubi-tuner
ok musubi || ko musubi

# ---------- models (single-pass, no duplicates) ----------
dl_one(){ # dl_one <repo> <relpath> <destdir>
  local repo="$1" rel="$2" out="$3"
  mkdir -p "$out"
  local base="$(basename "$rel")"
  local target="$out/$base"
  if [ -f "$target" ]; then
    echo "  - exists: $target"
    return 0
  fi
  echo "  - fetching $repo/$rel → $out"
  # huggingface-cli handles subfolders and resumes
  $HF download "$repo" "$rel" \
      --local-dir "$out" \
      --local-dir-use-symlinks False \
      --resume ${HF_TOKEN:+--token "$HF_TOKEN"}
  # ensure file is in place
  if [ ! -f "$target" ]; then
    local found="$(find "$out" -type f -name "$base" | head -n1 || true)"
    [ -n "$found" ] && mv -f "$found" "$target"
  fi
  [ -f "$target" ] || { echo "  ! missing after download: $target"; return 1; }
  echo "  - ready: $target"
}

echo "[MODELS] downloading…"
# DiT (2.2 high/low)
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models" && ok i2v_high || ko i2v_high
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "$MODELS_DIR/diffusion_models" && ok i2v_low  || ko i2v_low
# VAE (2.1 ONLY — works with musubi I2V)
dl_one "$REPACK_REPO" "split_files/vae/wan_2.1_vae.safetensors"                                  "$MODELS_DIR/vae"             && ok vae21   || ko vae21
# Correct T5 encoder .pth
dl_one "$WAN21_REPO"  "models_t5_umt5-xxl-enc-bf16.pth"                                         "$MODELS_DIR/text_encoders"   && ok t5pth   || ko t5pth

echo "[MODELS] present:"
find "$MODELS_DIR" -maxdepth 3 -type f \( -name "*.safetensors" -o -name "*.pth" \) -printf "  %p\n" || true

# ---------- accelerate (fp16) ----------
ACC="$HOME/.cache/huggingface/accelerate/default_config.yaml"
mkdir -p "$(dirname "$ACC")"
cat > "$ACC" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: NO
gpu_ids: '0'
mixed_precision: fp16
num_machines: 1
num_processes: 1
main_training_function: main
YAML
ok accelerate

# ---------- dataset import (clean, non-nested) ----------
TMP=/tmp/wan_lora.$$
UNPACK=/tmp/dsunpack
rm -rf "$UNPACK"
mkdir -p "$UNPACK"

echo "[DATASET] downloading $DATASET_ZIP_URL"
if curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$TMP" "$DATASET_ZIP_URL"; then
  (unzip -oq "$TMP" -d "$UNPACK" || tar -xof "$TMP" -C "$UNPACK") || true
  # find the real content root (handles zips with a single top-level dir)
  SRC="$UNPACK"
  if [ "$(find "$UNPACK" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ] && \
     [ "$(find "$UNPACK" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 0 ]; then
    SRC="$(find "$UNPACK" -mindepth 1 -maxdepth 1 -type d | head -n1)"
  fi
  rm -rf "$DATASETS_DIR/character_images"
  mkdir -p "$DATASETS_DIR/character_images"
  rsync -a --delete "$SRC"/ "$DATASETS_DIR/character_images"/
  ok dataset || ko dataset
else
  ko dataset
fi

# ---------- TOML config (expects .txt captions) ----------
CONF="$CONFIGS_DIR/dataset_i2v.toml"
cat > "$CONF" <<'TOML'
[general]
resolution = [1280, 720]         # lower to [960, 544] if OOM
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
caption_extension = ".txt"

[[datasets]]
image_directory = "/workspace/datasets/character_images"
cache_directory  = "/workspace/cache/i2v"
num_repeats = 2
TOML
sed -i 's/\r$//' "$CONF"
ok dataset_config

# ---------- env flags ----------
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- training scripts (HIGH / LOW) ----------
# SDPA only (no --sage_attn), LoRA rank/alpha=64, 50 epochs, save every 10, caption_dropout=0.05

cat > "$SCRIPTS_DIR/train_i2v_high.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M=/workspace/models
CONF=/workspace/configs/dataset_i2v.toml
LOG=/workspace/logs/train_high.log

# 1) cache latents
python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --i2v --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"

# 2) cache T5 outputs
python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda --batch_size 4 | tee -a "$LOG"

# 3) train LoRA (high-noise)
accelerate launch -m musubi_tuner.wan_train_network \
  --task i2v-A14B \
  --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --network_module networks.lora_wan \
  --network_dim 64 --network_alpha 64 \
  --learning_rate 1e-4 \
  --optimizer_type AdamW --optimizer_args weight_decay=0.01 betas=0.9,0.95 eps=1e-8 \
  --max_grad_norm 1.0 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 50 \
  --save_every_n_epochs 10 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --offload_inactive_dit \
  --network_args caption_dropout=0.05 \
  --output_dir /workspace/outputs/i2v_high --output_name i2v_high \
  --sdpa | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_high.sh" && ok script_high || ko script_high

cat > "$SCRIPTS_DIR/train_i2v_low.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M=/workspace/models
CONF=/workspace/configs/dataset_i2v.toml
LOG=/workspace/logs/train_low.log

# 1) cache latents
python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --i2v --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"

# 2) cache T5 outputs
python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda --batch_size 4 | tee -a "$LOG"

# 3) train LoRA (low-noise)
accelerate launch -m musubi_tuner.wan_train_network \
  --task i2v-A14B \
  --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --network_module networks.lora_wan \
  --network_dim 64 --network_alpha 64 \
  --learning_rate 5e-5 \
  --optimizer_type AdamW --optimizer_args weight_decay=0.01 betas=0.9,0.95 eps=1e-8 \
  --max_grad_norm 1.0 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 50 \
  --save_every_n_epochs 10 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --offload_inactive_dit \
  --network_args caption_dropout=0.05 \
  --output_dir /workspace/outputs/i2v_low --output_name i2v_low \
  --sdpa | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_low.sh" && ok script_low || ko script_low

echo "[SCRIPTS] ready:"
ls -la "$SCRIPTS_DIR" | sed -n '1,120p' || true

# ---------- summary ----------
echo
echo "================== SUMMARY =================="
echo "[TIME] $(date -Iseconds)"
[ ${#SUCCEEDED[@]} -gt 0 ] && { echo "Succeeded:"; for s in "${SUCCEEDED[@]}"; do echo "  • $s"; done; } || echo "Succeeded: (none)"
[ ${#FAILED[@]} -gt 0 ]    && { echo "FAILED (non-fatal):"; for f in "${FAILED[@]}"; do echo "  • $f"; done; } || echo "FAILED: (none)"
echo "============================================"
