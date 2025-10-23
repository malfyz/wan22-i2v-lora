#!/usr/bin/env bash
# WAN 2.2 I2V LoRA bootstrap (RunPod-ready, FINAL – VAE 2.1 ONLY)
# - Installs musubi-tuner
# - Downloads WAN 2.2 I2V (high/low), VAE 2.1, and correct T5 (.pth)
# - Imports dataset zip (with .txt captions) non-interactively
# - Creates high/low training scripts (SDPA, fp16, grad checkpointing, offload)
# - Uses cache reuse (skip recache if present)
# - 40 epochs, save every 10

set -u -o pipefail
echo "[BOOTSTRAP] start $(date -Iseconds)"

# ---------- paths ----------
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

# dataset zip (override with env if needed)
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://rsvp.ninja/wan_lora.zip}"

# HF repos
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
  DEBIAN_FRONTEND=noninteractive retry 3 5 -- apt-get update -y && \
  retry 3 5 -- apt-get install -y --no-install-recommends git curl unzip rsync && ok sys_pkgs || ko sys_pkgs
else ok sys_pkgs; fi
retry 3 5 -- python -m pip install --upgrade pip wheel setuptools && ok pip || ko pip

# ---------- HF CLI ----------
if command -v hf >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else
  if retry 3 5 -- python -m pip install --no-cache-dir "huggingface_hub[cli]==0.25.2"; then
    HF=huggingface-cli
  else
    HF=""
  fi
fi
[ -n "${HF:-}" ] && export HF_HUB_ENABLE_HF_TRANSFER=1
[ -n "${HF:-}" ] && ok hf_cli || ko hf_cli

# ---------- musubi-tuner ----------
if [ ! -d /opt/musubi-tuner ]; then
  if retry 3 5 -- git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && \
     retry 3 5 -- python -m pip install --no-cache-dir -e /opt/musubi-tuner; then ok musubi; else ko musubi; fi
else
  python -m pip install --no-cache-dir -e /opt/musubi-tuner && ok musubi || ko musubi
fi

# ---------- helpers ----------
dl_one(){ # dl_one <repo> <relpath> <destdir>
  local repo="$1"; local rel="$2"; local out="$3"
  mkdir -p "$out"
  local base
  base="$(basename "$rel")"
  local target="$out/$base"
  if [ -f "$target" ]; then
    echo "  - exists: $target"; return 0
  fi
  echo "  - fetching $repo/$rel → $out"
  if [ -n "${HF:-}" ]; then
    retry 5 5 -- $HF download "$repo" "$rel" --local-dir "$out" --resume ${HF_TOKEN:+--token "$HF_TOKEN"} || return 1
  else
    retry 5 5 -- curl -fL --retry 5 --retry-all-errors --retry-delay 5 \
      -o "$target" "https://huggingface.co/${repo}/resolve/main/${rel}" || return 1
  fi
  # normalize if nested by HF
  if [ ! -f "$target" ]; then
    local found; found="$(find "$out" -type f -name "$base" | head -n1 || true)"
    [ -n "$found" ] && mv -f "$found" "$target" || true
    find "$out" -type d -empty -delete || true
  fi
  [ -f "$target" ] || return 1
  echo "  - ready: $target"
}

# ---------- models ----------
echo "[MODELS] downloading…"
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models" && ok i2v_high || ko i2v_high
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "$MODELS_DIR/diffusion_models" && ok i2v_low  || ko i2v_low
# STRICT: VAE 2.1 ONLY (2.2 is NOT compatible for our flow)
dl_one "$REPACK_REPO" "split_files/vae/wan_2.1_vae.safetensors"                                  "$MODELS_DIR/vae"             && ok vae21   || ko vae21
# Correct T5 encoder .pth (NOT .safetensors)
dl_one "$WAN21_REPO"  "models_t5_umt5-xxl-enc-bf16.pth"                                           "$MODELS_DIR/text_encoders"   && ok t5pth   || ko t5pth

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

# ---------- dataset import (overwrite, no prompts) ----------
TMP="/tmp/wan_lora.$$.zip"
UNPACK="/tmp/dsunpack"
rm -rf "$UNPACK"
mkdir -p "$UNPACK" "$DATASETS_DIR/character_images"
echo "[DATASET] downloading $DATASET_ZIP_URL"
if retry 5 5 -- curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$TMP" "$DATASET_ZIP_URL" && \
   (unzip -oq "$TMP" -d "$UNPACK" || tar -xof "$TMP" -C "$UNPACK"); then
  rsync -a --delete "$UNPACK"/ "$DATASETS_DIR/character_images"/ && ok dataset || ko dataset
else ko dataset; fi

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
cache_directory = "/workspace/cache/i2v"
num_repeats = 1
TOML
sed -i 's/\r$//' "$CONF"
ok dataset_config

# ---------- env flags ----------
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- training scripts (HIGH / LOW) ----------
cat > "$SCRIPTS_DIR/train_i2v_high.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M=/workspace/models
CONF=/workspace/configs/dataset_i2v.toml
CACHE=/workspace/cache/i2v
OUT=/workspace/outputs/i2v_high
LOG=/workspace/logs/train_high.log
mkdir -p "$OUT" "$CACHE"

# (1) cache latents (skip if present)
if ! find "$CACHE" -type f -name "*.npz" -print -quit | grep -q .; then
  python -u -m musubi_tuner.wan_cache_latents \
    --dataset_config "$CONF" \
    --vae "$M/vae/wan_2.1_vae.safetensors" \
    --i2v --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"
else
  echo "[CACHE] latents present → skip" | tee "$LOG"
fi

# (2) cache T5 outputs (skip if present)
if ! find "$CACHE" -type f \( -name "*.pt" -o -name "*t5*" -o -name "*.npy" \) -print -quit | grep -q .; then
  python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
    --dataset_config "$CONF" \
    --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
    --device cuda --batch_size 4 | tee -a "$LOG"
else
  echo "[CACHE] T5 outputs present → skip" | tee -a "$LOG"
fi

# (3) train LoRA (high-noise) — 40 epochs, save every 10
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
  --output_dir "$OUT" --output_name i2v_high \
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
CACHE=/workspace/cache/i2v
OUT=/workspace/outputs/i2v_low
LOG=/workspace/logs/train_low.log
mkdir -p "$OUT" "$CACHE"

if ! find "$CACHE" -type f -name "*.npz" -print -quit | grep -q .; then
  python -u -m musubi_tuner.wan_cache_latents \
    --dataset_config "$CONF" \
    --vae "$M/vae/wan_2.1_vae.safetensors" \
    --i2v --device cuda --batch_size 2 --num_workers 1 | tee "$LOG"
else
  echo "[CACHE] latents present → skip" | tee "$LOG"
fi

if ! find "$CACHE" -type f \( -name "*.pt" -o -name "*t5*" -o -name "*.npy" \) -print -quit | grep -q .; then
  python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
    --dataset_config "$CONF" \
    --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
    --device cuda --batch_size 4 | tee -a "$LOG"
else
  echo "[CACHE] T5 outputs present → skip" | tee -a "$LOG"
fi

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
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --offload_inactive_dit \
  --output_dir "$OUT" --output_name i2v_low \
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
