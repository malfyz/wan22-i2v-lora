#!/usr/bin/env bash
# WAN 2.2 I2V LoRA bootstrap (RunPod-friendly, concise + resilient)

set -u -o pipefail  # (not -e: we want a summary even if some steps fail)
WORKDIR="${WORKDIR:-/workspace}"
MODELS_DIR="$WORKDIR/models"
DATASETS_DIR="$WORKDIR/datasets"
OUT_DIR="$WORKDIR/outputs"
CACHE_DIR="$WORKDIR/cache"
SCRIPTS_DIR="$WORKDIR/scripts"
CONFIGS_DIR="$WORKDIR/configs"
LOGS_DIR="$WORKDIR/logs"

# Change this if you host a different zip:
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://rsvp.ninja/wan_lora.zip}"

# HF repos
REPACK_REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN21_REPO="Wan-AI/Wan2.1-I2V-14B-720P"

mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae} \
         "$DATASETS_DIR/character_images" \
         "$OUT_DIR" "$CACHE_DIR" "$SCRIPTS_DIR" "$CONFIGS_DIR" "$LOGS_DIR"

FAILED=(); SUCCEEDED=()
ok(){ SUCCEEDED+=("$1"); echo "[OK]   $1"; }
ko(){ FAILED+=("$1");   echo "[FAIL] $1"; }

retry(){ # retry <times> <sleep_base> -- <cmd...>
  local t="$1"; shift; local s="$1"; shift
  if [ "$#" -gt 0 ] && [ "${1:-}" = "--" ]; then shift; fi
  local i=1; while :; do "$@" && return 0; [ $i -ge "$t" ] && return 1
    echo "  retry $i/$t: $*"; sleep $((s*i)); i=$((i+1)); done; }

need_cli(){
  if command -v hf >/dev/null 2>&1; then HF=hf
  elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
  else
    retry 3 5 -- python -m pip install --no-cache-dir "huggingface_hub[cli]==0.25.2" || return 1
    HF=huggingface-cli
  fi
  export HF_HUB_ENABLE_HF_TRANSFER=1
}

dl_one(){ # dl_one <repo> <relpath> <destdir>
  local repo="$1" rel="$2" out="$3"; mkdir -p "$out"
  local base="$(basename "$rel")" target="$out/$base"
  [ -f "$target" ] && { echo "  - exists: $target"; return 0; }
  echo "  - fetching $repo/$rel → $out"
  if [ -n "${HF:-$HF}" ]; then
    $HF download "$repo" "$rel" --local-dir "$out" --resume ${HF_TOKEN:+--token "$HF_TOKEN"} || return 1
  else
    curl -fL --retry 5 --retry-all-errors --retry-delay 5 \
      -o "$target" "https://huggingface.co/${repo}/resolve/main/${rel}" || return 1
  fi
  [ -f "$target" ] || { mv "$(find "$out" -type f -name "$base" | head -n1)" "$target" 2>/dev/null || true; }
  [ -f "$target" ] || return 1
  echo "  - ready: $target"
}

echo "[BOOTSTRAP] start $(date -Iseconds)"

# --- system / python basics ---
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive retry 3 5 -- apt-get update -y && \
  retry 3 5 -- apt-get install -y --no-install-recommends git curl unzip rsync && ok sys_pkgs || ko sys_pkgs
else ok sys_pkgs; fi
retry 3 5 -- python -m pip install --upgrade pip wheel setuptools && ok pip || ko pip

# --- HF CLI ---
if need_cli; then ok hf_cli; else ko hf_cli; fi

# --- musubi-tuner (editable) ---
if [ ! -d /opt/musubi-tuner ]; then
  if retry 3 5 -- git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && \
     retry 3 5 -- python -m pip install --no-cache-dir -e /opt/musubi-tuner; then ok musubi; else ko musubi; fi
else
  python -m pip install --no-cache-dir -e /opt/musubi-tuner && ok musubi || ko musubi
fi

# --- Models (WAN 2.2 I2V + Wan 2.1 VAE + correct T5 encoder) ---
echo "[MODELS] downloading…"
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models" && ok i2v_high || ko i2v_high
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "$MODELS_DIR/diffusion_models" && ok i2v_low  || ko i2v_low
dl_one "$REPACK_REPO" "split_files/vae/wan_2.1_vae.safetensors"                                  "$MODELS_DIR/vae"             && ok vae21   || ko vae21
# optional: 2.2 vae (not used for I2V 14B but handy to have)
dl_one "$REPACK_REPO" "split_files/vae/wan2.2_vae.safetensors"                                    "$MODELS_DIR/vae"             && ok vae22   || ko vae22
# CORRECT T5 encoder for Musubi WAN: .pth from Wan-AI repo
dl_one "$WAN21_REPO"  "models_t5_umt5-xxl-enc-bf16.pth"                                           "$MODELS_DIR/text_encoders"   && ok t5pth   || ko t5pth

echo "[MODELS] present:"
find "$MODELS_DIR" -maxdepth 3 -type f -name "*.safetensors" -o -name "*.pth" -printf "  %p\n" || true

# --- accelerate (bf16, 1 GPU) ---
ACC="$HOME/.cache/huggingface/accelerate/default_config.yaml"; mkdir -p "$(dirname "$ACC")"
cat > "$ACC" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: NO
gpu_ids: '0'
mixed_precision: bf16
num_machines: 1
num_processes: 1
main_training_function: main
YAML
ok accelerate

# --- dataset import (clean overwrite) ---
TMP=/tmp/wan_lora.$$
UNPACK=/tmp/dsunpack
rm -rf "$UNPACK" "$DATASETS_DIR/character_images"
mkdir -p "$UNPACK" "$DATASETS_DIR/character_images"
echo "[DATASET] downloading $DATASET_ZIP_URL"
if curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$TMP" "$DATASET_ZIP_URL" && \
   (unzip -oq "$TMP" -d "$UNPACK" || tar -xof "$TMP" -C "$UNPACK"); then
  rsync -a --delete "$UNPACK"/ "$DATASETS_DIR/character_images"/ && ok dataset || ko dataset
else ko dataset; fi

# --- TOML dataset (expects real .txt captions) ---
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
sed -i 's/\r$//' "$CONF"
ok dataset_config

# --- env flags to avoid CUDA/NCCL hiccups ---
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --- training scripts (HIGH / LOW) ---
cat > "$SCRIPTS_DIR/train_i2v_high.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
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
  --task i2v-A14B --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --network_module lora --rank 32 \
  --learning_rate 1e-4 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_epochs 30 --mixed_precision bf16 --xformers | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_high.sh" && ok script_high || ko script_high

cat > "$SCRIPTS_DIR/train_i2v_low.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
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
  --task i2v-A14B --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --network_module lora --rank 32 \
  --learning_rate 5e-5 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_epochs 30 --mixed_precision bf16 --xformers | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_low.sh" && ok script_low || ko script_low

echo "[SCRIPTS] ready:"
ls -la "$SCRIPTS_DIR" | sed -n '1,120p' || true

# --- summary ---
echo
echo "================== SUMMARY =================="
echo "[TIME] $(date -Iseconds)"
[ ${#SUCCEEDED[@]} -gt 0 ] && { echo "Succeeded:"; for s in "${SUCCEEDED[@]}"; do echo "  • $s"; done; } || echo "Succeeded: (none)"
[ ${#FAILED[@]} -gt 0 ]    && { echo "FAILED (non-fatal):"; for f in "${FAILED[@]}"; do echo "  • $f"; done; } || echo "FAILED: (none)"
echo "============================================"
echo
