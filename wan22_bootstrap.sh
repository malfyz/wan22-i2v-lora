#!/usr/bin/env bash
# WAN 2.2 I2V LoRA bootstrap (RunPod-friendly)
# - Installs musubi-tuner
# - Downloads WAN 2.2 I2V (high/low), VAE, and UMT5 XXL
# - Imports dataset from URL (zip)
# - Creates simple training scripts (high / low noise)
# - Uses cautious env defaults to avoid CUDA/NCCL init issues

set -u -o pipefail

# ---------- Tunables / overridable via env ----------
WORKDIR="${WORKDIR:-/workspace}"
MODELS_DIR="$WORKDIR/models"
DATASETS_DIR="$WORKDIR/datasets"
OUT_DIR="$WORKDIR/outputs"
CACHE_DIR="$WORKDIR/cache"
SCRIPTS_DIR="$WORKDIR/scripts"
CONFIGS_DIR="$WORKDIR/configs"
LOGS_DIR="$WORKDIR/logs"

# Dataset source (override if you like)
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://rsvp.ninja/wan_lora.zip}"

# HF repo containing WAN 2.2 repackaged weights
WAN_REPO="${WAN_REPO:-Comfy-Org/Wan_2.2_ComfyUI_Repackaged}"

# Optional: set KEEP_ALIVE=1 to keep container alive at end
KEEP_ALIVE="${KEEP_ALIVE:-0}"

mkdir -p \
  "$MODELS_DIR/diffusion_models" \
  "$MODELS_DIR/text_encoders" \
  "$MODELS_DIR/vae" \
  "$DATASETS_DIR/character_images" \
  "$OUT_DIR" "$CACHE_DIR" "$SCRIPTS_DIR" "$CONFIGS_DIR" "$LOGS_DIR"

echo "[BOOTSTRAP] start $(date -Iseconds)"

FAILED=() ; SUCCEEDED=()
note_fail(){ FAILED+=("$1"); echo "[FAIL] $1"; }
note_ok(){   SUCCEEDED+=("$1"); echo "[OK]   $1"; }

retry () {
  # retry <times> <sleep_base_sec> -- <cmd...>
  local times="$1"; shift
  local base="$1"; shift
  shift $(( $# > 0 && $1 == "--" ? 1 : 0 ))
  local i=1
  while true; do
    "$@" && return 0
    if [ $i -ge "$times" ]; then return 1; fi
    echo "  retry $i/$times: $*"
    sleep $(( base * i ))
    i=$((i+1))
  done
}

# ---------- system packages we rely on ----------
step="sys_pkgs"
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive retry 3 5 -- apt-get update -y && \
  retry 3 5 -- apt-get install -y --no-install-recommends \
    git curl unzip rsync && note_ok "$step" || note_fail "$step"
else
  echo "[WARN] apt-get not present; assuming git/curl/unzip/rsync available"
  note_ok "$step"
fi

# ---------- Python basics ----------
step="pip_upgrade"
retry 3 5 -- python -m pip install --upgrade pip wheel setuptools && note_ok "$step" || note_fail "$step"

# ---------- Hugging Face CLI (for robust downloads) ----------
step="hf_cli"
if command -v hf >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else
  if retry 3 5 -- python -m pip install "huggingface_hub[cli]==0.25.2"; then
    HF=huggingface-cli
  else
    HF=""
  fi
fi
[ -n "${HF:-}" ] && note_ok "$step" || note_fail "$step"

# Try high-speed transfer if available
python - <<'PY' 2>/dev/null || true
import pkgutil, os, sys
if not pkgutil.find_loader('hf_transfer'):
    os.system("python -m pip install -q hf_transfer")
print("OK")
PY
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---------- musubi-tuner (editable) ----------
step="musubi_install"
if [ ! -d /opt/musubi-tuner ]; then
  if retry 3 5 -- git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && \
     retry 3 5 -- python -m pip install -e /opt/musubi-tuner; then
    note_ok "$step"
  else
    note_fail "$step"
  fi
else
  echo "[SETUP] musubi-tuner already present; ensuring install…"
  if python -m pip install -e /opt/musubi-tuner; then note_ok "$step"; else note_fail "$step"; fi
fi

# ---------- helper: download one HF file (resume, token optional) ----------
dl_one () {
  # dl_one <repo> <repo_rel_path> <dest_dir>
  local repo="$1"; shift
  local rel="$1"; shift
  local out="$1"; shift
  mkdir -p "$out"
  local base="$(basename "$rel")"
  local target="$out/$base"

  if [ -f "$target" ]; then
    echo "  - exists: $target"
    return 0
  fi
  echo "  - fetching $repo/$rel → $out"
  if [ -n "${HF:-}" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
      retry 5 5 -- $HF download "$repo" "$rel" --local-dir "$out" --token "$HF_TOKEN" --resume || return 1
    else
      retry 5 5 -- $HF download "$repo" "$rel" --local-dir "$out" --resume || return 1
    fi
  else
    # Fallback: plain HTTP for public repos
    local url="https://huggingface.co/${repo}/resolve/main/${rel}"
    retry 5 5 -- curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$target" "$url" || return 1
  fi

  # normalize split_files nesting if needed
  if [ ! -f "$target" ]; then
    local found
    found="$(find "$out" -type f -name "$base" | head -n1 || true)"
    if [ -n "$found" ]; then
      mv -f "$found" "$target" || return 1
      find "$out" -type d -empty -delete || true
    fi
  fi

  [ -f "$target" ] || return 1
  echo "  - ready: $target"
}

# ---------- Models: WAN 2.2 I2V (high/low), VAE, UMT5 ----------
echo "[MODELS] downloading (WAN 2.2 repack)"
step="model_umt5"
dl_one "$WAN_REPO" "split_files/text_encoders/umt5_xxl_fp16.safetensors" "$MODELS_DIR/text_encoders" && note_ok "$step" || note_fail "$step"

step="model_vae_2p2"
dl_one "$WAN_REPO" "split_files/vae/wan2.2_vae.safetensors" "$MODELS_DIR/vae" && note_ok "$step" || note_fail "$step"

step="model_i2v_high"
dl_one "$WAN_REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models" && note_ok "$step" || note_fail "$step"

step="model_i2v_low"
dl_one "$WAN_REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models" && note_ok "$step" || note_fail "$step"

echo "[MODELS] present:"
find "$MODELS_DIR" -maxdepth 3 -type f -name "*.safetensors" -printf "  %p\n" || true

# ---------- Accelerate config (bf16, 1 GPU) ----------
step="accelerate_config"
ACC="$HOME/.cache/huggingface/accelerate/default_config.yaml"
mkdir -p "$(dirname "$ACC")"
cat > "$ACC" <<'YAML' && note_ok "$step" || note_fail "$step"
compute_environment: LOCAL_MACHINE
distributed_type: NO
gpu_ids: '0'
mixed_precision: bf16
num_machines: 1
num_processes: 1
main_training_function: main
YAML

# ---------- Dataset import ----------
step="dataset_import"
tmp="/tmp/ds.$$"
mkdir -p "$DATASETS_DIR/character_images"
echo "[DATASET] downloading $DATASET_ZIP_URL"
if retry 5 5 -- curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$tmp" "$DATASET_ZIP_URL"; then
  mkdir -p /tmp/dsunpack
  if unzip -q "$tmp" -d /tmp/dsunpack 2>/dev/null || tar -xf "$tmp" -C /tmp/dsunpack 2>/dev/null; then
    rsync -a --ignore-existing /tmp/dsunpack/ "$DATASETS_DIR/character_images/" && note_ok "$step" || note_fail "$step"
  else
    echo "[DATASET] unknown archive format; placing raw file into dataset dir"
    mv -f "$tmp" "$DATASETS_DIR/character_images/wan_lora.zip" && note_ok "$step" || note_fail "$step"
  fi
else
  note_fail "$step"
fi

# ---------- Dataset config (I2V) ----------
step="dataset_config"
CONF="$CONFIGS_DIR/dataset_i2v.json"
if [ ! -f "$CONF" ]; then
  cat > "$CONF" <<'JSON' && note_ok "$step" || note_fail "$step"
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
else
  note_ok "$step"
fi

# ---------- Environment flags to avoid CUDA/NCCL hiccups ----------
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ---------- Training scripts (simple, one per noise level) ----------
step="script_high"
cat > "$SCRIPTS_DIR/train_i2v_high.sh" <<'BASH' && chmod +x "$SCRIPTS_DIR/train_i2v_high.sh" && note_ok "$step" || note_fail "$step"
#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

M=/workspace/models
CONF=/workspace/configs/dataset_i2v.json
LOG=/workspace/logs/train_high.log

# (1) Cache latents (GPU)
python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan2.2_vae.safetensors" \
  --i2v \
  --device cuda \
  --batch_size 2 --num_workers 0 | tee "$LOG"

# (2) Cache UMT5 text enc outputs
python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --device cuda \
  --batch_size 4 | tee -a "$LOG"

# (3) Train LoRA (HIGH noise)
accelerate launch -m musubi_tuner.wan_train_network \
  --task i2v-A14B \
  --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --vae "$M/vae/wan2.2_vae.safetensors" \
  --t5 "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --network_module lora \
  --rank 32 \
  --learning_rate 1e-4 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 30 \
  --mixed_precision bf16 \
  --xformers | tee -a "$LOG"
BASH

step="script_low"
cat > "$SCRIPTS_DIR/train_i2v_low.sh" <<'BASH' && chmod +x "$SCRIPTS_DIR/train_i2v_low.sh" && note_ok "$step" || note_fail "$step"
#!/usr/bin/env bash
set -euo pipefail
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

M=/workspace/models
CONF=/workspace/configs/dataset_i2v.json
LOG=/workspace/logs/train_low.log

python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan2.2_vae.safetensors" \
  --i2v \
  --device cuda \
  --batch_size 2 --num_workers 0 | tee "$LOG"

python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --device cuda \
  --batch_size 4 | tee -a "$LOG"

accelerate launch -m musubi_tuner.wan_train_network \
  --task i2v-A14B \
  --dataset_config "$CONF" \
  --dit "$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --vae "$M/vae/wan2.2_vae.safetensors" \
  --t5 "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --network_module lora \
  --rank 32 \
  --learning_rate 5e-5 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 30 \
  --mixed_precision bf16 \
  --xformers | tee -a "$LOG"
BASH

echo "[SCRIPTS] ready:"
ls -la "$SCRIPTS_DIR" | sed -n '1,120p' || true

# ---------- Summary ----------
echo
echo "================== SUMMARY =================="
echo "[TIME] $(date -Iseconds)"
if [ "${#SUCCEEDED[@]}" -gt 0 ]; then
  echo "Succeeded:"; for s in "${SUCCEEDED[@]}"; do echo "  • $s"; done
else
  echo "Succeeded: (none)"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED (non-fatal):"; for f in "${FAILED[@]}"; do echo "  • $f"; done
else
  echo "FAILED: (none)"
fi
echo "============================================"
echo

# ---------- Keep alive optionally ----------
if [ "$KEEP_ALIVE" = "1" ]; then
  echo "[BOOTSTRAP] complete — keeping container alive."
  tail -f /dev/null
fi
