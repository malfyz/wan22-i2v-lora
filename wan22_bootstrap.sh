#!/usr/bin/env bash
# WAN 2.2 I2V LoRA bootstrap — final tuned version
# - Installs musubi-tuner
# - Installs triton + tries sage-attn
# - Downloads WAN2.2 i2v (high/low), WAN2.1 VAE, and correct T5 .pth
# - Imports dataset zip (non-interactive)
# - Writes dataset TOML (num_repeats=2, captions enabled)
# - Creates train scripts (network_dim/alpha=64, 50 epochs, save every 10)
# - Uses fp16, SDPA, gradient checkpointing, offload, allocator fix

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

mkdir -p \
  "$MODELS_DIR/diffusion_models" \
  "$MODELS_DIR/text_encoders" \
  "$MODELS_DIR/vae" \
  "$DATASETS_DIR/character_images" \
  "$OUT_DIR" "$CACHE_DIR" "$SCRIPTS_DIR" "$CONFIGS_DIR" "$LOGS_DIR"

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
  while :; do
    "$@" && return 0
    [ $i -ge "$t" ] && return 1
    echo "  retry $i/$t: $*"
    sleep $(( s * i ))
    i=$((i+1))
  done
}

# ---------- system basics ----------
step=sys_pkgs
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive retry 3 5 -- apt-get update -y && \
  retry 3 5 -- apt-get install -y --no-install-recommends git curl unzip rsync && ok $step || ko $step
else
  ok $step
fi

step=pip_upgrade
retry 3 5 -- python -m pip install --upgrade pip wheel setuptools >/dev/null 2>&1 && ok $step || ko $step

# ---------- HF CLI ----------
step=hf_cli
if command -v hf >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else
  if retry 3 5 -- python -m pip install --no-cache-dir "huggingface_hub[cli]==0.25.2"; then
    HF=huggingface-cli
  else
    HF=""
  fi
fi
if [ -n "${HF:-}" ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  ok $step
else
  ko $step
fi

# ---------- musubi-tuner ----------
step=musubi
if [ ! -d /opt/musubi-tuner ]; then
  if retry 3 5 -- git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && \
     retry 3 5 -- python -m pip install --no-cache-dir -e /opt/musubi-tuner; then
    ok $step
  else
    ko $step
  fi
else
  if python -m pip install --no-cache-dir -e /opt/musubi-tuner >/dev/null 2>&1; then ok $step; else ko $step; fi
fi

# ---------- optional speedups: triton + sage-attn ----------
step=triton_sage
echo "[SETUP] attempting to install triton + sage-attn (best-effort)"
TRIED_SAGE=0
if retry 2 5 -- python -m pip install --no-cache-dir "triton"; then
  echo "  - triton installed"
else
  echo "  - triton install failed (continuing)"
fi

# attempt several likely sage-attn package names (best-effort)
for pkg in "sage-attn" "sageattention" "sage-attention" "sage_attention"; do
  TRIED_SAGE=$((TRIED_SAGE+1))
  if python -m pip install --no-cache-dir "$pkg" >/dev/null 2>&1; then
    echo "  - installed $pkg"
    SAGE_PKG="$pkg"
    ok $step
    break
  fi
done
[ -n "${SAGE_PKG:-}" ] || echo "  - sage-attn install attempts failed (continuing without)"

# ---------- helper: download HF file robustly ----------
dl_one(){ # dl_one <repo> <relpath> <destdir>
  local repo="$1"; local rel="$2"; local out="$3"
  mkdir -p "$out"
  local base
  base="$(basename "$rel")"
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
    # fallback to curl (public assets only)
    local url="https://huggingface.co/${repo}/resolve/main/${rel}"
    retry 5 5 -- curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$target" "$url" || return 1
  fi

  # normalize split_files layout in case hf CLI wrote nested
  if [ ! -f "$target" ]; then
    local found
    found="$(find "$out" -type f -name "$base" 2>/dev/null | head -n1 || true)"
    if [ -n "$found" ]; then
      mv -f "$found" "$target" || true
      find "$out" -type d -empty -delete || true
    fi
  fi

  [ -f "$target" ] || return 1
  echo "  - ready: $target"
}

# ---------- models ----------
echo "[MODELS] downloading…"
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models" && ok i2v_high || ko i2v_high
dl_one "$REPACK_REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "$MODELS_DIR/diffusion_models" && ok i2v_low  || ko i2v_low
dl_one "$REPACK_REPO" "split_files/vae/wan_2.1_vae.safetensors"                                  "$MODELS_DIR/vae"             && ok vae21   || ko vae21
# (leave wan2.2_vae optional)
dl_one "$REPACK_REPO" "split_files/vae/wan2.2_vae.safetensors"                                    "$MODELS_DIR/vae"             && ok vae22   || echo "[MODELS] wan2.2 vae skipped/optional"
dl_one "$WAN21_REPO"  "models_t5_umt5-xxl-enc-bf16.pth"                                           "$MODELS_DIR/text_encoders"   && ok t5pth   || ko t5pth

echo "[MODELS] present:"
find "$MODELS_DIR" -maxdepth 3 -type f \( -name "*.safetensors" -o -name "*.pth" \) -printf "  %p\n" || true

# ---------- accelerate config (fp16) ----------
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

# ---------- dataset import (overwrite) ----------
step=dataset
TMP=/tmp/wan_lora.$$
UNPACK=/tmp/dsunpack
rm -rf "$UNPACK" "$DATASETS_DIR/character_images"
mkdir -p "$UNPACK" "$DATASETS_DIR/character_images"
echo "[DATASET] downloading $DATASET_ZIP_URL"
if retry 5 5 -- curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$TMP" "$DATASET_ZIP_URL" && \
   (unzip -oq "$TMP" -d "$UNPACK" || tar -xof "$TMP" -C "$UNPACK"); then
  rsync -a --delete "$UNPACK"/ "$DATASETS_DIR/character_images"/ && ok $step || ko $step
else
  ko $step
fi

# ---------- dataset TOML (captions expected) ----------
CONF="$CONFIGS_DIR/dataset_i2v.toml"
cat > "$CONF" <<'TOML'
[general]
resolution = [1280, 720]         # lower to [960,544] if OOM
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
caption_extension = ".txt"

[[datasets]]
image_directory = "/workspace/datasets/character_images"
cache_directory = "/workspace/cache/i2v"
num_repeats = 2
TOML
sed -i 's/\r$//' "$CONF"
ok dataset_config

# ---------- env flags to avoid CUDA/NCCL hiccups ----------
export CUDA_MODULE_LOADING=LAZY
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- training scripts ----------
# Notes:
# - network_dim/alpha = 64 (tuned)
# - epochs = 50 ; save_every_n_epochs = 10
# - caption_dropout via --network_args caption_dropout=0.05 (5%)
# - using sdpa (--sdpa) and --sage_attn switch to enable sage attention (if installed)
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

mkdir -p /workspace/outputs/i2v_high /workspace/cache/i2v

# cache latents (i2v)
python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --i2v --device cuda --batch_size 2 --num_workers 2 | tee "$LOG"

# cache T5 outputs
python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda --batch_size 4 | tee -a "$LOG"

# train LoRA (high noise)
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
  --sdpa --sage_attn | tee -a "$LOG"
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

mkdir -p /workspace/outputs/i2v_low /workspace/cache/i2v

python -u -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan_2.1_vae.safetensors" \
  --i2v --device cuda --batch_size 2 --num_workers 2 | tee "$LOG"

python -u -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --device cuda --batch_size 4 | tee -a "$LOG"

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
  --sdpa --sage_attn | tee -a "$LOG"
BASH
chmod +x "$SCRIPTS_DIR/train_i2v_low.sh" && ok script_low || ko script_low

echo "[SCRIPTS] ready:"
ls -la "$SCRIPTS_DIR" | sed -n '1,200p' || true

# ---------- summary ----------
echo
echo "================== SUMMARY =================="
echo "[TIME] $(date -Iseconds)"
if [ "${#SUCCEEDED[@]}" -gt 0 ]; then
  echo "Succeeded:"
  for s in "${SUCCEEDED[@]}"; do echo "  • $s"; done
else
  echo "Succeeded: (none)"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED (non-fatal):"
  for f in "${FAILED[@]}"; do echo "  • $f"; done
else
  echo "FAILED: (none)"
fi
echo "============================================"
echo

# exit normally (no keep-alive)
exit 0
