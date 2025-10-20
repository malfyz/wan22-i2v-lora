#!/usr/bin/env bash
# WAN 2.2 bootstrap — resilient version
# - Installs musubi-tuner (retries)
# - Downloads models (retries + resume; normalizes split_files/)
# - Creates training scripts (BF16 + xformers; 30 epochs)
# - Optional dataset import via URL
# - Continues on errors; summarizes failures at end; keeps pod alive

set -u -o pipefail  # NOT using -e so we can continue on individual step failures

WORKDIR="${WORKDIR:-/workspace}"
MODELS_DIR="$WORKDIR/models"
DATASETS_DIR="$WORKDIR/datasets"
OUT_DIR="$WORKDIR/outputs"
CACHE_DIR="$WORKDIR/cache"
SCRIPTS_DIR="$WORKDIR/scripts"
CONFIGS_DIR="$WORKDIR/configs"

mkdir -p \
  "$MODELS_DIR/diffusion_models" \
  "$MODELS_DIR/text_encoders" \
  "$MODELS_DIR/vae" \
  "$DATASETS_DIR/character_images" \
  "$OUT_DIR" "$CACHE_DIR" "$SCRIPTS_DIR" "$CONFIGS_DIR"

echo "[BOOTSTRAP] start $(date -Iseconds)"

FAILED=()    # collect step names that fail
SUCCEEDED=() # collect successes

note_fail () { FAILED+=("$1"); echo "[FAIL] $1"; }
note_ok   () { SUCCEEDED+=("$1"); echo "[OK]   $1"; }

# ---------- helpers ----------
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

need_cli () {
  if command -v hf >/dev/null 2>&1; then HF=hf
  elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
  else
    echo "[SETUP] installing huggingface_hub CLI…"
    if retry 3 5 -- python -m pip install --no-cache-dir "huggingface_hub[cli]==0.25.2"; then
      HF=huggingface-cli
    else
      HF=""
    fi
  fi
  echo "[SETUP] HF CLI: ${HF:-MISSING}"
}

# download single file; normalize split_files layout; resume; optional $HF_TOKEN
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
    # very last resort: direct HTTP (no auth); only works for public assets
    local url="https://huggingface.co/${repo}/resolve/main/${rel}"
    retry 5 5 -- curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$target" "$url" || return 1
  fi

  # normalize split_files if HF placed it nested
  if [ ! -f "$target" ]; then
    local found
    found="$(find "$out" -type f -name "$base" | head -n1 || true)"
    if [ -n "$found" ]; then
      mv -f "$found" "$target" || return 1
      # cleanup empties
      find "$out" -type d -empty -delete || true
    fi
  fi

  [ -f "$target" ] || return 1
  echo "  - ready: $target"
}

# ---------- 0) HF + speedup ----------
step="hf_cli"
need_cli
if [ -z "${HF:-}" ]; then note_fail "$step"; else note_ok "$step"; fi

# optional transfer speedup
if python -c "import pkgutil; import sys; sys.exit(0 if pkgutil.find_loader('hf_transfer') else 1)" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
else
  python -m pip install --no-cache-dir -q hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1 || true
fi

# ---------- 1) musubi-tuner ----------
step="musubi_install"
if [ ! -d /opt/musubi-tuner ]; then
  echo "[SETUP] Installing musubi-tuner…"
  if retry 3 5 -- git clone https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && \
     retry 3 5 -- python -m pip install --no-cache-dir -e /opt/musubi-tuner; then
    note_ok "$step"
  else
    note_fail "$step"
  fi
else
  echo "[SETUP] musubi-tuner already present."
  note_ok "$step"
fi

# ---------- 2) models (WAN 2.2 repack) ----------
REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

step="model_text_encoder"
dl_one "$REPO" "split_files/text_encoders/umt5_xxl_fp16.safetensors" "$MODELS_DIR/text_encoders" && note_ok "$step" || note_fail "$step"

step="model_vae_2p2"
dl_one "$REPO" "split_files/vae/wan2.2_vae.safetensors" "$MODELS_DIR/vae" && note_ok "$step" || note_fail "$step"

step="model_i2v_high"
dl_one "$REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models" && note_ok "$step" || note_fail "$step"

step="model_i2v_low"
dl_one "$REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" "$MODELS_DIR/diffusion_models" && note_ok "$step" || note_fail "$step"

echo "[MODELS] present:"
find "$MODELS_DIR" -maxdepth 3 -type f -name "*.safetensors" -printf "  %p\n" || true

# ---------- 3) accelerate config (bf16, 1 GPU) ----------
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

# ---------- 4) dataset import (optional; non-fatal) ----------
# Provide ONE of these env vars to auto-import:
#   DATASET_ZIP_URL : http(s) URL to a zip/tar(.gz) with images+captions
#   DATASET_GIT_URL : git repo with images+captions
# It will land in $DATASETS_DIR/character_images
step="dataset_import"
import_ok=0
if [ -n "${DATASET_ZIP_URL:-}" ]; then
  echo "[DATASET] downloading archive from DATASET_ZIP_URL → $DATASETS_DIR/character_images"
  mkdir -p "$DATASETS_DIR/character_images"
  tmp="/tmp/ds.$$"
  if retry 5 5 -- curl -fL --retry 5 --retry-all-errors --retry-delay 5 -o "$tmp" "$DATASET_ZIP_URL"; then
    mkdir -p /tmp/dsunpack && \
    (unzip -q "$tmp" -d /tmp/dsunpack || tar -xf "$tmp" -C /tmp/dsunpack || true) && \
    rsync -a --ignore-existing /tmp/dsunpack/ "$DATASETS_DIR/character_images/" && import_ok=1
  fi
elif [ -n "${DATASET_GIT_URL:-}" ]; then
  echo "[DATASET] cloning DATASET_GIT_URL → $DATASETS_DIR/character_images"
  if retry 3 5 -- git clone --depth 1 "$DATASET_GIT_URL" /tmp/dsgit; then
    mkdir -p "$DATASETS_DIR/character_images"
    rsync -a --ignore-existing /tmp/dsgit/ "$DATASETS_DIR/character_images/" && import_ok=1
  fi
else
  echo "[DATASET] no DATASET_ZIP_URL / DATASET_GIT_URL provided; skipping (not fatal)."
  import_ok=1  # skip isn’t a failure
fi
if [ "$import_ok" -eq 1 ]; then note_ok "$step"; else note_fail "$step"; fi

# ---------- 5) dataset config scaffold ----------
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

# ---------- 6) training scripts (Musubi CLI; BF16 + xformers; 30 epochs) ----------
step="scripts_high"
cat > "$SCRIPTS_DIR/train_i2v_high.sh" <<'BASH' && chmod +x "$SCRIPTS_DIR/train_i2v_high.sh" && note_ok "$step" || note_fail "$step"
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models
OUT=/workspace/outputs/i2v_high
CACHE=/workspace/cache/i2v_high
CONF=/workspace/configs/dataset_i2v.json
mkdir -p "$OUT" "$CACHE"

# (1) Cache latents for i2v (needs VAE only)
python -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan2.2_vae.safetensors" \
  --i2v \
  --batch_size 2 --num_workers 4

# (2) Cache text encoder outputs (T5)
python -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --batch_size 4

# (3) Train LoRA (high noise DiT)
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
  --xformers
BASH

step="scripts_low"
cat > "$SCRIPTS_DIR/train_i2v_low.sh" <<'BASH' && chmod +x "$SCRIPTS_DIR/train_i2v_low.sh" && note_ok "$step" || note_fail "$step"
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models
OUT=/workspace/outputs/i2v_low
CACHE=/workspace/cache/i2v_low
CONF=/workspace/configs/dataset_i2v.json
mkdir -p "$OUT" "$CACHE"

python -m musubi_tuner.wan_cache_latents \
  --dataset_config "$CONF" \
  --vae "$M/vae/wan2.2_vae.safetensors" \
  --i2v \
  --batch_size 2 --num_workers 4

python -m musubi_tuner.wan_cache_text_encoder_outputs \
  --dataset_config "$CONF" \
  --t5 "$M/text_encoders/umt5_xxl_fp16.safetensors" \
  --batch_size 4

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
  --xformers
BASH

echo "[SCRIPTS] ready:"
ls -la "$SCRIPTS_DIR" | sed -n '1,80p' || true

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
  echo "FAILED (non-fatal; you can resume):"
  for f in "${FAILED[@]}"; do echo "  • $f"; done
  echo
  echo "You can re-run *only* a failed step by exporting env and re-executing the relevant part,"
  echo "or simply run the training scripts if models are already in place."
else
  echo "FAILED: (none)"
fi
echo "============================================"
echo

# ---------- keep alive for SSH ----------
echo "[BOOTSTRAP] complete — keeping container alive for SSH."
tail -f /dev/null
