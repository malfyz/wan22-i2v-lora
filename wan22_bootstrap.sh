#!/usr/bin/env bash
set -euo pipefail

# ---------- ENV ----------
HF_TOKEN="${HF_TOKEN:-}"        # optional
DATASET_URL="${DATASET_URL:-}"  # optional

export WORKDIR=/workspace
export MODELS_DIR=$WORKDIR/models
export DATASETS_DIR=$WORKDIR/datasets
export OUTPUTS_DIR=$WORKDIR/outputs
export CACHE_DIR=$WORKDIR/cache

mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae" \
         "$DATASETS_DIR/character_images" "$DATASETS_DIR/val" "$OUTPUTS_DIR" "$CACHE_DIR" \
         "$WORKDIR/scripts" "$WORKDIR/configs" /root/.cache/huggingface/accelerate

# HF auth only if provided
if [ -n "$HF_TOKEN" ]; then
  mkdir -p /root/.huggingface
  echo -n "$HF_TOKEN" > /root/.huggingface/token
fi

# ---------- DOWNLOADS ----------
echo "[WAN22] Downloading/validating required models..."
dl() {
  # dl <repo> <path_in_repo> <local_dir>
  local repo="$1"; shift
  local path="$1"; shift
  local out="$1"; shift
  mkdir -p "$out"
  local fname
  fname="$(basename "$path")"
  if [ ! -f "$out/$fname" ]; then
    echo "  - fetching $repo/$path → $out"
    if [ -n "$HF_TOKEN" ]; then
      hf download "$repo" "$path" --local-dir "$out" --token "$HF_TOKEN"
    else
      hf download "$repo" "$path" --local-dir "$out"
    fi
  else
    echo "  - exists: $out/$fname"
  fi
}

REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Text encoder (WAN 2.x, FP8 scaled) — what you asked for
# NOTE: this lives under split_files/text_encoders/ in the repack
dl "$REPO" "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "$MODELS_DIR/text_encoders"

# VAE
dl "$REPO" "split_files/vae/wan_2.1_vae.safetensors" "$MODELS_DIR/vae"

# I2V diffusion weights (fp16); the repack stores them under split_files then we move/rename as needed
TMP_SPLIT="$WORKDIR/split_files/diffusion_models"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$WORKDIR"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"  "$WORKDIR"
mkdir -p "$MODELS_DIR/diffusion_models"
if [ -d "$TMP_SPLIT" ]; then
  mv -f "$TMP_SPLIT/"*.safetensors "$MODELS_DIR/diffusion_models/" 2>/dev/null || true
fi

# ---------- OPTIONAL: DATASET IMPORT ----------
if [ -n "$DATASET_URL" ]; then
  echo "[WAN22] Fetching dataset from $DATASET_URL"
  cd "$DATASETS_DIR"
  aria2c -x 8 -s 8 "$DATASET_URL" -o dataset_archive || true
  if file -b dataset_archive | grep -qi zip; then
    unzip -o dataset_archive -d character_images
  else
    mkdir -p tmp && tar -xf dataset_archive -C tmp || true
    rsync -a tmp/ character_images/ || true
    rm -rf tmp
  fi
  rm -f dataset_archive
fi

# ---------- DATASET CONFIG (560x720) ----------
cat > "$WORKDIR/configs/dataset_i2v.json" <<'JSON'
{
  "name": "wan22_i2v_char_560x720",
  "type": "i2v",
  "resolution": [560, 720],
  "frames_per_sample": 32,
  "bucket_resolutions": [[560, 720]],
  "train_data": [
    { "path": "/workspace/datasets/character_images", "caption_ext": ".txt", "shuffle": true, "repeat": 1 }
  ],
  "val_images": ["/workspace/datasets/val/portrait1.png"]
}
JSON

# ---------- TRAINING SCRIPTS ----------
# Use the same text encoder file we just downloaded above
TE="$MODELS_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

cat > "$WORKDIR/scripts/train_i2v_high.sh" <<BASH
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models; OUT=/workspace/outputs/i2v_high; CACHE=/workspace/cache/i2v_high; CONF=/workspace/configs/dataset_i2v.json
mkdir -p "\$OUT" "\$CACHE"
python /workspace/musubi-tuner/wan_cache_latents.py \
  --model_path "\$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --text_encoder_path "$TE" \
  --vae_path "\$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "\$CONF" --output_dir "\$CACHE" --i2v
python /workspace/musubi-tuner/wan_cache_text_encoder_outputs.py \
  --text_encoder_path "$TE" \
  --dataset_config "\$CONF" --output_dir "\$CACHE"
accelerate launch /workspace/musubi-tuner/wan_train_network.py \
  --model_path "\$M/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  --text_encoder_path "$TE" \
  --vae_path "\$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "\$CONF" --output_dir "\$OUT" \
  --network_module lora --rank 32 \
  --learning_rate 1e-4 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_epochs 2 --mixed_precision bf16 --i2v \
  --cache_latents_dir "\$CACHE" --cache_text_encoder_outputs_dir "\$CACHE" \
  --enable_xformers --fp8_llm --blocks_to_swap 16
echo "HN LoRA saved to \$OUT"
BASH
chmod +x "$WORKDIR/scripts/train_i2v_high.sh"

cat > "$WORKDIR/scripts/train_i2v_low.sh" <<BASH
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models; OUT=/workspace/outputs/i2v_low; CACHE=/workspace/cache/i2v_low; CONF=/workspace/configs/dataset_i2v.json
mkdir -p "\$OUT" "\$CACHE"
python /workspace/musubi-tuner/wan_cache_latents.py \
  --model_path "\$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --text_encoder_path "$TE" \
  --vae_path "\$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "\$CONF" --output_dir "\$CACHE" --i2v
python /workspace/musubi-tuner/wan_cache_text_encoder_outputs.py \
  --text_encoder_path "$TE" \
  --dataset_config "\$CONF" --output_dir "\$CACHE"
accelerate launch /workspace/musubi-tuner/wan_train_network.py \
  --model_path "\$M/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  --text_encoder_path "$TE" \
  --vae_path "\$M/vae/wan_2.1_vae.safetensors" \
  --dataset_config "\$CONF" --output_dir "\$OUT" \
  --network_module lora --rank 32 \
  --learning_rate 5e-5 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_epochs 2 --mixed_precision bf16 --i2v \
  --cache_latents_dir "\$CACHE" --cache_text_encoder_outputs_dir "\$CACHE" \
  --enable_xformers --fp8_llm --blocks_to_swap 16
echo "LN LoRA saved to \$OUT"
BASH
chmod +x "$WORKDIR/scripts/train_i2v_low.sh"

cat > "$WORKDIR/scripts/train_both.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
bash /workspace/scripts/train_i2v_high.sh
bash /workspace/scripts/train_i2v_low.sh
BASH
chmod +x "$WORKDIR/scripts/train_both.sh"

# ---------- READY ----------
echo "============================================================"
echo "READY: WAN 2.2 I2V HN/LN LoRA trainer."
echo "Models in: $MODELS_DIR"
echo "Configs:   $WORKDIR/configs/dataset_i2v.json"
echo "Scripts:   $WORKDIR/scripts/train_i2v_high.sh | train_i2v_low.sh"
echo "============================================================"

# Keep container alive for SSH/interactive use
tail -f /dev/null
