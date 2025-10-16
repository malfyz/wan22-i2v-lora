#!/usr/bin/env bash
set -euo pipefail

HF_TOKEN="${HF_TOKEN:-}"
DATASET_URL="${DATASET_URL:-}"

export WORKDIR=/workspace
export MODELS_DIR=$WORKDIR/models
export DATASETS_DIR=$WORKDIR/datasets
export OUTPUTS_DIR=$WORKDIR/outputs
export CACHE_DIR=$WORKDIR/cache

mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae" \
         "$DATASETS_DIR/character_images" "$DATASETS_DIR/val" "$OUTPUTS_DIR" "$CACHE_DIR" \
         "$WORKDIR/scripts" "$WORKDIR/configs" /root/.cache/huggingface/accelerate

if [ -n "$HF_TOKEN" ]; then
  mkdir -p /root/.huggingface
  echo -n "$HF_TOKEN" > /root/.huggingface/token
fi

echo "[WAN22] Downloading models…"

# downloader function
dl() {
  repo="$1"; shift
  path="$1"; shift
  out="$1"; shift
  mkdir -p "$out"
  fname="$(basename "$path")"
  if [ ! -f "$out/$fname" ]; then
    echo "  - $path → $out"
    if [ -n "$HF_TOKEN" ]; then
      hf download "$repo" "$path" --local-dir "$out" --token "$HF_TOKEN"
    else
      hf download "$repo" "$path" --local-dir "$out"
    fi
  else
    echo "  - exists: $fname"
  fi
}

REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Text encoder
dl "$REPO" "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "$MODELS_DIR/text_encoders"

# VAE
dl "$REPO" "split_files/vae/wan_2.1_vae.safetensors" "$MODELS_DIR/vae"

# Diffusion weights
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" "$WORKDIR"
dl "$REPO" "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" "$WORKDIR"

mkdir -p "$MODELS_DIR/diffusion_models"
if [ -d "$WORKDIR/split_files/diffusion_models" ]; then
  mv -f "$WORKDIR/split_files/diffusion_models/"*.safetensors "$MODELS_DIR/diffusion_models/" || true
fi

# Dataset fetch
if [ -n "$DATASET_URL" ]; then
  echo "Fetching dataset from $DATASET_URL"
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

# Generate dataset_i2v.json
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

# Write training scripts
TE="$MODELS_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE="$MODELS_DIR/vae/wan_2.1_vae.safetensors"
HN="$MODELS_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
LN="$MODELS_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

cat > "$WORKDIR/scripts/train_i2v_high.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models; OUT=/workspace/outputs/i2v_high; CACHE=/workspace/cache/i2v_high; CONF=/workspace/configs/dataset_i2v.json
mkdir -p "\$OUT" "\$CACHE"
python /workspace/musubi-tuner/wan_cache_latents.py \
  --dataset_config "\$CONF" --vae "$VAE" --i2v --output_dir "\$CACHE"
python /workspace/musubi-tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config "\$CONF" --clip "$TE" --output_dir "\$CACHE"
accelerate launch /workspace/musubi-tuner/wan_train_network.py \
  --dataset_config "\$CONF" --model "$HN" --clip "$TE" --vae "$VAE" --output_dir "\$OUT" \
  --network_module lora --rank 32 --learning_rate 1e-4 --train_batch_size 1 \
  --gradient_accumulation_steps 4 --max_train_epochs 2 --mixed_precision bf16 --i2v \
  --cache_latents_dir "\$CACHE" --cache_text_encoder_outputs_dir "\$CACHE" \
  --enable_xformers --fp8_llm --blocks_to_swap 16
echo "HN LoRA saved to \$OUT"
BASH
chmod +x "$WORKDIR/scripts/train_i2v_high.sh"

cat > "$WORKDIR/scripts/train_i2v_low.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
M=/workspace/models; OUT=/workspace/outputs/i2v_low; CACHE=/workspace/cache/i2v_low; CONF=/workspace/configs/dataset_i2v.json
mkdir -p "\$OUT" "\$CACHE"
python /workspace/musubi-tuner/wan_cache_latents.py \
  --dataset_config "\$CONF" --vae "$VAE" --i2v --output_dir "\$CACHE"
python /workspace/musubi-tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config "\$CONF" --clip "$TE" --output_dir "\$CACHE"
accelerate launch /workspace/musubi-tuner/wan_train_network.py \
  --dataset_config "\$CONF" --model "$LN" --clip "$TE" --vae "$VAE" --output_dir "\$OUT" \
  --network_module lora --rank 32 --learning_rate 5e-5 --train_batch_size 1 \
  --gradient_accumulation_steps 4 --max_train_epochs 2 --mixed_precision bf16 --i2v \
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

echo "============================================================"
echo "READY: WAN 2.2 I2V HN/LN LoRA trainer."
echo "Models in: $MODELS_DIR"
echo "Configs:   $WORKDIR/configs/dataset_i2v.json"
echo "Scripts:   $WORKDIR/scripts/train_i2v_high.sh | train_i2v_low.sh"
echo "============================================================"

tail -f /dev/null
