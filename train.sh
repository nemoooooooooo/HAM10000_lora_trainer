#!/bin/bash


# Model and dataset paths
MODEL_NAME="black-forest-labs/FLUX.1-dev"
OUTPUT_DIR="ham10000_mel_flux/lora_output"
DATASET_NAME="BoooomNing/ham10000-mel-flux"

# Training configuration
TRAIN_BATCH_SIZE=1  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 4
LEARNING_RATE=1e-5
MAX_TRAIN_STEPS=10000  # 2000 samples * 2 epochs with batch size 4
CHECKPOINTING_STEPS=1000
VALIDATION_EPOCHS=1

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Log configuration
echo "==========================================="
echo "HAM10000 Multi-Class LoRA Training"
echo "==========================================="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective Batch Size: $((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Steps: ${MAX_TRAIN_STEPS}"
echo "==========================================="

# Run training
accelerate launch train.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --dataset_name="$DATASET_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="bf16" \
  --instance_prompt="" \
  --caption_column="text" \
  --image_column="image" \
  --resolution 512 \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --learning_rate=$LEARNING_RATE \
  --seed=42 \
  --rank=128 \
  --lora_alpha=128 \
  --validation_prompt="melanomadet, clinical dermoscopy showing melanoma" \
  --validation_prompt="melanomadet dermoscopic image" \
  --validation_prompt="melanomadet" \
  --num_validation_images=9 \
  --validation_steps=500 \
  --checkpointing_steps=2000 \
  --report_to="wandb" \
  --guidance_scale=1 \