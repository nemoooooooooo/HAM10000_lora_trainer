# HAM10000 FLUX LoRA - Synthetic Dermoscopy Image Generator

A LoRA (Low-Rank Adaptation) fine-tuning implementation for FLUX.1-dev model to generate synthetic dermoscopy images of skin lesions, trained on the HAM10000 dataset for multi-class dermatological image synthesis.

## Overview

This project fine-tunes the FLUX.1-dev model using LoRA to generate high-quality synthetic dermoscopy images across multiple skin lesion categories. The model can be used for data augmentation.


## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
bash train.sh
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | FLUX.1-dev | Foundation model |
| Dataset | HAM10000 | 10,000+ dermoscopic images |
| LoRA Rank | 128 | Parameter efficiency |
| Steps | 10,000 | Training iterations |
| Precision | bf16 | Mixed precision training |

## Usage

After training:
```python
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
pipeline.load_lora_weights("./lora_output")
image = pipeline("melanomadet, clinical dermoscopy showing melanoma").images[0]
```

## Files
- `train.py` - Main training script
- `train.sh` - Training configuration
- `data_loader.py` - Dataset handling
- `utils.py` - Helper functions

## Monitoring
Training logs to W&B. Checkpoints saved every 2000 steps.

## Model Outputs

The trained LoRA adapter will be saved in the output directory with:
- `pytorch_lora_weights.safetensors` - LoRA weight file
- `checkpoint-*/` - Training checkpoints
- Validation images logged to Wandb (if enabled)

