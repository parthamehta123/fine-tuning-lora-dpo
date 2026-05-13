# Fine-Tuning with LoRA & DPO

Fine-tune small language models for structured JSON extraction and tool-calling using LoRA/QLoRA for supervised fine-tuning and DPO for preference-based alignment.

## Features

- **Supervised Fine-Tuning (SFT)**: LoRA/QLoRA on Qwen 3 8B for JSON extraction
- **Preference Tuning (DPO)**: Train model to prefer better outputs using ranked pairs
- **Evaluation Pipeline**: JSON validity rate, exact match accuracy, refusal correctness
- **Training Curves**: Visualize loss and metrics throughout training
- **Before/After Comparison**: Clear metrics showing improvement over base model

## Phases

### Phase 1: Supervised Fine-Tuning
- Clean dataset of 2K-10K examples
- LoRA/QLoRA parameter-efficient fine-tuning
- Evaluation on held-out test set
- Metrics: JSON validity, exact match, refusal correctness

### Phase 2: DPO Preference Tuning
- Generate multiple outputs per prompt
- Label chosen/rejected pairs
- Train with DPO loss
- Show incremental improvement over SFT

## Tech Stack

- **Training**: Hugging Face TRL (SFT + DPO trainers)
- **PEFT**: LoRA / QLoRA via peft library
- **Base Model**: Qwen 3 8B
- **Data**: Custom JSON extraction dataset
- **Evaluation**: Custom metrics + training curves

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Prepare dataset
python src/prepare_data.py

# Run SFT with LoRA
python src/train_sft.py --config configs/sft_config.yaml

# Run DPO training
python src/train_dpo.py --config configs/dpo_config.yaml

# Evaluate before/after
python src/evaluate.py --base-model qwen3:8b --finetuned-model ./models/sft-lora

# Generate report
python src/report.py
```

## Project Structure

```
fine-tuning-lora-dpo/
├── src/
│   ├── prepare_data.py     # Dataset preparation and formatting
│   ├── train_sft.py        # Supervised fine-tuning with LoRA
│   ├── train_dpo.py        # DPO preference optimization
│   ├── evaluate.py         # Before/after evaluation
│   └── report.py           # Generate training report
├── configs/
│   ├── sft_config.yaml     # SFT hyperparameters
│   └── dpo_config.yaml     # DPO hyperparameters
├── data/                   # Training and eval datasets
├── models/                 # Saved model checkpoints
└── eval/                   # Evaluation results
```
