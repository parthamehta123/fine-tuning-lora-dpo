# Fine-Tuning with LoRA & DPO

Fine-tune small language models for **financial function-calling** using LoRA/QLoRA for supervised fine-tuning and DPO for preference-based alignment.

## Results

Trained on 472 examples (320 augmented + 152 Glaive function calls), evaluated on 6 held-out test cases:

| Metric | Score |
|--------|-------|
| JSON Validity | **100%** |
| Schema Validity | **100%** |
| Function Accuracy | **100%** |
| Argument Accuracy | **83.3%** |

## Dataset

- **Augmented Custom**: 320 template-generated examples across 5 financial tools (stock prices, reports, ratios, SEC filings, currency conversion) + refusal examples
- **Glaive Function-Calling**: [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) — filtered to valid JSON function calls
- **DPO Pairs**: 8 chosen/rejected pairs comparing correct vs incorrect function calls

## Google Colab

The easiest way to run the full pipeline (no local GPU needed):

1. Upload this folder to Google Drive
2. Open `colab_notebook.ipynb` in Colab
3. Select a T4 or A100 GPU runtime
4. Run all cells

The notebook handles setup, data generation, SFT training, DPO training, evaluation, and interactive testing end-to-end.

## LoRA vs QLoRA

| Method | Precision | VRAM Needed | Config |
|--------|-----------|-------------|--------|
| **QLoRA** | 4-bit (NF4) | ~6GB (T4/A10) | `configs/sft_config.yaml` |
| **LoRA** | 16-bit (bf16) | ~16GB (A100) | `configs/sft_lora_fullprecision.yaml` |

Both configs target the same layers (attention + MLP projections) with identical hyperparameters for fair comparison. QLoRA uses `bitsandbytes` NF4 quantization with double quantization enabled.

## Features

- **Supervised Fine-Tuning (SFT)**: LoRA/QLoRA on Qwen 3 8B for JSON function-calling
- **Preference Tuning (DPO)**: Train model to prefer correct function calls using ranked pairs
- **Data Augmentation**: Template-based generation of diverse training examples
- **Evaluation Pipeline**: JSON validity, schema conformance, function accuracy, argument accuracy, refusal correctness
- **Colab Notebook**: Complete end-to-end pipeline runnable on free GPU

## Phases

### Phase 1: Supervised Fine-Tuning
- Augmented dataset of 320+ examples with diverse natural language queries
- Glaive function-calling dataset (filtered to JSON function calls)
- LoRA/QLoRA parameter-efficient fine-tuning on Qwen 3 8B
- Evaluation: 100% JSON validity, 100% function accuracy

### Phase 2: DPO Preference Tuning
- 8 chosen/rejected pairs comparing correct vs incorrect function calls
- Trains model to prefer proper ticker symbols, correct argument ordering, appropriate refusals
- Stacks on top of SFT checkpoint

## Tech Stack

- **Training**: Hugging Face TRL (SFT + DPO trainers)
- **PEFT**: LoRA / QLoRA via peft library
- **Base Model**: Qwen 3 8B
- **Data**: Augmented financial function-calling dataset + Glaive v2
- **Evaluation**: Pydantic schema validation + custom metrics

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
# Prepare custom dataset (20 hand-crafted examples)
python src/prepare_data.py

# Download Glaive dataset (filtered to function calls)
python src/download_dataset.py --max-examples 10000

# Run SFT with QLoRA (needs GPU, ~6GB VRAM)
python src/train_sft.py --config configs/sft_config.yaml

# Run DPO training (stacks on SFT checkpoint)
python src/train_dpo.py --config configs/dpo_config.yaml

# Evaluate
python src/evaluate.py --base-model Qwen/Qwen3-8B --finetuned-model ./models/sft-lora

# Run tests (no GPU needed)
pytest tests/ -v
```

## Project Structure

```
fine-tuning-lora-dpo/
├── colab_notebook.ipynb          # Full pipeline for Google Colab
├── src/
│   ├── prepare_data.py           # Custom dataset (20 examples + DPO pairs)
│   ├── download_dataset.py       # Glaive dataset downloader + parser
│   ├── train_sft.py              # Supervised fine-tuning with LoRA/QLoRA
│   ├── train_dpo.py              # DPO preference optimization
│   └── evaluate.py               # Function-calling accuracy metrics
├── configs/
│   ├── sft_config.yaml           # QLoRA SFT config (~6GB VRAM)
│   ├── sft_lora_fullprecision.yaml  # LoRA config (~16GB VRAM)
│   └── dpo_config.yaml           # DPO config (stacks on SFT)
├── data/                         # Training and eval datasets
├── models/                       # Saved LoRA adapter checkpoints
├── tests/                        # Unit tests (no GPU needed)
└── eval/                         # Evaluation results
```
