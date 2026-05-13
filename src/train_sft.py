"""Supervised fine-tuning with LoRA/QLoRA using TRL."""

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    # Quantization config for QLoRA
    bnb_config = None
    if config["model"].get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": config["data"]["train_path"],
            "eval": config["data"]["eval_path"],
        },
    )

    def format_example(example):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"

    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=config["output"]["dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        eval_strategy="steps",
        bf16=True,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
        formatting_func=format_example,
        max_seq_length=train_cfg["max_seq_length"],
    )

    trainer.train()
    trainer.save_model(config["output"]["dir"])
    tokenizer.save_pretrained(config["output"]["dir"])
    print(f"Model saved to {config['output']['dir']}")


if __name__ == "__main__":
    main()
