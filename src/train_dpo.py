"""DPO preference optimization training using TRL."""

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    sft_path = config["model"]["sft_checkpoint"]

    bnb_config = None
    if config["model"].get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load SFT model as base for DPO
    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    model = AutoModelForCausalLM.from_pretrained(
        sft_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load DPO preference dataset
    dataset = load_dataset("json", data_files={
        "train": config["data"]["train_path"],
        "eval": config["data"]["eval_path"],
    })

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
        bf16=True,
        report_to="wandb",
    )

    dpo_cfg = config["dpo"]
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=dpo_cfg["beta"],
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        max_length=train_cfg["max_seq_length"],
        loss_type=dpo_cfg["loss_type"],
    )

    trainer.train()
    trainer.save_model(config["output"]["dir"])
    tokenizer.save_pretrained(config["output"]["dir"])
    print(f"DPO model saved to {config['output']['dir']}")


if __name__ == "__main__":
    main()
