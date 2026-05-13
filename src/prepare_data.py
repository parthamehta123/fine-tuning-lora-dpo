"""Prepare training data for SFT and DPO fine-tuning."""

import json
from pathlib import Path


def create_sft_examples() -> list[dict]:
    """Create sample SFT training data for JSON extraction task."""
    examples = [
        {
            "instruction": "Extract structured information from the following text as JSON.",
            "input": "John Smith, a 35-year-old software engineer at Google, filed a patent on March 15, 2025 in San Francisco.",
            "output": json.dumps(
                {
                    "name": "John Smith",
                    "age": 35,
                    "occupation": "software engineer",
                    "company": "Google",
                    "action": "filed a patent",
                    "date": "2025-03-15",
                    "location": "San Francisco",
                }
            ),
        },
        {
            "instruction": "Extract structured information from the following text as JSON.",
            "input": "The quarterly earnings report from Apple showed revenue of $94.8 billion for Q1 2025, a 4% increase year-over-year.",
            "output": json.dumps(
                {
                    "company": "Apple",
                    "report_type": "quarterly earnings",
                    "revenue": "$94.8 billion",
                    "period": "Q1 2025",
                    "growth": "4% increase year-over-year",
                }
            ),
        },
    ]
    # In practice, you'd have 2,000-10,000 examples
    return examples


def create_dpo_examples() -> list[dict]:
    """Create DPO preference pairs (chosen vs rejected outputs)."""
    examples = [
        {
            "prompt": "Extract structured information from: 'Tesla CEO Elon Musk announced layoffs affecting 10% of the workforce on April 15, 2025.'",
            "chosen": json.dumps(
                {
                    "company": "Tesla",
                    "person": "Elon Musk",
                    "role": "CEO",
                    "event": "layoffs",
                    "impact": "10% of workforce",
                    "date": "2025-04-15",
                }
            ),
            "rejected": json.dumps(
                {
                    "text": "Tesla layoffs",
                    "info": "Elon Musk fired people",
                }
            ),
        },
    ]
    return examples


def save_jsonl(data: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} examples to {path}")


def main():
    sft_data = create_sft_examples()
    save_jsonl(sft_data[: int(len(sft_data) * 0.9)] or sft_data, "data/sft_train.jsonl")
    save_jsonl(
        sft_data[int(len(sft_data) * 0.9) :] or sft_data[-1:], "data/sft_eval.jsonl"
    )

    dpo_data = create_dpo_examples()
    save_jsonl(dpo_data, "data/dpo_train.jsonl")
    save_jsonl(dpo_data, "data/dpo_eval.jsonl")


if __name__ == "__main__":
    main()
