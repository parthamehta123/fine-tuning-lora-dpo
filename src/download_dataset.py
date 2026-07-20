"""Download and prepare the Glaive function-calling dataset from HuggingFace.

Dataset: glaiveai/glaive-function-calling-v2
Contains ~113k function-calling examples with system prompts, user queries,
and expected function calls.

Usage:
    python src/download_dataset.py
    python src/download_dataset.py --max-examples 5000
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def download_and_prepare(max_examples: int | None = None):
    """Download Glaive dataset and convert to SFT/DPO training format."""
    print("Downloading glaiveai/glaive-function-calling-v2 from HuggingFace...")
    dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    print(f"Processing {len(dataset)} examples...")

    sft_examples = []
    for row in dataset:
        # Parse the chat format into instruction/input/output
        chat = row.get("chat", "") or row.get("conversations", "")
        system = row.get("system", "") or ""

        if not chat:
            continue

        # Extract user message and first assistant function call
        # Glaive v2 format uses "USER:" and "ASSISTANT:" prefixes,
        # separated by <|endoftext|> tokens. Function calls are wrapped
        # in <functioncall> tags.
        chat_str = str(chat)

        # Split on common turn delimiters
        user_msg = ""
        assistant_msg = ""

        # Find first USER: message
        for prefix in ["USER:", "USER :"]:
            idx = chat_str.find(prefix)
            if idx != -1:
                start = idx + len(prefix)
                # User message ends at next turn marker or end-of-text
                end = len(chat_str)
                for marker in ["ASSISTANT:", "ASSISTANT :", "<|endoftext|>", "\nFUNCTION"]:
                    m = chat_str.find(marker, start)
                    if m != -1 and m < end:
                        end = m
                user_msg = chat_str[start:end].strip()
                break

        # Find first function call in any ASSISTANT turn
        if "<functioncall>" in chat_str:
            fc_start = chat_str.find("<functioncall>") + len("<functioncall>")
            fc_end = chat_str.find("<|endoftext|>", fc_start)
            if fc_end == -1:
                fc_end = chat_str.find("</functioncall>", fc_start)
            if fc_end == -1:
                fc_end = len(chat_str)
            raw_call = chat_str[fc_start:fc_end].strip()
            # Validate it's parseable JSON with a "name" key
            try:
                parsed = json.loads(raw_call)
                if isinstance(parsed, dict) and "name" in parsed:
                    # Convert Glaive format to our format
                    args = parsed.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    assistant_msg = json.dumps(
                        {"function": parsed["name"], "arguments": args}
                    )
            except json.JSONDecodeError:
                pass

        if user_msg and assistant_msg:
            sft_examples.append(
                {
                    "instruction": system[:500]
                    if system
                    else "Call the appropriate function based on the user's request.",
                    "input": user_msg[:500],
                    "output": assistant_msg[:1000],
                }
            )

    print(f"Prepared {len(sft_examples)} valid SFT examples")

    # Split: 90% train, 10% eval
    split = int(len(sft_examples) * 0.9)
    train_data = sft_examples[:split]
    eval_data = sft_examples[split:]

    Path("data").mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    for name, data in [
        ("glaive_sft_train.jsonl", train_data),
        ("glaive_sft_eval.jsonl", eval_data),
    ]:
        path = f"data/{name}"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} examples to {path}")

    print("\nTo use this data, update configs/sft_config.yaml:")
    print('  train_path: "data/glaive_sft_train.jsonl"')
    print('  eval_path: "data/glaive_sft_eval.jsonl"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Limit number of examples"
    )
    args = parser.parse_args()
    download_and_prepare(args.max_examples)


if __name__ == "__main__":
    main()
