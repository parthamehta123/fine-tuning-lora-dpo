"""Evaluate base model vs fine-tuned model on JSON extraction task."""

import argparse
import json

from pydantic import BaseModel, ValidationError


class ExtractedInfo(BaseModel):
    """Expected output schema for evaluation."""

    name: str | None = None
    company: str | None = None
    date: str | None = None
    location: str | None = None


TEST_CASES = [
    {
        "input": "Sarah Chen, VP of Engineering at Meta, presented at the AI Summit in New York on Jan 20, 2025.",
        "expected": {
            "name": "Sarah Chen",
            "company": "Meta",
            "date": "2025-01-20",
            "location": "New York",
        },
    },
    {
        "input": "No relevant entity information in this random sentence about weather.",
        "expected_refusal": True,
    },
]


def evaluate_json_output(raw_output: str) -> dict:
    """Evaluate a model's JSON output for validity and accuracy."""
    metrics = {"json_valid": False, "schema_valid": False, "exact_match": False}

    try:
        parsed = json.loads(raw_output)
        metrics["json_valid"] = True
    except json.JSONDecodeError:
        return metrics

    try:
        if not isinstance(parsed, dict):
            return metrics
        ExtractedInfo(**parsed)
        metrics["schema_valid"] = True
    except (ValidationError, TypeError):
        pass

    return metrics


def run_comparison(base_results: list[str], finetuned_results: list[str]):
    """Compare base model vs fine-tuned model outputs."""
    print(f"{'Metric':<25} {'Base Model':<15} {'Fine-Tuned':<15}")
    print("-" * 55)

    for label, results in [("Base", base_results), ("Fine-Tuned", finetuned_results)]:
        json_valid = sum(
            1 for r in results if evaluate_json_output(r)["json_valid"]
        ) / len(results)
        schema_valid = sum(
            1 for r in results if evaluate_json_output(r)["schema_valid"]
        ) / len(results)
        print(f"  JSON Validity:         {json_valid:.1%}")
        print(f"  Schema Validity:       {schema_valid:.1%}")

    report = {
        "test_cases": len(TEST_CASES),
        "base_model": {"json_validity": 0, "schema_validity": 0},
        "finetuned_model": {"json_validity": 0, "schema_validity": 0},
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="qwen3:8b")
    parser.add_argument("--finetuned-model", default="./models/sft-lora")
    args = parser.parse_args()

    print(f"Evaluation comparing {args.base_model} vs {args.finetuned_model}")
    print("Run inference on test cases and compare outputs.")
    print("See TEST_CASES in this file for the evaluation dataset.")


if __name__ == "__main__":
    main()
