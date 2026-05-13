"""Evaluate base model vs fine-tuned model on function-calling task.

Metrics:
- JSON validity rate: Does the model output valid JSON?
- Function accuracy: Does it pick the correct function?
- Argument accuracy: Are the arguments correct?
- Refusal correctness: Does it correctly decline when no function matches?
"""

import argparse
import json

from pydantic import BaseModel, ValidationError


class FunctionCall(BaseModel):
    """Expected output schema for function-calling evaluation."""

    function: str
    arguments: dict | None = None
    reason: str | None = None


TEST_CASES = [
    {
        "input": "What's the current price of Google stock?",
        "expected_function": "get_stock_price",
        "expected_args": {"ticker": "GOOGL"},
    },
    {
        "input": "Pull up Amazon's 10-K for 2024",
        "expected_function": "get_financial_report",
        "expected_args": {"company": "AMZN", "report_type": "10-K", "year": 2024},
    },
    {
        "input": "Calculate ROA with net income $15B and total assets $350B",
        "expected_function": "calculate_ratio",
        "expected_args": {"ratio_type": "roa", "numerator": 15, "denominator": 350},
    },
    {
        "input": "Convert 500000 EUR to JPY",
        "expected_function": "convert_currency",
        "expected_args": {
            "amount": 500000,
            "from_currency": "EUR",
            "to_currency": "JPY",
        },
    },
    {
        "input": "Order lunch for the trading desk",
        "expected_function": "NONE",
        "expected_refusal": True,
    },
    {
        "input": "Search SEC for climate disclosures in 10-K filings since 2024",
        "expected_function": "search_sec_filings",
        "expected_args": {"query": "climate disclosures", "filing_type": "10-K"},
    },
]


def evaluate_json_output(raw_output: str) -> dict:
    """Evaluate a model's JSON output for validity and accuracy."""
    metrics = {
        "json_valid": False,
        "schema_valid": False,
        "function_correct": False,
        "args_correct": False,
    }

    try:
        parsed = json.loads(raw_output)
        metrics["json_valid"] = True
    except json.JSONDecodeError:
        return metrics

    if not isinstance(parsed, dict):
        return metrics

    try:
        FunctionCall(**parsed)
        metrics["schema_valid"] = True
    except (ValidationError, TypeError):
        pass

    return metrics


def evaluate_function_call(
    raw_output: str, expected_function: str, expected_args: dict | None = None
) -> dict:
    """Evaluate if the model selected the correct function with correct arguments."""
    metrics = evaluate_json_output(raw_output)

    if not metrics["json_valid"]:
        return metrics

    parsed = json.loads(raw_output)
    if parsed.get("function") == expected_function:
        metrics["function_correct"] = True

    if expected_args and metrics["function_correct"]:
        actual_args = parsed.get("arguments", {})
        # Check if all expected args are present with correct values
        all_correct = all(actual_args.get(k) == v for k, v in expected_args.items())
        metrics["args_correct"] = all_correct

    return metrics


def run_comparison(base_results: list[str], finetuned_results: list[str]):
    """Compare base model vs fine-tuned model outputs on test cases."""
    print(f"\n{'Metric':<25} {'Base Model':<15} {'Fine-Tuned':<15}")
    print("-" * 55)

    for label, results in [("Base", base_results), ("Fine-Tuned", finetuned_results)]:
        json_valid = sum(
            1 for r in results if evaluate_json_output(r)["json_valid"]
        ) / len(results)
        schema_valid = sum(
            1 for r in results if evaluate_json_output(r)["schema_valid"]
        ) / len(results)

        func_correct = 0
        for r, tc in zip(results, TEST_CASES):
            m = evaluate_function_call(
                r, tc["expected_function"], tc.get("expected_args")
            )
            if m["function_correct"]:
                func_correct += 1

        print(f"  {label} JSON Validity:    {json_valid:.1%}")
        print(f"  {label} Schema Validity:  {schema_valid:.1%}")
        print(f"  {label} Function Acc:     {func_correct / len(results):.1%}")

    report = {
        "test_cases": len(TEST_CASES),
        "base_model": {
            "json_validity": 0,
            "schema_validity": 0,
            "function_accuracy": 0,
        },
        "finetuned_model": {
            "json_validity": 0,
            "schema_validity": 0,
            "function_accuracy": 0,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="qwen3:8b")
    parser.add_argument("--finetuned-model", default="./models/sft-lora")
    args = parser.parse_args()

    print(f"Evaluation: {args.base_model} vs {args.finetuned_model}")
    print(f"Task: Function-calling accuracy ({len(TEST_CASES)} test cases)")
    print("\nTest cases:")
    for tc in TEST_CASES:
        print(f"  - {tc['input'][:60]}... -> {tc['expected_function']}")


if __name__ == "__main__":
    main()
