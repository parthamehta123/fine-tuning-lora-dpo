"""Tests for function-calling evaluation logic."""

import json

from src.evaluate import (
    FunctionCall,
    evaluate_function_call,
    evaluate_json_output,
    run_comparison,
)


def test_evaluate_valid_json():
    valid = json.dumps({"function": "get_stock_price", "arguments": {"ticker": "AAPL"}})
    result = evaluate_json_output(valid)
    assert result["json_valid"] is True
    assert result["schema_valid"] is True


def test_evaluate_invalid_json():
    result = evaluate_json_output("not json at all")
    assert result["json_valid"] is False
    assert result["schema_valid"] is False


def test_evaluate_valid_json_invalid_schema():
    result = evaluate_json_output('"just a string"')
    assert result["json_valid"] is True
    assert result["schema_valid"] is False


def test_function_call_schema():
    fc = FunctionCall(function="get_stock_price", arguments={"ticker": "AAPL"})
    assert fc.function == "get_stock_price"
    assert fc.arguments["ticker"] == "AAPL"


def test_function_call_refusal():
    fc = FunctionCall(function="NONE", reason="No matching function")
    assert fc.function == "NONE"
    assert fc.reason == "No matching function"


def test_evaluate_function_call_correct():
    output = json.dumps(
        {"function": "get_stock_price", "arguments": {"ticker": "AAPL"}}
    )
    result = evaluate_function_call(output, "get_stock_price", {"ticker": "AAPL"})
    assert result["function_correct"] is True
    assert result["args_correct"] is True


def test_evaluate_function_call_wrong_function():
    output = json.dumps(
        {"function": "convert_currency", "arguments": {"ticker": "AAPL"}}
    )
    result = evaluate_function_call(output, "get_stock_price", {"ticker": "AAPL"})
    assert result["function_correct"] is False


def test_evaluate_function_call_wrong_args():
    output = json.dumps(
        {"function": "get_stock_price", "arguments": {"ticker": "MSFT"}}
    )
    result = evaluate_function_call(output, "get_stock_price", {"ticker": "AAPL"})
    assert result["function_correct"] is True
    assert result["args_correct"] is False


def test_run_comparison_outputs():
    base = [
        json.dumps({"function": "get_stock_price", "arguments": {"ticker": "AAPL"}}),
        "invalid json",
    ]
    finetuned = [
        json.dumps({"function": "get_stock_price", "arguments": {"ticker": "GOOGL"}}),
        json.dumps(
            {
                "function": "get_financial_report",
                "arguments": {"company": "AMZN", "report_type": "10-K", "year": 2024},
            }
        ),
    ]
    report = run_comparison(base, finetuned)
    assert report["test_cases"] > 0
