"""Tests for data preparation pipeline."""

import json

from src.prepare_data import (
    create_function_calling_dpo_examples,
    create_function_calling_sft_examples,
    save_jsonl,
)


def test_sft_examples_format():
    examples = create_function_calling_sft_examples()
    assert len(examples) >= 10
    for ex in examples:
        assert "instruction" in ex
        assert "input" in ex
        assert "output" in ex
        parsed = json.loads(ex["output"])
        assert isinstance(parsed, dict)
        assert "function" in parsed


def test_sft_examples_cover_all_functions():
    examples = create_function_calling_sft_examples()
    functions_seen = {json.loads(ex["output"])["function"] for ex in examples}
    expected = {
        "get_stock_price",
        "get_financial_report",
        "calculate_ratio",
        "search_sec_filings",
        "convert_currency",
        "NONE",
    }
    assert expected == functions_seen


def test_sft_examples_include_refusals():
    examples = create_function_calling_sft_examples()
    refusals = [ex for ex in examples if json.loads(ex["output"])["function"] == "NONE"]
    assert len(refusals) >= 2
    for r in refusals:
        parsed = json.loads(r["output"])
        assert "reason" in parsed


def test_dpo_examples_format():
    examples = create_function_calling_dpo_examples()
    assert len(examples) >= 5
    for ex in examples:
        assert "prompt" in ex
        assert "chosen" in ex
        assert "rejected" in ex
        chosen = json.loads(ex["chosen"])
        rejected = json.loads(ex["rejected"])
        assert isinstance(chosen, dict)
        assert isinstance(rejected, dict)


def test_save_jsonl(tmp_path):
    data = [{"a": 1}, {"b": 2}]
    path = str(tmp_path / "test.jsonl")
    save_jsonl(data, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
