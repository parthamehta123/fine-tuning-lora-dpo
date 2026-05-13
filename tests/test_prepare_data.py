"""Tests for data preparation pipeline."""

import json

from src.prepare_data import create_dpo_examples, create_sft_examples, save_jsonl


def test_sft_examples_format():
    examples = create_sft_examples()
    assert len(examples) >= 1
    for ex in examples:
        assert "instruction" in ex
        assert "input" in ex
        assert "output" in ex
        # Output should be valid JSON
        parsed = json.loads(ex["output"])
        assert isinstance(parsed, dict)


def test_dpo_examples_format():
    examples = create_dpo_examples()
    assert len(examples) >= 1
    for ex in examples:
        assert "prompt" in ex
        assert "chosen" in ex
        assert "rejected" in ex
        chosen = json.loads(ex["chosen"])
        rejected = json.loads(ex["rejected"])
        assert isinstance(chosen, dict)
        assert isinstance(rejected, dict)
        # Chosen should have more fields than rejected
        assert len(chosen) >= len(rejected)


def test_save_jsonl(tmp_path):
    data = [{"a": 1}, {"b": 2}]
    path = str(tmp_path / "test.jsonl")
    save_jsonl(data, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
