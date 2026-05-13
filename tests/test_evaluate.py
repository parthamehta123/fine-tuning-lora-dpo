"""Tests for evaluation logic."""

import json

from src.evaluate import ExtractedInfo, evaluate_json_output, run_comparison


def test_evaluate_valid_json():
    valid = json.dumps(
        {"name": "Alice", "company": "Acme", "date": "2025-01-01", "location": "NYC"}
    )
    result = evaluate_json_output(valid)
    assert result["json_valid"] is True
    assert result["schema_valid"] is True


def test_evaluate_invalid_json():
    result = evaluate_json_output("not json at all")
    assert result["json_valid"] is False
    assert result["schema_valid"] is False


def test_evaluate_valid_json_invalid_schema():
    # Extra fields are allowed by Pydantic by default, so use a non-dict
    result = evaluate_json_output('"just a string"')
    assert result["json_valid"] is True
    assert result["schema_valid"] is False


def test_extracted_info_optional_fields():
    info = ExtractedInfo()
    assert info.name is None
    assert info.company is None


def test_extracted_info_partial():
    info = ExtractedInfo(name="Bob", company="Test Corp")
    assert info.name == "Bob"
    assert info.date is None


def test_run_comparison_outputs():
    base = [
        json.dumps(
            {"name": "A", "company": "B", "date": "2025-01-01", "location": "C"}
        ),
        "invalid json",
    ]
    finetuned = [
        json.dumps(
            {"name": "A", "company": "B", "date": "2025-01-01", "location": "C"}
        ),
        json.dumps(
            {"name": "X", "company": "Y", "date": "2025-06-01", "location": "Z"}
        ),
    ]
    report = run_comparison(base, finetuned)
    assert report["test_cases"] > 0
