"""Tests for training configuration files."""

import yaml


def test_sft_config_valid():
    with open("configs/sft_config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "model" in cfg
    assert "name" in cfg["model"]
    assert "lora" in cfg
    assert cfg["lora"]["r"] > 0
    assert cfg["lora"]["lora_alpha"] > 0
    assert "training" in cfg
    assert cfg["training"]["num_epochs"] > 0
    assert cfg["training"]["learning_rate"] > 0
    assert "output" in cfg


def test_dpo_config_valid():
    with open("configs/dpo_config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "model" in cfg
    assert "sft_checkpoint" in cfg["model"]
    assert "dpo" in cfg
    assert 0 < cfg["dpo"]["beta"] <= 1.0
    assert "training" in cfg
    assert "output" in cfg
