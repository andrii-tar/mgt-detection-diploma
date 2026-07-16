from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from interactive_cli import input_int, select_key


def test_select_key_returns_numbered_choice(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: "2")

    choice = select_key("Select model", {"first": "repo/first", "second": "repo/second"})

    assert choice == "second"
    output = capsys.readouterr().out
    assert "1. first -> repo/first" in output
    assert "2. second -> repo/second" in output


def test_input_int_uses_default_on_empty_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    assert input_int("Count per split", default=100) == 100


def test_input_int_rejects_values_below_minimum(monkeypatch):
    responses = iter(["0", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert input_int("Inference batch size", default=2, minimum=1) == 3


def test_select_key_rejects_invalid_choice(monkeypatch):
    responses = iter(["bad", "3", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))

    assert select_key("Select dataset", {"kaggle_train": {}, "outfox_chatgpt_train": {}}) == "kaggle_train"
