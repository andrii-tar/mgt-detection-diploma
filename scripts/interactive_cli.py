from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def select_key(prompt: str, options: Mapping[str, Any]) -> str:
    keys = list(options)
    if not keys:
        raise ValueError(f"No options available for {prompt}")

    while True:
        print(f"\n{prompt}:")
        for index, key in enumerate(keys, start=1):
            value = options[key]
            suffix = f" -> {value}" if isinstance(value, str) else ""
            print(f"{index}. {key}{suffix}")

        choice = input("Number: ").strip()
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(keys):
                return keys[selected_index - 1]

        print(f"Enter a number from 1 to {len(keys)}.")


def input_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if value == "":
            return default

        try:
            parsed = int(value)
        except ValueError:
            print(f"Enter an integer greater than or equal to {minimum}.")
            continue

        if parsed >= minimum:
            return parsed

        print(f"Enter an integer greater than or equal to {minimum}.")


def optional_choice(prompt: str, choices: list[str]) -> str | None:
    choices_text = "/".join(choices)
    while True:
        value = input(f"{prompt} [{choices_text}, Enter=all]: ").strip()
        if value == "":
            return None

        normalized = value.upper()
        if normalized in choices:
            return normalized

        print(f"Enter one of {choices_text}, or press Enter for all.")
