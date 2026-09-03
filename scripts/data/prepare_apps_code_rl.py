#!/usr/bin/env python3
"""Prepare APPS rows for response-only Python execution during evaluation.

The input parquet is expected to follow the VERL schema and contain
``data_source='apps'``, chat-formatted prompts, and APPS ``inputs``/``outputs``
inside ``reward_model.ground_truth``. Existing columns are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


INSTRUCTION = (
    "\n\nWrite a correct Python 3 program that reads from standard input and writes "
    "to standard output. Return only the program in a ```python code block; do not "
    "include an explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_messages(value: Any) -> list[dict[str, str]]:
    messages = list(value)
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("Each APPS prompt must end with one user message")
    return [dict(message) for message in messages]


def add_instruction(value: Any) -> list[dict[str, str]]:
    messages = normalize_messages(value)
    content = str(messages[-1].get("content", ""))
    if "Return only the program in a ```python code block" not in content:
        messages[-1]["content"] = content.rstrip() + INSTRUCTION
    return messages


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; pass --overwrite if intended"
        )

    frame = pd.read_parquet(args.input)
    required = {"data_source", "prompt", "reward_model"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if set(frame["data_source"].dropna().unique()) != {"apps"}:
        raise ValueError("This preparer only accepts data_source='apps'")

    for index, reward in enumerate(frame["reward_model"]):
        parsed = json.loads(reward["ground_truth"])
        inputs = parsed.get("inputs")
        outputs = parsed.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise ValueError(f"Row {index} has non-list APPS inputs/outputs")
        if not inputs or len(inputs) != len(outputs):
            raise ValueError(f"Row {index} has invalid APPS inputs/outputs lengths")

    frame = frame.copy()
    frame["prompt"] = frame["prompt"].map(add_instruction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Wrote {len(frame)} APPS rows to {args.output}")


if __name__ == "__main__":
    main()
