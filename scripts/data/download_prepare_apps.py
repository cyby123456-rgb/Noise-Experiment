#!/usr/bin/env python3
"""Download APPS from Hugging Face and write a VERL/v8-ready parquet.

The downloader uses streaming by default so a small pilot does not have to
materialize the complete APPS dataset locally. Reference solutions are never
copied into the output; only prompts, executable tests, and metadata are kept.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


STDIN_INSTRUCTION = (
    "\n\nWrite a correct Python 3 program that reads from standard input and writes "
    "to standard output. Return only the complete program in a ```python code "
    "block; do not include an explanation."
)

CALL_BASED_INSTRUCTION = (
    "\n\nWrite a correct Python 3 solution implementing the requested callable. "
    "Return only the complete code in a ```python code block; do not include "
    "an explanation."
)

# The Hub's automatic Parquet conversion currently has these shard counts.
# They are only a fallback when the official datasets-server /parquet endpoint
# is temporarily unavailable.
KNOWN_CONVERTED_SHARDS = {
    ("all", "train"): 1,
    ("all", "test"): 2,
    ("competition", "train"): 1,
    ("competition", "test"): 1,
    ("interview", "train"): 1,
    ("interview", "test"): 2,
    ("introductory", "train"): 1,
    ("introductory", "test"): 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="codeparrot/apps")
    parser.add_argument(
        "--config",
        choices=("all", "introductory", "interview", "competition"),
        default="competition",
        help="APPS difficulty configuration; competition is the hardest subset",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100,
        help="Maximum number of valid rows to write; use 0 for every valid row",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1000,
        help="Streaming shuffle buffer; use 0 to preserve dataset order",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/materialize the selected split before conversion",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_test_cases(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        parsed = dict(value)
    elif isinstance(value, str):
        parsed = json.loads(value)
    else:
        raise TypeError(f"Unsupported input_output type: {type(value).__name__}")

    inputs = parsed.get("inputs")
    outputs = parsed.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("inputs and outputs must both be lists")
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must be non-empty and equally sized")
    return parsed


def build_prompt(row: dict[str, Any], test_cases: dict[str, Any]) -> str:
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("empty question")

    starter_code = str(row.get("starter_code") or "").strip()
    if starter_code:
        question += f"\n\nUse this starter code:\n```python\n{starter_code}\n```"

    if test_cases.get("fn_name"):
        return question + CALL_BASED_INSTRUCTION
    return question + STDIN_INSTRUCTION


def json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def converted_parquet_urls(dataset: str, config: str, split: str) -> list[str]:
    """Resolve Hub-generated Parquet files without executing dataset scripts."""
    api_url = "https://datasets-server.huggingface.co/parquet?" + urlencode(
        {"dataset": dataset}
    )
    try:
        request = Request(api_url, headers={"User-Agent": "noise-experiment-apps-preparer/1"})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        urls = [
            str(item["url"])
            for item in payload.get("parquet_files", [])
            if item.get("config") == config and item.get("split") == split
        ]
        if urls:
            return urls
    except Exception as exc:
        print(
            f"Warning: datasets-server Parquet discovery failed ({type(exc).__name__}: {exc}); "
            "using the known refs/convert/parquet layout.",
            flush=True,
        )

    shard_count = KNOWN_CONVERTED_SHARDS.get((config, split))
    if shard_count is None or dataset != "codeparrot/apps":
        raise RuntimeError(
            f"Could not resolve converted Parquet files for {dataset}/{config}/{split}"
        )
    revision = "refs%2Fconvert%2Fparquet"
    return [
        (
            f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/"
            f"{config}/{split}/{index:04d}.parquet"
        )
        for index in range(shard_count)
    ]


def main() -> None:
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")
    if args.shuffle_buffer < 0:
        raise ValueError("--shuffle-buffer must be non-negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; pass --overwrite if intended"
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face 'datasets' package is required: pip install datasets"
        ) from exc

    streaming = not args.no_streaming
    print(
        f"Loading {args.dataset}/{args.config} split={args.split} "
        f"streaming={streaming}",
        flush=True,
    )
    # ``codeparrot/apps`` still exposes a legacy apps.py loader on its main
    # branch. New releases of ``datasets`` reject dataset scripts. Resolve the
    # Hub-generated Parquet conversion explicitly and invoke only the built-in
    # Parquet loader, which works with both old and new ``datasets`` versions.
    parquet_urls = converted_parquet_urls(args.dataset, args.config, args.split)
    print(
        "Reading converted Parquet shards:\n  " + "\n  ".join(parquet_urls),
        flush=True,
    )
    dataset = load_dataset(
        "parquet",
        data_files={args.split: parquet_urls},
        split=args.split,
        streaming=streaming,
    )
    if args.shuffle_buffer:
        if streaming:
            dataset = dataset.shuffle(
                seed=args.seed,
                buffer_size=args.shuffle_buffer,
            )
        else:
            dataset = dataset.shuffle(seed=args.seed)

    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for source_index, raw_row in enumerate(dataset):
        row = dict(raw_row)
        try:
            test_cases = parse_test_cases(row.get("input_output"))
            prompt = build_prompt(row, test_cases)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            exclusions[type(exc).__name__] += 1
            continue

        rows.append(
            {
                "data_source": "apps",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "Code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(test_cases, ensure_ascii=False),
                },
                "extra_info": {
                    "split": args.split,
                    "source_index": source_index,
                    "problem_id": json_scalar(row.get("problem_id")),
                    "difficulty": json_scalar(row.get("difficulty")),
                    "url": json_scalar(row.get("url")),
                    "dataset": args.dataset,
                    "config": args.config,
                },
            }
        )
        if args.max_rows and len(rows) >= args.max_rows:
            break

    if not rows:
        raise RuntimeError(f"No valid APPS rows found; exclusions={dict(exclusions)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_parquet(args.output, index=False)
    print(
        f"Wrote {len(rows)} rows to {args.output}; exclusions={dict(exclusions)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
