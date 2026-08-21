#!/usr/bin/env python3
"""Reaggregate paired flip counts into the pre-registered clean 32-rollout bins."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bucket(correct_count: int, total: int) -> str:
    if total != 32:
        raise ValueError(f"Expected 32 clean rollouts, got {total}")
    if correct_count == 0:
        return "0/32"
    if correct_count <= 7:
        return "1-7/32"
    if correct_count <= 23:
        return "8-23/32"
    if correct_count <= 31:
        return "24-31/32"
    return "32/32"


def aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    counts = ["wrong_to_right", "right_to_wrong", "right_to_right", "wrong_to_wrong"]
    frame = frame.assign(question_key=frame["dataset"].astype(str) + "::" + frame["row_index"].astype(str))
    result = frame.groupby(group_columns, as_index=False).agg(
        questions=("question_key", "nunique"),
        **{name: (name, "sum") for name in counts},
    )
    result["trajectories"] = result[counts].sum(axis=1)
    result["wrong_to_right_rate"] = result["wrong_to_right"] / result["trajectories"]
    result["right_to_wrong_rate"] = result["right_to_wrong"] / result["trajectories"]
    result["delta_accuracy_pp"] = 100 * (result["wrong_to_right"] - result["right_to_wrong"]) / result["trajectories"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df["difficulty_bucket_exact"] = [bucket(c, t) for c, t in zip(df.difficulty_correct_count, df.difficulty_total, strict=True)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_repo = aggregate(df, ["repo", "difficulty_bucket_exact"])
    layer20 = df[df.repo.str.contains("layer20")].copy()
    layer20_pooled = aggregate(layer20, ["difficulty_bucket_exact"])
    layer20_by_dataset = aggregate(layer20, ["dataset", "difficulty_bucket_exact"])
    layer20_by_dataset_seed = aggregate(layer20, ["dataset", "repo", "difficulty_bucket_exact"])
    layer25 = df[df.repo.str.contains("layer25")].copy()
    layer25_pooled = aggregate(layer25, ["difficulty_bucket_exact"])
    per_repo.to_csv(args.output_dir / "difficulty_exact_by_repo.csv", index=False)
    layer20_pooled.to_csv(args.output_dir / "difficulty_exact_layer20_pooled.csv", index=False)
    layer20_by_dataset.to_csv(args.output_dir / "difficulty_exact_layer20_by_dataset.csv", index=False)
    layer20_by_dataset_seed.to_csv(args.output_dir / "difficulty_exact_layer20_by_dataset_seed.csv", index=False)
    layer25_pooled.to_csv(args.output_dir / "difficulty_exact_layer25_seed1238.csv", index=False)
    print("LAYER20 POOLED")
    print(layer20_pooled.to_string(index=False))
    print("\nLAYER25 SEED1238")
    print(layer25_pooled.to_string(index=False))


if __name__ == "__main__":
    main()
