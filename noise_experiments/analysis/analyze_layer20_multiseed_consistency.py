#!/usr/bin/env python3
"""Paired difficulty/flip/PPL analysis for the std=0.1 layer20 seed sweep.

The inputs are ModelScope generation parquets.  Every noisy trajectory is
paired with the clean trajectory at the same dataset row and response index.
This script deliberately streams parquet batches: response text makes full
tables too large for ordinary workstation memory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FILES = (
    "aime24-1.0-32-8192--1.parquet",
    "aime25-1.0-32-8192--1.parquet",
    "amc23-1.0-32-8192--1.parquet",
    "math-1.0-32-8192--1.parquet",
    "minerva-1.0-32-8192--1.parquet",
    "olympiad-1.0-32-8192--1.parquet",
)
DEFAULT_REPOS = (
    "r1-1p5b-32-8k-grpo-seed42-noise0.1-layer20-seed1235",
    "r1-1p5b-32-8k-grpo-seed42-noise0.1-layer20-seed1236",
    "r1-1p5b-32-8k-grpo-seed42-noise0.1-layer20-seed1237",
    "r1-1p5b-32-8k-grpo-seed42-noise0.1-layer20-seed1238",
)
TRANSITIONS = ("wrong_to_right", "right_to_wrong", "right_to_right", "wrong_to_wrong")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=root / "result-analysis" / "8k_noise0.2_layer_ranges" / "parquets" / "clean",
    )
    parser.add_argument("--noisy-root", type=Path, default=Path(r"D:\tmp\modelscope-layer20-std0.1-multiseed"))
    parser.add_argument("--repos", nargs="*", default=list(DEFAULT_REPOS))
    parser.add_argument(
        "--layer25-repo",
        default="r1-1p5b-32-8k-grpo-seed42-noise0.1-layer25-seed1238",
        help="Optional matched-seed layer25 repo under --noisy-root; pass an empty string to skip.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\tmp\layer20-multiseed-analysis"))
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def load_math_dapo(repo_root: Path):
    path = repo_root / "verl" / "verl" / "utils" / "reward_score" / "math_dapo.py"
    spec = importlib.util.spec_from_file_location("math_dapo_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{") : pos]
    return None


def extract_number(text: str) -> str | None:
    text = str(text).replace(",", "").replace("\\times", "e").replace("\\cdot", "e")
    text = re.sub(r"\^\{([+-]?\d+)\}", r"\1", text)
    text = re.sub(r"\^([+-]?\d+)", r"\1", text)
    found = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return found.group(0) if found else None


def is_correct(response: str, ground_truth: str, normalizer) -> bool:
    pred = last_boxed(str(response))
    if pred is None:
        return False
    pred_norm = normalizer(pred)
    gt_norm = normalizer(str(ground_truth).strip())
    if gt_norm.endswith(".0"):
        gt_norm = gt_norm[:-2]
    try:
        return math.isclose(float(extract_number(pred_norm)), float(extract_number(gt_norm)), rel_tol=0.05, abs_tol=1e-9)
    except (TypeError, ValueError):
        return pred_norm.replace(",", "") == gt_norm.replace(",", "")


def mean_ppl(token_logprobs) -> float:
    values = np.asarray(token_logprobs or [], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    # Clamp avoids overflow from a malformed trajectory while preserving normal PPL values.
    return float(math.exp(min(30.0, -float(values.mean()))))


def difficulty_bucket(correct_count: int, total: int) -> str:
    if correct_count == 0:
        return f"0/{total}"
    if correct_count == total:
        return f"{total}/{total}"
    fraction = correct_count / total
    if fraction <= 0.25:
        return f"1-25% ({correct_count}/{total})"
    if fraction <= 0.50:
        return f"25-50% ({correct_count}/{total})"
    if fraction <= 0.75:
        return f"50-75% ({correct_count}/{total})"
    return f"75-<100% ({correct_count}/{total})"


def transition(clean_ok: bool, noisy_ok: bool) -> str:
    return ("right" if clean_ok else "wrong") + "_to_" + ("right" if noisy_ok else "wrong")


def analyse_repo(repo: str, args: argparse.Namespace, normalizer) -> tuple[list[dict], list[dict]]:
    trajectory_rows, question_rows = [], []
    for filename in FILES:
        clean_path = args.clean_dir / filename
        noisy_path = args.noisy_root / repo / filename
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean input: {clean_path}")
        if not noisy_path.exists():
            print(f"[multiseed-analysis] skip unavailable {repo}/{filename}", flush=True)
            continue
        clean_pf, noisy_pf = pq.ParquetFile(clean_path), pq.ParquetFile(noisy_path)
        if clean_pf.metadata.num_rows != noisy_pf.metadata.num_rows:
            raise ValueError(f"Row mismatch in {filename}: {clean_pf.metadata.num_rows} vs {noisy_pf.metadata.num_rows}")
        cols = ["data_source", "reward_model", "extra_info", "responses", "chosen_logprobs"]
        clean_batches = clean_pf.iter_batches(batch_size=args.batch_size, columns=cols)
        noisy_batches = noisy_pf.iter_batches(batch_size=args.batch_size, columns=cols)
        row_offset = 0
        batch_index = 0
        print(f"[multiseed-analysis] start {repo}/{filename} ({clean_pf.metadata.num_rows} questions)", flush=True)
        for clean_batch, noisy_batch in zip(clean_batches, noisy_batches, strict=True):
            for clean, noisy in zip(clean_batch.to_pylist(), noisy_batch.to_pylist(), strict=True):
                ground_truth = clean["reward_model"]["ground_truth"]
                clean_responses, noisy_responses = clean["responses"] or [], noisy["responses"] or []
                clean_lps, noisy_lps = clean["chosen_logprobs"] or [], noisy["chosen_logprobs"] or []
                if len(clean_responses) != len(noisy_responses):
                    raise ValueError(f"Trajectory mismatch at {filename}:{row_offset}")
                clean_scores = [is_correct(x, ground_truth, normalizer) for x in clean_responses]
                difficulty = sum(clean_scores)
                total = len(clean_scores)
                bucket = difficulty_bucket(difficulty, total)
                flips = Counter()
                for trajectory_index, (clean_response, noisy_response, clean_ok) in enumerate(
                    zip(clean_responses, noisy_responses, clean_scores, strict=True)
                ):
                    noisy_ok = is_correct(noisy_response, ground_truth, normalizer)
                    label = transition(clean_ok, noisy_ok)
                    flips[label] += 1
                    trajectory_rows.append(
                        {
                            "repo": repo,
                            "dataset": filename.split("-")[0],
                            "row_index": row_offset,
                            "problem_index": (clean.get("extra_info") or {}).get("index", row_offset),
                            "trajectory_index": trajectory_index,
                            "difficulty_correct_count": difficulty,
                            "difficulty_total": total,
                            "difficulty_bucket": bucket,
                            "transition": label,
                            "clean_ppl": mean_ppl(clean_lps[trajectory_index] if trajectory_index < len(clean_lps) else []),
                            "noisy_ppl": mean_ppl(noisy_lps[trajectory_index] if trajectory_index < len(noisy_lps) else []),
                        }
                    )
                question_rows.append(
                    {
                        "repo": repo,
                        "dataset": filename.split("-")[0],
                        "row_index": row_offset,
                        "problem_index": (clean.get("extra_info") or {}).get("index", row_offset),
                        "difficulty_correct_count": difficulty,
                        "difficulty_total": total,
                        "difficulty_bucket": bucket,
                        **{key: flips[key] for key in TRANSITIONS},
                    }
                )
                row_offset += 1
            batch_index += 1
            if batch_index % 10 == 0:
                print(
                    f"[multiseed-analysis] {repo}/{filename}: {row_offset}/{clean_pf.metadata.num_rows} questions",
                    flush=True,
                )
        print(f"[multiseed-analysis] completed {repo}/{filename}", flush=True)
    return trajectory_rows, question_rows


def summarize(trajectory_df: pd.DataFrame, question_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    def rate_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if group_cols:
            pivot = frame.groupby(group_cols + ["transition"], dropna=False).size().unstack(fill_value=0)
        else:
            pivot = pd.DataFrame([frame["transition"].value_counts()]).fillna(0)
        for label in TRANSITIONS:
            if label not in pivot:
                pivot[label] = 0
        pivot["n"] = pivot[list(TRANSITIONS)].sum(axis=1)
        pivot["clean_accuracy"] = (pivot["right_to_right"] + pivot["right_to_wrong"]) / pivot["n"]
        pivot["noisy_accuracy"] = (pivot["right_to_right"] + pivot["wrong_to_right"]) / pivot["n"]
        pivot["delta_accuracy_pp"] = 100 * (pivot["noisy_accuracy"] - pivot["clean_accuracy"])
        pivot["wrong_to_right_rate"] = pivot["wrong_to_right"] / pivot["n"]
        pivot["right_to_wrong_rate"] = pivot["right_to_wrong"] / pivot["n"]
        return pivot.reset_index()

    seed_summary = rate_table(trajectory_df, ["repo"])
    difficulty = rate_table(trajectory_df, ["repo", "difficulty_bucket"])
    ppl = (
        trajectory_df.assign(ppl_delta=lambda x: x["noisy_ppl"] - x["clean_ppl"])
        .groupby(["repo", "transition"], dropna=False)
        .agg(n=("transition", "size"), clean_ppl_mean=("clean_ppl", "mean"), clean_ppl_median=("clean_ppl", "median"), noisy_ppl_mean=("noisy_ppl", "mean"), ppl_delta_mean=("ppl_delta", "mean"))
        .reset_index()
    )
    pooled = rate_table(trajectory_df, [])
    question_overlap = (
        question_df.assign(helped=lambda x: x["wrong_to_right"] > 0, hurt=lambda x: x["right_to_wrong"] > 0)
        .groupby(["dataset", "row_index", "problem_index", "difficulty_bucket"], dropna=False)
        .agg(seed_count=("repo", "nunique"), seeds_helped=("helped", "sum"), seeds_hurt=("hurt", "sum"))
        .reset_index()
    )
    summary = {
        "pooled": pooled.to_dict("records"),
        "question_seed_overlap": {
            "questions": int(len(question_overlap)),
            "helped_in_at_least_one_seed": int((question_overlap["seeds_helped"] > 0).sum()),
            "helped_in_at_least_two_seeds": int((question_overlap["seeds_helped"] >= 2).sum()),
            "hurt_in_at_least_one_seed": int((question_overlap["seeds_hurt"] > 0).sum()),
        },
    }
    return seed_summary, difficulty, ppl, summary


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    normalizer = load_math_dapo(repo_root).normalize_final_answer
    repos = list(args.repos)
    if args.layer25_repo:
        repos.append(args.layer25_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_trajectories, all_questions = [], []
    for repo in repos:
        trajectories, questions = analyse_repo(repo, args, normalizer)
        all_trajectories.extend(trajectories)
        all_questions.extend(questions)
    trajectory_df, question_df = pd.DataFrame(all_trajectories), pd.DataFrame(all_questions)
    seed_summary, difficulty, ppl, summary = summarize(trajectory_df, question_df)
    seed_summary.to_csv(args.output_dir / "seed_summary.csv", index=False)
    difficulty.to_csv(args.output_dir / "difficulty_by_seed.csv", index=False)
    ppl.to_csv(args.output_dir / "ppl_by_transition.csv", index=False)
    question_df.to_csv(args.output_dir / "question_seed_flips.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(seed_summary.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
