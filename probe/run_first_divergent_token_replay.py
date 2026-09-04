#!/usr/bin/env python3
"""Run first-divergent-token counterfactual replays on saved V8 trials.

The source is one or more batch-1 greedy Gaussian probe directories.  For each
selected W2R trial and a matched changed-but-still-wrong trial from the same
question and injection position, this collector first reproduces the saved
clean and noisy trajectories exactly.  It then runs, for each forced-prefix
length k, the two nontrivial cells of a State x Token intervention:

  * clean internal state + source-trial divergent prefix (token sufficiency)
  * noisy internal state + clean prefix (token necessity/state persistence)

The clean/noisy state is rebuilt by replaying from the original prompt with a
zero/the exact saved Gaussian vector.  We never deserialize and splice a KV
cache.  This preserves the original intervention semantics and lets the model
recompute all downstream KV entries from the counterfactual token prefix.

This implementation is deliberately strict batch-1.  Parallel V9 code shards
must not be mixed with it because changing their execution batch shape can
change the numerical baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = Path(os.environ.get("VERL_ROOT", EXPERIMENT_ROOT / "verl")).resolve()
sys.path.insert(0, str(VERL_ROOT))

from verl.models.transformers.noise_injection import _get_decoder_layers
from verl.trainer.main_ppo import _select_rm_score_fn


FORMAT_VERSION = "first-divergent-token-replay-v1-strict-batch1-state-x-token"
CODE_ALIASES = {
    "train-code-taco-medium": "taco",
    "train-code-taco-hard": "taco",
    "train-code-taco-medium_hard": "taco",
}
CODE_SOURCES = {"taco", "codecontests", "apps", "codeforces"}


@dataclass(frozen=True)
class TrialLocator:
    source_dir: str
    shard_path: str
    source_format_version: str
    trial_index: int
    label: str
    row_index: int
    question_index: int
    problem_index: Any
    question_fingerprint: str
    data_source: str
    injection_layer: int
    response_position: int
    clean_response_length: int
    source_response_length: int
    first_divergence_index: int
    first_divergence_position: int
    divergence_delay_after_first_changeable: int
    immediate_next_token_change: bool
    source_score: float
    baseline_score: float
    noise_seed: int
    noise_sha256: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.shard_path, self.trial_index)

    @property
    def group_key(self) -> str:
        # One source shard is exactly one question x injection-position group.
        return self.shard_path


class ForceResponsePrefix(LogitsProcessor):
    """Force a token sequence at absolute zero-based response indices."""

    def __init__(self, prompt_length: int, start_index: int, token_ids: list[int]):
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        self.prompt_length = int(prompt_length)
        self.start_index = int(start_index)
        self.token_ids = [int(token_id) for token_id in token_ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        response_index = int(input_ids.shape[1] - self.prompt_length)
        offset = response_index - self.start_index
        if 0 <= offset < len(self.token_ids):
            forced_token = self.token_ids[offset]
            if not 0 <= forced_token < scores.shape[-1]:
                raise IndexError(
                    f"Forced token {forced_token} is outside vocabulary size {scores.shape[-1]}"
                )
            forced_scores = torch.full_like(scores, -torch.inf)
            forced_scores[:, forced_token] = 0
            return forced_scores
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "A completed strict batch-1 V8 probe directory containing manifest.jsonl and "
            "tensor shards. Repeat this argument to pool multiple datasets/directories."
        ),
    )
    parser.add_argument(
        "--model",
        default="",
        help="Checkpoint path. If omitted, all source shards must name the same model path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-lengths", default="1,4,16,64")
    parser.add_argument(
        "--min-unforced-source-tokens",
        type=int,
        default=128,
        help=(
            "Require this many source tokens after the longest forced prefix in both clean "
            "and source trajectories. This prevents forcing a prefix that reaches the answer/end."
        ),
    )
    parser.add_argument(
        "--max-w2r-trials",
        type=int,
        default=50,
        help="Maximum matched W2R episodes; 0 means all eligible matched W2R episodes.",
    )
    parser.add_argument("--selection-seed", type=int, default=20260904)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=None,
        help="Defaults to the model_dtype stored in the source shards.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Allow reward scoring to execute generated code. Use only inside an isolated container.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def parse_positive_int_list(value: str, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicates")
    return sorted(parsed)


def as_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return as_python(value.as_py())
    if isinstance(value, dict):
        return {str(key): as_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_python(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return as_python(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(as_python(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_shard(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        # mmap and weights_only are unavailable in older supported PyTorch versions.
        return torch.load(path, map_location="cpu")


def tensor_to_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]


def tensor_to_bool_list(value: Any) -> list[bool]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return [bool(item) for item in value]


def tensor_to_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]


def first_divergence(clean: list[int], other: list[int]) -> int | None:
    for index, (clean_token, other_token) in enumerate(zip(clean, other)):
        if clean_token != other_token:
            return index
    if len(clean) != len(other):
        return min(len(clean), len(other))
    return None


def is_code_data_source(data_source: Any) -> bool:
    source = str(data_source)
    canonical = CODE_ALIASES.get(source, source)
    return canonical.startswith("livecodebench/") or canonical in CODE_SOURCES


def score_response(record: dict[str, Any], response: str) -> float:
    score_fn = _select_rm_score_fn(str(record["data_source"]), code_continuous=False)
    result = score_fn(
        solution_str=response,
        ground_truth=record["reward_model"]["ground_truth"],
    )
    if isinstance(result, dict):
        result = result.get("score", 0.0)
    return float(result)


def response_record(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_source": metadata["data_source"],
        "reward_model": {"ground_truth": metadata["ground_truth"]},
    }


def source_execution_batch_size(metadata: dict[str, Any]) -> int:
    if "execution_batch_size_including_control" in metadata:
        return int(metadata["execution_batch_size_including_control"])
    return int(metadata.get("trial_batch_size", 1))


def resolve_tensor_path(source_dir: Path, manifest_record: dict[str, Any]) -> Path:
    raw = Path(str(manifest_record["tensor_path"]))
    path = raw if raw.is_absolute() else source_dir / raw
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source tensor shard not found: {path}")
    return path


def scan_source_dirs(
    source_dirs: list[Path],
    max_prefix_length: int,
    min_unforced_source_tokens: int,
    allow_code_execution: bool,
) -> tuple[list[TrialLocator], dict[str, Any]]:
    locators: list[TrialLocator] = []
    seen_shards: set[Path] = set()
    model_paths: set[str] = set()
    model_dtypes: set[str] = set()
    counts: Counter[str] = Counter()

    for raw_source_dir in source_dirs:
        source_dir = raw_source_dir.resolve()
        manifest_path = source_dir / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Source manifest not found: {manifest_path}")

        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            manifest_record = json.loads(line)
            shard_path = resolve_tensor_path(source_dir, manifest_record)
            if shard_path in seen_shards:
                raise ValueError(f"Duplicate source shard supplied: {shard_path}")
            seen_shards.add(shard_path)
            shard = load_shard(shard_path)
            metadata = dict(shard["metadata"])
            source_format = str(shard.get("format_version", ""))

            execution_batch_size = source_execution_batch_size(metadata)
            if execution_batch_size != 1:
                raise ValueError(
                    "First-divergent-token replay currently requires strict batch-1 source shards, "
                    f"but {shard_path} used execution batch size {execution_batch_size}. "
                    "Do not replay V9 parallel code results with a different batch shape."
                )
            data_source = str(metadata["data_source"])
            if is_code_data_source(data_source) and not allow_code_execution:
                raise ValueError(
                    f"Source {shard_path} is a code task. Re-run with --allow-code-execution only "
                    "inside an isolated container."
                )

            model_paths.add(str(metadata["model"]))
            model_dtypes.add(str(metadata.get("model_dtype", "bfloat16")))
            clean_ids = tensor_to_int_list(shard["clean_response_token_ids"])
            noisy_ids_all = [tensor_to_int_list(tokens) for tokens in shard["noisy_response_token_ids"]]
            noisy_scores = tensor_to_float_list(shard["noisy_scores"])
            is_w2r = tensor_to_bool_list(shard["is_w2r"])
            noise_seeds = tensor_to_int_list(shard["noise_seeds"])
            noise_hashes = [str(value) for value in shard["noise_sha256"]]

            trial_count = len(noisy_ids_all)
            aligned_lengths = {
                trial_count,
                len(noisy_scores),
                len(is_w2r),
                len(noise_seeds),
                len(noise_hashes),
                int(shard["sampled_noise"].shape[0]),
                int(shard["applied_noise"].shape[0]),
            }
            if len(aligned_lengths) != 1:
                raise ValueError(
                    f"Misaligned per-trial arrays in {shard_path}: lengths={sorted(aligned_lengths)}"
                )
            baseline_score = float(shard["baseline_score"])
            if not math.isfinite(baseline_score) or baseline_score > 0:
                raise ValueError(f"Source shard does not have a finite wrong baseline: {shard_path}")

            response_position = int(metadata["response_position"])
            for trial_index, (source_ids, score, marked_w2r) in enumerate(
                zip(noisy_ids_all, noisy_scores, is_w2r)
            ):
                counts["trials_scanned"] += 1
                if not math.isfinite(score):
                    counts["non_finite_score"] += 1
                    continue
                divergence_index = first_divergence(clean_ids, source_ids)
                if divergence_index is None:
                    counts["unchanged"] += 1
                    continue
                if divergence_index < response_position:
                    raise ValueError(
                        f"Source trial diverges before noise can act in {shard_path}, "
                        f"trial={trial_index}: divergence_index={divergence_index}, "
                        f"response_position={response_position}"
                    )

                score_is_w2r = baseline_score <= 0 and score > 0
                if bool(marked_w2r) != bool(score_is_w2r):
                    raise ValueError(
                        f"Stored W2R flag disagrees with score in {shard_path}, trial={trial_index}"
                    )
                label = "w2r" if score_is_w2r else "changed_wrong"
                counts[label] += 1

                remaining_clean = len(clean_ids) - (divergence_index + max_prefix_length)
                remaining_source = len(source_ids) - (divergence_index + max_prefix_length)
                if min(remaining_clean, remaining_source) < min_unforced_source_tokens:
                    counts[f"{label}_answer_leakage_guard"] += 1
                    continue
                counts[f"{label}_eligible"] += 1

                locators.append(
                    TrialLocator(
                        source_dir=str(source_dir),
                        shard_path=str(shard_path),
                        source_format_version=source_format,
                        trial_index=trial_index,
                        label=label,
                        row_index=int(metadata["row_index"]),
                        question_index=int(metadata["question_index"]),
                        problem_index=as_python(metadata.get("problem_index")),
                        question_fingerprint=str(metadata["question_fingerprint"]),
                        data_source=data_source,
                        injection_layer=int(metadata["injection_layer"]),
                        response_position=response_position,
                        clean_response_length=len(clean_ids),
                        source_response_length=len(source_ids),
                        first_divergence_index=divergence_index,
                        first_divergence_position=divergence_index + 1,
                        divergence_delay_after_first_changeable=divergence_index - response_position,
                        immediate_next_token_change=divergence_index == response_position,
                        source_score=score,
                        baseline_score=baseline_score,
                        noise_seed=int(noise_seeds[trial_index]),
                        noise_sha256=noise_hashes[trial_index],
                    )
                )
            del shard

    if not seen_shards:
        raise RuntimeError("No tensor shards were found in the supplied source directories")
    scan_metadata = {
        "source_directories": [str(path.resolve()) for path in source_dirs],
        "source_shards": len(seen_shards),
        "source_model_paths": sorted(model_paths),
        "source_model_dtypes": sorted(model_dtypes),
        "counts": dict(counts),
    }
    return locators, scan_metadata


def match_trials(
    locators: list[TrialLocator],
    max_w2r_trials: int,
    selection_seed: int,
) -> tuple[list[tuple[TrialLocator, TrialLocator]], dict[str, Any]]:
    w2r_by_group: dict[str, list[TrialLocator]] = defaultdict(list)
    wrong_by_group: dict[str, list[TrialLocator]] = defaultdict(list)
    for locator in locators:
        if locator.label == "w2r":
            w2r_by_group[locator.group_key].append(locator)
        elif locator.label == "changed_wrong":
            wrong_by_group[locator.group_key].append(locator)

    rng = random.Random(selection_seed)
    # Round-robin across question-position groups before taking a second W2R
    # from any group. This prevents a high-correctability position from
    # dominating a small pilot while still allowing all episodes when desired.
    group_keys = sorted(w2r_by_group)
    rng.shuffle(group_keys)
    for group_locators in w2r_by_group.values():
        rng.shuffle(group_locators)
    ordered_w2r: list[TrialLocator] = []
    round_index = 0
    while True:
        added = False
        for group_key in group_keys:
            group_locators = w2r_by_group[group_key]
            if round_index < len(group_locators):
                ordered_w2r.append(group_locators[round_index])
                added = True
        if not added:
            break
        round_index += 1

    used_wrong: set[tuple[str, int]] = set()
    pairs: list[tuple[TrialLocator, TrialLocator]] = []
    unmatched = 0

    for w2r_locator in ordered_w2r:
        pool = [
            candidate
            for candidate in wrong_by_group.get(w2r_locator.group_key, [])
            if candidate.key not in used_wrong
        ]
        if not pool:
            unmatched += 1
            continue
        matched_wrong = min(
            pool,
            key=lambda candidate: (
                int(candidate.immediate_next_token_change != w2r_locator.immediate_next_token_change),
                abs(
                    candidate.divergence_delay_after_first_changeable
                    - w2r_locator.divergence_delay_after_first_changeable
                ),
                abs(candidate.source_response_length - w2r_locator.source_response_length),
                candidate.trial_index,
            ),
        )
        used_wrong.add(matched_wrong.key)
        pairs.append((w2r_locator, matched_wrong))
        if max_w2r_trials > 0 and len(pairs) >= max_w2r_trials:
            break

    if not pairs:
        raise RuntimeError(
            "No eligible W2R trial had an eligible changed-wrong match in the same question/position"
        )
    metadata = {
        "eligible_w2r": len(ordered_w2r),
        "eligible_w2r_question_position_groups": len(w2r_by_group),
        "eligible_changed_wrong": sum(len(items) for items in wrong_by_group.values()),
        "matched_pairs": len(pairs),
        "unmatched_w2r_encountered": unmatched,
        "selection_seed": selection_seed,
        "w2r_selection_order": (
            "seeded round-robin across question-position groups, then additional trials per group"
        ),
        "matching": (
            "same source shard (question and position), without replacement; minimize immediate-change "
            "mismatch, divergence-delay distance, then response-length distance"
        ),
    }
    return pairs, metadata


def _layer_hidden(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError(
            "Expected decoder layer output [batch, sequence, hidden], "
            f"got {type(hidden).__name__} shape={getattr(hidden, 'shape', None)}"
        )
    return hidden


def register_batch1_noise_hook(
    model,
    layer_idx: int,
    response_position: int,
    sampled_noise: torch.Tensor,
) -> tuple[Any, dict[str, Any]]:
    if sampled_noise.ndim != 1 or sampled_noise.dtype != torch.float32:
        raise ValueError("sampled_noise must be a CPU float32 vector [hidden]")
    if sampled_noise.device.type != "cpu":
        raise ValueError("sampled_noise must remain on CPU until the target hook fires")
    layers = _get_decoder_layers(model)
    if not 0 <= layer_idx < len(layers):
        raise IndexError(f"injection layer {layer_idx} outside decoder depth {len(layers)}")

    capture: dict[str, Any] = {}
    forward_call_index = -1

    def hook(module, inputs, output):
        nonlocal forward_call_index
        forward_call_index += 1
        if forward_call_index != response_position:
            return output
        hidden = _layer_hidden(output)
        if hidden.shape[0] != 1:
            raise RuntimeError(f"Branch replay requires batch size 1, got {hidden.shape[0]}")
        clean_token = hidden[:, -1, :]
        if clean_token.shape[-1] != sampled_noise.shape[0]:
            raise ValueError(
                f"Noise hidden size {sampled_noise.shape[0]} != model hidden size {clean_token.shape[-1]}"
            )
        noise_device = sampled_noise.to(device=hidden.device, dtype=torch.float32)
        modified_token = (clean_token.float() + noise_device.unsqueeze(0)).to(hidden.dtype)
        modified = hidden.clone()
        modified[:, -1, :] = modified_token
        capture["pre_noise_hidden_state"] = clean_token[0].detach().float().cpu()
        capture["sampled_noise"] = sampled_noise.clone()
        capture["applied_noise"] = (
            modified_token.float() - clean_token.float()
        )[0].detach().cpu()
        capture["target_forward_call_index"] = forward_call_index
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    return layers[layer_idx].register_forward_hook(hook), capture


def eos_token_ids_for_v8(tokenizer) -> list[int]:
    # The strict V8 source collector passed tokenizer.eos_token_id to generate.
    raw = tokenizer.eos_token_id
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    result = [int(value) for value in values if value is not None]
    if not result:
        raise ValueError("Tokenizer has no EOS token ID")
    return result


@torch.inference_mode()
def run_replay(
    *,
    model,
    tokenizer,
    prompt_token_ids: list[int],
    response_position: int,
    layer_idx: int,
    sampled_noise: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    forced_start_index: int | None = None,
    forced_token_ids: list[int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if (forced_start_index is None) != (forced_token_ids is None):
        raise ValueError("forced_start_index and forced_token_ids must be provided together")
    input_tensor = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_tensor)
    logits_processor = LogitsProcessorList()
    if forced_token_ids is not None:
        logits_processor.append(
            ForceResponsePrefix(
                prompt_length=len(prompt_token_ids),
                start_index=int(forced_start_index),
                token_ids=forced_token_ids,
            )
        )

    handle, capture = register_batch1_noise_hook(
        model=model,
        layer_idx=layer_idx,
        response_position=response_position,
        sampled_noise=sampled_noise,
    )
    try:
        generated = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            logits_processor=logits_processor,
        )
    finally:
        handle.remove()

    if capture.get("target_forward_call_index") != response_position:
        raise RuntimeError(
            f"Noise hook did not fire at response position {response_position}; "
            f"capture={capture.keys()}"
        )

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    eos_ids = set(eos_token_ids_for_v8(tokenizer))
    pad_id = tokenizer.pad_token_id
    response: list[int] = []
    for token_id in sequences[0, input_tensor.shape[1] :].tolist():
        if token_id in eos_ids or (pad_id is not None and token_id == pad_id):
            break
        response.append(int(token_id))
    return response, capture


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def noise_sha256(noise: torch.Tensor) -> str:
    array = noise.detach().cpu().contiguous().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def score_generated(
    tokenizer,
    record: dict[str, Any],
    token_ids: list[int],
) -> tuple[str, float, bool]:
    response = tokenizer.decode(token_ids, skip_special_tokens=True)
    score = score_response(record, response)
    if not math.isfinite(score):
        raise RuntimeError("Counterfactual response received a non-finite score")
    return response, score, score > 0


def verify_capture(
    capture: dict[str, Any],
    expected_clean_hidden: torch.Tensor,
    expected_applied_noise: torch.Tensor,
    context: str,
) -> dict[str, float]:
    pre_diff = max_abs_difference(capture["pre_noise_hidden_state"], expected_clean_hidden)
    applied_diff = max_abs_difference(capture["applied_noise"], expected_applied_noise)
    if pre_diff != 0 or applied_diff != 0:
        raise RuntimeError(
            f"{context}: replay did not reconstruct the saved intervention state exactly: "
            f"pre_hidden_max_diff={pre_diff}, applied_noise_max_diff={applied_diff}"
        )
    return {
        "pre_noise_hidden_max_abs_diff": pre_diff,
        "applied_noise_max_abs_diff": applied_diff,
    }


def forced_prefix_is_present(
    generated: list[int],
    start_index: int,
    forced: list[int],
) -> bool:
    return generated[start_index : start_index + len(forced)] == forced


def compact_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": run["score"],
        "is_correct": run["is_correct"],
        "response_length": len(run["token_ids"]),
        "first_divergence_from_clean_index": run["first_divergence_from_clean_index"],
        "forced_prefix_verified": run.get("forced_prefix_verified"),
        "pre_noise_hidden_max_abs_diff": run["diagnostics"][
            "pre_noise_hidden_max_abs_diff"
        ],
        "applied_noise_max_abs_diff": run["diagnostics"]["applied_noise_max_abs_diff"],
    }


def make_run_payload(
    *,
    tokenizer,
    record: dict[str, Any],
    token_ids: list[int],
    clean_ids: list[int],
    capture: dict[str, Any],
    expected_clean_hidden: torch.Tensor,
    expected_applied_noise: torch.Tensor,
    context: str,
    forced_start_index: int | None = None,
    forced_token_ids: list[int] | None = None,
) -> dict[str, Any]:
    response, score, is_correct = score_generated(tokenizer, record, token_ids)
    payload = {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "response": response,
        "score": score,
        "is_correct": is_correct,
        "first_divergence_from_clean_index": first_divergence(clean_ids, token_ids),
        "diagnostics": verify_capture(
            capture,
            expected_clean_hidden,
            expected_applied_noise,
            context,
        ),
    }
    if forced_token_ids is not None:
        verified = forced_prefix_is_present(token_ids, int(forced_start_index), forced_token_ids)
        if not verified:
            raise RuntimeError(f"{context}: generated sequence does not contain the forced prefix")
        payload["forced_prefix_verified"] = True
    return payload


def execute_episode(
    *,
    locator: TrialLocator,
    pair_id: int,
    shard: dict[str, Any],
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    prefix_lengths: list[int],
    clean_validation_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(shard["metadata"])
    prompt_ids = tensor_to_int_list(shard["prompt_token_ids"])
    clean_ids = tensor_to_int_list(shard["clean_response_token_ids"])
    source_ids = tensor_to_int_list(shard["noisy_response_token_ids"][locator.trial_index])
    sampled_noise = shard["sampled_noise"][locator.trial_index].detach().cpu().float().clone()
    saved_applied_noise = shard["applied_noise"][locator.trial_index].detach().cpu().float().clone()
    clean_hidden = shard["clean_hidden_state"].detach().cpu().float().clone()
    zero_noise = torch.zeros_like(sampled_noise)
    zero_applied = torch.zeros_like(saved_applied_noise)
    record = response_record(metadata)

    actual_hash = noise_sha256(sampled_noise)
    if actual_hash != locator.noise_sha256:
        raise RuntimeError(
            f"Saved noise hash mismatch in {locator.shard_path}, trial={locator.trial_index}: "
            f"manifest={locator.noise_sha256}, tensor={actual_hash}"
        )

    if locator.shard_path not in clean_validation_cache:
        clean_replay_ids, clean_capture = run_replay(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_ids,
            response_position=locator.response_position,
            layer_idx=locator.injection_layer,
            sampled_noise=zero_noise,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        if clean_replay_ids != clean_ids:
            raise RuntimeError(
                f"Clean replay is not token-identical for source shard {locator.shard_path}. "
                "Check model, dtype, max_new_tokens, software version, and batch size."
            )
        clean_payload = make_run_payload(
            tokenizer=tokenizer,
            record=record,
            token_ids=clean_replay_ids,
            clean_ids=clean_ids,
            capture=clean_capture,
            expected_clean_hidden=clean_hidden,
            expected_applied_noise=zero_applied,
            context=f"clean replay {locator.shard_path}",
        )
        if clean_payload["is_correct"]:
            raise RuntimeError(f"Saved clean-wrong baseline rescored as correct: {locator.shard_path}")
        clean_validation_cache[locator.shard_path] = clean_payload

    noisy_replay_ids, noisy_capture = run_replay(
        model=model,
        tokenizer=tokenizer,
        prompt_token_ids=prompt_ids,
        response_position=locator.response_position,
        layer_idx=locator.injection_layer,
        sampled_noise=sampled_noise,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    if noisy_replay_ids != source_ids:
        mismatch = first_divergence(source_ids, noisy_replay_ids)
        raise RuntimeError(
            f"Noisy replay is not token-identical for {locator.shard_path}, "
            f"trial={locator.trial_index}, first mismatch={mismatch}. "
            "Counterfactual results would not be valid."
        )
    noisy_payload = make_run_payload(
        tokenizer=tokenizer,
        record=record,
        token_ids=noisy_replay_ids,
        clean_ids=clean_ids,
        capture=noisy_capture,
        expected_clean_hidden=clean_hidden,
        expected_applied_noise=saved_applied_noise,
        context=f"noisy replay {locator.shard_path} trial={locator.trial_index}",
    )
    if noisy_payload["is_correct"] != (locator.label == "w2r"):
        raise RuntimeError(
            f"Replayed correctness disagrees with source label for {locator.shard_path}, "
            f"trial={locator.trial_index}"
        )

    by_k: dict[str, Any] = {}
    compact_by_k: dict[str, Any] = {}
    for prefix_length in prefix_lengths:
        start = locator.first_divergence_index
        source_prefix = source_ids[start : start + prefix_length]
        clean_prefix = clean_ids[start : start + prefix_length]
        if len(source_prefix) != prefix_length or len(clean_prefix) != prefix_length:
            raise RuntimeError("Answer-leakage eligibility check failed to retain a full prefix")

        target_on_clean_ids, target_on_clean_capture = run_replay(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_ids,
            response_position=locator.response_position,
            layer_idx=locator.injection_layer,
            sampled_noise=zero_noise,
            max_new_tokens=max_new_tokens,
            device=device,
            forced_start_index=start,
            forced_token_ids=source_prefix,
        )
        target_on_clean = make_run_payload(
            tokenizer=tokenizer,
            record=record,
            token_ids=target_on_clean_ids,
            clean_ids=clean_ids,
            capture=target_on_clean_capture,
            expected_clean_hidden=clean_hidden,
            expected_applied_noise=zero_applied,
            context=(
                f"target_on_clean pair={pair_id} label={locator.label} k={prefix_length}"
            ),
            forced_start_index=start,
            forced_token_ids=source_prefix,
        )

        clean_on_noisy_ids, clean_on_noisy_capture = run_replay(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_ids,
            response_position=locator.response_position,
            layer_idx=locator.injection_layer,
            sampled_noise=sampled_noise,
            max_new_tokens=max_new_tokens,
            device=device,
            forced_start_index=start,
            forced_token_ids=clean_prefix,
        )
        clean_on_noisy = make_run_payload(
            tokenizer=tokenizer,
            record=record,
            token_ids=clean_on_noisy_ids,
            clean_ids=clean_ids,
            capture=clean_on_noisy_capture,
            expected_clean_hidden=clean_hidden,
            expected_applied_noise=saved_applied_noise,
            context=(
                f"clean_on_noisy pair={pair_id} label={locator.label} k={prefix_length}"
            ),
            forced_start_index=start,
            forced_token_ids=clean_prefix,
        )

        by_k[str(prefix_length)] = {
            "prefix_length": prefix_length,
            "forced_start_index": start,
            "forced_start_position": start + 1,
            "unforced_source_tokens_after_prefix": min(
                len(clean_ids), len(source_ids)
            ) - (start + prefix_length),
            "source_prefix_token_ids": torch.tensor(source_prefix, dtype=torch.long),
            "clean_prefix_token_ids": torch.tensor(clean_prefix, dtype=torch.long),
            "target_on_clean_state": target_on_clean,
            "clean_on_noisy_state": clean_on_noisy,
        }
        compact_by_k[str(prefix_length)] = {
            "target_on_clean_state": compact_run(target_on_clean),
            "clean_on_noisy_state": compact_run(clean_on_noisy),
        }

    episode = {
        "format_version": FORMAT_VERSION,
        "pair_id": pair_id,
        "label": locator.label,
        "source": asdict(locator),
        "prompt_token_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "clean_response_token_ids": torch.tensor(clean_ids, dtype=torch.long),
        "source_response_token_ids": torch.tensor(source_ids, dtype=torch.long),
        "clean_response": shard["baseline_response"],
        "source_response": shard["noisy_responses"][locator.trial_index],
        "clean_hidden_state": clean_hidden,
        "sampled_noise": sampled_noise,
        "applied_noise": saved_applied_noise,
        "clean_unforced_replay": clean_validation_cache[locator.shard_path],
        "noisy_unforced_replay": noisy_payload,
        "by_prefix_length": by_k,
    }
    manifest_record = {
        "format_version": FORMAT_VERSION,
        "pair_id": pair_id,
        "label": locator.label,
        "source": asdict(locator),
        "clean_reproduction_exact": True,
        "noisy_reproduction_exact": True,
        "by_prefix_length": compact_by_k,
    }
    return episode, manifest_record


def aggregate_results(rows: list[dict[str, Any]], prefix_lengths: list[int]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for label in ("w2r", "changed_wrong"):
        label_rows = [row for row in rows if row["label"] == label]
        episode_keys = {(row["pair_id"], row["label"]) for row in label_rows}
        aggregate[label] = {"episodes": len(episode_keys), "by_prefix_length": {}}
        for prefix_length in prefix_lengths:
            k_rows = [row for row in label_rows if row["prefix_length"] == prefix_length]
            n = len(k_rows)
            target_correct = sum(bool(row["target_on_clean_is_correct"]) for row in k_rows)
            clean_on_noisy_correct = sum(
                bool(row["clean_on_noisy_is_correct"]) for row in k_rows
            )
            aggregate[label]["by_prefix_length"][str(prefix_length)] = {
                "n": n,
                "target_on_clean_correct": target_correct,
                "target_on_clean_correct_rate": target_correct / n if n else None,
                "clean_on_noisy_correct": clean_on_noisy_correct,
                "clean_on_noisy_correct_rate": clean_on_noisy_correct / n if n else None,
                "conditional_token_necessity_rate": (
                    (n - clean_on_noisy_correct) / n if n and label == "w2r" else None
                ),
            }
    return aggregate


def resolve_model_and_dtype(
    args: argparse.Namespace,
    scan_metadata: dict[str, Any],
) -> tuple[str, str]:
    source_models = scan_metadata["source_model_paths"]
    if args.model:
        model_path = args.model
    elif len(source_models) == 1:
        model_path = source_models[0]
    else:
        raise ValueError(
            "--model is required when source shards name zero or multiple checkpoint paths: "
            f"{source_models}"
        )
    source_dtypes = scan_metadata["source_model_dtypes"]
    if args.dtype:
        dtype = args.dtype
    elif len(source_dtypes) == 1:
        dtype = source_dtypes[0]
    else:
        raise ValueError(
            "--dtype is required when source shards contain multiple model dtypes: "
            f"{source_dtypes}"
        )
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported resolved dtype: {dtype}")
    return model_path, dtype


def main() -> None:
    args = parse_args()
    prefix_lengths = parse_positive_int_list(args.prefix_lengths, "--prefix-lengths")
    if args.min_unforced_source_tokens < 0:
        raise ValueError("--min-unforced-source-tokens must be non-negative")
    if args.max_w2r_trials < 0:
        raise ValueError("--max-w2r-trials must be non-negative")
    if args.max_new_tokens <= 0 or args.log_every <= 0:
        raise ValueError("--max-new-tokens and --log-every must be positive")

    args.output_dir = args.output_dir.resolve()
    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing replay manifest: {manifest_path}"
        )

    print("[branch-replay] scanning source shards", flush=True)
    locators, scan_metadata = scan_source_dirs(
        source_dirs=args.source_dir,
        max_prefix_length=max(prefix_lengths),
        min_unforced_source_tokens=args.min_unforced_source_tokens,
        allow_code_execution=args.allow_code_execution,
    )
    pairs, matching_metadata = match_trials(
        locators,
        max_w2r_trials=args.max_w2r_trials,
        selection_seed=args.selection_seed,
    )
    model_path, dtype_name = resolve_model_and_dtype(args, scan_metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = args.output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "selection.json",
        {
            "format_version": FORMAT_VERSION,
            "scan": scan_metadata,
            "matching": matching_metadata,
            "pairs": [
                {"pair_id": index, "w2r": asdict(w2r), "changed_wrong": asdict(wrong)}
                for index, (w2r, wrong) in enumerate(pairs)
            ],
        },
    )

    device = torch.device(args.device)
    dtype = getattr(torch, dtype_name)
    print(f"[branch-replay] loading tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[branch-replay] loading model on {device} ({dtype_name})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    started_at = time.monotonic()
    clean_validation_cache: dict[str, dict[str, Any]] = {}
    current_shard_path: str | None = None
    current_shard: dict[str, Any] | None = None
    result_rows: list[dict[str, Any]] = []
    episode_count = 0

    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        for pair_id, pair in enumerate(pairs):
            for locator in pair:
                if locator.shard_path != current_shard_path:
                    current_shard = load_shard(Path(locator.shard_path))
                    current_shard_path = locator.shard_path
                assert current_shard is not None
                episode, manifest_record = execute_episode(
                    locator=locator,
                    pair_id=pair_id,
                    shard=current_shard,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    prefix_lengths=prefix_lengths,
                    clean_validation_cache=clean_validation_cache,
                )
                episode_name = f"pair_{pair_id:04d}_{locator.label}.pt"
                episode_path = episode_dir / episode_name
                if episode_path.exists():
                    raise FileExistsError(f"Refusing to overwrite episode: {episode_path}")
                torch.save(episode, episode_path)
                manifest_record["episode_path"] = str(episode_path.relative_to(args.output_dir))
                manifest_file.write(json.dumps(as_python(manifest_record), ensure_ascii=False) + "\n")
                manifest_file.flush()

                for prefix_length in prefix_lengths:
                    k_record = manifest_record["by_prefix_length"][str(prefix_length)]
                    target = k_record["target_on_clean_state"]
                    state = k_record["clean_on_noisy_state"]
                    result_rows.append(
                        {
                            "pair_id": pair_id,
                            "label": locator.label,
                            "source_dir": locator.source_dir,
                            "source_shard": locator.shard_path,
                            "trial_index": locator.trial_index,
                            "row_index": locator.row_index,
                            "question_index": locator.question_index,
                            "problem_index": locator.problem_index,
                            "question_fingerprint": locator.question_fingerprint,
                            "data_source": locator.data_source,
                            "injection_layer": locator.injection_layer,
                            "response_position": locator.response_position,
                            "first_divergence_index": locator.first_divergence_index,
                            "first_divergence_position": locator.first_divergence_position,
                            "divergence_delay_after_first_changeable": (
                                locator.divergence_delay_after_first_changeable
                            ),
                            "immediate_next_token_change": locator.immediate_next_token_change,
                            "prefix_length": prefix_length,
                            "target_on_clean_score": target["score"],
                            "target_on_clean_is_correct": target["is_correct"],
                            "target_on_clean_response_length": target["response_length"],
                            "clean_on_noisy_score": state["score"],
                            "clean_on_noisy_is_correct": state["is_correct"],
                            "clean_on_noisy_response_length": state["response_length"],
                        }
                    )
                episode_count += 1
                if episode_count % args.log_every == 0:
                    print(
                        f"[branch-replay] completed episodes={episode_count}/{len(pairs) * 2}",
                        flush=True,
                    )

    csv_path = args.output_dir / "results.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    summary = {
        "format_version": FORMAT_VERSION,
        "model": model_path,
        "dtype": dtype_name,
        "device": str(device),
        "output_dir": str(args.output_dir),
        "prefix_lengths": prefix_lengths,
        "min_unforced_source_tokens": args.min_unforced_source_tokens,
        "max_new_tokens": args.max_new_tokens,
        "causal_estimands": {
            "target_on_clean_state": (
                "clean/zero-noise state plus the source trajectory's divergent prefix; "
                "tests token-prefix sufficiency"
            ),
            "clean_on_noisy_state": (
                "original noisy state plus the clean prefix; residual correctness measures "
                "state persistence, while a W2R-to-wrong change supports conditional token necessity"
            ),
        },
        "reproduction_gate": (
            "Every source clean and noisy trajectory must reproduce token-for-token before "
            "counterfactual results are saved"
        ),
        "scan": scan_metadata,
        "matching": matching_metadata,
        "completed_pairs": len(pairs),
        "completed_episodes": episode_count,
        "aggregate": aggregate_results(result_rows, prefix_lengths),
        "elapsed_seconds": time.monotonic() - started_at,
        "files": {
            "selection": "selection.json",
            "manifest": "manifest.jsonl",
            "flat_results": "results.csv",
            "episodes": "episodes/*.pt",
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"[branch-replay] complete: {args.output_dir / 'summary.json'}", flush=True)
    print(json.dumps(as_python(summary["aggregate"]), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
