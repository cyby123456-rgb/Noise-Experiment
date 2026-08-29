#!/usr/bin/env python3
"""Collect fixed-state Gaussian W2R trials from greedy-wrong questions.

For every accepted question, the script first produces one complete greedy
wrong answer.  It then fixes one response-token position, replays the clean
prompt and response prefix through that position, adds one Gaussian vector at
one decoder-block output, and greedily continues the answer.  Noise is applied
only at that fixed response token, never at prompt tokens or later decode steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = Path(os.environ.get("VERL_ROOT", EXPERIMENT_ROOT / "verl")).resolve()
sys.path.insert(0, str(VERL_ROOT))

from verl.models.transformers.noise_injection import _get_decoder_layers
from verl.trainer.main_ppo import _select_rm_score_fn


FORMAT_VERSION = "greedy-wrong-gaussian-probe-v6"


class PositionCleanReplayCorrect(Exception):
    """Skip a position whose matched zero-noise replay is already correct."""

    def __init__(self, score: float) -> None:
        super().__init__(f"position clean replay is correct (score={score})")
        self.score = float(score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Hugging Face checkpoint path or identifier")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True, help="Decoder layer whose output receives noise")
    parser.add_argument(
        "--noise-std",
        type=float,
        required=True,
        help=(
            "Noise scale. With relative_rms this is the per-coordinate std as a fraction of "
            "the clean token hidden RMS; with absolute it is the std in raw hidden units."
        ),
    )
    parser.add_argument(
        "--noise-scale-mode",
        choices=("relative_rms", "absolute"),
        default="relative_rms",
        help="Normalize Gaussian noise to the fixed token's hidden RMS, or use raw hidden units",
    )
    parser.add_argument("--num-noise-seeds", type=int, default=128)
    parser.add_argument("--base-noise-seed", type=int, default=20260827)
    parser.add_argument(
        "--noise-seeds",
        default="",
        help="Optional comma-separated explicit seed bank; overrides count and base seed",
    )
    parser.add_argument("--noise-batch-size", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=1)
    position_group = parser.add_mutually_exclusive_group()
    position_group.add_argument(
        "--response-position",
        type=int,
        default=None,
        help="One-based clean response-token position to perturb",
    )
    position_group.add_argument(
        "--response-position-fraction",
        type=float,
        default=None,
        help="Choose round(response_length * fraction); default is 0.5",
    )
    position_group.add_argument(
        "--num-response-positions",
        type=int,
        default=None,
        help="Probe K approximately uniform interior positions of the clean response",
    )
    parser.add_argument(
        "--row-indices",
        default="",
        help="Optional comma-separated zero-based parquet row indices to consider",
    )
    parser.add_argument(
        "--require-all-input-rollouts-wrong",
        action="store_true",
        help="Also require every existing response in the input row to score <= 0",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str, name: str) -> list[int]:
    if not value.strip():
        return []
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not parsed:
        raise ValueError(f"{name} must contain at least one integer when provided")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate values")
    return parsed


def as_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return as_python(value.as_py())
    if isinstance(value, dict):
        return {str(key): as_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_python(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return as_python(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def jsonable(value: Any) -> Any:
    value = as_python(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def score_response(record: dict[str, Any], response: str) -> float:
    score_fn = _select_rm_score_fn(str(record["data_source"]))
    ground_truth = record["reward_model"]["ground_truth"]
    result = score_fn(solution_str=response, ground_truth=ground_truth)
    if isinstance(result, dict):
        result = result.get("score", 0.0)
    return float(result)


def prompt_ids(tokenizer, prompt: Any, max_input_tokens: int) -> list[int]:
    prompt = as_python(prompt)
    if isinstance(prompt, list):
        token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    elif isinstance(prompt, str):
        token_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")
    token_ids = list(token_ids[-max_input_tokens:])
    if not token_ids:
        raise ValueError("Prompt tokenization produced no input IDs")
    return token_ids


def input_rollouts_all_wrong(record: dict[str, Any]) -> tuple[bool, str]:
    responses = record.get("responses")
    if not isinstance(responses, list) or not responses:
        return False, "missing_input_responses"

    stored_scores = record.get("response_scores")
    if isinstance(stored_scores, list) and len(stored_scores) == len(responses):
        scores = [float(score) for score in stored_scores]
        source = "response_scores"
    else:
        scores = [score_response(record, str(response)) for response in responses]
        source = "rescored_responses"

    if not all(math.isfinite(score) for score in scores):
        return False, "non_finite_input_score"
    return all(score <= 0 for score in scores), source


def resolve_layer(layer: int, layer_count: int) -> int:
    resolved = int(layer)
    if resolved < 0:
        resolved += layer_count
    if not 0 <= resolved < layer_count:
        raise ValueError(f"--layer must resolve into [0, {layer_count - 1}], got {layer}")
    return resolved


def _layer_hidden(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError(
            "Expected decoder layer output with shape [batch, sequence, hidden], "
            f"got {type(hidden).__name__} shape={getattr(hidden, 'shape', None)}"
        )
    return hidden


def _lm_head_hidden(inputs: tuple[Any, ...]) -> torch.Tensor:
    if not inputs:
        raise ValueError("LM head pre-hook received no positional input")
    hidden = inputs[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError(
            "Expected LM head input with shape [batch, sequence, hidden], "
            f"got {type(hidden).__name__} shape={getattr(hidden, 'shape', None)}"
        )
    return hidden


def register_clean_state_capture(model, layer_idx: int) -> tuple[list[Any], dict[str, torch.Tensor]]:
    """Capture the fixed response token at the target layer and LM-head input."""

    layers = _get_decoder_layers(model)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("Model does not expose an output embedding / LM head")
    capture: dict[str, torch.Tensor] = {}

    def target_hook(module, inputs, output):
        if "clean_hidden_state_batch" not in capture:
            hidden = _layer_hidden(output)
            capture["clean_hidden_state_batch"] = hidden[:, -1, :].detach().float().cpu()
        return output

    def final_pre_hook(module, inputs):
        if "clean_final_hidden_state_batch" not in capture:
            hidden = _lm_head_hidden(inputs)
            capture["clean_final_hidden_state_batch"] = hidden[:, -1, :].detach().float().cpu()

    handles = [
        layers[layer_idx].register_forward_hook(target_hook),
        lm_head.register_forward_pre_hook(final_pre_hook),
    ]
    return handles, capture


def register_noisy_state_capture(
    model,
    layer_idx: int,
    sampled_noise: torch.Tensor,
) -> tuple[list[Any], dict[str, torch.Tensor]]:
    """Inject one noise vector per batch item and capture fixed-token states."""

    if sampled_noise.ndim != 2 or sampled_noise.dtype != torch.float32 or sampled_noise.device.type != "cpu":
        raise ValueError("sampled_noise must be a CPU float32 tensor with shape [batch, hidden]")
    layers = _get_decoder_layers(model)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("Model does not expose an output embedding / LM head")
    capture: dict[str, torch.Tensor] = {}
    applied = False

    def target_hook(module, inputs, output):
        nonlocal applied
        if applied:
            return output
        hidden = _layer_hidden(output)
        if hidden.shape[0] != sampled_noise.shape[0] or hidden.shape[-1] != sampled_noise.shape[1]:
            raise ValueError(
                "Noise shape does not match the fixed response hidden state: "
                f"noise={tuple(sampled_noise.shape)}, hidden={tuple(hidden.shape)}"
            )
        clean_token = hidden[:, -1, :]
        noise_device = sampled_noise.to(device=hidden.device, dtype=torch.float32)
        modified_token = (clean_token.float() + noise_device).to(dtype=hidden.dtype)
        modified = hidden.clone()
        modified[:, -1, :] = modified_token

        capture["pre_noise_hidden_state_batch"] = clean_token.detach().float().cpu()
        capture["sampled_noise_batch"] = sampled_noise.clone()
        capture["applied_noise_batch"] = (modified_token.float() - clean_token.float()).detach().cpu()
        applied = True
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    def final_pre_hook(module, inputs):
        if "noisy_final_hidden_state_batch" not in capture:
            hidden = _lm_head_hidden(inputs)
            capture["noisy_final_hidden_state_batch"] = hidden[:, -1, :].detach().float().cpu()

    handles = [
        layers[layer_idx].register_forward_hook(target_hook),
        lm_head.register_forward_pre_hook(final_pre_hook),
    ]
    return handles, capture


def remove_handles(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def require_capture(capture: dict[str, torch.Tensor], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in capture]
    if missing:
        raise RuntimeError(f"Generation completed without required hook captures: {missing}")


@torch.inference_mode()
def greedy_generate_batch(
    model,
    tokenizer,
    input_ids: list[int],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[list[int]]:
    inputs = torch.tensor([input_ids] * batch_size, device=device, dtype=torch.long)
    attention_mask = torch.ones_like(inputs)
    generated = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    eos_ids = tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
    eos_ids.discard(None)
    pad_id = tokenizer.pad_token_id
    completions: list[list[int]] = []
    for row in sequences:
        completion: list[int] = []
        for token_id in row[inputs.shape[1] :].tolist():
            if token_id in eos_ids or (pad_id is not None and token_id == pad_id):
                break
            completion.append(int(token_id))
        completions.append(completion)
    return completions


def decode_response(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def choose_response_positions(
    args: argparse.Namespace,
    response_length: int,
) -> list[int]:
    """Return sorted unique one-based response-token positions.

    Uniform multi-position probing uses the interior fractions j/(K+1).  It
    therefore covers early, middle, and late response states without selecting
    the terminal endpoint by construction.
    """

    # Exclude the final clean response token: every intervention must leave at
    # least one clean-response token after the fixed point, in addition to a
    # positive generation budget.
    max_position = min(response_length - 1, args.max_new_tokens - 1)
    if max_position <= 0:
        return []
    if args.response_position is not None:
        return [args.response_position] if args.response_position <= max_position else []
    if args.response_position_fraction is not None:
        position = max(1, min(max_position, round(response_length * args.response_position_fraction)))
        return [position]
    if args.num_response_positions is not None:
        positions = {
            max(1, min(max_position, round(response_length * index / (args.num_response_positions + 1))))
            for index in range(1, args.num_response_positions + 1)
        }
        return sorted(positions)
    return [max(1, min(max_position, round(response_length * 0.5)))]


def sample_standard_normal(seed: int, hidden_size: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(hidden_size, generator=generator, dtype=torch.float32)


def max_batch_spread(batch: torch.Tensor) -> float:
    if batch.shape[0] <= 1:
        return 0.0
    return float((batch - batch[0:1]).abs().max().item())


def max_difference(batch: torch.Tensor, reference: torch.Tensor) -> float:
    return float((batch - reference.unsqueeze(0)).abs().max().item())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=jsonable), encoding="utf-8")


def collect_one_response_position(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    device: torch.device,
    record: dict[str, Any],
    row_index: int,
    problem_index: Any,
    question_index: int,
    input_rollout_filter_source: str | None,
    prompt_token_ids: list[int],
    clean_response_token_ids: list[int],
    original_greedy_response: str,
    original_greedy_score: float,
    response_position: int,
    layer_idx: int,
    hidden_size: int,
    noise_seeds: list[int],
    completed_trials_before: int,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Collect every noise seed at one fixed clean response-token state."""

    remaining_new_tokens = args.max_new_tokens - response_position
    if remaining_new_tokens <= 0:
        raise ValueError(f"response_position={response_position} leaves no continuation budget")
    fixed_response_prefix_ids = clean_response_token_ids[:response_position]
    replay_input_ids = prompt_token_ids + fixed_response_prefix_ids
    if len(replay_input_ids) > args.max_input_tokens:
        raise ValueError(
            f"Replay input has {len(replay_input_ids)} tokens, exceeding --max-input-tokens={args.max_input_tokens}"
        )

    clean_handles, clean_capture = register_clean_state_capture(model, layer_idx)
    try:
        replay_continuation_batches = greedy_generate_batch(
            model,
            tokenizer,
            replay_input_ids,
            args.noise_batch_size,
            remaining_new_tokens,
            device,
        )
    finally:
        remove_handles(clean_handles)
    require_capture(
        clean_capture,
        ("clean_hidden_state_batch", "clean_final_hidden_state_batch"),
    )
    replay_response_token_batches = [
        fixed_response_prefix_ids + continuation
        for continuation in replay_continuation_batches
    ]
    position_clean_response_token_ids = replay_response_token_batches[0]
    if any(tokens != position_clean_response_token_ids for tokens in replay_response_token_batches[1:]):
        raise RuntimeError(
            f"Repeated zero-noise prefix replays produced different token sequences at row {row_index}, "
            f"response_position={response_position}; the position has no stable clean reference."
        )
    position_clean_response = decode_response(tokenizer, position_clean_response_token_ids)
    position_clean_score = score_response(record, position_clean_response)
    if not math.isfinite(position_clean_score):
        raise RuntimeError(
            f"Position clean replay produced a non-finite score at row {row_index}, "
            f"response_position={response_position}."
        )
    if position_clean_score > 0:
        raise PositionCleanReplayCorrect(position_clean_score)

    clean_hidden_batch = clean_capture["clean_hidden_state_batch"]
    clean_final_batch = clean_capture["clean_final_hidden_state_batch"]
    clean_hidden_state = clean_hidden_batch[0].clone()
    clean_final_hidden_state = clean_final_batch[0].clone()
    clean_hidden_rms = float(clean_hidden_state.square().mean().sqrt().item())
    if not math.isfinite(clean_hidden_rms) or clean_hidden_rms <= 0:
        raise RuntimeError(
            f"Invalid clean hidden RMS at row {row_index}, position={response_position}: "
            f"{clean_hidden_rms}"
        )
    effective_noise_std = (
        float(args.noise_std) * clean_hidden_rms
        if args.noise_scale_mode == "relative_rms"
        else float(args.noise_std)
    )
    clean_hidden_batch_spread = max_batch_spread(clean_hidden_batch)
    clean_final_batch_spread = max_batch_spread(clean_final_batch)
    if clean_hidden_batch_spread != 0 or clean_final_batch_spread != 0:
        raise RuntimeError(
            f"Repeated clean prefixes produced different states at row {row_index}, "
            f"position={response_position}: hidden_spread={clean_hidden_batch_spread}, "
            f"final_spread={clean_final_batch_spread}."
        )

    # Matched placebo group: execute the exact intervention hook and batching
    # path with epsilon=0 for as many trials as the noisy group. Any token or
    # state difference means the noisy comparison is not attributable solely
    # to the saved Gaussian perturbation, so fail before collecting noisy data.
    zero_noise_control_trials = 0
    zero_noise_max_pre_difference = 0.0
    zero_noise_max_applied = 0.0
    zero_noise_max_final_difference = 0.0
    for chunk_start in range(0, len(noise_seeds), args.noise_batch_size):
        valid_count = min(args.noise_batch_size, len(noise_seeds) - chunk_start)
        zero_noise = torch.zeros(args.noise_batch_size, hidden_size, dtype=torch.float32)
        control_handles, control_capture = register_noisy_state_capture(model, layer_idx, zero_noise)
        try:
            control_continuations = greedy_generate_batch(
                model,
                tokenizer,
                replay_input_ids,
                args.noise_batch_size,
                remaining_new_tokens,
                device,
            )
        finally:
            remove_handles(control_handles)
        require_capture(
            control_capture,
            (
                "pre_noise_hidden_state_batch",
                "sampled_noise_batch",
                "applied_noise_batch",
                "noisy_final_hidden_state_batch",
            ),
        )
        control_pre = control_capture["pre_noise_hidden_state_batch"][:valid_count]
        control_applied = control_capture["applied_noise_batch"][:valid_count]
        control_final = control_capture["noisy_final_hidden_state_batch"][:valid_count]
        zero_noise_max_pre_difference = max(
            zero_noise_max_pre_difference,
            max_difference(control_pre, clean_hidden_state),
        )
        zero_noise_max_applied = max(
            zero_noise_max_applied,
            float(control_applied.abs().max().item()),
        )
        zero_noise_max_final_difference = max(
            zero_noise_max_final_difference,
            max_difference(control_final, clean_final_hidden_state),
        )
        for continuation in control_continuations[:valid_count]:
            if fixed_response_prefix_ids + continuation != position_clean_response_token_ids:
                raise RuntimeError(
                    f"Zero-noise placebo changed the position-level clean replay at row {row_index}, "
                    f"position={response_position}."
                )
            zero_noise_control_trials += 1

    if zero_noise_control_trials != len(noise_seeds):
        raise RuntimeError(
            f"Zero-noise control count mismatch: {zero_noise_control_trials} vs {len(noise_seeds)}"
        )
    if (
        zero_noise_max_pre_difference != 0
        or zero_noise_max_applied != 0
        or zero_noise_max_final_difference != 0
    ):
        raise RuntimeError(
            f"Zero-noise placebo changed model states at row {row_index}, position={response_position}: "
            f"pre={zero_noise_max_pre_difference}, applied={zero_noise_max_applied}, "
            f"final={zero_noise_max_final_difference}."
        )

    standard_normal_noise_parts: list[torch.Tensor] = []
    sampled_noise_parts: list[torch.Tensor] = []
    applied_noise_parts: list[torch.Tensor] = []
    noisy_final_parts: list[torch.Tensor] = []
    noisy_responses: list[str] = []
    noisy_response_token_ids: list[list[int]] = []
    noisy_scores: list[float] = []
    is_w2r: list[bool] = []
    noise_hashes: list[str] = []
    max_pre_noise_difference = 0.0
    completed_here = 0

    for chunk_start in range(0, len(noise_seeds), args.noise_batch_size):
        chunk_seeds = noise_seeds[chunk_start : chunk_start + args.noise_batch_size]
        valid_count = len(chunk_seeds)
        valid_standard_normal = torch.stack(
            [sample_standard_normal(seed, hidden_size) for seed in chunk_seeds]
        )
        valid_noise = valid_standard_normal * effective_noise_std
        padded_noise = torch.zeros(args.noise_batch_size, hidden_size, dtype=torch.float32)
        padded_noise[:valid_count] = valid_noise

        noisy_handles, noisy_capture = register_noisy_state_capture(model, layer_idx, padded_noise)
        try:
            batch_continuations = greedy_generate_batch(
                model,
                tokenizer,
                replay_input_ids,
                args.noise_batch_size,
                remaining_new_tokens,
                device,
            )
        finally:
            remove_handles(noisy_handles)
        require_capture(
            noisy_capture,
            (
                "pre_noise_hidden_state_batch",
                "sampled_noise_batch",
                "applied_noise_batch",
                "noisy_final_hidden_state_batch",
            ),
        )

        pre_noise_batch = noisy_capture["pre_noise_hidden_state_batch"][:valid_count]
        max_pre_noise_difference = max(
            max_pre_noise_difference,
            max_difference(pre_noise_batch, clean_hidden_state),
        )
        standard_normal_noise_parts.append(valid_standard_normal.clone())
        sampled_noise_parts.append(noisy_capture["sampled_noise_batch"][:valid_count].clone())
        applied_noise_parts.append(noisy_capture["applied_noise_batch"][:valid_count].clone())
        noisy_final_parts.append(noisy_capture["noisy_final_hidden_state_batch"][:valid_count].clone())

        for local_index, continuation in enumerate(batch_continuations[:valid_count]):
            response_token_ids = fixed_response_prefix_ids + continuation
            response = decode_response(tokenizer, response_token_ids)
            score = score_response(record, response)
            flipped = bool(math.isfinite(score) and score > 0)
            noisy_responses.append(response)
            noisy_response_token_ids.append(response_token_ids)
            noisy_scores.append(score)
            is_w2r.append(flipped)
            noise_hashes.append(hashlib.sha256(valid_noise[local_index].numpy().tobytes()).hexdigest())
            completed_here += 1
            completed_total = completed_trials_before + completed_here
            if completed_total % args.log_every == 0:
                print(
                    f"[greedy-gaussian] trials={completed_total}, question={question_index}, "
                    f"position={response_position}, position_W2R={sum(is_w2r)}/{completed_here}",
                    flush=True,
                )

    standard_normal_noise_tensor = torch.cat(standard_normal_noise_parts, dim=0)
    sampled_noise_tensor = torch.cat(sampled_noise_parts, dim=0)
    applied_noise_tensor = torch.cat(applied_noise_parts, dim=0)
    noisy_final_tensor = torch.cat(noisy_final_parts, dim=0)
    if max_pre_noise_difference != 0:
        raise RuntimeError(
            f"Noisy trials did not start from the same clean state at row {row_index}, "
            f"position={response_position}: max_diff={max_pre_noise_difference}."
        )
    expected_shape = (len(noise_seeds), hidden_size)
    tensor_shapes = {
        "standard_normal_noise": tuple(standard_normal_noise_tensor.shape),
        "sampled_noise": tuple(sampled_noise_tensor.shape),
        "applied_noise": tuple(applied_noise_tensor.shape),
        "noisy_final_hidden_state": tuple(noisy_final_tensor.shape),
    }
    malformed = {name: shape for name, shape in tensor_shapes.items() if shape != expected_shape}
    if malformed:
        raise RuntimeError(f"Collected tensor shapes do not match {expected_shape}: {malformed}")
    if not (
        torch.isfinite(clean_hidden_state).all()
        and torch.isfinite(clean_final_hidden_state).all()
        and torch.isfinite(standard_normal_noise_tensor).all()
        and torch.isfinite(sampled_noise_tensor).all()
        and torch.isfinite(applied_noise_tensor).all()
        and torch.isfinite(noisy_final_tensor).all()
    ):
        raise RuntimeError(f"Non-finite hidden state or noise collected at row {row_index}, position {response_position}")

    diagnostics = {
        "clean_hidden_batch_max_abs_spread": clean_hidden_batch_spread,
        "clean_final_batch_max_abs_spread": clean_final_batch_spread,
        "zero_noise_max_pre_hidden_vs_clean_abs_diff": zero_noise_max_pre_difference,
        "zero_noise_max_applied_abs": zero_noise_max_applied,
        "zero_noise_max_final_hidden_vs_clean_abs_diff": zero_noise_max_final_difference,
        "max_pre_noise_hidden_vs_clean_abs_diff": max_pre_noise_difference,
    }
    question_w2r = sum(is_w2r)
    shard = {
        "format_version": FORMAT_VERSION,
        "metadata": {
            "input_parquet": str(args.input_parquet),
            "model": args.model,
            "row_index": row_index,
            "problem_index": problem_index,
            "question_index": question_index,
            "data_source": record["data_source"],
            "ground_truth": record["reward_model"]["ground_truth"],
            "prompt": record["prompt"],
            "injection_layer": layer_idx,
            "response_position": response_position,
            "response_position_fraction": response_position / len(clean_response_token_ids),
            "response_position_indexing": "one_based",
            "clean_response_length": len(clean_response_token_ids),
            "injection_location": "decoder_layer_output/fixed_response_token/prefix_replay_once",
            "final_state_location": "lm_head_input/fixed_response_token",
            "noise_distribution": "isotropic_gaussian",
            "noise_std": float(args.noise_std),
            "noise_scale_mode": args.noise_scale_mode,
            "clean_hidden_rms": clean_hidden_rms,
            "effective_noise_std_hidden_units": effective_noise_std,
            "model_dtype": args.dtype,
            "input_rollout_filter_source": input_rollout_filter_source,
        },
        "prompt_token_ids": torch.tensor(prompt_token_ids, dtype=torch.long),
        "clean_response_token_ids": torch.tensor(clean_response_token_ids, dtype=torch.long),
        "fixed_response_prefix_token_ids": torch.tensor(fixed_response_prefix_ids, dtype=torch.long),
        "position_clean_response_token_ids": torch.tensor(position_clean_response_token_ids, dtype=torch.long),
        "clean_hidden_state": clean_hidden_state,
        "clean_final_hidden_state": clean_final_hidden_state,
        "original_greedy_response": original_greedy_response,
        "original_greedy_score": float(original_greedy_score),
        "baseline_response": position_clean_response,
        "baseline_score": float(position_clean_score),
        "zero_noise_control": {
            "trials": zero_noise_control_trials,
            "all_token_sequences_match_clean": True,
            "w2r_count": 0,
            "w2r_rate": 0.0,
        },
        "noise_seeds": torch.tensor(noise_seeds, dtype=torch.long),
        "noise_sha256": noise_hashes,
        "standard_normal_noise": standard_normal_noise_tensor,
        "sampled_noise": sampled_noise_tensor,
        "applied_noise": applied_noise_tensor,
        "noisy_final_hidden_state": noisy_final_tensor,
        "noisy_responses": noisy_responses,
        "noisy_response_token_ids": noisy_response_token_ids,
        "noisy_scores": torch.tensor(noisy_scores, dtype=torch.float64),
        "is_w2r": torch.tensor(is_w2r, dtype=torch.bool),
        "diagnostics": diagnostics,
    }
    manifest_record = {
        "format_version": FORMAT_VERSION,
        "question_index": question_index,
        "row_index": row_index,
        "problem_index": problem_index,
        "data_source": record["data_source"],
        "injection_layer": layer_idx,
        "response_position": response_position,
        "response_position_fraction": response_position / len(clean_response_token_ids),
        "clean_response_length": len(clean_response_token_ids),
        "noise_std": float(args.noise_std),
        "noise_scale_mode": args.noise_scale_mode,
        "clean_hidden_rms": clean_hidden_rms,
        "effective_noise_std_hidden_units": effective_noise_std,
        "num_noise_seeds": len(noise_seeds),
        "zero_noise_control_trials": zero_noise_control_trials,
        "zero_noise_control_w2r_count": 0,
        "zero_noise_control_w2r_rate": 0.0,
        "original_greedy_score": float(original_greedy_score),
        "baseline_score": float(position_clean_score),
        "position_clean_matches_original_greedy": (
            position_clean_response_token_ids == clean_response_token_ids
        ),
        "w2r_count": question_w2r,
        "w2r_rate": question_w2r / len(noise_seeds),
        "diagnostics": diagnostics,
    }
    return shard, manifest_record, completed_here, question_w2r


def main() -> None:
    args = parse_args()
    if args.noise_std <= 0:
        raise ValueError("--noise-std must be positive")
    if args.num_noise_seeds <= 0 or args.noise_batch_size <= 0 or args.max_questions <= 0:
        raise ValueError("--num-noise-seeds, --noise-batch-size, and --max-questions must be positive")
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0 or args.log_every <= 0:
        raise ValueError("--max-new-tokens, --max-input-tokens, and --log-every must be positive")
    if args.response_position is not None and args.response_position <= 0:
        raise ValueError("--response-position is one-based and must be positive")
    if args.response_position_fraction is not None and not 0 < args.response_position_fraction <= 1:
        raise ValueError("--response-position-fraction must be in (0, 1]")
    if args.num_response_positions is not None and args.num_response_positions <= 0:
        raise ValueError("--num-response-positions must be positive")

    explicit_seeds = parse_int_list(args.noise_seeds, "--noise-seeds")
    noise_seeds = explicit_seeds or [args.base_noise_seed + offset for offset in range(args.num_noise_seeds)]
    row_indices = set(parse_int_list(args.row_indices, "--row-indices")) if args.row_indices.strip() else None
    if any(seed < 0 for seed in noise_seeds):
        raise ValueError("Noise seeds must be non-negative")
    if row_indices is not None and any(index < 0 for index in row_indices):
        raise ValueError("--row-indices must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment manifest: {manifest_path}. Use a new --output-dir."
        )

    started_at = time.monotonic()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"[greedy-gaussian] loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[greedy-gaussian] loading model on {device} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    layers = _get_decoder_layers(model)
    layer_idx = resolve_layer(args.layer, len(layers))
    hidden_size = int(model.config.hidden_size)
    if hidden_size <= 0:
        raise ValueError(f"Invalid model hidden_size={hidden_size}")

    print(f"[greedy-gaussian] reading input: {args.input_parquet}", flush=True)
    df = pd.read_parquet(args.input_parquet)
    required_columns = {"prompt", "data_source", "reward_model"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")
    if row_indices is not None:
        absent = sorted(row_indices - set(range(len(df))))
        if absent:
            raise IndexError(f"Requested row indices are outside the parquet: {absent}")

    accepted = 0
    completed_trials = 0
    total_w2r = 0
    exclusions: Counter[str] = Counter()
    question_summaries: list[dict[str, Any]] = []
    clean_question_summaries: list[dict[str, Any]] = []

    with manifest_path.open("x", encoding="utf-8") as manifest:
        for row_index, raw_record in enumerate(df.to_dict("records")):
            if row_indices is not None and row_index not in row_indices:
                continue
            record = as_python(raw_record)

            input_rollout_filter_source = None
            if args.require_all_input_rollouts_wrong:
                all_wrong, input_rollout_filter_source = input_rollouts_all_wrong(record)
                if not all_wrong:
                    exclusions[f"input_not_all_wrong:{input_rollout_filter_source}"] += 1
                    continue

            try:
                token_ids = prompt_ids(tokenizer, record["prompt"], args.max_input_tokens)
            except (TypeError, ValueError) as exc:
                exclusions[f"invalid_prompt:{type(exc).__name__}"] += 1
                continue

            print(f"[greedy-gaussian] row={row_index}: running clean greedy baseline", flush=True)
            baseline_token_batches = greedy_generate_batch(
                model,
                tokenizer,
                token_ids,
                1,
                args.max_new_tokens,
                device,
            )
            clean_response_token_ids = baseline_token_batches[0]
            if not clean_response_token_ids:
                exclusions["empty_greedy_response"] += 1
                continue
            baseline_response = decode_response(tokenizer, clean_response_token_ids)
            baseline_score = score_response(record, baseline_response)
            if not math.isfinite(baseline_score):
                exclusions["non_finite_greedy_score"] += 1
                continue
            if baseline_score > 0:
                exclusions["greedy_correct"] += 1
                continue

            response_positions = choose_response_positions(args, len(clean_response_token_ids))
            if not response_positions:
                exclusions["no_valid_response_positions"] += 1
                continue
            print(
                f"[greedy-gaussian] row={row_index}: clean_response_tokens={len(clean_response_token_ids)}, "
                f"selected_positions={response_positions}",
                flush=True,
            )

            extra_info = record.get("extra_info") or {}
            problem_index = extra_info.get("index", row_index) if isinstance(extra_info, dict) else row_index
            question_index = accepted
            completed_positions = 0
            for response_position in response_positions:
                if len(token_ids) + response_position > args.max_input_tokens:
                    exclusions["replay_prefix_exceeds_max_input_tokens"] += 1
                    continue
                try:
                    shard, manifest_record, position_trials, position_w2r = collect_one_response_position(
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        record=record,
                        row_index=row_index,
                        problem_index=problem_index,
                        question_index=question_index,
                        input_rollout_filter_source=input_rollout_filter_source,
                        prompt_token_ids=token_ids,
                        clean_response_token_ids=clean_response_token_ids,
                        original_greedy_response=baseline_response,
                        original_greedy_score=baseline_score,
                        response_position=response_position,
                        layer_idx=layer_idx,
                        hidden_size=hidden_size,
                        noise_seeds=noise_seeds,
                        completed_trials_before=completed_trials,
                    )
                except PositionCleanReplayCorrect as exc:
                    exclusions["position_clean_replay_correct"] += 1
                    print(
                        f"[greedy-gaussian] row={row_index}, position={response_position}: "
                        f"skipped because matched clean replay is already correct (score={exc.score})",
                        flush=True,
                    )
                    continue
                shard_name = (
                    f"question_{question_index:04d}_row_{row_index}_pos_{response_position:06d}.pt"
                )
                shard_path = tensor_dir / shard_name
                if shard_path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing tensor shard: {shard_path}")
                torch.save(shard, shard_path)
                manifest_record["tensor_path"] = str(shard_path.relative_to(args.output_dir))
                manifest.write(json.dumps(manifest_record, ensure_ascii=False, default=jsonable) + "\n")
                manifest.flush()
                question_summaries.append(manifest_record)
                completed_trials += position_trials
                total_w2r += position_w2r
                completed_positions += 1
                print(
                    f"[greedy-gaussian] row={row_index}, position={response_position}/"
                    f"{len(clean_response_token_ids)}: W2R={position_w2r}/{position_trials}; "
                    f"saved={shard_path}",
                    flush=True,
                )
            if completed_positions == 0:
                exclusions["no_positions_completed"] += 1
                continue
            clean_question_summaries.append(
                {
                    "question_index": question_index,
                    "row_index": row_index,
                    "problem_index": problem_index,
                    "clean_response_length": len(clean_response_token_ids),
                    "original_greedy_score": float(baseline_score),
                    "selected_response_positions": response_positions,
                    "completed_response_positions": completed_positions,
                }
            )
            accepted += 1
            print(
                f"[greedy-gaussian] accepted row={row_index}; positions={completed_positions}/"
                f"{len(response_positions)}",
                flush=True,
            )
            if accepted >= args.max_questions:
                break

    if accepted == 0:
        raise RuntimeError(
            "No greedy-wrong question was accepted. "
            f"Filter/exclusion counts: {dict(exclusions)}"
        )

    summary = {
        "format_version": FORMAT_VERSION,
        "input_parquet": str(args.input_parquet),
        "model": args.model,
        "output_dir": str(args.output_dir),
        "injection_layer": layer_idx,
        "decoder_layer_count": len(layers),
        "hidden_size": hidden_size,
        "response_position_selection": (
            {"mode": "absolute_one_based", "value": args.response_position}
            if args.response_position is not None
            else (
                {"mode": "fraction_of_clean_response", "value": args.response_position_fraction}
                if args.response_position_fraction is not None
                else (
                    {"mode": "uniform_interior", "value": args.num_response_positions}
                    if args.num_response_positions is not None
                    else {"mode": "fraction_of_clean_response", "value": 0.5}
                )
            )
        ),
        "noise_std": float(args.noise_std),
        "noise_scale_mode": args.noise_scale_mode,
        "noise_seeds": noise_seeds,
        "noise_batch_size": args.noise_batch_size,
        "require_all_input_rollouts_wrong": args.require_all_input_rollouts_wrong,
        "accepted_questions": accepted,
        "completed_response_positions": len(question_summaries),
        "completed_zero_noise_control_trials": sum(
            int(position["zero_noise_control_trials"]) for position in question_summaries
        ),
        "completed_trials": completed_trials,
        "w2r_count": total_w2r,
        "w2r_rate": total_w2r / completed_trials if completed_trials else None,
        "exclusions": dict(exclusions),
        "clean_questions": clean_question_summaries,
        "positions": question_summaries,
        "elapsed_seconds": time.monotonic() - started_at,
    }
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"[greedy-gaussian] complete: summary={summary_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=jsonable))


if __name__ == "__main__":
    main()
