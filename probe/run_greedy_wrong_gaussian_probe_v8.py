#!/usr/bin/env python3
"""Collect strict full-path Gaussian W2R trials from greedy-wrong questions.

For every accepted question, the script first produces one complete batch-1
greedy wrong answer from the original prompt.  For each selected response-token
position, every control/noisy trial starts again from that same original prompt
and follows the same batch-1 greedy decoding path.  The intervention hook is
idle during prompt prefill and all earlier response-token steps; it injects one
Gaussian vector only while processing the selected response token.  Therefore
tokens through the selected position must match the clean baseline exactly, and
only later tokens are allowed to diverge.
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


FORMAT_VERSION = (
    "greedy-wrong-gaussian-probe-v8-strict-full-path-seed-modes-"
    "suffix-traces-logits-strict-code"
)

CODE_ALIASES = {
    "train-code-taco-medium": "taco",
    "train-code-taco-hard": "taco",
    "train-code-taco-medium_hard": "taco",
}
CODE_SOURCES = {"taco", "codecontests", "apps", "codeforces"}


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
        "--noise-seed-mode",
        choices=("paired", "independent"),
        default="paired",
        help=(
            "paired reuses the same per-trial Gaussian directions across questions/positions; "
            "independent deterministically derives a distinct seed for every "
            "(row, response position, trial) while remaining reproducible."
        ),
    )
    parser.add_argument(
        "--noise-namespace",
        default="",
        help=(
            "Required in independent mode. A stable dataset/experiment identifier included in "
            "effective seed derivation so different datasets do not reuse Gaussian directions."
        ),
    )
    parser.add_argument(
        "--noise-seeds",
        default="",
        help="Optional comma-separated explicit seed bank; overrides count and base seed",
    )
    parser.add_argument(
        "--noise-batch-size",
        type=int,
        default=1,
        help=(
            "Strict full-path mode requires 1. Each clean/noisy trial is run from the original "
            "prompt with batch size 1 so batching cannot alter the pre-intervention greedy path."
        ),
    )
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


def derive_independent_noise_seed(
    base_seed: int,
    noise_namespace: str,
    question_fingerprint: str,
    layer_idx: int,
    response_position: int,
    trial_index: int,
) -> int:
    """Derive a stable, condition-specific 63-bit seed without Python hash randomization."""
    payload = (
        f"greedy-gaussian-v8:{base_seed}:{noise_namespace}:{question_fingerprint}:"
        f"{layer_idx}:{response_position}:{trial_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def resolve_position_noise_seeds(
    seed_bank: list[int],
    mode: str,
    noise_namespace: str,
    question_fingerprint: str,
    layer_idx: int,
    response_position: int,
) -> list[int]:
    if mode == "paired":
        return list(seed_bank)
    if mode != "independent":
        raise ValueError(f"Unsupported noise seed mode: {mode}")
    resolved = [
        derive_independent_noise_seed(
            seed,
            noise_namespace,
            question_fingerprint,
            layer_idx,
            response_position,
            trial_index,
        )
        for trial_index, seed in enumerate(seed_bank)
    ]
    if len(set(resolved)) != len(resolved):
        raise RuntimeError(
            f"Independent seed derivation produced a collision for question={question_fingerprint}, "
            f"response_position={response_position}"
        )
    return resolved


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


def question_fingerprint(record: dict[str, Any]) -> str:
    """Return a stable content fingerprint independent of parquet row numbering."""
    reward_model = record.get("reward_model") or {}
    ground_truth = (
        reward_model.get("ground_truth")
        if isinstance(reward_model, dict)
        else reward_model
    )
    payload = {
        "data_source": record.get("data_source"),
        "prompt": as_python(record.get("prompt")),
        "ground_truth": as_python(ground_truth),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=jsonable,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def is_code_data_source(data_source: Any) -> bool:
    source = str(data_source)
    canonical = CODE_ALIASES.get(source, source)
    return canonical.startswith("livecodebench/") or canonical in CODE_SOURCES


def validate_code_ground_truth(record: dict[str, Any]) -> None:
    """Fail closed when a code row does not contain executable test cases."""
    if not is_code_data_source(record.get("data_source")):
        return

    from verl.utils.reward_score.prime_code import load_test_cases

    reward_model = record.get("reward_model")
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        raise ValueError("Code row is missing reward_model.ground_truth")
    test_cases = load_test_cases(reward_model["ground_truth"])
    if not isinstance(test_cases, dict):
        raise ValueError("Code ground truth must decode to a test-case dictionary")
    inputs = test_cases.get("inputs")
    outputs = test_cases.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("Code ground truth must contain list-valued inputs and outputs")
    if not inputs or len(inputs) != len(outputs):
        raise ValueError(
            "Code ground truth must contain a non-empty, equally sized inputs/outputs pair"
        )


def score_response(record: dict[str, Any], response: str) -> float:
    # W2R is an evaluation-time correctness transition. Code must therefore be
    # binary pass-all-tests, never the continuous partial-test reward used for
    # training. For non-code sources this flag leaves scorer selection intact.
    score_fn = _select_rm_score_fn(
        str(record["data_source"]),
        code_continuous=False,
    )
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
    if is_code_data_source(record.get("data_source")):
        # Stored code scores may have been produced with a continuous training
        # reward. Re-execute in strict pass-all-tests mode so the input filter
        # and the W2R outcome use exactly the same correctness definition.
        scores = [score_response(record, str(response)) for response in responses]
        source = "rescored_code_responses_pass_all_tests"
    elif isinstance(stored_scores, list) and len(stored_scores) == len(responses):
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


def _lm_head_logits(output: Any) -> torch.Tensor:
    logits = output[0] if isinstance(output, tuple) else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError(
            "Expected LM head output with shape [batch, sequence, vocabulary], "
            f"got {type(logits).__name__} shape={getattr(logits, 'shape', None)}"
        )
    return logits


def register_position_intervention(
    model,
    layer_idx: int,
    response_position: int,
    sampled_noise: torch.Tensor | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Capture/intervene only when the selected response token is processed.

    With cached greedy decoding, decoder-layer hook invocation 0 is the prompt
    prefill. Invocation 1 processes response token 1, invocation 2 processes
    response token 2, and so on. Noise at response position p therefore cannot
    affect response tokens 1..p; the first token it can change is p+1.
    """

    if response_position <= 0:
        raise ValueError("response_position must be one-based and positive")
    if sampled_noise is not None:
        if sampled_noise.ndim != 1 or sampled_noise.dtype != torch.float32 or sampled_noise.device.type != "cpu":
            raise ValueError("sampled_noise must be a CPU float32 vector with shape [hidden]")

    layers = _get_decoder_layers(model)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("Model does not expose an output embedding / LM head")

    capture: dict[str, Any] = {"suffix_decoder_layer_hidden_states": {}}
    forward_call_index = -1
    waiting_for_target_lm_head = False
    waiting_for_target_logits = False

    def target_hook(module, inputs, output):
        nonlocal forward_call_index, waiting_for_target_lm_head
        forward_call_index += 1
        if forward_call_index != response_position:
            return output

        hidden = _layer_hidden(output)
        if hidden.shape[0] != 1:
            raise RuntimeError(
                "Strict full-path intervention requires batch size 1, "
                f"got hidden batch={hidden.shape[0]}"
            )
        clean_token = hidden[:, -1, :]
        capture["pre_noise_hidden_state"] = clean_token[0].detach().float().cpu()
        capture["target_forward_call_index"] = int(forward_call_index)
        waiting_for_target_lm_head = True

        if sampled_noise is None:
            capture["sampled_noise"] = torch.zeros(clean_token.shape[-1], dtype=torch.float32)
            capture["applied_noise"] = torch.zeros(clean_token.shape[-1], dtype=torch.float32)
            capture["post_intervention_hidden_state"] = clean_token[0].detach().float().cpu()
            capture["suffix_decoder_layer_hidden_states"][layer_idx] = (
                clean_token[0].detach().float().cpu()
            )
            return output

        if clean_token.shape[-1] != sampled_noise.shape[0]:
            raise ValueError(
                "Noise shape does not match target hidden state: "
                f"noise={tuple(sampled_noise.shape)}, hidden={tuple(clean_token.shape)}"
            )
        noise_device = sampled_noise.to(device=hidden.device, dtype=torch.float32)
        modified_token = (clean_token.float() + noise_device.unsqueeze(0)).to(dtype=hidden.dtype)
        modified = hidden.clone()
        modified[:, -1, :] = modified_token
        capture["sampled_noise"] = sampled_noise.clone()
        capture["applied_noise"] = (modified_token.float() - clean_token.float())[0].detach().cpu()
        capture["post_intervention_hidden_state"] = modified_token[0].detach().float().cpu()
        capture["suffix_decoder_layer_hidden_states"][layer_idx] = (
            modified_token[0].detach().float().cpu()
        )
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    def make_suffix_layer_hook(suffix_layer_idx: int):
        def suffix_layer_hook(module, inputs, output):
            if not waiting_for_target_lm_head:
                return output
            hidden = _layer_hidden(output)
            if hidden.shape[0] != 1:
                raise RuntimeError(
                    "Strict full-path suffix capture requires batch size 1, "
                    f"got hidden batch={hidden.shape[0]} at decoder layer {suffix_layer_idx}"
                )
            capture["suffix_decoder_layer_hidden_states"][suffix_layer_idx] = (
                hidden[0, -1, :].detach().float().cpu()
            )
            return output

        return suffix_layer_hook

    def final_pre_hook(module, inputs):
        nonlocal waiting_for_target_lm_head, waiting_for_target_logits
        if not waiting_for_target_lm_head:
            return None
        hidden = _lm_head_hidden(inputs)
        if hidden.shape[0] != 1:
            raise RuntimeError(
                "Strict full-path intervention requires batch size 1 at LM head, "
                f"got batch={hidden.shape[0]}"
            )
        capture["final_hidden_state"] = hidden[0, -1, :].detach().float().cpu()
        waiting_for_target_lm_head = False
        waiting_for_target_logits = True
        return None

    def final_output_hook(module, inputs, output):
        nonlocal waiting_for_target_logits
        if not waiting_for_target_logits:
            return output
        logits = _lm_head_logits(output)
        if logits.shape[0] != 1:
            raise RuntimeError(
                "Strict full-path logit capture requires batch size 1 at LM head, "
                f"got batch={logits.shape[0]}"
            )
        capture["next_token_logits"] = logits[0, -1, :].detach().float().cpu()
        waiting_for_target_logits = False
        return output

    handles = [layers[layer_idx].register_forward_hook(target_hook)]
    handles.extend(
        layers[suffix_layer_idx].register_forward_hook(make_suffix_layer_hook(suffix_layer_idx))
        for suffix_layer_idx in range(layer_idx + 1, len(layers))
    )
    handles.append(lm_head.register_forward_pre_hook(final_pre_hook))
    handles.append(lm_head.register_forward_hook(final_output_hook))
    return handles, capture


def run_full_greedy_path(
    *,
    model,
    tokenizer,
    prompt_token_ids: list[int],
    response_position: int,
    layer_idx: int,
    sampled_noise: torch.Tensor | None,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[int], dict[str, Any]]:
    """Run one batch-1 greedy generation from the original prompt."""

    handles, capture = register_position_intervention(
        model,
        response_position=response_position,
        layer_idx=layer_idx,
        sampled_noise=sampled_noise,
    )
    try:
        response_token_ids = greedy_generate_batch(
            model,
            tokenizer,
            prompt_token_ids,
            1,
            max_new_tokens,
            device,
        )[0]
    finally:
        remove_handles(handles)

    required = (
        "pre_noise_hidden_state",
        "post_intervention_hidden_state",
        "sampled_noise",
        "applied_noise",
        "final_hidden_state",
        "next_token_logits",
    )
    missing = [key for key in required if key not in capture]
    if missing:
        raise RuntimeError(
            f"Full-path generation never captured target response_position={response_position}: missing={missing}"
        )
    if capture.get("target_forward_call_index") != response_position:
        raise RuntimeError(
            f"Target hook fired at unexpected forward call {capture.get('target_forward_call_index')} "
            f"for response_position={response_position}"
        )
    suffix_decoder_layer_indices = list(range(layer_idx, len(_get_decoder_layers(model))))
    layer_states = capture["suffix_decoder_layer_hidden_states"]
    missing_suffix_layers = [
        suffix_layer_idx
        for suffix_layer_idx in suffix_decoder_layer_indices
        if suffix_layer_idx not in layer_states
    ]
    if missing_suffix_layers:
        raise RuntimeError(
            f"Target token suffix capture missed decoder layers {missing_suffix_layers} "
            f"at response_position={response_position}"
        )
    capture["suffix_decoder_layer_indices"] = suffix_decoder_layer_indices
    capture["suffix_state_labels"] = [
        f"decoder_layer_{suffix_layer_idx}_output"
        for suffix_layer_idx in suffix_decoder_layer_indices
    ] + ["final_norm_lm_head_input"]
    capture["suffix_hidden_states"] = torch.stack(
        [layer_states[suffix_layer_idx] for suffix_layer_idx in suffix_decoder_layer_indices]
        + [capture["final_hidden_state"]],
        dim=0,
    )
    return response_token_ids, capture


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
    baseline_response: str,
    baseline_score: float,
    response_position: int,
    layer_idx: int,
    hidden_size: int,
    noise_seeds: list[int],
    completed_trials_before: int,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Collect strict W2R trials at one response position.

    Invariants:
      1. A second clean batch-1 run from the original prompt must reproduce the
         complete original greedy response token-for-token.
      2. A zero-noise hook run from the original prompt must also reproduce it.
      3. Every noisy run must have the identical clean hidden state before the
         intervention, and its response tokens 1..p must equal the baseline.
      4. W2R is scored only after those invariants hold.
    """

    record_question_fingerprint = question_fingerprint(record)
    fixed_response_prefix_ids = clean_response_token_ids[:response_position]

    # Clean state capture: same original prompt, same batch-1 greedy path, no
    # prefix replay and no alternative baseline definition.
    clean_regen_tokens, clean_capture = run_full_greedy_path(
        model=model,
        tokenizer=tokenizer,
        prompt_token_ids=prompt_token_ids,
        response_position=response_position,
        layer_idx=layer_idx,
        sampled_noise=None,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    if clean_regen_tokens != clean_response_token_ids:
        raise RuntimeError(
            f"Strict clean greedy regeneration did not reproduce the original baseline at row {row_index}, "
            f"response_position={response_position}. No W2R data are valid for this position."
        )

    clean_hidden_state = clean_capture["pre_noise_hidden_state"].clone()
    clean_final_hidden_state = clean_capture["final_hidden_state"].clone()
    suffix_decoder_layer_indices = list(clean_capture["suffix_decoder_layer_indices"])
    suffix_state_labels = list(clean_capture["suffix_state_labels"])
    clean_suffix_hidden_states = clean_capture["suffix_hidden_states"].clone()
    clean_next_token_logits = clean_capture["next_token_logits"].clone()
    vocab_size = int(clean_next_token_logits.shape[0])
    if vocab_size <= 0:
        raise RuntimeError(
            f"Captured invalid vocabulary size at row={row_index}, position={response_position}: "
            f"{vocab_size}"
        )
    expected_suffix_depth = len(suffix_decoder_layer_indices) + 1
    if clean_suffix_hidden_states.shape != (expected_suffix_depth, hidden_size):
        raise RuntimeError(
            "Clean suffix hidden-state trace has unexpected shape: "
            f"expected={(expected_suffix_depth, hidden_size)}, "
            f"actual={tuple(clean_suffix_hidden_states.shape)}"
        )
    clean_hidden_rms = float(clean_hidden_state.square().mean().sqrt().item())
    if not math.isfinite(clean_hidden_rms) or clean_hidden_rms <= 0:
        raise RuntimeError(
            f"Invalid clean hidden RMS at row {row_index}, position={response_position}: {clean_hidden_rms}"
        )
    effective_noise_std = (
        float(args.noise_std) * clean_hidden_rms
        if args.noise_scale_mode == "relative_rms"
        else float(args.noise_std)
    )

    # One deterministic zero-noise placebo is sufficient in strict batch-1
    # mode. It executes the intervention code path but must remain bit-identical
    # in tokens and captured states.
    zero_noise = torch.zeros(hidden_size, dtype=torch.float32)
    zero_tokens, zero_capture = run_full_greedy_path(
        model=model,
        tokenizer=tokenizer,
        prompt_token_ids=prompt_token_ids,
        response_position=response_position,
        layer_idx=layer_idx,
        sampled_noise=zero_noise,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    if zero_tokens != clean_response_token_ids:
        raise RuntimeError(
            f"Zero-noise full-path control changed the greedy answer at row {row_index}, "
            f"response_position={response_position}."
        )
    zero_pre_diff = float((zero_capture["pre_noise_hidden_state"] - clean_hidden_state).abs().max().item())
    zero_applied_max = float(zero_capture["applied_noise"].abs().max().item())
    zero_final_diff = float((zero_capture["final_hidden_state"] - clean_final_hidden_state).abs().max().item())
    zero_suffix_diff = float(
        (zero_capture["suffix_hidden_states"] - clean_suffix_hidden_states).abs().max().item()
    )
    zero_logits_diff = float(
        (zero_capture["next_token_logits"] - clean_next_token_logits).abs().max().item()
    )
    if (
        zero_capture["suffix_decoder_layer_indices"] != suffix_decoder_layer_indices
        or zero_capture["suffix_state_labels"] != suffix_state_labels
    ):
        raise RuntimeError(
            f"Zero-noise suffix capture schema differs from clean at row={row_index}, "
            f"position={response_position}."
        )
    if (
        zero_pre_diff != 0
        or zero_applied_max != 0
        or zero_final_diff != 0
        or zero_suffix_diff != 0
        or zero_logits_diff != 0
    ):
        raise RuntimeError(
            f"Zero-noise full-path control changed model states at row {row_index}, "
            f"position={response_position}: pre={zero_pre_diff}, applied={zero_applied_max}, "
            f"final={zero_final_diff}, suffix={zero_suffix_diff}, logits={zero_logits_diff}."
        )

    standard_normal_noise_parts: list[torch.Tensor] = []
    sampled_noise_parts: list[torch.Tensor] = []
    applied_noise_parts: list[torch.Tensor] = []
    noisy_final_parts: list[torch.Tensor] = []
    noisy_suffix_hidden_parts: list[torch.Tensor] = []
    noisy_next_token_logits_parts: list[torch.Tensor] = []
    noisy_responses: list[str] = []
    noisy_response_token_ids: list[list[int]] = []
    noisy_scores: list[float] = []
    is_w2r: list[bool] = []
    noise_hashes: list[str] = []
    max_pre_noise_difference = 0.0
    completed_here = 0

    for seed in noise_seeds:
        standard_normal = sample_standard_normal(seed, hidden_size)
        sampled_noise = standard_normal * effective_noise_std
        response_token_ids, noisy_capture = run_full_greedy_path(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            response_position=response_position,
            layer_idx=layer_idx,
            sampled_noise=sampled_noise,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        # Noise while processing token p can only affect token p+1 onward.
        if len(response_token_ids) < response_position or response_token_ids[:response_position] != fixed_response_prefix_ids:
            raise RuntimeError(
                f"Noisy run diverged before the intervention could causally affect generation at row {row_index}, "
                f"position={response_position}, seed={seed}. Tokens 1..p must equal clean exactly."
            )

        pre_diff = float((noisy_capture["pre_noise_hidden_state"] - clean_hidden_state).abs().max().item())
        max_pre_noise_difference = max(max_pre_noise_difference, pre_diff)
        if pre_diff != 0:
            raise RuntimeError(
                f"Noisy trial did not reach the exact same clean pre-intervention state at row {row_index}, "
                f"position={response_position}, seed={seed}: max_diff={pre_diff}."
            )
        if (
            noisy_capture["suffix_decoder_layer_indices"] != suffix_decoder_layer_indices
            or noisy_capture["suffix_state_labels"] != suffix_state_labels
        ):
            raise RuntimeError(
                f"Noisy suffix capture schema differs from clean at row={row_index}, "
                f"position={response_position}, seed={seed}."
            )

        response = decode_response(tokenizer, response_token_ids)
        score = score_response(record, response)
        flipped = bool(math.isfinite(score) and baseline_score <= 0 and score > 0)

        standard_normal_noise_parts.append(standard_normal.unsqueeze(0))
        sampled_noise_parts.append(noisy_capture["sampled_noise"].unsqueeze(0))
        applied_noise_parts.append(noisy_capture["applied_noise"].unsqueeze(0))
        noisy_final_parts.append(noisy_capture["final_hidden_state"].unsqueeze(0))
        noisy_suffix_hidden_parts.append(noisy_capture["suffix_hidden_states"].unsqueeze(0))
        noisy_next_token_logits_parts.append(noisy_capture["next_token_logits"].unsqueeze(0))
        noisy_responses.append(response)
        noisy_response_token_ids.append(response_token_ids)
        noisy_scores.append(score)
        is_w2r.append(flipped)
        noise_hashes.append(hashlib.sha256(sampled_noise.numpy().tobytes()).hexdigest())

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
    noisy_suffix_hidden_tensor = torch.cat(noisy_suffix_hidden_parts, dim=0)
    delta_suffix_hidden_tensor = noisy_suffix_hidden_tensor - clean_suffix_hidden_states.unsqueeze(0)
    noisy_next_token_logits_tensor = torch.cat(noisy_next_token_logits_parts, dim=0)
    expected_shape = (len(noise_seeds), hidden_size)
    expected_suffix_shape = (len(noise_seeds), expected_suffix_depth, hidden_size)
    expected_logits_shape = (len(noise_seeds), vocab_size)
    tensor_shapes = {
        "standard_normal_noise": tuple(standard_normal_noise_tensor.shape),
        "sampled_noise": tuple(sampled_noise_tensor.shape),
        "applied_noise": tuple(applied_noise_tensor.shape),
        "noisy_final_hidden_state": tuple(noisy_final_tensor.shape),
        "noisy_suffix_hidden_states": tuple(noisy_suffix_hidden_tensor.shape),
        "delta_suffix_hidden_states": tuple(delta_suffix_hidden_tensor.shape),
        "noisy_next_token_logits": tuple(noisy_next_token_logits_tensor.shape),
    }
    malformed = {
        name: shape
        for name, shape in tensor_shapes.items()
        if shape
        != (
            expected_suffix_shape
            if "suffix_hidden_states" in name
            else expected_logits_shape
            if "logits" in name
            else expected_shape
        )
    }
    if malformed:
        raise RuntimeError(f"Collected tensor shapes do not match {expected_shape}: {malformed}")
    if not (
        torch.isfinite(clean_hidden_state).all()
        and torch.isfinite(clean_final_hidden_state).all()
        and torch.isfinite(clean_suffix_hidden_states).all()
        and torch.isfinite(clean_next_token_logits).all()
        and torch.isfinite(standard_normal_noise_tensor).all()
        and torch.isfinite(sampled_noise_tensor).all()
        and torch.isfinite(applied_noise_tensor).all()
        and torch.isfinite(noisy_final_tensor).all()
        and torch.isfinite(noisy_suffix_hidden_tensor).all()
        and torch.isfinite(delta_suffix_hidden_tensor).all()
        and torch.isfinite(noisy_next_token_logits_tensor).all()
    ):
        raise RuntimeError(f"Non-finite hidden state or noise collected at row {row_index}, position {response_position}")

    first_suffix_delta_vs_applied_max_diff = float(
        (delta_suffix_hidden_tensor[:, 0, :] - applied_noise_tensor).abs().max().item()
    )
    final_suffix_delta = noisy_final_tensor - clean_final_hidden_state.unsqueeze(0)
    final_suffix_delta_max_diff = float(
        (delta_suffix_hidden_tensor[:, -1, :] - final_suffix_delta).abs().max().item()
    )
    if first_suffix_delta_vs_applied_max_diff != 0 or final_suffix_delta_max_diff != 0:
        raise RuntimeError(
            f"Suffix trace endpoint validation failed at row={row_index}, position={response_position}: "
            f"first_vs_applied={first_suffix_delta_vs_applied_max_diff}, "
            f"final_vs_lm_head={final_suffix_delta_max_diff}."
        )

    coordinate_stds = standard_normal_noise_tensor.std(dim=1, unbiased=False)
    min_coordinate_std = float(coordinate_stds.min().item())
    unique_noise_vectors = len(set(noise_hashes))
    if min_coordinate_std <= 0:
        raise RuntimeError(
            f"A Gaussian trial has zero across-coordinate variance at row={row_index}, "
            f"position={response_position}; this would indicate scalar broadcasting."
        )
    if unique_noise_vectors != len(noise_seeds):
        raise RuntimeError(
            f"Expected {len(noise_seeds)} distinct noise vectors but found "
            f"{unique_noise_vectors} at row={row_index}, position={response_position}."
        )

    diagnostics = {
        "strict_clean_regeneration_matches_original_greedy": True,
        "zero_noise_tokens_match_original_greedy": True,
        "zero_noise_max_pre_hidden_vs_clean_abs_diff": zero_pre_diff,
        "zero_noise_max_applied_abs": zero_applied_max,
        "zero_noise_max_final_hidden_vs_clean_abs_diff": zero_final_diff,
        "zero_noise_max_suffix_hidden_vs_clean_abs_diff": zero_suffix_diff,
        "zero_noise_max_next_token_logits_vs_clean_abs_diff": zero_logits_diff,
        "max_pre_noise_hidden_vs_clean_abs_diff": max_pre_noise_difference,
        "noisy_prefix_through_target_matches_clean": True,
        "first_causally_changeable_response_position": response_position + 1,
        "noise_coordinates_sampled_iid": True,
        "noise_vector_shape": [len(noise_seeds), hidden_size],
        "minimum_standard_normal_across_coordinate_std": min_coordinate_std,
        "unique_noise_vectors": unique_noise_vectors,
        "scalar_broadcast_detected": False,
        "suffix_trace_endpoint_applied_noise_max_diff": first_suffix_delta_vs_applied_max_diff,
        "suffix_trace_endpoint_final_hidden_max_diff": final_suffix_delta_max_diff,
        "suffix_trace_shape_clean": list(clean_suffix_hidden_states.shape),
        "suffix_trace_shape_noisy": list(noisy_suffix_hidden_tensor.shape),
        "clean_next_token_logits_shape": list(clean_next_token_logits.shape),
        "noisy_next_token_logits_shape": list(noisy_next_token_logits_tensor.shape),
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
            "question_fingerprint": record_question_fingerprint,
            "data_source": record["data_source"],
            "ground_truth": record["reward_model"]["ground_truth"],
            "prompt": record["prompt"],
            "injection_layer": layer_idx,
            "response_position": response_position,
            "response_position_fraction": response_position / len(clean_response_token_ids),
            "response_position_indexing": "one_based",
            "clean_response_length": len(clean_response_token_ids),
            "injection_location": "decoder_layer_output/response_token/full_clean_greedy_path_once",
            "first_causally_changeable_response_position": response_position + 1,
            "final_state_location": "lm_head_input/same_target_forward",
            "suffix_capture_scope": "same_target_token_from_injection_layer_through_final_norm",
            "suffix_decoder_layer_indices": suffix_decoder_layer_indices,
            "suffix_decoder_layer_indexing": "zero_based",
            "suffix_state_labels": suffix_state_labels,
            "suffix_trace_tensor_layout": "[state, hidden] clean; [trial, state, hidden] noisy/delta",
            "logits_capture_scope": "raw_lm_head_output_for_response_token_p_plus_1",
            "logits_source_state": "final_norm_lm_head_input/same_target_forward",
            "logits_tensor_layout": "[vocabulary] clean; [trial, vocabulary] noisy",
            "logits_saved_dtype": "float32",
            "vocab_size": vocab_size,
            "noise_distribution": "isotropic_gaussian_iid_per_hidden_coordinate",
            "noise_seed_mode": args.noise_seed_mode,
            "noise_namespace": args.noise_namespace.strip(),
            "noise_coordinate_sampling": "torch.randn(hidden_size), not scalar broadcasting",
            "noise_std": float(args.noise_std),
            "noise_scale_mode": args.noise_scale_mode,
            "clean_hidden_rms": clean_hidden_rms,
            "effective_noise_std_hidden_units": effective_noise_std,
            "model_dtype": args.dtype,
            "trial_batch_size": 1,
            "input_rollout_filter_source": input_rollout_filter_source,
            "scoring_mode": (
                "binary_pass_all_tests"
                if is_code_data_source(record["data_source"])
                else "task_default"
            ),
        },
        "prompt_token_ids": torch.tensor(prompt_token_ids, dtype=torch.long),
        "clean_response_token_ids": torch.tensor(clean_response_token_ids, dtype=torch.long),
        "fixed_response_prefix_token_ids": torch.tensor(fixed_response_prefix_ids, dtype=torch.long),
        "clean_hidden_state": clean_hidden_state,
        "clean_final_hidden_state": clean_final_hidden_state,
        "suffix_decoder_layer_indices": torch.tensor(
            suffix_decoder_layer_indices, dtype=torch.long
        ),
        "suffix_state_labels": suffix_state_labels,
        "clean_suffix_hidden_states": clean_suffix_hidden_states,
        "clean_next_token_logits": clean_next_token_logits,
        "baseline_response": baseline_response,
        "baseline_score": float(baseline_score),
        "zero_noise_control": {
            "trials": 1,
            "all_token_sequences_match_clean": True,
            "all_suffix_hidden_states_match_clean": True,
            "all_next_token_logits_match_clean": True,
            "w2r_count": 0,
            "w2r_rate": 0.0,
        },
        "noise_seeds": torch.tensor(noise_seeds, dtype=torch.long),
        "noise_sha256": noise_hashes,
        "standard_normal_noise": standard_normal_noise_tensor,
        "sampled_noise": sampled_noise_tensor,
        "applied_noise": applied_noise_tensor,
        "noisy_final_hidden_state": noisy_final_tensor,
        "noisy_suffix_hidden_states": noisy_suffix_hidden_tensor,
        "delta_suffix_hidden_states": delta_suffix_hidden_tensor,
        "noisy_next_token_logits": noisy_next_token_logits_tensor,
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
        "question_fingerprint": record_question_fingerprint,
        "data_source": record["data_source"],
        "injection_layer": layer_idx,
        "response_position": response_position,
        "response_position_fraction": response_position / len(clean_response_token_ids),
        "clean_response_length": len(clean_response_token_ids),
        "first_causally_changeable_response_position": response_position + 1,
        "suffix_capture_scope": "same_target_token_from_injection_layer_through_final_norm",
        "suffix_decoder_layer_indices": suffix_decoder_layer_indices,
        "suffix_decoder_layer_indexing": "zero_based",
        "suffix_state_labels": suffix_state_labels,
        "clean_suffix_hidden_states_shape": list(clean_suffix_hidden_states.shape),
        "noisy_suffix_hidden_states_shape": list(noisy_suffix_hidden_tensor.shape),
        "delta_suffix_hidden_states_shape": list(delta_suffix_hidden_tensor.shape),
        "logits_capture_scope": "raw_lm_head_output_for_response_token_p_plus_1",
        "clean_next_token_logits_shape": list(clean_next_token_logits.shape),
        "noisy_next_token_logits_shape": list(noisy_next_token_logits_tensor.shape),
        "logits_saved_dtype": "float32",
        "vocab_size": vocab_size,
        "noise_std": float(args.noise_std),
        "noise_scale_mode": args.noise_scale_mode,
        "noise_seed_mode": args.noise_seed_mode,
        "noise_namespace": args.noise_namespace.strip(),
        "noise_seeds": noise_seeds,
        "noise_coordinate_sampling": "iid_per_hidden_coordinate",
        "clean_hidden_rms": clean_hidden_rms,
        "effective_noise_std_hidden_units": effective_noise_std,
        "num_noise_seeds": len(noise_seeds),
        "zero_noise_control_trials": 1,
        "zero_noise_control_w2r_count": 0,
        "zero_noise_control_w2r_rate": 0.0,
        "baseline_score": float(baseline_score),
        "scoring_mode": (
            "binary_pass_all_tests"
            if is_code_data_source(record["data_source"])
            else "task_default"
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
    if args.noise_batch_size != 1:
        raise ValueError(
            "Strict full-path W2R requires --noise-batch-size 1 so clean and noisy trials use the same "
            "batch-1 greedy computation path. Do not batch noise seeds in this mode."
        )
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0 or args.log_every <= 0:
        raise ValueError("--max-new-tokens, --max-input-tokens, and --log-every must be positive")
    if args.response_position is not None and args.response_position <= 0:
        raise ValueError("--response-position is one-based and must be positive")
    if args.response_position_fraction is not None and not 0 < args.response_position_fraction <= 1:
        raise ValueError("--response-position-fraction must be in (0, 1]")
    if args.num_response_positions is not None and args.num_response_positions <= 0:
        raise ValueError("--num-response-positions must be positive")
    noise_namespace = args.noise_namespace.strip()
    if args.noise_seed_mode == "independent" and not noise_namespace:
        raise ValueError(
            "--noise-namespace is required in independent mode so seeds cannot be "
            "accidentally reused across datasets/experiments"
        )

    explicit_seeds = parse_int_list(args.noise_seeds, "--noise-seeds")
    noise_seed_bank = explicit_seeds or [
        args.base_noise_seed + offset for offset in range(args.num_noise_seeds)
    ]
    row_indices = set(parse_int_list(args.row_indices, "--row-indices")) if args.row_indices.strip() else None
    if any(seed < 0 for seed in noise_seed_bank):
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
        dtype=dtype,
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
    seen_independent_noise_seeds: set[int] = set()

    with manifest_path.open("x", encoding="utf-8") as manifest:
        for row_index, raw_record in enumerate(df.to_dict("records")):
            if row_indices is not None and row_index not in row_indices:
                continue
            record = as_python(raw_record)

            try:
                validate_code_ground_truth(record)
            except Exception as exc:
                exclusions[f"invalid_code_ground_truth:{type(exc).__name__}"] += 1
                continue

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
            record_question_fingerprint = question_fingerprint(record)
            position_noise_seed_map = {
                response_position: resolve_position_noise_seeds(
                    seed_bank=noise_seed_bank,
                    mode=args.noise_seed_mode,
                    noise_namespace=noise_namespace,
                    question_fingerprint=record_question_fingerprint,
                    layer_idx=layer_idx,
                    response_position=response_position,
                )
                for response_position in response_positions
            }
            effective_question_noise_seeds = [
                seed
                for response_position in response_positions
                for seed in position_noise_seed_map[response_position]
            ]
            unique_effective_question_noise_seed_count = len(set(effective_question_noise_seeds))
            if args.noise_seed_mode == "independent":
                if unique_effective_question_noise_seed_count != len(effective_question_noise_seeds):
                    raise RuntimeError(
                        f"Independent mode did not produce unique seeds across all positions at row={row_index}: "
                        f"unique={unique_effective_question_noise_seed_count}, "
                        f"total={len(effective_question_noise_seeds)}"
                    )
                cross_question_collisions = (
                    set(effective_question_noise_seeds) & seen_independent_noise_seeds
                )
                if cross_question_collisions:
                    raise RuntimeError(
                        "Independent mode produced effective seed collisions across questions: "
                        f"{sorted(cross_question_collisions)}"
                    )
                seen_independent_noise_seeds.update(effective_question_noise_seeds)
            completed_positions = 0
            for response_position in response_positions:
                position_noise_seeds = position_noise_seed_map[response_position]
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
                    baseline_response=baseline_response,
                    baseline_score=baseline_score,
                    response_position=response_position,
                    layer_idx=layer_idx,
                    hidden_size=hidden_size,
                    noise_seeds=position_noise_seeds,
                    completed_trials_before=completed_trials,
                )
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
                    "question_fingerprint": record_question_fingerprint,
                    "clean_response_length": len(clean_response_token_ids),
                    "selected_response_positions": response_positions,
                    "effective_noise_seed_count": len(effective_question_noise_seeds),
                    "unique_effective_noise_seed_count": unique_effective_question_noise_seed_count,
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
        "logits_capture_scope": "raw_lm_head_output_for_response_token_p_plus_1",
        "full_vocabulary_logits_saved": True,
        "logit_delta_storage": "not_duplicated; compute noisy_next_token_logits - clean_next_token_logits",
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
        "noise_seed_mode": args.noise_seed_mode,
        "noise_namespace": noise_namespace,
        "noise_seed_bank": noise_seed_bank,
        "noise_seed_derivation": (
            "same seed bank reused across conditions"
            if args.noise_seed_mode == "paired"
            else (
                "sha256(base_seed,noise_namespace,question_fingerprint,layer_idx,"
                "response_position,trial_index) -> 63-bit seed"
            )
        ),
        "noise_coordinate_sampling": "iid torch.randn(hidden_size), never scalar broadcasting",
        "code_scoring_mode": "binary_pass_all_tests",
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
