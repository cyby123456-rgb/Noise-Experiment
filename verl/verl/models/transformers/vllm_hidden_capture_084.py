"""Runtime hidden-state capture for VERL's vLLM 0.8.4 V0 model runner.

This module is deliberately version-locked.  It captures states *inside* the
actual vLLM forward, rather than replaying generated text with a Hugging Face
model.  Only eager V0 execution is supported: CUDA graphs can replace the
eager model object and would make a layer hook ambiguous.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class _CaptureState:
    enabled: bool = False
    layers: tuple[int, ...] = ()
    response_positions: tuple[int, ...] = ()
    hidden_size: int | None = None
    # request id -> most recently processed response position
    response_position: dict[str, int] = field(default_factory=dict)
    # request id -> layer -> response position -> CPU tensor
    records: dict[str, dict[int, dict[int, torch.Tensor]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )

    def reset(self, *, layers: list[int], response_positions: list[int], hidden_size: int) -> None:
        self.enabled = True
        self.layers = tuple(layers)
        self.response_positions = tuple(response_positions)
        self.hidden_size = int(hidden_size)
        self.response_position.clear()
        self.records.clear()

    def clear(self) -> None:
        self.enabled = False
        self.response_position.clear()
        self.records.clear()


def _decoder_layers(model: torch.nn.Module):
    """Locate vLLM Qwen/Llama decoder blocks without model-type assumptions."""
    for root_name in ("model", "transformer", "decoder", "backbone"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        for layers_name in ("layers", "h", "block"):
            layers = getattr(root, layers_name, None)
            if layers is not None:
                return layers
    raise AttributeError("Cannot locate decoder layers on the vLLM model")


def _hidden_tensor(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
        raise RuntimeError(
            "vLLM hidden capture expected a [tokens, hidden] or "
            f"[batch, tokens, hidden] layer output, got {type(value).__name__} "
            f"with shape={getattr(value, 'shape', None)}"
        )
    return value


def _request_ids(model_input: Any, selected_count: int) -> list[str]:
    request_map = getattr(model_input, "request_ids_to_seq_ids", None)
    if not request_map:
        raise RuntimeError("vLLM did not provide request_ids_to_seq_ids for hidden-state capture")
    request_ids = [str(request_id) for request_id, seq_ids in request_map.items() for _ in seq_ids]
    if len(request_ids) != selected_count:
        raise RuntimeError(
            "Cannot safely map vLLM selected states to requests: "
            f"{selected_count} selected states but {len(request_ids)} sequences. "
            "This capture patch supports n=1 only."
        )
    return request_ids


def install_vllm_hidden_capture_084() -> None:
    """Install the process-local hook around vLLM 0.8.4's V0 ModelRunner."""
    import vllm
    from vllm.worker.model_runner import ModelRunner

    if getattr(ModelRunner, "_verl_hidden_capture_084_installed", False):
        return
    if str(vllm.__version__) != "0.8.4":
        raise RuntimeError(
            "VERL hidden-state capture is pinned to vLLM 0.8.4, "
            f"but found vLLM {vllm.__version__}."
        )

    original_execute_model = ModelRunner.execute_model

    def execute_model_with_hidden_capture(self, model_input, kv_caches, *args, **kwargs):
        capture: _CaptureState | None = getattr(self, "_verl_hidden_capture_084", None)
        if capture is None or not capture.enabled:
            return original_execute_model(self, model_input, kv_caches, *args, **kwargs)

        if not getattr(model_input, "is_prompt", False):
            decode_meta = getattr(getattr(model_input, "attn_metadata", None), "decode_metadata", None)
            if decode_meta is not None and getattr(decode_meta, "use_cuda_graph", False):
                raise RuntimeError("Hidden-state capture requires rollout.enforce_eager=true")

        sampling_metadata = getattr(model_input, "sampling_metadata", None)
        selected_indices = getattr(sampling_metadata, "selected_token_indices", None)
        if selected_indices is None:
            raise RuntimeError("vLLM did not expose selected_token_indices for hidden-state capture")
        selected_indices = selected_indices.detach()
        request_ids = _request_ids(model_input, len(selected_indices))

        is_prompt = bool(getattr(model_input, "is_prompt", False))
        positions = []
        for request_id in request_ids:
            position = 0 if is_prompt else capture.response_position.get(request_id, 0) + 1
            positions.append(position)
            capture.response_position[request_id] = position

        decoder_layers = _decoder_layers(self.model)
        handles = []
        for layer_id in capture.layers:
            if not 0 <= layer_id < len(decoder_layers):
                raise IndexError(f"Requested capture layer {layer_id} is outside [0, {len(decoder_layers) - 1}]")

            def make_hook(current_layer_id: int):
                def hook(_, __, output):
                    hidden = _hidden_tensor(output)
                    if hidden.ndim != 2:
                        raise RuntimeError(
                            "vLLM 0.8.4 V0 capture expects packed [tokens, hidden] "
                            f"states, got {tuple(hidden.shape)}. Refusing to guess the request mapping."
                        )
                    if int(selected_indices.max().item()) >= hidden.shape[0]:
                        raise RuntimeError(
                            "selected_token_indices exceed captured layer output; "
                            "the vLLM runner layout is not the expected 0.8.4 V0 layout."
                        )
                    selected = hidden.index_select(0, selected_indices).detach().to("cpu", dtype=torch.float16)
                    for row, (request_id, position) in enumerate(zip(request_ids, positions, strict=True)):
                        if position in capture.response_positions:
                            capture.records[request_id][current_layer_id][position] = selected[row].clone()

                return hook

            handles.append(decoder_layers[layer_id].register_forward_hook(make_hook(layer_id)))

        try:
            return original_execute_model(self, model_input, kv_caches, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

    ModelRunner.execute_model = execute_model_with_hidden_capture
    ModelRunner._verl_hidden_capture_084_installed = True


def begin_capture(model_runner: Any, *, layers: list[int], response_positions: list[int]) -> None:
    """Start one synchronous ``LLM.generate`` capture session."""
    from vllm.worker.model_runner import ModelRunner

    if not isinstance(model_runner, ModelRunner):
        raise RuntimeError(
            "Hidden-state capture requires vLLM 0.8.4 V0 ModelRunner, but got "
            f"{type(model_runner).__module__}.{type(model_runner).__name__}. "
            "Set VLLM_USE_V1=0 before importing vLLM."
        )
    install_vllm_hidden_capture_084()
    if 0 not in response_positions:
        raise ValueError(
            "response_positions must include 0 (the prompt-final state) so request mapping can be verified"
        )
    if not getattr(model_runner, "enforce_eager", True):
        raise RuntimeError("Hidden-state capture requires rollout.enforce_eager=true")
    hidden_size = int(getattr(model_runner.model, "hidden_size", 0) or getattr(model_runner.model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise RuntimeError("Cannot determine vLLM model hidden size")
    state = getattr(model_runner, "_verl_hidden_capture_084", None)
    if state is None:
        state = _CaptureState()
        model_runner._verl_hidden_capture_084 = state
    state.reset(layers=layers, response_positions=response_positions, hidden_size=hidden_size)


def end_capture(model_runner: Any, request_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [request, layer, position, hidden] states and presence mask."""
    state: _CaptureState | None = getattr(model_runner, "_verl_hidden_capture_084", None)
    if state is None or not state.enabled or state.hidden_size is None:
        raise RuntimeError("Hidden-state capture was not active")
    values = torch.full(
        (len(request_ids), len(state.layers), len(state.response_positions), state.hidden_size),
        float("nan"),
        dtype=torch.float16,
    )
    present = torch.zeros((len(request_ids), len(state.response_positions)), dtype=torch.bool)
    for request_index, request_id in enumerate(request_ids):
        records = state.records.get(str(request_id), {})
        for layer_index, layer_id in enumerate(state.layers):
            for position_index, position in enumerate(state.response_positions):
                value = records.get(layer_id, {}).get(position)
                if position == 0 and value is None:
                    raise RuntimeError(
                        "Missing mandatory prompt-final hidden state for request "
                        f"{request_id}, layer {layer_id}. The vLLM request mapping "
                        "did not round-trip; refusing to write mispaired states."
                    )
                if value is not None:
                    values[request_index, layer_index, position_index] = value
                    present[request_index, position_index] = True
    state.clear()
    return values, present
