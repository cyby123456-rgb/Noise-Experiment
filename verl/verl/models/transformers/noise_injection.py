import hashlib
import logging, torch
from typing import Dict, List, Optional, Sequence, Union
from transformers import PreTrainedModel
logger = logging.getLogger(__name__)

def _get_decoder_layers(model: PreTrainedModel):
    for root_name in ("model", "transformer", "decoder", "backbone"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        for layers_name in ("layers", "h", "block"):
            layers = getattr(root, layers_name, None)
            if layers is not None:
                return layers
    raise AttributeError("Cannot locate decoder layers for noise injection")

def register_hidden_state_noise(
    model: PreTrainedModel,
    *,
    std: float,
    layer_idx: Union[int, Sequence[int], None] = None,
    apply_phase: str = "train",
    train_only: Optional[bool] = None,
    all_layers: bool = False,
    seed: Optional[int] = None,
):
    if std is None or std <= 0:
        return None

    if train_only is not None:
        apply_phase = "train" if train_only else "both"
    apply_phase = (apply_phase or "train").lower()
    if apply_phase not in ("train", "eval", "both"):
        raise ValueError(f"Unknown apply_phase={apply_phase}")

    layers = _get_decoder_layers(model)
    num_layers = len(layers)
    target_indices: List[int]
    if all_layers:
        target_indices = list(range(num_layers))
    elif isinstance(layer_idx, (list, tuple)):
        target_indices = []
        for idx in layer_idx:
            if idx is None:
                continue
            idx_int = int(idx)
            if idx_int < 0:
                idx_int += num_layers
            target_indices.append(idx_int)
    else:
        idx = layer_idx if layer_idx is not None else num_layers // 2
        idx = int(idx)
        if idx < 0:
            idx += num_layers
        target_indices = [idx]

    counters: Dict[int, int] = {}

    def _mix_seed(base_seed: int, target_idx: int, counter: int) -> int:
        payload = f"{int(base_seed)}::{int(target_idx)}::{int(counter)}".encode("utf-8", errors="backslashreplace")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)

    def _noise_tensor(x, target_idx: int):
        if seed is None:
            return x + torch.randn_like(x) * std
        counter = counters.get(target_idx, 0)
        counters[target_idx] = counter + 1
        generator = torch.Generator(device=x.device)
        generator.manual_seed(_mix_seed(seed, target_idx, counter))
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=torch.float32).to(dtype=x.dtype)
        return x + noise * std

    def _build_hook(target_idx: int):
        def _hook(module, inputs, output):
            if apply_phase == "train" and not module.training:
                return output
            if apply_phase == "eval" and module.training:
                return output
            if isinstance(output, tuple):
                head, *tail = output
                return (_noise_tensor(head, target_idx), *tail)
            return _noise_tensor(output, target_idx)
        return _hook

    handles: List = []
    for idx in target_indices:
        target_layer = layers[idx]
        handle = target_layer.register_forward_hook(_build_hook(idx))
        handles.append(handle)
    logger.info("Injecting Gaussian noise std=%s on layers %s (phase=%s, seed=%s)", std, target_indices, apply_phase, seed)
    if len(handles) == 1:
        return handles[0]
    return handles


def register_one_shot_hidden_perturbation(
    model: PreTrainedModel,
    *,
    layer_idx: int,
    direction: torch.Tensor,
    target_position: int,
    alpha: float,
):
    """Add one normalized direction to one token during the next model forward.

    This is deliberately distinct from :func:`register_hidden_state_noise`:
    it is for causal fixed-state probes, not whole-rollout Gaussian noise.  It
    modifies only ``target_position`` in the first forward call that reaches
    ``layer_idx``; subsequent decode forwards are left untouched.  The caller
    must remove the returned hook after generation.

    ``alpha`` is a relative radius.  The applied perturbation is
    ``alpha * RMS(hidden[target_position]) * normalize(direction)``.
    The returned diagnostics dictionary is populated when the hook fires.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if direction.ndim not in (1, 2):
        raise ValueError(f"direction must have shape [hidden] or [batch, hidden], got {tuple(direction.shape)}")

    layers = _get_decoder_layers(model)
    num_layers = len(layers)
    layer_idx = int(layer_idx)
    if layer_idx < 0:
        layer_idx += num_layers
    if not 0 <= layer_idx < num_layers:
        raise IndexError(f"layer_idx={layer_idx} is outside [0, {num_layers})")

    diagnostics = {
        "applied": False,
        "layer_idx": layer_idx,
        "target_position": int(target_position),
        "alpha": float(alpha),
        "hidden_rms": None,
        "perturbation_rms": None,
        "direction_norm": None,
    }

    def _hook(module, inputs, output):
        if diagnostics["applied"]:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError(
                "One-shot direction probing requires an unpacked [batch, sequence, hidden] layer output; "
                f"got {type(hidden).__name__} with shape={getattr(hidden, 'shape', None)}."
            )
        position = int(target_position)
        if position < 0:
            position += hidden.shape[1]
        if not 0 <= position < hidden.shape[1]:
            raise IndexError(f"target_position={target_position} is outside sequence length {hidden.shape[1]}")
        if direction.shape[-1] != hidden.shape[-1]:
            raise ValueError(
                f"direction hidden size {direction.shape[-1]} does not match {hidden.shape[-1]}"
            )

        unit_direction = direction.to(device=hidden.device, dtype=torch.float32)
        if unit_direction.ndim == 1:
            unit_direction = unit_direction.unsqueeze(0).expand(hidden.shape[0], -1)
        elif unit_direction.shape[0] != hidden.shape[0]:
            raise ValueError(
                f"Batched direction count {unit_direction.shape[0]} does not match model batch size {hidden.shape[0]}"
            )
        direction_norm = torch.linalg.vector_norm(unit_direction, dim=-1, keepdim=True)
        if not torch.isfinite(direction_norm).all() or (direction_norm == 0).any():
            raise ValueError("direction must have a finite non-zero norm")
        unit_direction = unit_direction / direction_norm

        token_hidden = hidden[:, position, :]
        # One scalar radius per sequence makes alpha directly comparable across
        # states while preserving one fixed direction for the whole batch item.
        hidden_rms = token_hidden.float().square().mean(dim=-1, keepdim=True).sqrt()
        perturbation = (float(alpha) * hidden_rms * unit_direction).to(dtype=hidden.dtype)
        modified = hidden.clone()
        modified[:, position, :] = token_hidden + perturbation

        diagnostics.update(
            {
                "applied": True,
                "hidden_rms": hidden_rms.mean().item(),
                "perturbation_rms": perturbation.float().square().mean(dim=-1).sqrt().mean().item(),
                "direction_norm": direction_norm.mean().item(),
            }
        )
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = layers[layer_idx].register_forward_hook(_hook)
    return handle, diagnostics
