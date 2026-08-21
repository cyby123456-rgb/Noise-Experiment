import contextlib
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from verl.models.transformers.noise_injection import _get_decoder_layers

logger = logging.getLogger(__name__)


def resolve_weight_noise_std(initial_sigma: float, global_step: int, total_steps: Optional[int], min_sigma: float = 0.0) -> float:
    try:
        sigma = float(initial_sigma)
    except (TypeError, ValueError):
        return 0.0

    if sigma <= 0:
        return 0.0

    try:
        min_sigma = float(min_sigma)
    except (TypeError, ValueError):
        min_sigma = 0.0

    if total_steps is None:
        return max(min_sigma, sigma)

    try:
        total_steps = int(total_steps)
        global_step = int(global_step)
    except (TypeError, ValueError):
        return max(min_sigma, sigma)

    if total_steps <= 0:
        return max(min_sigma, sigma)

    ratio = max(0.0, 1.0 - (global_step / total_steps))
    return max(min_sigma, sigma * ratio)


def resolve_decoder_layer_indices(model, num_layers: Optional[int] = None, layer_indices: Optional[Sequence[int]] = None, all_layers: bool = False) -> List[int]:
    layers = _get_decoder_layers(model)
    total_layers = len(layers)

    if all_layers:
        return list(range(total_layers))

    if layer_indices not in (None, "null"):
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        resolved = []
        for idx in layer_indices:
            idx = int(idx)
            if idx < 0:
                idx += total_layers
            if 0 <= idx < total_layers:
                resolved.append(idx)
        return sorted(set(resolved))

    try:
        num_layers = int(num_layers or 0)
    except (TypeError, ValueError):
        num_layers = 0

    if num_layers <= 0:
        return []

    num_layers = min(num_layers, total_layers)
    return list(range(total_layers - num_layers, total_layers))


def get_weight_noise_named_parameters(model, num_layers: Optional[int] = None, layer_indices: Optional[Sequence[int]] = None, all_layers: bool = False) -> List[Tuple[str, torch.nn.Parameter]]:
    layers = _get_decoder_layers(model)
    target_indices = resolve_decoder_layer_indices(model, num_layers=num_layers, layer_indices=layer_indices, all_layers=all_layers)
    named_params: List[Tuple[str, torch.nn.Parameter]] = []
    seen = set()
    for idx in target_indices:
        for name, param in layers[idx].named_parameters(recurse=True):
            if id(param) in seen:
                continue
            if param is None or not torch.is_floating_point(param):
                continue
            seen.add(id(param))
            named_params.append((f"layers.{idx}.{name}", param))
    return named_params


def get_weight_noise_parameter_name_suffixes(model, num_layers: Optional[int] = None, layer_indices: Optional[Sequence[int]] = None, all_layers: bool = False) -> List[str]:
    return [name for name, _ in get_weight_noise_named_parameters(model, num_layers=num_layers, layer_indices=layer_indices, all_layers=all_layers)]


def build_functional_weight_noise_overrides(
    model,
    *,
    sigma: float,
    noise_seed: int,
    num_layers: Optional[int] = None,
    layer_indices: Optional[Sequence[int]] = None,
    all_layers: bool = False,
) -> Dict[str, torch.Tensor]:
    if sigma <= 0:
        return {}

    suffixes = get_weight_noise_parameter_name_suffixes(
        model,
        num_layers=num_layers,
        layer_indices=layer_indices,
        all_layers=all_layers,
    )
    if not suffixes:
        return {}

    suffix_set = set(suffixes)
    overrides: Dict[str, torch.Tensor] = {}
    for full_name, param in model.named_parameters():
        matched_suffix = None
        for suffix in suffix_set:
            if full_name == suffix or full_name.endswith("." + suffix):
                matched_suffix = suffix
                break
        if matched_suffix is None:
            continue
        generator = torch.Generator(device=param.device)
        generator.manual_seed(mix_noise_seed(noise_seed, full_name))
        noise = torch.randn(
            param.shape,
            generator=generator,
            device=param.device,
            dtype=torch.float32,
        ).to(dtype=param.dtype)
        overrides[full_name] = param + noise * sigma
    return overrides


def mix_noise_seed(base_seed: int, name: str) -> int:
    payload = f"{int(base_seed)}::{name}".encode("utf-8", errors="backslashreplace")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


@dataclass
class AppliedWeightNoise:
    name: str
    parameter: torch.nn.Parameter
    noise: torch.Tensor


class TemporaryGaussianWeightNoise(contextlib.AbstractContextManager):
    def __init__(
        self,
        model,
        *,
        sigma: float,
        noise_seed: int,
        num_layers: Optional[int] = None,
        layer_indices: Optional[Sequence[int]] = None,
        all_layers: bool = False,
    ):
        self.model = model
        self.sigma = float(sigma)
        self.noise_seed = int(noise_seed)
        self.num_layers = num_layers
        self.layer_indices = layer_indices
        self.all_layers = all_layers
        self._applied: List[AppliedWeightNoise] = []

    def __enter__(self):
        if self.sigma <= 0:
            return self

        named_params = get_weight_noise_named_parameters(
            self.model,
            num_layers=self.num_layers,
            layer_indices=self.layer_indices,
            all_layers=self.all_layers,
        )
        if not named_params:
            return self

        with torch.no_grad():
            for name, param in named_params:
                if param.device.type == "meta":
                    continue
                generator = torch.Generator(device=param.device)
                generator.manual_seed(mix_noise_seed(self.noise_seed, name))
                noise = torch.randn(
                    param.shape,
                    generator=generator,
                    device=param.device,
                    dtype=torch.float32,
                ).to(dtype=param.dtype)
                noise.mul_(self.sigma)
                param.add_(noise)
                self._applied.append(AppliedWeightNoise(name=name, parameter=param, noise=noise))

        logger.info("Applied Gaussian weight noise sigma=%s to %d tensors", self.sigma, len(self._applied))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._applied:
            return False

        with torch.no_grad():
            for item in reversed(self._applied):
                item.parameter.sub_(item.noise)
        self._applied.clear()
        return False


def build_weight_noise_state(config: Dict, global_step: int, total_training_steps: Optional[int]) -> Dict:
    if not config:
        return {"enabled": False}

    sigma = resolve_weight_noise_std(
        initial_sigma=config.get("initial_sigma", 0.0),
        global_step=global_step,
        total_steps=config.get("total_decay_steps", total_training_steps),
        min_sigma=config.get("min_sigma", 0.0),
    )
    return {
        "enabled": sigma > 0,
        "sigma": sigma,
        "num_layers": config.get("num_layers", 0),
        "layer_indices": config.get("layer_indices"),
        "all_layers": bool(config.get("all_layers", False)),
    }
