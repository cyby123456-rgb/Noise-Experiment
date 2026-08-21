# VERL implementation map

The executable experiment wrappers live in `noise_experiments/`.  The actual
framework implementation remains in the active project `verl/`, and an exact
full-source backup now exists at `noise_experiments/verl/` for GitHub archival.
The noise-relevant files are:

| Purpose | Canonical implementation |
|---|---|
| Gaussian hidden-state noise and one-shot direction hook | `verl/verl/models/transformers/noise_injection.py` |
| Training rollout/update noise wiring | `verl/verl/trainer/ppo/ray_trainer.py` and `verl/verl/workers/fsdp_workers.py` |
| Validation/generation noise wiring | `verl/verl/workers/fsdp_workers.py` |
| Generation metadata forwarding | `verl/verl/trainer/main_generation.py` |
| Training config defaults | `verl/verl/trainer/config/ppo_trainer.yaml` |
| Generation config defaults | `verl/verl/trainer/config/generation.yaml` |

During ordinary experiments, use the active project `verl/`.  To verify or run
the archived source tree, set `VERL_ROOT` as documented in `ENVIRONMENT.md`.
