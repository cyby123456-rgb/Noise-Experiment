# Hidden-state noise experiments (VERL)

This directory is the single entry point for the hidden-state noise project.
It keeps runnable training, inference, causal-probing, and analysis commands
together without duplicating VERL's framework implementation.

## Layout

- `train/run_ppo_hidden_noise.sh`: GRPO/PPO training with optional rollout or
  update hidden-state Gaussian noise.
- `inference/run_generation_hidden_noise.sh`: generation-only paired clean or
  noisy evaluation through `verl.trainer.main_generation`.
- `probe/run_fixed_state_direction_probe.sh`: fixed-state one-shot directional
  Monte-Carlo probe.
- `analysis/run_exact_difficulty_summary.sh`: clean-32-rollout difficulty
  analysis of paired W->R / R->W transitions.
- `configs/noise_fields.yaml`: canonical noise field reference.
- `FRAMEWORK.md`: the exact VERL source files used by these entry points.

## Workflow

1. Run clean generation with `NOISE_STD=0`; retain its parquet.
2. Run one noisy generation per noise seed with exactly the same model, data,
   decoding settings, and rollout count.
3. Analyse paired outputs with the difficulty script.  Difficulty is defined
   *only* by the number of correct clean trajectories out of 32.
4. Use the fixed-state probe only after observing a trajectory-level W->R
   phenomenon.  It tests individual directions, not whole-rollout noise.

All launchers default to the current project's `verl/` source tree.  Set
`VERL_ROOT=/path/to/noise_experiments/verl` to run from the complete backup
snapshot instead.  They expose only noise-specific arguments; pass ordinary
VERL/model/data overrides after `--` or as additional Hydra overrides.

## Safety

These scripts create or overwrite only explicitly named output files.  They do
not delete checkpoints, source code, datasets, or prior experiment outputs.
