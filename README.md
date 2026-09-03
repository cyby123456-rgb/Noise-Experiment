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
- `probe/run_greedy_wrong_gaussian_probe.sh`: generate one complete greedy
  wrong answer, select uniform response positions, and at each position replay
  the clean prefix and inject one Gaussian vector only at that token. A matched
  zero-noise group defines the position-level clean baseline and must reproduce
  its tokens and states exactly; positions whose replay is already correct are
  skipped rather than counted as W2R.
- `probe/run_greedy_wrong_gaussian_probe_v8.py`: strict full-path v8 collector
  that saves clean/noisy suffix-layer states, deltas, logits, responses, scores,
  and auditable per-position manifests.
- `probe/run_greedy_wrong_gaussian_probe_v8_code.sh`: code-dataset launcher.
  Its default `independent` seed mode gives every `(question, position, trial)`
  a distinct Gaussian vector while remaining exactly reproducible.
- `scripts/data/download_prepare_apps.py`: reproducibly shuffle and prepare a
  small APPS parquet without relying on deprecated Hugging Face dataset scripts.
- `analysis/run_exact_difficulty_summary.sh`: clean-32-rollout difficulty
  analysis of paired W->R / R->W transitions.
- `analysis/run_hidden_state_kmeans_pipeline.sh`: batch `main_eval`, raw-space
  KMeans, original-space centroid-distance plots, and shared-PCA plots for
  saved rollout hidden states. Multiple sampled seeds are configured as
  explicit clean/noisy pairs and are analysed separately.
- `configs/noise_fields.yaml`: canonical noise field reference.
- `FRAMEWORK.md`: the exact VERL source files used by these entry points.
- `GREEDY_W2R_GAUSSIAN_PROBE.md`: exact first-stage W2R collection protocol,
  tensor schema, fixed-state checks, and interpretation boundary.

## Workflow

1. Run clean generation with `NOISE_STD=0`; it samples 32 stochastic responses
   per question and `main_eval` writes the aligned per-rollout
   `response_scores` into the same parquet. `ROLLOUT_SEED` is a reproducible
   base seed, not a request to clone one answer 32 times: the code derives a
   distinct seed for every `(dataset row, rollout index)` pair.
2. Run one noisy generation per noise seed with exactly the same model, data,
   decoding settings, and rollout count.
3. Analyse paired outputs with the difficulty script. Difficulty is defined
   *only* by `score > 0` over the existing clean `response_scores` (32
   rollouts); analysis scripts do not independently re-score answers.
4. Use the fixed-state probe only after observing a trajectory-level W->R
   phenomenon.  It tests individual directions, not whole-rollout noise.

For the greedy single-token probe, `NOISE_SCALE_MODE=relative_rms` is the
default. Conditional on the fixed clean state, it samples
`epsilon = noise_std * RMS(h_t) * z`, where `z ~ N(0, I)`. Thus `noise_std=0.1`
means a per-coordinate noise RMS of about 10% of the clean token hidden RMS.
Use `absolute` only when reproducing an older raw-hidden-unit experiment.

For code experiments, correctness means passing every executable test case.
Run generated programs only in an isolated container. The code launcher accepts
`apps`, `taco`, `codecontests`, `codeforces`, and `livecodebench/*` sources and
defaults to independent directions across questions and response positions.

All launchers default to the current project's `verl/` source tree.  Set
`VERL_ROOT=/path/to/noise_experiments/verl` to run from the complete backup
snapshot instead.  They expose only noise-specific arguments; pass ordinary
VERL/model/data overrides after `--` or as additional Hydra overrides.

## Safety

These scripts create or overwrite only explicitly named output files.  They do
not delete checkpoints, source code, datasets, or prior experiment outputs.
