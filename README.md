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
- `probe/run_greedy_wrong_gaussian_probe_v8.py`: strict batch-1 math/non-code
  collector. It rejects code datasets so serial math and parallel code results
  cannot be accidentally mixed.
- `probe/run_greedy_wrong_gaussian_probe_v9_code_parallel.py`: code-only
  collector that evaluates multiple independent noisy rollouts in one GPU
  batch and saves suffix-layer states, deltas, logits, responses, scores, and
  auditable controls.
- `probe/run_greedy_wrong_gaussian_probe_v9_code_parallel.sh`: launcher for the parallel
  code collector. Its default `independent` seed mode gives every
  `(question, position, trial)` a distinct Gaussian vector. Every noisy GPU
  batch also contains a matched zero-noise row, and the collector aborts if
  that row or any pre-intervention prefix differs from clean greedy.
- `probe/run_greedy_wrong_gaussian_probe_v8_code.sh`: compatibility alias that
  immediately forwards to the v9 parallel launcher; it no longer contains a
  serial code experiment.
- `probe/run_two_scripts_2gpu.sh`: assigns two different shell scripts to two
  GPUs, launches them concurrently, keeps separate logs, and propagates either
  worker's failure status.
- `probe/run_first_divergent_token_replay.py`: consumes completed strict
  batch-1 V8 shards, exactly reproduces selected W2R and matched changed-wrong
  trajectories, and runs the clean/noisy-state x clean/source-token-prefix
  counterfactual replay. The launcher is
  `probe/run_first_divergent_token_replay.sh`; the full protocol is documented
  in `BRANCH_REPLAY_EXPERIMENT.md`.
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
`NOISE_BATCH_SIZE=8` means eight noisy rollouts plus one matched zero-control
row per model call; the clean greedy baseline is also generated with that same
9-row execution shape. This avoids treating batch-shape numerical drift as a
noise effect. `NUM_NOISE_SEEDS` must be divisible by the noisy batch size.
Parallel code outputs use `greedy_gaussian_w2r_parallel/` and a v9 format
identifier. Treat earlier serial code outputs as a separate pilot rather than
pooling them with this dataset.

The code launcher defaults to `ENABLE_THINKING=false`. This is passed into the
chat template as `enable_thinking=False`, which avoids Qwen3's unsupported
thinking-plus-greedy combination. It also preserves all EOS IDs from the model
generation configuration instead of replacing them with one tokenizer ID.

A minimal one-position smoke test on one code question is:

```bash
INPUT_PARQUET=/path/to/apps_competition_test_100.parquet \
MODEL_PATH=/path/to/Qwen3-8B \
MAX_QUESTIONS=1 \
NUM_RESPONSE_POSITIONS=1 \
NUM_NOISE_SEEDS=8 \
NOISE_BATCH_SIZE=8 \
bash probe/run_greedy_wrong_gaussian_probe_v9_code_parallel.sh
```

After that succeeds, use `MAX_QUESTIONS=20`, `NUM_RESPONSE_POSITIONS=10`, and
`NUM_NOISE_SEEDS=32`. Start with `NOISE_BATCH_SIZE=8`; try 16 only after an
8-row smoke test fits GPU memory. `MODEL_PATH` is required explicitly so a code
run cannot silently fall back to the math checkpoint.

To launch two already-configured, different scripts on separate GPUs:

```bash
bash probe/run_two_scripts_2gpu.sh \
  probe/first_experiment.sh \
  probe/second_experiment.sh
```

The first script receives `CUDA_VISIBLE_DEVICES=0`; the second receives
`CUDA_VISIBLE_DEVICES=1`. Override those physical IDs with `GPU0_ID` and
`GPU1_ID`. Both child scripts must respect the inherited variable (for example,
`export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"`) rather than
unconditionally resetting it. Each child remains responsible for using a
distinct experiment/output directory.

All launchers default to the current project's `verl/` source tree.  Set
`VERL_ROOT=/path/to/noise_experiments/verl` to run from the complete backup
snapshot instead.  They expose only noise-specific arguments; pass ordinary
VERL/model/data overrides after `--` or as additional Hydra overrides.

## Safety

These scripts create or overwrite only explicitly named output files.  They do
not delete checkpoints, source code, datasets, or prior experiment outputs.
