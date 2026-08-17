# Environment compatibility

This backup intentionally uses the **same environment contract as the active
POLARIS-main project**, excluding `verl-new`:

- Python package source: `noise_experiments/verl/` is a complete snapshot of
  the project's current `verl/` directory.
- Core Python dependencies: `noise_experiments/verl/requirements.txt`.
- Optional SGLang dependencies: `noise_experiments/verl/requirements_sglang.txt`.
- Package metadata and extras: `noise_experiments/verl/pyproject.toml`.

For existing POLARIS servers, activate the same conda environment already used
for training and generation; do **not** create a second environment solely for
the backup.  Then choose which source tree to import:

```bash
# Use active POLARIS source (default used by launchers)
export VERL_ROOT=/path/to/POLARIS-main/verl

# Or verify the GitHub backup snapshot itself
export VERL_ROOT=/path/to/POLARIS-main/noise_experiments/verl
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH}"
python -c 'import verl; print(verl.__file__)'
```

The GPU stack (PyTorch/CUDA, vLLM, FlashAttention, Ray and any cluster-specific
libraries) must remain the versions from the working POLARIS environment.  It
is not safely reproducible from a generic CPU machine's package export.
