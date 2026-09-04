# First-divergent-token branch replay

This experiment asks whether an observed W2R transition is mediated primarily
by the first different output token or by the continuous hidden/KV state left
by the earlier single-token noise intervention.

## Source data

The first implementation consumes completed strict batch-1 V8 probe outputs.
Every source directory must contain `manifest.jsonl` and its `tensors/*.pt`
files. The shard already stores the exact prompt tokens, clean/noisy responses,
sampled and applied noise vectors, scores, layer, and injection position.

Do not pass V9 parallel code shards. Their clean/noisy trajectories were
generated at a fixed batch size greater than one. Replaying them at batch size
one would invalidate the exact numerical control, so the script rejects them.

## Intervention matrix

For a source trajectory, let `t*` be the zero-based index of its first token
that differs from the clean wrong response. The saved clean and noisy runs give
the two diagonal cells:

| State before `t*` | Token prefix at `t*` | Meaning |
| --- | --- | --- |
| clean / zero-noise | clean | clean baseline |
| original noisy | source trajectory | original W2R or changed-wrong replay |
| clean / zero-noise | source trajectory | token-prefix sufficiency |
| original noisy | clean | conditional token necessity / noisy-state persistence |

The two counterfactual cells are run at each configured prefix length. After
the forced prefix, generation returns to unrestricted greedy decoding.

`t*` is the first *observable discrete* divergence. It is not assumed to be
the first internal causal difference: noisy suffix-layer KV entries may differ
before a token changes.

## Validity gates

Before saving any counterfactual result, the collector requires:

1. batch-1 clean replay exactly matches the saved clean response;
2. the original saved noise exactly reproduces the complete source response;
3. clean pre-noise hidden state and applied noise match the saved tensors
   exactly;
4. every forced token appears at the requested response index;
5. W2R and changed-wrong labels agree with fresh reward evaluation;
6. both source sequences retain at least `MIN_UNFORCED_SOURCE_TOKENS` after the
   longest forced prefix.

The last rule is an answer-leakage guard. It keeps a long free-generation
suffix and reduces the risk that a large `k` trivially copies the terminal
answer; it is not a semantic proof that the prefix contains no answer content.

## Matched control

W2R selection proceeds round-robin across question-position groups before a
second W2R is selected from any group. Each selected W2R is then matched
without replacement to one changed-but-still-wrong trial from the same question
and injection position. Matching first preserves immediate-versus-delayed
divergence, then minimizes divergence-delay and response-length differences.
Unchanged wrong trials are not controls for this experiment because they never
entered another observable branch.

## Smoke test

Use a new output directory for every run. The collector never overwrites an
existing manifest.

```bash
SOURCE_DIRS="/path/to/completed/v8/aime24-result" \
MODEL_PATH=/path/to/exact/source/model \
OUTPUT_DIR=/path/to/output/branch-replay-smoke \
MAX_W2R_TRIALS=2 \
PREFIX_LENGTHS=1,4 \
bash probe/run_first_divergent_token_replay.sh
```

The source model, dtype and `MAX_NEW_TOKENS` must match the original V8 run.
If the source answer hit its generation cap, a different token budget will
fail the exact-reproduction gate.

After the smoke test passes:

```bash
SOURCE_DIRS="/path/to/v8/aime24 /path/to/v8/aime25 /path/to/v8/amc23" \
MODEL_PATH=/path/to/exact/source/model \
OUTPUT_DIR=/path/to/output/branch-replay-50 \
MAX_W2R_TRIALS=50 \
PREFIX_LENGTHS=1,4,16,64 \
MIN_UNFORCED_SOURCE_TOKENS=128 \
bash probe/run_first_divergent_token_replay.sh
```

## Outputs

- `selection.json`: eligible population and the frozen matched pairs;
- `manifest.jsonl`: compact per-episode outcomes;
- `results.csv`: one flat row per episode and prefix length;
- `episodes/*.pt`: complete response token IDs/text, saved noise, and all
  counterfactual generations;
- `summary.json`: aggregate token-sufficiency and state-persistence rates.

For W2R episodes:

- `target_on_clean_correct_rate` estimates trajectory-prefix sufficiency;
- `clean_on_noisy_correct_rate` measures how often the noisy state still
  reaches correctness despite forcing the clean prefix;
- `conditional_token_necessity_rate = 1 - clean_on_noisy_correct_rate`.

Always compare these curves with the matched `changed_wrong` episodes. A rise
that occurs equally for both groups indicates generic branch exploration, not
a correctness-specific branch.
