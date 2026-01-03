# NFP

NFP is a Lean 4 project for **mathematically rigorous** reasoning about transformer-style computations, with a focus on mechanistic interpretability (e.g. induction heads) and provable norm/error bounds.

NFP stands for **Neural Formal Pathways**.

This repo contains:

- A **Lean library** (under `Nfp/`) for finite probability and a lightweight “transformer semantics” layer.
- A **CLI executable** (`lake exe nfp …`) that loads transformer weights stored in a compact binary format (`.nfpt`) and produces rigorous bounds and diagnostics.

> Goal: *no “hand-wavy” numerics in the bound path.* Heuristic estimates (e.g. power iteration) may exist for diagnostics, but the bounds reported as “rigorous” are computed via conservative inequalities.

## Status

This is research tooling. Interfaces may change; please treat results as experimental unless they are backed by a certificate/check you trust.

## Tabula Rasa Rewrite (current state)

The `tabula-rasa` branch is a fresh, minimal Lean 4 core focused on circuit certification.

Current core modules (new):
- `Nfp/Core`, `Nfp/Prob`, `Nfp/Mixer`, `Nfp/System` define basic mass/probability, mixers, and DAG-backed local systems.
- `Nfp/Circuit` defines DAG-based circuits with typed interfaces, well-formedness, and equivalence checkers.
- `Nfp/Circuit/Compose` adds sequential and residual wiring combinators for typed circuits.
- `Nfp/Circuit/Layers/Attention` contains Q/K/V projection wiring plus an attention score/mixing core.
- `Nfp/Circuit/Layers/Induction` provides induction-head specs and the core attention one-hot lemma.
- `Nfp/Circuit/Layers/TransformerBlock` wires LN/attention/MLP into a GPT-style block skeleton.
- `Nfp/Cli` and `Main.lean` remain thin placeholders (no full transformer pipeline yet).

Module map and invariants are tracked in `AGENTS.md`.

## Induction Certification (prototype)

The current prototype checks **head-level induction certificates** and can optionally compose
them with a **downstream error bound**. Certificates are produced by **untrusted** helper scripts
and verified by the CLI.

Generate certificates (untrusted):

```bash
python scripts/build_gpt2_induction_cert.py \
  --output reports/gpt2_induction.cert \
  --layer 5 --head 1 --seq 32 --pattern-length 16 \
  --values-out reports/gpt2_induction.values --value-dim 0 \
  --active-eps-max 1/2
```

To produce value-range certificates aligned with a logit-diff direction, add:

```
--direction-target <token_id> --direction-negative <token_id>
```

Verify it (trusted checker):

```bash
lake exe nfp induction certify --scores reports/gpt2_induction.cert \
  --values reports/gpt2_induction.values
```

You can enforce non-vacuity checks with:

```
--min-margin <rat>   --max-eps <rat>   --min-active <n>   --min-logit-diff <rat>
```

To recompute `eps`/`margin` and `lo`/`hi` inside Lean (sound builder), run:

```bash
lake exe nfp induction certify_sound --scores reports/gpt2_induction.cert \
  --values reports/gpt2_induction.values
```

To compute scores/values inside Lean from exact head inputs, run:

```bash
lake exe nfp induction certify_head --inputs reports/gpt2_induction.head
```

To add a downstream error bound (end-to-end check), supply a downstream certificate
that records a nonnegative error bound computed externally:

```bash
python scripts/build_downstream_linear_cert.py \
  --output reports/gpt2_downstream.cert \
  --gain 3/2 --input-bound 5/4

lake exe nfp induction certify_end_to_end --scores reports/gpt2_induction.cert \
  --values reports/gpt2_induction.values --downstream reports/gpt2_downstream.cert
```

To build a head-input file from an exported `.nfpt` binary:

```bash
python scripts/build_gpt2_head_inputs.py --model models/gpt2_rigorous.nfpt \
  --layer 5 --head 1 --direction-target 17850 --direction-negative 31215 \
  --output reports/gpt2_induction.head
```

This extractor is **untrusted** and currently ignores LN/bias terms, so treat it as a
convenience path for exercising the `certify_head` pipeline rather than a full
end-to-end verification of GPT-2 internals.

Softmax-margin certificate format (line-oriented):

```
seq <n>
eps <rat>
margin <rat>
active <q>
prev <q> <k>
score <q> <k> <rat>
weight <q> <k> <rat>
```

`active <q>` lines declare the queries on which the bounds are required; if omitted,
the checker defaults to all nonzero queries.

Value-range certificate format (line-oriented):

```
seq <n>
direction-target <tok_id>
direction-negative <tok_id>
lo <rat>
hi <rat>
val <k> <rat>
```

`direction-*` lines are optional metadata for directional (logit-diff) values.

Downstream linear certificate format (line-oriented):

```
error <rat>
gain <rat>
input-bound <rat>
```

The checker enforces `error = gain * input-bound` and nonnegativity of all fields.

Head input format for `certify_head` (line-oriented):

```
seq <n>
d_model <n>
d_head <n>
scale <rat>
direction-target <tok_id>
direction-negative <tok_id>
direction <d> <rat>
active <q>
prev <q> <k>
embed <q> <d> <rat>
wq <i> <j> <rat>
wk <i> <j> <rat>
wv <i> <j> <rat>
wo <i> <j> <rat>
```

All `direction`, `embed`, and projection matrices must be fully specified. If no
`active` lines appear, the checker defaults to all nonzero queries.

The checker derives a softmax tolerance from the score margins and validates the value-range
bounds. The CLI reports a tolerance `eps * (hi - lo)` for the approximate induction spec.

For tighter, non-vacuous bounds, use `--active-eps-max` when building the certificate to restrict
`active` queries to positions with small `eps` (at the cost of fewer certified positions).
You can enforce a minimum active coverage at check time with `--min-active <n>`.
The default minimum is `max 1 (seq / 8)` when the flag is omitted.

`certify`/`certify_sound`/`certify_end_to_end` also accept `--min-margin` and `--max-eps` to
reject vacuous score gaps or overly large tolerances (defaults: `0` and `1/2`).

If the value-range certificate is built from a logit-diff direction (see below),
the checker also reports `logitDiffLB`. When `direction-target`/`direction-negative`
metadata is present, the checker defaults `--min-logit-diff` to `0` to avoid
vacuous directional bounds. You can override with a higher rational literal.

`certify-sound` ignores any supplied `eps`/`margin`/`lo`/`hi` lines and recomputes
those bounds from the raw entries.

`certify_head` reads a single input file with exact head inputs (embeddings,
projection weights, direction vector, and scale) and recomputes
scores/values inside Lean.

## Soundness statement (what is proven vs checked)

The Lean library defines the core math objects (finite probability, mixers, linearizations, and operator-norm-style bounds) and proves a number of lemmas about them. The CLI sound path produces certificates using exact `Rat` arithmetic and a trusted checker that verifies internal arithmetic relationships between certificate fields.

At present, the checker does **not** include a bridge theorem that connects certificate validity to
Lean-defined Jacobian bounds (for example, a theorem of the form `||layerJacobian - I|| <= C`).
The downstream error certificate is only checked for internal arithmetic consistency.
Treat sound certificates as **internally consistent bound reports**, not as a fully formal
end-to-end verification of transformer Jacobians.

Margin-based softmax tightening exists, but only **best-match margin evidence** is accepted today. Direct `--softmaxMargin` is rejected by the checker, and best-match logit bounds are generated in untrusted code and only checked for internal consistency.

For known gaps and ongoing upgrades, see `SOUNDNESS_LIMITATIONS.md`.

## North Star

NFP’s long-term direction is **verified circuit discovery**:

- Use fast, exploratory tooling to **propose** candidate circuits (e.g. induction-style head interactions),
- then produce **checkable evidence** (bounds / certificates) that a skeptical reader can re-run and validate.

Concretely, the intended split is:

- **Discovery / exploration (untrusted, fast):**
  Heuristic search, ranking, and diagnostics are allowed here (and should be clearly labelled as such).
  This includes things like candidate search (`induction`) and comparison estimates printed under diagnostics/verbose flags.

- **Certification / checking (trusted, boring):**
  Anything described as “rigorous” should be justified by conservative inequalities or by a certificate that a checker can validate.
  The long-term aim is that Lean does as little “real inference” as possible: instead of running large forward passes,
  it should mostly **check small, structured proof obligations** (e.g. inequality chains, norm bounds, interval/rational arithmetic).

Current state: `certify` is already an example of this direction (sound-mode reporting using exact `Rat` arithmetic rather than trusted floats),
but the certificate story is still evolving and interfaces may change.

Model trajectory: GPT-2 support is currently a proving ground for the end-to-end workflow (export → analyze/search → bound/certify).
The goal is to gradually cover more modern decoder blocks (e.g. RoPE-style position handling) while keeping the certification/checking layer lightweight.

## Reproduce results

Minimal local demo (no network needed):

```bash
lake build -q --wfail
lake build nfp -q --wfail
lake exe nfp certify tests/fixtures/tiny_sound_binary.nfpt \
  --output reports/tiny_sound_demo.txt
```

Expected artifacts:
- `reports/tiny_sound_demo.txt`

Optional (rebuild the tiny binary from text fixtures and run a fixed induction cert):

```bash
./scripts/demo_tiny_local_binary.sh
./scripts/demo_tiny_induction_cert.sh
```

Expected artifacts (optional path):
- `reports/tiny_sound_local_binary.txt`
- `reports/tiny_induction_cert.txt`

End-to-end GPT-2 demo (requires network/model download):

```bash
./scripts/demo_gpt2_sound.sh
./scripts/demo_gpt2_induction_sound.sh
```

Expected artifacts:
- `reports/gpt2_sound_demo.txt`
- `reports/gpt2_induction_sound_scan.txt`

Notes:
- If a legacy `.nfpt` header is missing `gelu_kind`, `demo_gpt2_sound.sh` writes
  `models/gpt2_with_gelu_kind.nfpt` and uses that for certification.
- `demo_gpt2_induction_sound.sh` can take a while on CPU; use `--top 1`,
  `--fast`, or `--jobs 2` to shorten the scan or run it on a larger machine.
- You can also set `NFP_BIN=./.lake/build/bin/nfp` to avoid repeated `lake exe`
  startup overhead.


## Requirements

- **Lean 4** (pinned by `lean-toolchain`) and **Lake**.
  - Easiest install: `elan` (Lean toolchain manager).
- A standard build toolchain for Lean (C/C++ compiler, `make`, etc.).
- (Optional) **Python** for the export scripts in `scripts/`.

Lean version is pinned in `lean-toolchain` (currently `leanprover/lean4:v4.26`).

## Getting started

Clone and build:

```bash
lake update
lake build
```

Run the CLI (see subcommands below):

```bash
lake exe nfp --help
```

## Models

The CLI expects a model file in **`.nfpt`** format (NFP_BINARY_V1).
Most commands (analysis/induction/diagnostics) require `NFP_BINARY_V1`; legacy `NFP_TEXT_V1/V2`
is supported only for local SOUND certification.

- Create a local `models/` directory and place your `.nfpt` files there (the repo does not version model files; the author’s setup may have used local symlinks).
- You can export GPT-2 weights from Hugging Face using the scripts in `scripts/`.

`.nfpt` files use a small text header followed by a binary payload:

```
NFP_BINARY_V1
num_layers=...
num_heads=...
model_dim=...
head_dim=...
hidden_dim=...
vocab_size=...
seq_len=...
layer_norm_eps=...
gelu_kind=...
BINARY_START
```

The payload is raw little-endian bytes in a fixed order (tokens, embeddings, then weights).

Notes:
- `layer_norm_eps` (or legacy `eps`) and `gelu_kind` (or legacy `gelu_deriv`) are required for
  SOUND certification.
- Global sound certification supports `NFP_BINARY_V1`. Local sound certification supports
  `NFP_BINARY_V1` (fixed-point union-box) and legacy `NFP_TEXT_V1/V2`.

### Exporting GPT-2 to `.nfpt`

The export scripts use `torch` + `transformers`.

Example (write `models/gpt2_rigorous.nfpt`):

```bash
python scripts/export_gpt2.py models/gpt2_rigorous.nfpt
```

If you prefer a locked Python environment, use `uv` or a venv and install dependencies from `pyproject.toml`:

```bash
uv run python scripts/export_gpt2.py models/gpt2_rigorous.nfpt
```

### GPT-2 sound demo (global)

This demo downloads GPT-2 weights on demand, exports a binary `.nfpt`, and runs the
global sound certificate.

```bash
./scripts/demo_gpt2_sound.sh
```

Artifacts:
- `models/gpt2.nfpt` (binary export)
- `reports/gpt2_sound_demo.txt` (sound certificate report)

### GPT-2 induction sound scan

This demo builds the rigorous induction dataset (if needed), finds candidate
induction head pairs, and ranks them by sound logit-diff lower bounds.

```bash
./scripts/demo_gpt2_induction_sound.sh
```

Artifacts:
- `models/gpt2_rigorous.nfpt` (binary export)
- `reports/gpt2_induction_sound_scan.txt` (sound scan report)

### Tiny local binary demo

This demo converts the tiny text fixtures into a binary `.nfpt` and runs a local
sound certificate (with `--delta`).

```bash
./scripts/demo_tiny_local_binary.sh
```

Artifacts:
- `tests/fixtures/tiny_sound_binary.nfpt` (binary fixture)
- `reports/tiny_sound_local_binary.txt` (local sound certificate report)

### Tiny induction cert demo

This demo computes a minimal induction head certificate on the tiny fixture.

```bash
./scripts/demo_tiny_induction_cert.sh
```

Artifacts:
- `reports/tiny_induction_cert.txt` (induction cert report)

## CLI overview

The main entrypoint is:

```bash
lake exe nfp <command> [args] [flags]
```

By default, `nfp` mirrors everything printed to stdout into `logs/` as a timestamped `.log` file.

### `analyze`

Runs the default end-to-end analysis for the supplied model and prints a human-readable report.

```bash
lake exe nfp analyze models/gpt2_rigorous.nfpt \
  --threshold 0.1 --verify --verbose --output report.txt
```

- `--threshold` (`-t`) sets the minimum effect threshold used for verification (default: `0.1`).
- `--verify` optionally runs causal verification using model-provided inputs.
- `--verbose` prints model metadata and per-stage status messages.
- `--output` (`-o`) writes the report to a file instead of stdout.

### `induction`

Searches for **candidate induction circuits** and ranks head pairs by a mechanical score.

```bash
lake exe nfp induction models/gpt2_rigorous.nfpt \
  --threshold 0.0 --diagnostics --diagTop 5 --adaptive --verbose
```

- `--threshold` (`-t`) sets the minimum normalized effect (default: `0.0`).
- `--correct` / `--incorrect` manually pick logit IDs for the induction target (otherwise the target is inferred from tokens).
- `--verify` runs causal verification via head ablation on the top-10 candidates.
- `--diagnostics` enables bound breakdowns; `--diagTop` controls how many candidates receive diagnostics (default: `5`).
- `--adaptive` turns on the adaptive bound scheduler. Tuning flags include `--targetSlack` (default: `8.0`),
  `--maxUpgrades` (default: `120`), `--minRelImprove` (default: `0.01`), `--krylovSteps` (default: `2`),
  and `--adaptiveScope` (`layernorm | all`, default: `layernorm`).
- `--verbose` prints detailed scoring metrics for each candidate.

### `certify`

Computes a conservative **certificate report** in sound mode using exact `Rat` arithmetic (no trusted floats).

Note: global sound certification supports `NFP_BINARY_V1`. Local sound certification
supports `NFP_BINARY_V1` (fixed-point union-box) and legacy `NFP_TEXT_V1/V2`.

`certify` supports both:
- **global certification** (weights only), and
- **local certification** (weights + a small input region around a concrete prompt/input).

```bash
lake exe nfp certify models/gpt2_rigorous.nfpt \
  --output cert.txt
```

- For local (input-dependent) LayerNorm certification, pass an ℓ∞ radius `δ`:

```bash
lake exe nfp certify models/gpt2_rigorous.nfpt \
  --delta 0.01
```

If you want to override the embedded input, pass a separate input `.nfpt`:

- LayerNorm ε is read from the model header (`layer_norm_eps`).
- `gelu_kind` in the model header selects the GeLU derivative target (`tanh` or `exact`).
- `--delta` sets the local ℓ∞ radius `δ` (default: `0`). Providing `--delta` enables local certification.
- `--input` optionally provides an input `.nfpt` file used for local certification; if omitted and the
  model file embeds `EMBEDDINGS`, `certify` reuses the model file as its input source.
- `--softmaxMargin` provides a logit-margin lower bound, but it is currently **rejected** by the
  verifier (use `--bestMatchMargins` instead).
- `--softmaxExpEffort` controls exp lower-bound effort used for margin-based softmax tightening (default: `1`).
- `--bestMatchMargins` runs a full best-match sweep (binary + local only) and tightens layer
  softmax bounds using verified margin evidence. It is incompatible with `--softmaxMargin`.
- `--targetOffset` selects the target-token offset for best-match margins (default: `-1`).
- `--maxSeqLen` caps the sequence length used in best-match margin sweeps (default: `0` = full `seq_len`).
- `--tightPattern`, `--tightPatternLayers`, and `--perRowPatternLayers` control pattern tightening
  during best-match sweeps.
- `--scalePow10` sets fixed-point scaling for best-match sweeps (default: `9`).
- `--noncausalPattern` disables the causal-prefix restriction (required for non-causal models).
- `--soundnessBits` sets dyadic sqrt precision for LayerNorm bounds (default: `20`).
- `--partitionDepth` requests input partitioning depth (default: `0`; scaffold only, must remain `0` for now).
- `--output` (`-o`) writes the report to a file (otherwise it prints to stdout).

### `head_bounds`

Computes sound per-head contribution bounds (global weight-only, or local with `--delta`).

```bash
lake exe nfp head_bounds models/gpt2_rigorous.nfpt
```

For local bounds (uses input embeddings in the model file when present):

```bash
lake exe nfp head_bounds models/gpt2_rigorous.nfpt --delta 0.01
```

- `--delta` enables local head bounds; `--input` can override the embedded input.
- LayerNorm ε is read from the model header (`layer_norm_eps`).
- `--soundnessBits` sets dyadic sqrt precision for LayerNorm bounds (default: `20`).
- `--scalePow10` controls fixed-point scaling for global bounds (default: `9`).
- `--output` (`-o`) writes the report to a file (otherwise it prints to stdout).

### `head_pattern`

Computes a sound local attention pattern bound for a single head (binary only),
propagating per-position intervals up to the target layer (bounded by `maxSeqLen`).
The pattern compares logits for keys whose **shifted-key token** matches the
query’s **offset token** (e.g., `--offset -1` matches the previous token, and
`--offset 0 --keyOffset -1` matches the copy-next pattern).

```bash
lake exe nfp head_pattern models/gpt2_rigorous.nfpt --layer 0 --head 0 --delta 0.01 --offset -1
```

- `--offset` selects the target key position relative to the query (default: `-1` for previous token).
- `--keyOffset` selects which key-position token is matched (default: `0` for the key token itself).
- `--maxSeqLen` caps the sequence length analyzed for pattern bounds (default: `256`).
- `--input` optionally provides an input `.nfpt` file; required for legacy text models.
- `--delta` sets the local input radius; LayerNorm ε is read from the model header (`layer_norm_eps`).
- `--soundnessBits` sets dyadic sqrt precision for LayerNorm bounds (default: `20`).
- `--tightPattern` enables a slower but tighter pattern bound near the target layer.
- `--tightPatternLayers` sets how many layers use tight bounds (default: `1`; implies `--tightPattern`).
- `--perRowPatternLayers` sets how many layers use per-row MLP propagation (default: `0`).
- `--softmaxExpEffort` sets the exp lower-bound effort for margin-derived softmax bounds (default: `1`).
- `--scalePow10` sets fixed-point scaling for best-match bounds (default: `9`).
- `--noncausalPattern` disables the causal-prefix restriction (required for non-causal models).
- `--bestMatch` switches to a single-query best-match bound (default query: last position).
- `--affine` uses affine Q/K dot bounds in best-match mode.
- `--sweep` prints best-match bounds for all valid query positions (requires `--bestMatch`).
- `--queryPos` chooses the query position for best-match bounds (default: last position).
- `--output` (`-o`) writes the report to a file (otherwise it prints to stdout).

### `induction_cert`

Computes a minimal sound induction-head certificate by combining two pattern
certificates and a value-coordinate lower bound (binary only).

```bash
lake exe nfp induction_cert models/gpt2_rigorous.nfpt \
  --layer1 0 --head1 0 --layer2 1 --head2 0 --coord 0 --delta 0.01 \
  --target 42 --negative 17
```

- `--layer1/--head1` selects the previous-token head; `--layer2/--head2` selects the
  token-match head.
- `--coord` chooses the output coordinate used for the value lower bound.
- `--offset1/--offset2` adjust the token-match offsets (default: `-1`).
- `--keyOffset1/--keyOffset2` adjust the key-token offsets (default: `0`;
  use `--offset2 0 --keyOffset2 -1` for copy-next induction).
- `--target/--negative` optionally add a logit-diff lower bound using unembedding columns.
- `--input` optionally provides an input `.nfpt` file; required for legacy text models.
- `--delta` sets the local input radius (default: `0`).
- `--soundnessBits` sets dyadic sqrt precision for LayerNorm bounds (default: `20`).
- `--tightPattern` enables a slower but tighter pattern bound near the target layer.
- `--tightPatternLayers` sets how many layers use tight bounds (default: `1`; implies `--tightPattern`).
- `--perRowPatternLayers` sets how many layers use per-row MLP propagation (default: `0`).
- `--softmaxExpEffort` sets the exp lower-bound effort for margin-derived softmax bounds (default: `1`).
- `--maxSeqLen` caps the sequence length analyzed for best-match bounds (default: `256`).
- `--scalePow10` sets fixed-point scaling for best-match bounds (default: `9`).
- `--noncausalPattern` disables the causal-prefix restriction (required for non-causal models).
- `--bestMatch` switches to single-query best-match bounds (default query: last position).
- `--affine` uses affine Q/K dot bounds in best-match mode.
- `--queryPos` chooses the query position for best-match bounds (default: last position).
- `--iterTighten` iteratively tightens best-match bounds (tight/per-row layers and scale precision).
- `--output` (`-o`) writes the report to a file (otherwise it prints to stdout).

### `rope`

Generates RoPE-related linearization bounds used by the certificate/checking pipeline.

```bash
lake exe nfp rope --seqLen 4 --pairs 8
```

- `--seqLen` instantiates the bound at the given sequence length (default: `4`).
- `--pairs` sets the number of RoPE pairs; the dimension is `2 * pairs` (default: `8`).

### `bench`

Runs repeatable microbenchmarks for analysis or induction search.

```bash
lake exe nfp bench models/gpt2_rigorous.nfpt --mode analysis --runs 5 --repeats 1
```

- `--mode` selects `analysis` or `induction` (default: `analysis`).
- `--runs` sets the number of timed runs (default: `5`).
- `--repeats` repeats the inner workload per run (default: `1`).
- `--threshold` sets the analyze threshold (default: `0.1`).
- `--minEffect` sets the induction minEffect (default: `0.0`).
- `--correct/--incorrect` override induction target tokens.
- `--verbose` prints per-run timing details.
- `--breakdown` emits per-phase averages (analysis only).

### `sound_cache_check`

Checks SOUND fixed-point cache soundness (CI / small fixtures).

```bash
lake exe nfp sound_cache_check tests/fixtures/tiny_sound_binary.nfpt
```

- `--scalePow10` sets the fixed-point scale exponent (default: `9`).
- `--maxTokens` checks at most this many numeric tokens (default: `0` = all).

### `sound_cache_bench`

Benchmarks SOUND fixed-point cache build (text or binary).

```bash
lake exe nfp sound_cache_bench models/gpt2_rigorous.nfpt --runs 3
```

- `--scalePow10` sets the fixed-point scale exponent (default: `9`).
- `--runs` sets the number of benchmark runs (default: `1`).

### `dump`

Dumps a small forward-pass slice for PyTorch sanity checking.

```bash
lake exe nfp dump models/gpt2_rigorous.nfpt --layer 0 --pos 0 --kind afterLayer
```

- `--layer` selects the layer index (default: `0`).
- `--pos` selects the token position / row index (default: `0`).
- `--take` limits columns from the start (default: `16`).
- `--kind` chooses `embeddings | layerInput | postAttn | afterLayer` (default: `afterLayer`).

### `logit_diff`

Computes an empirical logit-difference for a target vs. negative token.

```bash
lake exe nfp logit_diff models/gpt2_rigorous.nfpt 42 17 --autoNegative
```

- `--pos` selects the token position (default: last position).
- `--input` provides an input `.nfpt` with TOKENS + EMBEDDINGS.
- `--autoNegative` uses the top non-target logit as the negative token.

### `--version`

Prints the CLI version string.

## What “rigorous” means here

At a high level, the “rigorous” path avoids heuristic operator-norm estimation and instead uses **upper bounds** derived from standard inequalities (examples you may see in logs):

- Frobenius-norm based bounds.
- Gram-matrix based bounds.
- Schur / Brauer-style eigenvalue bounds for symmetric matrices.
- Row-wise softmax operator bounds using quantities like `rowMaxP`, `rowTrace`, Gershgorin-style estimates, and a “moment” bound.

The CLI may still compute **power-iteration estimates** for comparison, but those are explicitly labelled as diagnostics and are not used to produce the rigorous `ub=…` values.

## Reproducing the example command

A typical workflow:

```bash
# 1) Build
lake update
lake build

# 2) Export a model (optional)
python scripts/export_gpt2.py models/gpt2_rigorous.nfpt

# 3) Run induction search with diagnostics
lake exe nfp induction models/gpt2_rigorous.nfpt -v -d | sed -n '1,220p'
```

## Project layout

- `Main.lean` — CLI wiring and command definitions.
- `Nfp/` — library code (probability, transformer semantics, soundness/cert machinery, discovery routines).
- `scripts/` — Python helpers to export models and generate induction datasets.
- `models/` — local model files (not versioned here if large; author’s setup may have used local symlinks).

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See the LICENSE file.
