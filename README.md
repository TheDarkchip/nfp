# NFP

NFP is a Lean 4 project for **mathematically rigorous** reasoning about transformer-style computations, with a focus on mechanistic interpretability (e.g. induction heads) and provable norm/error bounds.

NFP stands for **Neural Formal Pathways**.

This repo contains:

- A **Lean library** (under `Nfp/`) for finite probability and a lightweight “transformer semantics” layer.
- A **CLI executable** (`lake exe nfp …`) that loads transformer weights stored in a compact binary format (`.nfpt`) and produces rigorous bounds and diagnostics.

> Goal: *no “hand-wavy” numerics in the bound path.* Heuristic estimates (e.g. power iteration) may exist for diagnostics, but the bounds reported as “rigorous” are computed via conservative inequalities.

## Status

This is research tooling. Interfaces may change; please treat results as experimental unless they are backed by a certificate/check you trust.

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


## Requirements

- **Lean 4** (pinned by `lean-toolchain`) and **Lake**.
  - Easiest install: `elan` (Lean toolchain manager).
- A standard build toolchain for Lean (C/C++ compiler, `make`, etc.).
- (Optional) **Python** for the export scripts in `scripts/`.

Lean version is pinned in `lean-toolchain` (currently `leanprover/lean4:v4.25.2`).

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
BINARY_START
```

The payload is raw little-endian bytes in a fixed order (tokens, embeddings, then weights).

Note: global sound certification supports `NFP_BINARY_V1`. Local sound certification
supports `NFP_BINARY_V1` (fixed-point union-box) and legacy `NFP_TEXT_V1/V2`.

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
  `--maxUpgrades` (default: `200`), `--minRelImprove` (default: `0.01`), `--krylovSteps` (default: `4`),
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
  --actDeriv 2 --output cert.txt
```

- For local (input-dependent) LayerNorm certification, pass an ℓ∞ radius `δ`:

```bash
lake exe nfp certify models/gpt2_rigorous.nfpt \
  --delta 0.01 --actDeriv 2
```

If you want to override the embedded input, pass a separate input `.nfpt`:

- LayerNorm ε is read from the model header (`layer_norm_eps`).
- `--actDeriv` bounds the activation derivative (default: `2`).
- `--delta` sets the local ℓ∞ radius `δ` (default: `0`). Providing `--delta` enables local certification.
- `--input` optionally provides an input `.nfpt` file used for local certification.
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
- `--scalePow10` controls fixed-point scaling for global bounds (default: `9`).
- `--output` (`-o`) writes the report to a file (otherwise it prints to stdout).

### `head_pattern`

Computes a sound local attention pattern bound for a single head (binary only),
propagating per-position intervals up to the target layer (bounded by `maxSeqLen`).
The pattern compares logits for keys whose token matches the query’s offset token
(e.g., `--offset -1` matches the previous token).

```bash
lake exe nfp head_pattern models/gpt2_rigorous.nfpt --layer 0 --head 0 --delta 0.01 --offset -1
```

- `--offset` selects the target key position relative to the query (default: `-1` for previous token).
- `--maxSeqLen` caps the sequence length analyzed for pattern bounds (default: `256`).
- `--delta` sets the local input radius; LayerNorm ε is read from the model header (`layer_norm_eps`).
- `--tightPattern` enables a slower but tighter pattern bound near the target layer.
- `--tightPatternLayers` sets how many layers use tight bounds (default: `1`; implies `--tightPattern`).
- `--perRowPatternLayers` sets how many layers use per-row MLP propagation (default: `0`).
- `--bestMatch` switches to a single-query best-match bound (default query: last position).
- `--sweep` prints best-match bounds for all valid query positions (requires `--bestMatch`).
- `--queryPos` chooses the query position for best-match bounds (default: last position).

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
- `--target/--negative` optionally add a logit-diff lower bound using unembedding columns.
- `--tightPattern` enables a slower but tighter pattern bound near the target layer.
- `--tightPatternLayers` sets how many layers use tight bounds (default: `1`; implies `--tightPattern`).
- `--perRowPatternLayers` sets how many layers use per-row MLP propagation (default: `0`).
- `--bestMatch` switches to single-query best-match bounds (default query: last position).
- `--queryPos` chooses the query position for best-match bounds (default: last position).

### `rope`

Generates RoPE-related linearization bounds used by the certificate/checking pipeline.

```bash
lake exe nfp rope --seqLen 4 --pairs 8
```

- `--seqLen` instantiates the bound at the given sequence length (default: `4`).
- `--pairs` sets the number of RoPE pairs; the dimension is `2 * pairs` (default: `8`).

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
