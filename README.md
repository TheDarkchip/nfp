# NFP

NFP is a Lean 4 project for **mathematically rigorous** reasoning about transformer-style computations, with a focus on mechanistic interpretability (e.g. induction heads) and provable norm/error bounds.

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

The CLI expects a model file in **`.nfpt`** format.

- Create a local `models/` directory and place your `.nfpt` files there (the repo does not version model files; the author’s setup may have used local symlinks).
- You can export GPT-2 weights from Hugging Face using the scripts in `scripts/`.

### Exporting GPT-2 to `.nfpt`

The export scripts use `torch` + `transformers`.

Example (write `models/gpt2_rigorous.nfpt`):

```bash
python scripts/export_gpt2.py models/gpt2_rigorous.nfpt
```

If you prefer a locked Python environment, use `uv` or a venv and install dependencies from `pyproject.toml`.

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

`certify` supports both:
- **global certification** (weights only), and
- **local certification** (weights + a small input region around a concrete prompt/input).

```bash
lake exe nfp certify models/gpt2_rigorous.nfpt \
  --eps 1e-5 --actDeriv 2 --output cert.txt
```

- For local (input-dependent) LayerNorm certification, pass an input `.nfpt` containing `EMBEDDINGS` and an ℓ∞ radius `δ`:

```bash
lake exe nfp certify models/gpt2_rigorous.nfpt \
  --input models/gpt2_rigorous.nfpt --delta 1/100 --eps 1e-5 --actDeriv 2
```

- `--eps` sets the LayerNorm ε (default: `1e-5`).
- `--actDeriv` bounds the activation derivative (default: `2`).
- `--input` optionally provides an input `.nfpt` file used for local certification.
- `--delta` sets the local ℓ∞ radius `δ` (default: `0`).
- `--output` (`-o`) writes the report to a file (otherwise it prints to stdout).

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

## Contributing

- Keep the **bound-producing** code path free of heuristic numeric methods.
- Prefer changes that either:
  - tighten a bound via a proven inequality, or
  - add a *checked* certificate that justifies a tighter constant.

If you add a heuristic for exploration, keep it clearly labelled and gated behind diagnostics/debug flags.

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See the LICENSE file.
