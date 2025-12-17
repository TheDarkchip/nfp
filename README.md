# NFP

NFP is a Lean 4 project for **mathematically rigorous** reasoning about transformer-style computations, with a focus on mechanistic interpretability (e.g. induction heads) and provable norm/error bounds.

This repo contains:

- A **Lean library** (under `Nfp/`) for finite probability and a lightweight “transformer semantics” layer.
- A **CLI executable** (`lake exe nfp …`) that loads transformer weights stored in a compact binary format (`.nfpt`) and produces rigorous bounds and diagnostics.

> Goal: *no “hand-wavy” numerics in the bound path.* Heuristic estimates (e.g. power iteration) may exist for diagnostics, but the bounds reported as “rigorous” are computed via conservative inequalities.

## Status

This is research tooling. Interfaces may change; please treat results as experimental unless they are backed by a certificate/check you trust.

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

Run the CLI:

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

### `induction`

Searches for **candidate induction circuits** and prints ranked head pairs along with rigorous error terms.

```bash
lake exe nfp induction models/gpt2_rigorous.nfpt -v -d
```

Useful flags:

- `-v` / `--verbose`: more internal metrics.
- `-d` / `--diagnostics`: print extra bound decompositions and comparisons.

The output contains:

- A ranked list of head pairs (e.g. `L2H2 -> L5H5`).
- A “rigorous” **Error** composed from per-head ε terms.
- When diagnostics are enabled, a section like:
  - `LAYER NORM DIAGNOSTICS (PI vs rigorous)`
    - “PI” (power iteration) is *diagnostics only*.
    - “rigorous ub” is the bound actually used.

### `analyze`

Analyzes a specific head pair and (optionally) prints a detailed decomposition of each bound.

```bash
lake exe nfp analyze models/gpt2_rigorous.nfpt \
  --l1 2 --h1 2 --l2 5 --h2 5 -v -d
```

### `rope`

Computes (and prints) RoPE-related bounds used by the certificate/checking pipeline.

```bash
lake exe nfp rope models/gpt2_rigorous.nfpt
```

### `certify`

Generates a conservative **certificate report** for a model (RoPE error, activation derivative bounds, etc.).

```bash
lake exe nfp certify models/gpt2_rigorous.nfpt \
  --eps 1e-5 --actDeriv 2 --output cert.txt
```

- `--output` writes the report to a file (otherwise it prints to stdout).
- This is meant as a bridge between floating-point computation and a checkable “sound mode” story.

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
