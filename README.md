# NFP

NFP is a Lean 4 project for **mathematically rigorous** reasoning about transformer-style
computations, with a focus on mechanistic interpretability (e.g. induction heads) and provable
norm/error bounds.

NFP stands for **Neural Formal Pathways**.

## Status

This repository is in a **tabula rasa rewrite**. The new core is intentionally minimal and the API
surface is still settling. Expect breaking changes.

## Build

```bash
lake build -q --wfail
lake build nfp -q --wfail
```

## CLI

```bash
lake exe nfp --help
lake exe nfp induction --help
```

Current subcommands are limited to **induction certificate checking**. The CLI does **not** run a
full model forward pass and does **not** ingest `.nfpt` weights directly; weight ingestion is done
by untrusted helper scripts (see below).

## Module map

The authoritative module map and invariants are tracked in `AGENTS.md`.

High-level layout:
- `Nfp/Core`, `Nfp/Prob`, `Nfp/Mixer`, `Nfp/System`: core math infrastructure.
- `Nfp/Circuit`: circuits, typed interfaces, and layer wiring (attention, induction).
- `Nfp/Sound`: sound builders and verified helpers.
- `Nfp/IO`, `Nfp/Cli`: parsing and CLI entrypoints.

## Induction Certification (prototype)

The current prototype checks **head-level induction certificates** and can optionally compose them
with a **downstream error bound**. Certificates are produced by **untrusted** helper scripts and
verified by the CLI.

### Build a head certificate (untrusted)

```bash
python scripts/build_gpt2_induction_cert.py \
  --output reports/gpt2_induction.cert \
  --layer 5 --head 1 --seq 32 --pattern-length 16 \
  --values-out reports/gpt2_induction.values --value-dim 0 \
  --active-eps-max 1/2
```

If you want values aligned to a logit-diff direction, add:

```
--direction-target <token_id> --direction-negative <token_id>
```

### Verify a head certificate (trusted checker)

```bash
lake exe nfp induction certify \
  --scores reports/gpt2_induction.cert \
  --values reports/gpt2_induction.values
```

Non-vacuity gates (optional):

```
--min-margin <rat>   --max-eps <rat>   --min-active <n>   --min-logit-diff <rat>
```

### Recompute bounds inside Lean (sound builder)

```bash
lake exe nfp induction certify_sound \
  --scores reports/gpt2_induction.cert \
  --values reports/gpt2_induction.values
```

This ignores any `eps`/`margin`/`lo`/`hi` lines and recomputes them from the raw entries.

### Compute exact head inputs inside Lean (experimental)

```bash
lake exe nfp induction certify_head --inputs reports/gpt2_induction.head
```

This path recomputes scores/values in Lean from exact head inputs. It is **experimental** and can
be slow for nontrivial sequence lengths.

You can also derive the head inputs directly from an `NFP_BINARY_V1` model file:

```bash
lake exe nfp induction certify_head_model \
  --model models/gpt2_rigorous_with_gelu_kind_seq32.nfpt \
  --layer 5 --head 1 --period 16 \
  --direction-target 1 --direction-negative 2
```

### End-to-end check with downstream bound (prototype)

```bash
python scripts/build_downstream_linear_cert.py \
  --output reports/gpt2_downstream.cert \
  --gain 3/2 --input-bound 5/4

lake exe nfp induction certify_end_to_end \
  --scores reports/gpt2_induction.cert \
  --values reports/gpt2_induction.values \
  --downstream reports/gpt2_downstream.cert
```

The downstream certificate is **checked for internal arithmetic consistency** but is externally
computed. You can also compute the downstream bound inside Lean from a matrix payload:

```bash
lake exe nfp induction certify_end_to_end_matrix \
  --scores reports/gpt2_induction.cert \
  --values reports/gpt2_induction.values \
  --matrix reports/gpt2_downstream.matrix
```

Or derive the downstream matrix directly from an `NFP_BINARY_V1` model file
(currently uses the unembedding direction only):

```bash
lake exe nfp induction certify_end_to_end_model \
  --scores reports/gpt2_induction.cert \
  --values reports/gpt2_induction.values \
  --model models/gpt2_rigorous.nfpt \
  --residual-interval reports/gpt2_residual.interval
```

## File formats

### Softmax-margin certificate

```
seq <n>
eps <rat>
margin <rat>
active <q>
prev <q> <k>
score <q> <k> <rat>
weight <q> <k> <rat>
```

`active <q>` lines declare the queries on which bounds are required; if omitted, the checker
defaults to all nonzero queries.

### Value-range certificate

```
seq <n>
direction-target <tok_id>
direction-negative <tok_id>
lo <rat>
hi <rat>
val <k> <rat>
```

`direction-*` lines are optional metadata for directional (logit-diff) values.

### Downstream linear certificate

```
error <rat>
gain <rat>
input-bound <rat>
```

The checker enforces `error = gain * input-bound` and nonnegativity of all fields.

### Downstream matrix payload

```
rows <n>
cols <n>
input-bound <rat>
w <i> <j> <rat>
```

The checker computes a row-sum norm bound from the matrix entries.

### Residual-interval certificate

```
dim <n>
lo <i> <rat>
hi <i> <rat>
```

Each `lo`/`hi` entry supplies an interval bound for residual vector coordinate `i`,
used to compute downstream error.

### Head input format (for `certify_head`)

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

All `direction`, `embed`, and projection matrices must be fully specified. If no `active` lines
appear, the checker defaults to all nonzero queries.

## Soundness boundary

- Untrusted scripts may use floating-point numerics to generate candidate certificates.
- The CLI **only verifies** certificate constraints inside Lean; it does not search for witnesses.
- Downstream error certificates are currently **not derived in Lean** (work in progress).

For known gaps, see `SOUNDNESS_LIMITATIONS.md`.

## Requirements

- **Lean 4** (pinned in `lean-toolchain`) and **Lake**.
- Optional: **Python** for helper scripts (`scripts/`).

## Contributing

Please follow the project rules in `AGENTS.md` (no `sorry`, no linter disables, total soundness in
trusted namespaces).
