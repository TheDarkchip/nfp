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
full model forward pass; certificate generation is done by untrusted helper scripts (see below).

## Module map

The authoritative module map and invariants are tracked in `AGENTS.md`.

High-level layout:
- `Nfp/Core`, `Nfp/Prob`, `Nfp/Mixer`, `Nfp/System`: core math infrastructure.
- `Nfp/Circuit`: circuits, typed interfaces, and layer wiring (attention, induction).
- `Nfp/Sound`: soundness theorems and verified helpers.
- `Nfp/IO`, `Nfp/Cli`: parsing and CLI entrypoints.

## Induction Certification (prototype)

The current prototype checks **explicit induction-head certificates**. Certificates are produced
by **untrusted** Python scripts and verified by the Lean CLI; no model forward pass runs in Lean.

### Build a head certificate (untrusted)

```bash
python scripts/build_gpt2_induction_cert.py \
  --output reports/gpt2_induction.cert \
  --layer 1 --head 6 --seq 32 --pattern-length 16 \
  --random-pattern --seed 0 \
  --active-eps-max 1/2
```

Layer/head indices in the generator are 1-based to match the literature.

Optional direction metadata:

```
--direction-target <token_id> --direction-negative <token_id>
```

### Verify a head certificate (trusted checker)

```bash
lake exe nfp induction certify --cert reports/gpt2_induction.cert
```

Optional gates:

```
--min-active <n>   --min-margin <rat>   --max-eps <rat>   --min-logit-diff <rat>
```

## File formats

### Induction-head certificate

```
seq <n>
direction-target <tok_id>
direction-negative <tok_id>
eps <rat>
margin <rat>
active <q>
prev <q> <k>
score <q> <k> <rat>
weight <q> <k> <rat>
eps-at <q> <rat>
weight-bound <q> <k> <rat>
lo <rat>
hi <rat>
val <k> <rat>
val-lo <k> <rat>
val-hi <k> <rat>
```

All sequence indices (`q`, `k`) are **1-based** (literature convention). Direction token IDs
(`direction-target`, `direction-negative`) are raw model IDs (tokenizer convention).
`direction-*` lines are optional metadata; if present, both must appear. If no `active` lines
appear, the checker defaults to all non-initial queries (indices 2.. in 1-based indexing).

## Soundness boundary

- Untrusted scripts may use floating-point numerics to generate candidate certificates.
- The CLI **only verifies** explicit certificates; it does not search for witnesses or run models.

For known gaps, see `SOUNDNESS_LIMITATIONS.md`.

## Requirements

- **Lean 4** (pinned in `lean-toolchain`) and **Lake**.
- Optional: **Python** for helper scripts (`scripts/`).

## Contributing

Please follow the project rules in `AGENTS.md` (no `sorry`, no linter disables, total soundness in
trusted namespaces).
