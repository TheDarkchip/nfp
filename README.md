# NFP

NFP is a Lean 4 project for **mathematically rigorous** reasoning about transformer-style
computations, with a focus on mechanistic interpretability (e.g. induction heads) and provable
norm/error bounds.

NFP stands for **Neural Formal Pathways**.

## Status
The core has recently been rewritten and the API is still settling. Expect breaking changes.

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
The input setup follows the standard literature diagnostic: repeated token patterns (pattern
repeated twice) and attention stripes that look back by one period.

For a step-by-step walkthrough, see `docs/demo.md`.
For a careful statement of what certificates do and do not claim, see
`docs/cert_usefulness.md`.

### Build a head certificate (untrusted)

```bash
python scripts/build_gpt2_induction_cert.py \
  --output reports/gpt2_induction.cert \
  --layer 0 --head 5 --seq 32 --pattern-length 16 \
  --random-pattern --seed 0 \
  --active-eps-max 1/2
```

Layer/head indices in the generator are 0-based to match the literature.

To certify a **non-vacuous** logit-diff lower bound, supply a direction:

```bash
python scripts/build_gpt2_induction_cert.py \
  --output reports/gpt2_induction.cert \
  --layer 0 --head 5 --seq 32 --pattern-length 16 \
  --random-pattern --seed 0 \
  --active-eps-max 1/2 \
  --direction-target 1268 --direction-negative 1796
```

Or let the untrusted script search for a direction in a vocab slice:

```bash
python scripts/build_gpt2_induction_cert.py \
  --output reports/gpt2_induction.cert \
  --layer 0 --head 5 --seq 32 --pattern-length 16 \
  --random-pattern --seed 0 \
  --active-eps-max 1/2 \
  --search-direction --direction-vocab-min 1000 --direction-vocab-max 2000 \
  --direction-min-lb 1/10 \
  --direction-report-out reports/direction_report.txt --direction-topk 10 \
  --tokens-out reports/gpt2_induction.tokens
```

Direction search is **untrusted witness generation**; the Lean CLI only verifies the resulting
explicit certificate. The direction report lists the top-ranked candidates by estimated lower
bound so you can pick a stable non-vacuous direction.

Optional direction metadata:

```
--direction-target <token_id> --direction-negative <token_id>
```

### Verify a head certificate (trusted checker)

```bash
lake exe nfp induction verify --cert reports/gpt2_induction.cert
```

Optional gates:

```
--min-active <n>   --min-margin <rat>   --max-eps <rat>   --min-logit-diff <rat>   --tokens <path>
```

If `--tokens` is provided, the CLI verifies that the certificate's `prev` and `active`
match the token-sequence semantics for repeated tokens (previous occurrence).

Example non-vacuous check:

```bash
lake exe nfp induction verify --cert reports/gpt2_induction.cert --min-logit-diff 1/10
```

### Verify a batch (trusted checker)

```bash
lake exe nfp induction verify --batch reports/gpt2_induction.batch
```

Batch file format (all fields optional unless noted):

```
min-active <n>
min-logit-diff <rat>
min-margin <rat>
max-eps <rat>
min-avg-logit-diff <rat>
item <cert_path> [tokens_path]
```

`item` entries are required. `min-avg-logit-diff` enforces an average
logit-diff lower bound across all items (requires direction metadata in each cert).

### Verify a stripe certificate (trusted checker)

```bash
lake exe nfp induction verify --stripe-cert reports/gpt2_stripe.cert
```

Optional gates:

```
--min-stripe-mean <rat>   --min-stripe-top1 <rat>
```

### Verify a stripe batch (trusted checker)

```bash
lake exe nfp induction verify --stripe-batch reports/gpt2_stripe.batch
```

Stripe batch file format:

```
min-stripe-mean <rat>
min-stripe-top1 <rat>
min-avg-stripe-mean <rat>
min-avg-stripe-top1 <rat>
item <cert_path>
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

All sequence indices (`q`, `k`) are **0-based** (file-format convention). Layer/head indices
elsewhere in the tooling are **0-based** to match the literature. Direction token IDs
(`direction-target`, `direction-negative`) are raw model IDs (tokenizer convention).
`direction-*` lines are optional metadata; if present, both must appear. If no `active` lines
appear, the checker defaults to all non-initial queries (indices 1.. in 0-based indexing).

### Stripe-attention certificate

```
seq <n>
period <n>
stripe-mean-lb <rat>
stripe-top1-lb <rat>
weight <q> <k> <rat>
```

All sequence indices (`q`, `k`) are **0-based**. The checker recomputes stripe-mean and
stripe-top1 from the provided weights and enforces the certificate’s lower bounds.

### Direction report (untrusted)

```
direction_report
vocab_min=<n> vocab_max=<n> seed=<n>
rank\tlb\ttarget\tnegative
```

This file is an **untrusted helper artifact**; it only ranks candidate directions and does not
change what the Lean checker accepts.

### Token list (untrusted)

```
seq <n>
token <q> <tok_id>
```

This file is an **untrusted helper artifact** used to check that `prev` and `active` match the
token sequence (previous-occurrence semantics) when `--tokens` is supplied to the CLI. Sequence
indices are 0-based.

## Soundness boundary

- Untrusted scripts may use floating-point numerics to generate candidate certificates.
- The CLI **only verifies** explicit certificates; it does not search for witnesses or run models.

For known gaps, see `SOUNDNESS_LIMITATIONS.md`.

## Requirements

- **Lean 4** (pinned in `lean-toolchain`) and **Lake**.
- Optional: **Python** for helper scripts (`scripts/`), plus `torch`, `transformers`, and `numpy`.

## References

- Elhage et al., “A Mathematical Framework for Transformer Circuits.”
  Link: `https://transformer-circuits.pub/2021/framework/index.html`
- Olsson et al., “In-context Learning and Induction Heads.”
  Link: `https://arxiv.org/abs/2209.11895`

## Contributing

Please follow the project rules in `AGENTS.md` (no `sorry`, no linter disables, total soundness in
trusted namespaces).
