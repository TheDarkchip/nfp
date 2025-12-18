# Contributing to NFP

Thank you for your interest in NFP. This project is a Lean 4 library for **mathematically rigorous** circuit discovery. To maintain scientific soundness, we enforce a strict set of verification standards.

**Note:** If you are using LLM-based coding assistants, please configure them to read `AGENTS.md`, which contains specific system prompts aligned with these guidelines.

## 1. Verification Standards

### 1.1 No "Fake Proofs"
* **Incomplete Proofs:** The use of `sorry` or `admit` is strictly forbidden in the library kernel. All theorems must be fully proven.
* **Axioms:** Do not introduce new axioms. The library relies on `mathlib` and the standard Lean axioms.

### 1.2 Strict Linting
* **Warnings are Errors:** The build pipeline treats all warnings as errors.
* **Linter Configuration:** Do not disable linters (e.g., `set_option linter.* false`) to silence warnings. Address the underlying issue instead.

### 1.3 Exact Arithmetic
* **Sound Mode:** Certification logic must use exact `Rat` (rational) arithmetic.
* **Floating Point:** `Float` operations are permitted only for heuristic discovery/diagnostics, not for final verification bounds.

## 2. Core Invariants
Contributions must preserve the following mathematical properties:
* **Row-Stochasticity:** All `Mixer` structures must maintain the row-sum-one invariant.
* **Non-Negativity:** Mass and capacity values must be non-negative (prefer `NNReal`).
* **Finiteness:** Types used in probability vectors must be finite (`[Fintype]`).

## 3. Workflow
Before submitting a change, ensure the following commands pass:

```bash
lake build -q --wfail      # Must pass with zero warnings
lake build nfp -q --wfail  # CLI must compile cleanly
```

## 4. Design Philosophy

We prioritize **correctness over convenience**.

* If a proof is difficult, factor it into helper lemmas rather than skipping steps.
* Avoid "hand-wavy" numerics in the bound-producing code paths.