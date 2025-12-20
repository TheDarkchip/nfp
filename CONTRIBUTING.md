# Contributing to NFP

Thank you for your interest in NFP. This project is a Lean 4 library for **mathematically rigorous** circuit discovery.

## Contribution Policy

To maintain strict scientific soundness and clear copyright for any future research grant, **I generally do not accept external Pull Requests for core logic.**

Instead of submitting code, please **open an Issue** to report bugs or suggest features:
* If you have found a bug, please describe how to reproduce it.
* If you have a mathematical improvement or new algorithm, please describe the proof strategy in the issue.

I prefer to implement core logic changes myself to ensure they meet the project's strict verification standards and licensing requirements.

**Acknowledgments & Credit**
I value every contribution. If you report a bug or suggest a feature that ends up in the library, I will add you to the **Acknowledgments** section of the `README.md` to ensure you get proper credit for your ideas.

## 1. Verification Standards

Even though direct code contributions are restricted, we document our standards here so users understand the library's design constraints.

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
The library maintains the following mathematical properties:
* **Row-Stochasticity:** All `Mixer` structures must maintain the row-sum-one invariant.
* **Non-Negativity:** Mass and capacity values must be non-negative (prefer `NNReal`).
* **Finiteness:** Types used in probability vectors must be finite (`[Fintype]`).

## 3. Local Development
If you are experimenting with the code locally or verifying a bug report, you can check that the build passes with:

```bash
lake build -q --wfail      # Must pass with zero warnings
lake build nfp -q --wfail  # CLI must compile cleanly

```

**Note:** If you are using LLM-based coding assistants for your local experiments, please configure them to read `AGENTS.md`, which contains specific system prompts aligned with these guidelines.

## 4. Design Philosophy

We prioritize **correctness over convenience**.

* If a proof is difficult, factor it into helper lemmas rather than skipping steps.
* Avoid "hand-wavy" numerics in the bound-producing code paths.
