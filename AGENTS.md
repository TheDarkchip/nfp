# AGENTS.md

This repo is a Lean 4 **math library + CLI tool** formalizing an `Nfp`-style system:
finite probability on finite types, row-stochastic “mixers”, influence specs, and
uniqueness / attribution results (including neural-network–interpretation layers),
with an executable `nfp` CLI for analysis and verification.

You are an automated agent extending a *library* (with a small CLI shell), not an app.

**Lean version note:** The Lean version is pinned by `lean-toolchain`.

The design may evolve. If an abstraction is genuinely painful, you may refactor it—
but keep the core invariants and the “no fake proofs” ethos.

---

## 0. Quick Start (What to run)

### Build (warnings are errors)
- `lake build -q --wfail`

### Build the CLI
- `lake build nfp -q --wfail`

### Run the CLI (preferred integration path)
One of these typically works (depending on your Lake setup):
- `lake exe nfp --help`

If you add or change CLI behavior, validate at least:
- `nfp --help`
- `nfp analyze --help`
- `nfp induction --help`
- `nfp --version` (if supported)

Before you finish any change:
- `lake build -q --wfail`
- `lake build nfp -q --wfail`

Note: `models/` is gitignored, so `rg` will skip it unless you pass `--no-ignore`
or `-uuu` (or equivalent) when searching.

---

## 1. Non-Negotiables (Hard Rules)

### 1.1 No fake proofs
- **Forbidden:** `sorry`
- **Forbidden:** introducing new nontrivial axioms beyond what mathlib already uses.
- If you can’t prove a lemma as stated:
  - reconsider the statement (missing assumptions? wrong generality?),
  - introduce helper lemmas,
  - or refactor the structure so the proof becomes natural.
  - Do **not** “paper over” gaps.

> Lean 4.26+ exploration tools (`finish?`, `try?`, `grind => finish?`, etc.) may *suggest* scripts that
> contain `sorry` (useful for debugging). Treat those suggestions as **scratch** only.
> **No `sorry` may reach the branch.**

### 1.2 Linting stays on
- **Never** disable linters globally or locally.
- **Forbidden:** any `set_option linter.* false` (including e.g. `linter.unnecessarySimpa`).
- Fix the code/proofs instead.

### 1.3 Clean build
- `lake build -q --wfail` must succeed.
- Any warning is treated as an error: resolve it, do not ignore it.

### 1.4 Core invariants must remain true
The library’s claims rest on these being preserved (preferably with explicit lemmas):
- Probability vectors sum to `1`
- Nonnegativity for masses/capacities/probabilities
- Mixers are row-stochastic when intended
- DAG / acyclicity assumptions are not silently violated
- Finiteness assumptions (`[Fintype _]`) are used intentionally and consistently

### 1.5 Trusted Code Verification (Total Soundness)
**All code** in trusted namespaces (e.g., `Nfp.Sound.*`) must be **verified**.
- **Requirement:** Every pure definition in the trusted scope must be characterized by a theorem
  or return a proof-carrying structure.
    - *Example (Bad):* `def addOne (x : Nat) := x + 1` (Unverified logic)
    - *Example (Good):* `def addOne (x : Nat) : { y // y > x } := ⟨x + 1, Nat.lt_succ_self _⟩`
    - *Example (Good):* `def addOne ...` followed immediately by `theorem addOne_gt_input ...`
- **Scope:** This applies to **everything**: parsers, converters, arithmetic helpers, and bound computations.
- **IO Exception:** Low-level IO primitives (reading bytes/files) cannot be "proven" correct but must be kept **logic-free**.
    - IO code should only read data and pass it to verified Pure code.
    - No mathematical transformations or complex branching allowed in IO functions.

---

## 2. Design Principles (Strong Preferences)

### 2.1 Finite + nonnegative by default
- Prefer `NNReal` for masses/capacities/probabilities.
- Prefer finite types (`[Fintype ι]`) where possible.

### 2.2 Keep proofs readable and local
- Prefer: `simp`, `rw`, `linarith`/`nlinarith` when appropriate, small `calc` blocks,
  and restrained `aesop` usage backed by named helper lemmas.
- Avoid huge opaque “mega proofs”. If a proof is long, factor it.

> Lean 4.26+ note for agents:
> - Use stronger automation (`simp?`, `finish?`, `grind`, `try?`) primarily as **proof discovery** tools.
> - The final committed proof should be **explicit, minimal, and stable** (often: a small lemma + `simp [..]` / `rw [..]`).

### 2.3 Don’t duplicate mathlib
- Search for existing lemmas before inventing new ones.
- If you introduce a lemma that feels “standard”, consider whether mathlib already has it
  (or whether it belongs in a more general file in this repo).

### 2.4 Verify, Don't Trust
- Distinguish between **witness generation** (untrusted, can use heuristics) and **verification** (trusted, must contain proofs).
- The trusted kernel should only check that a candidate witness is valid; it should not be responsible for finding it if the search is complex.

---

## 3. Workflow Expectations (How to make changes)

### 3.1 Before coding
- Identify the right module (see §5 Module Map).
- Skim the top docstring / main definitions in that module.
- Look for existing lemmas and naming patterns to match.

### 3.2 While coding
- Keep imports minimal and local.
- Add small helper lemmas rather than forcing a single theorem to do everything.
- If you add a new definition/theorem:
  - add a short docstring (what it is + why it exists),
  - and (when relevant) state the invariant it preserves/uses.
- Prefer *local* changes that keep compilation and elaboration fast:
  - small lemmas, smaller proof terms, fewer global simp rules.

### 3.3 After coding
- Ensure `lake build -q --wfail` passes.
- Ensure no `sorry`.
- Ensure no linter toggles were introduced.
- If you changed module responsibilities/structure, update §5 in the same commit.

---

## 4. Lean Style Guide (Project-specific)

### 4.1 Naming and organization
- Prefer consistent, descriptive names:
  - `_lemma`, `_iff`, `_eq`, `_mono`, `_nonneg`, `_sum_one`, etc.
- Keep namespaces coherent:
  - attach lemmas to the structure/namespace they conceptually belong to.

### 4.2 `simp` discipline
- Don’t spam `[simp]`. Only mark lemmas simp when they are:
  - terminating,
  - non-explosive,
  - and broadly safe.
- Prefer `simp [foo]` over global simp-set growth.
- Prefer `simp?` **only to discover** what `simp [..]` should be.

### 4.3 Tactic usage
- `aesop` is allowed, but:
  - avoid relying on “magic” if it makes failures hard to debug,
  - extract key steps into named lemmas so proofs stay stable.

### 4.4 Refactors are allowed—but must be principled
- You may do nontrivial refactors to improve conceptual cleanliness.
- If you rename/reshape core APIs:
  - update all call sites,
  - leave a brief comment (or commit message rationale),
  - keep the module map (§5) accurate.

### 4.5 Lean 4.26+ proof exploration toolkit (for LLM agents)
These tools can dramatically reduce “stuck time” for lemma discovery. Use them like a **search assistant**.
They are *not* a substitute for readable proofs.

**Allowed for exploration (scratch / development):**
- `simp?` (optionally with suggestions, if available)
- `finish?`
- `grind` / `grind?`, and `grind => finish?`
- `try?` (as a hint generator)

**Rules for using exploration tools:**
1. **Never commit generated `sorry`.** If an exploration tactic suggests a script with `sorry`, treat it as debugging output and delete it.
2. **Never commit giant opaque scripts.** If a generated script is long:
   - identify the key lemmas it used,
   - create named helper lemmas,
   - replace the script with a small proof built from those lemmas.
3. **Minimize lemma sets.**
   - If `simp?` / `finish?` / `grind` suggests many lemmas, shrink to the smallest stable subset.
4. **Prefer stable shapes:**
   - a short `calc` block,
   - or a couple of `simp [..]` / `rw [..]` steps,
   - plus one helper lemma if necessary.
5. **Keep it local.** Prefer adding lemmas to the local simp set (`simp [foo, bar]`) over tagging globally `[simp]`.

**Agent “proof playbook” (recommended loop):**
- Step A: Try the obvious: `simp`, `simp [defs]`, `rw [defs]`, `linarith`, `nlinarith`, `ring`, `field_simp` (as appropriate).
- Step B: If stuck, run `simp?` to discover missing rewrite/simp lemmas.
- Step C: If still stuck, use `finish?` or `grind => finish?` to learn the *shape* of the proof and which lemmas matter.
- Step D: Replace the discovered script with:
  - a helper lemma (named + documented) capturing the crucial step,
  - and a short final proof using `simp`/`rw`/`calc`.
- Step E: Re-run `lake build -q --wfail`.

---

## Lean 4 performance & scalability (use when justified)

Default: write the simplest correct thing first. Use the levers below only when there is a clear payoff
(hot path, large workload, or expensive work that’s often unused). Add a short comment explaining the trigger.

### Parallelism: `Task` (opt-in, deterministic-by-construction)
Use `Task` when work is independent and CPU-heavy (e.g., per-candidate / per-layer computations).
- Prefer *pure* tasks: `Task.spawn (fun () => ...)` and later `Task.get`.
  Tasks cache their result; subsequent `get`s do not recompute. (Tasks are like “opportunistic thunks”.)
- Use `IO.asTask` only when you truly need effects; remember a task is spawned each time the returned `IO` action is executed.
- Keep results deterministic: never depend on completion order; aggregate by stable keys.
- Keep granularity coarse enough to amortize scheduling overhead.
- Cancellation: pure tasks stop when dropped; `IO.asTask` tasks must check for cancellation (`IO.checkCanceled`), and can be canceled via `IO.cancel`.
- If benchmarking, note that the runtime task thread pool size is controlled by `LEAN_NUM_THREADS` (or defaults to logical CPU count).

### Laziness: `Thunk` / delayed computations (opt-in, for expensive-but-often-unused work)
Use `Thunk` to defer work that is expensive and frequently unused (debug traces, optional certificates, rare branches).
- Prefer `Thunk` over “manual caching”: the runtime forces at most once and caches the value.
- Force explicitly at the boundary (`Thunk.get`), not “deep inside” unrelated logic.
- If a thunk is forced from multiple threads, other threads will wait while one thread computes it—avoid forcing in places where blocking could deadlock.

### Compile-time / elaboration performance nudge
When proofs or declarations get large, prefer factoring them into smaller independent theorems/lemmas when it improves clarity.
Lean can elaborate theorem bodies in parallel, so smaller independent units can help the compiler do more work concurrently.

### Transparency / unfolding control (use sparingly)
Unfolding choices affect performance of simplification and typeclass search.
- The simplifier unfolds *reducible* definitions by default; semireducible/irreducible require explicit rewrite rules or different settings.
- `opaque` definitions are not δ-reduced in the kernel; use them to prevent expensive kernel reduction when unfolding is not needed for reasoning.
- Avoid cargo-culting reducibility attributes: use `local`/`scoped` when possible, and leave a short comment about why.

Note: Recent Lean versions changed the story around well-founded recursion transparency; don’t rely on old recipes like making well-founded recursion “reducible” via attributes.

---

## 5. Module Map (Where Things Live)

This is a *map*, not a prison. You may reshuffle if a better design emerges,
but you **must** update this list in the same commit.

### 5.1 Core types
- `Nfp/Core/Basic.lean`
  - `Mass` alias for nonnegative weights used throughout the rewrite.
- `Nfp/Core.lean`
  - Aggregator for core shared definitions.

### 5.2 Probability vectors
- `Nfp/Prob/Basic.lean`
  - `ProbVec` definition + invariants.
- `Nfp/Prob/Operations.lean`
  - `pure`, `mix`, and basic lemmas.
- `Nfp/Prob.lean`
  - Aggregator for probability modules.

### 5.3 Mixers
- `Nfp/Mixer/Basic.lean`
  - `Mixer` structure and row-stochastic invariant.
- `Nfp/Mixer/Operations.lean`
  - `push`, `comp`, and `id` mixers.
- `Nfp/Mixer.lean`
  - Aggregator for mixer modules.

### 5.4 Systems (DAG + local mixing)
- `Nfp/System/Dag.lean`
  - DAG relation + parent/child sets.
- `Nfp/System/LocalSystem.lean`
  - `LocalSystem` with edge support, row-stochastic predicate, and evaluation semantics.
- `Nfp/System.lean`
  - Aggregator for system modules.

### 5.5 Circuits (certification core)
- `Nfp/Circuit/Basic.lean`
  - DAG-based circuit structure with inputs/outputs and gate semantics.
- `Nfp/Circuit/Combinators.lean`
  - Core circuit combinators (relabeling, interface transport).
- `Nfp/Circuit/Interface.lean`
  - Typed input/output interfaces and interface-based evaluation.
- `Nfp/Circuit/Semantics.lean`
  - Well-founded evaluation semantics for circuits.
- `Nfp/Circuit/WellFormed.lean`
  - Basic well-formedness conditions for circuit inputs.
- `Nfp/Circuit/Cert.lean`
  - Equivalence definition and finite checker.
- `Nfp/Circuit/Cert/SoftmaxMargin.lean`
  - Softmax-margin certificate payloads and checker soundness.
- `Nfp/Circuit/Cert/ValueRange.lean`
  - Value-range certificate payloads and checker soundness.
- `Nfp/Circuit/Typed.lean`
  - Typed circuit wrapper and interface-level equivalence checker.
- `Nfp/Circuit/Compose.lean`
  - Sequential composition and residual wiring for typed circuits.
- `Nfp/Circuit/Gates/Basic.lean`
  - Basic gate combinators for aggregating parent values.
- `Nfp/Circuit/Gates/Linear.lean`
  - Linear and affine gate combinators built from `Matrix.mulVec`.
- `Nfp/Circuit/Gates.lean`
  - Aggregator for gate combinator modules.
- `Nfp/Circuit/Tensor.lean`
  - Typed tensor indices and tensor aliases.
- `Nfp/Circuit/Layers/Linear.lean`
  - Linear/affine layer circuits with typed interfaces.
- `Nfp/Circuit/Layers/Tensor.lean`
  - Batched linear/affine layer circuits for tensor-shaped data.
- `Nfp/Circuit/Layers/Reshape.lean`
  - Reshape combinators for product-typed circuit interfaces.
- `Nfp/Circuit/Layers/Heads.lean`
  - Head split/merge combinators for transformer-shaped indices.
- `Nfp/Circuit/Layers/Attention.lean`
  - Q/K/V, output projection wiring, and attention score/mixing core.
- `Nfp/Circuit/Layers/Induction.lean`
  - Induction-head weight specs and attention-core output lemmas.
- `Nfp/Circuit/Layers/TransformerBlock.lean`
  - GPT-style transformer block wiring from LN/attention/MLP circuits.
- `Nfp/Circuit/Layers.lean`
  - Aggregator for circuit layer modules.
- `Nfp/Circuit.lean`
  - Aggregator for circuit modules.

### 5.6 CLI surface
- `Nfp/IO/Pure.lean`
  - Pure parsing helpers for CLI inputs.
- `Nfp/IO.lean`
  - IO-only wrappers for loading inputs and running checks.
- `Nfp/Cli.lean`
  - CLI commands and `main` implementation.
- `Main.lean`
  - Thin entrypoint delegating to `Nfp.Cli.main`.
- `Nfp.lean`
  - Top-level reexports and axioms dashboard (`#print axioms`).

If you introduce a new conceptual layer:
- either extend the closest existing file,
- or add a new module with a clear name + top docstring,
- and update this map in the same commit.

---

## 6. Axioms & Trust Boundary

This repo treats “axioms creep” as a serious regression.

- Do not add axioms.
- Keep an eye on classical assumptions; they may be unavoidable, but should be explicit.
- Use `Nfp.lean` as the “trust dashboard” for `#print axioms` / dependency visibility.

---

## 7. Definition of Done (Checklist)

- [ ] `lake build -q --wfail` succeeds.
- [ ] No `sorry`.
- [ ] No new axioms were introduced.
- [ ] **Total Soundness:** Every pure definition in the trusted section is verified/proven.
- [ ] No linters were disabled (`set_option linter.* false` is absent).
- [ ] New nontrivial definitions/theorems have short, accurate docstrings.
- [ ] Core invariants (nonnegativity, normalization, finiteness, acyclicity) are preserved and, where possible, explicitly proved.
- [ ] §5 Module Map is accurate (updated in the same commit if needed).
- [ ] If CLI behavior changed: `lake build nfp -q --wfail` succeeds and basic `nfp ... --help` works.
- [ ] If you used Lean 4.26+ exploration tools, the final committed proof is short, explicit, and stable (no giant generated scripts).

When forced to choose between:
- “slightly breaking but conceptually clean redesign”
- vs “preserve an awkward design forever”

prefer the **clean redesign**, but do it consciously and document the rationale.
