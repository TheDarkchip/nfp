# AGENTS.md

This repo is a Lean 4 **math library + CLI tool** formalizing an `Nfp`-style system:
finite probability on finite types, row-stochastic “mixers”, influence specs, and
uniqueness / attribution results (including neural-network–interpretation layers),
with an executable `nfp` CLI for analysis and verification.

You are an automated agent extending a *library* (with a small CLI shell), not an app.

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
- `./.lake/build/bin/nfp --help`

If you add or change CLI behavior, validate at least:
- `nfp --help`
- `nfp analyze --help`
- `nfp induction --help`
- `nfp --version` (if supported)

Before you finish any change:
- `lake build -q --wfail`
- `lake build nfp -q --wfail`

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

---

## 2. Design Principles (Strong Preferences)

### 2.1 Finite + nonnegative by default
- Prefer `NNReal` for masses/capacities/probabilities.
- Prefer finite types (`[Fintype ι]`) where possible.

### 2.2 Keep proofs readable and local
- Prefer: `simp`, `rw`, `linarith`/`nlinarith` when appropriate, small `calc` blocks,
  and restrained `aesop` usage backed by named helper lemmas.
- Avoid huge opaque “mega proofs”. If a proof is long, factor it.

### 2.3 Don’t duplicate mathlib
- Search for existing lemmas before inventing new ones.
- If you introduce a lemma that feels “standard”, consider whether mathlib already has it
  (or whether it belongs in a more general file in this repo).

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

### 5.1 Core probability + mixing
- `Prob.lean`
  - Probability vectors (`ProbVec`) on finite types; normalization + basic lemmas.
- `Mixer.lean`
  - Row-stochastic operators (“mixers”), composition/pushforward/support tools.
- `Influence.lean`
  - Influence specifications/families, capacities, scaling, and conversion into mixers.
- `Reroute/Partition.lean`
  - Finite partitions + reroute planning structures.
- `Reroute/Heat.lean`
  - Weighted reroute plans and induced “heat” distributions.
- `PCC.lean`
  - Tracer/contribution utilities; discrete AUC / interval machinery.
- `Uniqueness.lean`
  - `LocalSystem` for finite DAG mixing systems; uniqueness theorem(s) for tracers.
- `MixerLocalSystem.lean`
  - Bridges mixers-on-DAGs to `LocalSystem` (interpreters using a topo order).
- `Appendix.lean`
  - Supplemental lemmas and wrappers that don’t belong elsewhere.

### 5.2 Interpretability / NN-oriented layers (mathematical, mostly proofs)
- `Layers.lean`
  - Neural-network layer operations modeled as mixers; attribution/ablation/reachability laws.
- `Attribution.lean`
  - Interpretability axioms and bridges from tracer-based notions.
- `Induction.lean`
  - True induction head definitions and certification theorems (pattern + faithfulness + functional effect).
- `SignedMixer.lean`
  - Signed/real-weight generalization (negative weights, affine maps, etc.).
- `Linearization.lean`
  - Jacobian-based linearizations, decomposition results, deep composition/error theorems.
- `Abstraction.lean`
  - Causal-consistency / intervention correspondence between “real” networks and abstract DAG views.

### 5.3 Executable analysis & CLI surface
- `Discovery.lean`
  - Executable discovery + bound computations and verification pipeline.
  - May be performance-sensitive; keep proofs minimal and move them to proof modules when possible.
- `Sound/Decimal.lean`
  - Exact parsing of decimal/scientific numerals into `Rat` for sound mode.
- `Sound/Bounds.lean`
  - Exact (`Rat`) norm/bound utilities used by sound certification (no Float).
- `Sound/Bridge.lean`
  - Lemmas connecting `Rat`-level bounds to `SignedMixer` operator-norm bounds.
- `Sound/Cert.lean`
  - Certificate/report structures and pretty-printing for SOUND-mode output.
- `Sound/IO.lean`
  - Minimal `.nfpt` scanner for sound certification (parses only what is needed).
- `Sound/Demo.lean`
  - Tiny end-to-end lemma demo bridging to `Linearization.operatorNormBound`.
- `Verification.lean`
  - Executable **causal verification** via head ablation + runtime axiom checks (competence, control independence, energy matching).
- `IO.lean`
  - Parsing/loading/tokenization/report formatting glue.
  - **IO-only principle:** no heavy proofs; keep it as a bridge to filesystem/CLI.
- `Main.lean`
  - CLI entrypoint and subcommand wiring. Keep it thin:
    - argument parsing + calling into `Nfp.IO` / `Discovery` / `Nfp.Sound.*` reporting helpers,
    - minimal logic, minimal proof content.
- `Nfp.lean`
  - Top-level reexports and an axioms check (`#print axioms` / trust dashboard).

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
- [ ] No linters were disabled (`set_option linter.* false` is absent).
- [ ] New nontrivial definitions/theorems have short, accurate docstrings.
- [ ] Core invariants (nonnegativity, normalization, finiteness, acyclicity) are preserved and, where possible, explicitly proved.
- [ ] §5 Module Map is accurate (updated in the same commit if needed).
- [ ] If CLI behavior changed: `lake build nfp -q --wfail` succeeds and basic `nfp ... --help` works.

When forced to choose between:
- “slightly breaking but conceptually clean redesign”
- vs “preserve an awkward design forever”

prefer the **clean redesign**, but do it consciously and document the rationale.
