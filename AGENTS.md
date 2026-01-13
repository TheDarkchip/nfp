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

**Use 'bd' for task tracking**

## 0. Quick Start (What to run)

### Build
- `lake build --wfail`

### Build the CLI
- `lake build nfp --wfail`

### Run the CLI (preferred integration path)
One of these typically works (depending on your Lake setup):
- `lake exe nfp --help`

### Search tips
Note: `models/` is gitignored, so `rg` will skip it unless you pass `--no-ignore`
or `-uuu` (or equivalent) when searching.

---

## 1. Non-Negotiables (Hard Rules)

### 1.1 No fake proofs
- **Forbidden:** `sorry`

### 1.2 Linting stays on
- **Never** disable linters globally or locally.
- **Forbidden:** any `set_option linter.* false` (including e.g. `linter.unnecessarySimpa`).
- Fix the code/proofs instead.
- If linters warn about line length or file length, prefer principled refactors
  (split modules, extract helpers) and keep docstrings with their code; avoid
  squashing whitespace or formatting.

### 1.3 Clean build
- `lake build --wfail` must succeed.
- Any warning is treated as an error: resolve it, do not ignore it.

### 1.4 Core invariants must remain true
The library’s claims rest on these being preserved (preferably with explicit lemmas):
- Probability vectors sum to `1`
- Nonnegativity for masses/capacities/probabilities
- Mixers are row-stochastic when intended
- DAG / acyclicity assumptions are not silently violated
- Finiteness assumptions (`[Fintype _]`) are used intentionally and consistently

### 1.5 Trusted Code Verification (Total Soundness)
**All code** in trusted namespaces (see §6) must be **verified**.
- **Requirement:** Every pure definition in the trusted scope must be characterized by a theorem
  or return a proof-carrying structure.
    - *Example (Bad):* `def addOne (x : Nat) := x + 1` (Unverified logic)
    - *Example (Good):* `def addOne (x : Nat) : { y // y > x } := ⟨x + 1, Nat.lt_succ_self _⟩`
    - *Example (Good):* `def addOne ...` followed immediately by `theorem addOne_gt_input ...`
- **Scope:** This applies to **everything**: parsers, converters, arithmetic helpers, and bound computations.
- **IO Exception:** Low-level IO primitives (reading bytes/files) cannot be "proven"
  correct but must be kept **logic-free**.
    - IO code should only read data and pass it to verified Pure code.
    - No mathematical transformations or complex branching allowed in IO functions.

---

## 2. Design Principles (Strong Preferences)

### 2.1 Finite + nonnegative by default
- Prefer `NNReal` for masses/capacities/probabilities.
- Prefer finite types (`[Fintype ι]`) where possible.

### 2.2 Don’t duplicate mathlib
- Search for existing lemmas before inventing new ones.
- If you introduce a lemma that feels “standard”, consider whether mathlib already has it
  (or whether it belongs in a more general file in this repo).

### 2.3 Verify, Don't Trust
- Distinguish between **witness generation** (untrusted, can use heuristics) and
  **verification** (trusted, must contain proofs).
- The trusted kernel should only check that a candidate witness is valid; it
  should not be responsible for finding it if the search is complex.

### 2.4 Prefer principled redesigns
When forced to choose between:
- “slightly breaking but conceptually clean redesign”
- vs “preserve an awkward design forever”

prefer the **clean redesign**, but do it consciously and document the rationale.

---

## 3. Workflow Expectations (How to make changes)

### 3.1 Before coding
- Identify the right module.
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
- Ensure the Definition of Done checklist is satisfied.

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

### 4.3 Proof automation discipline (Aesop-aware)

- Use `aesop?` (and `simp?`) to *discover* a proof or a minimal set of steps, then
  prefer to:
  - write a small explicit proof (`simp only [...]`, `constructor`, `cases`, `refine`, `exact`, etc.), or
  - keep Aesop but constrain it with a local ruleset and/or targeted rules.

- Avoid unconditional `by aesop` in core/trusted library code unless:
  - the goal is genuinely routine,
  - it stays fast and stable under small refactors, and
  - it does not rely on a large implicit rule universe.

- Prefer local rules over global rules:
  - If a lemma is meant to be reused by Aesop, tag it deliberately (e.g. `@[aesop safe]`)
    and explain why it is safe.
  - Avoid tagging “utility” lemmas as Aesop rules unless you want them participating
    in search broadly.

- If Aesop succeeds but produces a long/fragile search:
  - extract a helper lemma that expresses the key reasoning step,
  - prove that lemma explicitly (or with tightly scoped automation),
  - then let Aesop use the helper.

- Keep automation predictable:
  - prefer `simp only [...]` and small custom simp sets locally,
  - avoid growing the global simp set and global Aesop rule set without strong reason.


### 4.4 Refactors are allowed—but must be principled
- You may do nontrivial refactors to improve conceptual cleanliness.
- If you rename/reshape core APIs:
  - update all call sites,
  - leave a brief comment (or commit message rationale).

### 4.5 Mathlib Module Structure (Local Baseline)
Based on `.lake/packages/mathlib` in this workspace:
- Use module headers (`module`) in Lean source files, with `public import` in the header.
- Keep imports minimal; add only what the file directly uses.
- Put exported declarations in a `public section`; otherwise they remain private to the module.
- Do not expose bodies by default; use `@[expose] public section` only when downstream unfolding is required.
- Prefer explicit `*_def` lemmas over exposing bodies when you just need definitional rewriting.
- Avoid `import all` unless a proof truly needs the private scope; keep such imports local and documented.
- Keep aggregator modules (e.g., `Nfp.Core`, `Nfp.Sound`) as thin reexports.

---

## 6. Axioms & Trust Boundary

This repo treats “axioms creep” as a serious regression.

- Do not add axioms.
- Keep an eye on classical assumptions; they may be unavoidable, but should be explicit.
- Trusted namespaces are `Nfp.Sound.*`, `Nfp.IO.Pure.*`, and `Nfp.IO.NfptPure`.
  If another module is intended to be trusted, say so explicitly in its docstring
  and treat it as in-scope here.
- Use `TheoremAxioms.lean` / `lake build theorem-axioms --wfail` as the trust dashboard for
  `#print axioms` / dependency visibility.

---

## 7. Definition of Done (Checklist)

- [ ] `lake build --wfail` succeeds.
- [ ] No `sorry`.
- [ ] No new axioms were introduced.
- [ ] **Total Soundness:** Every pure definition in trusted namespaces is verified/proven.
- [ ] No linters were disabled (`set_option linter.* false` is absent).
- [ ] New nontrivial definitions/theorems have short, accurate docstrings.
- [ ] Core invariants (nonnegativity, normalization, finiteness, acyclicity) are
  preserved and, where possible, explicitly proved.
- [ ] If CLI behavior changed: `lake build nfp --wfail` succeeds and basic `nfp ... --help` works.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
