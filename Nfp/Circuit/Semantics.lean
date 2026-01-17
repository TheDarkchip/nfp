-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Basic

/-!
Evaluation semantics for finite circuits.
-/

public section

namespace Nfp

universe u v

namespace Circuit

variable {ι : Type u} [Fintype ι] [DecidableEq ι]
variable {α : Type v}

/-- One-step evaluation functional used by `eval`. -/
def evalStep (C : Circuit ι α) (input : ι → α)
    (i : ι) (rec : ∀ j, C.dag.rel j i → α) : α :=
  if _ : i ∈ C.inputs then input i else C.gate i rec

/-- Evaluate a circuit with a given input assignment. -/
def eval (C : Circuit ι α) (input : ι → α) : ι → α :=
  C.dag.wf.fix (fun i rec => evalStep C input i rec)

/-- Unfolding equation for `eval`. -/
theorem eval_eq (C : Circuit ι α) (input : ι → α) (i : ι) :
    eval C input i =
      if _ : i ∈ C.inputs then input i else C.gate i (fun j _ => eval C input j) := by
  set F : ∀ i, (∀ j, C.dag.rel j i → α) → α := fun i rec => evalStep C input i rec
  change C.dag.wf.fix F i =
    if _ : i ∈ C.inputs then input i else C.gate i (fun j _ => C.dag.wf.fix F j)
  rw [WellFounded.fix_eq]
  simp [F, evalStep]

/-- Input nodes evaluate to their assigned input value. -/
theorem eval_eq_input (C : Circuit ι α) (input : ι → α) {i : ι} (h : i ∈ C.inputs) :
    eval C input i = input i := by
  simpa [h] using (eval_eq C input i)

/-- Non-input nodes evaluate via their gate semantics. -/
theorem eval_eq_gate (C : Circuit ι α) (input : ι → α) {i : ι} (h : i ∉ C.inputs) :
    eval C input i = C.gate i (fun j _ => eval C input j) := by
  simpa [h] using (eval_eq C input i)

/-- One-step evaluation functional used by `evalInput`. -/
def evalInputStep (C : Circuit ι α) (input : C.InputAssignment)
    (i : ι) (rec : ∀ j, C.dag.rel j i → α) : α :=
  if h : i ∈ C.inputs then input ⟨i, h⟩ else C.gate i rec

/-- Evaluate a circuit with an input assignment defined on input nodes. -/
def evalInput (C : Circuit ι α) (input : C.InputAssignment) : ι → α :=
  C.dag.wf.fix (fun i rec => evalInputStep C input i rec)

/-- Unfolding equation for `evalInput`. -/
theorem evalInput_eq (C : Circuit ι α) (input : C.InputAssignment) (i : ι) :
    evalInput C input i =
      if h : i ∈ C.inputs then input ⟨i, h⟩ else C.gate i (fun j _ => evalInput C input j) := by
  set F : ∀ i, (∀ j, C.dag.rel j i → α) → α := fun i rec => evalInputStep C input i rec
  change C.dag.wf.fix F i =
    if h : i ∈ C.inputs then input ⟨i, h⟩ else C.gate i (fun j _ => C.dag.wf.fix F j)
  rw [WellFounded.fix_eq]
  simp [F, evalInputStep]

/-- Input nodes evaluate to their assigned input value (input-only form). -/
theorem evalInput_eq_input (C : Circuit ι α) (input : C.InputAssignment) {i : ι}
    (h : i ∈ C.inputs) :
    evalInput C input i = input ⟨i, h⟩ := by
  simpa [h] using (evalInput_eq C input i)

/-- Non-input nodes evaluate via their gate semantics (input-only form). -/
theorem evalInput_eq_gate (C : Circuit ι α) (input : C.InputAssignment) {i : ι}
    (h : i ∉ C.inputs) :
    evalInput C input i = C.gate i (fun j _ => evalInput C input j) := by
  simpa [h] using (evalInput_eq C input i)

end Circuit

end Nfp
