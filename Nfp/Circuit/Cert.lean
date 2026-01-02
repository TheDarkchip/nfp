-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Semantics

/-!
Circuit equivalence and a finite checker.
-/

namespace Nfp

universe u v

namespace Circuit

variable {ι : Type u} [Fintype ι] [DecidableEq ι]
variable {α : Type v}

/-- Circuits share the same input/output interface. -/
def SameInterface (C₁ C₂ : Circuit ι α) : Prop :=
  C₁.inputs = C₂.inputs ∧ C₁.outputs = C₂.outputs

/-- Circuits are equivalent if they agree on outputs for all inputs on the same interface. -/
def Equiv (C₁ C₂ : Circuit ι α) : Prop :=
  SameInterface C₁ C₂ ∧
    ∀ input, ∀ i ∈ C₁.outputs, eval C₁ input i = eval C₂ input i

/-- Decide equivalence (classically); computational checkers can refine this. -/
noncomputable def checkEquiv (C₁ C₂ : Circuit ι α) : Bool := by
  classical
  exact decide (Equiv C₁ C₂)

/-- `checkEquiv` is sound and complete for `Equiv`. -/
theorem checkEquiv_eq_true_iff (C₁ C₂ : Circuit ι α) :
    checkEquiv C₁ C₂ = true ↔ Equiv C₁ C₂ := by
  classical
  simp [checkEquiv]

end Circuit

end Nfp
