-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Fold
import Mathlib.Data.Finset.Insert
import Mathlib.Data.Fintype.Pi
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

section

local instance : Std.Commutative (α := Bool) (· && ·) := ⟨Bool.and_comm⟩
local instance : Std.Associative (α := Bool) (· && ·) := ⟨Bool.and_assoc⟩

/-- Boolean `all` over a finset. -/
def finsetAll {β : Type v} (s : Finset β) (p : β → Bool) : Bool :=
  s.fold (· && ·) true p

theorem finsetAll_eq_true_iff {β : Type v} {s : Finset β} {p : β → Bool} :
    finsetAll s p = true ↔ ∀ a ∈ s, p a = true := by
  classical
  induction s using Finset.induction_on with
  | empty =>
      simp [finsetAll]
  | @insert a s ha ih =>
      have hfold : finsetAll (insert a s) p = true ↔ p a = true ∧ finsetAll s p = true := by
        simp [finsetAll, ha, Bool.and_eq_true]
      calc
        finsetAll (insert a s) p = true
            ↔ p a = true ∧ finsetAll s p = true := hfold
        _ ↔ p a = true ∧ ∀ a ∈ s, p a = true := by simp [ih]
        _ ↔ ∀ x ∈ insert a s, p x = true := by
              constructor
              · intro h x hx
                rcases h with ⟨ha', hs⟩
                by_cases hx' : x = a
                · simpa [hx'] using ha'
                · exact hs x (Finset.mem_of_mem_insert_of_ne hx hx')
              · intro h
                refine ⟨?_, ?_⟩
                · exact h a (Finset.mem_insert_self a s)
                · intro x hx
                  exact h x (Finset.mem_insert_of_mem hx)

/-- Boolean check for interface equality. -/
def sameInterface (C₁ C₂ : Circuit ι α) : Bool :=
  decide (C₁.inputs = C₂.inputs) && decide (C₁.outputs = C₂.outputs)

theorem sameInterface_eq_true_iff (C₁ C₂ : Circuit ι α) :
    sameInterface C₁ C₂ = true ↔ SameInterface C₁ C₂ := by
  simp [sameInterface, SameInterface, Bool.and_eq_true]

/-- Decide equivalence by enumerating all inputs on a finite value type. -/
def checkEquiv (C₁ C₂ : Circuit ι α) [Fintype α] [DecidableEq α] : Bool :=
  sameInterface C₁ C₂ &&
    finsetAll (Finset.univ : Finset (ι → α)) (fun input =>
      finsetAll C₁.outputs (fun i => decide (eval C₁ input i = eval C₂ input i)))

/-- `checkEquiv` is sound and complete for `Equiv`. -/
theorem checkEquiv_eq_true_iff (C₁ C₂ : Circuit ι α) [Fintype α] [DecidableEq α] :
    checkEquiv C₁ C₂ = true ↔ Equiv C₁ C₂ := by
  classical
  simp [checkEquiv, Equiv, sameInterface_eq_true_iff, finsetAll_eq_true_iff, Bool.and_eq_true]

end

end Circuit

end Nfp
