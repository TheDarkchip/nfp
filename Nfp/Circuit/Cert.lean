-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Fold
import Mathlib.Data.Finset.Insert
import Mathlib.Data.Fintype.Pi
import Nfp.Circuit.Interface
import Nfp.Circuit.Semantics

/-!
Circuit equivalence and a finite checker.
-/

namespace Nfp

universe u v u' u_in u_out

namespace Circuit

variable {ι : Type u} [Fintype ι] [DecidableEq ι]
variable {α : Type v}

/-- Circuits share the same input/output interface. -/
def SameInterface (C₁ C₂ : Circuit ι α) : Prop :=
  C₁.inputs = C₂.inputs ∧ C₁.outputs = C₂.outputs

/-- `SameInterface` is decidable. -/
instance (C₁ C₂ : Circuit ι α) : Decidable (SameInterface C₁ C₂) := by
  dsimp [SameInterface]
  infer_instance

/-- Circuits agree on outputs for all input assignments on a fixed interface. -/
def EquivOn (C₁ C₂ : Circuit ι α) (h : SameInterface C₁ C₂) : Prop :=
  ∀ input : C₁.InputAssignment, ∀ i ∈ C₁.outputs,
    evalInput C₁ input i = evalInput C₂ (InputAssignment.cast h.1 input) i

/-- Circuits are equivalent if they share an interface and agree on all inputs. -/
def Equiv (C₁ C₂ : Circuit ι α) : Prop :=
  ∃ h : SameInterface C₁ C₂, EquivOn C₁ C₂ h

section Interface

variable {ι₁ : Type u} [Fintype ι₁] [DecidableEq ι₁]
variable {ι₂ : Type u'} [Fintype ι₂] [DecidableEq ι₂]
variable {ι_in : Type u_in} {ι_out : Type u_out}

/-- Circuits agree on outputs for all typed inputs on a shared interface. -/
def EquivOnInterface (C₁ : Circuit ι₁ α) (C₂ : Circuit ι₂ α)
    (I₁ : Interface C₁ ι_in ι_out) (I₂ : Interface C₂ ι_in ι_out) : Prop :=
  ∀ input : ι_in → α, ∀ o : ι_out, I₁.eval input o = I₂.eval input o

end Interface

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

/-- Decide equivalence by enumerating all input assignments on a finite value type. -/
def checkEquiv (C₁ C₂ : Circuit ι α) [Fintype α] [DecidableEq α] : Bool :=
  if h : SameInterface C₁ C₂ then
    finsetAll (Finset.univ : Finset C₁.InputAssignment) (fun input =>
      finsetAll C₁.outputs (fun i =>
        decide (evalInput C₁ input i = evalInput C₂ (InputAssignment.cast h.1 input) i)))
  else
    false

/-- `checkEquiv` is sound and complete for `Equiv`. -/
theorem checkEquiv_eq_true_iff (C₁ C₂ : Circuit ι α) [Fintype α] [DecidableEq α] :
    checkEquiv C₁ C₂ = true ↔ Equiv C₁ C₂ := by
  classical
  by_cases h : SameInterface C₁ C₂
  · have hcheck : checkEquiv C₁ C₂ = true ↔ EquivOn C₁ C₂ h := by
      simp [checkEquiv, h, EquivOn, finsetAll_eq_true_iff]
    constructor
    · intro hc
      exact ⟨h, hcheck.mp hc⟩
    · intro hEquiv
      rcases hEquiv with ⟨h', hEq⟩
      have hh : h' = h := Subsingleton.elim _ _
      exact hcheck.mpr (by simpa [hh] using hEq)
  · simp [checkEquiv, h, Equiv]

end

section InterfaceCheck

variable {ι₁ : Type u} [Fintype ι₁] [DecidableEq ι₁]
variable {ι₂ : Type u'} [Fintype ι₂] [DecidableEq ι₂]
variable {ι_in : Type u_in} [Fintype ι_in] [DecidableEq ι_in]
variable {ι_out : Type u_out} [Fintype ι_out]

/-- Decide interface-based equivalence by enumerating typed inputs. -/
def checkEquivOnInterface (C₁ : Circuit ι₁ α) (C₂ : Circuit ι₂ α)
    (I₁ : Interface C₁ ι_in ι_out) (I₂ : Interface C₂ ι_in ι_out)
    [Fintype α] [DecidableEq α] : Bool :=
  finsetAll (Finset.univ : Finset (ι_in → α)) (fun input =>
    finsetAll (Finset.univ : Finset ι_out) (fun o =>
      decide (I₁.eval input o = I₂.eval input o)))

/-- `checkEquivOnInterface` is sound and complete for `EquivOnInterface`. -/
theorem checkEquivOnInterface_eq_true_iff (C₁ : Circuit ι₁ α) (C₂ : Circuit ι₂ α)
    (I₁ : Interface C₁ ι_in ι_out) (I₂ : Interface C₂ ι_in ι_out)
    [Fintype α] [DecidableEq α] :
    checkEquivOnInterface C₁ C₂ I₁ I₂ = true ↔ EquivOnInterface C₁ C₂ I₁ I₂ := by
  classical
  simp [checkEquivOnInterface, EquivOnInterface, finsetAll_eq_true_iff]

end InterfaceCheck

end Circuit

end Nfp
