-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Fold
public import Mathlib.Data.Finset.Insert
public import Mathlib.Data.Fintype.Pi
public import Nfp.Circuit.Interface
public import Nfp.Circuit.Semantics

/-!
Circuit equivalence and a finite checker.
-/

public section

namespace Nfp

universe u v u' u_in u_out

namespace Circuit

variable {ι : Type u} [Fintype ι] [DecidableEq ι]
variable {α : Type v}

/-- Circuits share the same input/output interface. -/
def SameInterface (C₁ C₂ : Circuit ι α) : Prop :=
  C₁.inputs = C₂.inputs ∧ C₁.outputs = C₂.outputs

/-- `SameInterface` is decidable. -/
private instance (C₁ C₂ : Circuit ι α) : Decidable (SameInterface C₁ C₂) := by
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

/-- Boolean `all` over a finset (tail-recursive fold over the multiset). -/
def finsetAll {β : Type v} (s : Finset β) (p : β → Bool) : Bool := by
  classical
  let f : Bool → β → Bool := fun acc a => acc && p a
  have hf : RightCommutative f := by
    refine ⟨?_⟩
    intro b a c
    calc
      f (f b a) c = ((b && p a) && p c) := rfl
      _ = (b && (p a && p c)) := by simp [Bool.and_assoc]
      _ = (b && (p c && p a)) := by simp [Bool.and_comm]
      _ = ((b && p c) && p a) := by simp [Bool.and_assoc]
      _ = f (f b c) a := rfl
  let _ : RightCommutative f := hf
  exact Multiset.foldl (f := f) (b := true) s.1

theorem finsetAll_eq_true_iff {β : Type v} {s : Finset β} {p : β → Bool} :
    finsetAll s p = true ↔ ∀ a ∈ s, p a = true := by
  classical
  let f : Bool → β → Bool := fun acc a => acc && p a
  have hf : RightCommutative f := by
    refine ⟨?_⟩
    intro b a c
    calc
      f (f b a) c = ((b && p a) && p c) := rfl
      _ = (b && (p a && p c)) := by simp [Bool.and_assoc]
      _ = (b && (p c && p a)) := by simp [Bool.and_comm]
      _ = ((b && p c) && p a) := by simp [Bool.and_assoc]
      _ = f (f b c) a := rfl
  let _ : RightCommutative f := hf
  have hfoldl :
      ∀ (s : Multiset β) (acc : Bool),
        Multiset.foldl (f := f) (b := acc) s = true ↔
          acc = true ∧ Multiset.foldl (f := f) (b := true) s = true := by
    intro s acc
    revert acc
    refine Multiset.induction_on s ?h0 ?hcons
    · intro acc
      simp [Multiset.foldl_zero]
    · intro a s ih acc
      have ih_acc :
          Multiset.foldl (f := f) (b := acc && p a) s = true ↔
            (acc && p a) = true ∧ Multiset.foldl (f := f) (b := true) s = true := by
        simpa using (ih (acc := acc && p a))
      have ih_pa :
          Multiset.foldl (f := f) (b := p a) s = true ↔
            p a = true ∧ Multiset.foldl (f := f) (b := true) s = true := by
        simpa using (ih (acc := p a))
      have hgoal :
          Multiset.foldl (f := f) (b := acc && p a) s = true ↔
            acc = true ∧ Multiset.foldl (f := f) (b := p a) s = true := by
        constructor
        · intro h
          have haccpa := ih_acc.mp h
          have haccpa' : acc = true ∧ p a = true := by
            simpa [Bool.and_eq_true] using haccpa.1
          have hacc : acc = true := haccpa'.1
          have hpa : p a = true := haccpa'.2
          have hfold : Multiset.foldl (f := f) (b := p a) s = true :=
            ih_pa.mpr ⟨hpa, haccpa.2⟩
          exact ⟨hacc, hfold⟩
        · intro h
          rcases h with ⟨hacc, hfold⟩
          have hpa := ih_pa.mp hfold
          have haccpa : (acc && p a) = true := by
            simpa [Bool.and_eq_true] using And.intro hacc hpa.1
          exact ih_acc.mpr ⟨haccpa, hpa.2⟩
      simpa [Multiset.foldl_cons, f] using hgoal
  induction s using Finset.induction_on with
  | empty =>
      simp [finsetAll, Multiset.foldl_zero]
  | @insert a s ha ih =>
      have hfold :
          finsetAll (insert a s) p = true ↔
            p a = true ∧ finsetAll s p = true := by
        have hval : (insert a s).1 = a ::ₘ s.1 := by
          simpa using (Finset.insert_val_of_notMem (a := a) (s := s) ha)
        calc
          finsetAll (insert a s) p = true ↔
              Multiset.foldl (f := f) (b := true) (insert a s).1 = true := by
                simp [finsetAll, f]
          _ ↔ Multiset.foldl (f := f) (b := true) (a ::ₘ s.1) = true := by
                simp [hval]
          _ ↔ Multiset.foldl (f := f) (b := f true a) s.1 = true := by
                simp [Multiset.foldl_cons]
          _ ↔ Multiset.foldl (f := f) (b := p a) s.1 = true := by
                simp [f]
          _ ↔ p a = true ∧ Multiset.foldl (f := f) (b := true) s.1 = true := by
                simpa using (hfoldl (s := s.1) (acc := p a))
          _ ↔ p a = true ∧ finsetAll s p = true := by
                simp [finsetAll, f]
      have hfold' :
          finsetAll (insert a s) p = true ↔ p a = true ∧ finsetAll s p = true := hfold
      simpa [Finset.forall_mem_insert, ih] using hfold'

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
