-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Image
public import Mathlib.Logic.Embedding.Basic
public import Nfp.Circuit.Basic
public import Nfp.Circuit.Gates.Linear
public import Nfp.Circuit.Typed

/-!
Linear and affine layer circuits.
-/

public section

namespace Nfp

namespace Circuit

namespace Layers

open Function

universe u v

variable {Row Col : Type u}

/-- Node type for a linear/affine layer from `Col` inputs to `Row` outputs. -/
abbrev LinearNode (Row Col : Type u) : Type u := Sum Col Row

/-- Rank function used to orient layer edges from inputs to outputs. -/
@[expose] def linearRank : LinearNode Row Col → Nat
  | Sum.inl _ => 0
  | Sum.inr _ => 1

section Dag

variable [Fintype Row] [Fintype Col]

/-- DAG for a single linear/affine layer. -/
def linearDag : Dag (LinearNode Row Col) :=
  { graph := { Adj := fun j i => linearRank (Row := Row) (Col := Col) j <
      linearRank (Row := Row) (Col := Col) i }
    decAdj := by
      intro j i
      infer_instance
    wf := by
      simpa using (InvImage.wf (f := linearRank (Row := Row) (Col := Col))
        (h := Nat.lt_wfRel.wf)) }

/-- Every input node has an edge to every output node. -/
theorem linearDag_rel_inl_inr (c : Col) (r : Row) :
    (linearDag (Row := Row) (Col := Col)).rel (Sum.inl c) (Sum.inr r) := by
  dsimp [linearDag, linearRank]
  exact Nat.zero_lt_one

end Dag

section Inputs

variable [Fintype Col]

/-- Input nodes for a linear/affine layer circuit. -/
def linearInputs : Finset (LinearNode Row Col) :=
  (Finset.univ : Finset Col).map Embedding.inl

/-- Membership in the input nodes corresponds to being a left injection. -/
theorem mem_linearInputs_iff {s : LinearNode Row Col} :
    s ∈ linearInputs (Row := Row) (Col := Col) ↔ ∃ c, s = Sum.inl c := by
  constructor
  · intro hs
    rcases (Finset.mem_map.1 hs) with ⟨c, _hc, hcs⟩
    exact ⟨c, hcs.symm⟩
  · rintro ⟨c, rfl⟩
    refine Finset.mem_map.2 ?_
    exact ⟨c, by simp, rfl⟩

/-- Right injections are not input nodes. -/
theorem not_mem_linearInputs_inr (r : Row) :
    Sum.inr r ∉ linearInputs (Row := Row) (Col := Col) := by
  intro h
  rcases (mem_linearInputs_iff (Row := Row) (Col := Col)).1 h with ⟨c, hcs⟩
  cases hcs

/-- Input labels correspond to input nodes in a linear/affine layer. -/
def linearInputEquiv : Col ≃ { i // i ∈ linearInputs (Row := Row) (Col := Col) } :=
  { toFun := fun c =>
      ⟨Sum.inl c, (mem_linearInputs_iff (Row := Row) (Col := Col)).2 ⟨c, rfl⟩⟩
    invFun := fun i =>
      match i with
      | ⟨Sum.inl c, _⟩ => c
      | ⟨Sum.inr r, h⟩ => False.elim
          (not_mem_linearInputs_inr (Row := Row) (Col := Col) r h)
    left_inv := by
      intro c
      rfl
    right_inv := by
      intro i
      cases i with
      | mk s hs =>
          cases s with
          | inl c => rfl
          | inr r =>
              cases (not_mem_linearInputs_inr (Row := Row) (Col := Col) r hs) }

end Inputs

section Outputs

variable [Fintype Row]

/-- Output nodes for a linear/affine layer circuit. -/
def linearOutputs : Finset (LinearNode Row Col) :=
  (Finset.univ : Finset Row).map Embedding.inr

/-- Membership in the output nodes corresponds to being a right injection. -/
theorem mem_linearOutputs_iff {s : LinearNode Row Col} :
    s ∈ linearOutputs (Row := Row) (Col := Col) ↔ ∃ r, s = Sum.inr r := by
  constructor
  · intro hs
    rcases (Finset.mem_map.1 hs) with ⟨r, _hr, hrs⟩
    exact ⟨r, hrs.symm⟩
  · rintro ⟨r, rfl⟩
    refine Finset.mem_map.2 ?_
    exact ⟨r, by simp, rfl⟩

/-- Left injections are not output nodes. -/
theorem not_mem_linearOutputs_inl (c : Col) :
    Sum.inl c ∉ linearOutputs (Row := Row) (Col := Col) := by
  intro h
  rcases (mem_linearOutputs_iff (Row := Row) (Col := Col)).1 h with ⟨r, hrs⟩
  cases hrs

/-- Output labels correspond to output nodes in a linear/affine layer. -/
def linearOutputEquiv : Row ≃ { i // i ∈ linearOutputs (Row := Row) (Col := Col) } :=
  { toFun := fun r =>
      ⟨Sum.inr r, (mem_linearOutputs_iff (Row := Row) (Col := Col)).2 ⟨r, rfl⟩⟩
    invFun := fun i =>
      match i with
      | ⟨Sum.inr r, _⟩ => r
      | ⟨Sum.inl c, h⟩ => False.elim
          (not_mem_linearOutputs_inl (Row := Row) (Col := Col) c h)
    left_inv := by
      intro r
      rfl
    right_inv := by
      intro i
      cases i with
      | mk s hs =>
          cases s with
          | inr r => rfl
          | inl c =>
              cases (not_mem_linearOutputs_inl (Row := Row) (Col := Col) c hs) }

end Outputs

section Circuits

variable [Fintype Row] [Fintype Col]
variable {Val : Type v} [NonUnitalNonAssocSemiring Val]

/-- Gate semantics for a linear layer circuit. -/
def linearGate (W : Matrix Row Col Val) :
    ∀ i, (∀ j, (linearDag (Row := Row) (Col := Col)).rel j i → Val) → Val := by
  intro i rec
  cases i with
  | inl _ =>
      exact 0
  | inr r =>
      let x : Col → Val := fun c =>
        rec (Sum.inl c) (linearDag_rel_inl_inr (Row := Row) (Col := Col) c r)
      exact Gates.linear W x r

/-- Gate semantics for an affine layer circuit. -/
def affineGate (W : Matrix Row Col Val) (b : Row → Val) :
    ∀ i, (∀ j, (linearDag (Row := Row) (Col := Col)).rel j i → Val) → Val := by
  intro i rec
  cases i with
  | inl _ =>
      exact 0
  | inr r =>
      let x : Col → Val := fun c =>
        rec (Sum.inl c) (linearDag_rel_inl_inr (Row := Row) (Col := Col) c r)
      exact Gates.affine W b x r

/-- Circuit for a linear layer. -/
def linearCircuit (W : Matrix Row Col Val) : Circuit (LinearNode Row Col) Val :=
  { dag := linearDag (Row := Row) (Col := Col)
    inputs := linearInputs (Row := Row) (Col := Col)
    outputs := linearOutputs (Row := Row) (Col := Col)
    gate := linearGate (Row := Row) (Col := Col) W }

/-- Circuit for an affine layer. -/
def affineCircuit (W : Matrix Row Col Val) (b : Row → Val) :
    Circuit (LinearNode Row Col) Val :=
  { dag := linearDag (Row := Row) (Col := Col)
    inputs := linearInputs (Row := Row) (Col := Col)
    outputs := linearOutputs (Row := Row) (Col := Col)
    gate := affineGate (Row := Row) (Col := Col) W b }

/-- Typed interface for a linear layer circuit. -/
def linearInterface (W : Matrix Row Col Val) :
    Interface (linearCircuit (Row := Row) (Col := Col) W) Col Row :=
  { inputs := linearInputEquiv (Row := Row) (Col := Col)
    outputs := linearOutputEquiv (Row := Row) (Col := Col) }

/-- Typed interface for an affine layer circuit. -/
def affineInterface (W : Matrix Row Col Val) (b : Row → Val) :
    Interface (affineCircuit (Row := Row) (Col := Col) W b) Col Row :=
  { inputs := linearInputEquiv (Row := Row) (Col := Col)
    outputs := linearOutputEquiv (Row := Row) (Col := Col) }

end Circuits

section Typed

variable [Fintype Row] [Fintype Col]
variable [DecidableEq Row] [DecidableEq Col]
variable {Val : Type v} [NonUnitalNonAssocSemiring Val]

/-- Typed linear layer circuit. -/
def linearTyped (W : Matrix Row Col Val) :
    TypedCircuit (LinearNode Row Col) Val Col Row :=
  { circuit := linearCircuit (Row := Row) (Col := Col) W
    interface := linearInterface (Row := Row) (Col := Col) W }

/-- Typed affine layer circuit. -/
def affineTyped (W : Matrix Row Col Val) (b : Row → Val) :
    TypedCircuit (LinearNode Row Col) Val Col Row :=
  { circuit := affineCircuit (Row := Row) (Col := Col) W b
    interface := affineInterface (Row := Row) (Col := Col) W b }

end Typed

end Layers

end Circuit

end Nfp
