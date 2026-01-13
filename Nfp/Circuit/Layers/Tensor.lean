-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Layers.Linear

/-!
Tensor-shaped layer builders (batched linear and affine layers).
-/

public section

namespace Nfp

namespace Circuit

namespace Layers

universe u v

variable {Batch Row Col : Type u}

/-- Node type for a batched linear/affine layer. -/
abbrev BatchedLinearNode (Batch Row Col : Type u) : Type u :=
  LinearNode (Batch × Row) (Batch × Col)

/-- Adjacency for batched linear layers: inputs connect only to outputs in the same batch. -/
def batchedLinearAdj (Batch Row Col : Type u) :
    BatchedLinearNode Batch Row Col → BatchedLinearNode Batch Row Col → Prop
  | Sum.inl (b, _), Sum.inr (b', _) => b = b'
  | _, _ => False

section Dag

variable [Fintype Batch] [Fintype Row] [Fintype Col]
variable [DecidableEq Batch]

/-- DAG for a batched linear/affine layer. -/
def batchedLinearDag : Dag (BatchedLinearNode Batch Row Col) :=
  { graph := { Adj := batchedLinearAdj Batch Row Col }
    decAdj := by
      intro j i
      cases j with
      | inl bc =>
          cases i with
          | inl _ =>
              exact isFalse (by intro h; cases h)
          | inr br =>
              exact (inferInstance : Decidable (bc.1 = br.1))
      | inr _ =>
          cases i with
          | inl _ =>
              exact isFalse (by intro h; cases h)
          | inr _ =>
              exact isFalse (by intro h; cases h)
    wf := by
      have hsub : Subrelation (batchedLinearAdj Batch Row Col)
          (fun j i =>
            linearRank (Row := Batch × Row) (Col := Batch × Col) j <
              linearRank (Row := Batch × Row) (Col := Batch × Col) i) := by
        intro j i h
        cases j <;> cases i <;> simp [batchedLinearAdj, linearRank] at h ⊢
      have hwf : WellFounded (fun j i =>
          linearRank (Row := Batch × Row) (Col := Batch × Col) j <
            linearRank (Row := Batch × Row) (Col := Batch × Col) i) := by
        simpa using (InvImage.wf
          (f := linearRank (Row := Batch × Row) (Col := Batch × Col))
          (h := Nat.lt_wfRel.wf))
      exact Subrelation.wf hsub hwf }

/-- Edges connect each batch's inputs to its outputs. -/
theorem batchedLinearDag_rel_inl_inr (b : Batch) (c : Col) (r : Row) :
    (batchedLinearDag (Batch := Batch) (Row := Row) (Col := Col)).rel
      (Sum.inl (b, c)) (Sum.inr (b, r)) := by
  change batchedLinearAdj Batch Row Col (Sum.inl (b, c)) (Sum.inr (b, r))
  simp [batchedLinearAdj]

end Dag

section Circuits

variable [Fintype Batch] [Fintype Row] [Fintype Col]
variable [DecidableEq Batch]
variable {Val : Type v} [NonUnitalNonAssocSemiring Val]

/-- Gate semantics for a batched linear layer circuit. -/
def batchedLinearGate (W : Matrix Row Col Val) :
    ∀ i,
      (∀ j,
          (batchedLinearDag (Batch := Batch) (Row := Row) (Col := Col)).rel j i → Val) →
        Val := by
  intro i rec
  cases i with
  | inl _ =>
      exact 0
  | inr br =>
      cases br with
      | mk b r =>
          let x : Col → Val := fun c =>
            rec (Sum.inl (b, c))
              (batchedLinearDag_rel_inl_inr (Batch := Batch) (Row := Row) (Col := Col) b c r)
          exact Gates.linear W x r

/-- Gate semantics for a batched affine layer circuit. -/
def batchedAffineGate (W : Matrix Row Col Val) (bias : Row → Val) :
    ∀ i,
      (∀ j,
          (batchedLinearDag (Batch := Batch) (Row := Row) (Col := Col)).rel j i → Val) →
        Val := by
  intro i rec
  cases i with
  | inl _ =>
      exact 0
  | inr br =>
      cases br with
      | mk b r =>
          let x : Col → Val := fun c =>
            rec (Sum.inl (b, c))
              (batchedLinearDag_rel_inl_inr (Batch := Batch) (Row := Row) (Col := Col) b c r)
          exact Gates.affine W bias x r

/-- Circuit for a batched linear layer. -/
def batchedLinearCircuit (W : Matrix Row Col Val) :
    Circuit (BatchedLinearNode Batch Row Col) Val :=
  { dag := batchedLinearDag (Batch := Batch) (Row := Row) (Col := Col)
    inputs := linearInputs (Row := Batch × Row) (Col := Batch × Col)
    outputs := linearOutputs (Row := Batch × Row) (Col := Batch × Col)
    gate := batchedLinearGate (Batch := Batch) (Row := Row) (Col := Col) W }

/-- Circuit for a batched affine layer. -/
def batchedAffineCircuit (W : Matrix Row Col Val) (bias : Row → Val) :
    Circuit (BatchedLinearNode Batch Row Col) Val :=
  { dag := batchedLinearDag (Batch := Batch) (Row := Row) (Col := Col)
    inputs := linearInputs (Row := Batch × Row) (Col := Batch × Col)
    outputs := linearOutputs (Row := Batch × Row) (Col := Batch × Col)
    gate := batchedAffineGate (Batch := Batch) (Row := Row) (Col := Col) W bias }

/-- Typed interface for a batched linear layer circuit. -/
def batchedLinearInterface (W : Matrix Row Col Val) :
    Interface (batchedLinearCircuit (Batch := Batch) (Row := Row) (Col := Col) W)
      (Batch × Col) (Batch × Row) :=
  { inputs := linearInputEquiv (Row := Batch × Row) (Col := Batch × Col)
    outputs := linearOutputEquiv (Row := Batch × Row) (Col := Batch × Col) }

/-- Typed interface for a batched affine layer circuit. -/
def batchedAffineInterface (W : Matrix Row Col Val) (bias : Row → Val) :
    Interface (batchedAffineCircuit (Batch := Batch) (Row := Row) (Col := Col) W bias)
      (Batch × Col) (Batch × Row) :=
  { inputs := linearInputEquiv (Row := Batch × Row) (Col := Batch × Col)
    outputs := linearOutputEquiv (Row := Batch × Row) (Col := Batch × Col) }

end Circuits

section Typed

variable [Fintype Batch] [Fintype Row] [Fintype Col]
variable [DecidableEq Batch] [DecidableEq Row] [DecidableEq Col]
variable {Val : Type v} [NonUnitalNonAssocSemiring Val]

/-- Typed batched linear layer circuit. -/
def batchedLinearTyped (W : Matrix Row Col Val) :
    TypedCircuit (BatchedLinearNode Batch Row Col) Val (Batch × Col) (Batch × Row) :=
  { circuit := batchedLinearCircuit (Batch := Batch) (Row := Row) (Col := Col) W
    interface := batchedLinearInterface (Batch := Batch) (Row := Row) (Col := Col) W }

/-- Typed batched affine layer circuit. -/
def batchedAffineTyped (W : Matrix Row Col Val) (bias : Row → Val) :
    TypedCircuit (BatchedLinearNode Batch Row Col) Val (Batch × Col) (Batch × Row) :=
  { circuit := batchedAffineCircuit (Batch := Batch) (Row := Row) (Col := Col) W bias
    interface := batchedAffineInterface (Batch := Batch) (Row := Row) (Col := Col) W bias }

end Typed

end Layers

end Circuit

end Nfp
