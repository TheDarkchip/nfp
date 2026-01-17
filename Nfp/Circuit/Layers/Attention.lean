-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Image
public import Mathlib.Data.Matrix.Mul
public import Mathlib.Logic.Embedding.Basic
public import Nfp.Circuit.Layers.Heads
public import Nfp.Circuit.Layers.Tensor

/-!
QKV and output projection wiring for attention layers, plus attention score/mixing core.
-/

public section

namespace Nfp

namespace Circuit

namespace Layers

open Function

universe v

variable {Batch : Type} [Fintype Batch]
variable {Val : Type v} [NonUnitalNonAssocSemiring Val]

/-- Hidden dimension type for a `heads × dim` factorization. -/
abbrev Hidden (heads dim : Nat) : Type := Fin (heads * dim)

/-- Node type for Q/K/V and output projection layers. -/
abbrev QkvNode (Batch : Type) (heads dim : Nat) : Type :=
  BatchedLinearNode Batch (Hidden heads dim) (Hidden heads dim)

section Projections

variable [DecidableEq Batch]

/-- Q projection with head-split output. -/
def qProj (heads dim : Nat) (Wq : Matrix (Hidden heads dim) (Hidden heads dim) Val) :
    TypedCircuit (QkvNode Batch heads dim) Val (Batch × Hidden heads dim)
      (Batch × Fin heads × Fin dim) :=
  splitHeadsOutput (Batch := Batch) heads dim
    (batchedLinearTyped (Batch := Batch)
      (Row := Hidden heads dim) (Col := Hidden heads dim) Wq)

/-- K projection with head-split output. -/
def kProj (heads dim : Nat) (Wk : Matrix (Hidden heads dim) (Hidden heads dim) Val) :
    TypedCircuit (QkvNode Batch heads dim) Val (Batch × Hidden heads dim)
      (Batch × Fin heads × Fin dim) :=
  splitHeadsOutput (Batch := Batch) heads dim
    (batchedLinearTyped (Batch := Batch)
      (Row := Hidden heads dim) (Col := Hidden heads dim) Wk)

/-- V projection with head-split output. -/
def vProj (heads dim : Nat) (Wv : Matrix (Hidden heads dim) (Hidden heads dim) Val) :
    TypedCircuit (QkvNode Batch heads dim) Val (Batch × Hidden heads dim)
      (Batch × Fin heads × Fin dim) :=
  splitHeadsOutput (Batch := Batch) heads dim
    (batchedLinearTyped (Batch := Batch)
      (Row := Hidden heads dim) (Col := Hidden heads dim) Wv)

/-- Output projection with head-merged input. -/
def outProj (heads dim : Nat) (Wo : Matrix (Hidden heads dim) (Hidden heads dim) Val) :
    TypedCircuit (QkvNode Batch heads dim) Val (Batch × Fin heads × Fin dim)
      (Batch × Hidden heads dim) :=
  splitHeadsInput (Batch := Batch) heads dim
    (batchedLinearTyped (Batch := Batch)
      (Row := Hidden heads dim) (Col := Hidden heads dim) Wo)

end Projections

/-
Attention score/mixing core.
-/

variable {seq heads dim : Nat}

/-- Index for per-token head-split vectors. -/
abbrev QkvIndex (Batch : Type) (seq heads dim : Nat) : Type :=
  Batch × Fin seq × Fin heads × Fin dim

/-- Index for attention score/weight entries. -/
abbrev ScoreIndex (Batch : Type) (seq heads : Nat) : Type :=
  Batch × Fin heads × Fin seq × Fin seq

/-- Index for attention weight entries (same shape as scores). -/
abbrev WeightIndex (Batch : Type) (seq heads : Nat) : Type :=
  ScoreIndex Batch seq heads

/-- Input-node labels for attention core circuits. -/
abbrev AttentionInputNode (Batch : Type) (seq heads dim : Nat) : Type :=
  Sum (QkvIndex Batch seq heads dim)
    (Sum (QkvIndex Batch seq heads dim) (QkvIndex Batch seq heads dim))

/-- Typed input labels for attention core circuits (Q/K/V). -/
abbrev AttentionInput (Batch : Type) (seq heads dim : Nat) : Type :=
  AttentionInputNode Batch seq heads dim

/-- Typed output labels for attention core circuits. -/
abbrev AttentionOutput (Batch : Type) (seq heads dim : Nat) : Type :=
  QkvIndex Batch seq heads dim

/-- Node type for the attention score/mixing core. -/
abbrev AttentionNode (Batch : Type) (seq heads dim : Nat) : Type :=
  Sum (AttentionInputNode Batch seq heads dim)
    (Sum (ScoreIndex Batch seq heads) (Sum (WeightIndex Batch seq heads)
      (AttentionOutput Batch seq heads dim)))

section Decidable

variable [DecidableEq Batch]

/-- Decidable equality for attention-core input nodes. -/
instance instDecidableEqAttentionInputNode :
    DecidableEq (AttentionInputNode Batch seq heads dim) := by
  intro x y
  cases x with
  | inl qx =>
      cases y with
      | inl qy =>
          simpa using (inferInstance : Decidable (qx = qy))
      | inr _ =>
          exact isFalse (by intro h; cases h)
  | inr x =>
      cases x with
      | inl kx =>
          cases y with
          | inl _ =>
              exact isFalse (by intro h; cases h)
          | inr y =>
              cases y with
              | inl ky =>
                  simpa using (inferInstance : Decidable (kx = ky))
              | inr _ =>
                  exact isFalse (by intro h; cases h)
      | inr vx =>
          cases y with
          | inl _ =>
              exact isFalse (by intro h; cases h)
          | inr y =>
              cases y with
              | inl _ =>
                  exact isFalse (by intro h; cases h)
              | inr vy =>
                  simpa using (inferInstance : Decidable (vx = vy))

/-- Decidable equality for attention-core nodes. -/
instance instDecidableEqAttentionNode :
    DecidableEq (AttentionNode Batch seq heads dim) := by
  intro x y
  cases x with
  | inl x =>
      cases y with
      | inl y =>
          simpa using (inferInstance : Decidable (x = y))
      | inr _ =>
          exact isFalse (by intro h; cases h)
  | inr x =>
      cases x with
      | inl sx =>
          cases y with
          | inl _ =>
              exact isFalse (by intro h; cases h)
          | inr y =>
              cases y with
              | inl sy =>
                  simpa using (inferInstance : Decidable (sx = sy))
              | inr _ =>
                  exact isFalse (by intro h; cases h)
      | inr x =>
          cases x with
          | inl wx =>
              cases y with
              | inl _ =>
                  exact isFalse (by intro h; cases h)
              | inr y =>
                  cases y with
                  | inl _ =>
                      exact isFalse (by intro h; cases h)
                  | inr y =>
                      cases y with
                      | inl wy =>
                          simpa using (inferInstance : Decidable (wx = wy))
                      | inr _ =>
                          exact isFalse (by intro h; cases h)
          | inr ox =>
              cases y with
              | inl _ =>
                  exact isFalse (by intro h; cases h)
              | inr y =>
                  cases y with
                  | inl _ =>
                      exact isFalse (by intro h; cases h)
                  | inr y =>
                      cases y with
                      | inl _ =>
                          exact isFalse (by intro h; cases h)
                      | inr oy =>
                          simpa using (inferInstance : Decidable (ox = oy))

end Decidable

/-- Inject a Q input node into the attention core node type. -/
abbrev attnQ (q : QkvIndex Batch seq heads dim) : AttentionNode Batch seq heads dim :=
  Sum.inl (Sum.inl q)

/-- Inject a K input node into the attention core node type. -/
abbrev attnK (k : QkvIndex Batch seq heads dim) : AttentionNode Batch seq heads dim :=
  Sum.inl (Sum.inr (Sum.inl k))

/-- Inject a V input node into the attention core node type. -/
abbrev attnV (v : QkvIndex Batch seq heads dim) : AttentionNode Batch seq heads dim :=
  Sum.inl (Sum.inr (Sum.inr v))

/-- Inject a score node into the attention core node type. -/
abbrev attnScore (s : ScoreIndex Batch seq heads) : AttentionNode Batch seq heads dim :=
  Sum.inr (Sum.inl s)

/-- Inject a weight node into the attention core node type. -/
abbrev attnWeight (w : WeightIndex Batch seq heads) : AttentionNode Batch seq heads dim :=
  Sum.inr (Sum.inr (Sum.inl w))

/-- Inject an output node into the attention core node type. -/
abbrev attnOut (o : AttentionOutput Batch seq heads dim) : AttentionNode Batch seq heads dim :=
  Sum.inr (Sum.inr (Sum.inr o))

/-- Rank function used to orient attention-core edges from inputs to outputs. -/
def attentionRank : AttentionNode Batch seq heads dim → Nat
  | Sum.inl _ => 0
  | Sum.inr (Sum.inl _) => 1
  | Sum.inr (Sum.inr (Sum.inl _)) => 2
  | Sum.inr (Sum.inr (Sum.inr _)) => 3

/-- Adjacency relation for attention-core wiring. -/
def attentionAdj : AttentionNode Batch seq heads dim → AttentionNode Batch seq heads dim → Prop
  | Sum.inl (Sum.inl (b, q, h, _)),
      Sum.inr (Sum.inl (b', h', q', _)) =>
      b = b' ∧ h = h' ∧ q = q'
  | Sum.inl (Sum.inr (Sum.inl (b, k, h, _))),
      Sum.inr (Sum.inl (b', h', _, k')) =>
      b = b' ∧ h = h' ∧ k = k'
  | Sum.inr (Sum.inl (b, h, q, _)),
      Sum.inr (Sum.inr (Sum.inl (b', h', q', _))) =>
      b = b' ∧ h = h' ∧ q = q'
  | Sum.inr (Sum.inr (Sum.inl (b, h, q, _))),
      Sum.inr (Sum.inr (Sum.inr (b', q', h', _))) =>
      b = b' ∧ h = h' ∧ q = q'
  | Sum.inl (Sum.inr (Sum.inr (b, _, h, d))),
      Sum.inr (Sum.inr (Sum.inr (b', _, h', d'))) =>
      b = b' ∧ h = h' ∧ d = d'
  | _, _ => False

section Dag

variable [DecidableEq Batch]

/-- DAG for the attention score/mixing core. -/
def attentionDag : Dag (AttentionNode Batch seq heads dim) :=
  { graph := { Adj := attentionAdj (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) }
    decAdj := by
      intro j i
      cases j with
      | inl j =>
          cases j with
          | inl q =>
              rcases q with ⟨b, q, h, _d⟩
              cases i with
              | inl _ =>
                  exact isFalse (by intro h; cases h)
              | inr i =>
                  cases i with
                  | inl s =>
                      rcases s with ⟨b', h', q', _k⟩
                      exact (inferInstance : Decidable (b = b' ∧ h = h' ∧ q = q'))
                  | inr _ =>
                      exact isFalse (by intro h; cases h)
          | inr j =>
              cases j with
              | inl k =>
                  rcases k with ⟨b, k, h, _d⟩
                  cases i with
                  | inl _ =>
                      exact isFalse (by intro h; cases h)
                  | inr i =>
                      cases i with
                      | inl s =>
                          rcases s with ⟨b', h', _q, k'⟩
                          exact (inferInstance : Decidable (b = b' ∧ h = h' ∧ k = k'))
                      | inr _ =>
                          exact isFalse (by intro h; cases h)
              | inr v =>
                  rcases v with ⟨b, _k, h, d⟩
                  cases i with
                  | inl _ =>
                      exact isFalse (by intro h; cases h)
                  | inr i =>
                      cases i with
                      | inl _ =>
                          exact isFalse (by intro h; cases h)
                      | inr i =>
                          cases i with
                          | inl _ =>
                              exact isFalse (by intro h; cases h)
                          | inr o =>
                              rcases o with ⟨b', _q, h', d'⟩
                              exact (inferInstance : Decidable (b = b' ∧ h = h' ∧ d = d'))
      | inr j =>
          cases j with
          | inl s =>
              rcases s with ⟨b, h, q, _k⟩
              cases i with
              | inl _ =>
                  exact isFalse (by intro h; cases h)
              | inr i =>
                  cases i with
                  | inl _ =>
                      exact isFalse (by intro h; cases h)
                  | inr i =>
                      cases i with
                      | inl w =>
                          rcases w with ⟨b', h', q', _k'⟩
                          exact (inferInstance : Decidable (b = b' ∧ h = h' ∧ q = q'))
                      | inr _ =>
                          exact isFalse (by intro h; cases h)
          | inr j =>
              cases j with
              | inl w =>
                  rcases w with ⟨b, h, q, _k⟩
                  cases i with
                  | inl _ =>
                      exact isFalse (by intro h; cases h)
                  | inr i =>
                      cases i with
                      | inl _ =>
                          exact isFalse (by intro h; cases h)
                      | inr i =>
                          cases i with
                          | inl _ =>
                              exact isFalse (by intro h; cases h)
                          | inr o =>
                              rcases o with ⟨b', q', h', _d⟩
                              exact (inferInstance : Decidable (b = b' ∧ h = h' ∧ q = q'))
              | inr _ =>
                  exact isFalse (by intro h; cases h)
    wf := by
      have hsub :
          Subrelation (attentionAdj (Batch := Batch) (seq := seq) (heads := heads) (dim := dim))
            (fun j i =>
              attentionRank (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) j <
                attentionRank (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) i) := by
        intro j i h
        cases j with
        | inl j =>
            cases j with
            | inl _ =>
                cases i with
                | inl _ =>
                    cases h
                | inr i =>
                    cases i with
                    | inl _ =>
                        simp [attentionRank]
                    | inr i =>
                        cases i with
                        | inl _ =>
                            cases h
                        | inr _ =>
                            cases h
            | inr j =>
                cases j with
                | inl _ =>
                    cases i with
                    | inl _ =>
                        cases h
                    | inr i =>
                        cases i with
                        | inl _ =>
                            simp [attentionRank]
                        | inr i =>
                            cases i with
                            | inl _ =>
                                cases h
                            | inr _ =>
                                cases h
                | inr _ =>
                    cases i with
                    | inl _ =>
                        cases h
                    | inr i =>
                        cases i with
                        | inl _ =>
                            cases h
                        | inr i =>
                            cases i with
                            | inl _ =>
                                cases h
                            | inr _ =>
                                simp [attentionRank]
        | inr j =>
            cases j with
            | inl _ =>
                cases i with
                | inl _ =>
                    cases h
                | inr i =>
                    cases i with
                    | inl _ =>
                        cases h
                    | inr i =>
                        cases i with
                        | inl _ =>
                            simp [attentionRank]
                        | inr _ =>
                            cases h
            | inr j =>
                cases j with
                | inl _ =>
                    cases i with
                    | inl _ =>
                        cases h
                    | inr i =>
                        cases i with
                        | inl _ =>
                            cases h
                        | inr i =>
                            cases i with
                            | inl _ =>
                                cases h
                            | inr _ =>
                                simp [attentionRank]
                | inr _ =>
                    cases h
      have hwf : WellFounded (fun j i =>
          attentionRank (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) j <
            attentionRank (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) i) := by
        simpa using (InvImage.wf
          (f := attentionRank (Batch := Batch) (seq := seq) (heads := heads) (dim := dim))
          (h := Nat.lt_wfRel.wf))
      exact Subrelation.wf hsub hwf }

/-- Q nodes feed score nodes for matching batch/head/query. -/
theorem attentionDag_rel_q_score (b : Batch) (q : Fin seq) (h : Fin heads) (d : Fin dim)
    (k : Fin seq) :
    (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel
      (attnQ (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d))
      (attnScore (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, h, q, k)) := by
  simp [Dag.rel, attentionDag, attentionAdj]

/-- K nodes feed score nodes for matching batch/head/key. -/
theorem attentionDag_rel_k_score (b : Batch) (k : Fin seq) (h : Fin heads) (d : Fin dim)
    (q : Fin seq) :
    (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel
      (attnK (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, k, h, d))
      (attnScore (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, h, q, k)) := by
  simp [Dag.rel, attentionDag, attentionAdj]

/-- Score nodes feed weight nodes for matching batch/head/query. -/
theorem attentionDag_rel_score_weight (b : Batch) (h : Fin heads) (q : Fin seq) (k k' : Fin seq) :
    (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel
      (attnScore (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, h, q, k'))
      (attnWeight (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, h, q, k)) := by
  simp [Dag.rel, attentionDag, attentionAdj]

/-- Weight nodes feed output nodes for matching batch/head/query. -/
theorem attentionDag_rel_weight_out (b : Batch) (h : Fin heads) (q : Fin seq)
    (k : Fin seq) (d : Fin dim) :
    (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel
      (attnWeight (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, h, q, k))
      (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) := by
  simp [Dag.rel, attentionDag, attentionAdj]

/-- V nodes feed output nodes for matching batch/head/dimension. -/
theorem attentionDag_rel_v_out (b : Batch) (k : Fin seq) (h : Fin heads) (d : Fin dim)
    (q : Fin seq) :
    (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel
      (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, k, h, d))
      (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) := by
  simp [Dag.rel, attentionDag, attentionAdj]

end Dag

section Inputs

/-- Input nodes for the attention core. -/
def attentionInputs : Finset (AttentionNode Batch seq heads dim) :=
  (Finset.univ : Finset (AttentionInput Batch seq heads dim)).map Embedding.inl

/-- Definitional characterization of `attentionInputs`. -/
theorem attentionInputs_def :
    attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) =
      (Finset.univ : Finset (AttentionInput Batch seq heads dim)).map Embedding.inl := by
  rfl

open scoped Classical in
/-- Membership in attention inputs corresponds to being a left injection. -/
theorem mem_attentionInputs_iff {s : AttentionNode Batch seq heads dim} :
    s ∈ attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) ↔
      ∃ a, s = Sum.inl a := by
  constructor
  · intro hs
    rcases (Finset.mem_map.1 hs) with ⟨a, _ha, hsa⟩
    exact ⟨a, hsa.symm⟩
  · rintro ⟨a, rfl⟩
    refine Finset.mem_map.2 ?_
    exact ⟨a, by simp, rfl⟩

open scoped Classical in
/-- Right injections are not attention input nodes. -/
theorem not_mem_attentionInputs_inr (s : Sum (ScoreIndex Batch seq heads)
    (Sum (WeightIndex Batch seq heads) (AttentionOutput Batch seq heads dim))) :
    Sum.inr s ∉ attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) := by
  intro h
  rcases (mem_attentionInputs_iff (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).1 h
    with ⟨a, ha⟩
  cases ha

open scoped Classical in
/-- Input labels correspond to input nodes in the attention core. -/
def attentionInputEquiv :
    AttentionInput Batch seq heads dim ≃
      { i // i ∈ attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) } :=
  { toFun := fun a =>
      ⟨Sum.inl a,
        (mem_attentionInputs_iff (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).2
          ⟨a, rfl⟩⟩
    invFun := fun i =>
      match i with
      | ⟨Sum.inl a, _⟩ => a
      | ⟨Sum.inr s, h⟩ =>
          False.elim
            (not_mem_attentionInputs_inr (Batch := Batch) (seq := seq) (heads := heads)
              (dim := dim) s h)
    left_inv := by
      intro a
      rfl
    right_inv := by
      intro i
      cases i with
      | mk s hs =>
          cases s with
          | inl a => rfl
          | inr s =>
              cases (not_mem_attentionInputs_inr (Batch := Batch) (seq := seq) (heads := heads)
                (dim := dim) s hs) }

/-- Definitional characterization of `attentionInputEquiv`. -/
theorem attentionInputEquiv_def :
    attentionInputEquiv (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) =
      { toFun := fun a =>
          ⟨Sum.inl a,
            (mem_attentionInputs_iff (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).2
              ⟨a, rfl⟩⟩
        invFun := fun i =>
          match i with
          | ⟨Sum.inl a, _⟩ => a
          | ⟨Sum.inr s, h⟩ =>
              False.elim
                (not_mem_attentionInputs_inr (Batch := Batch) (seq := seq) (heads := heads)
                  (dim := dim) s h)
        left_inv := by
          intro a
          rfl
        right_inv := by
          intro i
          cases i with
          | mk s hs =>
              cases s with
              | inl a => rfl
              | inr s =>
                  cases (not_mem_attentionInputs_inr (Batch := Batch) (seq := seq) (heads := heads)
                    (dim := dim) s hs) } := by
  rfl

end Inputs

section Outputs

/-- Output nodes for the attention core. -/
def attentionOutputs : Finset (AttentionNode Batch seq heads dim) :=
  (Finset.univ : Finset (AttentionOutput Batch seq heads dim)).map
    { toFun := attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
      inj' := by
        intro a b h
        cases h
        rfl }

/-- Definitional characterization of `attentionOutputs`. -/
theorem attentionOutputs_def :
    attentionOutputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) =
      (Finset.univ : Finset (AttentionOutput Batch seq heads dim)).map
        { toFun := attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          inj' := by
            intro a b h
            cases h
            rfl } := by
  rfl

open scoped Classical in
/-- Membership in attention outputs corresponds to being an output injection. -/
theorem mem_attentionOutputs_iff {s : AttentionNode Batch seq heads dim} :
    s ∈ attentionOutputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) ↔
      ∃ o, s = attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) o := by
  constructor
  · intro hs
    rcases (Finset.mem_map.1 hs) with ⟨o, _ho, hso⟩
    exact ⟨o, hso.symm⟩
  · rintro ⟨o, rfl⟩
    refine Finset.mem_map.2 ?_
    exact ⟨o, by simp, rfl⟩

open scoped Classical in
/-- Left injections are not attention output nodes. -/
theorem not_mem_attentionOutputs_inl (s : AttentionInputNode Batch seq heads dim) :
    Sum.inl s ∉ attentionOutputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) := by
  intro h
  rcases (Finset.mem_map.1 h) with ⟨o, _ho, ho⟩
  cases ho

open scoped Classical in
/-- Score nodes are not attention output nodes. -/
theorem not_mem_attentionOutputs_score (s : ScoreIndex Batch seq heads) :
    Sum.inr (Sum.inl s) ∉ attentionOutputs (Batch := Batch) (seq := seq) (heads := heads)
      (dim := dim) := by
  intro h
  rcases (Finset.mem_map.1 h) with ⟨o, _ho, ho⟩
  cases ho

open scoped Classical in
/-- Weight nodes are not attention output nodes. -/
theorem not_mem_attentionOutputs_weight (w : WeightIndex Batch seq heads) :
    Sum.inr (Sum.inr (Sum.inl w)) ∉
      attentionOutputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) := by
  intro h
  rcases (Finset.mem_map.1 h) with ⟨o, _ho, ho⟩
  cases ho

open scoped Classical in
/-- Output labels correspond to output nodes in the attention core. -/
def attentionOutputEquiv :
    AttentionOutput Batch seq heads dim ≃
      { i // i ∈ attentionOutputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) } :=
  { toFun := fun o =>
      ⟨attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) o,
        (mem_attentionOutputs_iff (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).2
          ⟨o, rfl⟩⟩
    invFun := fun i =>
      match i with
      | ⟨Sum.inr (Sum.inr (Sum.inr o)), _⟩ => o
      | ⟨Sum.inl s, h⟩ =>
          False.elim
            (not_mem_attentionOutputs_inl (Batch := Batch) (seq := seq) (heads := heads)
              (dim := dim) s h)
      | ⟨Sum.inr (Sum.inl s), h⟩ =>
          False.elim
            (not_mem_attentionOutputs_score (Batch := Batch) (seq := seq) (heads := heads)
              (dim := dim) s h)
      | ⟨Sum.inr (Sum.inr (Sum.inl w)), h⟩ =>
          False.elim
            (not_mem_attentionOutputs_weight (Batch := Batch) (seq := seq) (heads := heads)
              (dim := dim) w h)
    left_inv := by
      intro o
      rfl
    right_inv := by
      intro i
      cases i with
      | mk s hs =>
          cases s with
          | inl s =>
              cases (not_mem_attentionOutputs_inl (Batch := Batch) (seq := seq) (heads := heads)
                (dim := dim) s hs)
          | inr s =>
              cases s with
              | inl s =>
                  cases (not_mem_attentionOutputs_score (Batch := Batch) (seq := seq)
                    (heads := heads) (dim := dim) s hs)
              | inr s =>
                  cases s with
                  | inl w =>
                      cases (not_mem_attentionOutputs_weight (Batch := Batch) (seq := seq)
                        (heads := heads) (dim := dim) w hs)
                  | inr _ =>
                      rfl }

/-- Definitional characterization of `attentionOutputEquiv`. -/
theorem attentionOutputEquiv_def :
    attentionOutputEquiv (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) =
      { toFun := fun o =>
          ⟨attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) o,
            (mem_attentionOutputs_iff (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).2
              ⟨o, rfl⟩⟩
        invFun := fun i =>
          match i with
          | ⟨Sum.inr (Sum.inr (Sum.inr o)), _⟩ => o
          | ⟨Sum.inl s, h⟩ =>
              False.elim
                (not_mem_attentionOutputs_inl (Batch := Batch) (seq := seq) (heads := heads)
                  (dim := dim) s h)
          | ⟨Sum.inr (Sum.inl s), h⟩ =>
              False.elim
                (not_mem_attentionOutputs_score (Batch := Batch) (seq := seq) (heads := heads)
                  (dim := dim) s h)
          | ⟨Sum.inr (Sum.inr (Sum.inl w)), h⟩ =>
              False.elim
                (not_mem_attentionOutputs_weight (Batch := Batch) (seq := seq) (heads := heads)
                  (dim := dim) w h)
        left_inv := by
          intro o
          rfl
        right_inv := by
          intro i
          cases i with
          | mk s hs =>
              cases s with
              | inl s =>
                  cases (not_mem_attentionOutputs_inl (Batch := Batch) (seq := seq) (heads := heads)
                    (dim := dim) s hs)
              | inr s =>
                  cases s with
                  | inl s =>
                      cases (not_mem_attentionOutputs_score (Batch := Batch) (seq := seq)
                        (heads := heads) (dim := dim) s hs)
                  | inr s =>
                      cases s with
                      | inl w =>
                          cases (not_mem_attentionOutputs_weight (Batch := Batch) (seq := seq)
                            (heads := heads) (dim := dim) w hs)
                      | inr _ =>
                          rfl } := by
  rfl

end Outputs

section Circuits

variable [DecidableEq Batch]

/-- Gate semantics for attention score/mixing circuits. -/
def attentionGate (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val) :
    ∀ i,
      (∀ j,
          (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel j i →
            Val) →
        Val := by
  intro i rec
  cases i with
  | inl _ =>
      exact 0
  | inr i =>
      cases i with
      | inl s =>
          rcases s with ⟨b, h, q, k⟩
          let qNode : Fin dim → AttentionNode Batch seq heads dim := fun d =>
            attnQ (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
              (b, q, h, d)
          let kNode : Fin dim → AttentionNode Batch seq heads dim := fun d =>
            attnK (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
              (b, k, h, d)
          let qVals : Fin dim → Val := fun d =>
            rec (qNode d)
              (attentionDag_rel_q_score (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                b q h d k)
          let kVals : Fin dim → Val := fun d =>
            rec (kNode d)
              (attentionDag_rel_k_score (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                b k h d q)
          exact scale * dotProduct qVals kVals
      | inr i =>
          cases i with
          | inl w =>
              rcases w with ⟨b, h, q, k⟩
              let scoreNode : Fin seq → AttentionNode Batch seq heads dim := fun k' =>
                attnScore (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                  (b, h, q, k')
              let scores : Fin seq → Val := fun k' =>
                rec (scoreNode k')
                  (attentionDag_rel_score_weight (Batch := Batch) (seq := seq) (heads := heads)
                    (dim := dim) b h q k k')
              exact softmax scores k
          | inr o =>
              rcases o with ⟨b, q, h, d⟩
              let weightNode : Fin seq → AttentionNode Batch seq heads dim := fun k =>
                attnWeight (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                  (b, h, q, k)
              let valueNode : Fin seq → AttentionNode Batch seq heads dim := fun k =>
                attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                  (b, k, h, d)
              let weights : Fin seq → Val := fun k =>
                rec (weightNode k)
                  (attentionDag_rel_weight_out (Batch := Batch) (seq := seq) (heads := heads)
                    (dim := dim) b h q k d)
              let vals : Fin seq → Val := fun k =>
                rec (valueNode k)
                  (attentionDag_rel_v_out (Batch := Batch) (seq := seq) (heads := heads)
                    (dim := dim) b k h d q)
              exact dotProduct weights vals

/-- Definitional characterization of `attentionGate` on output nodes. -/
theorem attentionGate_out_def (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val)
    (b : Batch) (q : Fin seq) (h : Fin heads) (d : Fin dim)
    (rec :
      ∀ j,
        (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel j
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) →
          Val) :
    attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
        (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) rec =
      let weightNode : Fin seq → AttentionNode Batch seq heads dim := fun k =>
        attnWeight (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, h, q, k)
      let valueNode : Fin seq → AttentionNode Batch seq heads dim := fun k =>
        attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, k, h, d)
      let weights : Fin seq → Val := fun k =>
        rec (weightNode k)
          (attentionDag_rel_weight_out (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
            b h q k d)
      let vals : Fin seq → Val := fun k =>
        rec (valueNode k)
          (attentionDag_rel_v_out (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
            b k h d q)
      dotProduct weights vals := by
  simp [attentionGate]

/-- Circuit for attention score/mixing. -/
def attentionCircuit (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val) :
    Circuit (AttentionNode Batch seq heads dim) Val :=
  { dag := attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
    inputs := attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
    outputs := attentionOutputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
    gate :=
      attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax }

/-- Definitional characterization of `attentionCircuit`. -/
theorem attentionCircuit_def (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val) :
    attentionCircuit (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax =
      { dag := attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        inputs := attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        outputs := attentionOutputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        gate := attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale
          softmax } := by
  rfl

/-- Typed interface for attention score/mixing circuits. -/
def attentionInterface (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val) :
    Interface
      (attentionCircuit (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax)
      (AttentionInput Batch seq heads dim) (AttentionOutput Batch seq heads dim) :=
  { inputs := attentionInputEquiv (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
    outputs := attentionOutputEquiv (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) }

end Circuits

section Typed

variable [DecidableEq Batch]

/-- Typed attention score/mixing circuit. -/
def attentionTyped (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val) :
    TypedCircuit (AttentionNode Batch seq heads dim) Val
      (AttentionInput Batch seq heads dim) (AttentionOutput Batch seq heads dim) :=
  { circuit := attentionCircuit (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
      scale softmax
    interface := attentionInterface (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
      scale softmax }

/-- Definitional characterization of `attentionTyped`. -/
theorem attentionTyped_def (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val) :
    attentionTyped (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax =
      { circuit := attentionCircuit (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          scale softmax
        interface := attentionInterface (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          scale softmax } := by
  rfl

end Typed

end Layers

end Circuit

end Nfp
