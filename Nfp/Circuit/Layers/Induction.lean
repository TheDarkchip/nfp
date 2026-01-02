-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Monoid.Unbundled.Basic
import Nfp.Circuit.Layers.Attention

/-!
Induction-head specifications for attention cores.
-/

namespace Nfp

namespace Circuit

namespace Layers

universe v

open scoped BigOperators

section Weights

variable {Val : Type v} [NonAssocSemiring Val]
variable {seq : Nat}

/-- Induction weights are one-hot at `prev` for each query position. -/
def InductionWeights (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop :=
  ∀ q, weights q = Pi.single (prev q) 1

/-- A one-hot weight vector selects the corresponding value in a dot product. -/
theorem dotProduct_eq_of_oneHot (k : Fin seq) (vals : Fin seq → Val) :
    dotProduct (Pi.single k 1) vals = vals k := by
  simp

/-- Induction weights select the `prev` value in each dot product. -/
theorem dotProduct_eq_prev (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) (vals : Fin seq → Fin seq → Val)
    (hweights : InductionWeights (Val := Val) prev weights) (q : Fin seq) :
    dotProduct (weights q) (vals q) = vals q (prev q) := by
  have hq : weights q = Pi.single (prev q) 1 := hweights q
  simp [hq]

end Weights

section Spec

variable {Val : Type v}
variable {n : Nat}

/-- Induction-head spec: for nonzero queries, outputs copy `prev` values. -/
def InductionSpec (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val) : Prop :=
  ∀ q, q ≠ 0 → out q = vals (prev q)

/-- Concrete `prev` map on `Fin (n + 1)` (with `0 ↦ 0`). -/
def prevIndex : Fin (Nat.succ n) → Fin (Nat.succ n)
  | ⟨0, _⟩ => 0
  | ⟨Nat.succ k, hk⟩ =>
      ⟨k, Nat.lt_trans (Nat.lt_of_succ_lt_succ hk) (Nat.lt_succ_self n)⟩

end Spec

section ApproxSpec

variable {Val : Type v} [AddCommMonoid Val] [PartialOrder Val] [IsOrderedAddMonoid Val]
variable {n : Nat}

/-- Approximate induction-head spec: outputs are within `ε` of `prev` values. -/
def InductionSpecApprox (ε : Val)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val) : Prop :=
  ∀ q, q ≠ 0 → out q ≤ vals (prev q) + ε ∧ vals (prev q) ≤ out q + ε

/-- Exact induction spec implies the approximate spec for any nonnegative tolerance. -/
theorem inductionSpecApprox_of_spec (ε : Val) (hε : 0 ≤ ε)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val)
    (h : InductionSpec prev out vals) :
    InductionSpecApprox (Val := Val) (n := n) ε prev out vals := by
  intro q hq
  have hq' : out q = vals (prev q) := h q hq
  constructor <;>
    simpa [hq'] using
      (le_add_of_nonneg_right hε :
        vals (prev q) ≤ vals (prev q) + ε)

end ApproxSpec

section Bounds

variable {Val : Type v} [Semiring Val] [PartialOrder Val]
variable {seq : Nat} [NeZero seq]

/-- Numeric bounds certifying one-hot weights on nonzero queries. -/
structure OneHotBoundsOn (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop where
  /-- All weights are nonnegative on nonzero queries. -/
  nonneg : ∀ q, q ≠ 0 → ∀ k, 0 ≤ weights q k
  /-- Weights sum to one on nonzero queries. -/
  sum_one : ∀ q, q ≠ 0 → (∑ k, weights q k) = 1
  /-- Non-prev weights are nonpositive on nonzero queries. -/
  other_le_zero : ∀ q, q ≠ 0 → ∀ k, k ≠ prev q → weights q k ≤ 0

/-- Certified bounds imply one-hot weights on nonzero queries. -/
theorem oneHot_of_boundsOn (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) [DecidableEq (Fin seq)]
    (h : OneHotBoundsOn prev weights) :
    ∀ q, q ≠ 0 → weights q = Pi.single (prev q) 1 := by
  intro q hq
  funext k
  by_cases hk : k = prev q
  · subst hk
    have hzero :
        (∑ k ∈ (Finset.univ.erase (prev q)), weights q k) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro k hk'
      have hkne : k ≠ prev q := (Finset.mem_erase.1 hk').1
      have hle : weights q k ≤ 0 := h.other_le_zero q hq k hkne
      have hge : 0 ≤ weights q k := h.nonneg q hq k
      exact le_antisymm hle hge
    have hsum :
        weights q (prev q) +
            ∑ k ∈ (Finset.univ.erase (prev q)), weights q k =
          ∑ k, weights q k := by
      simpa using
        (Finset.add_sum_erase
          (s := (Finset.univ : Finset (Fin seq)))
          (f := weights q) (a := prev q) (by simp))
    have hprev : weights q (prev q) = 1 := by
      have hsum' :
          weights q (prev q) + 0 = 1 := by
        simpa [hzero, h.sum_one q hq] using hsum
      simpa using hsum'
    simp [Pi.single, hprev]
  · have hle : weights q k ≤ 0 := h.other_le_zero q hq k hk
    have hge : 0 ≤ weights q k := h.nonneg q hq k
    have hzero : weights q k = 0 := le_antisymm hle hge
    simp [Pi.single, hk, hzero]

end Bounds

section ApproxBounds

variable {Val : Type v} [Semiring Val] [PartialOrder Val]
variable {seq : Nat} [NeZero seq]

/-- Approximate one-hot bounds for attention weights on nonzero queries. -/
structure OneHotApproxBoundsOn (ε : Val) (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop where
  /-- All weights are nonnegative on nonzero queries. -/
  nonneg : ∀ q, q ≠ 0 → ∀ k, 0 ≤ weights q k
  /-- Weights sum to one on nonzero queries. -/
  sum_one : ∀ q, q ≠ 0 → (∑ k, weights q k) = 1
  /-- The `prev` weight is within `ε` of one on nonzero queries. -/
  prev_large : ∀ q, q ≠ 0 → 1 ≤ weights q (prev q) + ε
  /-- Non-prev weights are at most `ε` on nonzero queries. -/
  other_le : ∀ q, q ≠ 0 → ∀ k, k ≠ prev q → weights q k ≤ ε

/-- Approximate induction weights: prev weight near one, others at most `ε`. -/
def InductionWeightsApprox (ε : Val) (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop :=
  ∀ q, q ≠ 0 →
    1 ≤ weights q (prev q) + ε ∧
      ∀ k, k ≠ prev q → weights q k ≤ ε

/-- Approximate bounds imply approximate induction weights. -/
theorem inductionWeightsApprox_of_boundsOn (ε : Val) (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val)
    (h : OneHotApproxBoundsOn ε prev weights) :
    InductionWeightsApprox (Val := Val) ε prev weights := by
  intro q hq
  exact ⟨h.prev_large q hq, h.other_le q hq⟩

end ApproxBounds

section SoftmaxMargin

variable {Val : Type v} [Semiring Val] [PartialOrder Val]
variable {seq : Nat} [NeZero seq]

/-- Softmax margin certificates for approximate one-hot weights. -/
structure SoftmaxMarginBounds (ε margin : Val) (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val) : Prop where
  /-- Score gap between `prev` and other keys on nonzero queries. -/
  score_margin : ∀ q, q ≠ 0 → ∀ k, k ≠ prev q → scores q k + margin ≤ scores q (prev q)
  /-- All weights are nonnegative on nonzero queries. -/
  nonneg : ∀ q, q ≠ 0 → ∀ k, 0 ≤ weights q k
  /-- Weights sum to one on nonzero queries. -/
  sum_one : ∀ q, q ≠ 0 → (∑ k, weights q k) = 1
  /-- The `prev` weight is within `ε` of one on nonzero queries. -/
  prev_large : ∀ q, q ≠ 0 → 1 ≤ weights q (prev q) + ε
  /-- Non-prev weights are at most `ε` on nonzero queries. -/
  other_le : ∀ q, q ≠ 0 → ∀ k, k ≠ prev q → weights q k ≤ ε

/-- Margin certificates yield approximate one-hot bounds for the weights. -/
theorem oneHotApproxBounds_of_softmaxMargin (ε margin : Val) (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val)
    (h : SoftmaxMarginBounds (Val := Val) ε margin prev scores weights) :
    OneHotApproxBoundsOn (Val := Val) ε prev weights := by
  exact
    { nonneg := h.nonneg
      sum_one := h.sum_one
      prev_large := h.prev_large
      other_le := h.other_le }

/-- Margin certificates imply approximate induction-weight bounds. -/
theorem inductionWeightsApprox_of_softmaxMargin (ε margin : Val)
    (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val)
    (h : SoftmaxMarginBounds (Val := Val) ε margin prev scores weights) :
    InductionWeightsApprox (Val := Val) ε prev weights := by
  exact inductionWeightsApprox_of_boundsOn
    (Val := Val)
    (seq := seq)
    (ε := ε)
    (prev := prev)
    (weights := weights)
    (h := oneHotApproxBounds_of_softmaxMargin
      (Val := Val)
      (seq := seq)
      (ε := ε)
      (margin := margin)
      (prev := prev)
      (scores := scores)
      (weights := weights)
      h)

end SoftmaxMargin

section Attention

variable {Batch : Type} [Fintype Batch] [DecidableEq Batch]
variable {seq heads dim : Nat}
variable {Val : Type v} [NonAssocSemiring Val]

/-- Typed V-input label for attention cores. -/
abbrev attnInputV (v : QkvIndex Batch seq heads dim) :
    AttentionInput Batch seq heads dim :=
  Sum.inr (Sum.inr v)

/-- Weight function feeding an attention output node. -/
def attentionOutWeights (b : Batch) (h : Fin heads) (q : Fin seq) (d : Fin dim)
    (rec :
      ∀ j,
        (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel j
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) →
          Val) :
    Fin seq → Val :=
  fun k =>
    rec (attnWeight (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, h, q, k))
      (attentionDag_rel_weight_out (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        b h q k d)

/-- Value function feeding an attention output node. -/
def attentionOutValues (b : Batch) (h : Fin heads) (q : Fin seq) (d : Fin dim)
    (rec :
      ∀ j,
        (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel j
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) →
          Val) :
    Fin seq → Val :=
  fun k =>
    rec (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, k, h, d))
      (attentionDag_rel_v_out (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        b k h d q)

/-- One-hot attention weights force the output to copy the selected value. -/
theorem attentionGate_out_eq_of_oneHot (scale : Val)
    (softmax : (Fin seq → Val) → Fin seq → Val) (prev : Fin seq → Fin seq)
    (b : Batch) (h : Fin heads) (q : Fin seq) (d : Fin dim)
    (rec :
      ∀ j,
        (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel j
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) →
          Val)
    (hweights :
      attentionOutWeights (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d rec =
        Pi.single (prev q) 1) :
    attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
        (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) rec =
      attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        b h q d rec (prev q) := by
  simp only [attentionGate]
  change
    dotProduct
        (attentionOutWeights (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d rec)
        (attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d rec) =
      attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        b h q d rec (prev q)
  rw [hweights]
  exact dotProduct_eq_of_oneHot (Val := Val) (seq := seq) (k := prev q)
    (vals := attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
      b h q d rec)

section Typed

variable (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val)

/-- Attention output equals the selected V input when weights are one-hot. -/
theorem attentionTyped_eval_out_eq_of_oneHot (prev : Fin seq → Fin seq)
    (input : AttentionInput Batch seq heads dim → Val)
    (b : Batch) (h : Fin heads) (q : Fin seq) (d : Fin dim)
    (hweights :
      attentionOutWeights (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d
          (fun j _ =>
            Circuit.evalInput
              (attentionCircuit (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                scale softmax)
              ((attentionInterface (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                scale softmax).toInputAssignment input) j) =
        Pi.single (prev q) 1) :
    (attentionTyped (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax).eval
        input (b, q, h, d) =
      input
        (attnInputV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          (b, prev q, h, d)) := by
  let C :=
    attentionCircuit (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
  let I :=
    attentionInterface (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
  let inputAssign := I.toInputAssignment input
  have hnot :
      attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d) ∉
        attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) := by
    simpa using
      (not_mem_attentionInputs_inr (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        (s := Sum.inr (Sum.inr (b, q, h, d))))
  have hgate :
      Circuit.evalInput C inputAssign
          (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) =
        attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
          (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d))
          (fun j _ => Circuit.evalInput C inputAssign j) := by
    exact Circuit.evalInput_eq_gate (C := C) (input := inputAssign)
      (i := attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d))
      hnot
  have hcopy :
      Circuit.evalInput C inputAssign
          (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) =
        attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d (fun j _ => Circuit.evalInput C inputAssign j) (prev q) := by
    have hgate' :
        attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d))
            (fun j _ => Circuit.evalInput C inputAssign j) =
          attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
            b h q d (fun j _ => Circuit.evalInput C inputAssign j) (prev q) :=
      attentionGate_out_eq_of_oneHot (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        (scale := scale) (softmax := softmax) (prev := prev) (b := b) (h := h) (q := q) (d := d)
        (rec := fun j _ => Circuit.evalInput C inputAssign j) hweights
    exact hgate.trans hgate'
  have hmem :
      attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, prev q, h, d) ∈
        attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) := by
    refine (mem_attentionInputs_iff (Batch := Batch) (seq := seq) (heads := heads)
      (dim := dim)).2 ?_
    exact ⟨attnInputV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
      (b, prev q, h, d), rfl⟩
  have hinput :
      Circuit.evalInput C inputAssign
          (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, prev q, h, d)) =
        input (attnInputV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          (b, prev q, h, d)) := by
    have h :=
      Circuit.evalInput_eq_input (C := C) (input := inputAssign)
        (i := attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          (b, prev q, h, d)) hmem
    simpa [inputAssign, I, attentionInterface, attnInputV] using h
  have hvals :
      attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d (fun j _ => Circuit.evalInput C inputAssign j) (prev q) =
        Circuit.evalInput C inputAssign
          (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
            (b, prev q, h, d)) := rfl
  calc
    (attentionTyped (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax).eval
        input (b, q, h, d) =
      Circuit.evalInput C inputAssign (I.outputs (b, q, h, d)).1 := by
        simp [TypedCircuit.eval, Interface.eval, C, I, inputAssign, attentionTyped]
    _ = Circuit.evalInput C inputAssign
        (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) := by
        rfl
    _ = attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d (fun j _ => Circuit.evalInput C inputAssign j) (prev q) := hcopy
    _ = Circuit.evalInput C inputAssign
          (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
            (b, prev q, h, d)) := hvals
    _ = input (attnInputV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          (b, prev q, h, d)) := hinput

end Typed

end Attention

section InductionSpecTyped

variable {Batch : Type} [Fintype Batch] [DecidableEq Batch]
variable {heads dim n : Nat}
variable {Val : Type v} [NonAssocSemiring Val]

variable (scale : Val)
variable (softmax : (Fin (Nat.succ n) → Val) → Fin (Nat.succ n) → Val)

/-- One-hot weights on nonzero queries imply the induction spec for typed evaluation. -/
theorem attentionTyped_eval_inductionSpec_of_oneHot
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (input : AttentionInput Batch (Nat.succ n) heads dim → Val)
    (b : Batch) (h : Fin heads) (d : Fin dim)
    (hweights :
      ∀ q, q ≠ 0 →
        attentionOutWeights
            (Batch := Batch)
            (seq := Nat.succ n)
            (heads := heads)
            (dim := dim)
            b h q d
            (fun j _ =>
              Circuit.evalInput
                (attentionCircuit
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax)
                ((attentionInterface
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax).toInputAssignment input) j) =
          Pi.single (prev q) 1) :
    InductionSpec (n := n) prev
      (fun q =>
        (attentionTyped
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          scale softmax).eval input (b, q, h, d))
      (fun k =>
        input (attnInputV
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          (b, k, h, d))) := by
  intro q hq
  have hweights_q := hweights q hq
  exact attentionTyped_eval_out_eq_of_oneHot
    (Batch := Batch)
    (seq := Nat.succ n)
    (heads := heads)
    (dim := dim)
    (scale := scale)
    (softmax := softmax)
    (prev := prev)
    (input := input)
    (b := b)
    (h := h)
    (q := q)
    (d := d)
    hweights_q

/-- Induction spec for `prevIndex` under one-hot weight hypotheses. -/
theorem attentionTyped_eval_inductionSpec_prevIndex
    (input : AttentionInput Batch (Nat.succ n) heads dim → Val)
    (b : Batch) (h : Fin heads) (d : Fin dim)
    (hweights :
      ∀ q, q ≠ 0 →
        attentionOutWeights
            (Batch := Batch)
            (seq := Nat.succ n)
            (heads := heads)
            (dim := dim)
            b h q d
            (fun j _ =>
              Circuit.evalInput
                (attentionCircuit
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax)
                ((attentionInterface
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax).toInputAssignment input) j) =
          Pi.single (prevIndex (n := n) q) 1) :
    InductionSpec (n := n) (prevIndex (n := n))
      (fun q =>
        (attentionTyped
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          scale softmax).eval input (b, q, h, d))
      (fun k =>
        input (attnInputV
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          (b, k, h, d))) := by
  exact attentionTyped_eval_inductionSpec_of_oneHot
    (Batch := Batch)
    (heads := heads)
    (dim := dim)
    (n := n)
    (scale := scale)
    (softmax := softmax)
    (prev := prevIndex (n := n))
    (input := input)
    (b := b)
    (h := h)
    (d := d)
    hweights

end InductionSpecTyped

section InductionSpecApproxTyped

variable {Batch : Type} [Fintype Batch] [DecidableEq Batch]
variable {heads dim n : Nat}
variable {Val : Type v} [NonAssocSemiring Val] [PartialOrder Val] [IsOrderedAddMonoid Val]

variable (scale : Val)
variable (softmax : (Fin (Nat.succ n) → Val) → Fin (Nat.succ n) → Val)

/-- One-hot weights imply the approximate induction spec for any nonnegative tolerance. -/
theorem attentionTyped_eval_inductionSpecApprox_of_oneHot (ε : Val) (hε : 0 ≤ ε)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (input : AttentionInput Batch (Nat.succ n) heads dim → Val)
    (b : Batch) (h : Fin heads) (d : Fin dim)
    (hweights :
      ∀ q, q ≠ 0 →
        attentionOutWeights
            (Batch := Batch)
            (seq := Nat.succ n)
            (heads := heads)
            (dim := dim)
            b h q d
            (fun j _ =>
              Circuit.evalInput
                (attentionCircuit
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax)
                ((attentionInterface
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax).toInputAssignment input) j) =
          Pi.single (prev q) 1) :
    InductionSpecApprox (Val := Val) (n := n) ε prev
      (fun q =>
        (attentionTyped
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          scale softmax).eval input (b, q, h, d))
      (fun k =>
        input (attnInputV
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          (b, k, h, d))) := by
  apply inductionSpecApprox_of_spec (Val := Val) (n := n) (ε := ε) hε
  exact attentionTyped_eval_inductionSpec_of_oneHot
    (Batch := Batch)
    (heads := heads)
    (dim := dim)
    (n := n)
    (scale := scale)
    (softmax := softmax)
    (prev := prev)
    (input := input)
    (b := b)
    (h := h)
    (d := d)
    hweights

end InductionSpecApproxTyped

end Layers

end Circuit

end Nfp
