-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Layers.Attention

/-!
Induction-head specifications for attention cores.
-/

namespace Nfp

namespace Circuit

namespace Layers

universe v

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

end Layers

end Circuit

end Nfp
