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

section Attention

variable {Batch : Type} [Fintype Batch] [DecidableEq Batch]
variable {seq heads dim : Nat}
variable {Val : Type v} [NonAssocSemiring Val]

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

end Attention

end Layers

end Circuit

end Nfp
