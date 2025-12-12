import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Algebra.Order.Monoid.Defs
import Aesop
import Nfp.Prob
import Nfp.Reroute.Heat

/-!
# PCC Helpers (Appendix A.4)

This module provides small probability/contribution utilities used in the
formalization of Appendix A.4 of the accompanying documentation:

* `tracerOfContrib` – builds a probability vector from nonnegative contributions.
* `sum_monotone_chain` / `monotone_removed_mass` – monotonicity of accumulated mass
  along nested mask chains (with tiny nonnegativity side-conditions handled by `aesop`).

All proofs are elementary (`simp`, small `aesop` calls on nonnegativity), and avoid `sorry`.
-/

namespace Nfp

open scoped BigOperators
open Finset

variable {S : Type*}

/-- Nonnegative contributions over `S`. -/
abbrev Contrib (S : Type*) [Fintype S] := S → NNReal

/-- Lemma A.4: The tracer distribution equals normalized absolute contributions. -/
noncomputable def tracerOfContrib [Fintype S] (m : Contrib S) (h : (∑ i, m i) ≠ 0) : ProbVec S :=
  {
    mass := fun i => m i / (∑ j, m j),
    norm_one := by
      classical
      have hsum : (∑ j, m j) ≠ 0 := h
      have : (∑ i, m i / (∑ j, m j)) = (∑ i, m i) / (∑ j, m j) := by
        simp [Finset.sum_div]
      simp [this, div_self hsum]
  }

@[simp] lemma tracerOfContrib_mass [Fintype S] (m : Contrib S) (h : (∑ i, m i) ≠ 0) :
  (tracerOfContrib (S:=S) m h).mass = fun i => m i / (∑ j, m j) := rfl

-- PCC monotonicity surrogate: if masks grow, removed normalized mass grows.
/-- Appendix A.4 (monotonicity helper): if masks grow, the removed mass sum grows. -/
lemma sum_monotone_chain [Fintype S] (A : ℕ → Finset S) (w : S → NNReal)
    (hchain : ∀ k, A k ⊆ A (k + 1)) : Monotone (fun k => (A k).sum (fun i => w i)) := by
  classical
  intro k₁ k₂ hk
  refine Nat.le_induction ?base ?step _ hk
  · exact le_rfl
  · intro k₂ _ ih
    have hstep : (A k₂).sum (fun i => w i) ≤ (A (k₂+1)).sum (fun i => w i) := by
      refine Finset.sum_le_sum_of_subset_of_nonneg (hchain k₂) ?_
      intro i hi _
      aesop
    exact ih.trans hstep

/-- Appendix A.4 (monotonicity helper): removed mass is monotone along a nested mask chain. -/
lemma monotone_removed_mass [Fintype S] (A : ℕ → Finset S) (m : Contrib S)
    (hchain : ∀ k, A k ⊆ A (k + 1)) : Monotone (fun k => (A k).sum m) := by
  simpa using (sum_monotone_chain (A:=A) (w:=m) hchain)

namespace PCC

/-- Finite interval `[lower, upper]` used to accumulate PCC area.  We only require
`lower ≤ upper` so that the width is realizable in `NNReal`. -/
structure AInterval where
  lower : NNReal
  upper : NNReal
  hle : lower ≤ upper

namespace AInterval

@[simp] lemma width_nonneg (I : AInterval) : 0 ≤ I.upper - I.lower :=
  (I.upper - I.lower).property

@[simp] lemma coe_width (I : AInterval) : (I.upper - I.lower : NNReal) = I.upper - I.lower := rfl

end AInterval

noncomputable def intervalsFromWeightsAux : NNReal → NNReal → List NNReal → List AInterval
  | _, _, [] => []
  | total, acc, w :: ws =>
      let width := w / total
      have hle : acc ≤ acc + width :=
        le_add_of_nonneg_right (show 0 ≤ width from bot_le)
      { lower := acc, upper := acc + width, hle := hle } ::
        intervalsFromWeightsAux total (acc + width) ws

/-- Build the discrete PCC intervals from a list of widths and a total scale. -/
noncomputable def intervalsFromWeights (total : NNReal) (weights : List NNReal) : List AInterval :=
  intervalsFromWeightsAux total 0 weights

variable (f : NNReal → NNReal)

/-- Evaluate the discrete AUC directly from widths (without constructing intervals). -/
noncomputable def evalFromWeightsAux :
    NNReal → NNReal → List NNReal → NNReal
  | _, _, [] => 0
  | total, acc, w :: ws =>
      let width := w / total
      width * f acc + evalFromWeightsAux total (acc + width) ws

noncomputable def evalFromWeights (total : NNReal) (weights : List NNReal) : NNReal :=
  evalFromWeightsAux (f:=f) total 0 weights

/-- Discrete AUC over a finite list of intervals for an arbitrary nonnegative
function `f`.  Each interval contributes `(upper - lower) * f lower`. -/
def AUC (L : List AInterval) : NNReal :=
  L.foldr (fun I acc => (I.upper - I.lower) * f I.lower + acc) 0

@[simp] lemma AUC_nil : AUC (f:=f) [] = 0 := rfl

@[simp] lemma AUC_cons (I : AInterval) (L : List AInterval) :
    AUC (f:=f) (I :: L) = (I.upper - I.lower) * f I.lower + AUC (f:=f) L := rfl

lemma AUC_append (L₁ L₂ : List AInterval) :
    AUC (f:=f) (L₁ ++ L₂) = AUC (f:=f) L₁ + AUC (f:=f) L₂ := by
  induction L₁ with
  | nil => simp [AUC]
  | cons I L₁ ih =>
      calc
        AUC (f:=f) ((I :: L₁) ++ L₂)
            = (I.upper - I.lower) * f I.lower + AUC (f:=f) (L₁ ++ L₂) := by
                simp [List.cons_append]
        _ = (I.upper - I.lower) * f I.lower + (AUC (f:=f) L₁ + AUC (f:=f) L₂) := by
                simp [ih]
        _ = ((I.upper - I.lower) * f I.lower + AUC (f:=f) L₁) + AUC (f:=f) L₂ := by
                ac_rfl
        _ = AUC (f:=f) (I :: L₁) + AUC (f:=f) L₂ := by
                simp

lemma AUC_nonneg (L : List AInterval) : 0 ≤ AUC (f:=f) L := by
  induction L with
  | nil => simp
  | cons I L ih =>
      have hterm : 0 ≤ (I.upper - I.lower) * f I.lower := by
        exact mul_nonneg I.width_nonneg (show 0 ≤ f I.lower from bot_le)
      have hsum : 0 ≤ (I.upper - I.lower) * f I.lower + AUC (f:=f) L :=
        add_nonneg hterm ih
      calc
        0 ≤ (I.upper - I.lower) * f I.lower + AUC (f:=f) L := hsum
        _ = AUC (f:=f) (I :: L) := (AUC_cons (f:=f) I L).symm

lemma AUC_monotone_append (L₁ L₂ : List AInterval) :
    AUC (f:=f) L₁ ≤ AUC (f:=f) (L₁ ++ L₂) := by
  have hnonneg := AUC_nonneg (f:=f) L₂
  have hle : AUC (f:=f) L₁ ≤ AUC (f:=f) L₁ + AUC (f:=f) L₂ := by
    exact le_add_of_nonneg_right hnonneg
  calc
    AUC (f:=f) L₁
        ≤ AUC (f:=f) L₁ + AUC (f:=f) L₂ := hle
    _ = AUC (f:=f) (L₁ ++ L₂) := (AUC_append (f:=f) L₁ L₂).symm

lemma AUC_add (L₁ L₂ : List AInterval) :
    AUC (f:=f) (L₁ ++ L₂) = AUC (f:=f) L₁ + AUC (f:=f) L₂ :=
  AUC_append (f:=f) L₁ L₂

lemma AUC_intervalsFromWeightsAux (total acc : NNReal) (weights : List NNReal) :
    AUC (f:=f) (intervalsFromWeightsAux total acc weights)
      = evalFromWeightsAux (f:=f) total acc weights := by
  induction weights generalizing acc with
  | nil =>
      simp [intervalsFromWeightsAux, evalFromWeightsAux, AUC]
  | cons w ws ih =>
      have htail := ih (acc + w / total)
      simp [intervalsFromWeightsAux, evalFromWeightsAux, htail]

lemma AUC_intervalsFromWeights (total : NNReal) (weights : List NNReal) :
    AUC (f:=f) (intervalsFromWeights total weights)
      = evalFromWeights (f:=f) total weights := by
  simpa [intervalsFromWeights, evalFromWeights] using
    AUC_intervalsFromWeightsAux (f:=f) total 0 weights

lemma intervalsFromWeightsAux_take (total acc : NNReal) (weights : List NNReal) :
    ∀ k,
      intervalsFromWeightsAux total acc (weights.take k)
        = (intervalsFromWeightsAux total acc weights).take k
  | 0 => by simp [intervalsFromWeightsAux]
  | Nat.succ k =>
      by
        cases weights with
        | nil => simp [intervalsFromWeightsAux]
        | cons w ws =>
            have ih :=
              intervalsFromWeightsAux_take (total:=total) (acc:=acc + w / total) (weights:=ws) k
            simp [intervalsFromWeightsAux, ih, Nat.succ_eq_add_one]

lemma AUC_intervalsFromWeights_take (total : NNReal) (weights : List NNReal) (k : ℕ) :
    AUC (f:=f) ((intervalsFromWeights total weights).take k)
      = evalFromWeights (f:=f) total (weights.take k) := by
  have h :=
    AUC_intervalsFromWeightsAux (f:=f) total 0 (weights.take k)
  simpa [intervalsFromWeights, evalFromWeights,
    intervalsFromWeightsAux_take (total:=total) (acc:=0) (weights:=weights)]
      using h

variable {f}

end PCC

namespace WeightedReroutePlan

open PCC

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- Normalized PCC intervals for a weighted reroute plan (widths sum to one). -/
noncomputable def aucIntervals (P : WeightedReroutePlan (S := S)) : List PCC.AInterval :=
  intervalsFromWeights (total:=P.weightsSum) P.weights

lemma auc_eval (P : WeightedReroutePlan (S := S)) (f : NNReal → NNReal) :
    PCC.AUC (f:=f) (P.aucIntervals)
      = PCC.evalFromWeights (f:=f) P.weightsSum P.weights := by
  simpa [aucIntervals]
    using PCC.AUC_intervalsFromWeights (f:=f) P.weightsSum P.weights

end WeightedReroutePlan

end Nfp
