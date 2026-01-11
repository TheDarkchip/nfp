-- SPDX-License-Identifier: AGPL-3.0-or-later
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Nfp.Sound.Induction.CoreDefs
import Nfp.Sound.Linear.FinFold

/-!
Helper lemmas for value-direction bounds in induction-head soundness.

These isolate the algebra needed to rewrite direction-value projections into
dot products over cached `wvDir`/`bDir` terms.
-/

namespace Nfp

namespace Sound

open scoped BigOperators

open Nfp.Sound.Linear

variable {seq dModel dHead : Nat}

/-- Cast a cached `wvDir` dot to a Real-valued sum over head weights. -/
theorem wvDir_real_eq_sum (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dirHead : Fin dHead → Rat) (wvDir : Fin dModel → Rat)
    (hwvDir : ∀ j, wvDir j = Linear.dotFin dHead dirHead (fun d => inputs.wv j d)) :
    ∀ j, (wvDir j : Real) = ∑ d, (dirHead d : Real) * (inputs.wv j d : Real) := by
  intro j
  calc
    (wvDir j : Real)
        = ((∑ d, dirHead d * inputs.wv j d : Rat) : Real) := by
          simp [hwvDir j, Linear.dotFin_eq_dotProduct, dotProduct]
    _ = ∑ d, (dirHead d : Real) * (inputs.wv j d : Real) := by
          simp

/-- Cast a cached `bDir` dot to a Real-valued sum over head biases. -/
theorem bDir_real_eq_sum (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dirHead : Fin dHead → Rat) (bDir : Rat)
    (hbDir : bDir = Linear.dotFin dHead dirHead (fun d => inputs.bv d)) :
    (bDir : Real) = ∑ d, (dirHead d : Real) * (inputs.bv d : Real) := by
  calc
    (bDir : Real)
        = ((∑ d, dirHead d * inputs.bv d : Rat) : Real) := by
          simp [hbDir, Linear.dotFin_eq_dotProduct, dotProduct]
    _ = ∑ d, (dirHead d : Real) * (inputs.bv d : Real) := by
          simp

/-- Rewrite direction values using cached `wvDir` and `bDir` sums. -/
theorem valsReal_eq_of_dir (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dirHead : Fin dHead → Rat)
    (hdirHead : dirHead = fun d => (dirHeadVecOfInputs inputs).get d)
    (wvDir : Fin dModel → Rat) (bDir : Rat)
    (hdir_wv : ∀ j, (wvDir j : Real) = ∑ d, (dirHead d : Real) * (inputs.wv j d : Real))
    (hdir_bv : (bDir : Real) = ∑ d, (dirHead d : Real) * (inputs.bv d : Real)) :
    ∀ k,
      valsRealOfInputs inputs k =
        dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) + (bDir : Real) := by
  intro k
  classical
  have hdirHead_real :
      (fun d => (dirHeadVecOfInputs inputs).get d : Fin dHead → Real) =
        fun d => (dirHead d : Real) := by
    funext d
    simp [hdirHead]
  have hdot_add :
      dotProduct (fun d => (dirHead d : Real))
          (fun d =>
            dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs k) +
              (inputs.bv d : Real)) =
        dotProduct (fun d => (dirHead d : Real))
            (fun d => dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs k)) +
          dotProduct (fun d => (dirHead d : Real)) (fun d => (inputs.bv d : Real)) := by
    simp [dotProduct, mul_add, Finset.sum_add_distrib]
  have hdot_wv :
      dotProduct (fun d => (dirHead d : Real))
          (fun d => dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs k)) =
        dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) := by
    calc
      dotProduct (fun d => (dirHead d : Real))
          (fun d => dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs k)) =
        ∑ d, (dirHead d : Real) * ∑ j,
          (inputs.wv j d : Real) * lnRealOfInputs inputs k j := by
          simp [dotProduct]
      _ = ∑ d, ∑ j,
            (dirHead d : Real) *
              ((inputs.wv j d : Real) * lnRealOfInputs inputs k j) := by
          simp [Finset.mul_sum]
      _ = ∑ j, ∑ d,
            (dirHead d : Real) *
              ((inputs.wv j d : Real) * lnRealOfInputs inputs k j) := by
          simpa using
            (Finset.sum_comm (s := (Finset.univ : Finset (Fin dHead)))
              (t := (Finset.univ : Finset (Fin dModel)))
              (f := fun d j =>
                (dirHead d : Real) * ((inputs.wv j d : Real) * lnRealOfInputs inputs k j)))
      _ = ∑ j, (∑ d, (dirHead d : Real) * (inputs.wv j d : Real)) *
            lnRealOfInputs inputs k j := by
          refine Finset.sum_congr rfl ?_
          intro j _
          have hsum :
              (∑ d, (dirHead d : Real) * (inputs.wv j d : Real)) *
                  lnRealOfInputs inputs k j =
                ∑ d, (dirHead d : Real) * (inputs.wv j d : Real) *
                  lnRealOfInputs inputs k j := by
            simp [Finset.sum_mul, mul_assoc]
          simpa [mul_assoc] using hsum.symm
      _ = ∑ j, (wvDir j : Real) * lnRealOfInputs inputs k j := by
          refine Finset.sum_congr rfl ?_
          intro j _
          simp [hdir_wv j]
      _ = dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) := by
          simp [dotProduct]
  calc
    valsRealOfInputs inputs k =
        dotProduct (fun d => (dirHead d : Real))
          (fun d =>
            dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs k) +
              (inputs.bv d : Real)) := by
      simp [valsRealOfInputs, vRealOfInputs, hdirHead_real]
    _ =
        dotProduct (fun d => (dirHead d : Real))
            (fun d => dotProduct (fun j => (inputs.wv j d : Real))
              (lnRealOfInputs inputs k)) +
          dotProduct (fun d => (dirHead d : Real)) (fun d => (inputs.bv d : Real)) := hdot_add
    _ =
        dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) +
          dotProduct (fun d => (dirHead d : Real)) (fun d => (inputs.bv d : Real)) := by
      simp [hdot_wv]
    _ =
        dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) + (bDir : Real) := by
      have hb :
          dotProduct (fun d => (dirHead d : Real)) (fun d => (inputs.bv d : Real)) =
            (bDir : Real) := by
        simpa [dotProduct] using hdir_bv.symm
      simp [hb]

end Sound
end Nfp
