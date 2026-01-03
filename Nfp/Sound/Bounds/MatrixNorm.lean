-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Matrix.Mul
import Nfp.Circuit.Cert.DownstreamLinear

/-!
Row-sum matrix norms for downstream linear certificates.

These bounds are used to compute verified downstream error certificates
from explicit Rat matrices.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

/-- Row-sum of absolute values for a matrix row. -/
def rowSum {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) (i : Fin m) : Rat :=
  ∑ j, |W i j|

/-- Maximum row-sum norm (defaults to `0` on empty matrices). -/
def rowSumNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) : Rat :=
  if h : (Finset.univ : Finset (Fin m)).Nonempty then
    (Finset.univ).sup' h (fun i => rowSum W i)
  else
    0

/-- Row-sums are nonnegative. -/
theorem rowSum_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) (i : Fin m) :
    0 ≤ rowSum W i := by
  refine Finset.sum_nonneg ?_
  intro j _
  exact abs_nonneg (W i j)

/-- Each row-sum is bounded by the row-sum norm. -/
theorem rowSum_le_rowSumNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) (i : Fin m) :
    rowSum W i ≤ rowSumNorm W := by
  classical
  have h : (Finset.univ : Finset (Fin m)).Nonempty := ⟨i, by simp⟩
  have hle :
      rowSum W i ≤ (Finset.univ).sup' h (fun i => rowSum W i) := by
    simpa using
      (Finset.le_sup'
        (s := (Finset.univ : Finset (Fin m)))
        (f := fun i => rowSum W i)
        (by simp : i ∈ (Finset.univ : Finset (Fin m))))
  simpa [rowSumNorm, h] using hle

/-- The row-sum norm is nonnegative. -/
theorem rowSumNorm_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) :
    0 ≤ rowSumNorm W := by
  classical
  by_cases h : (Finset.univ : Finset (Fin m)).Nonempty
  · rcases h with ⟨i, hi⟩
    have hrow : 0 ≤ rowSum W i := rowSum_nonneg W i
    have hle : rowSum W i ≤ rowSumNorm W := rowSum_le_rowSumNorm W i
    exact le_trans hrow hle
  · simp [rowSumNorm, h]

/-- Row-sum norm bounds a matrix-vector product under a uniform input bound. -/
theorem abs_mulVec_le_rowSumNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (x : Fin n → Rat) (inputBound : Rat)
    (hx : ∀ j, |x j| ≤ inputBound) (hinput : 0 ≤ inputBound) :
    ∀ i, |Matrix.mulVec W x i| ≤ rowSumNorm W * inputBound := by
  intro i
  have hrow : |Matrix.mulVec W x i| ≤ rowSum W i * inputBound := by
    have h1 : |∑ j, W i j * x j| ≤ ∑ j, |W i j * x j| := by
      simpa using
        (Finset.abs_sum_le_sum_abs
          (f := fun j => W i j * x j)
          (s := (Finset.univ : Finset (Fin n))))
    have h2 : ∑ j, |W i j * x j| ≤ ∑ j, |W i j| * inputBound := by
      refine Finset.sum_le_sum ?_
      intro j _
      have hxj := hx j
      have hnonneg : 0 ≤ |W i j| := abs_nonneg (W i j)
      calc
        |W i j * x j| = |W i j| * |x j| := by
          simp [abs_mul]
        _ ≤ |W i j| * inputBound := by
          exact mul_le_mul_of_nonneg_left hxj hnonneg
    have h3 : ∑ j, |W i j| * inputBound = rowSum W i * inputBound := by
      have hsum :
          (∑ j, |W i j|) * inputBound = ∑ j, |W i j| * inputBound := by
        simpa using
          (Finset.sum_mul
            (s := (Finset.univ : Finset (Fin n)))
            (f := fun j => |W i j|)
            (a := inputBound))
      simpa [rowSum] using hsum.symm
    have hmul : |Matrix.mulVec W x i| ≤ rowSum W i * inputBound := by
      simpa [Matrix.mulVec, dotProduct] using h1.trans (h2.trans_eq h3)
    exact hmul
  have hle : rowSum W i ≤ rowSumNorm W := rowSum_le_rowSumNorm W i
  have hmul : rowSum W i * inputBound ≤ rowSumNorm W * inputBound :=
    mul_le_mul_of_nonneg_right hle hinput
  exact hrow.trans hmul

/-- Build a downstream linear certificate from a matrix and input bound. -/
def buildDownstreamLinearCert {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (inputBound : Rat) (hinput : 0 ≤ inputBound) :
    {c : Circuit.DownstreamLinearCert // Circuit.DownstreamLinearBounds c} := by
  let gain := rowSumNorm W
  let error := gain * inputBound
  refine ⟨{ error := error, gain := gain, inputBound := inputBound }, ?_⟩
  refine
    { error_nonneg := ?_
      gain_nonneg := ?_
      input_nonneg := hinput
      error_eq := rfl }
  · exact mul_nonneg (rowSumNorm_nonneg W) hinput
  · exact rowSumNorm_nonneg W

end Bounds

end Sound

end Nfp
