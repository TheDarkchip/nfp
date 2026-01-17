-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.Ring.Abs
public import Mathlib.Data.Fintype.Basic
public import Mathlib.Data.Matrix.Mul
public import Mathlib.Data.Real.Basic
public import Nfp.Core.Basic
public import Nfp.Bounds.MatrixNorm.Interval
public import Nfp.Linear.FinFold

/-!
Row-sum matrix norms for downstream linear certificates.

These bounds are used to compute verified downstream error certificates
from explicit Rat matrices.
-/

public section

namespace Nfp


namespace Bounds

open scoped BigOperators

/-- Row-sum of absolute values for a matrix row. -/
def rowSum {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) (i : Fin m) : Rat :=
  Linear.sumFin n (fun j => |W i j|)

/-- Weighted row-sum using per-coordinate bounds. -/
def rowSumWeighted {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) (i : Fin m) : Rat :=
  Linear.sumFin n (fun j => |W i j| * bound j)

/-- Maximum row-sum norm (defaults to `0` on empty matrices). -/
def rowSumNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) : Rat :=
  Linear.foldlFin m (fun acc i => max acc (rowSum W i)) 0

/-- Maximum weighted row-sum (defaults to `0` on empty matrices). -/
def rowSumWeightedNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) : Rat :=
  Linear.foldlFin m (fun acc i => max acc (rowSumWeighted W bound i)) 0

/-- Row-sums are nonnegative. -/
theorem rowSum_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) (i : Fin m) :
    0 ≤ rowSum W i := by
  simpa [rowSum, Linear.sumFin_eq_sum_univ] using
    (Finset.sum_nonneg (fun j _ => abs_nonneg (W i j)))

/-- Weighted row-sums are nonnegative under nonnegative bounds. -/
theorem rowSumWeighted_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) (i : Fin m) (hbound : ∀ j, 0 ≤ bound j) :
    0 ≤ rowSumWeighted W bound i := by
  classical
  have hsum : 0 ≤ ∑ j, |W i j| * bound j := by
    refine Finset.sum_nonneg ?_
    intro j _
    exact mul_nonneg (abs_nonneg (W i j)) (hbound j)
  simpa [rowSumWeighted, Linear.sumFin_eq_sum_univ] using hsum

/-- Each row-sum is bounded by the row-sum norm. -/
theorem rowSum_le_rowSumNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) (i : Fin m) :
    rowSum W i ≤ rowSumNorm W := by
  simpa [rowSumNorm] using
    (foldlFin_max_ge (f := fun j => rowSum W j) i)

/-- Each weighted row-sum is bounded by the weighted row-sum norm. -/
theorem rowSumWeighted_le_rowSumWeightedNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) (i : Fin m) :
    rowSumWeighted W bound i ≤ rowSumWeightedNorm W bound := by
  simpa [rowSumWeightedNorm] using
    (foldlFin_max_ge (f := fun j => rowSumWeighted W bound j) i)

/-- The row-sum norm is nonnegative. -/
theorem rowSumNorm_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) :
    0 ≤ rowSumNorm W := by
  simpa [rowSumNorm] using
    (foldlFin_max_ge_init (f := fun i => rowSum W i) (init := (0 : Rat)))

/-- Weighted row-sum norm is nonnegative. -/
theorem rowSumWeightedNorm_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) :
    0 ≤ rowSumWeightedNorm W bound := by
  simpa [rowSumWeightedNorm] using
    (foldlFin_max_ge_init (f := fun i => rowSumWeighted W bound i) (init := (0 : Rat)))

/-- Downstream error from per-coordinate residual bounds. -/
def downstreamErrorFromBounds {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) : Rat :=
  rowSumWeightedNorm W bound

/-- `downstreamErrorFromBounds` is nonnegative. -/
theorem downstreamErrorFromBounds_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) :
    0 ≤ downstreamErrorFromBounds W bound := by
  simpa [downstreamErrorFromBounds] using rowSumWeightedNorm_nonneg W bound

/-- Summed absolute row entries factor out a scalar bound. -/
theorem sum_abs_row_mul_eq_rowSum_mul {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (i : Fin m) (inputBound : Rat) :
    (∑ j, |W i j| * inputBound) = rowSum W i * inputBound := by
  have hsum :
      (∑ j, |W i j|) * inputBound = ∑ j, |W i j| * inputBound := by
    simpa using
      (Finset.sum_mul
        (s := (Finset.univ : Finset (Fin n)))
        (f := fun j => |W i j|)
        (a := inputBound))
  simpa [rowSum, Linear.sumFin_eq_sum_univ] using hsum.symm

/-- Row-sum norm bounds a matrix-vector product under a uniform input bound. -/
theorem abs_mulVec_le_rowSum {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (x : Fin n → Rat) (inputBound : Rat)
    (hx : ∀ j, |x j| ≤ inputBound) :
    ∀ i, |Matrix.mulVec W x i| ≤ rowSum W i * inputBound := by
  intro i
  have h1 : |∑ j, W i j * x j| ≤ ∑ j, |W i j * x j| := by
    simpa using
      (Finset.abs_sum_le_sum_abs
        (f := fun j => W i j * x j)
        (s := (Finset.univ : Finset (Fin n))))
  have h2 : ∑ j, |W i j * x j| ≤ ∑ j, |W i j| * inputBound := by
    refine Finset.sum_le_sum ?_
    intro j _
    have hnonneg : 0 ≤ |W i j| := abs_nonneg (W i j)
    calc
      |W i j * x j| = |W i j| * |x j| := by
        simp [abs_mul]
      _ ≤ |W i j| * inputBound := by
        exact mul_le_mul_of_nonneg_left (hx j) hnonneg
  have h3 : ∑ j, |W i j| * inputBound = rowSum W i * inputBound :=
    sum_abs_row_mul_eq_rowSum_mul W i inputBound
  simpa [Matrix.mulVec, dotProduct] using h1.trans (h2.trans_eq h3)

/-- Row-sum norm bounds a matrix-vector product under a uniform input bound. -/
theorem abs_mulVec_le_rowSumNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (x : Fin n → Rat) (inputBound : Rat)
    (hx : ∀ j, |x j| ≤ inputBound) (hinput : 0 ≤ inputBound) :
    ∀ i, |Matrix.mulVec W x i| ≤ rowSumNorm W * inputBound := by
  intro i
  have hrow : |Matrix.mulVec W x i| ≤ rowSum W i * inputBound :=
    abs_mulVec_le_rowSum W x inputBound hx i
  have hle : rowSum W i ≤ rowSumNorm W := rowSum_le_rowSumNorm W i
  have hmul : rowSum W i * inputBound ≤ rowSumNorm W * inputBound :=
    mul_le_mul_of_nonneg_right hle hinput
  exact hrow.trans hmul

end Bounds


end Nfp
