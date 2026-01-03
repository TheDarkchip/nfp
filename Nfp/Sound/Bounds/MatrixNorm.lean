-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Rat.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Data.Real.Basic
import Nfp.Circuit.Cert.DownstreamLinear
import Nfp.Circuit.Cert.ResidualInterval

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

/-- Weighted row-sum using per-coordinate bounds. -/
def rowSumWeighted {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) (i : Fin m) : Rat :=
  ∑ j, |W i j| * bound j

/-- Maximum row-sum norm (defaults to `0` on empty matrices). -/
def rowSumNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) : Rat :=
  if h : (Finset.univ : Finset (Fin m)).Nonempty then
    (Finset.univ).sup' h (fun i => rowSum W i)
  else
    0

/-- Maximum weighted row-sum (defaults to `0` on empty matrices). -/
def rowSumWeightedNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) : Rat :=
  if h : (Finset.univ : Finset (Fin m)).Nonempty then
    (Finset.univ).sup' h (fun i => rowSumWeighted W bound i)
  else
    0

/-- Row-sums are nonnegative. -/
theorem rowSum_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat) (i : Fin m) :
    0 ≤ rowSum W i := by
  refine Finset.sum_nonneg ?_
  intro j _
  exact abs_nonneg (W i j)

/-- Weighted row-sums are nonnegative under nonnegative bounds. -/
theorem rowSumWeighted_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) (i : Fin m) (hbound : ∀ j, 0 ≤ bound j) :
    0 ≤ rowSumWeighted W bound i := by
  refine Finset.sum_nonneg ?_
  intro j _
  exact mul_nonneg (abs_nonneg (W i j)) (hbound j)

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

/-- Each weighted row-sum is bounded by the weighted row-sum norm. -/
theorem rowSumWeighted_le_rowSumWeightedNorm {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) (i : Fin m) :
    rowSumWeighted W bound i ≤ rowSumWeightedNorm W bound := by
  classical
  have h : (Finset.univ : Finset (Fin m)).Nonempty := ⟨i, by simp⟩
  have hle :
      rowSumWeighted W bound i ≤
        (Finset.univ).sup' h (fun i => rowSumWeighted W bound i) := by
    simpa using
      (Finset.le_sup'
        (s := (Finset.univ : Finset (Fin m)))
        (f := fun i => rowSumWeighted W bound i)
        (by simp : i ∈ (Finset.univ : Finset (Fin m))))
  simpa [rowSumWeightedNorm, h] using hle

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

/-- Weighted row-sum norm is nonnegative under nonnegative bounds. -/
theorem rowSumWeightedNorm_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) (hbound : ∀ j, 0 ≤ bound j) :
    0 ≤ rowSumWeightedNorm W bound := by
  classical
  by_cases h : (Finset.univ : Finset (Fin m)).Nonempty
  · rcases h with ⟨i, hi⟩
    have hrow : 0 ≤ rowSumWeighted W bound i :=
      rowSumWeighted_nonneg W bound i hbound
    have hle : rowSumWeighted W bound i ≤ rowSumWeightedNorm W bound :=
      rowSumWeighted_le_rowSumWeightedNorm W bound i
    exact le_trans hrow hle
  · simp [rowSumWeightedNorm, h]

/-- Downstream error from per-coordinate residual bounds. -/
def downstreamErrorFromBounds {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) : Rat :=
  rowSumWeightedNorm W bound

/-- `downstreamErrorFromBounds` is nonnegative. -/
theorem downstreamErrorFromBounds_nonneg {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (bound : Fin n → Rat) (hbound : ∀ j, 0 ≤ bound j) :
    0 ≤ downstreamErrorFromBounds W bound := by
  simpa [downstreamErrorFromBounds] using rowSumWeightedNorm_nonneg W bound hbound

/-- Lower interval endpoint for a dot product with per-coordinate bounds. -/
def dotIntervalLower {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  ∑ j, if 0 ≤ v j then v j * lo j else v j * hi j

/-- Upper interval endpoint for a dot product with per-coordinate bounds. -/
def dotIntervalUpper {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  ∑ j, if 0 ≤ v j then v j * hi j else v j * lo j

/-- Absolute bound from interval endpoints for a dot product. -/
def dotIntervalAbsBound {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  max |dotIntervalLower v lo hi| |dotIntervalUpper v lo hi|

/-- Lower interval endpoint for a matrix-vector product under input intervals. -/
def mulVecIntervalLower {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (lo hi : Fin n → Rat) : Fin m → Rat :=
  fun i => dotIntervalLower (fun j => W i j) lo hi

/-- Upper interval endpoint for a matrix-vector product under input intervals. -/
def mulVecIntervalUpper {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (lo hi : Fin n → Rat) : Fin m → Rat :=
  fun i => dotIntervalUpper (fun j => W i j) lo hi

theorem dotIntervalLower_le_dotProduct {n : Nat} (v lo hi x : Fin n → Rat)
    (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    dotIntervalLower v lo hi ≤ dotProduct v x := by
  classical
  refine Finset.sum_le_sum ?_
  intro j _
  by_cases hv : 0 ≤ v j
  · have h1 : v j * lo j ≤ v j * x j :=
      mul_le_mul_of_nonneg_left (hlo j) hv
    simpa [hv] using h1
  · have hv' : v j ≤ 0 := le_of_lt (lt_of_not_ge hv)
    have h1 : v j * hi j ≤ v j * x j :=
      mul_le_mul_of_nonpos_left (hhi j) hv'
    simpa [hv] using h1

theorem dotProduct_le_dotIntervalUpper {n : Nat} (v lo hi x : Fin n → Rat)
    (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    dotProduct v x ≤ dotIntervalUpper v lo hi := by
  classical
  refine Finset.sum_le_sum ?_
  intro j _
  by_cases hv : 0 ≤ v j
  · have h1 : v j * x j ≤ v j * hi j :=
      mul_le_mul_of_nonneg_left (hhi j) hv
    simpa [hv] using h1
  · have hv' : v j ≤ 0 := le_of_lt (lt_of_not_ge hv)
    have h1 : v j * x j ≤ v j * lo j :=
      mul_le_mul_of_nonpos_left (hlo j) hv'
    simpa [hv] using h1

theorem abs_le_max_abs_abs_of_interval {a b x : Rat} (hlo : a ≤ x) (hhi : x ≤ b) :
    |x| ≤ max |a| |b| := by
  by_cases hx : 0 ≤ x
  · have hb : 0 ≤ b := le_trans hx hhi
    have hx' : |x| = x := abs_of_nonneg hx
    have hb' : |b| = b := abs_of_nonneg hb
    calc
      |x| = x := hx'
      _ ≤ b := hhi
      _ = |b| := hb'.symm
      _ ≤ max |a| |b| := le_max_right _ _
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have ha : a ≤ 0 := le_trans hlo hx'
    have hxabs : |x| = -x := abs_of_nonpos hx'
    have haabs : |a| = -a := abs_of_nonpos ha
    calc
      |x| = -x := hxabs
      _ ≤ -a := neg_le_neg hlo
      _ = |a| := by simp [haabs]
      _ ≤ max |a| |b| := le_max_left _ _

theorem abs_dotProduct_le_dotIntervalAbsBound {n : Nat} (v lo hi x : Fin n → Rat)
    (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    |dotProduct v x| ≤ dotIntervalAbsBound v lo hi := by
  have hlow : dotIntervalLower v lo hi ≤ dotProduct v x :=
    dotIntervalLower_le_dotProduct v lo hi x hlo hhi
  have hhigh : dotProduct v x ≤ dotIntervalUpper v lo hi :=
    dotProduct_le_dotIntervalUpper v lo hi x hlo hhi
  have habs : |dotProduct v x| ≤
      max |dotIntervalLower v lo hi| |dotIntervalUpper v lo hi| :=
    abs_le_max_abs_abs_of_interval hlow hhigh
  unfold dotIntervalAbsBound
  exact habs

/-! Real-valued bounds from rational intervals. -/

theorem dotIntervalLower_le_dotProduct_real {n : Nat} (v lo hi : Fin n → Rat)
    (x : Fin n → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    (dotIntervalLower v lo hi : Real) ≤ dotProduct (fun j => (v j : Real)) x := by
  classical
  have hcast :
      (dotIntervalLower v lo hi : Real) =
        ∑ j, if 0 ≤ v j then (v j : Real) * (lo j : Real) else (v j : Real) * (hi j : Real) := by
    conv_lhs => simp [dotIntervalLower]
    refine Finset.sum_congr rfl ?_
    intro j _
    by_cases hv : 0 ≤ v j
    · simp [hv]
    · simp [hv]
  have hsum :
      (∑ j, if 0 ≤ v j then (v j : Real) * (lo j : Real) else (v j : Real) * (hi j : Real)) ≤
        ∑ j, (v j : Real) * x j := by
    refine Finset.sum_le_sum ?_
    intro j _
    by_cases hv : 0 ≤ v j
    · have h1 : (v j : Real) * (lo j : Real) ≤ (v j : Real) * x j := by
        exact mul_le_mul_of_nonneg_left (hlo j) (by exact_mod_cast hv)
      simpa [hv] using h1
    · have hv' : (v j : Real) ≤ 0 := by
        exact_mod_cast (le_of_lt (lt_of_not_ge hv))
      have h1 : (v j : Real) * (hi j : Real) ≤ (v j : Real) * x j := by
        exact mul_le_mul_of_nonpos_left (hhi j) hv'
      simpa [hv] using h1
  simpa [hcast, dotProduct] using hsum

theorem dotProduct_le_dotIntervalUpper_real {n : Nat} (v lo hi : Fin n → Rat)
    (x : Fin n → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    dotProduct (fun j => (v j : Real)) x ≤ (dotIntervalUpper v lo hi : Real) := by
  classical
  have hcast :
      (dotIntervalUpper v lo hi : Real) =
        ∑ j, if 0 ≤ v j then (v j : Real) * (hi j : Real) else (v j : Real) * (lo j : Real) := by
    conv_lhs => simp [dotIntervalUpper]
    refine Finset.sum_congr rfl ?_
    intro j _
    by_cases hv : 0 ≤ v j
    · simp [hv]
    · simp [hv]
  have hsum :
      ∑ j, (v j : Real) * x j ≤
        ∑ j, if 0 ≤ v j then (v j : Real) * (hi j : Real) else (v j : Real) * (lo j : Real) := by
    refine Finset.sum_le_sum ?_
    intro j _
    by_cases hv : 0 ≤ v j
    · have h1 : (v j : Real) * x j ≤ (v j : Real) * (hi j : Real) := by
        exact mul_le_mul_of_nonneg_left (hhi j) (by exact_mod_cast hv)
      simpa [hv] using h1
    · have hv' : (v j : Real) ≤ 0 := by
        exact_mod_cast (le_of_lt (lt_of_not_ge hv))
      have h1 : (v j : Real) * x j ≤ (v j : Real) * (lo j : Real) := by
        exact mul_le_mul_of_nonpos_left (hlo j) hv'
      simpa [hv] using h1
  simpa [hcast, dotProduct] using hsum

theorem abs_le_max_abs_abs_of_interval_real {a b x : Real} (hlo : a ≤ x) (hhi : x ≤ b) :
    |x| ≤ max |a| |b| := by
  by_cases hx : 0 ≤ x
  · have hb : 0 ≤ b := le_trans hx hhi
    have hx' : |x| = x := abs_of_nonneg hx
    have hb' : |b| = b := abs_of_nonneg hb
    calc
      |x| = x := hx'
      _ ≤ b := hhi
      _ = |b| := hb'.symm
      _ ≤ max |a| |b| := le_max_right _ _
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have ha : a ≤ 0 := le_trans hlo hx'
    have hxabs : |x| = -x := abs_of_nonpos hx'
    have haabs : |a| = -a := abs_of_nonpos ha
    calc
      |x| = -x := hxabs
      _ ≤ -a := neg_le_neg hlo
      _ = |a| := by simp [haabs]
      _ ≤ max |a| |b| := le_max_left _ _

theorem abs_dotProduct_le_dotIntervalAbsBound_real {n : Nat} (v lo hi : Fin n → Rat)
    (x : Fin n → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    |dotProduct (fun j => (v j : Real)) x| ≤ (dotIntervalAbsBound v lo hi : Real) := by
  have hlow :
      (dotIntervalLower v lo hi : Real) ≤ dotProduct (fun j => (v j : Real)) x :=
    dotIntervalLower_le_dotProduct_real v lo hi x hlo hhi
  have hhigh :
      dotProduct (fun j => (v j : Real)) x ≤ (dotIntervalUpper v lo hi : Real) :=
    dotProduct_le_dotIntervalUpper_real v lo hi x hlo hhi
  have habs :
      |dotProduct (fun j => (v j : Real)) x| ≤
        max |(dotIntervalLower v lo hi : Real)| |(dotIntervalUpper v lo hi : Real)| :=
    abs_le_max_abs_abs_of_interval_real hlow hhigh
  have hcast :
      (dotIntervalAbsBound v lo hi : Real) =
        max |(dotIntervalLower v lo hi : Real)| |(dotIntervalUpper v lo hi : Real)| := by
    simp [dotIntervalAbsBound]
  simpa [hcast] using habs

/-- Matrix-interval lower bounds dominate matrix-vector products. -/
theorem mulVecIntervalLower_le_mulVec {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (lo hi x : Fin n → Rat) (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    ∀ i, mulVecIntervalLower W lo hi i ≤ Matrix.mulVec W x i := by
  intro i
  have h :=
    dotIntervalLower_le_dotProduct (v := fun j => W i j) lo hi x hlo hhi
  simpa [mulVecIntervalLower, Matrix.mulVec, dotProduct] using h

/-- Matrix-interval upper bounds dominate matrix-vector products. -/
theorem mulVec_le_mulVecIntervalUpper {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (lo hi x : Fin n → Rat) (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    ∀ i, Matrix.mulVec W x i ≤ mulVecIntervalUpper W lo hi i := by
  intro i
  have h :=
    dotProduct_le_dotIntervalUpper (v := fun j => W i j) lo hi x hlo hhi
  simpa [mulVecIntervalUpper, Matrix.mulVec, dotProduct] using h

/-- Interval endpoints for `mulVec` are ordered when the input interval is ordered. -/
theorem mulVecIntervalLower_le_upper {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (lo hi : Fin n → Rat) (hlohi : ∀ j, lo j ≤ hi j) :
    ∀ i, mulVecIntervalLower W lo hi i ≤ mulVecIntervalUpper W lo hi i := by
  intro i
  have hlow :
      dotIntervalLower (fun j => W i j) lo hi ≤ dotProduct (fun j => W i j) lo :=
    dotIntervalLower_le_dotProduct (v := fun j => W i j) lo hi lo
      (fun j => le_rfl) hlohi
  have hhigh :
      dotProduct (fun j => W i j) lo ≤ dotIntervalUpper (fun j => W i j) lo hi :=
    dotProduct_le_dotIntervalUpper (v := fun j => W i j) lo hi lo
      (fun j => le_rfl) hlohi
  exact le_trans hlow hhigh

/-- Build a residual-interval certificate by applying a matrix to an input interval. -/
def buildResidualIntervalCertFromMatrix {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (lo hi : Fin n → Rat) (hlohi : ∀ j, lo j ≤ hi j) :
    {c : Circuit.ResidualIntervalCert m // Circuit.ResidualIntervalBounds c} := by
  let lo' := mulVecIntervalLower W lo hi
  let hi' := mulVecIntervalUpper W lo hi
  refine ⟨{ lo := lo', hi := hi' }, ?_⟩
  refine { lo_le_hi := ?_ }
  intro i
  exact mulVecIntervalLower_le_upper W lo hi hlohi i

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
