-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Real.Basic
import Nfp.Core.Basic
import Nfp.Sound.Linear.FinFold

/-!
Interval bounds for dot products and matrix-vector products.

This module isolates interval-bound helpers used across downstream certificates.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

lemma foldl_max_ge_init {α : Type _} (f : α → Dyadic) :
    ∀ (l : List α) (init : Dyadic),
      init ≤ l.foldl (fun acc x => max acc (f x)) init := by
  intro l init
  induction l generalizing init with
  | nil =>
      simp
  | cons a l ih =>
      have hinit : init ≤ max init (f a) := le_max_left _ _
      have hrest : max init (f a) ≤ l.foldl (fun acc x => max acc (f x)) (max init (f a)) :=
        ih (max init (f a))
      simpa [List.foldl] using le_trans hinit hrest

lemma foldl_max_ge_mem {α : Type _} (f : α → Dyadic) :
    ∀ (l : List α) (a : α) (init : Dyadic),
      a ∈ l → f a ≤ l.foldl (fun acc x => max acc (f x)) init := by
  intro l a init hmem
  induction l generalizing init with
  | nil =>
      cases hmem
  | cons b l ih =>
      have hmem' : a = b ∨ a ∈ l := by
        simpa using hmem
      cases hmem' with
      | inl h =>
          subst h
          have hstep : f a ≤ max init (f a) := le_max_right _ _
          have hrest :
              max init (f a) ≤ l.foldl (fun acc x => max acc (f x)) (max init (f a)) :=
            foldl_max_ge_init (f := f) l (max init (f a))
          simpa [List.foldl] using le_trans hstep hrest
      | inr h =>
          have h' := ih (init := max init (f b)) h
          simpa [List.foldl] using h'

lemma foldlFin_max_ge_init {n : Nat} (f : Fin n → Dyadic) (init : Dyadic) :
    init ≤ Linear.foldlFin n (fun acc j => max acc (f j)) init := by
  classical
  have hlist :
      init ≤ (List.finRange n).foldl (fun acc j => max acc (f j)) init :=
    foldl_max_ge_init (f := f) (List.finRange n) init
  have hfold :
      Linear.foldlFin n (fun acc j => max acc (f j)) init =
        (List.finRange n).foldl (fun acc j => max acc (f j)) init := by
    simpa [Linear.foldlFin_eq_foldl] using
      (Fin.foldl_eq_foldl_finRange
        (f := fun acc j => max acc (f j)) (x := init) (n := n))
  simpa [hfold] using hlist

lemma foldlFin_max_ge {n : Nat} (f : Fin n → Dyadic) (i : Fin n) :
    f i ≤ Linear.foldlFin n (fun acc j => max acc (f j)) 0 := by
  classical
  have hmem : i ∈ List.finRange n := by
    simp
  have hlist :
      f i ≤ (List.finRange n).foldl (fun acc j => max acc (f j)) 0 :=
    foldl_max_ge_mem (f := f) (List.finRange n) i 0 hmem
  have hfold :
      Linear.foldlFin n (fun acc j => max acc (f j)) 0 =
        (List.finRange n).foldl (fun acc j => max acc (f j)) 0 := by
    simpa [Linear.foldlFin_eq_foldl] using
      (Fin.foldl_eq_foldl_finRange
        (f := fun acc j => max acc (f j)) (x := (0 : Dyadic)) (n := n))
  simpa [hfold] using hlist

/-- Lower interval endpoint for a dot product with per-coordinate bounds. -/
def dotIntervalLower {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  Linear.sumFin n (fun j => if 0 ≤ v j then v j * lo j else v j * hi j)

/-- Upper interval endpoint for a dot product with per-coordinate bounds. -/
def dotIntervalUpper {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  Linear.sumFin n (fun j => if 0 ≤ v j then v j * hi j else v j * lo j)

/-- Lower interval endpoint using a shared-denominator accumulator. -/
def dotIntervalLowerCommonDen {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  Linear.sumFinCommonDen n (fun j => if 0 ≤ v j then v j * lo j else v j * hi j)

/-- Upper interval endpoint using a shared-denominator accumulator. -/
def dotIntervalUpperCommonDen {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  Linear.sumFinCommonDen n (fun j => if 0 ≤ v j then v j * hi j else v j * lo j)

/-- Lower interval endpoint using unnormalized accumulation. -/
def dotIntervalLowerUnnorm {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  dotIntervalLower v lo hi

/-- Upper interval endpoint using unnormalized accumulation. -/
def dotIntervalUpperUnnorm {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  dotIntervalUpper v lo hi

theorem dotIntervalLowerCommonDen_eq {n : Nat} (v lo hi : Fin n → Dyadic) :
    dotIntervalLowerCommonDen v lo hi = dotIntervalLower v lo hi := by
  simp [dotIntervalLowerCommonDen, dotIntervalLower, Linear.sumFinCommonDen_eq_sumFin]

theorem dotIntervalUpperCommonDen_eq {n : Nat} (v lo hi : Fin n → Dyadic) :
    dotIntervalUpperCommonDen v lo hi = dotIntervalUpper v lo hi := by
  simp [dotIntervalUpperCommonDen, dotIntervalUpper, Linear.sumFinCommonDen_eq_sumFin]

theorem dotIntervalLowerUnnorm_eq {n : Nat} (v lo hi : Fin n → Dyadic) :
    dotIntervalLowerUnnorm v lo hi = dotIntervalLower v lo hi := rfl

theorem dotIntervalUpperUnnorm_eq {n : Nat} (v lo hi : Fin n → Dyadic) :
    dotIntervalUpperUnnorm v lo hi = dotIntervalUpper v lo hi := rfl

/-! Cached endpoints. -/

/-- Cached-array lower interval endpoint for a dot product using normalized dyadic sums. -/
def dotIntervalLowerCachedDyadic {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  let vArr := Array.ofFn v
  let loArr := Array.ofFn lo
  let hiArr := Array.ofFn hi
  Linear.sumFin n (fun j =>
    let vj := vArr[j.1]'(by
      have hsize : vArr.size = n := by simp [vArr]
      simp [hsize, j.isLt])
    let loj := loArr[j.1]'(by
      have hsize : loArr.size = n := by simp [loArr]
      simp [hsize, j.isLt])
    let hij := hiArr[j.1]'(by
      have hsize : hiArr.size = n := by simp [hiArr]
      simp [hsize, j.isLt])
    if 0 ≤ vj then vj * loj else vj * hij)

/-- Cached-array upper interval endpoint for a dot product using normalized dyadic sums. -/
def dotIntervalUpperCachedDyadic {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  let vArr := Array.ofFn v
  let loArr := Array.ofFn lo
  let hiArr := Array.ofFn hi
  Linear.sumFin n (fun j =>
    let vj := vArr[j.1]'(by
      have hsize : vArr.size = n := by simp [vArr]
      simp [hsize, j.isLt])
    let loj := loArr[j.1]'(by
      have hsize : loArr.size = n := by simp [loArr]
      simp [hsize, j.isLt])
    let hij := hiArr[j.1]'(by
      have hsize : hiArr.size = n := by simp [hiArr]
      simp [hsize, j.isLt])
    if 0 ≤ vj then vj * hij else vj * loj)

theorem dotIntervalLowerCachedRat_eq {n : Nat} (v lo hi : Fin n → Dyadic) :
    dotIntervalLowerCachedDyadic v lo hi = dotIntervalLower v lo hi := by
  classical
  simp [dotIntervalLowerCachedDyadic, dotIntervalLower, Linear.sumFin_eq_list_foldl,
    Array.getElem_ofFn]

theorem dotIntervalUpperCachedRat_eq {n : Nat} (v lo hi : Fin n → Dyadic) :
    dotIntervalUpperCachedDyadic v lo hi = dotIntervalUpper v lo hi := by
  classical
  simp [dotIntervalUpperCachedDyadic, dotIntervalUpper, Linear.sumFin_eq_list_foldl,
    Array.getElem_ofFn]

/-! Absolute bounds. -/

/-- Absolute bound from interval endpoints for a dot product. -/
def dotIntervalAbsBound {n : Nat} (v lo hi : Fin n → Dyadic) : Dyadic :=
  max |dotIntervalLower v lo hi| |dotIntervalUpper v lo hi|

/-- Lower interval endpoint for a matrix-vector product under input intervals. -/
def mulVecIntervalLower {m n : Nat} (W : Matrix (Fin m) (Fin n) Dyadic)
    (lo hi : Fin n → Dyadic) : Fin m → Dyadic :=
  fun i => dotIntervalLower (fun j => W i j) lo hi

/-- Upper interval endpoint for a matrix-vector product under input intervals. -/
def mulVecIntervalUpper {m n : Nat} (W : Matrix (Fin m) (Fin n) Dyadic)
    (lo hi : Fin n → Dyadic) : Fin m → Dyadic :=
  fun i => dotIntervalUpper (fun j => W i j) lo hi

theorem dotIntervalLower_le_dotProduct {n : Nat} (v lo hi x : Fin n → Dyadic)
    (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    dotIntervalLower v lo hi ≤ dotProduct v x := by
  classical
  simp only [dotIntervalLower, Linear.sumFin_eq_sum_univ, dotProduct]
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

theorem dotProduct_le_dotIntervalUpper {n : Nat} (v lo hi x : Fin n → Dyadic)
    (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    dotProduct v x ≤ dotIntervalUpper v lo hi := by
  classical
  simp only [dotIntervalUpper, Linear.sumFin_eq_sum_univ, dotProduct]
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

theorem abs_le_max_abs_abs_of_interval {a b x : Dyadic} (hlo : a ≤ x) (hhi : x ≤ b) :
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

/-- Global absolute bound from interval endpoints. -/
def intervalAbsBound {n : Nat} (lo hi : Fin n → Dyadic) : Dyadic :=
  Linear.foldlFin n (fun acc i => max acc (max |lo i| |hi i|)) 0

/-- `intervalAbsBound` dominates each endpoint absolute value. -/
theorem max_abs_le_intervalAbsBound {n : Nat} (lo hi : Fin n → Dyadic) (i : Fin n) :
    max |lo i| |hi i| ≤ intervalAbsBound lo hi := by
  simpa [intervalAbsBound] using
    (foldlFin_max_ge (f := fun j => max |lo j| |hi j|) i)

/-- `intervalAbsBound` bounds any element inside the interval. -/
theorem abs_le_intervalAbsBound {n : Nat} (lo hi x : Fin n → Dyadic)
    (hlo : ∀ i, lo i ≤ x i) (hhi : ∀ i, x i ≤ hi i) (i : Fin n) :
    |x i| ≤ intervalAbsBound lo hi := by
  have hbound : |x i| ≤ max |lo i| |hi i| :=
    abs_le_max_abs_abs_of_interval (hlo i) (hhi i)
  have hsup : max |lo i| |hi i| ≤ intervalAbsBound lo hi :=
    max_abs_le_intervalAbsBound lo hi i
  exact le_trans hbound hsup

theorem abs_dotProduct_le_dotIntervalAbsBound {n : Nat} (v lo hi x : Fin n → Dyadic)
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

theorem dotIntervalLower_le_dotProduct_real {n : Nat} (v lo hi : Fin n → Dyadic)
    (x : Fin n → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    (dotIntervalLower v lo hi : Real) ≤ dotProduct (fun j => (v j : Real)) x := by
  classical
  have hcast :
      (dotIntervalLower v lo hi : Real) =
        ∑ j, if 0 ≤ v j then (v j : Real) * (lo j : Real) else (v j : Real) * (hi j : Real) := by
    simpa [dotIntervalLower, dyadicToReal_mul, dyadicToReal_if] using
      (Linear.dyadicToReal_sumFin
        (f := fun j => if 0 ≤ v j then v j * lo j else v j * hi j))
  have hsum :
      (∑ j, if 0 ≤ v j then (v j : Real) * (lo j : Real) else (v j : Real) * (hi j : Real)) ≤
        ∑ j, (v j : Real) * x j := by
    refine Finset.sum_le_sum ?_
    intro j _
    by_cases hv : 0 ≤ v j
    · have h1 : (v j : Real) * (lo j : Real) ≤ (v j : Real) * x j := by
        have hv' : (0 : Real) ≤ (v j : Real) := dyadicToReal_nonneg_of_nonneg hv
        exact mul_le_mul_of_nonneg_left (hlo j) hv'
      simpa [hv] using h1
    · have hv' : (v j : Real) ≤ 0 := by
        exact (dyadicToReal_nonpos_iff (x := v j)).2 (le_of_not_ge hv)
      have h1 : (v j : Real) * (hi j : Real) ≤ (v j : Real) * x j := by
        exact mul_le_mul_of_nonpos_left (hhi j) hv'
      simpa [hv] using h1
  simpa [hcast, dotProduct] using hsum

theorem dotProduct_le_dotIntervalUpper_real {n : Nat} (v lo hi : Fin n → Dyadic)
    (x : Fin n → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    dotProduct (fun j => (v j : Real)) x ≤ (dotIntervalUpper v lo hi : Real) := by
  classical
  have hcast :
      (dotIntervalUpper v lo hi : Real) =
        ∑ j, if 0 ≤ v j then (v j : Real) * (hi j : Real) else (v j : Real) * (lo j : Real) := by
    simpa [dotIntervalUpper, dyadicToReal_mul, dyadicToReal_if] using
      (Linear.dyadicToReal_sumFin
        (f := fun j => if 0 ≤ v j then v j * hi j else v j * lo j))
  have hsum :
      ∑ j, (v j : Real) * x j ≤
        ∑ j, if 0 ≤ v j then (v j : Real) * (hi j : Real) else (v j : Real) * (lo j : Real) := by
    refine Finset.sum_le_sum ?_
    intro j _
    by_cases hv : 0 ≤ v j
    · have h1 : (v j : Real) * x j ≤ (v j : Real) * (hi j : Real) := by
        have hv' : (0 : Real) ≤ (v j : Real) := dyadicToReal_nonneg_of_nonneg hv
        exact mul_le_mul_of_nonneg_left (hhi j) hv'
      simpa [hv] using h1
    · have hv' : (v j : Real) ≤ 0 := by
        exact (dyadicToReal_nonpos_iff (x := v j)).2 (le_of_not_ge hv)
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

/-- `intervalAbsBound` controls real-valued coordinates inside a rational interval. -/
theorem abs_le_intervalAbsBound_real {n : Nat} (lo hi : Fin n → Dyadic) (x : Fin n → Real)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) (i : Fin n) :
    |x i| ≤ (intervalAbsBound lo hi : Real) := by
  have hbound : |x i| ≤ max |(lo i : Real)| |(hi i : Real)| :=
    abs_le_max_abs_abs_of_interval_real (hlo i) (hhi i)
  have hsup_real :
      max |(lo i : Real)| |(hi i : Real)| ≤ (intervalAbsBound lo hi : Real) := by
    have hsup : max |lo i| |hi i| ≤ intervalAbsBound lo hi :=
      max_abs_le_intervalAbsBound lo hi i
    have hlo : |lo i| ≤ intervalAbsBound lo hi :=
      le_trans (le_max_left _ _) hsup
    have hhi : |hi i| ≤ intervalAbsBound lo hi :=
      le_trans (le_max_right _ _) hsup
    have hlo_real :
        |(lo i : Real)| ≤ (intervalAbsBound lo hi : Real) := by
      exact dyadicToReal_abs_le_of_le hlo
    have hhi_real :
        |(hi i : Real)| ≤ (intervalAbsBound lo hi : Real) := by
      exact dyadicToReal_abs_le_of_le hhi
    exact max_le_iff.mpr ⟨hlo_real, hhi_real⟩
  exact le_trans hbound hsup_real

theorem abs_dotProduct_le_dotIntervalAbsBound_real {n : Nat} (v lo hi : Fin n → Dyadic)
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
    simp [dotIntervalAbsBound, dyadicToReal_abs, dyadicToReal_max]
  simpa [hcast] using habs

/-! Matrix-vector interval bounds. -/

theorem mulVecIntervalLower_le_mulVec {m n : Nat} (W : Matrix (Fin m) (Fin n) Dyadic)
    (lo hi x : Fin n → Dyadic) (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    ∀ i, mulVecIntervalLower W lo hi i ≤ Matrix.mulVec W x i := by
  intro i
  have h :=
    dotIntervalLower_le_dotProduct (v := fun j => W i j) lo hi x hlo hhi
  simpa [mulVecIntervalLower, Matrix.mulVec, dotProduct] using h

theorem mulVec_le_mulVecIntervalUpper {m n : Nat} (W : Matrix (Fin m) (Fin n) Dyadic)
    (lo hi x : Fin n → Dyadic) (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    ∀ i, Matrix.mulVec W x i ≤ mulVecIntervalUpper W lo hi i := by
  intro i
  have h :=
    dotProduct_le_dotIntervalUpper (v := fun j => W i j) lo hi x hlo hhi
  simpa [mulVecIntervalUpper, Matrix.mulVec, dotProduct] using h

theorem mulVecIntervalLower_le_upper {m n : Nat} (W : Matrix (Fin m) (Fin n) Dyadic)
    (lo hi : Fin n → Dyadic) (hlohi : ∀ j, lo j ≤ hi j) :
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

end Bounds

end Sound

end Nfp
