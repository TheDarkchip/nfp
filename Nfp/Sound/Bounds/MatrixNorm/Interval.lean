-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.Ring.Abs
public import Mathlib.Data.Matrix.Mul
public import Mathlib.Data.Real.Basic
public import Nfp.Core.Basic
public import Nfp.Sound.Linear.FinFold

/-!
Interval bounds for dot products and matrix-vector products.

This module isolates interval-bound helpers used across downstream certificates.
-/

public section

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

lemma foldl_max_ge_init {α : Type _} (f : α → Rat) :
    ∀ (l : List α) (init : Rat),
      init ≤ l.foldl (fun acc x => max acc (f x)) init := by
  intro l init
  induction l generalizing init with
  | nil =>
      simp
  | cons a l ih =>
      simpa [List.foldl] using
        le_trans (le_max_left _ _) (ih (max init (f a)))

lemma foldl_max_ge_mem {α : Type _} (f : α → Rat) :
    ∀ (l : List α) (a : α) (init : Rat),
      a ∈ l → f a ≤ l.foldl (fun acc x => max acc (f x)) init := by
  intro l a init hmem
  induction l generalizing init with
  | nil =>
      cases hmem
  | cons b l ih =>
      rcases (List.mem_cons.mp hmem) with rfl | hmem
      · simpa [List.foldl] using
          le_trans (le_max_right _ _)
            (foldl_max_ge_init (f := f) l (max init (f a)))
      · simpa [List.foldl] using ih (init := max init (f b)) hmem

lemma foldlFin_max_ge_init {n : Nat} (f : Fin n → Rat) (init : Rat) :
    init ≤ Linear.foldlFin n (fun acc j => max acc (f j)) init := by
  classical
  simpa [Linear.foldlFin_eq_foldl, Fin.foldl_eq_foldl_finRange] using
    (foldl_max_ge_init (f := f) (List.finRange n) init)

lemma foldlFin_max_ge {n : Nat} (f : Fin n → Rat) (i : Fin n) :
    f i ≤ Linear.foldlFin n (fun acc j => max acc (f j)) 0 := by
  classical
  have hmem : i ∈ List.finRange n := by
    simp
  simpa [Linear.foldlFin_eq_foldl, Fin.foldl_eq_foldl_finRange] using
    (foldl_max_ge_mem (f := f) (List.finRange n) i 0 hmem)

/-- Lower interval endpoint for a dot product with per-coordinate bounds. -/
def dotIntervalLower {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun j => if 0 ≤ v j then v j * lo j else v j * hi j)

/-- Upper interval endpoint for a dot product with per-coordinate bounds. -/
def dotIntervalUpper {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun j => if 0 ≤ v j then v j * hi j else v j * lo j)

/-- Lower interval endpoint for a product of two intervals. -/
def mulIntervalLower (a b c d : Rat) : Rat :=
  min (min (a * c) (a * d)) (min (b * c) (b * d))

/-- Upper interval endpoint for a product of two intervals. -/
def mulIntervalUpper (a b c d : Rat) : Rat :=
  max (max (a * c) (a * d)) (max (b * c) (b * d))

/-- `x * y` lies between `min (a * y) (b * y)` and `max (a * y) (b * y)` when `a ≤ x ≤ b`. -/
lemma mul_between_of_bounds {a b x y : Rat} (hx : a ≤ x) (hx' : x ≤ b) :
    min (a * y) (b * y) ≤ x * y ∧ x * y ≤ max (a * y) (b * y) := by
  have hab : a ≤ b := le_trans hx hx'
  by_cases hy : 0 ≤ y
  · have hmin : min (a * y) (b * y) = a * y := by
      have hle : a * y ≤ b * y := by
        exact mul_le_mul_of_nonneg_right hab hy
      exact min_eq_left hle
    have hmax : max (a * y) (b * y) = b * y := by
      have hle : a * y ≤ b * y := by
        exact mul_le_mul_of_nonneg_right hab hy
      exact max_eq_right hle
    constructor
    · have hmul : a * y ≤ x * y := by
        exact mul_le_mul_of_nonneg_right hx hy
      simpa [hmin] using hmul
    · have hmul : x * y ≤ b * y := by
        exact mul_le_mul_of_nonneg_right hx' hy
      simpa [hmax] using hmul
  · have hy' : y ≤ 0 := le_of_not_ge hy
    have hmin : min (a * y) (b * y) = b * y := by
      have hle : b * y ≤ a * y := by
        exact mul_le_mul_of_nonpos_right hab hy'
      exact min_eq_right hle
    have hmax : max (a * y) (b * y) = a * y := by
      have hle : b * y ≤ a * y := by
        exact mul_le_mul_of_nonpos_right hab hy'
      exact max_eq_left hle
    constructor
    · have hmul : b * y ≤ x * y := by
        exact mul_le_mul_of_nonpos_right hx' hy'
      simpa [hmin] using hmul
    · have hmul : x * y ≤ a * y := by
        exact mul_le_mul_of_nonpos_right hx hy'
      simpa [hmax] using hmul

/-- Lower interval endpoint bounds `x * y` when both factors are interval-bounded. -/
lemma mulIntervalLower_le_mul {a b c d x y : Rat}
    (hx : a ≤ x) (hx' : x ≤ b) (hy : c ≤ y) (hy' : y ≤ d) :
    mulIntervalLower a b c d ≤ x * y := by
  have hAy :
      min (a * c) (a * d) ≤ a * y := by
    have h := mul_between_of_bounds (a := c) (b := d) (x := y) (y := a) hy hy'
    simpa only [mul_comm] using h.1
  have hBy :
      min (b * c) (b * d) ≤ b * y := by
    have h := mul_between_of_bounds (a := c) (b := d) (x := y) (y := b) hy hy'
    simpa only [mul_comm] using h.1
  have hmin :
      min (min (a * c) (a * d)) (min (b * c) (b * d)) ≤ min (a * y) (b * y) := by
    apply le_min
    · exact le_trans (min_le_left _ _) hAy
    · exact le_trans (min_le_right _ _) hBy
  have hxy := (mul_between_of_bounds (a := a) (b := b) (x := x) (y := y) hx hx').1
  exact le_trans hmin hxy

/-- Upper interval endpoint bounds `x * y` when both factors are interval-bounded. -/
lemma mul_le_mulIntervalUpper {a b c d x y : Rat}
    (hx : a ≤ x) (hx' : x ≤ b) (hy : c ≤ y) (hy' : y ≤ d) :
    x * y ≤ mulIntervalUpper a b c d := by
  have hAy :
      a * y ≤ max (a * c) (a * d) := by
    have h := mul_between_of_bounds (a := c) (b := d) (x := y) (y := a) hy hy'
    simpa only [mul_comm] using h.2
  have hBy :
      b * y ≤ max (b * c) (b * d) := by
    have h := mul_between_of_bounds (a := c) (b := d) (x := y) (y := b) hy hy'
    simpa only [mul_comm] using h.2
  have hmax :
      max (a * y) (b * y) ≤ max (max (a * c) (a * d)) (max (b * c) (b * d)) := by
    apply max_le
    · exact le_trans hAy (le_max_left _ _)
    · exact le_trans hBy (le_max_right _ _)
  have hxy := (mul_between_of_bounds (a := a) (b := b) (x := x) (y := y) hx hx').2
  exact le_trans hxy hmax

/-- Lower interval endpoint for a dot product with bounds on both vectors. -/
def dotIntervalLower2 {n : Nat} (lo1 hi1 lo2 hi2 : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun j => mulIntervalLower (lo1 j) (hi1 j) (lo2 j) (hi2 j))

/-- Upper interval endpoint for a dot product with bounds on both vectors. -/
def dotIntervalUpper2 {n : Nat} (lo1 hi1 lo2 hi2 : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun j => mulIntervalUpper (lo1 j) (hi1 j) (lo2 j) (hi2 j))

/-- Lower/upper interval endpoints for a dot product with bounds on both vectors. -/
def dotIntervalLowerUpper2CommonDen {n : Nat} (lo1 hi1 lo2 hi2 : Fin n → Rat) : Rat × Rat :=
  Linear.foldlFin n
    (fun acc j =>
      (acc.1 + mulIntervalLower (lo1 j) (hi1 j) (lo2 j) (hi2 j),
        acc.2 + mulIntervalUpper (lo1 j) (hi1 j) (lo2 j) (hi2 j)))
    (0, 0)

/-! Sign-splitting bounds. -/

/-- Clamp a single coordinate interval to be nonnegative or nonpositive. -/
def clampAt {n : Nat} (i : Fin n) (nonneg : Bool) (lo hi : Fin n → Rat) :
    (Fin n → Rat) × (Fin n → Rat) :=
  if nonneg then
    (fun j => if j = i then max 0 (lo j) else lo j, hi)
  else
    (lo, fun j => if j = i then min 0 (hi j) else hi j)

/-- Lower/upper interval endpoints with sign-splitting on selected coordinates. -/
def dotIntervalLowerUpper2SignSplit {n : Nat} (dims : List (Fin n))
    (lo1 hi1 lo2 hi2 : Fin n → Rat) : Rat × Rat :=
  match dims with
  | [] =>
      dotIntervalLowerUpper2CommonDen lo1 hi1 lo2 hi2
  | i :: rest =>
      let boundsPos :=
        let clamped := clampAt i true lo1 hi1
        dotIntervalLowerUpper2SignSplit rest clamped.1 clamped.2 lo2 hi2
      let boundsNeg :=
        let clamped := clampAt i false lo1 hi1
        dotIntervalLowerUpper2SignSplit rest clamped.1 clamped.2 lo2 hi2
      (min boundsPos.1 boundsNeg.1, max boundsPos.2 boundsNeg.2)

/-- Lower/upper interval endpoints with sign-splitting on both sides. -/
def dotIntervalLowerUpper2SignSplitBoth {n : Nat} (dims1 dims2 : List (Fin n))
    (lo1 hi1 lo2 hi2 : Fin n → Rat) : Rat × Rat :=
  let bounds1 := dotIntervalLowerUpper2SignSplit dims1 lo1 hi1 lo2 hi2
  let bounds2 := dotIntervalLowerUpper2SignSplit dims2 lo2 hi2 lo1 hi1
  (max bounds1.1 bounds2.1, min bounds1.2 bounds2.2)

/-- Sum of lower interval products bounds the dot-product sum (Rat). -/
private theorem sum_mulIntervalLower_le_sum_mul {n : Nat}
    (lo1 hi1 lo2 hi2 x y : Fin n → Rat)
    (hlo1 : ∀ j, lo1 j ≤ x j) (hhi1 : ∀ j, x j ≤ hi1 j)
    (hlo2 : ∀ j, lo2 j ≤ y j) (hhi2 : ∀ j, y j ≤ hi2 j) :
    ∑ j, mulIntervalLower (lo1 j) (hi1 j) (lo2 j) (hi2 j) ≤
      ∑ j, x j * y j := by
  refine Finset.sum_le_sum ?_
  intro j _
  exact mulIntervalLower_le_mul (hlo1 j) (hhi1 j) (hlo2 j) (hhi2 j)

/-- Sum of products is bounded by the upper interval products (Rat). -/
private theorem sum_mul_le_sum_mulIntervalUpper {n : Nat}
    (lo1 hi1 lo2 hi2 x y : Fin n → Rat)
    (hlo1 : ∀ j, lo1 j ≤ x j) (hhi1 : ∀ j, x j ≤ hi1 j)
    (hlo2 : ∀ j, lo2 j ≤ y j) (hhi2 : ∀ j, y j ≤ hi2 j) :
    ∑ j, x j * y j ≤
      ∑ j, mulIntervalUpper (lo1 j) (hi1 j) (lo2 j) (hi2 j) := by
  refine Finset.sum_le_sum ?_
  intro j _
  exact mul_le_mulIntervalUpper (hlo1 j) (hhi1 j) (hlo2 j) (hhi2 j)

theorem dotIntervalLower2_le_dotProduct {n : Nat} (lo1 hi1 lo2 hi2 x y : Fin n → Rat)
    (hlo1 : ∀ j, lo1 j ≤ x j) (hhi1 : ∀ j, x j ≤ hi1 j)
    (hlo2 : ∀ j, lo2 j ≤ y j) (hhi2 : ∀ j, y j ≤ hi2 j) :
    dotIntervalLower2 lo1 hi1 lo2 hi2 ≤ dotProduct x y := by
  classical
  have hsum :=
    sum_mulIntervalLower_le_sum_mul
      (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2) (x := x) (y := y)
      hlo1 hhi1 hlo2 hhi2
  simpa [dotIntervalLower2, Linear.sumFin_eq_sum_univ, dotProduct] using hsum

theorem dotProduct_le_dotIntervalUpper2 {n : Nat} (lo1 hi1 lo2 hi2 x y : Fin n → Rat)
    (hlo1 : ∀ j, lo1 j ≤ x j) (hhi1 : ∀ j, x j ≤ hi1 j)
    (hlo2 : ∀ j, lo2 j ≤ y j) (hhi2 : ∀ j, y j ≤ hi2 j) :
    dotProduct x y ≤ dotIntervalUpper2 lo1 hi1 lo2 hi2 := by
  classical
  have hsum :=
    sum_mul_le_sum_mulIntervalUpper
      (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2) (x := x) (y := y)
      hlo1 hhi1 hlo2 hhi2
  simpa [dotIntervalUpper2, Linear.sumFin_eq_sum_univ, dotProduct] using hsum
/-- Lower interval endpoint using a shared-denominator accumulator. -/
def dotIntervalLowerCommonDen {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  Linear.sumFinCommonDen n (fun j => if 0 ≤ v j then v j * lo j else v j * hi j)

/-- Upper interval endpoint using a shared-denominator accumulator. -/
def dotIntervalUpperCommonDen {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  Linear.sumFinCommonDen n (fun j => if 0 ≤ v j then v j * hi j else v j * lo j)

/-- Lower/upper interval endpoints computed in a single pass. -/
def dotIntervalLowerUpperCommonDen {n : Nat} (v lo hi : Fin n → Rat) : Rat × Rat :=
  Linear.foldlFin n
    (fun acc j =>
      (acc.1 + if 0 ≤ v j then v j * lo j else v j * hi j,
        acc.2 + if 0 ≤ v j then v j * hi j else v j * lo j))
    (0, 0)

/-- Lower interval endpoint using unnormalized accumulation. -/
def dotIntervalLowerUnnorm {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  dotIntervalLower v lo hi

/-- Upper interval endpoint using unnormalized accumulation. -/
def dotIntervalUpperUnnorm {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  dotIntervalUpper v lo hi

theorem dotIntervalLowerCommonDen_eq {n : Nat} (v lo hi : Fin n → Rat) :
    dotIntervalLowerCommonDen v lo hi = dotIntervalLower v lo hi := by
  simp only [dotIntervalLowerCommonDen, dotIntervalLower, Linear.sumFinCommonDen_eq_sumFin]

theorem dotIntervalUpperCommonDen_eq {n : Nat} (v lo hi : Fin n → Rat) :
    dotIntervalUpperCommonDen v lo hi = dotIntervalUpper v lo hi := by
  simp only [dotIntervalUpperCommonDen, dotIntervalUpper, Linear.sumFinCommonDen_eq_sumFin]

private lemma foldl_pair {α : Type _} (xs : List α) (f g : α → Rat) (a b : Rat) :
    xs.foldl (fun acc x => (acc.1 + f x, acc.2 + g x)) (a, b) =
      (xs.foldl (fun acc x => acc + f x) a, xs.foldl (fun acc x => acc + g x) b) := by
  induction xs generalizing a b with
  | nil =>
      simp
  | cons x xs ih =>
      simp [List.foldl, ih]

theorem dotIntervalLowerUpper2CommonDen_fst {n : Nat} (lo1 hi1 lo2 hi2 : Fin n → Rat) :
    (dotIntervalLowerUpper2CommonDen lo1 hi1 lo2 hi2).1 =
      dotIntervalLower2 lo1 hi1 lo2 hi2 := by
  classical
  have hpair :=
    foldl_pair (xs := List.finRange n)
      (f := fun j => mulIntervalLower (lo1 j) (hi1 j) (lo2 j) (hi2 j))
      (g := fun j => mulIntervalUpper (lo1 j) (hi1 j) (lo2 j) (hi2 j))
      (a := 0) (b := 0)
  have hfst := congrArg Prod.fst hpair
  simpa [dotIntervalLowerUpper2CommonDen, dotIntervalLower2, Linear.foldlFin_eq_foldl,
    Linear.sumFin_eq_list_foldl, Fin.foldl_eq_foldl_finRange] using hfst

theorem dotIntervalLowerUpper2CommonDen_snd {n : Nat} (lo1 hi1 lo2 hi2 : Fin n → Rat) :
    (dotIntervalLowerUpper2CommonDen lo1 hi1 lo2 hi2).2 =
      dotIntervalUpper2 lo1 hi1 lo2 hi2 := by
  classical
  have hpair :=
    foldl_pair (xs := List.finRange n)
      (f := fun j => mulIntervalLower (lo1 j) (hi1 j) (lo2 j) (hi2 j))
      (g := fun j => mulIntervalUpper (lo1 j) (hi1 j) (lo2 j) (hi2 j))
      (a := 0) (b := 0)
  have hsnd := congrArg Prod.snd hpair
  simpa [dotIntervalLowerUpper2CommonDen, dotIntervalUpper2, Linear.foldlFin_eq_foldl,
    Linear.sumFin_eq_list_foldl, Fin.foldl_eq_foldl_finRange] using hsnd

theorem dotIntervalLowerUpper2CommonDen_eq {n : Nat} (lo1 hi1 lo2 hi2 : Fin n → Rat) :
    dotIntervalLowerUpper2CommonDen lo1 hi1 lo2 hi2 =
      (dotIntervalLower2 lo1 hi1 lo2 hi2, dotIntervalUpper2 lo1 hi1 lo2 hi2) := by
  ext <;> simp only [dotIntervalLowerUpper2CommonDen_fst, dotIntervalLowerUpper2CommonDen_snd]

theorem dotIntervalLowerUpperCommonDen_fst {n : Nat} (v lo hi : Fin n → Rat) :
    (dotIntervalLowerUpperCommonDen v lo hi).1 = dotIntervalLowerCommonDen v lo hi := by
  classical
  have hpair :=
    foldl_pair (xs := List.finRange n)
      (f := fun j => if 0 ≤ v j then v j * lo j else v j * hi j)
      (g := fun j => if 0 ≤ v j then v j * hi j else v j * lo j)
      (a := 0) (b := 0)
  have hfst := congrArg Prod.fst hpair
  simpa [dotIntervalLowerUpperCommonDen, dotIntervalLowerCommonDen,
    Linear.foldlFin_eq_foldl, Linear.sumFinCommonDen_eq_sumFin,
    Linear.sumFin_eq_list_foldl, Fin.foldl_eq_foldl_finRange] using hfst

theorem dotIntervalLowerUpperCommonDen_snd {n : Nat} (v lo hi : Fin n → Rat) :
    (dotIntervalLowerUpperCommonDen v lo hi).2 = dotIntervalUpperCommonDen v lo hi := by
  classical
  have hpair :=
    foldl_pair (xs := List.finRange n)
      (f := fun j => if 0 ≤ v j then v j * lo j else v j * hi j)
      (g := fun j => if 0 ≤ v j then v j * hi j else v j * lo j)
      (a := 0) (b := 0)
  have hsnd := congrArg Prod.snd hpair
  simpa [dotIntervalLowerUpperCommonDen, dotIntervalUpperCommonDen,
    Linear.foldlFin_eq_foldl, Linear.sumFinCommonDen_eq_sumFin,
    Linear.sumFin_eq_list_foldl, Fin.foldl_eq_foldl_finRange] using hsnd

/-- Single-pass lower/upper endpoints agree with the common-denominator bounds. -/
theorem dotIntervalLowerUpperCommonDen_eq {n : Nat} (v lo hi : Fin n → Rat) :
    dotIntervalLowerUpperCommonDen v lo hi =
      (dotIntervalLowerCommonDen v lo hi, dotIntervalUpperCommonDen v lo hi) := by
  ext <;> simp only [dotIntervalLowerUpperCommonDen_fst, dotIntervalLowerUpperCommonDen_snd]

private theorem dotIntervalLowerUnnorm_eq {n : Nat} (v lo hi : Fin n → Rat) :
    dotIntervalLowerUnnorm v lo hi = dotIntervalLower v lo hi := rfl

private theorem dotIntervalUpperUnnorm_eq {n : Nat} (v lo hi : Fin n → Rat) :
    dotIntervalUpperUnnorm v lo hi = dotIntervalUpper v lo hi := rfl

/-! Cached endpoints. -/

/-- Cached-array lower interval endpoint for a dot product using normalized rational sums. -/
def dotIntervalLowerCachedRat {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
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

/-- Cached-array upper interval endpoint for a dot product using normalized rational sums. -/
def dotIntervalUpperCachedRat {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
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

theorem dotIntervalLowerCachedRat_eq {n : Nat} (v lo hi : Fin n → Rat) :
    dotIntervalLowerCachedRat v lo hi = dotIntervalLower v lo hi := by
  classical
  simp only [dotIntervalLowerCachedRat, dotIntervalLower, Linear.sumFin_eq_list_foldl,
    Array.getElem_ofFn]

theorem dotIntervalUpperCachedRat_eq {n : Nat} (v lo hi : Fin n → Rat) :
    dotIntervalUpperCachedRat v lo hi = dotIntervalUpper v lo hi := by
  classical
  simp only [dotIntervalUpperCachedRat, dotIntervalUpper, Linear.sumFin_eq_list_foldl,
    Array.getElem_ofFn]

/-! Absolute bounds. -/

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
  simp only [dotIntervalLower, Linear.sumFin_eq_sum_univ, dotProduct]
  refine Finset.sum_le_sum ?_
  intro j _
  by_cases hv : 0 ≤ v j
  · have hmul : v j * lo j ≤ v j * x j := by
      exact mul_le_mul_of_nonneg_left (hlo j) hv
    simpa [hv] using hmul
  · have hv' : v j ≤ 0 := le_of_lt (lt_of_not_ge hv)
    have hmul : v j * hi j ≤ v j * x j := by
      have hmul' : hi j * v j ≤ x j * v j := by
        exact mul_le_mul_of_nonpos_right (hhi j) hv'
      simpa only [mul_comm] using hmul'
    simpa [hv] using hmul

theorem dotProduct_le_dotIntervalUpper {n : Nat} (v lo hi x : Fin n → Rat)
    (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    dotProduct v x ≤ dotIntervalUpper v lo hi := by
  classical
  simp only [dotIntervalUpper, Linear.sumFin_eq_sum_univ, dotProduct]
  refine Finset.sum_le_sum ?_
  intro j _
  by_cases hv : 0 ≤ v j
  · have hmul : v j * x j ≤ v j * hi j := by
      exact mul_le_mul_of_nonneg_left (hhi j) hv
    simpa [hv] using hmul
  · have hv' : v j ≤ 0 := le_of_lt (lt_of_not_ge hv)
    have hmul : v j * x j ≤ v j * lo j := by
      have hmul' : x j * v j ≤ lo j * v j := by
        exact mul_le_mul_of_nonpos_right (hlo j) hv'
      simpa only [mul_comm] using hmul'
    simpa [hv] using hmul

theorem abs_le_max_abs_abs_of_interval {a b x : Rat} (hlo : a ≤ x) (hhi : x ≤ b) :
    |x| ≤ max |a| |b| := by
  exact abs_le_max_abs_abs hlo hhi

/-- Global absolute bound from interval endpoints. -/
def intervalAbsBound {n : Nat} (lo hi : Fin n → Rat) : Rat :=
  Linear.foldlFin n (fun acc i => max acc (max |lo i| |hi i|)) 0

/-- `intervalAbsBound` dominates each endpoint absolute value. -/
theorem max_abs_le_intervalAbsBound {n : Nat} (lo hi : Fin n → Rat) (i : Fin n) :
    max |lo i| |hi i| ≤ intervalAbsBound lo hi := by
  simpa [intervalAbsBound] using
    (foldlFin_max_ge (f := fun j => max |lo j| |hi j|) i)

/-- `intervalAbsBound` bounds any element inside the interval. -/
theorem abs_le_intervalAbsBound {n : Nat} (lo hi x : Fin n → Rat)
    (hlo : ∀ i, lo i ≤ x i) (hhi : ∀ i, x i ≤ hi i) (i : Fin n) :
    |x i| ≤ intervalAbsBound lo hi := by
  have hbound : |x i| ≤ max |lo i| |hi i| :=
    abs_le_max_abs_abs_of_interval (hlo i) (hhi i)
  have hsup : max |lo i| |hi i| ≤ intervalAbsBound lo hi :=
    max_abs_le_intervalAbsBound lo hi i
  exact le_trans hbound hsup

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
  simpa [dotIntervalAbsBound] using habs

/-! Real-valued bounds from rational intervals. -/

lemma mul_between_of_bounds_real {a b x y : Real} (hx : a ≤ x) (hx' : x ≤ b) :
    min (a * y) (b * y) ≤ x * y ∧ x * y ≤ max (a * y) (b * y) := by
  have hab : a ≤ b := le_trans hx hx'
  by_cases hy : 0 ≤ y
  · have hmin : min (a * y) (b * y) = a * y := by
      have hle : a * y ≤ b * y := by
        exact mul_le_mul_of_nonneg_right hab hy
      exact min_eq_left hle
    have hmax : max (a * y) (b * y) = b * y := by
      have hle : a * y ≤ b * y := by
        exact mul_le_mul_of_nonneg_right hab hy
      exact max_eq_right hle
    constructor
    · have hmul : a * y ≤ x * y := by
        exact mul_le_mul_of_nonneg_right hx hy
      simpa [hmin] using hmul
    · have hmul : x * y ≤ b * y := by
        exact mul_le_mul_of_nonneg_right hx' hy
      simpa [hmax] using hmul
  · have hy' : y ≤ 0 := le_of_not_ge hy
    have hmin : min (a * y) (b * y) = b * y := by
      have hle : b * y ≤ a * y := by
        exact mul_le_mul_of_nonpos_right hab hy'
      exact min_eq_right hle
    have hmax : max (a * y) (b * y) = a * y := by
      have hle : b * y ≤ a * y := by
        exact mul_le_mul_of_nonpos_right hab hy'
      exact max_eq_left hle
    constructor
    · have hmul : b * y ≤ x * y := by
        exact mul_le_mul_of_nonpos_right hx' hy'
      simpa [hmin] using hmul
    · have hmul : x * y ≤ a * y := by
        exact mul_le_mul_of_nonpos_right hx hy'
      simpa [hmax] using hmul

lemma mulIntervalLower_le_mul_real {a b c d : Rat} {x y : Real}
    (hx : (a : Real) ≤ x) (hx' : x ≤ (b : Real))
    (hy : (c : Real) ≤ y) (hy' : y ≤ (d : Real)) :
    (mulIntervalLower a b c d : Real) ≤ x * y := by
  have hAy :
      min ((a : Real) * (c : Real)) ((a : Real) * (d : Real)) ≤ (a : Real) * y := by
    have h := mul_between_of_bounds_real (a := (c : Real)) (b := (d : Real)) (x := y)
      (y := (a : Real)) hy hy'
    simpa only [mul_comm] using h.1
  have hBy :
      min ((b : Real) * (c : Real)) ((b : Real) * (d : Real)) ≤ (b : Real) * y := by
    have h := mul_between_of_bounds_real (a := (c : Real)) (b := (d : Real)) (x := y)
      (y := (b : Real)) hy hy'
    simpa only [mul_comm] using h.1
  have hmin :
      min (min ((a : Real) * (c : Real)) ((a : Real) * (d : Real)))
          (min ((b : Real) * (c : Real)) ((b : Real) * (d : Real))) ≤
        min ((a : Real) * y) ((b : Real) * y) := by
    apply le_min
    · exact le_trans (min_le_left _ _) hAy
    · exact le_trans (min_le_right _ _) hBy
  have hxy := (mul_between_of_bounds_real (a := (a : Real)) (b := (b : Real)) (x := x)
    (y := y) hx hx').1
  have hcast :
      (mulIntervalLower a b c d : Real) =
        min (min ((a : Real) * (c : Real)) ((a : Real) * (d : Real)))
          (min ((b : Real) * (c : Real)) ((b : Real) * (d : Real))) := by
    simp only [mulIntervalLower, Rat.cast_min, Rat.cast_mul]
  calc
    (mulIntervalLower a b c d : Real)
        = min (min ((a : Real) * (c : Real)) ((a : Real) * (d : Real)))
            (min ((b : Real) * (c : Real)) ((b : Real) * (d : Real))) := hcast
    _ ≤ min ((a : Real) * y) ((b : Real) * y) := hmin
    _ ≤ x * y := hxy

lemma mul_le_mulIntervalUpper_real {a b c d : Rat} {x y : Real}
    (hx : (a : Real) ≤ x) (hx' : x ≤ (b : Real))
    (hy : (c : Real) ≤ y) (hy' : y ≤ (d : Real)) :
    x * y ≤ (mulIntervalUpper a b c d : Real) := by
  have hAy :
      (a : Real) * y ≤ max ((a : Real) * (c : Real)) ((a : Real) * (d : Real)) := by
    have h := mul_between_of_bounds_real (a := (c : Real)) (b := (d : Real)) (x := y)
      (y := (a : Real)) hy hy'
    simpa only [mul_comm] using h.2
  have hBy :
      (b : Real) * y ≤ max ((b : Real) * (c : Real)) ((b : Real) * (d : Real)) := by
    have h := mul_between_of_bounds_real (a := (c : Real)) (b := (d : Real)) (x := y)
      (y := (b : Real)) hy hy'
    simpa only [mul_comm] using h.2
  have hmax :
      max ((a : Real) * y) ((b : Real) * y) ≤
        max (max ((a : Real) * (c : Real)) ((a : Real) * (d : Real)))
          (max ((b : Real) * (c : Real)) ((b : Real) * (d : Real))) := by
    apply max_le
    · exact le_trans hAy (le_max_left _ _)
    · exact le_trans hBy (le_max_right _ _)
  have hxy := (mul_between_of_bounds_real (a := (a : Real)) (b := (b : Real)) (x := x)
    (y := y) hx hx').2
  have hcast :
      (mulIntervalUpper a b c d : Real) =
        max (max ((a : Real) * (c : Real)) ((a : Real) * (d : Real)))
          (max ((b : Real) * (c : Real)) ((b : Real) * (d : Real))) := by
    simp only [mulIntervalUpper, Rat.cast_max, Rat.cast_mul]
  calc
    x * y ≤ max ((a : Real) * y) ((b : Real) * y) := hxy
    _ ≤ max (max ((a : Real) * (c : Real)) ((a : Real) * (d : Real)))
          (max ((b : Real) * (c : Real)) ((b : Real) * (d : Real))) := hmax
    _ = (mulIntervalUpper a b c d : Real) := hcast.symm

/-- Sum of lower interval products bounds the dot-product sum (Real). -/
private theorem sum_mulIntervalLower_le_sum_mul_real {n : Nat}
    (lo1 hi1 lo2 hi2 : Fin n → Rat) (x y : Fin n → Real)
    (hlo1 : ∀ j, (lo1 j : Real) ≤ x j) (hhi1 : ∀ j, x j ≤ (hi1 j : Real))
    (hlo2 : ∀ j, (lo2 j : Real) ≤ y j) (hhi2 : ∀ j, y j ≤ (hi2 j : Real)) :
    (∑ j, (mulIntervalLower (lo1 j) (hi1 j) (lo2 j) (hi2 j) : Real)) ≤
      ∑ j, x j * y j := by
  refine Finset.sum_le_sum ?_
  intro j _
  exact mulIntervalLower_le_mul_real (hlo1 j) (hhi1 j) (hlo2 j) (hhi2 j)

/-- Sum of products is bounded by the upper interval products (Real). -/
private theorem sum_mul_le_sum_mulIntervalUpper_real {n : Nat}
    (lo1 hi1 lo2 hi2 : Fin n → Rat) (x y : Fin n → Real)
    (hlo1 : ∀ j, (lo1 j : Real) ≤ x j) (hhi1 : ∀ j, x j ≤ (hi1 j : Real))
    (hlo2 : ∀ j, (lo2 j : Real) ≤ y j) (hhi2 : ∀ j, y j ≤ (hi2 j : Real)) :
    ∑ j, x j * y j ≤
      ∑ j, (mulIntervalUpper (lo1 j) (hi1 j) (lo2 j) (hi2 j) : Real) := by
  refine Finset.sum_le_sum ?_
  intro j _
  exact mul_le_mulIntervalUpper_real (hlo1 j) (hhi1 j) (hlo2 j) (hhi2 j)

theorem dotIntervalLower2_le_dotProduct_real {n : Nat} (lo1 hi1 lo2 hi2 : Fin n → Rat)
    (x y : Fin n → Real)
    (hlo1 : ∀ j, (lo1 j : Real) ≤ x j) (hhi1 : ∀ j, x j ≤ (hi1 j : Real))
    (hlo2 : ∀ j, (lo2 j : Real) ≤ y j) (hhi2 : ∀ j, y j ≤ (hi2 j : Real)) :
    (dotIntervalLower2 lo1 hi1 lo2 hi2 : Real) ≤ dotProduct x y := by
  classical
  have hcast :
      (dotIntervalLower2 lo1 hi1 lo2 hi2 : Real) =
        ∑ j, (mulIntervalLower (lo1 j) (hi1 j) (lo2 j) (hi2 j) : Real) := by
    simpa [dotIntervalLower2, ratToReal_def] using
      (Linear.ratToReal_sumFin
        (f := fun j => mulIntervalLower (lo1 j) (hi1 j) (lo2 j) (hi2 j)))
  have hsum :=
    sum_mulIntervalLower_le_sum_mul_real
      (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2) (x := x) (y := y)
      hlo1 hhi1 hlo2 hhi2
  simpa [hcast, dotProduct] using hsum

theorem dotProduct_le_dotIntervalUpper2_real {n : Nat} (lo1 hi1 lo2 hi2 : Fin n → Rat)
    (x y : Fin n → Real)
    (hlo1 : ∀ j, (lo1 j : Real) ≤ x j) (hhi1 : ∀ j, x j ≤ (hi1 j : Real))
    (hlo2 : ∀ j, (lo2 j : Real) ≤ y j) (hhi2 : ∀ j, y j ≤ (hi2 j : Real)) :
    dotProduct x y ≤ (dotIntervalUpper2 lo1 hi1 lo2 hi2 : Real) := by
  classical
  have hcast :
      (dotIntervalUpper2 lo1 hi1 lo2 hi2 : Real) =
        ∑ j, (mulIntervalUpper (lo1 j) (hi1 j) (lo2 j) (hi2 j) : Real) := by
    simpa [dotIntervalUpper2, ratToReal_def] using
      (Linear.ratToReal_sumFin
        (f := fun j => mulIntervalUpper (lo1 j) (hi1 j) (lo2 j) (hi2 j)))
  have hsum :=
    sum_mul_le_sum_mulIntervalUpper_real
      (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2) (x := x) (y := y)
      hlo1 hhi1 hlo2 hhi2
  simpa [hcast, dotProduct] using hsum

theorem dotIntervalLowerUpper2SignSplit_spec_real {n : Nat} (dims : List (Fin n))
    (lo1 hi1 lo2 hi2 : Fin n → Rat) (x y : Fin n → Real)
    (hlo1 : ∀ j, (lo1 j : Real) ≤ x j) (hhi1 : ∀ j, x j ≤ (hi1 j : Real))
    (hlo2 : ∀ j, (lo2 j : Real) ≤ y j) (hhi2 : ∀ j, y j ≤ (hi2 j : Real)) :
    let bounds := dotIntervalLowerUpper2SignSplit dims lo1 hi1 lo2 hi2
    (bounds.1 : Real) ≤ dotProduct x y ∧ dotProduct x y ≤ (bounds.2 : Real) := by
  classical
  induction dims generalizing lo1 hi1 with
  | nil =>
      have hlow :=
        dotIntervalLower2_le_dotProduct_real
          (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
          (x := x) (y := y) hlo1 hhi1 hlo2 hhi2
      have hhigh :=
        dotProduct_le_dotIntervalUpper2_real
          (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
          (x := x) (y := y) hlo1 hhi1 hlo2 hhi2
      simpa only [dotIntervalLowerUpper2SignSplit, dotIntervalLowerUpper2CommonDen_fst,
        dotIntervalLowerUpper2CommonDen_snd] using And.intro hlow hhigh
  | cons i rest ih =>
      by_cases hx : 0 ≤ x i
      · let clamped := clampAt i true lo1 hi1
        let boundsPos := dotIntervalLowerUpper2SignSplit rest clamped.1 clamped.2 lo2 hi2
        let boundsNeg :=
          dotIntervalLowerUpper2SignSplit rest (clampAt i false lo1 hi1).1
            (clampAt i false lo1 hi1).2 lo2 hi2
        have hlo1' : ∀ j, (clamped.1 j : Real) ≤ x j := by
          intro j
          by_cases hji : j = i
          · have hmax : max (0 : Real) (lo1 i : Real) ≤ x i :=
              (max_le_iff).2 ⟨hx, hlo1 i⟩
            simpa [clamped, clampAt, hji, ratToReal_max] using hmax
          · simpa [clamped, clampAt, hji] using hlo1 j
        have hhi1' : ∀ j, x j ≤ (clamped.2 j : Real) := by
          intro j
          simpa [clamped, clampAt] using hhi1 j
        have hpos :=
          ih (lo1 := clamped.1) (hi1 := clamped.2) hlo1' hhi1'
        have hlow : (min boundsPos.1 boundsNeg.1 : Real) ≤ dotProduct x y := by
          exact le_trans (min_le_left _ _) hpos.1
        have hhigh : dotProduct x y ≤ (max boundsPos.2 boundsNeg.2 : Real) := by
          exact le_trans hpos.2 (le_max_left _ _)
        simpa [dotIntervalLowerUpper2SignSplit, boundsPos, boundsNeg, clamped] using
          And.intro hlow hhigh
      · have hxneg : x i ≤ 0 := le_of_lt (lt_of_not_ge hx)
        let clamped := clampAt i false lo1 hi1
        let boundsPos :=
          dotIntervalLowerUpper2SignSplit rest (clampAt i true lo1 hi1).1
            (clampAt i true lo1 hi1).2 lo2 hi2
        let boundsNeg := dotIntervalLowerUpper2SignSplit rest clamped.1 clamped.2 lo2 hi2
        have hlo1' : ∀ j, (clamped.1 j : Real) ≤ x j := by
          intro j
          simpa [clamped, clampAt] using hlo1 j
        have hhi1' : ∀ j, x j ≤ (clamped.2 j : Real) := by
          intro j
          by_cases hji : j = i
          · have hmin : x i ≤ min (0 : Real) (hi1 i : Real) :=
              (le_min_iff).2 ⟨hxneg, hhi1 i⟩
            simpa [clamped, clampAt, hji, ratToReal_min] using hmin
          · simpa [clamped, clampAt, hji] using hhi1 j
        have hneg :=
          ih (lo1 := clamped.1) (hi1 := clamped.2) hlo1' hhi1'
        have hlow : (min boundsPos.1 boundsNeg.1 : Real) ≤ dotProduct x y := by
          exact le_trans (min_le_right _ _) hneg.1
        have hhigh : dotProduct x y ≤ (max boundsPos.2 boundsNeg.2 : Real) := by
          exact le_trans hneg.2 (le_max_right _ _)
        simpa [dotIntervalLowerUpper2SignSplit, boundsPos, boundsNeg, clamped] using
          And.intro hlow hhigh

theorem dotIntervalLowerUpper2SignSplitBoth_spec_real {n : Nat} (dims1 dims2 : List (Fin n))
    (lo1 hi1 lo2 hi2 : Fin n → Rat) (x y : Fin n → Real)
    (hlo1 : ∀ j, (lo1 j : Real) ≤ x j) (hhi1 : ∀ j, x j ≤ (hi1 j : Real))
    (hlo2 : ∀ j, (lo2 j : Real) ≤ y j) (hhi2 : ∀ j, y j ≤ (hi2 j : Real)) :
    let bounds := dotIntervalLowerUpper2SignSplitBoth dims1 dims2 lo1 hi1 lo2 hi2
    (bounds.1 : Real) ≤ dotProduct x y ∧ dotProduct x y ≤ (bounds.2 : Real) := by
  classical
  let bounds1 := dotIntervalLowerUpper2SignSplit dims1 lo1 hi1 lo2 hi2
  let bounds2 := dotIntervalLowerUpper2SignSplit dims2 lo2 hi2 lo1 hi1
  have h1 :=
    dotIntervalLowerUpper2SignSplit_spec_real
      (dims := dims1) (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
      (x := x) (y := y) hlo1 hhi1 hlo2 hhi2
  have h2swap :=
    dotIntervalLowerUpper2SignSplit_spec_real
      (dims := dims2) (lo1 := lo2) (hi1 := hi2) (lo2 := lo1) (hi2 := hi1)
      (x := y) (y := x) hlo2 hhi2 hlo1 hhi1
  have h2 : (bounds2.1 : Real) ≤ dotProduct x y ∧ dotProduct x y ≤ (bounds2.2 : Real) := by
    simpa only [dotProduct_comm] using h2swap
  have hlow' : max (bounds1.1 : Real) (bounds2.1 : Real) ≤ dotProduct x y :=
    (max_le_iff).2 ⟨h1.1, h2.1⟩
  have hhigh' : dotProduct x y ≤ min (bounds1.2 : Real) (bounds2.2 : Real) :=
    (le_min_iff).2 ⟨h1.2, h2.2⟩
  have hlow : ((max bounds1.1 bounds2.1 : Rat) : Real) ≤ dotProduct x y := by
    simpa [ratToReal_max] using hlow'
  have hhigh : dotProduct x y ≤ ((min bounds1.2 bounds2.2 : Rat) : Real) := by
    simpa [ratToReal_min] using hhigh'
  simpa only [dotIntervalLowerUpper2SignSplitBoth, bounds1, bounds2] using And.intro hlow hhigh

theorem dotIntervalLower_le_dotProduct_real {n : Nat} (v lo hi : Fin n → Rat)
    (x : Fin n → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    (dotIntervalLower v lo hi : Real) ≤ dotProduct (fun j => (v j : Real)) x := by
  classical
  have hcast :
      (dotIntervalLower v lo hi : Real) =
        ∑ j, if 0 ≤ v j then (v j : Real) * (lo j : Real) else (v j : Real) * (hi j : Real) := by
    have hcast' :
        ratToReal (dotIntervalLower v lo hi) =
          ∑ j, if 0 ≤ v j then ratToReal (v j) * ratToReal (lo j) else
            ratToReal (v j) * ratToReal (hi j) := by
      simpa [dotIntervalLower, ratToReal_if, ratToReal_mul] using
        (Linear.ratToReal_sumFin
          (f := fun j => if 0 ≤ v j then v j * lo j else v j * hi j))
    simpa [ratToReal_def] using hcast'
  have hsum :
      (∑ j, if 0 ≤ v j then (v j : Real) * (lo j : Real) else (v j : Real) * (hi j : Real)) ≤
        ∑ j, (v j : Real) * x j := by
    refine Finset.sum_le_sum ?_
    intro j _
    by_cases hv : 0 ≤ v j
    · have hv' : (0 : Real) ≤ (v j : Real) := by
        simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg hv
      have hmul : (v j : Real) * (lo j : Real) ≤ (v j : Real) * x j := by
        exact mul_le_mul_of_nonneg_left (hlo j) hv'
      simpa [hv] using hmul
    · have hv' : (v j : Real) ≤ 0 := by
        simpa [ratToReal_def] using (ratToReal_nonpos_iff (x := v j)).2 (le_of_not_ge hv)
      have hmul : (v j : Real) * (hi j : Real) ≤ (v j : Real) * x j := by
        exact mul_le_mul_of_nonpos_left (hhi j) hv'
      simpa [hv] using hmul
  simpa [hcast, dotProduct] using hsum

theorem dotIntervalLower_le_dotProduct_real_add {n : Nat}
    (v lo hi : Fin n → Rat) (x : Fin n → Real) (b : Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    (dotIntervalLower v lo hi : Real) + b ≤
      dotProduct (fun j => (v j : Real)) x + b := by
  have hlow :=
    dotIntervalLower_le_dotProduct_real (v := v) (lo := lo) (hi := hi)
      (x := x) hlo hhi
  simpa [add_comm] using add_le_add_left hlow b

theorem dotProduct_le_dotIntervalUpper_real {n : Nat} (v lo hi : Fin n → Rat)
    (x : Fin n → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    dotProduct (fun j => (v j : Real)) x ≤ (dotIntervalUpper v lo hi : Real) := by
  classical
  have hcast :
      (dotIntervalUpper v lo hi : Real) =
        ∑ j, if 0 ≤ v j then (v j : Real) * (hi j : Real) else (v j : Real) * (lo j : Real) := by
    have hcast' :
        ratToReal (dotIntervalUpper v lo hi) =
          ∑ j, if 0 ≤ v j then ratToReal (v j) * ratToReal (hi j) else
            ratToReal (v j) * ratToReal (lo j) := by
      simpa [dotIntervalUpper, ratToReal_if, ratToReal_mul] using
        (Linear.ratToReal_sumFin
          (f := fun j => if 0 ≤ v j then v j * hi j else v j * lo j))
    simpa [ratToReal_def] using hcast'
  have hsum :
      ∑ j, (v j : Real) * x j ≤
        ∑ j, if 0 ≤ v j then (v j : Real) * (hi j : Real) else (v j : Real) * (lo j : Real) := by
    refine Finset.sum_le_sum ?_
    intro j _
    by_cases hv : 0 ≤ v j
    · have hv' : (0 : Real) ≤ (v j : Real) := by
        simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg hv
      have hmul : (v j : Real) * x j ≤ (v j : Real) * (hi j : Real) := by
        exact mul_le_mul_of_nonneg_left (hhi j) hv'
      simpa [hv] using hmul
    · have hv' : (v j : Real) ≤ 0 := by
        simpa [ratToReal_def] using (ratToReal_nonpos_iff (x := v j)).2 (le_of_not_ge hv)
      have hmul : (v j : Real) * x j ≤ (v j : Real) * (lo j : Real) := by
        exact mul_le_mul_of_nonpos_left (hlo j) hv'
      simpa [hv] using hmul
  simpa [hcast, dotProduct] using hsum

theorem dotProduct_le_dotIntervalUpper_real_add {n : Nat}
    (v lo hi : Fin n → Rat) (x : Fin n → Real) (b : Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    dotProduct (fun j => (v j : Real)) x + b ≤
      (dotIntervalUpper v lo hi : Real) + b := by
  have hhigh :=
    dotProduct_le_dotIntervalUpper_real (v := v) (lo := lo) (hi := hi)
      (x := x) hlo hhi
  simpa [add_comm] using add_le_add_left hhigh b

theorem abs_le_max_abs_abs_of_interval_real {a b x : Real} (hlo : a ≤ x) (hhi : x ≤ b) :
    |x| ≤ max |a| |b| := by
  exact abs_le_max_abs_abs hlo hhi

/-- `intervalAbsBound` controls real-valued coordinates inside a rational interval. -/
theorem abs_le_intervalAbsBound_real {n : Nat} (lo hi : Fin n → Rat) (x : Fin n → Real)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) (i : Fin n) :
    |x i| ≤ (intervalAbsBound lo hi : Real) := by
  have hbound : |x i| ≤ max |(lo i : Real)| |(hi i : Real)| :=
    abs_le_max_abs_abs_of_interval_real (hlo i) (hhi i)
  have hsup : max |lo i| |hi i| ≤ intervalAbsBound lo hi :=
    max_abs_le_intervalAbsBound lo hi i
  have hsup_real :
      max |(lo i : Real)| |(hi i : Real)| ≤ (intervalAbsBound lo hi : Real) := by
    refine max_le_iff.mpr ?_
    constructor
    · have hleft : |lo i| ≤ intervalAbsBound lo hi := by
        exact le_trans (le_max_left _ _) (max_abs_le_intervalAbsBound lo hi i)
      simpa [ratToReal_def] using ratToReal_abs_le_of_le hleft
    · have hright : |hi i| ≤ intervalAbsBound lo hi := by
        exact le_trans (le_max_right _ _) (max_abs_le_intervalAbsBound lo hi i)
      simpa [ratToReal_def] using ratToReal_abs_le_of_le hright
  exact le_trans hbound hsup_real

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
  simpa [dotIntervalAbsBound] using habs

/-! Matrix-vector interval bounds. -/

theorem mulVecIntervalLower_le_mulVec {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (lo hi x : Fin n → Rat) (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    ∀ i, mulVecIntervalLower W lo hi i ≤ Matrix.mulVec W x i := by
  intro i
  simpa [mulVecIntervalLower, Matrix.mulVec, dotProduct] using
    (dotIntervalLower_le_dotProduct (v := fun j => W i j) lo hi x hlo hhi)

theorem mulVec_le_mulVecIntervalUpper {m n : Nat} (W : Matrix (Fin m) (Fin n) Rat)
    (lo hi x : Fin n → Rat) (hlo : ∀ j, lo j ≤ x j) (hhi : ∀ j, x j ≤ hi j) :
    ∀ i, Matrix.mulVec W x i ≤ mulVecIntervalUpper W lo hi i := by
  intro i
  simpa [mulVecIntervalUpper, Matrix.mulVec, dotProduct] using
    (dotProduct_le_dotIntervalUpper (v := fun j => W i j) lo hi x hlo hhi)

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

end Bounds

end Sound

end Nfp
