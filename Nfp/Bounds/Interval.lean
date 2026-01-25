-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Init.Data.Array.FinRange
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Real.Basic
public import Nfp.Core.Basic
public import Nfp.Linear.FinFold

/-!
Interval bounds for dot products and absolute-value envelopes.
-/

public section

namespace Nfp

namespace Bounds

open scoped BigOperators

/-- Lower bound for a dot product given per-coordinate intervals. -/
def dotIntervalLower {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun i => if 0 ≤ v i then v i * lo i else v i * hi i)

/-- Upper bound for a dot product given per-coordinate intervals. -/
def dotIntervalUpper {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun i => if 0 ≤ v i then v i * hi i else v i * lo i)

/-- Lower/upper bounds for a dot product, computed in one pass. -/
def dotIntervalBoundsFast {n : Nat} (v lo hi : Fin n → Rat) : Rat × Rat :=
  Linear.foldlFin n (fun acc i =>
    (acc.1 + (if 0 ≤ v i then v i * lo i else v i * hi i),
      acc.2 + (if 0 ≤ v i then v i * hi i else v i * lo i))) (0, 0)

/-- Array-based bounds for a dot product, computed in one pass. -/
def dotIntervalBoundsFastArr (n : Nat) (v lo hi : Array Rat)
    (hv : v.size = n) (hlo : lo.size = n) (hhi : hi.size = n) : Rat × Rat :=
  (Array.finRange n).foldl (fun acc i =>
    let vi := v[i.1]'(Nat.lt_of_lt_of_eq i.isLt hv.symm)
    let loi := lo[i.1]'(Nat.lt_of_lt_of_eq i.isLt hlo.symm)
    let hii := hi[i.1]'(Nat.lt_of_lt_of_eq i.isLt hhi.symm)
    (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
      acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))) (0, 0)

/-- `dotIntervalBoundsFastArr` matches `dotIntervalBoundsFast` on `Array.ofFn` inputs. -/
theorem dotIntervalBoundsFastArr_ofFn {n : Nat}
    (v lo hi : Fin n → Rat) :
    dotIntervalBoundsFastArr n (Array.ofFn v) (Array.ofFn lo) (Array.ofFn hi)
        (by simp) (by simp) (by simp) =
      dotIntervalBoundsFast v lo hi := by
  classical
  have hv : (Array.ofFn v).size = n := by simp
  have hlo : (Array.ofFn lo).size = n := by simp
  have hhi : (Array.ofFn hi).size = n := by simp
  -- Convert the array fold to a list fold, then use the `Fin` fold lemma.
  have hfold :
      (Array.finRange n).foldl (fun acc i =>
        let vi := (Array.ofFn v)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hv.symm)
        let loi := (Array.ofFn lo)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hlo.symm)
        let hii := (Array.ofFn hi)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hhi.symm)
        (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
          acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))) (0, 0)
        =
      (Array.finRange n).toList.foldl (fun acc i =>
        let vi := (Array.ofFn v)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hv.symm)
        let loi := (Array.ofFn lo)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hlo.symm)
        let hii := (Array.ofFn hi)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hhi.symm)
        (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
          acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))) (0, 0) := by
    simp
  -- Simplify the list and array indices to the original `Fin` functions.
  have hlist :
      (Array.finRange n).toList =
        List.finRange n := by
    apply List.ext_getElem
    · simp [Array.size_finRange, List.length_finRange]
    · intro i hi₁ hi₂
      have hi : i < (Array.finRange n).size := by
        simpa [Array.size_finRange] using hi₁
      have hi' : i < (List.finRange n).length := by
        simpa [List.length_finRange] using hi₂
      simp [Array.getElem_toList, Array.getElem_finRange, List.getElem_finRange]
  have hget_v : ∀ i : Fin n,
      (Array.ofFn v)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hv.symm) = v i := by
    intro i
    simp
  have hget_lo : ∀ i : Fin n,
      (Array.ofFn lo)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hlo.symm) = lo i := by
    intro i
    simp
  have hget_hi : ∀ i : Fin n,
      (Array.ofFn hi)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hhi.symm) = hi i := by
    intro i
    simp
  have hpair : ∀ acc : Rat × Rat, ∀ i : Fin n,
      (if 0 ≤ v i then (acc.1 + v i * lo i, acc.2 + v i * hi i)
        else (acc.1 + v i * hi i, acc.2 + v i * lo i)) =
        (acc.1 + (if 0 ≤ v i then v i * lo i else v i * hi i),
          acc.2 + (if 0 ≤ v i then v i * hi i else v i * lo i)) := by
    intro acc i
    by_cases h : 0 ≤ v i <;> simp [h]
  have hfold' :
      (Array.finRange n).toList.foldl (fun acc i =>
        let vi := (Array.ofFn v)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hv.symm)
        let loi := (Array.ofFn lo)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hlo.symm)
        let hii := (Array.ofFn hi)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hhi.symm)
        (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
          acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))) (0, 0)
      =
      (List.finRange n).foldl (fun acc i =>
        if 0 ≤ v i then (acc.1 + v i * lo i, acc.2 + v i * hi i)
        else (acc.1 + v i * hi i, acc.2 + v i * lo i)) (0, 0) := by
    simp [hlist, hget_v, hget_lo, hget_hi, hpair]
  have hdef :
      dotIntervalBoundsFastArr n (Array.ofFn v) (Array.ofFn lo) (Array.ofFn hi)
          (by simp) (by simp) (by simp)
        =
        (Array.finRange n).foldl (fun acc i =>
          let vi := (Array.ofFn v)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hv.symm)
          let loi := (Array.ofFn lo)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hlo.symm)
          let hii := (Array.ofFn hi)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hhi.symm)
          (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
            acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))) (0, 0) := by
    dsimp [dotIntervalBoundsFastArr]
  calc
    dotIntervalBoundsFastArr n (Array.ofFn v) (Array.ofFn lo) (Array.ofFn hi)
        (by simp) (by simp) (by simp)
        = (Array.finRange n).foldl (fun acc i =>
            let vi := (Array.ofFn v)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hv.symm)
            let loi := (Array.ofFn lo)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hlo.symm)
            let hii := (Array.ofFn hi)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hhi.symm)
            (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
              acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))) (0, 0) := hdef
    _ = (Array.finRange n).toList.foldl (fun acc i =>
          let vi := (Array.ofFn v)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hv.symm)
          let loi := (Array.ofFn lo)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hlo.symm)
          let hii := (Array.ofFn hi)[i.1]'(Nat.lt_of_lt_of_eq i.isLt hhi.symm)
          (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
            acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))) (0, 0) := hfold
    _ = (List.finRange n).foldl (fun acc i =>
          if 0 ≤ v i then (acc.1 + v i * lo i, acc.2 + v i * hi i)
          else (acc.1 + v i * hi i, acc.2 + v i * lo i)) (0, 0) := hfold'
    _ = dotIntervalBoundsFast v lo hi := by
          simp [dotIntervalBoundsFast, Linear.foldlFin_eq_foldl,
            Fin.foldl_eq_foldl_finRange, hpair]

/-- `dotIntervalBoundsFast` agrees with `dotIntervalLower` on the first component. -/
theorem dotIntervalBoundsFast_fst {n : Nat} (v lo hi : Fin n → Rat) :
    (dotIntervalBoundsFast v lo hi).1 = dotIntervalLower v lo hi := by
  classical
  let g1 : Rat → Fin n → Rat := fun acc i =>
    acc + (if 0 ≤ v i then v i * lo i else v i * hi i)
  let g2 : Rat → Fin n → Rat := fun acc i =>
    acc + (if 0 ≤ v i then v i * hi i else v i * lo i)
  let f : Rat × Rat → Fin n → Rat × Rat := fun acc i =>
    (g1 acc.1 i, g2 acc.2 i)
  have hfold : ∀ xs : List (Fin n), ∀ acc1 acc2,
      (List.foldl f (acc1, acc2) xs).1 = List.foldl g1 acc1 xs := by
    intro xs; induction xs with
    | nil => intro acc1 acc2; simp [f, g1, g2]
    | cons x xs ih =>
        intro acc1 acc2
        simp [List.foldl, f, g1, g2, ih]
  have h1 :
      (dotIntervalBoundsFast v lo hi).1 =
        (List.foldl f (0, 0) (List.finRange n)).1 := by
    simp [dotIntervalBoundsFast, f, g1, g2, Linear.foldlFin_eq_foldl,
      Fin.foldl_eq_foldl_finRange]
  have h2 : dotIntervalLower v lo hi = List.foldl g1 0 (List.finRange n) := by
    simp [dotIntervalLower, Linear.sumFin_eq_list_foldl, g1]
  calc
    (dotIntervalBoundsFast v lo hi).1
        = (List.foldl f (0, 0) (List.finRange n)).1 := h1
    _ = List.foldl g1 0 (List.finRange n) := hfold _ 0 0
    _ = dotIntervalLower v lo hi := by
      simp [h2]

/-- `dotIntervalBoundsFast` agrees with `dotIntervalUpper` on the second component. -/
theorem dotIntervalBoundsFast_snd {n : Nat} (v lo hi : Fin n → Rat) :
    (dotIntervalBoundsFast v lo hi).2 = dotIntervalUpper v lo hi := by
  classical
  let g1 : Rat → Fin n → Rat := fun acc i =>
    acc + (if 0 ≤ v i then v i * lo i else v i * hi i)
  let g2 : Rat → Fin n → Rat := fun acc i =>
    acc + (if 0 ≤ v i then v i * hi i else v i * lo i)
  let f : Rat × Rat → Fin n → Rat × Rat := fun acc i =>
    (g1 acc.1 i, g2 acc.2 i)
  have hfold : ∀ xs : List (Fin n), ∀ acc1 acc2,
      (List.foldl f (acc1, acc2) xs).2 = List.foldl g2 acc2 xs := by
    intro xs; induction xs with
    | nil => intro acc1 acc2; simp [f, g1, g2]
    | cons x xs ih =>
        intro acc1 acc2
        simp [List.foldl, f, g1, g2, ih]
  have h1 :
      (dotIntervalBoundsFast v lo hi).2 =
        (List.foldl f (0, 0) (List.finRange n)).2 := by
    simp [dotIntervalBoundsFast, f, g1, g2, Linear.foldlFin_eq_foldl,
      Fin.foldl_eq_foldl_finRange]
  have h2 : dotIntervalUpper v lo hi = List.foldl g2 0 (List.finRange n) := by
    simp [dotIntervalUpper, Linear.sumFin_eq_list_foldl, g2]
  calc
    (dotIntervalBoundsFast v lo hi).2
        = (List.foldl f (0, 0) (List.finRange n)).2 := h1
    _ = List.foldl g2 0 (List.finRange n) := hfold _ 0 0
    _ = dotIntervalUpper v lo hi := by
      simp [h2]

/-- Dot products over interval inputs are bounded below by `dotIntervalLower`. -/
theorem dotIntervalLower_le_dotProduct_real {n : Nat}
    (v lo hi : Fin n → Rat) (x : Fin n → Real)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) :
    (dotIntervalLower v lo hi : Real) ≤ dotProduct (fun i => (v i : Real)) x := by
  classical
  have hterm : ∀ i,
      (if 0 ≤ v i then (v i : Real) * (lo i : Real) else (v i : Real) * (hi i : Real))
        ≤ (v i : Real) * x i := by
    intro i
    by_cases h : 0 ≤ v i
    · have hv : 0 ≤ (v i : Real) := by
        simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg h
      have hmul : (v i : Real) * (lo i : Real) ≤ (v i : Real) * x i :=
        mul_le_mul_of_nonneg_left (hlo i) hv
      simpa [h] using hmul
    · have hv' : v i ≤ 0 := le_of_lt (lt_of_not_ge h)
      have hv : (v i : Real) ≤ 0 := by
        simpa [ratToReal_nonpos_iff] using hv'
      have hmul : (v i : Real) * (hi i : Real) ≤ (v i : Real) * x i :=
        mul_le_mul_of_nonpos_left (hhi i) hv
      simpa [h] using hmul
  have hsum :
      ∑ i, (if 0 ≤ v i then (v i : Real) * (lo i : Real) else (v i : Real) * (hi i : Real)) ≤
        ∑ i, (v i : Real) * x i := by
    refine Finset.sum_le_sum ?_
    intro i _
    exact hterm i
  have hcast : (dotIntervalLower v lo hi : Real) =
      ∑ i, (if 0 ≤ v i then (v i : Real) * (lo i : Real) else (v i : Real) * (hi i : Real)) := by
    classical
    have hcast' :
        ratToReal (Linear.sumFin n (fun i => if 0 ≤ v i then v i * lo i else v i * hi i)) =
          ∑ i, ratToReal (if 0 ≤ v i then v i * lo i else v i * hi i) := by
      simpa using
        (Linear.ratToReal_sumFin (n := n)
          (f := fun i => if 0 ≤ v i then v i * lo i else v i * hi i))
    have hcast'' :
        ratToReal (Linear.sumFin n (fun i => if 0 ≤ v i then v i * lo i else v i * hi i)) =
          ∑ i, if 0 ≤ v i then ratToReal (v i * lo i) else ratToReal (v i * hi i) := by
      simpa [ratToReal_if] using hcast'
    have hcast''' :
        ratToReal (Linear.sumFin n (fun i => if 0 ≤ v i then v i * lo i else v i * hi i)) =
          ∑ i, if 0 ≤ v i then ratToReal (v i) * ratToReal (lo i)
            else ratToReal (v i) * ratToReal (hi i) := by
      simpa [ratToReal_mul] using hcast''
    have hcast'''' :
        ratToReal (Linear.sumFin n (fun i => if 0 ≤ v i then v i * lo i else v i * hi i)) =
          ∑ i,
            (if 0 ≤ v i then (v i : Real) * (lo i : Real) else (v i : Real) * (hi i : Real)) := by
      simpa [ratToReal_def] using hcast'''
    simpa [dotIntervalLower, ratToReal_def] using hcast''''
  have hdot : dotProduct (fun i => (v i : Real)) x = ∑ i, (v i : Real) * x i := by
    simp [dotProduct]
  simpa [hcast, hdot] using hsum

/-- Dot products over interval inputs are bounded above by `dotIntervalUpper`. -/
theorem dotProduct_le_dotIntervalUpper_real {n : Nat}
    (v lo hi : Fin n → Rat) (x : Fin n → Real)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) :
    dotProduct (fun i => (v i : Real)) x ≤ (dotIntervalUpper v lo hi : Real) := by
  classical
  have hterm : ∀ i,
      (v i : Real) * x i ≤
        (if 0 ≤ v i then (v i : Real) * (hi i : Real) else (v i : Real) * (lo i : Real)) := by
    intro i
    by_cases h : 0 ≤ v i
    · have hv : 0 ≤ (v i : Real) := by
        simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg h
      have hmul : (v i : Real) * x i ≤ (v i : Real) * (hi i : Real) :=
        mul_le_mul_of_nonneg_left (hhi i) hv
      simpa [h] using hmul
    · have hv' : v i ≤ 0 := le_of_lt (lt_of_not_ge h)
      have hv : (v i : Real) ≤ 0 := by
        simpa [ratToReal_nonpos_iff] using hv'
      have hmul : (v i : Real) * x i ≤ (v i : Real) * (lo i : Real) :=
        mul_le_mul_of_nonpos_left (hlo i) hv
      simpa [h] using hmul
  have hsum :
      ∑ i, (v i : Real) * x i ≤
        ∑ i, (if 0 ≤ v i then (v i : Real) * (hi i : Real) else (v i : Real) * (lo i : Real)) := by
    refine Finset.sum_le_sum ?_
    intro i _
    exact hterm i
  have hcast : (dotIntervalUpper v lo hi : Real) =
      ∑ i, (if 0 ≤ v i then (v i : Real) * (hi i : Real) else (v i : Real) * (lo i : Real)) := by
    classical
    have hcast' :
        ratToReal (Linear.sumFin n (fun i => if 0 ≤ v i then v i * hi i else v i * lo i)) =
          ∑ i, ratToReal (if 0 ≤ v i then v i * hi i else v i * lo i) := by
      simpa using
        (Linear.ratToReal_sumFin (n := n)
          (f := fun i => if 0 ≤ v i then v i * hi i else v i * lo i))
    have hcast'' :
        ratToReal (Linear.sumFin n (fun i => if 0 ≤ v i then v i * hi i else v i * lo i)) =
          ∑ i, if 0 ≤ v i then ratToReal (v i * hi i) else ratToReal (v i * lo i) := by
      simpa [ratToReal_if] using hcast'
    have hcast''' :
        ratToReal (Linear.sumFin n (fun i => if 0 ≤ v i then v i * hi i else v i * lo i)) =
          ∑ i, if 0 ≤ v i then ratToReal (v i) * ratToReal (hi i)
            else ratToReal (v i) * ratToReal (lo i) := by
      simpa [ratToReal_mul] using hcast''
    have hcast'''' :
        ratToReal (Linear.sumFin n (fun i => if 0 ≤ v i then v i * hi i else v i * lo i)) =
          ∑ i,
            (if 0 ≤ v i then (v i : Real) * (hi i : Real) else (v i : Real) * (lo i : Real)) := by
      simpa [ratToReal_def] using hcast'''
    simpa [dotIntervalUpper, ratToReal_def] using hcast''''
  have hdot : dotProduct (fun i => (v i : Real)) x = ∑ i, (v i : Real) * x i := by
    simp [dotProduct]
  simpa [hcast, hdot] using hsum

/-- Dot products plus a bias are bounded below by `dotIntervalLower` plus the bias. -/
theorem dotIntervalLower_le_dotProduct_real_add {n : Nat}
    (v lo hi : Fin n → Rat) (x : Fin n → Real) (b : Real)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) :
    (dotIntervalLower v lo hi : Real) + b ≤ dotProduct (fun i => (v i : Real)) x + b := by
  simpa [add_comm] using
    add_le_add_left (dotIntervalLower_le_dotProduct_real v lo hi x hlo hhi) b

/-- Dot products plus a bias are bounded above by `dotIntervalUpper` plus the bias. -/
theorem dotProduct_le_dotIntervalUpper_real_add {n : Nat}
    (v lo hi : Fin n → Rat) (x : Fin n → Real) (b : Real)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) :
    dotProduct (fun i => (v i : Real)) x + b ≤ (dotIntervalUpper v lo hi : Real) + b := by
  simpa [add_comm] using
    add_le_add_left (dotProduct_le_dotIntervalUpper_real v lo hi x hlo hhi) b

/-- Lower bound for a product over two intervals. -/
def intervalMulLower (alo ahi blo bhi : Rat) : Rat :=
  min (min (alo * blo) (alo * bhi)) (min (ahi * blo) (ahi * bhi))

/-- Upper bound for a product over two intervals. -/
def intervalMulUpper (alo ahi blo bhi : Rat) : Rat :=
  max (max (alo * blo) (alo * bhi)) (max (ahi * blo) (ahi * bhi))

private theorem min_mul_left_le_mul_real {a c d : Rat} {y : Real}
    (hylo : (c : Real) ≤ y) (hyhi : y ≤ (d : Real)) :
    (min (a * c) (a * d) : Real) ≤ (a : Real) * y := by
  classical
  by_cases ha : 0 ≤ a
  · have ha' : 0 ≤ (a : Real) := by
      simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg ha
    have hmul : (a : Real) * (c : Real) ≤ (a : Real) * y :=
      mul_le_mul_of_nonneg_left hylo ha'
    have hmin : (min (a * c) (a * d) : Real) ≤ (a : Real) * (c : Real) := by
      exact min_le_left _ _
    exact le_trans hmin hmul
  · have ha_rat : a ≤ 0 := le_of_lt (lt_of_not_ge ha)
    have ha' : (a : Real) ≤ 0 := by
      have : ratToReal a ≤ 0 := (ratToReal_nonpos_iff).2 ha_rat
      simpa [ratToReal_def] using this
    have hmul : (a : Real) * (d : Real) ≤ (a : Real) * y :=
      mul_le_mul_of_nonpos_left hyhi ha'
    have hmin : (min (a * c) (a * d) : Real) ≤ (a : Real) * (d : Real) := by
      exact min_le_right _ _
    exact le_trans hmin hmul

private theorem mul_left_le_max_mul_real {a c d : Rat} {y : Real}
    (hylo : (c : Real) ≤ y) (hyhi : y ≤ (d : Real)) :
    (a : Real) * y ≤ (max (a * c) (a * d) : Real) := by
  classical
  by_cases ha : 0 ≤ a
  · have ha' : 0 ≤ (a : Real) := by
      simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg ha
    have hmul : (a : Real) * y ≤ (a : Real) * (d : Real) :=
      mul_le_mul_of_nonneg_left hyhi ha'
    have hmax : (a : Real) * (d : Real) ≤ (max (a * c) (a * d) : Real) := by
      exact le_max_right _ _
    exact le_trans hmul hmax
  · have ha_rat : a ≤ 0 := le_of_lt (lt_of_not_ge ha)
    have ha' : (a : Real) ≤ 0 := by
      have : ratToReal a ≤ 0 := (ratToReal_nonpos_iff).2 ha_rat
      simpa [ratToReal_def] using this
    have hmul : (a : Real) * y ≤ (a : Real) * (c : Real) :=
      mul_le_mul_of_nonpos_left hylo ha'
    have hmax : (a : Real) * (c : Real) ≤ (max (a * c) (a * d) : Real) := by
      exact le_max_left _ _
    exact le_trans hmul hmax

/-- Products over intervals are bounded below by `intervalMulLower`. -/
theorem intervalMulLower_le_mul_real {alo ahi blo bhi : Rat} {x y : Real}
    (hxlo : (alo : Real) ≤ x) (hxhi : x ≤ (ahi : Real))
    (hylo : (blo : Real) ≤ y) (hyhi : y ≤ (bhi : Real)) :
    (intervalMulLower alo ahi blo bhi : Real) ≤ x * y := by
  classical
  have hmin_a : (min (alo * blo) (alo * bhi) : Real) ≤ (alo : Real) * y :=
    min_mul_left_le_mul_real (a := alo) (c := blo) (d := bhi) hylo hyhi
  have hmin_b : (min (ahi * blo) (ahi * bhi) : Real) ≤ (ahi : Real) * y :=
    min_mul_left_le_mul_real (a := ahi) (c := blo) (d := bhi) hylo hyhi
  have hleft : (intervalMulLower alo ahi blo bhi : Real) ≤ (alo : Real) * y := by
    have hmin : (intervalMulLower alo ahi blo bhi : Real) ≤
        (min (alo * blo) (alo * bhi) : Real) := by
      simp [intervalMulLower]
    exact le_trans hmin hmin_a
  have hright : (intervalMulLower alo ahi blo bhi : Real) ≤ (ahi : Real) * y := by
    have hmin : (intervalMulLower alo ahi blo bhi : Real) ≤
        (min (ahi * blo) (ahi * bhi) : Real) := by
      simp [intervalMulLower]
    exact le_trans hmin hmin_b
  have hmin_xy : min ((alo : Real) * y) ((ahi : Real) * y) ≤ x * y := by
    by_cases hy : 0 ≤ y
    · have hmul : (alo : Real) * y ≤ x * y :=
        mul_le_mul_of_nonneg_right hxlo hy
      exact le_trans (min_le_left _ _) hmul
    · have hy' : y ≤ 0 := le_of_not_ge hy
      have hmul : (ahi : Real) * y ≤ x * y :=
        mul_le_mul_of_nonpos_right hxhi hy'
      exact le_trans (min_le_right _ _) hmul
  have hmin' : (intervalMulLower alo ahi blo bhi : Real) ≤
      min ((alo : Real) * y) ((ahi : Real) * y) :=
    le_min hleft hright
  exact le_trans hmin' hmin_xy

/-- Products over intervals are bounded above by `intervalMulUpper`. -/
theorem mul_le_intervalMulUpper_real {alo ahi blo bhi : Rat} {x y : Real}
    (hxlo : (alo : Real) ≤ x) (hxhi : x ≤ (ahi : Real))
    (hylo : (blo : Real) ≤ y) (hyhi : y ≤ (bhi : Real)) :
    x * y ≤ (intervalMulUpper alo ahi blo bhi : Real) := by
  classical
  have hmax_a : (alo : Real) * y ≤ (max (alo * blo) (alo * bhi) : Real) :=
    mul_left_le_max_mul_real (a := alo) (c := blo) (d := bhi) hylo hyhi
  have hmax_b : (ahi : Real) * y ≤ (max (ahi * blo) (ahi * bhi) : Real) :=
    mul_left_le_max_mul_real (a := ahi) (c := blo) (d := bhi) hylo hyhi
  have hleft : (alo : Real) * y ≤ (intervalMulUpper alo ahi blo bhi : Real) := by
    have hmax : (max (alo * blo) (alo * bhi) : Real) ≤
        (intervalMulUpper alo ahi blo bhi : Real) := by
      simp [intervalMulUpper]
    exact le_trans hmax_a hmax
  have hright : (ahi : Real) * y ≤ (intervalMulUpper alo ahi blo bhi : Real) := by
    have hmax : (max (ahi * blo) (ahi * bhi) : Real) ≤
        (intervalMulUpper alo ahi blo bhi : Real) := by
      simp [intervalMulUpper]
    exact le_trans hmax_b hmax
  have hmax_xy : x * y ≤ max ((alo : Real) * y) ((ahi : Real) * y) := by
    by_cases hy : 0 ≤ y
    · have hmul : x * y ≤ (ahi : Real) * y :=
        mul_le_mul_of_nonneg_right hxhi hy
      exact le_trans hmul (le_max_right _ _)
    · have hy' : y ≤ 0 := le_of_not_ge hy
      have hmul : x * y ≤ (alo : Real) * y :=
        mul_le_mul_of_nonpos_right hxlo hy'
      exact le_trans hmul (le_max_left _ _)
  have hmax' : max ((alo : Real) * y) ((ahi : Real) * y) ≤
      (intervalMulUpper alo ahi blo bhi : Real) :=
    max_le hleft hright
  exact le_trans hmax_xy hmax'

/-- Lower bound for dot products of two interval vectors. -/
def dotIntervalMulLower {n : Nat} (lo₁ hi₁ lo₂ hi₂ : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun i => intervalMulLower (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i))

/-- Upper bound for dot products of two interval vectors. -/
def dotIntervalMulUpper {n : Nat} (lo₁ hi₁ lo₂ hi₂ : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun i => intervalMulUpper (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i))

/-- Dot products over interval inputs are bounded below by `dotIntervalMulLower`. -/
theorem dotIntervalMulLower_le_dotProduct_real {n : Nat}
    (lo₁ hi₁ lo₂ hi₂ : Fin n → Rat) (x y : Fin n → Real)
    (hlo₁ : ∀ i, (lo₁ i : Real) ≤ x i) (hhi₁ : ∀ i, x i ≤ (hi₁ i : Real))
    (hlo₂ : ∀ i, (lo₂ i : Real) ≤ y i) (hhi₂ : ∀ i, y i ≤ (hi₂ i : Real)) :
    (dotIntervalMulLower lo₁ hi₁ lo₂ hi₂ : Real) ≤ dotProduct x y := by
  classical
  have hterm : ∀ i,
      (intervalMulLower (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i) : Real) ≤ x i * y i := by
    intro i
    exact intervalMulLower_le_mul_real (hxlo := hlo₁ i) (hxhi := hhi₁ i)
      (hylo := hlo₂ i) (hyhi := hhi₂ i)
  have hsum :
      ∑ i, (intervalMulLower (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i) : Real) ≤
        ∑ i, x i * y i := by
    refine Finset.sum_le_sum ?_
    intro i _
    exact hterm i
  have hcast : (dotIntervalMulLower lo₁ hi₁ lo₂ hi₂ : Real) =
      ∑ i, (intervalMulLower (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i) : Real) := by
    classical
    have hcast' :
        ratToReal (Linear.sumFin n
          (fun i => intervalMulLower (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i))) =
          ∑ i, ratToReal (intervalMulLower (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i)) := by
      simpa using
        (Linear.ratToReal_sumFin (n := n)
          (f := fun i => intervalMulLower (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i)))
    simpa [dotIntervalMulLower, ratToReal_def] using hcast'
  have hdot : dotProduct x y = ∑ i, x i * y i := by
    simp [dotProduct]
  simpa [hcast, hdot] using hsum

/-- Dot products over interval inputs are bounded above by `dotIntervalMulUpper`. -/
theorem dotProduct_le_dotIntervalMulUpper_real {n : Nat}
    (lo₁ hi₁ lo₂ hi₂ : Fin n → Rat) (x y : Fin n → Real)
    (hlo₁ : ∀ i, (lo₁ i : Real) ≤ x i) (hhi₁ : ∀ i, x i ≤ (hi₁ i : Real))
    (hlo₂ : ∀ i, (lo₂ i : Real) ≤ y i) (hhi₂ : ∀ i, y i ≤ (hi₂ i : Real)) :
    dotProduct x y ≤ (dotIntervalMulUpper lo₁ hi₁ lo₂ hi₂ : Real) := by
  classical
  have hterm : ∀ i,
      x i * y i ≤ (intervalMulUpper (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i) : Real) := by
    intro i
    exact mul_le_intervalMulUpper_real (hxlo := hlo₁ i) (hxhi := hhi₁ i)
      (hylo := hlo₂ i) (hyhi := hhi₂ i)
  have hsum :
      ∑ i, x i * y i ≤
        ∑ i, (intervalMulUpper (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i) : Real) := by
    refine Finset.sum_le_sum ?_
    intro i _
    exact hterm i
  have hcast : (dotIntervalMulUpper lo₁ hi₁ lo₂ hi₂ : Real) =
      ∑ i, (intervalMulUpper (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i) : Real) := by
    classical
    have hcast' :
        ratToReal (Linear.sumFin n
          (fun i => intervalMulUpper (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i))) =
          ∑ i, ratToReal (intervalMulUpper (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i)) := by
      simpa using
        (Linear.ratToReal_sumFin (n := n)
          (f := fun i => intervalMulUpper (lo₁ i) (hi₁ i) (lo₂ i) (hi₂ i)))
    simpa [dotIntervalMulUpper, ratToReal_def] using hcast'
  have hdot : dotProduct x y = ∑ i, x i * y i := by
    simp [dotProduct]
  simpa [hcast, hdot] using hsum

/-- Absolute bound for interval-valued inputs (sum of per-coordinate maxima). -/
def intervalAbsBound {n : Nat} (lo hi : Fin n → Rat) : Rat :=
  ∑ i, max |lo i| |hi i|

/-- Each coordinate's max-abs is bounded by `intervalAbsBound`. -/
theorem max_abs_le_intervalAbsBound {n : Nat}
    (lo hi : Fin n → Rat) (i : Fin n) :
    max |lo i| |hi i| ≤ intervalAbsBound lo hi := by
  classical
  have hnonneg : ∀ j ∈ (Finset.univ : Finset (Fin n)), 0 ≤ max |lo j| |hi j| := by
    intro j _
    exact (abs_nonneg (lo j)).trans (le_max_left _ _)
  have hmem : i ∈ (Finset.univ : Finset (Fin n)) := by
    simp
  have hsum :=
    Finset.single_le_sum (s := (Finset.univ : Finset (Fin n)))
      (f := fun j => max |lo j| |hi j|) hnonneg hmem
  simpa [intervalAbsBound] using hsum

/-- If `x` lies in `[lo, hi]`, its absolute value is bounded by the endpoint max. -/
theorem abs_le_max_abs_abs_of_interval_real {lo hi x : Real}
    (hlo : lo ≤ x) (hhi : x ≤ hi) : |x| ≤ max |lo| |hi| := by
  exact abs_le_max_abs_abs hlo hhi

end Bounds

end Nfp
