-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Init.Data.Array.FinRange
public import Init.Data.Array.Lemmas
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Control.Fold
public import Mathlib.Data.Nat.GCD.Basic
public import Mathlib.Data.Rat.Defs
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
open scoped Rat

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
  Nat.fold n (fun i h acc =>
    let vi := v[i]'(by simpa [hv] using h)
    let loi := lo[i]'(by simpa [hlo] using h)
    let hii := hi[i]'(by simpa [hhi] using h)
    (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
      acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))) (0, 0)

/-- Scale a rational to a numerator at a fixed denominator when compatible. -/
def ratScaledNum? (den : Nat) (x : Rat) : Option Int :=
  if _ : x.den ∣ den then
    let k := den / x.den
    some (x.num * Int.ofNat k)
  else
    none

/-- Scale a rational to a numerator at a fixed denominator (caller ensures divisibility). -/
def ratScaledNum (den : Nat) (x : Rat) : Int :=
  x.num * Int.ofNat (den / x.den)

/-- Divisible denominators make `ratScaledNum?` return the scaled numerator. -/
theorem ratScaledNum?_eq_some {den : Nat} {x : Rat} (hdiv : x.den ∣ den) :
    ratScaledNum? den x = some (ratScaledNum den x) := by
  classical
  unfold ratScaledNum? ratScaledNum
  simp [hdiv]

/-- Successful scaling recovers the exact `Rat.divInt` representation. -/
theorem ratScaledNum?_eq_divInt {den : Nat} {x : Rat} {n : Int}
    (hden : den ≠ 0) (h : ratScaledNum? den x = some n) :
    x = n /. (Int.ofNat den) := by
  classical
  unfold ratScaledNum? at h
  by_cases hdiv : x.den ∣ den
  · have h' : ratScaledNum? den x = some n := h
    simp only [ratScaledNum?, hdiv] at h'
    cases h'
    have hmulNat : den = x.den * (den / x.den) := by
      simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
        (Nat.mul_div_cancel' hdiv).symm
    have hmulInt :
        (Int.ofNat den) =
          (Int.ofNat x.den) * (Int.ofNat (den / x.den)) := by
      simpa [Int.natCast_mul] using congrArg Int.ofNat hmulNat
    have hdenInt : (Int.ofNat den) ≠ 0 := by
      exact Int.ofNat_ne_zero.mpr hden
    have hxdenInt : (Int.ofNat x.den) ≠ 0 := by
      exact Int.ofNat_ne_zero.mpr (Rat.den_ne_zero x)
    have hmul :
        x.num * (Int.ofNat den) =
          (x.num * Int.ofNat (den / x.den)) * (Int.ofNat x.den) := by
      calc
        x.num * (Int.ofNat den)
            = x.num * ((Int.ofNat x.den) * (Int.ofNat (den / x.den))) := by
                rw [hmulInt]
        _ = (x.num * Int.ofNat (den / x.den)) * (Int.ofNat x.den) := by
            simp [mul_comm, mul_left_comm]
    have hdivInt :
        x.num /. (Int.ofNat x.den) =
          (x.num * Int.ofNat (den / x.den)) /. (Int.ofNat den) := by
      exact (Rat.divInt_eq_divInt_iff hxdenInt hdenInt).2 hmul
    calc
      x = x.num /. (Int.ofNat x.den) := by
        simp
      _ = (x.num * Int.ofNat (den / x.den)) /. (Int.ofNat den) := hdivInt
  · have h' : ratScaledNum? den x = some n := h
    simp only [ratScaledNum?, hdiv] at h'
    cases h'

/-- Scaled numerators recover the exact `Rat.divInt` representation. -/
theorem ratScaledNum_eq_divInt {den : Nat} {x : Rat} (hden : den ≠ 0) (hdiv : x.den ∣ den) :
    x = ratScaledNum den x /. (Int.ofNat den) := by
  apply ratScaledNum?_eq_divInt (den := den) (x := x) (n := ratScaledNum den x) hden
  exact ratScaledNum?_eq_some (den := den) (x := x) hdiv

/-- Folding `lcm` preserves divisibility by the accumulator. -/
lemma dvd_foldl_lcm_den_of_dvd (xs : List Rat) (acc n : Nat) (h : n ∣ acc) :
    n ∣ List.foldl (fun acc v => Nat.lcm acc v.den) acc xs := by
  induction xs generalizing acc with
  | nil =>
      exact h
  | cons v xs ih =>
      have h' : n ∣ Nat.lcm acc v.den := h.trans (Nat.dvd_lcm_left acc v.den)
      exact ih (acc := Nat.lcm acc v.den) h'

/-- Each denominator divides the `lcm` fold that includes it. -/
lemma dvd_foldl_lcm_den_of_mem (xs : List Rat) (acc : Nat) (x : Rat) (hmem : x ∈ xs) :
    x.den ∣ List.foldl (fun acc v => Nat.lcm acc v.den) acc xs := by
  induction xs generalizing acc with
  | nil => cases hmem
  | cons v xs ih =>
      simp only [List.mem_cons] at hmem
      cases hmem with
      | inl h =>
          subst h
          apply dvd_foldl_lcm_den_of_dvd xs (acc := Nat.lcm acc x.den) (n := x.den)
          exact Nat.dvd_lcm_right acc x.den
      | inr h =>
          exact ih (acc := Nat.lcm acc v.den) h

lemma foldl_lcm_den_pos (xs : List Rat) (acc : Nat) (hacc : 0 < acc) :
    0 < List.foldl (fun acc v => Nat.lcm acc v.den) acc xs := by
  induction xs generalizing acc with
  | nil =>
      exact hacc
  | cons v xs ih =>
      have hacc' : 0 < Nat.lcm acc v.den :=
        Nat.lcm_pos hacc (Rat.den_pos v)
      exact ih (acc := Nat.lcm acc v.den) hacc'

/-- Array `lcm` fold is divisible by each entry denominator. -/
lemma dvd_foldl_lcm_den_get (arr : Array Rat) (i : Fin arr.size) :
    (arr[i]).den ∣
      arr.foldl (fun acc v => Nat.lcm acc v.den) 1 := by
  classical
  let x := arr[i]
  have hmem : x ∈ arr := by
    refine (Array.mem_iff_getElem).2 ?_
    exact ⟨i.1, i.2, rfl⟩
  have hmem_list : x ∈ arr.toList := (Array.mem_toList_iff).2 hmem
  have hdiv_list :
      x.den ∣ List.foldl (fun acc v => Nat.lcm acc v.den) 1 arr.toList := by
    exact dvd_foldl_lcm_den_of_mem (xs := arr.toList) (acc := 1) (x := x) hmem_list
  have hfold :
      arr.foldl (fun acc v => Nat.lcm acc v.den) 1 =
        List.foldl (fun acc v => Nat.lcm acc v.den) 1 arr.toList := by
    -- `Array.foldl_toList` is a simp lemma.
    simp
  have hdiv' :
      x.den ∣ arr.foldl (fun acc v => Nat.lcm acc v.den) 1 := by
    rw [hfold]
    exact hdiv_list
  simpa [x] using hdiv'

lemma foldl_lcm_den_pos_array (arr : Array Rat) :
    0 < arr.foldl (fun acc v => Nat.lcm acc v.den) 1 := by
  classical
  have hpos :
      0 < List.foldl (fun acc v => Nat.lcm acc v.den) 1 arr.toList := by
    exact foldl_lcm_den_pos (xs := arr.toList) (acc := 1) (by decide)
  have hfold :
      arr.foldl (fun acc v => Nat.lcm acc v.den) 1 =
        List.foldl (fun acc v => Nat.lcm acc v.den) 1 arr.toList := by
    -- `Array.foldl_toList` is a simp lemma.
    simp
  have hpos' : 0 < arr.foldl (fun acc v => Nat.lcm acc v.den) 1 := by
    rw [hfold]
    exact hpos
  exact hpos'

/-- Integer-scaled dot-interval bounds, when a common denominator is available. -/
def dotIntervalBoundsFastArrScaled? (n den : Nat) (v lo hi : Array Rat)
    (hv : v.size = n) (hlo : lo.size = n) (hhi : hi.size = n) : Option (Rat × Rat) :=
  if hden : den = 0 then
    none
  else
    let denInt : Int := Int.ofNat den
    let denSqInt : Int := denInt * denInt
    let vFn : Fin n → Rat := fun i => v[i.1]'(by
      simp [hv])
    let loFn : Fin n → Rat := fun i => lo[i.1]'(by
      simp [hlo])
    let hiFn : Fin n → Rat := fun i => hi[i.1]'(by
      simp [hhi])
    let stepInt : Int × Int → Fin n → Option (Int × Int) := fun acc i =>
      match ratScaledNum? den (vFn i),
          ratScaledNum? den (loFn i),
          ratScaledNum? den (hiFn i) with
      | some vnum, some lonum, some hinum =>
          let termLo := if 0 ≤ vFn i then vnum * lonum else vnum * hinum
          let termHi := if 0 ≤ vFn i then vnum * hinum else vnum * lonum
          some (acc.1 + termLo, acc.2 + termHi)
      | _, _, _ => none
    let acc? :=
      (List.finRange n).foldlM (m := Option) stepInt (0, 0)
    match acc? with
    | some acc => some (acc.1 /. denSqInt, acc.2 /. denSqInt)
    | none => none

/-- Integer-scaled dot-interval bounds (total, assumes denominators divide `den`). -/
def dotIntervalBoundsFastArrScaled (n den : Nat) (v lo hi : Array Rat)
    (hv : v.size = n) (hlo : lo.size = n) (hhi : hi.size = n) : Rat × Rat :=
  let denInt : Int := Int.ofNat den
  let denSqInt : Int := denInt * denInt
  let vFn : Fin n → Rat := fun i => v[i.1]'(by
    simp [hv])
  let loFn : Fin n → Rat := fun i => lo[i.1]'(by
    simp [hlo])
  let hiFn : Fin n → Rat := fun i => hi[i.1]'(by
    simp [hhi])
  let stepInt : Int × Int → Fin n → Int × Int := fun acc i =>
    let vnum := ratScaledNum den (vFn i)
    let lonum := ratScaledNum den (loFn i)
    let hinum := ratScaledNum den (hiFn i)
    let termLo := if 0 ≤ vFn i then vnum * lonum else vnum * hinum
    let termHi := if 0 ≤ vFn i then vnum * hinum else vnum * lonum
    (acc.1 + termLo, acc.2 + termHi)
  let acc := (List.finRange n).foldl stepInt (0, 0)
  (acc.1 /. denSqInt, acc.2 /. denSqInt)

lemma foldlM_some_eq_foldl {α β : Type} (step : α → β → Option α) (step' : α → β → α)
    (init : α) (xs : List β)
    (hstep : ∀ acc b, step acc b = some (step' acc b)) :
    List.foldlM (m := Option) step init xs = some (List.foldl step' init xs) := by
  induction xs generalizing init with
  | nil =>
      simp
  | cons b xs ih =>
      simp [List.foldlM, hstep, ih]

/-- Scaling succeeds when all denominators divide the scale. -/
theorem dotIntervalBoundsFastArrScaled?_eq_some {n den : Nat}
    (v lo hi : Array Rat) (hv : v.size = n) (hlo : lo.size = n) (hhi : hi.size = n)
    (hden : den ≠ 0)
    (hdiv_v : ∀ i : Fin n, (v[i.1]'(by simp [hv, i.2])).den ∣ den)
    (hdiv_lo : ∀ i : Fin n, (lo[i.1]'(by simp [hlo, i.2])).den ∣ den)
    (hdiv_hi : ∀ i : Fin n, (hi[i.1]'(by simp [hhi, i.2])).den ∣ den) :
    dotIntervalBoundsFastArrScaled? n den v lo hi hv hlo hhi =
      some (dotIntervalBoundsFastArrScaled n den v lo hi hv hlo hhi) := by
  classical
  unfold dotIntervalBoundsFastArrScaled? dotIntervalBoundsFastArrScaled
  simp only [hden]
  -- reuse the same definitions as the scaled fold
  let vFn : Fin n → Rat := fun i => v[i.1]'(by
    simp [hv])
  let loFn : Fin n → Rat := fun i => lo[i.1]'(by
    simp [hlo])
  let hiFn : Fin n → Rat := fun i => hi[i.1]'(by
    simp [hhi])
  let stepInt? : Int × Int → Fin n → Option (Int × Int) := fun acc i =>
    match ratScaledNum? den (vFn i),
        ratScaledNum? den (loFn i),
        ratScaledNum? den (hiFn i) with
    | some vnum, some lonum, some hinum =>
        let termLo := if 0 ≤ vFn i then vnum * lonum else vnum * hinum
        let termHi := if 0 ≤ vFn i then vnum * hinum else vnum * lonum
        some (acc.1 + termLo, acc.2 + termHi)
    | _, _, _ => none
  let stepInt : Int × Int → Fin n → Int × Int := fun acc i =>
    let vnum := ratScaledNum den (vFn i)
    let lonum := ratScaledNum den (loFn i)
    let hinum := ratScaledNum den (hiFn i)
    let termLo := if 0 ≤ vFn i then vnum * lonum else vnum * hinum
    let termHi := if 0 ≤ vFn i then vnum * hinum else vnum * lonum
    (acc.1 + termLo, acc.2 + termHi)
  have hstep : ∀ acc i, stepInt? acc i = some (stepInt acc i) := by
    intro acc i
    have hv' : ratScaledNum? den (vFn i) = some (ratScaledNum den (vFn i)) :=
      ratScaledNum?_eq_some (den := den) (x := vFn i) (hdiv_v i)
    have hlo' : ratScaledNum? den (loFn i) = some (ratScaledNum den (loFn i)) :=
      ratScaledNum?_eq_some (den := den) (x := loFn i) (hdiv_lo i)
    have hhi' : ratScaledNum? den (hiFn i) = some (ratScaledNum den (hiFn i)) :=
      ratScaledNum?_eq_some (den := den) (x := hiFn i) (hdiv_hi i)
    simp [stepInt?, stepInt, hv', hlo', hhi']
  have hfold :
      (List.finRange n).foldlM (m := Option) stepInt? (0, 0) =
        some ((List.finRange n).foldl stepInt (0, 0)) := by
    exact foldlM_some_eq_foldl (step := stepInt?) (step' := stepInt) (init := (0, 0))
      (xs := List.finRange n) hstep
  simp [hfold, stepInt?, stepInt, vFn, loFn, hiFn]

/-- Scaled bounds agree with `dotIntervalBoundsFastArr` when they succeed. -/
theorem dotIntervalBoundsFastArrScaled?_eq {n den : Nat}
    (v lo hi : Array Rat) (hv : v.size = n) (hlo : lo.size = n) (hhi : hi.size = n)
    {bounds : Rat × Rat} (hden : den ≠ 0) :
    dotIntervalBoundsFastArrScaled? n den v lo hi hv hlo hhi = some bounds →
      bounds = dotIntervalBoundsFastArr n v lo hi hv hlo hhi := by
  classical
  intro hscaled
  unfold dotIntervalBoundsFastArrScaled? at hscaled
  -- reuse the same definitions as the scaled fold
  let denInt : Int := Int.ofNat den
  let denSqInt : Int := denInt * denInt
  let vFn : Fin n → Rat := fun i => v[i.1]'(by
    simp [hv])
  let loFn : Fin n → Rat := fun i => lo[i.1]'(by
    simp [hlo])
  let hiFn : Fin n → Rat := fun i => hi[i.1]'(by
    simp [hhi])
  let stepInt : Int × Int → Fin n → Option (Int × Int) := fun acc i =>
    match ratScaledNum? den (vFn i),
        ratScaledNum? den (loFn i),
        ratScaledNum? den (hiFn i) with
    | some vnum, some lonum, some hinum =>
        let termLo := if 0 ≤ vFn i then vnum * lonum else vnum * hinum
        let termHi := if 0 ≤ vFn i then vnum * hinum else vnum * lonum
        some (acc.1 + termLo, acc.2 + termHi)
    | _, _, _ => none
  let stepRat : Rat × Rat → Fin n → Rat × Rat := fun acc i =>
    let vi := vFn i
    let loi := loFn i
    let hii := hiFn i
    (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
      acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))
  let acc? : Option (Int × Int) :=
    (List.finRange n).foldlM (m := Option) stepInt (0, 0)
  have hscaled' :
      (match acc? with
        | some acc => some (acc.1 /. denSqInt, acc.2 /. denSqInt)
        | none => none) = some bounds := by
    simpa [hden, acc?, stepInt, vFn, loFn, hiFn, denSqInt, denInt] using hscaled
  cases hacc : acc? with
  | none =>
      simp only [hacc] at hscaled'
      cases hscaled'
  | some acc =>
      have hbounds :
          bounds = (acc.1 /. denSqInt, acc.2 /. denSqInt) := by
        have htmp :
            (acc.1 /. denSqInt, acc.2 /. denSqInt) = bounds := by
          simpa [hacc] using hscaled'
        exact htmp.symm
      -- core fold-correctness lemma on lists
      have hfoldCorrect :
          List.foldl stepRat (0, 0) (List.finRange n) =
            (acc.1 /. denSqInt, acc.2 /. denSqInt) := by
        -- helper: step correctness
        have step_correct :
            ∀ accInt accRat i accInt',
              accRat = (accInt.1 /. denSqInt, accInt.2 /. denSqInt) →
              stepInt accInt i = some accInt' →
              stepRat accRat i = (accInt'.1 /. denSqInt, accInt'.2 /. denSqInt) := by
          intro accInt accRat i accInt' haccRat hstep
          -- unpack scaling successes
          cases hvi : ratScaledNum? den (vFn i) with
          | none =>
              simp only [stepInt, hvi] at hstep
              cases hstep
          | some vnum =>
              cases hlo' : ratScaledNum? den (loFn i) with
              | none =>
                  simp only [stepInt, hvi, hlo'] at hstep
                  cases hstep
              | some lonum =>
                  cases hhi' : ratScaledNum? den (hiFn i) with
                  | none =>
                      simp only [stepInt, hvi, hlo', hhi'] at hstep
                      cases hstep
                  | some hinum =>
                      simp only [stepInt, hvi, hlo', hhi'] at hstep
                      cases hstep
                      have hviEq : vFn i = vnum /. denInt := by
                        exact
                          (ratScaledNum?_eq_divInt (den := den) (x := vFn i) hden (by simp [hvi]))
                      have hloEq : loFn i = lonum /. denInt := by
                        exact
                          (ratScaledNum?_eq_divInt (den := den) (x := loFn i) hden (by simp [hlo']))
                      have hhiEq : hiFn i = hinum /. denInt := by
                        exact
                          (ratScaledNum?_eq_divInt (den := den) (x := hiFn i) hden (by simp [hhi']))
                      -- compute the Rat step
                      have htermLo :
                          (if 0 ≤ vFn i then vFn i * loFn i else vFn i * hiFn i) =
                            (if 0 ≤ vFn i then vnum * lonum else vnum * hinum) /. denSqInt := by
                        by_cases hpos : 0 ≤ vFn i
                        · simp only [hpos]
                          simp [hviEq, hloEq, denSqInt, Rat.divInt_mul_divInt]
                        · simp only [hpos]
                          simp [hviEq, hhiEq, denSqInt, Rat.divInt_mul_divInt]
                      have htermHi :
                          (if 0 ≤ vFn i then vFn i * hiFn i else vFn i * loFn i) =
                            (if 0 ≤ vFn i then vnum * hinum else vnum * lonum) /. denSqInt := by
                        by_cases hpos : 0 ≤ vFn i
                        · simp only [hpos]
                          simp [hviEq, hhiEq, denSqInt, Rat.divInt_mul_divInt]
                        · simp only [hpos]
                          simp [hviEq, hloEq, denSqInt, Rat.divInt_mul_divInt]
                      -- combine with accumulator
                      have hsum1 :
                          accRat.1 + (if 0 ≤ vFn i then vFn i * loFn i else vFn i * hiFn i) =
                            (accInt.1 + (if 0 ≤ vFn i then vnum * lonum else vnum * hinum))
                              /. denSqInt := by
                        calc
                          accRat.1 + (if 0 ≤ vFn i then vFn i * loFn i else vFn i * hiFn i)
                              = (accInt.1 /. denSqInt) +
                                  ((if 0 ≤ vFn i then vnum * lonum else vnum * hinum)
                                    /. denSqInt) := by
                                simp [haccRat, htermLo]
                          _ =
                              (accInt.1 + (if 0 ≤ vFn i then vnum * lonum else vnum * hinum))
                                /. denSqInt := by
                            symm
                            exact Rat.add_divInt accInt.1
                              (if 0 ≤ vFn i then vnum * lonum else vnum * hinum) denSqInt
                      have hsum2 :
                          accRat.2 + (if 0 ≤ vFn i then vFn i * hiFn i else vFn i * loFn i) =
                            (accInt.2 + (if 0 ≤ vFn i then vnum * hinum else vnum * lonum))
                              /. denSqInt := by
                        calc
                          accRat.2 + (if 0 ≤ vFn i then vFn i * hiFn i else vFn i * loFn i)
                              = (accInt.2 /. denSqInt) +
                                  ((if 0 ≤ vFn i then vnum * hinum else vnum * lonum)
                                    /. denSqInt) := by
                                simp [haccRat, htermHi]
                          _ =
                              (accInt.2 + (if 0 ≤ vFn i then vnum * hinum else vnum * lonum))
                                /. denSqInt := by
                            symm
                            exact Rat.add_divInt accInt.2
                              (if 0 ≤ vFn i then vnum * hinum else vnum * lonum) denSqInt
                      simp [stepRat, hsum1, hsum2]
        -- list-fold induction using step_correct
        have hfoldGen :
            ∀ xs accInt accRat,
              accRat = (accInt.1 /. denSqInt, accInt.2 /. denSqInt) →
              List.foldlM (m := Option) stepInt accInt xs = some acc →
              List.foldl stepRat accRat xs =
                (acc.1 /. denSqInt, acc.2 /. denSqInt) := by
          intro xs
          induction xs with
          | nil =>
              intro accInt accRat haccRat hfold
              simp only [List.foldlM] at hfold
              cases hfold
              simp [haccRat]
          | cons i xs ih =>
              intro accInt accRat haccRat hfold
              cases hstep : stepInt accInt i with
              | none =>
                  simp only [List.foldlM, hstep] at hfold
                  cases hfold
              | some accInt' =>
                  have hfold' := hfold
                  simp only [List.foldlM, hstep] at hfold'
                  have haccRat' :
                      stepRat accRat i = (accInt'.1 /. denSqInt, accInt'.2 /. denSqInt) :=
                    step_correct accInt accRat i accInt' haccRat hstep
                  have ih' := ih accInt' (stepRat accRat i) haccRat' hfold'
                  exact ih'
        have hfoldMain :
            List.foldl stepRat (0, 0) (List.finRange n) =
              (acc.1 /. denSqInt, acc.2 /. denSqInt) := by
          apply hfoldGen (List.finRange n) (0, 0) (0, 0) (by simp)
          simpa [acc?] using hacc
        exact hfoldMain
      -- relate Nat.fold to List.foldl
      have hfoldNat :
          dotIntervalBoundsFastArr n v lo hi hv hlo hhi =
            List.foldl stepRat (0, 0) (List.finRange n) := by
        -- rewrite Nat.fold into Fin.foldl, then List.foldl
        have natFold_eq_foldlFin {α : Type} (n : Nat)
            (f : (i : Nat) → i < n → α → α) (init : α) :
            Nat.fold n f init = Fin.foldl n (fun acc i => f i.1 i.2 acc) init := by
          induction n with
          | zero => simp [Nat.fold_zero, Fin.foldl_zero]
          | succ n ih => simp [Nat.fold_succ, Fin.foldl_succ_last, ih]
        let f : (i : Nat) → i < n → Rat × Rat → Rat × Rat := fun i h acc =>
          let vi := v[i]'(by simpa [hv] using h)
          let loi := lo[i]'(by simpa [hlo] using h)
          let hii := hi[i]'(by simpa [hhi] using h)
          (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
            acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))
        have hfold := natFold_eq_foldlFin (n := n) (f := f) (init := (0, 0))
        simpa [dotIntervalBoundsFastArr, stepRat, vFn, loFn, hiFn, f,
          Linear.foldlFin_eq_foldl, Fin.foldl_eq_foldl_finRange] using hfold
      -- finish
      calc
        bounds = (acc.1 /. denSqInt, acc.2 /. denSqInt) := hbounds
        _ = dotIntervalBoundsFastArr n v lo hi hv hlo hhi := by
          symm
          simpa [hfoldNat] using hfoldCorrect

/-- Scaled bounds agree with `dotIntervalBoundsFastArr` when denominators divide. -/
theorem dotIntervalBoundsFastArrScaled_eq {n den : Nat}
    (v lo hi : Array Rat) (hv : v.size = n) (hlo : lo.size = n) (hhi : hi.size = n)
    (hden : den ≠ 0)
    (hdiv_v : ∀ i : Fin n, (v[i.1]'(by simp [hv, i.2])).den ∣ den)
    (hdiv_lo : ∀ i : Fin n, (lo[i.1]'(by simp [hlo, i.2])).den ∣ den)
    (hdiv_hi : ∀ i : Fin n, (hi[i.1]'(by simp [hhi, i.2])).den ∣ den) :
    dotIntervalBoundsFastArrScaled n den v lo hi hv hlo hhi =
      dotIntervalBoundsFastArr n v lo hi hv hlo hhi := by
  have hsome :
      dotIntervalBoundsFastArrScaled? n den v lo hi hv hlo hhi =
        some (dotIntervalBoundsFastArrScaled n den v lo hi hv hlo hhi) :=
    dotIntervalBoundsFastArrScaled?_eq_some (n := n) (den := den) (v := v) (lo := lo) (hi := hi)
      hv hlo hhi hden hdiv_v hdiv_lo hdiv_hi
  have hEq :=
    dotIntervalBoundsFastArrScaled?_eq (n := n) (den := den) (v := v) (lo := lo) (hi := hi)
      hv hlo hhi (bounds := dotIntervalBoundsFastArrScaled n den v lo hi hv hlo hhi) hden hsome
  simpa using hEq

/-- Fast path for `dotIntervalBoundsFastArr`, falling back on exact bounds. -/
def dotIntervalBoundsFastArrFast (n : Nat) (den? : Option Nat) (v lo hi : Array Rat)
    (hv : v.size = n) (hlo : lo.size = n) (hhi : hi.size = n) : Rat × Rat :=
  match den? with
  | some den =>
      match dotIntervalBoundsFastArrScaled? n den v lo hi hv hlo hhi with
      | some bounds => bounds
      | none => dotIntervalBoundsFastArr n v lo hi hv hlo hhi
  | none => dotIntervalBoundsFastArr n v lo hi hv hlo hhi

/-- The fast path always agrees with `dotIntervalBoundsFastArr`. -/
theorem dotIntervalBoundsFastArrFast_eq {n : Nat} (den? : Option Nat)
    (v lo hi : Array Rat) (hv : v.size = n) (hlo : lo.size = n) (hhi : hi.size = n) :
    dotIntervalBoundsFastArrFast n den? v lo hi hv hlo hhi =
      dotIntervalBoundsFastArr n v lo hi hv hlo hhi := by
  classical
  cases den? with
  | none => rfl
  | some den =>
      by_cases hden : den = 0
      · simp [dotIntervalBoundsFastArrFast, dotIntervalBoundsFastArrScaled?, hden]
      · cases hscaled : dotIntervalBoundsFastArrScaled? n den v lo hi hv hlo hhi with
        | none =>
            simp [dotIntervalBoundsFastArrFast, hscaled]
        | some bounds =>
            have hEq :=
              dotIntervalBoundsFastArrScaled?_eq (n := n) (den := den)
                (v := v) (lo := lo) (hi := hi) hv hlo hhi (bounds := bounds)
                (by exact hden) hscaled
            simp [dotIntervalBoundsFastArrFast, hscaled, hEq]

/-- `dotIntervalBoundsFastArr` matches `dotIntervalBoundsFast` on `Array.ofFn` inputs. -/
theorem dotIntervalBoundsFastArr_ofFn {n : Nat}
    (v lo hi : Fin n → Rat) :
    dotIntervalBoundsFastArr n (Array.ofFn v) (Array.ofFn lo) (Array.ofFn hi)
        (by simp) (by simp) (by simp) =
      dotIntervalBoundsFast v lo hi := by
  classical
  have natFold_eq_foldlFin {α : Type} (n : Nat)
      (f : (i : Nat) → i < n → α → α) (init : α) :
      Nat.fold n f init = Fin.foldl n (fun acc i => f i.1 i.2 acc) init := by
    induction n with
    | zero => simp [Nat.fold_zero, Fin.foldl_zero]
    | succ n ih => simp [Nat.fold_succ, Fin.foldl_succ_last, ih]
  let f : (i : Nat) → i < n → Rat × Rat → Rat × Rat := fun i h acc =>
    let vi := (Array.ofFn v)[i]'(by simpa using h)
    let loi := (Array.ofFn lo)[i]'(by simpa using h)
    let hii := (Array.ofFn hi)[i]'(by simpa using h)
    (acc.1 + (if 0 ≤ vi then vi * loi else vi * hii),
      acc.2 + (if 0 ≤ vi then vi * hii else vi * loi))
  have hfold := natFold_eq_foldlFin (n := n) (f := f) (init := (0, 0))
  -- Simplify the array indices to the original `Fin` functions.
  simpa [dotIntervalBoundsFastArr, dotIntervalBoundsFast, Linear.foldlFin_eq_foldl, f] using hfold

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
