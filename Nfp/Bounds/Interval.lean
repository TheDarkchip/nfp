-- SPDX-License-Identifier: AGPL-3.0-or-later

module

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
