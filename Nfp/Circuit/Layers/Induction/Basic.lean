-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.Monoid.Unbundled.Basic
public import Mathlib.Algebra.Order.Ring.Defs
import all Nfp.Circuit.Layers.Attention
public import Nfp.Circuit.Layers.Attention

/-!
Induction-head specifications for attention cores.
-/

public section

namespace Nfp

namespace Circuit

namespace Layers

universe v

open scoped BigOperators

section Weights

variable {Val : Type v} [NonAssocSemiring Val]
variable {seq : Nat}

/-- Induction weights are one-hot at `prev` for each query position. -/
def InductionWeights (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop :=
  ∀ q, weights q = Pi.single (prev q) 1

/-- A one-hot weight vector selects the corresponding value in a dot product. -/
theorem dotProduct_eq_of_oneHot (k : Fin seq) (vals : Fin seq → Val) :
    dotProduct (Pi.single k 1) vals = vals k := by
  simp

/-- Induction weights select the `prev` value in each dot product. -/
theorem dotProduct_eq_prev (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) (vals : Fin seq → Fin seq → Val)
    (hweights : InductionWeights (Val := Val) prev weights) (q : Fin seq) :
    dotProduct (weights q) (vals q) = vals q (prev q) := by
  have hq : weights q = Pi.single (prev q) 1 := hweights q
  simp [hq]

end Weights

section Spec

variable {Val : Type v}
variable {n : Nat}

/-- Induction-head spec: for non-initial queries (1-based indices ≥ 2),
    outputs copy `prev` values. -/
def InductionSpec (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val) : Prop :=
  ∀ q, q ≠ 0 → out q = vals (prev q)

/-- Unfolding lemma for `InductionSpec`. -/
theorem InductionSpec_def (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val) :
    InductionSpec (n := n) prev out vals = ∀ q, q ≠ 0 → out q = vals (prev q) := by
  rfl

/-- Concrete `prev` map on `Fin (n + 1)` (with `0 ↦ 0`). -/
def prevIndex : Fin (Nat.succ n) → Fin (Nat.succ n)
  | ⟨0, _⟩ => 0
  | ⟨Nat.succ k, hk⟩ =>
      ⟨k, Nat.lt_trans (Nat.lt_of_succ_lt_succ hk) (Nat.lt_succ_self n)⟩

/-- Previous-token head spec: copies the immediately preceding token. -/
def PrevTokenSpec (out vals : Fin (Nat.succ n) → Val) : Prop :=
  InductionSpec (n := n) (prevIndex (n := n)) out vals

/-- Unfolding lemma for `PrevTokenSpec`. -/
theorem PrevTokenSpec_def (out vals : Fin (Nat.succ n) → Val) :
    PrevTokenSpec (n := n) out vals =
      InductionSpec (n := n) (prevIndex (n := n)) out vals := by
  rfl

end Spec

section ApproxSpec

variable {Val : Type v} [AddCommMonoid Val] [PartialOrder Val]
variable {n : Nat}

/-- Approximate induction-head spec: outputs are within `ε` of `prev` values. -/
def InductionSpecApprox (ε : Val)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val) : Prop :=
  ∀ q, q ≠ 0 → out q ≤ vals (prev q) + ε ∧ vals (prev q) ≤ out q + ε

/-- Approximate induction-head spec restricted to active queries. -/
def InductionSpecApproxOn (ε : Val) (active : Fin (Nat.succ n) → Prop)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val) : Prop :=
  ∀ q, active q → out q ≤ vals (prev q) + ε ∧ vals (prev q) ≤ out q + ε

/-- Definitional characterization of `InductionSpecApproxOn`. -/
theorem InductionSpecApproxOn_def (ε : Val) (active : Fin (Nat.succ n) → Prop)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val) :
    InductionSpecApproxOn (Val := Val) (n := n) ε active prev out vals =
      ∀ q, active q → out q ≤ vals (prev q) + ε ∧ vals (prev q) ≤ out q + ε := by
  rfl

variable [IsOrderedAddMonoid Val]

/-- Exact induction spec implies the approximate spec for any nonnegative tolerance. -/
theorem inductionSpecApprox_of_spec (ε : Val) (hε : 0 ≤ ε)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (out vals : Fin (Nat.succ n) → Val)
    (h : InductionSpec prev out vals) :
    InductionSpecApprox (Val := Val) (n := n) ε prev out vals := by
  intro q hq
  have hq' : out q = vals (prev q) := h q hq
  constructor <;>
    simpa [hq'] using
      (le_add_of_nonneg_right hε :
        vals (prev q) ≤ vals (prev q) + ε)

end ApproxSpec

section ValueRange

variable {Val : Type v} [PartialOrder Val]
variable {seq : Nat}

/-- Value-range bounds for a vector of attention values. -/
structure ValueRangeBounds (lo hi : Val) (vals : Fin seq → Val) : Prop where
  /-- Lower and upper bounds are ordered. -/
  lo_le_hi : lo ≤ hi
  /-- All values are at least `lo`. -/
  lo_le : ∀ k, lo ≤ vals k
  /-- All values are at most `hi`. -/
  le_hi : ∀ k, vals k ≤ hi

end ValueRange

section Bounds

variable {Val : Type v} [Semiring Val] [PartialOrder Val]
variable {seq : Nat} [NeZero seq]

/-- Numeric bounds certifying one-hot weights on non-initial queries
    (1-based indices ≥ 2). -/
structure OneHotBoundsOn (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop where
  /-- All weights are nonnegative on non-initial queries (1-based indices ≥ 2). -/
  nonneg : ∀ q, q ≠ 0 → ∀ k, 0 ≤ weights q k
  /-- Weights sum to one on non-initial queries (1-based indices ≥ 2). -/
  sum_one : ∀ q, q ≠ 0 → (∑ k, weights q k) = 1
  /-- Non-prev weights are nonpositive on non-initial queries (1-based indices ≥ 2). -/
  other_le_zero : ∀ q, q ≠ 0 → ∀ k, k ≠ prev q → weights q k ≤ 0

/-- Certified bounds imply one-hot weights on non-initial queries
    (1-based indices ≥ 2). -/
theorem oneHot_of_boundsOn (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) [DecidableEq (Fin seq)]
    (h : OneHotBoundsOn prev weights) :
    ∀ q, q ≠ 0 → weights q = Pi.single (prev q) 1 := by
  intro q hq
  funext k
  by_cases hk : k = prev q
  · subst hk
    have hzero :
        (∑ k ∈ (Finset.univ.erase (prev q)), weights q k) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro k hk'
      have hkne : k ≠ prev q := (Finset.mem_erase.1 hk').1
      have hle : weights q k ≤ 0 := h.other_le_zero q hq k hkne
      have hge : 0 ≤ weights q k := h.nonneg q hq k
      exact le_antisymm hle hge
    have hsum :
        weights q (prev q) +
            ∑ k ∈ (Finset.univ.erase (prev q)), weights q k =
          ∑ k, weights q k := by
      simpa using
        (Finset.add_sum_erase
          (s := (Finset.univ : Finset (Fin seq)))
          (f := weights q) (a := prev q) (by simp))
    have hprev : weights q (prev q) = 1 := by
      have hsum' :
          weights q (prev q) + 0 = 1 := by
        simpa [hzero, h.sum_one q hq] using hsum
      simpa using hsum'
    simp [Pi.single, hprev]
  · have hle : weights q k ≤ 0 := h.other_le_zero q hq k hk
    have hge : 0 ≤ weights q k := h.nonneg q hq k
    have hzero : weights q k = 0 := le_antisymm hle hge
    simp [Pi.single, hk, hzero]

end Bounds

section ApproxBounds

variable {Val : Type v} [Semiring Val] [PartialOrder Val]
variable {seq : Nat} [NeZero seq]

/-- Approximate one-hot bounds for attention weights on non-initial queries
    (1-based indices ≥ 2). -/
structure OneHotApproxBoundsOn (ε : Val) (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop where
  /-- All weights are nonnegative on non-initial queries (1-based indices ≥ 2). -/
  nonneg : ∀ q, q ≠ 0 → ∀ k, 0 ≤ weights q k
  /-- Weights sum to one on non-initial queries (1-based indices ≥ 2). -/
  sum_one : ∀ q, q ≠ 0 → (∑ k, weights q k) = 1
  /-- The `prev` weight is within `ε` of one on non-initial queries
      (1-based indices ≥ 2). -/
  prev_large : ∀ q, q ≠ 0 → 1 ≤ weights q (prev q) + ε
  /-- Non-prev weights are at most `ε` on non-initial queries (1-based indices ≥ 2). -/
  other_le : ∀ q, q ≠ 0 → ∀ k, k ≠ prev q → weights q k ≤ ε

/-- Approximate one-hot bounds for attention weights on active queries. -/
structure OneHotApproxBoundsOnActive (ε : Val) (active : Fin seq → Prop)
    (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop where
  /-- All weights are nonnegative on active queries. -/
  nonneg : ∀ q, active q → ∀ k, 0 ≤ weights q k
  /-- Weights sum to one on active queries. -/
  sum_one : ∀ q, active q → (∑ k, weights q k) = 1
  /-- The `prev` weight is within `ε` of one on active queries. -/
  prev_large : ∀ q, active q → 1 ≤ weights q (prev q) + ε
  /-- Non-prev weights are at most `ε` on active queries. -/
  other_le : ∀ q, active q → ∀ k, k ≠ prev q → weights q k ≤ ε

/-- Lift global approximate bounds to an active-set version. -/
theorem oneHotApproxBoundsOnActive_of_on (ε : Val) (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val)
    (h : OneHotApproxBoundsOn (Val := Val) ε prev weights) :
    OneHotApproxBoundsOnActive (Val := Val) ε (fun q => q ≠ 0) prev weights := by
  refine { nonneg := ?_, sum_one := ?_, prev_large := ?_, other_le := ?_ }
  · intro q hq k
    exact h.nonneg q hq k
  · intro q hq
    exact h.sum_one q hq
  · intro q hq
    exact h.prev_large q hq
  · intro q hq k hk
    exact h.other_le q hq k hk

/-- Approximate induction weights: prev weight near one, others at most `ε`. -/
def InductionWeightsApprox (ε : Val) (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val) : Prop :=
  ∀ q, q ≠ 0 →
    1 ≤ weights q (prev q) + ε ∧
      ∀ k, k ≠ prev q → weights q k ≤ ε

/-- Approximate bounds imply approximate induction weights. -/
theorem inductionWeightsApprox_of_boundsOn (ε : Val) (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Val)
    (h : OneHotApproxBoundsOn ε prev weights) :
    InductionWeightsApprox (Val := Val) ε prev weights := by
  intro q hq
  exact ⟨h.prev_large q hq, h.other_le q hq⟩

end ApproxBounds

section ApproxOutput

variable {Val : Type v} [Ring Val] [LinearOrder Val] [IsOrderedRing Val]
variable {n : Nat}

local instance : NeZero (Nat.succ n) := ⟨by simp⟩

/-- Approximate one-hot weights plus bounded values yield an approximate induction spec
    on active queries. -/
theorem inductionSpecApproxOn_of_oneHotApprox_valueRange
    (ε lo hi : Val) (active : Fin (Nat.succ n) → Prop)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Val)
    (vals : Fin (Nat.succ n) → Val)
    (hweights : OneHotApproxBoundsOnActive (Val := Val) ε active prev weights)
    (hvals : ValueRangeBounds (Val := Val) lo hi vals) :
    InductionSpecApproxOn (Val := Val) (n := n) (ε * (hi - lo)) active prev
      (fun q => dotProduct (weights q) vals) vals := by
  classical
  intro q hq
  let others : Finset (Fin (Nat.succ n)) :=
    (Finset.univ : Finset (Fin (Nat.succ n))).erase (prev q)
  have hsum_decomp :
      weights q (prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := by
    simp [others]
  have hsum :
      weights q (prev q) + ∑ k ∈ others, weights q k = 1 := by
    simpa [hweights.sum_one q hq] using hsum_decomp
  have hsum_others_le : (∑ k ∈ others, weights q k) ≤ ε := by
    have hprev : 1 ≤ weights q (prev q) + ε := hweights.prev_large q hq
    have hprev' :
        weights q (prev q) + ∑ k ∈ others, weights q k ≤ weights q (prev q) + ε := by
      simpa [hsum] using hprev
    exact (add_le_add_iff_left (weights q (prev q))).1 hprev'
  have hsum_others_nonneg : 0 ≤ ∑ k ∈ others, weights q k := by
    refine Finset.sum_nonneg ?_
    intro k hk
    exact hweights.nonneg q hq k
  have hvals_hi : ∀ k, vals k ≤ hi := hvals.le_hi
  have hvals_lo : ∀ k, lo ≤ vals k := hvals.lo_le
  have hdiff_nonneg : 0 ≤ hi - lo := sub_nonneg.mpr hvals.lo_le_hi
  have hsum_vals_le :
      (∑ k ∈ others, weights q k * vals k) ≤ (∑ k ∈ others, weights q k) * hi := by
    have hle : ∀ k ∈ others, weights q k * vals k ≤ weights q k * hi := by
      intro k hk
      have hval : vals k ≤ hi := hvals_hi k
      have hnonneg : 0 ≤ weights q k := hweights.nonneg q hq k
      exact mul_le_mul_of_nonneg_left hval hnonneg
    calc
      ∑ k ∈ others, weights q k * vals k
          ≤ ∑ k ∈ others, weights q k * hi := Finset.sum_le_sum hle
      _ = (∑ k ∈ others, weights q k) * hi := by
          simpa using
            (Finset.sum_mul (s := others) (f := fun k => weights q k) (a := hi)).symm
  have hsum_vals_ge :
      (∑ k ∈ others, weights q k) * lo ≤ (∑ k ∈ others, weights q k * vals k) := by
    have hle : ∀ k ∈ others, weights q k * lo ≤ weights q k * vals k := by
      intro k hk
      have hval : lo ≤ vals k := hvals_lo k
      have hnonneg : 0 ≤ weights q k := hweights.nonneg q hq k
      exact mul_le_mul_of_nonneg_left hval hnonneg
    calc
      (∑ k ∈ others, weights q k) * lo
          = ∑ k ∈ others, weights q k * lo := by
              exact
                (Finset.sum_mul (s := others) (f := fun k => weights q k) (a := lo))
      _ ≤ ∑ k ∈ others, weights q k * vals k := Finset.sum_le_sum hle
  have hsum_prod :
      weights q (prev q) * vals (prev q) + ∑ k ∈ others, weights q k * vals k =
        ∑ k, weights q k * vals k := by
    simp [others]
  have hout_eq :
      dotProduct (weights q) vals =
        weights q (prev q) * vals (prev q) + ∑ k ∈ others, weights q k * vals k := by
    simpa [dotProduct] using hsum_prod.symm
  have hsum_val_prev :
      weights q (prev q) * vals (prev q) +
          (∑ k ∈ others, weights q k) * vals (prev q) =
        vals (prev q) := by
    calc
      weights q (prev q) * vals (prev q) +
          (∑ k ∈ others, weights q k) * vals (prev q) =
        (weights q (prev q) + ∑ k ∈ others, weights q k) * vals (prev q) := by
          simpa using
            (add_mul (weights q (prev q)) (∑ k ∈ others, weights q k) (vals (prev q))).symm
      _ = 1 * vals (prev q) := by
          simp [hsum]
      _ = vals (prev q) := by simp
  have hsplit :
      (∑ k ∈ others, weights q k) * hi =
        (∑ k ∈ others, weights q k) * lo +
          (∑ k ∈ others, weights q k) * (hi - lo) := by
    calc
      (∑ k ∈ others, weights q k) * hi =
          (∑ k ∈ others, weights q k) * lo +
            (∑ k ∈ others, weights q k) * hi -
            (∑ k ∈ others, weights q k) * lo := by
        exact
          (add_sub_cancel_left
            ((∑ k ∈ others, weights q k) * lo) ((∑ k ∈ others, weights q k) * hi)).symm
      _ = (∑ k ∈ others, weights q k) * lo +
          ((∑ k ∈ others, weights q k) * hi -
            (∑ k ∈ others, weights q k) * lo) := by
        simp [sub_eq_add_neg, add_assoc]
      _ = (∑ k ∈ others, weights q k) * lo +
          (∑ k ∈ others, weights q k) * (hi - lo) := by
        simp [mul_sub]
  have hsum_prev_le :
      weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * lo ≤
        vals (prev q) := by
    have hmul : (∑ k ∈ others, weights q k) * lo ≤
        (∑ k ∈ others, weights q k) * vals (prev q) :=
      mul_le_mul_of_nonneg_left (hvals_lo (prev q)) hsum_others_nonneg
    calc
      weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * lo
          ≤ weights q (prev q) * vals (prev q) +
              (∑ k ∈ others, weights q k) * vals (prev q) := by
                have h :=
                  add_le_add_left hmul (weights q (prev q) * vals (prev q))
                simpa [add_comm, add_left_comm, add_assoc] using h
      _ = vals (prev q) := hsum_val_prev
  have hupper_mid :
      weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * hi ≤
        vals (prev q) + (∑ k ∈ others, weights q k) * (hi - lo) := by
    calc
      weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * hi =
          weights q (prev q) * vals (prev q) +
            ((∑ k ∈ others, weights q k) * lo +
              (∑ k ∈ others, weights q k) * (hi - lo)) := by
            simp [hsplit]
      _ = weights q (prev q) * vals (prev q) +
            (∑ k ∈ others, weights q k) * lo +
            (∑ k ∈ others, weights q k) * (hi - lo) := by
          simp [add_assoc]
      _ ≤ vals (prev q) + (∑ k ∈ others, weights q k) * (hi - lo) := by
          have h :=
            add_le_add_right hsum_prev_le
              ((∑ k ∈ others, weights q k) * (hi - lo))
          simpa [add_comm, add_left_comm, add_assoc] using h
  have hupper :
      dotProduct (weights q) vals ≤ vals (prev q) + ε * (hi - lo) := by
    have hmul :
        (∑ k ∈ others, weights q k) * (hi - lo) ≤ ε * (hi - lo) := by
      exact mul_le_mul_of_nonneg_right hsum_others_le hdiff_nonneg
    calc
      dotProduct (weights q) vals =
          weights q (prev q) * vals (prev q) + ∑ k ∈ others, weights q k * vals k := hout_eq
      _ ≤ weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * hi := by
          have h :=
            add_le_add_left hsum_vals_le (weights q (prev q) * vals (prev q))
          simpa [add_comm, add_left_comm, add_assoc] using h
      _ ≤ vals (prev q) + (∑ k ∈ others, weights q k) * (hi - lo) := hupper_mid
      _ ≤ vals (prev q) + ε * (hi - lo) := by
          have h :=
            add_le_add_left hmul (vals (prev q))
          simpa [add_comm, add_left_comm, add_assoc] using h
  have hprev_le :
      vals (prev q) ≤
        weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * hi := by
    have hmul : (∑ k ∈ others, weights q k) * vals (prev q) ≤
        (∑ k ∈ others, weights q k) * hi :=
      mul_le_mul_of_nonneg_left (hvals_hi (prev q)) hsum_others_nonneg
    have hmul' :
        weights q (prev q) * vals (prev q) +
            (∑ k ∈ others, weights q k) * vals (prev q) ≤
          weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * hi := by
      have h :=
        add_le_add_left hmul (weights q (prev q) * vals (prev q))
      simpa [add_comm, add_left_comm, add_assoc] using h
    calc
      vals (prev q) =
          weights q (prev q) * vals (prev q) +
            (∑ k ∈ others, weights q k) * vals (prev q) := by
          simpa using hsum_val_prev.symm
      _ ≤ weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * hi := hmul'
  have hprev_le' :
      vals (prev q) ≤
        weights q (prev q) * vals (prev q) +
          (∑ k ∈ others, weights q k) * lo +
          (∑ k ∈ others, weights q k) * (hi - lo) := by
    calc
      vals (prev q) ≤
          weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * hi := hprev_le
      _ =
          weights q (prev q) * vals (prev q) +
            (∑ k ∈ others, weights q k) * lo +
            (∑ k ∈ others, weights q k) * (hi - lo) := by
          simp [hsplit, add_assoc]
  have hsub :
      vals (prev q) - (∑ k ∈ others, weights q k) * (hi - lo) ≤
        weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * lo := by
    exact (sub_le_iff_le_add).2 hprev_le'
  have hlowershift :
      vals (prev q) - ε * (hi - lo) ≤
        vals (prev q) - (∑ k ∈ others, weights q k) * (hi - lo) := by
    have hmul :
        (∑ k ∈ others, weights q k) * (hi - lo) ≤ ε * (hi - lo) := by
      exact mul_le_mul_of_nonneg_right hsum_others_le hdiff_nonneg
    exact sub_le_sub_left hmul (vals (prev q))
  have hlow :
      vals (prev q) - ε * (hi - lo) ≤ dotProduct (weights q) vals := by
    calc
      vals (prev q) - ε * (hi - lo) ≤
          vals (prev q) - (∑ k ∈ others, weights q k) * (hi - lo) := hlowershift
      _ ≤ weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * lo := hsub
      _ ≤ dotProduct (weights q) vals := by
          calc
            weights q (prev q) * vals (prev q) + (∑ k ∈ others, weights q k) * lo
                ≤ weights q (prev q) * vals (prev q) + ∑ k ∈ others, weights q k * vals k := by
                    have h :=
                      add_le_add_left hsum_vals_ge (weights q (prev q) * vals (prev q))
                    simpa [add_comm, add_left_comm, add_assoc] using h
            _ = dotProduct (weights q) vals := by
                simp [hout_eq]
  have hlower :
      vals (prev q) ≤ dotProduct (weights q) vals + ε * (hi - lo) := by
    exact (sub_le_iff_le_add).1 hlow
  exact ⟨hupper, hlower⟩

/-- Approximate one-hot weights plus bounded values yield an approximate induction spec. -/
theorem inductionSpecApprox_of_oneHotApprox_valueRange
    (ε lo hi : Val)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Val)
    (vals : Fin (Nat.succ n) → Val)
    (hweights : OneHotApproxBoundsOn (Val := Val) ε prev weights)
    (hvals : ValueRangeBounds (Val := Val) lo hi vals) :
    InductionSpecApprox (Val := Val) (n := n) (ε * (hi - lo)) prev
      (fun q => dotProduct (weights q) vals) vals := by
  have hweights' :
      OneHotApproxBoundsOnActive (Val := Val) ε (fun q => q ≠ 0) prev weights :=
    oneHotApproxBoundsOnActive_of_on (Val := Val) (seq := Nat.succ n)
      (ε := ε) (prev := prev) (weights := weights) hweights
  exact
    inductionSpecApproxOn_of_oneHotApprox_valueRange
      (Val := Val)
      (n := n)
      (ε := ε)
      (lo := lo)
      (hi := hi)
      (active := fun q => q ≠ 0)
      (prev := prev)
      (weights := weights)
      (vals := vals)
      (hweights := hweights')
      (hvals := hvals)

end ApproxOutput

section SoftmaxMargin

variable {Val : Type v} [Semiring Val] [PartialOrder Val]
variable {seq : Nat} [NeZero seq]

/-- Softmax margin certificates for approximate one-hot weights. -/
structure SoftmaxMarginBounds (ε margin : Val) (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val) : Prop where
  /-- Score gap between `prev` and other keys on non-initial queries
      (1-based indices ≥ 2). -/
  score_margin : ∀ q, q ≠ 0 → ∀ k, k ≠ prev q → scores q k + margin ≤ scores q (prev q)
  /-- All weights are nonnegative on non-initial queries (1-based indices ≥ 2). -/
  nonneg : ∀ q, q ≠ 0 → ∀ k, 0 ≤ weights q k
  /-- Weights sum to one on non-initial queries (1-based indices ≥ 2). -/
  sum_one : ∀ q, q ≠ 0 → (∑ k, weights q k) = 1
  /-- The `prev` weight is within `ε` of one on non-initial queries
      (1-based indices ≥ 2). -/
  prev_large : ∀ q, q ≠ 0 → 1 ≤ weights q (prev q) + ε
  /-- Non-prev weights are at most `ε` on non-initial queries (1-based indices ≥ 2). -/
  other_le : ∀ q, q ≠ 0 → ∀ k, k ≠ prev q → weights q k ≤ ε

/-- Softmax margin certificates for approximate one-hot weights on active queries. -/
structure SoftmaxMarginBoundsOn (ε margin : Val) (active : Fin seq → Prop)
    (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val) : Prop where
  /-- Score gap between `prev` and other keys on active queries. -/
  score_margin : ∀ q, active q → ∀ k, k ≠ prev q → scores q k + margin ≤ scores q (prev q)
  /-- All weights are nonnegative on active queries. -/
  nonneg : ∀ q, active q → ∀ k, 0 ≤ weights q k
  /-- Weights sum to one on active queries. -/
  sum_one : ∀ q, active q → (∑ k, weights q k) = 1
  /-- The `prev` weight is within `ε` of one on active queries. -/
  prev_large : ∀ q, active q → 1 ≤ weights q (prev q) + ε
  /-- Non-prev weights are at most `ε` on active queries. -/
  other_le : ∀ q, active q → ∀ k, k ≠ prev q → weights q k ≤ ε

/-- Lift global softmax-margin bounds to an active-set version. -/
theorem softmaxMarginBoundsOn_of_on (ε margin : Val) (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val)
    (h : SoftmaxMarginBounds (Val := Val) ε margin prev scores weights) :
    SoftmaxMarginBoundsOn (Val := Val) ε margin (fun q => q ≠ 0) prev scores weights := by
  refine
    { score_margin := ?_
      nonneg := ?_
      sum_one := ?_
      prev_large := ?_
      other_le := ?_ }
  · intro q hq k hk
    exact h.score_margin q hq k hk
  · intro q hq k
    exact h.nonneg q hq k
  · intro q hq
    exact h.sum_one q hq
  · intro q hq
    exact h.prev_large q hq
  · intro q hq k hk
    exact h.other_le q hq k hk

/-- Margin certificates yield approximate one-hot bounds for the weights. -/
theorem oneHotApproxBounds_of_softmaxMargin (ε margin : Val) (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val)
    (h : SoftmaxMarginBounds (Val := Val) ε margin prev scores weights) :
    OneHotApproxBoundsOn (Val := Val) ε prev weights := by
  exact
    { nonneg := h.nonneg
      sum_one := h.sum_one
      prev_large := h.prev_large
      other_le := h.other_le }

/-- Margin certificates imply approximate induction-weight bounds. -/
theorem inductionWeightsApprox_of_softmaxMargin (ε margin : Val)
    (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val)
    (h : SoftmaxMarginBounds (Val := Val) ε margin prev scores weights) :
    InductionWeightsApprox (Val := Val) ε prev weights := by
  exact inductionWeightsApprox_of_boundsOn
    (Val := Val)
    (seq := seq)
    (ε := ε)
    (prev := prev)
    (weights := weights)
    (h := oneHotApproxBounds_of_softmaxMargin
      (Val := Val)
      (seq := seq)
      (ε := ε)
      (margin := margin)
      (prev := prev)
      (scores := scores)
      (weights := weights)
      h)

end SoftmaxMargin

section SoftmaxMarginActive

variable {Val : Type v} [Semiring Val] [PartialOrder Val]
variable {seq : Nat}

/-- Margin certificates yield approximate one-hot bounds on active queries. -/
theorem oneHotApproxBoundsOnActive_of_softmaxMargin (ε margin : Val)
    (active : Fin seq → Prop)
    (prev : Fin seq → Fin seq)
    (scores weights : Fin seq → Fin seq → Val)
    (h : SoftmaxMarginBoundsOn (Val := Val) ε margin active prev scores weights) :
    OneHotApproxBoundsOnActive (Val := Val) ε active prev weights := by
  exact
    { nonneg := h.nonneg
      sum_one := h.sum_one
      prev_large := h.prev_large
      other_le := h.other_le }

end SoftmaxMarginActive

section Attention

variable {Batch : Type} [Fintype Batch] [DecidableEq Batch]
variable {seq heads dim : Nat}
variable {Val : Type v} [NonAssocSemiring Val]

/-- Typed V-input label for attention cores. -/
abbrev attnInputV (v : QkvIndex Batch seq heads dim) :
    AttentionInput Batch seq heads dim :=
  Sum.inr (Sum.inr v)

/-- Weight function feeding an attention output node. -/
def attentionOutWeights (b : Batch) (h : Fin heads) (q : Fin seq) (d : Fin dim)
    (rec :
      ∀ j,
        (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel j
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) →
          Val) :
    Fin seq → Val :=
  fun k =>
    rec (attnWeight (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, h, q, k))
      (attentionDag_rel_weight_out (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        b h q k d)

/-- Value function feeding an attention output node. -/
def attentionOutValues (b : Batch) (h : Fin heads) (q : Fin seq) (d : Fin dim)
    (rec :
      ∀ j,
        (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel j
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) →
          Val) :
    Fin seq → Val :=
  fun k =>
    rec (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, k, h, d))
      (attentionDag_rel_v_out (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        b k h d q)

/-- One-hot attention weights force the output to copy the selected value. -/
theorem attentionGate_out_eq_of_oneHot (scale : Val)
    (softmax : (Fin seq → Val) → Fin seq → Val) (prev : Fin seq → Fin seq)
    (b : Batch) (h : Fin heads) (q : Fin seq) (d : Fin dim)
    (rec :
      ∀ j,
        (attentionDag (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)).rel j
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) →
          Val)
    (hweights :
      attentionOutWeights (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d rec =
        Pi.single (prev q) 1) :
    attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
        (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) rec =
      attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        b h q d rec (prev q) := by
  simp only [attentionGate_out_def]
  change
    dotProduct
        (attentionOutWeights (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d rec)
        (attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d rec) =
      attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        b h q d rec (prev q)
  rw [hweights]
  exact dotProduct_eq_of_oneHot (Val := Val) (seq := seq) (k := prev q)
    (vals := attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
      b h q d rec)

section Typed

variable (scale : Val) (softmax : (Fin seq → Val) → Fin seq → Val)

/-- Attention output equals the selected V input when weights are one-hot. -/
theorem attentionTyped_eval_out_eq_of_oneHot (prev : Fin seq → Fin seq)
    (input : AttentionInput Batch seq heads dim → Val)
    (b : Batch) (h : Fin heads) (q : Fin seq) (d : Fin dim)
    (hweights :
      attentionOutWeights (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d
          (fun j _ =>
            Circuit.evalInput
              (attentionCircuit (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                scale softmax)
              ((attentionInterface (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
                scale softmax).toInputAssignment input) j) =
        Pi.single (prev q) 1) :
    (attentionTyped (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax).eval
        input (b, q, h, d) =
      input
        (attnInputV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          (b, prev q, h, d)) := by
  let C :=
    attentionCircuit (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
  let I :=
    attentionInterface (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
  let inputAssign := I.toInputAssignment input
  have hnot :
      attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d) ∉
        attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) := by
    simpa using
      (not_mem_attentionInputs_inr (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        (s := Sum.inr (Sum.inr (b, q, h, d))))
  have hgate :
      Circuit.evalInput C inputAssign
          (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) =
        attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
          (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d))
          (fun j _ => Circuit.evalInput C inputAssign j) := by
    exact Circuit.evalInput_eq_gate (C := C) (input := inputAssign)
      (i := attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d))
      hnot
  have hcopy :
      Circuit.evalInput C inputAssign
          (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) =
        attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d (fun j _ => Circuit.evalInput C inputAssign j) (prev q) := by
    have hgate' :
        attentionGate (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax
            (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d))
            (fun j _ => Circuit.evalInput C inputAssign j) =
          attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
            b h q d (fun j _ => Circuit.evalInput C inputAssign j) (prev q) :=
      attentionGate_out_eq_of_oneHot (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
        (scale := scale) (softmax := softmax) (prev := prev) (b := b) (h := h) (q := q) (d := d)
        (rec := fun j _ => Circuit.evalInput C inputAssign j) hweights
    exact hgate.trans hgate'
  have hmem :
      attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, prev q, h, d) ∈
        attentionInputs (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) := by
    refine (mem_attentionInputs_iff (Batch := Batch) (seq := seq) (heads := heads)
      (dim := dim)).2 ?_
    exact ⟨attnInputV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
      (b, prev q, h, d), rfl⟩
  have hinput :
      Circuit.evalInput C inputAssign
          (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, prev q, h, d)) =
        input (attnInputV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          (b, prev q, h, d)) := by
    have h :=
      Circuit.evalInput_eq_input (C := C) (input := inputAssign)
        (i := attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          (b, prev q, h, d)) hmem
    simpa [inputAssign, I, attentionInterface, attentionInputEquiv_def,
      Interface.toInputAssignment_def, attnInputV] using h
  have hvals :
      attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d (fun j _ => Circuit.evalInput C inputAssign j) (prev q) =
        Circuit.evalInput C inputAssign
          (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
            (b, prev q, h, d)) := rfl
  calc
    (attentionTyped (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) scale softmax).eval
        input (b, q, h, d) =
      Circuit.evalInput C inputAssign (I.outputs (b, q, h, d)).1 := by
        simp [TypedCircuit.eval_def, Interface.eval_def, attentionTyped_def, C, I, inputAssign]
    _ = Circuit.evalInput C inputAssign
        (attnOut (Batch := Batch) (seq := seq) (heads := heads) (dim := dim) (b, q, h, d)) := by
        rfl
    _ = attentionOutValues (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          b h q d (fun j _ => Circuit.evalInput C inputAssign j) (prev q) := hcopy
    _ = Circuit.evalInput C inputAssign
          (attnV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
            (b, prev q, h, d)) := hvals
    _ = input (attnInputV (Batch := Batch) (seq := seq) (heads := heads) (dim := dim)
          (b, prev q, h, d)) := hinput

end Typed

end Attention

section InductionSpecTyped

variable {Batch : Type} [Fintype Batch] [DecidableEq Batch]
variable {heads dim n : Nat}
variable {Val : Type v} [NonAssocSemiring Val]

variable (scale : Val)
variable (softmax : (Fin (Nat.succ n) → Val) → Fin (Nat.succ n) → Val)

/-- One-hot weights on non-initial queries (1-based indices ≥ 2) imply the induction spec
    for typed evaluation. -/
theorem attentionTyped_eval_inductionSpec_of_oneHot
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (input : AttentionInput Batch (Nat.succ n) heads dim → Val)
    (b : Batch) (h : Fin heads) (d : Fin dim)
    (hweights :
      ∀ q, q ≠ 0 →
        attentionOutWeights
            (Batch := Batch)
            (seq := Nat.succ n)
            (heads := heads)
            (dim := dim)
            b h q d
            (fun j _ =>
              Circuit.evalInput
                (attentionCircuit
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax)
                ((attentionInterface
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax).toInputAssignment input) j) =
          Pi.single (prev q) 1) :
    InductionSpec (n := n) prev
      (fun q =>
        (attentionTyped
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          scale softmax).eval input (b, q, h, d))
      (fun k =>
        input (attnInputV
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          (b, k, h, d))) := by
  intro q hq
  have hweights_q := hweights q hq
  exact attentionTyped_eval_out_eq_of_oneHot
    (Batch := Batch)
    (seq := Nat.succ n)
    (heads := heads)
    (dim := dim)
    (scale := scale)
    (softmax := softmax)
    (prev := prev)
    (input := input)
    (b := b)
    (h := h)
    (q := q)
    (d := d)
    hweights_q

/-- Induction spec for `prevIndex` under one-hot weight hypotheses. -/
theorem attentionTyped_eval_inductionSpec_prevIndex
    (input : AttentionInput Batch (Nat.succ n) heads dim → Val)
    (b : Batch) (h : Fin heads) (d : Fin dim)
    (hweights :
      ∀ q, q ≠ 0 →
        attentionOutWeights
            (Batch := Batch)
            (seq := Nat.succ n)
            (heads := heads)
            (dim := dim)
            b h q d
            (fun j _ =>
              Circuit.evalInput
                (attentionCircuit
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax)
                ((attentionInterface
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax).toInputAssignment input) j) =
          Pi.single (prevIndex (n := n) q) 1) :
    InductionSpec (n := n) (prevIndex (n := n))
      (fun q =>
        (attentionTyped
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          scale softmax).eval input (b, q, h, d))
      (fun k =>
        input (attnInputV
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          (b, k, h, d))) := by
  exact attentionTyped_eval_inductionSpec_of_oneHot
    (Batch := Batch)
    (heads := heads)
    (dim := dim)
    (n := n)
    (scale := scale)
    (softmax := softmax)
    (prev := prevIndex (n := n))
    (input := input)
    (b := b)
    (h := h)
    (d := d)
    hweights

end InductionSpecTyped

section InductionSpecApproxTyped

variable {Batch : Type} [Fintype Batch] [DecidableEq Batch]
variable {heads dim n : Nat}
variable {Val : Type v} [NonAssocSemiring Val] [PartialOrder Val] [IsOrderedAddMonoid Val]

variable (scale : Val)
variable (softmax : (Fin (Nat.succ n) → Val) → Fin (Nat.succ n) → Val)

/-- One-hot weights imply the approximate induction spec for any nonnegative tolerance. -/
theorem attentionTyped_eval_inductionSpecApprox_of_oneHot (ε : Val) (hε : 0 ≤ ε)
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (input : AttentionInput Batch (Nat.succ n) heads dim → Val)
    (b : Batch) (h : Fin heads) (d : Fin dim)
    (hweights :
      ∀ q, q ≠ 0 →
        attentionOutWeights
            (Batch := Batch)
            (seq := Nat.succ n)
            (heads := heads)
            (dim := dim)
            b h q d
            (fun j _ =>
              Circuit.evalInput
                (attentionCircuit
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax)
                ((attentionInterface
                  (Batch := Batch)
                  (seq := Nat.succ n)
                  (heads := heads)
                  (dim := dim)
                  scale softmax).toInputAssignment input) j) =
          Pi.single (prev q) 1) :
    InductionSpecApprox (Val := Val) (n := n) ε prev
      (fun q =>
        (attentionTyped
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          scale softmax).eval input (b, q, h, d))
      (fun k =>
        input (attnInputV
          (Batch := Batch)
          (seq := Nat.succ n)
          (heads := heads)
          (dim := dim)
          (b, k, h, d))) := by
  apply inductionSpecApprox_of_spec (Val := Val) (n := n) (ε := ε) hε
  exact attentionTyped_eval_inductionSpec_of_oneHot
    (Batch := Batch)
    (heads := heads)
    (dim := dim)
    (n := n)
    (scale := scale)
    (softmax := softmax)
    (prev := prev)
    (input := input)
    (b := b)
    (h := h)
    (d := d)
    hweights

end InductionSpecApproxTyped

end Layers

end Circuit

end Nfp
