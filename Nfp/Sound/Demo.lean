-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Nfp.Linearization

namespace Nfp.Sound

open scoped BigOperators

open Nfp

/-- A tiny 2×2 signed mixer used to demonstrate the sound row-sum bound story. -/
noncomputable def demoMixer : SignedMixer (Fin 1 × Fin 2) (Fin 1 × Fin 2) :=
  ⟨fun i j =>
    match i.2.val, j.2.val with
    | 0, 0 => (1 : ℝ)
    | 0, 1 => (2 : ℝ)
    | 1, 0 => (-3 : ℝ)
    | 1, 1 => (4 : ℝ)
    | _, _ => 0⟩

/-- End-to-end toy lemma: the abstract `Linearization.operatorNormBound` is controlled by an
explicit row-sum bound for a concrete small matrix.

This is intentionally tiny (2×2) but exercises the same definition (`max row sum of abs`).
-/
theorem demo_operatorNormBound_le :
    Nfp.operatorNormBound demoMixer ≤ (7 : ℝ) := by
  classical
  -- Unfold to a `Finset.sup'` of row sums.
  -- `simp` reduces the `sup' ≤ 7` goal to a per-row bound.
  simp [Nfp.operatorNormBound, SignedMixer.operatorNormBound, demoMixer]
  constructor
  · -- Row 0: |1| + |2| = 3 ≤ 7.
    have hsum :
        (∑ x : Fin 1 × Fin 2,
            abs
              (match (0 : Nat), (x.2 : Nat) with
                | 0, 0 => (1 : ℝ)
                | 0, 1 => (2 : ℝ)
                | 1, 0 => (-3 : ℝ)
                | 1, 1 => (4 : ℝ)
                | _, _ => 0)) = 3 := by
      simp [Fintype.sum_prod_type, Fin.sum_univ_two]
      norm_num
    nlinarith [hsum]
  · -- Row 1: |−3| + |4| = 7 ≤ 7.
    have hsum :
        (∑ x : Fin 1 × Fin 2,
            abs
              (match (1 : Nat), (x.2 : Nat) with
                | 0, 0 => (1 : ℝ)
                | 0, 1 => (2 : ℝ)
                | 1, 0 => (-3 : ℝ)
                | 1, 1 => (4 : ℝ)
                | _, _ => 0)) = 7 := by
      simp [Fintype.sum_prod_type, Fin.sum_univ_two]
      norm_num
    nlinarith [hsum]

end Nfp.Sound
