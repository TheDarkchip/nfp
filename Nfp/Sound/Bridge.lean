-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Rat.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Data.Finset.Lattice.Fold
import Nfp.SignedMixer
import Nfp.Sound.Bounds

namespace Nfp.Sound

open scoped BigOperators

namespace RatMatrix

variable {S T : Type*} [Fintype S] [Fintype T] [DecidableEq S] [DecidableEq T]

/-- Cast a rational matrix to a real SignedMixer. -/
noncomputable def toSignedMixer (M : RatMatrix S T) : SignedMixer S T :=
  ⟨fun i j => (M.w i j : ℝ)⟩

lemma ratAbs_eq_abs (x : Rat) : ratAbs x = |x| := by
  by_cases h : x < 0
  · simp [ratAbs, h, abs_of_neg h]
  · have h' : 0 ≤ x := le_of_not_gt h
    simp [ratAbs, h, abs_of_nonneg h']

/-- Casting the rational bound matches the real row-sum operator norm bound. -/
theorem operatorNormBound_cast (M : RatMatrix S T) [Nonempty S] :
    (operatorNormBound M : ℝ) = SignedMixer.operatorNormBound (M.toSignedMixer) := by
  classical
  have hsup_cast :
      (operatorNormBound M : ℝ) =
        Finset.sup' Finset.univ (Finset.univ_nonempty (α := S))
          (fun i => ((rowAbsSum M i : Rat) : ℝ)) := by
    apply le_antisymm
    · rcases Finset.exists_mem_eq_sup'
          (s := Finset.univ) (H := Finset.univ_nonempty (α := S))
          (f := fun i => rowAbsSum M i) with ⟨i, hi, hsup⟩
      have hcast : (operatorNormBound M : ℝ) = (rowAbsSum M i : ℝ) := by
        simp [operatorNormBound, hsup]
      calc
        (operatorNormBound M : ℝ) = (rowAbsSum M i : ℝ) := hcast
        _ ≤ Finset.sup' Finset.univ (Finset.univ_nonempty (α := S))
            (fun j => ((rowAbsSum M j : Rat) : ℝ)) := by
              exact Finset.le_sup' (s := Finset.univ)
                (f := fun j => ((rowAbsSum M j : Rat) : ℝ)) hi
    · refine (Finset.sup'_le_iff (s := Finset.univ)
          (H := Finset.univ_nonempty (α := S))
          (f := fun i => ((rowAbsSum M i : Rat) : ℝ))
          (a := (operatorNormBound M : ℝ))).2 ?_
      intro i hi
      have hle : rowAbsSum M i ≤ operatorNormBound M := by
        exact Finset.le_sup' (s := Finset.univ) (f := fun j => rowAbsSum M j) hi
      exact (Rat.cast_le (K := ℝ)).2 hle
  calc
    (operatorNormBound M : ℝ)
        = Finset.sup' Finset.univ (Finset.univ_nonempty (α := S))
            (fun i => ((rowAbsSum M i : Rat) : ℝ)) := hsup_cast
    _ = SignedMixer.operatorNormBound (M.toSignedMixer) := by
          simp [SignedMixer.operatorNormBound, rowAbsSum, toSignedMixer,
            ratAbs_eq_abs, Rat.cast_sum, Rat.cast_abs]

/-- Casted row-major bound agrees with the `SignedMixer` operator norm bound. -/
theorem matrixNormInfOfRowMajor_cast (rows cols : Nat) (data : Array Rat)
    [Nonempty (Fin rows)] (h : rows ≠ 0) :
    (matrixNormInfOfRowMajor rows cols data : ℝ) =
      SignedMixer.operatorNormBound (RatMatrix.ofRowMajor rows cols data).toSignedMixer := by
  classical
  simpa [matrixNormInfOfRowMajor, h] using
    (operatorNormBound_cast (M := RatMatrix.ofRowMajor rows cols data))

end RatMatrix

end Nfp.Sound
