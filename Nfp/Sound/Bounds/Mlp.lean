-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Ring.Rat
import Nfp.Sound.Bounds.Gelu
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Bounds.MatrixNorm

/-!
Interval bounds for GPT-2 MLP blocks (linear + GELU + linear).
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

/-- Real-valued MLP with tanh-based GELU activations. -/
noncomputable def mlpReal {dModel hidden : Nat}
    (wIn : Fin dModel → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin dModel → Rat) (bOut : Fin dModel → Rat)
    (x : Fin dModel → Real) : Fin dModel → Real :=
  fun i =>
    let hidden : Fin hidden → Real := fun h =>
      geluTanh (dotProduct (fun j => (wIn j h : Real)) x + (bIn h : Real))
    dotProduct (fun h => (wOut h i : Real)) hidden + (bOut i : Real)

/-- Interval bounds for a tanh-GELU MLP given input intervals. -/
def mlpBounds {dModel hidden : Nat}
    (wIn : Fin dModel → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin dModel → Rat) (bOut : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let preLo : Fin hidden → Rat := fun h =>
    dotIntervalLower (fun j => wIn j h) lo hi + bIn h
  let preHi : Fin hidden → Rat := fun h =>
    dotIntervalUpper (fun j => wIn j h) lo hi + bIn h
  let geluLo : Fin hidden → Rat := fun h => min (preLo h) 0
  let geluHi : Fin hidden → Rat := fun h => max (preHi h) 0
  let outLo : Fin dModel → Rat := fun i =>
    dotIntervalLower (fun h => wOut h i) geluLo geluHi + bOut i
  let outHi : Fin dModel → Rat := fun i =>
    dotIntervalUpper (fun h => wOut h i) geluLo geluHi + bOut i
  (outLo, outHi)

/-- `mlpBounds` soundness for real MLP outputs. -/
theorem mlpBounds_spec {dModel hidden : Nat}
    (wIn : Fin dModel → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin dModel → Rat) (bOut : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) (x : Fin dModel → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    let bounds := mlpBounds wIn bIn wOut bOut lo hi
    ∀ i, (bounds.1 i : Real) ≤ mlpReal wIn bIn wOut bOut x i ∧
      mlpReal wIn bIn wOut bOut x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let preLo : Fin hidden → Rat := fun h =>
    dotIntervalLower (fun j => wIn j h) lo hi + bIn h
  let preHi : Fin hidden → Rat := fun h =>
    dotIntervalUpper (fun j => wIn j h) lo hi + bIn h
  let pre : Fin hidden → Real := fun h =>
    dotProduct (fun j => (wIn j h : Real)) x + (bIn h : Real)
  have hpre_lower : ∀ h, (preLo h : Real) ≤ pre h := by
    intro h
    have hdot :=
      dotIntervalLower_le_dotProduct_real (v := fun j => wIn j h) lo hi x hlo hhi
    have hdot' := add_le_add_right hdot (bIn h : Real)
    simpa [pre, preLo, Rat.cast_add] using hdot'
  have hpre_upper : ∀ h, pre h ≤ (preHi h : Real) := by
    intro h
    have hdot :=
      dotProduct_le_dotIntervalUpper_real (v := fun j => wIn j h) lo hi x hlo hhi
    have hdot' := add_le_add_right hdot (bIn h : Real)
    simpa [pre, preHi, Rat.cast_add] using hdot'
  let geluLo : Fin hidden → Rat := fun h => min (preLo h) 0
  let geluHi : Fin hidden → Rat := fun h => max (preHi h) 0
  let hidden : Fin hidden → Real := fun h => geluTanh (pre h)
  have hgelu : ∀ h, (geluLo h : Real) ≤ hidden h ∧ hidden h ≤ (geluHi h : Real) := by
    intro h
    have hbounds := geluInterval_bounds (lo := preLo h) (hi := preHi h)
      (hpre_lower h) (hpre_upper h)
    dsimp [geluLo, geluHi, hidden, geluInterval]
    exact hbounds
  let outLo : Fin dModel → Rat := fun i =>
    dotIntervalLower (fun h => wOut h i) geluLo geluHi + bOut i
  let outHi : Fin dModel → Rat := fun i =>
    dotIntervalUpper (fun h => wOut h i) geluLo geluHi + bOut i
  have hout_lower :
      (outLo i : Real) ≤ dotProduct (fun h => (wOut h i : Real)) hidden + (bOut i : Real) := by
    have hdot :=
      dotIntervalLower_le_dotProduct_real (v := fun h => wOut h i) geluLo geluHi hidden
        (fun h => (hgelu h).1) (fun h => (hgelu h).2)
    have hdot' := add_le_add_right hdot (bOut i : Real)
    simpa [outLo, Rat.cast_add] using hdot'
  have hout_upper :
      dotProduct (fun h => (wOut h i : Real)) hidden + (bOut i : Real) ≤ (outHi i : Real) := by
    have hdot :=
      dotProduct_le_dotIntervalUpper_real (v := fun h => wOut h i) geluLo geluHi hidden
        (fun h => (hgelu h).1) (fun h => (hgelu h).2)
    have hdot' := add_le_add_right hdot (bOut i : Real)
    simpa [outHi, Rat.cast_add] using hdot'
  have hlo' : (outLo i : Real) ≤ mlpReal wIn bIn wOut bOut x i := by
    simpa [mlpReal, hidden, pre] using hout_lower
  have hhi' : mlpReal wIn bIn wOut bOut x i ≤ (outHi i : Real) := by
    simpa [mlpReal, hidden, pre] using hout_upper
  simpa [bounds, mlpBounds, preLo, preHi, geluLo, geluHi, outLo, outHi] using
    And.intro hlo' hhi'

/-- Interval bounds for a LayerNorm + MLP sublayer from exact inputs. -/
def layerNormMlpBounds {n hidden : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat)
    (wIn : Fin n → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin n → Rat) (bOut : Fin n → Rat)
    (x : Fin n → Rat) : (Fin n → Rat) × (Fin n → Rat) :=
  let ln := layerNormBounds eps gamma beta x
  mlpBounds wIn bIn wOut bOut ln.1 ln.2

/-- `layerNormMlpBounds` soundness for real LayerNorm + MLP outputs. -/
theorem layerNormMlpBounds_spec {n hidden : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat)
    (wIn : Fin n → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin n → Rat) (bOut : Fin n → Rat)
    (x : Fin n → Rat) (hne : n ≠ 0) (heps : 0 < eps) :
    let bounds := layerNormMlpBounds eps gamma beta wIn bIn wOut bOut x
    ∀ i, (bounds.1 i : Real) ≤
        mlpReal wIn bIn wOut bOut (layerNormReal eps gamma beta x) i ∧
      mlpReal wIn bIn wOut bOut (layerNormReal eps gamma beta x) i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let ln := layerNormBounds eps gamma beta x
  have hln := layerNormBounds_spec eps gamma beta x hne heps
  have hlo : ∀ j, (ln.1 j : Real) ≤ layerNormReal eps gamma beta x j := fun j => (hln j).1
  have hhi : ∀ j, layerNormReal eps gamma beta x j ≤ (ln.2 j : Real) := fun j => (hln j).2
  have hmlp := mlpBounds_spec wIn bIn wOut bOut ln.1 ln.2
    (layerNormReal eps gamma beta x) hlo hhi
  simpa [bounds, layerNormMlpBounds, ln] using hmlp i

/-- Interval bounds for LayerNorm + MLP sublayer from interval inputs. -/
def layerNormAbsMlpBounds {n hidden : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat)
    (wIn : Fin n → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin n → Rat) (bOut : Fin n → Rat)
    (lo hi : Fin n → Rat) : (Fin n → Rat) × (Fin n → Rat) :=
  let absBound := intervalAbsBound lo hi
  let ln := layerNormAbsBounds eps gamma beta absBound
  mlpBounds wIn bIn wOut bOut ln.1 ln.2

/-- `layerNormAbsMlpBounds` soundness for real LayerNorm + MLP outputs. -/
theorem layerNormAbsMlpBounds_spec {n hidden : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat)
    (wIn : Fin n → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin n → Rat) (bOut : Fin n → Rat)
    (lo hi : Fin n → Rat) (x : Fin n → Real)
    (hne : n ≠ 0) (heps : 0 < eps)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) :
    let bounds := layerNormAbsMlpBounds eps gamma beta wIn bIn wOut bOut lo hi
    ∀ i, (bounds.1 i : Real) ≤
        mlpReal wIn bIn wOut bOut (layerNormRealOfReal eps gamma beta x) i ∧
      mlpReal wIn bIn wOut bOut (layerNormRealOfReal eps gamma beta x) i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let absBound := intervalAbsBound lo hi
  let ln := layerNormAbsBounds eps gamma beta absBound
  have habs : ∀ j, |x j| ≤ (absBound : Real) := by
    intro j
    have hbound :
        |x j| ≤ max |(lo j : Real)| |(hi j : Real)| :=
      abs_le_max_abs_abs_of_interval_real (hlo j) (hhi j)
    have hnonempty : (Finset.univ : Finset (Fin n)).Nonempty := ⟨j, by simp⟩
    have hsup :
        max |lo j| |hi j| ≤ intervalAbsBound lo hi := by
      have hsup' :
          max |lo j| |hi j| ≤
            (Finset.univ).sup' hnonempty (fun k => max |lo k| |hi k|) := by
        simpa using
          (Finset.le_sup'
            (s := (Finset.univ : Finset (Fin n)))
            (f := fun k => max |lo k| |hi k|)
            (by simp : j ∈ (Finset.univ : Finset (Fin n))))
      simpa [intervalAbsBound, hnonempty] using hsup'
    have hsup_real :
        max |(lo j : Real)| |(hi j : Real)| ≤ (absBound : Real) := by
      exact_mod_cast hsup
    exact le_trans hbound hsup_real
  have hln :=
    layerNormAbsBounds_spec_real eps gamma beta absBound x hne heps habs
  have hlo_ln : ∀ j, (ln.1 j : Real) ≤ layerNormRealOfReal eps gamma beta x j := fun j => (hln j).1
  have hhi_ln : ∀ j, layerNormRealOfReal eps gamma beta x j ≤ (ln.2 j : Real) := fun j => (hln j).2
  have hmlp := mlpBounds_spec wIn bIn wOut bOut ln.1 ln.2
    (layerNormRealOfReal eps gamma beta x) hlo_ln hhi_ln
  simpa [bounds, layerNormAbsMlpBounds, absBound, ln] using hmlp i

/-- Add residual inputs to interval bounds. -/
def residualAddBounds {n : Nat} (x : Fin n → Rat) (lo hi : Fin n → Rat) :
    (Fin n → Rat) × (Fin n → Rat) :=
  (fun i => x i + lo i, fun i => x i + hi i)

/-- `residualAddBounds` soundness for residual addition. -/
theorem residualAddBounds_spec {n : Nat} (x : Fin n → Rat)
    (lo hi : Fin n → Rat) (y : Fin n → Real)
    (hlo : ∀ i, (lo i : Real) ≤ y i) (hhi : ∀ i, y i ≤ (hi i : Real)) :
    let bounds := residualAddBounds x lo hi
    ∀ i, (bounds.1 i : Real) ≤ (x i : Real) + y i ∧
      (x i : Real) + y i ≤ (bounds.2 i : Real) := by
  intro bounds i
  have hlow := add_le_add_left (hlo i) (x i : Real)
  have hhigh := add_le_add_left (hhi i) (x i : Real)
  constructor
  · simpa [bounds, residualAddBounds, Rat.cast_add] using hlow
  · simpa [bounds, residualAddBounds, Rat.cast_add] using hhigh

/-- Interval bounds for a full MLP residual path (LayerNorm + MLP + residual add). -/
def layerNormMlpResidualBounds {n hidden : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat)
    (wIn : Fin n → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin n → Rat) (bOut : Fin n → Rat)
    (x : Fin n → Rat) : (Fin n → Rat) × (Fin n → Rat) :=
  let mlp := layerNormMlpBounds eps gamma beta wIn bIn wOut bOut x
  residualAddBounds x mlp.1 mlp.2

/-- `layerNormMlpResidualBounds` soundness for the MLP residual path. -/
theorem layerNormMlpResidualBounds_spec {n hidden : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat)
    (wIn : Fin n → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin n → Rat) (bOut : Fin n → Rat)
    (x : Fin n → Rat) (hne : n ≠ 0) (heps : 0 < eps) :
    let bounds := layerNormMlpResidualBounds eps gamma beta wIn bIn wOut bOut x
    ∀ i,
      (bounds.1 i : Real) ≤
        (x i : Real) +
          mlpReal wIn bIn wOut bOut (layerNormReal eps gamma beta x) i ∧
        (x i : Real) +
          mlpReal wIn bIn wOut bOut (layerNormReal eps gamma beta x) i ≤
          (bounds.2 i : Real) := by
  classical
  intro bounds i
  let mlp := layerNormMlpBounds eps gamma beta wIn bIn wOut bOut x
  have hmlp := layerNormMlpBounds_spec eps gamma beta wIn bIn wOut bOut x hne heps
  have hres := residualAddBounds_spec x mlp.1 mlp.2
    (mlpReal wIn bIn wOut bOut (layerNormReal eps gamma beta x))
    (fun j => (hmlp j).1) (fun j => (hmlp j).2)
  simpa [bounds, layerNormMlpResidualBounds, mlp] using hres i

/-- Interval bounds for a full MLP residual path (LayerNorm + MLP + residual add) from intervals. -/
def layerNormAbsMlpResidualBounds {n hidden : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat)
    (wIn : Fin n → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin n → Rat) (bOut : Fin n → Rat)
    (lo hi : Fin n → Rat) : (Fin n → Rat) × (Fin n → Rat) :=
  let mlp := layerNormAbsMlpBounds eps gamma beta wIn bIn wOut bOut lo hi
  (fun i => lo i + mlp.1 i, fun i => hi i + mlp.2 i)

/-- `layerNormAbsMlpResidualBounds` soundness for the MLP residual path. -/
theorem layerNormAbsMlpResidualBounds_spec {n hidden : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat)
    (wIn : Fin n → Fin hidden → Rat) (bIn : Fin hidden → Rat)
    (wOut : Fin hidden → Fin n → Rat) (bOut : Fin n → Rat)
    (lo hi : Fin n → Rat) (x : Fin n → Real)
    (hne : n ≠ 0) (heps : 0 < eps)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) :
    let bounds := layerNormAbsMlpResidualBounds eps gamma beta wIn bIn wOut bOut lo hi
    ∀ i,
      (bounds.1 i : Real) ≤
        x i + mlpReal wIn bIn wOut bOut (layerNormRealOfReal eps gamma beta x) i ∧
      x i + mlpReal wIn bIn wOut bOut (layerNormRealOfReal eps gamma beta x) i ≤
        (bounds.2 i : Real) := by
  classical
  intro bounds i
  let mlp := layerNormAbsMlpBounds eps gamma beta wIn bIn wOut bOut lo hi
  have hmlp := layerNormAbsMlpBounds_spec eps gamma beta wIn bIn wOut bOut lo hi x hne heps hlo hhi
  have hlo' := (hmlp i).1
  have hhi' := (hmlp i).2
  have hlow := add_le_add (hlo i) hlo'
  have hhigh := add_le_add (hhi i) hhi'
  constructor
  · simpa [bounds, layerNormAbsMlpResidualBounds, mlp, Rat.cast_add] using hlow
  · simpa [bounds, layerNormAbsMlpResidualBounds, mlp, Rat.cast_add] using hhigh

end Bounds

end Sound

end Nfp
