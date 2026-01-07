-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Nfp.Core.Basic
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
    (wIn : Fin dModel → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin dModel → Dyadic) (bOut : Fin dModel → Dyadic)
    (x : Fin dModel → Real) : Fin dModel → Real :=
  fun i =>
    let hidden : Fin hidden → Real := fun h =>
      geluTanh (dotProduct (fun j => (wIn j h : Real)) x + (bIn h : Real))
    dotProduct (fun h => (wOut h i : Real)) hidden + (bOut i : Real)

/-- Interval bounds for a tanh-GELU MLP given input intervals. -/
def mlpBounds {dModel hidden : Nat}
    (wIn : Fin dModel → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin dModel → Dyadic) (bOut : Fin dModel → Dyadic)
    (lo hi : Fin dModel → Dyadic) : (Fin dModel → Dyadic) × (Fin dModel → Dyadic) :=
  let preLo : Fin hidden → Dyadic := fun h =>
    dotIntervalLower (fun j => wIn j h) lo hi + bIn h
  let preHi : Fin hidden → Dyadic := fun h =>
    dotIntervalUpper (fun j => wIn j h) lo hi + bIn h
  let geluBounds : Fin hidden → Dyadic × Dyadic := fun h => geluInterval (preLo h) (preHi h)
  let geluLo : Fin hidden → Dyadic := fun h => (geluBounds h).1
  let geluHi : Fin hidden → Dyadic := fun h => (geluBounds h).2
  let outLo : Fin dModel → Dyadic := fun i =>
    dotIntervalLower (fun h => wOut h i) geluLo geluHi + bOut i
  let outHi : Fin dModel → Dyadic := fun i =>
    dotIntervalUpper (fun h => wOut h i) geluLo geluHi + bOut i
  (outLo, outHi)

/-- `mlpBounds` soundness for real MLP outputs. -/
theorem mlpBounds_spec {dModel hidden : Nat}
    (wIn : Fin dModel → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin dModel → Dyadic) (bOut : Fin dModel → Dyadic)
    (lo hi : Fin dModel → Dyadic) (x : Fin dModel → Real)
    (hlo : ∀ j, (lo j : Real) ≤ x j) (hhi : ∀ j, x j ≤ (hi j : Real)) :
    let bounds := mlpBounds wIn bIn wOut bOut lo hi
    ∀ i, (bounds.1 i : Real) ≤ mlpReal wIn bIn wOut bOut x i ∧
      mlpReal wIn bIn wOut bOut x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let preLo : Fin hidden → Dyadic := fun h =>
    dotIntervalLower (fun j => wIn j h) lo hi + bIn h
  let preHi : Fin hidden → Dyadic := fun h =>
    dotIntervalUpper (fun j => wIn j h) lo hi + bIn h
  let pre : Fin hidden → Real := fun h =>
    dotProduct (fun j => (wIn j h : Real)) x + (bIn h : Real)
  have hpre_lower : ∀ h, (preLo h : Real) ≤ pre h := by
    intro h
    simpa [pre, preLo] using
      add_le_add_right
        (dotIntervalLower_le_dotProduct_real (v := fun j => wIn j h) lo hi x hlo hhi)
        (bIn h : Real)
  have hpre_upper : ∀ h, pre h ≤ (preHi h : Real) := by
    intro h
    simpa [pre, preHi] using
      add_le_add_right
        (dotProduct_le_dotIntervalUpper_real (v := fun j => wIn j h) lo hi x hlo hhi)
        (bIn h : Real)
  let geluBounds : Fin hidden → Dyadic × Dyadic := fun h => geluInterval (preLo h) (preHi h)
  let geluLo : Fin hidden → Dyadic := fun h => (geluBounds h).1
  let geluHi : Fin hidden → Dyadic := fun h => (geluBounds h).2
  let hidden : Fin hidden → Real := fun h => geluTanh (pre h)
  have hgelu : ∀ h, (geluLo h : Real) ≤ hidden h ∧ hidden h ≤ (geluHi h : Real) := by
    intro h
    have hbounds := geluInterval_bounds (lo := preLo h) (hi := preHi h)
      (hpre_lower h) (hpre_upper h)
    simpa [geluLo, geluHi, geluBounds, hidden] using hbounds
  let outLo : Fin dModel → Dyadic := fun i =>
    dotIntervalLower (fun h => wOut h i) geluLo geluHi + bOut i
  let outHi : Fin dModel → Dyadic := fun i =>
    dotIntervalUpper (fun h => wOut h i) geluLo geluHi + bOut i
  have hout_lower :
      (outLo i : Real) ≤ dotProduct (fun h => (wOut h i : Real)) hidden + (bOut i : Real) := by
    simpa [outLo] using
      add_le_add_right
        (dotIntervalLower_le_dotProduct_real (v := fun h => wOut h i) geluLo geluHi hidden
          (fun h => (hgelu h).1) (fun h => (hgelu h).2))
        (bOut i : Real)
  have hout_upper :
      dotProduct (fun h => (wOut h i : Real)) hidden + (bOut i : Real) ≤ (outHi i : Real) := by
    simpa [outHi] using
      add_le_add_right
        (dotProduct_le_dotIntervalUpper_real (v := fun h => wOut h i) geluLo geluHi hidden
          (fun h => (hgelu h).1) (fun h => (hgelu h).2))
        (bOut i : Real)
  have hlo' : (outLo i : Real) ≤ mlpReal wIn bIn wOut bOut x i := by
    simpa [mlpReal, hidden, pre] using hout_lower
  have hhi' : mlpReal wIn bIn wOut bOut x i ≤ (outHi i : Real) := by
    simpa [mlpReal, hidden, pre] using hout_upper
  simpa [bounds, mlpBounds, preLo, preHi, geluLo, geluHi, outLo, outHi] using
    And.intro hlo' hhi'

/-- Interval bounds for a LayerNorm + MLP sublayer from exact inputs. -/
def layerNormMlpBounds {n hidden : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic)
    (wIn : Fin n → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin n → Dyadic) (bOut : Fin n → Dyadic)
    (x : Fin n → Dyadic) : (Fin n → Dyadic) × (Fin n → Dyadic) :=
  let ln := layerNormBounds eps gamma beta x
  mlpBounds wIn bIn wOut bOut ln.1 ln.2

/-- `layerNormMlpBounds` soundness for real LayerNorm + MLP outputs. -/
theorem layerNormMlpBounds_spec {n hidden : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic)
    (wIn : Fin n → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin n → Dyadic) (bOut : Fin n → Dyadic)
    (x : Fin n → Dyadic) (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
    let bounds := layerNormMlpBounds eps gamma beta wIn bIn wOut bOut x
    ∀ i, (bounds.1 i : Real) ≤
        mlpReal wIn bIn wOut bOut (layerNormReal eps gamma beta x) i ∧
      mlpReal wIn bIn wOut bOut (layerNormReal eps gamma beta x) i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let ln := layerNormBounds eps gamma beta x
  have hln := layerNormBounds_spec eps gamma beta x hne heps hsqrt
  have hlo : ∀ j, (ln.1 j : Real) ≤ layerNormReal eps gamma beta x j := fun j => (hln j).1
  have hhi : ∀ j, layerNormReal eps gamma beta x j ≤ (ln.2 j : Real) := fun j => (hln j).2
  have hmlp := mlpBounds_spec wIn bIn wOut bOut ln.1 ln.2
    (layerNormReal eps gamma beta x) hlo hhi
  simpa [bounds, layerNormMlpBounds, ln] using hmlp i

/-- Interval bounds for LayerNorm + MLP sublayer from interval inputs. -/
def layerNormAbsMlpBounds {n hidden : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic)
    (wIn : Fin n → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin n → Dyadic) (bOut : Fin n → Dyadic)
    (lo hi : Fin n → Dyadic) : (Fin n → Dyadic) × (Fin n → Dyadic) :=
  let absBound := intervalAbsBound lo hi
  let ln := layerNormAbsBounds eps gamma beta absBound
  mlpBounds wIn bIn wOut bOut ln.1 ln.2

/-- `layerNormAbsMlpBounds` soundness for real LayerNorm + MLP outputs. -/
theorem layerNormAbsMlpBounds_spec {n hidden : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic)
    (wIn : Fin n → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin n → Dyadic) (bOut : Fin n → Dyadic)
    (lo hi : Fin n → Dyadic) (x : Fin n → Real)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
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
    have hsup : max |lo j| |hi j| ≤ intervalAbsBound lo hi :=
      max_abs_le_intervalAbsBound lo hi j
    have hsup_real :
        max |(lo j : Real)| |(hi j : Real)| ≤ (absBound : Real) := by
      have hsup' :
          dyadicToReal (max |lo j| |hi j|) ≤ dyadicToReal absBound :=
        dyadicToReal_le_of_le hsup
      simpa [dyadicToReal_abs, dyadicToReal_max] using hsup'
    exact le_trans hbound hsup_real
  have hln :=
    layerNormAbsBounds_spec_real eps gamma beta absBound x hne heps hsqrt habs
  have hlo_ln : ∀ j, (ln.1 j : Real) ≤ layerNormRealOfReal eps gamma beta x j := fun j => (hln j).1
  have hhi_ln : ∀ j, layerNormRealOfReal eps gamma beta x j ≤ (ln.2 j : Real) := fun j => (hln j).2
  have hmlp := mlpBounds_spec wIn bIn wOut bOut ln.1 ln.2
    (layerNormRealOfReal eps gamma beta x) hlo_ln hhi_ln
  simpa [bounds, layerNormAbsMlpBounds, absBound, ln] using hmlp i

/-- Add residual inputs to interval bounds. -/
def residualAddBounds {n : Nat} (x : Fin n → Dyadic) (lo hi : Fin n → Dyadic) :
    (Fin n → Dyadic) × (Fin n → Dyadic) :=
  (fun i => x i + lo i, fun i => x i + hi i)

/-- `residualAddBounds` soundness for residual addition. -/
theorem residualAddBounds_spec {n : Nat} (x : Fin n → Dyadic)
    (lo hi : Fin n → Dyadic) (y : Fin n → Real)
    (hlo : ∀ i, (lo i : Real) ≤ y i) (hhi : ∀ i, y i ≤ (hi i : Real)) :
    let bounds := residualAddBounds x lo hi
    ∀ i, (bounds.1 i : Real) ≤ (x i : Real) + y i ∧
      (x i : Real) + y i ≤ (bounds.2 i : Real) := by
  intro bounds i
  have hlow := add_le_add_left (hlo i) (x i : Real)
  have hhigh := add_le_add_left (hhi i) (x i : Real)
  constructor
  · simpa [bounds, residualAddBounds] using hlow
  · simpa [bounds, residualAddBounds] using hhigh

/-- Interval bounds for a full MLP residual path (LayerNorm + MLP + residual add). -/
def layerNormMlpResidualBounds {n hidden : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic)
    (wIn : Fin n → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin n → Dyadic) (bOut : Fin n → Dyadic)
    (x : Fin n → Dyadic) : (Fin n → Dyadic) × (Fin n → Dyadic) :=
  let mlp := layerNormMlpBounds eps gamma beta wIn bIn wOut bOut x
  residualAddBounds x mlp.1 mlp.2

/-- `layerNormMlpResidualBounds` soundness for the MLP residual path. -/
theorem layerNormMlpResidualBounds_spec {n hidden : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic)
    (wIn : Fin n → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin n → Dyadic) (bOut : Fin n → Dyadic)
    (x : Fin n → Dyadic) (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
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
  have hmlp := layerNormMlpBounds_spec eps gamma beta wIn bIn wOut bOut x hne heps hsqrt
  have hres := residualAddBounds_spec x mlp.1 mlp.2
    (mlpReal wIn bIn wOut bOut (layerNormReal eps gamma beta x))
    (fun j => (hmlp j).1) (fun j => (hmlp j).2)
  simpa [bounds, layerNormMlpResidualBounds, mlp] using hres i

/-- Interval bounds for a full MLP residual path (LayerNorm + MLP + residual add) from intervals. -/
def layerNormAbsMlpResidualBounds {n hidden : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic)
    (wIn : Fin n → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin n → Dyadic) (bOut : Fin n → Dyadic)
    (lo hi : Fin n → Dyadic) : (Fin n → Dyadic) × (Fin n → Dyadic) :=
  let mlp := layerNormAbsMlpBounds eps gamma beta wIn bIn wOut bOut lo hi
  (fun i => lo i + mlp.1 i, fun i => hi i + mlp.2 i)

/-- `layerNormAbsMlpResidualBounds` soundness for the MLP residual path. -/
theorem layerNormAbsMlpResidualBounds_spec {n hidden : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic)
    (wIn : Fin n → Fin hidden → Dyadic) (bIn : Fin hidden → Dyadic)
    (wOut : Fin hidden → Fin n → Dyadic) (bOut : Fin n → Dyadic)
    (lo hi : Fin n → Dyadic) (x : Fin n → Real)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
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
  have hmlp :=
    layerNormAbsMlpBounds_spec eps gamma beta wIn bIn wOut bOut lo hi x hne heps hsqrt hlo hhi
  have hlo' := (hmlp i).1
  have hhi' := (hmlp i).2
  have hlow := add_le_add (hlo i) hlo'
  have hhigh := add_le_add (hhi i) hhi'
  constructor
  · simpa [bounds, layerNormAbsMlpResidualBounds, mlp] using hlow
  · simpa [bounds, layerNormAbsMlpResidualBounds, mlp] using hhigh

end Bounds

end Sound

end Nfp
