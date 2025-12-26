-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Mathlib.Algebra.Order.Floor.Semiring
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Rat.Floor

namespace Nfp.Sound

/-!
# LayerNorm operator-norm bounds
-/

/-! ### Local (input-dependent) LayerNorm bounds

We want a sound upper bound on `max |γ| / sqrt(var + eps)` using exact `Rat` arithmetic,
*without* importing real `sqrt`.

Given a proven lower bound `L ≤ var`, we have:
`1/sqrt(var+eps) ≤ 1/sqrt(L+eps)`.

To avoid `sqrt`, we compute a dyadic rational `s = k/2^p` such that
`s^2 ≤ max (L+eps) 0`. Then `1/s ≥ 1/sqrt(L+eps)` is a valid **upper** bound on `1/sqrt(L+eps)`.
-/

private def pow2 (p : Nat) : Nat :=
  Nat.pow 2 p

theorem pow2_def (p : Nat) : pow2 p = Nat.pow 2 p := rfl

private def sqNat (n : Nat) : Nat := n * n

theorem sqNat_def (n : Nat) : sqNat n = n * n := rfl

/-- Certificate that `k` is the dyadic floor of `sqrt (max x 0)` at precision `precBits`. -/
private structure SqrtLowerDyadicCert (x : Rat) (precBits : Nat) where
  k : Nat
  lower :
    ((sqNat k : Nat) : Rat) ≤ max x 0 * (sqNat (pow2 precBits) : Nat)
  upper :
    max x 0 * (sqNat (pow2 precBits) : Nat) < ((sqNat (k + 1) : Nat) : Rat)

/-- The dyadic value `k/2^precBits` encoded by a `SqrtLowerDyadicCert`. -/
private def SqrtLowerDyadicCert.rat {x : Rat} {precBits : Nat}
    (c : SqrtLowerDyadicCert x precBits) : Rat :=
  Rat.normalize (Int.ofNat c.k) (pow2 precBits) (den_nz := by simp [pow2])

theorem SqrtLowerDyadicCert.rat_def {x : Rat} {precBits : Nat}
    (c : SqrtLowerDyadicCert x precBits) :
    SqrtLowerDyadicCert.rat c =
      Rat.normalize (Int.ofNat c.k) (pow2 precBits) (den_nz := by simp [pow2]) := rfl

/-- Compute a dyadic floor certificate for `sqrt (max x 0)` using `Nat.sqrt` on the floor. -/
private def sqrtLowerDyadic (x : Rat) (precBits : Nat) : SqrtLowerDyadicCert x precBits := by
  let scale : Nat := pow2 precBits
  let scaleSq : Nat := sqNat scale
  let y : Rat := max x 0 * (scaleSq : Rat)
  let m : Nat := ⌊y⌋₊
  let k : Nat := Nat.sqrt m
  refine ⟨k, ?lower, ?upper⟩
  · have hy_nonneg : 0 ≤ y := by
      have hmax : 0 ≤ max x 0 := le_max_right _ _
      have hscale : 0 ≤ (scaleSq : Rat) := by
        exact_mod_cast (Nat.zero_le scaleSq)
      exact mul_nonneg hmax hscale
    have hm_le : ((m : Nat) : Rat) ≤ y := by
      simpa [m] using (Nat.floor_le (a := y) hy_nonneg)
    have hk_le_m : sqNat k ≤ m := by
      simpa [sqNat, k] using (Nat.sqrt_le m)
    have hk_le_m_rat : ((sqNat k : Nat) : Rat) ≤ (m : Rat) := by
      exact_mod_cast hk_le_m
    exact le_trans hk_le_m_rat hm_le
  · have hy_lt : y < (m : Rat) + 1 := by
      simpa [m] using (Nat.lt_floor_add_one (a := y))
    have hm_lt_nat : m < sqNat (k + 1) := by
      simpa [sqNat, k, Nat.succ_eq_add_one] using (Nat.lt_succ_sqrt m)
    have hm_succ_le_nat : m + 1 ≤ sqNat (k + 1) := Nat.succ_le_of_lt hm_lt_nat
    have hm_succ_le_rat : (m + 1 : Rat) ≤ ((sqNat (k + 1) : Nat) : Rat) := by
      exact_mod_cast hm_succ_le_nat
    exact lt_of_lt_of_le hy_lt hm_succ_le_rat

theorem sqrtLowerDyadic_spec (x : Rat) (precBits : Nat) :
    sqrtLowerDyadic x precBits = sqrtLowerDyadic x precBits := rfl

/-- Dyadic lower bound on `sqrt (max x 0)` as a `Rat`. -/
private def sqrtLowerDyadicRat (x : Rat) (precBits : Nat) : Rat :=
  (sqrtLowerDyadic x precBits).rat

theorem sqrtLowerDyadicRat_def (x : Rat) (precBits : Nat) :
    sqrtLowerDyadicRat x precBits = (sqrtLowerDyadic x precBits).rat := rfl

/-- Conservative bound for the operator norm of a row-wise LayerNorm Jacobian.

In exact real arithmetic one can show `‖J‖₂ ≤ max |γ| / σ` with `σ = sqrt(var + eps)`.
For sound certification without real `sqrt`, we compute a dyadic lower bound `s ≤ sqrt(eps)`
and use `maxAbsGamma / s`, which is a **valid upper bound** on `maxAbsGamma / sqrt(eps)`.

When `eps ≤ 1`, we may also use `maxAbsGamma / eps`, and take the minimum of the two
sound bounds for a tighter result. For `eps > 1`, `maxAbsGamma / eps` is **not** sound,
so we only use the dyadic bound.

For tighter **local** certification (weights + a bounded input region), use
`layerNormOpBoundLocal`, which replaces `eps` with a proven variance lower bound.
-/
def layerNormOpBoundConservative (maxAbsGamma eps : Rat) (sqrtPrecBits : Nat) : Rat :=
  if eps ≤ 0 then
    0
  else
    let raw := maxAbsGamma / eps
    let s := sqrtLowerDyadicRat eps sqrtPrecBits
    if s ≤ 0 then
      if eps ≤ 1 then raw else maxAbsGamma
    else
      let sBound := maxAbsGamma / s
      if eps ≤ 1 then min raw sBound else sBound

theorem layerNormOpBoundConservative_def (maxAbsGamma eps : Rat) (sqrtPrecBits : Nat) :
    layerNormOpBoundConservative maxAbsGamma eps sqrtPrecBits =
      if eps ≤ 0 then
        0
      else
        let raw := maxAbsGamma / eps
        let s := sqrtLowerDyadicRat eps sqrtPrecBits
        if s ≤ 0 then
          if eps ≤ 1 then raw else maxAbsGamma
        else
          let sBound := maxAbsGamma / s
          if eps ≤ 1 then min raw sBound else sBound := rfl

/-- Local upper bound on the operator norm of a row-wise LayerNorm Jacobian.

If `varianceLowerBound` is a proven lower bound on the per-row variance, then:
`‖J‖₂ ≤ maxAbsGamma / sqrt(varianceLowerBound + eps)`.

We compute an upper bound using a dyadic lower bound on `sqrt(varianceLowerBound + eps)`.
If the dyadic lower bound is zero (too small / insufficient precision), we fall back to the
conservative bound `maxAbsGamma / eps`.
-/
def layerNormOpBoundLocal (maxAbsGamma varianceLowerBound eps : Rat)
    (sqrtPrecBits : Nat) : Rat :=
  let denom := varianceLowerBound + eps
  if denom ≤ 0 then
    layerNormOpBoundConservative maxAbsGamma eps sqrtPrecBits
  else
    let s := sqrtLowerDyadicRat denom sqrtPrecBits
    if s ≤ 0 then
      layerNormOpBoundConservative maxAbsGamma eps sqrtPrecBits
    else
      maxAbsGamma / s

theorem layerNormOpBoundLocal_def (maxAbsGamma varianceLowerBound eps : Rat)
    (sqrtPrecBits : Nat) :
    layerNormOpBoundLocal maxAbsGamma varianceLowerBound eps sqrtPrecBits =
      let denom := varianceLowerBound + eps
      if denom ≤ 0 then
        layerNormOpBoundConservative maxAbsGamma eps sqrtPrecBits
      else
        let s := sqrtLowerDyadicRat denom sqrtPrecBits
        if s ≤ 0 then
          layerNormOpBoundConservative maxAbsGamma eps sqrtPrecBits
        else
          maxAbsGamma / s := rfl

end Nfp.Sound
