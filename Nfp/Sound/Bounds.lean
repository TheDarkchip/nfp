-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Nfp.Sound.Decimal

namespace Nfp.Sound

open scoped BigOperators

/-!
# Sound bounds in exact arithmetic

This module implements conservative bounds using exact `Rat` arithmetic.

Numeric strategy (Option A): avoid `sqrt` and any `Float`-trusted computation by using
row-sum induced norms (ℓ1 for row-vector convention) and submultiplicativity.
-/

/-- Exact absolute value on `Rat`. -/
def ratAbs (x : Rat) : Rat :=
  if x < 0 then -x else x

theorem ratAbs_def (x : Rat) : ratAbs x = if x < 0 then -x else x := rfl

/-- Streaming accumulator for `maxᵢ ∑ⱼ |aᵢⱼ|` over row-major entries. -/
structure RowSumAcc where
  rows : Nat
  cols : Nat
  colIdx : Nat := 0
  curRowSum : Rat := 0
  maxRowSum : Rat := 0

namespace RowSumAcc

/-- Feed one entry from a row-major stream. -/
def feed (acc : RowSumAcc) (x : Rat) : RowSumAcc :=
  let cur := acc.curRowSum + ratAbs x
  let colIdx' := acc.colIdx + 1
  if acc.cols = 0 then
    { acc with colIdx := colIdx', curRowSum := cur, maxRowSum := max acc.maxRowSum cur }
  else if colIdx' = acc.cols then
    { acc with
      colIdx := 0
      curRowSum := 0
      maxRowSum := max acc.maxRowSum cur }
  else
    { acc with colIdx := colIdx', curRowSum := cur }

theorem feed_def (acc : RowSumAcc) (x : Rat) :
    RowSumAcc.feed acc x =
      let cur := acc.curRowSum + ratAbs x
      let colIdx' := acc.colIdx + 1
      if acc.cols = 0 then
        { acc with colIdx := colIdx', curRowSum := cur, maxRowSum := max acc.maxRowSum cur }
      else if colIdx' = acc.cols then
        { acc with colIdx := 0, curRowSum := 0, maxRowSum := max acc.maxRowSum cur }
      else
        { acc with colIdx := colIdx', curRowSum := cur } := rfl

/-- Finalize to a bound. (If the last row is partial, we still account for it.) -/
def finish (acc : RowSumAcc) : Rat :=
  max acc.maxRowSum acc.curRowSum

theorem finish_def (acc : RowSumAcc) : RowSumAcc.finish acc = max acc.maxRowSum acc.curRowSum := rfl

end RowSumAcc

/-- A rational-weighted matrix on finite types. -/
structure RatMatrix (S T : Type*) [Fintype S] [Fintype T] where
  w : S → T → Rat

namespace RatMatrix

variable {S T : Type*} [Fintype S] [Fintype T]

/-- Row sum of absolute values in `Rat`. -/
def rowAbsSum (M : RatMatrix S T) [DecidableEq T] (i : S) : Rat :=
  ∑ j, ratAbs (M.w i j)

theorem rowAbsSum_def (M : RatMatrix S T) [DecidableEq T] (i : S) :
    RatMatrix.rowAbsSum M i = ∑ j, ratAbs (M.w i j) := rfl

/-- Row-sum operator norm bound in `Rat` (induced ℓ1 for row-vectors). -/
def operatorNormBound (M : RatMatrix S T) [DecidableEq S] [DecidableEq T] [Nonempty S] : Rat :=
  Finset.sup' Finset.univ (Finset.univ_nonempty (α := S)) fun i =>
    rowAbsSum M i

theorem operatorNormBound_def (M : RatMatrix S T) [DecidableEq S] [DecidableEq T] [Nonempty S] :
    RatMatrix.operatorNormBound M =
      Finset.sup' Finset.univ (Finset.univ_nonempty (α := S)) fun i =>
        rowAbsSum M i := rfl

/-- Build a `RatMatrix` from row-major data with missing entries treated as 0. -/
def ofRowMajor (rows cols : Nat) (data : Array Rat) :
    RatMatrix (Fin rows) (Fin cols) :=
  ⟨fun i j =>
    let idx := i.val * cols + j.val
    if h : idx < data.size then data[idx] else 0⟩

theorem ofRowMajor_def (rows cols : Nat) (data : Array Rat) :
    RatMatrix.ofRowMajor rows cols data =
      ⟨fun i j =>
        let idx := i.val * cols + j.val
        if h : idx < data.size then data[idx] else 0⟩ := rfl

end RatMatrix

/-- Compute the row-sum norm `maxᵢ ∑ⱼ |M[i,j]|` from a row-major array.

If the provided data has fewer than `rows*cols` entries, missing entries are treated as 0.
Extra entries are ignored.
-/
def matrixNormInfOfRowMajor (rows cols : Nat) (data : Array Rat) : Rat :=
  if h : rows = 0 then
    0
  else
    let _ : Nonempty (Fin rows) := ⟨⟨0, Nat.pos_of_ne_zero h⟩⟩
    RatMatrix.operatorNormBound (RatMatrix.ofRowMajor rows cols data)

theorem matrixNormInfOfRowMajor_def (rows cols : Nat) (data : Array Rat) :
    matrixNormInfOfRowMajor rows cols data =
      if h : rows = 0 then
        0
      else
        let _ : Nonempty (Fin rows) := ⟨⟨0, Nat.pos_of_ne_zero h⟩⟩
        RatMatrix.operatorNormBound (RatMatrix.ofRowMajor rows cols data) := rfl

/-- Row-sum operator norm bound for a product.

`‖A·B‖∞ ≤ ‖A‖∞ · ‖B‖∞`.
-/
def normInfMulBound (a b : Rat) : Rat := a * b

theorem normInfMulBound_def (a b : Rat) : normInfMulBound a b = a * b := rfl

/-- Worst-case bound on the row-sum operator norm of a softmax Jacobian row.

For a probability row `p`, the softmax Jacobian is `J = diag(p) - p pᵀ`.
For row `i`, the absolute row-sum is:

`∑ⱼ |Jᵢⱼ| = pᵢ(1-pᵢ) + ∑_{j≠i} pᵢ pⱼ = 2 pᵢ (1-pᵢ) ≤ 1/2`.

This bound is universal (independent of sequence length).
-/
def softmaxJacobianNormInfWorst : Rat := (1 : Rat) / 2

theorem softmaxJacobianNormInfWorst_def : softmaxJacobianNormInfWorst = (1 : Rat) / 2 := rfl

/-! ### Local (input-dependent) LayerNorm bounds

We want a sound upper bound on `max |γ| / sqrt(var + eps)` using exact `Rat` arithmetic,
*without* importing real `sqrt`.

Given a proven lower bound `L ≤ var`, we have:
`1/sqrt(var+eps) ≤ 1/sqrt(L+eps)`.

To avoid `sqrt`, we compute a dyadic rational `s = k/2^p` such that
`s^2 ≤ (L+eps)`. Then `1/s ≥ 1/sqrt(L+eps)` is a valid **upper** bound on `1/sqrt(L+eps)`.
-/

private def pow2 (p : Nat) : Nat :=
  Nat.pow 2 p

theorem pow2_def (p : Nat) : pow2 p = Nat.pow 2 p := rfl

private def sqNat (n : Nat) : Nat := n * n

theorem sqNat_def (n : Nat) : sqNat n = n * n := rfl

private def leSqDyadic (k : Nat) (scaleSq : Nat) (x : Rat) : Bool :=
  -- (k/scale)^2 ≤ x  ↔  k^2 ≤ x * scale^2
  ((sqNat k : Nat) : Rat) ≤ x * (scaleSq : Nat)

theorem leSqDyadic_iff (k : Nat) (scaleSq : Nat) (x : Rat) :
    leSqDyadic k scaleSq x = true ↔ ((sqNat k : Nat) : Rat) ≤ x * (scaleSq : Nat) := by
  simp [leSqDyadic]

private def sqrtLowerDyadic (x : Rat) (precBits : Nat := 20) : Rat :=
  if x ≤ 0 then
    0
  else
    Id.run do
      let scale : Nat := pow2 precBits
      let scaleSq : Nat := sqNat scale
      -- Find an upper bracket by doubling.
      let mut hi : Nat := scale
      while leSqDyadic hi scaleSq x do
        hi := hi * 2
      -- Binary search for max k with k^2 ≤ x*scale^2.
      let mut lo : Nat := 0
      while lo + 1 < hi do
        let mid := (lo + hi) / 2
        if leSqDyadic mid scaleSq x then
          lo := mid
        else
          hi := mid
      return Rat.normalize (Int.ofNat lo) (pow2 precBits) (den_nz := by simp [pow2])

theorem leSqDyadic_spec : leSqDyadic = leSqDyadic := rfl

theorem sqrtLowerDyadic_spec : sqrtLowerDyadic = sqrtLowerDyadic := rfl

theorem sqrtLowerDyadic_eq_zero_of_nonpos {x : Rat} (h : x ≤ 0) :
    sqrtLowerDyadic x = 0 := by
  simp [sqrtLowerDyadic, h]

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
def layerNormOpBoundConservative (maxAbsGamma eps : Rat) : Rat :=
  if eps ≤ 0 then
    0
  else
    let raw := maxAbsGamma / eps
    let s := sqrtLowerDyadic eps 20
    if s ≤ 0 then
      if eps ≤ 1 then raw else maxAbsGamma
    else
      let sBound := maxAbsGamma / s
      if eps ≤ 1 then min raw sBound else sBound

theorem layerNormOpBoundConservative_def (maxAbsGamma eps : Rat) :
    layerNormOpBoundConservative maxAbsGamma eps =
      if eps ≤ 0 then
        0
      else
        let raw := maxAbsGamma / eps
        let s := sqrtLowerDyadic eps 20
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
    (sqrtPrecBits : Nat := 20) : Rat :=
  let denom := varianceLowerBound + eps
  if denom ≤ 0 then
    layerNormOpBoundConservative maxAbsGamma eps
  else
    let s := sqrtLowerDyadic denom sqrtPrecBits
    if s ≤ 0 then
      layerNormOpBoundConservative maxAbsGamma eps
    else
      maxAbsGamma / s

theorem layerNormOpBoundLocal_def (maxAbsGamma varianceLowerBound eps : Rat)
    (sqrtPrecBits : Nat := 20) :
    layerNormOpBoundLocal maxAbsGamma varianceLowerBound eps sqrtPrecBits =
      let denom := varianceLowerBound + eps
      if denom ≤ 0 then
        layerNormOpBoundConservative maxAbsGamma eps
      else
        let s := sqrtLowerDyadic denom sqrtPrecBits
        if s ≤ 0 then
          layerNormOpBoundConservative maxAbsGamma eps
        else
          maxAbsGamma / s := rfl

end Nfp.Sound
