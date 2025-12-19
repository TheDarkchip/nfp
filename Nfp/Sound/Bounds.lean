-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Decimal

namespace Nfp.Sound

/-!
# Sound bounds in exact arithmetic

This module implements conservative bounds using exact `Rat` arithmetic.

Numeric strategy (Option A): avoid `sqrt` and any `Float`-trusted computation by using
row-sum induced norms (ℓ∞) and submultiplicativity.
-/

/-- Exact absolute value on `Rat`. -/
def ratAbs (x : Rat) : Rat :=
  if x < 0 then -x else x

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

/-- Finalize to a bound. (If the last row is partial, we still account for it.) -/
def finish (acc : RowSumAcc) : Rat :=
  max acc.maxRowSum acc.curRowSum

end RowSumAcc

/-- Compute `‖M‖∞ = maxᵢ ∑ⱼ |M[i,j]|` from a row-major array.

If the provided data has fewer than `rows*cols` entries, missing entries are treated as 0.
Extra entries are ignored.
-/
def matrixNormInfOfRowMajor (rows cols : Nat) (data : Array Rat) : Rat :=
  Id.run do
    let expected := rows * cols
    let mut acc : RowSumAcc := { rows := rows, cols := cols }
    for idx in [:min expected data.size] do
      acc := acc.feed (data[idx]!)
    return acc.finish

/-- ℓ∞ induced operator norm bound for a product.

`‖A·B‖∞ ≤ ‖A‖∞ · ‖B‖∞`.
-/
def normInfMulBound (a b : Rat) : Rat := a * b

/-- Worst-case bound on the induced ℓ∞ operator norm of a softmax Jacobian row.

For a probability row `p`, the softmax Jacobian is `J = diag(p) - p pᵀ`.
For row `i`, the absolute row-sum is:

`∑ⱼ |Jᵢⱼ| = pᵢ(1-pᵢ) + ∑_{j≠i} pᵢ pⱼ = 2 pᵢ (1-pᵢ) ≤ 1/2`.

This bound is universal (independent of sequence length).
-/
def softmaxJacobianNormInfWorst : Rat := (1 : Rat) / 2

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

private def sqNat (n : Nat) : Nat := n * n

private def leSqDyadic (k : Nat) (scaleSq : Nat) (x : Rat) : Bool :=
  -- (k/scale)^2 ≤ x  ↔  k^2 ≤ x * scale^2
  ((sqNat k : Nat) : Rat) ≤ x * (scaleSq : Nat)

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

end Nfp.Sound
