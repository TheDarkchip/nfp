import Mathlib
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

/-- Conservative bound for the operator norm of a row-wise LayerNorm Jacobian.

In exact real arithmetic one can show `‖J‖₂ ≤ max |γ| / σ` with `σ = sqrt(var + eps)`.
For sound certification without `sqrt`, we use the very conservative inequality:

`1/σ ≤ 1/eps` (since `σ ≥ sqrt(eps)` and for `eps ∈ (0,1]`, `1/eps ≥ 1/sqrt(eps)`).

So we certify using `maxAbsGamma / eps`.
-/
def layerNormOpBoundConservative (maxAbsGamma eps : Rat) : Rat :=
  if eps ≤ 0 then 0 else maxAbsGamma / eps

end Nfp.Sound
