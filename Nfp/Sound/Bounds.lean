-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Mathlib.Algebra.Order.Floor.Semiring
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Rat.Floor
import Nfp.Sound.Activation
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

/-! ### GeLU derivative bounds -/

/-- Global conservative GeLU derivative bound (independent of interval). -/
def geluDerivBoundGlobal : GeluDerivTarget → Rat
  | .tanh => 2
  | .exact => 2

theorem geluDerivBoundGlobal_def (t : GeluDerivTarget) :
    geluDerivBoundGlobal t = match t with | .tanh => 2 | .exact => 2 := rfl

/-- Clamp a rational to the unit interval `[0,1]`. -/
private def clamp01 (x : Rat) : Rat :=
  max 0 (min x 1)

theorem clamp01_def (x : Rat) : clamp01 x = max 0 (min x 1) := rfl

/-- Local upper bound on the row-sum softmax Jacobian norm given `p ∈ [pLo, pHi]`. -/
def softmaxJacobianNormInfBound (pLo pHi : Rat) : Rat :=
  let lo0 := min pLo pHi
  let hi0 := max pLo pHi
  let lo := clamp01 lo0
  let hi := clamp01 hi0
  if hi < lo then
    0
  else
    let half : Rat := (1 : Rat) / 2
    let f : Rat → Rat := fun p => (2 : Rat) * p * (1 - p)
    if lo ≤ half ∧ half ≤ hi then
      half
    else
      max (f lo) (f hi)

theorem softmaxJacobianNormInfBound_def (pLo pHi : Rat) :
    softmaxJacobianNormInfBound pLo pHi =
      let lo0 := min pLo pHi
      let hi0 := max pLo pHi
      let lo := clamp01 lo0
      let hi := clamp01 hi0
      if hi < lo then
        0
      else
        let half : Rat := (1 : Rat) / 2
        let f : Rat → Rat := fun p => (2 : Rat) * p * (1 - p)
        if lo ≤ half ∧ half ≤ hi then
          half
        else
          max (f lo) (f hi) := rfl

/-! ### Exp lower bounds (scaled Taylor + squaring) -/

/-- Power function on `Rat` for natural exponents. -/
private def ratPow (x : Rat) : Nat → Rat
  | 0 => 1
  | n + 1 => ratPow x n * x

theorem ratPow_def (x : Rat) (n : Nat) :
    ratPow x n = match n with | 0 => 1 | n + 1 => ratPow x n * x := by
  cases n <;> rfl

/-- Factorial as a rational. -/
private def ratFactorial (n : Nat) : Rat := (Nat.factorial n : Nat)

theorem ratFactorial_def (n : Nat) : ratFactorial n = (Nat.factorial n : Nat) := rfl

/-- Taylor partial sum for `exp` (all terms are nonnegative when `x ≥ 0`). -/
private def expTaylorLowerBound (x : Rat) (deg : Nat) : Rat :=
  Finset.sum (Finset.range (deg + 1)) fun k => ratPow x k / ratFactorial k

theorem expTaylorLowerBound_def (x : Rat) (deg : Nat) :
    expTaylorLowerBound x deg =
      Finset.sum (Finset.range (deg + 1)) fun k => ratPow x k / ratFactorial k := rfl

/-- Lower bound on `exp` via scaled Taylor partial sums and repeated squaring. -/
def expLBScaledTaylor (x : Rat) (deg scalePow : Nat) : Rat :=
  if x < 0 then
    0
  else
    let scale : Rat := (Nat.pow 2 scalePow : Nat)
    let z := x / scale
    let t := expTaylorLowerBound z deg
    ratPow t (Nat.pow 2 scalePow)

theorem expLBScaledTaylor_def (x : Rat) (deg scalePow : Nat) :
    expLBScaledTaylor x deg scalePow =
      if x < 0 then
        0
      else
        let scale : Rat := (Nat.pow 2 scalePow : Nat)
        let z := x / scale
        let t := expTaylorLowerBound z deg
        ratPow t (Nat.pow 2 scalePow) := rfl

/-- Default portfolio of `(scalePow, taylorDeg)` candidates for `expLB`. -/
def expLBPortfolio : Array (Nat × Nat) :=
  #[(2, 4), (3, 6), (4, 8)]

theorem expLBPortfolio_def : expLBPortfolio = #[(2, 4), (3, 6), (4, 8)] := rfl

/-- Portfolio lower bound on `exp`, with a baseline `1 + x` candidate. -/
def expLB (x : Rat) (effort : Nat) : Rat :=
  let base : Rat := max 0 ((1 : Rat) + x)
  let limit := min effort expLBPortfolio.size
  Id.run do
    let mut best := base
    for i in [:limit] do
      let cand := expLBScaledTaylor x (expLBPortfolio[i]!).2 (expLBPortfolio[i]!).1
      best := max best cand
    return best

theorem expLB_def (x : Rat) (effort : Nat) :
    expLB x effort =
      let base : Rat := max 0 ((1 : Rat) + x)
      let limit := min effort expLBPortfolio.size
      Id.run do
        let mut best := base
        for i in [:limit] do
          let cand := expLBScaledTaylor x (expLBPortfolio[i]!).2 (expLBPortfolio[i]!).1
          best := max best cand
        return best := rfl

/-- Default effort used for margin-derived softmax bounds. -/
def defaultSoftmaxExpEffort : Nat := 1

theorem defaultSoftmaxExpEffort_def : defaultSoftmaxExpEffort = 1 := rfl

/-! ### Margin-derived softmax bounds -/

/-- Lower bound on the maximum softmax probability from a logit margin.

Uses a portfolio `expLB` to lower bound `exp(m)` and maps it to
`p_max ≥ exp(m) / (exp(m) + (n-1))` for `m > 0`, with `n = seqLen`. -/
def softmaxMaxProbLowerBound (seqLen : Nat) (margin : Rat) (expEffort : Nat) : Rat :=
  if seqLen = 0 then
    0
  else if margin > 0 then
    let nRat : Rat := (seqLen : Nat)
    let e := expLB margin expEffort
    e / (e + (nRat - 1))
  else
    0

theorem softmaxMaxProbLowerBound_def (seqLen : Nat) (margin : Rat) (expEffort : Nat) :
    softmaxMaxProbLowerBound seqLen margin expEffort =
      if seqLen = 0 then
        0
      else if margin > 0 then
        let nRat : Rat := (seqLen : Nat)
        let e := expLB margin expEffort
        e / (e + (nRat - 1))
      else
        0 := rfl

/-- Lower bound on total target softmax weight from a logit margin.

If at least `targetCount` logits exceed the rest by `margin`, then the total
target weight is at least `t*exp(m)/(t*exp(m)+(n-t))`. -/
def softmaxTargetWeightLowerBound (seqLen targetCount : Nat) (margin : Rat)
    (expEffort : Nat) : Rat :=
  if seqLen = 0 || targetCount = 0 then
    0
  else if margin > 0 then
    let nRat : Rat := (seqLen : Nat)
    let tRat : Rat := (targetCount : Nat)
    let base := tRat / nRat
    let e := expLB margin expEffort
    let cand := (tRat * e) / (tRat * e + (nRat - tRat))
    max base cand
  else
    0

theorem softmaxTargetWeightLowerBound_def (seqLen targetCount : Nat) (margin : Rat)
    (expEffort : Nat) :
    softmaxTargetWeightLowerBound seqLen targetCount margin expEffort =
      if seqLen = 0 || targetCount = 0 then
        0
      else if margin > 0 then
        let nRat : Rat := (seqLen : Nat)
        let tRat : Rat := (targetCount : Nat)
        let base := tRat / nRat
        let e := expLB margin expEffort
        let cand := (tRat * e) / (tRat * e + (nRat - tRat))
        max base cand
      else
        0 := rfl

/-- Upper bound on the row-sum softmax Jacobian norm from a max-probability lower bound.

If the maximum probability is at least `pLo` and `pLo > 1/2`, then every row
satisfies `2 p (1-p) ≤ 2 pLo (1-pLo)`; otherwise the universal `1/2` bound applies. -/
def softmaxJacobianNormInfBoundFromMaxProb (pLo : Rat) : Rat :=
  let half : Rat := (1 : Rat) / 2
  let p := clamp01 pLo
  if p > half then
    (2 : Rat) * p * (1 - p)
  else
    half

theorem softmaxJacobianNormInfBoundFromMaxProb_def (pLo : Rat) :
    softmaxJacobianNormInfBoundFromMaxProb pLo =
      let half : Rat := (1 : Rat) / 2
      let p := clamp01 pLo
      if p > half then
        (2 : Rat) * p * (1 - p)
      else
        half := rfl

/-- Upper bound on the row-sum softmax Jacobian norm from a logit margin. -/
def softmaxJacobianNormInfBoundFromMargin (seqLen : Nat) (margin : Rat) (expEffort : Nat) : Rat :=
  softmaxJacobianNormInfBoundFromMaxProb (softmaxMaxProbLowerBound seqLen margin expEffort)

theorem softmaxJacobianNormInfBoundFromMargin_def (seqLen : Nat) (margin : Rat)
    (expEffort : Nat) :
    softmaxJacobianNormInfBoundFromMargin seqLen margin expEffort =
      softmaxJacobianNormInfBoundFromMaxProb
        (softmaxMaxProbLowerBound seqLen margin expEffort) := rfl

/-! ### Attention pattern-term helpers -/

/-- Upper bound on `sqrt(n)` using `Nat.sqrt` (floor) plus one. -/
def sqrtUpperNat (n : Nat) : Nat := Nat.sqrt n + 1

theorem sqrtUpperNat_def (n : Nat) : sqrtUpperNat n = Nat.sqrt n + 1 := rfl

/-- Upper bound on `sqrt(n)` as a rational. -/
def sqrtUpperRat (n : Nat) : Rat := (sqrtUpperNat n : Nat)

theorem sqrtUpperRat_def (n : Nat) : sqrtUpperRat n = (sqrtUpperNat n : Nat) := rfl

/-- Upper bound on `1 / sqrt(n)` using `Nat.sqrt` (floor). -/
def invSqrtUpperBound (n : Nat) : Rat :=
  if n = 0 then 0 else (1 : Rat) / (Nat.sqrt n : Nat)

theorem invSqrtUpperBound_def (n : Nat) :
    invSqrtUpperBound n = if n = 0 then 0 else (1 : Rat) / (Nat.sqrt n : Nat) := rfl

/-- Conservative bound on `max |LayerNorm(x)|` after affine (uses only `γ`, `β`, and `dim`). -/
def layerNormOutputMaxAbsBound (dim : Nat) (maxAbsGamma maxAbsBeta : Rat) : Rat :=
  maxAbsGamma * sqrtUpperRat dim + maxAbsBeta

theorem layerNormOutputMaxAbsBound_def (dim : Nat) (maxAbsGamma maxAbsBeta : Rat) :
    layerNormOutputMaxAbsBound dim maxAbsGamma maxAbsBeta =
      maxAbsGamma * sqrtUpperRat dim + maxAbsBeta := rfl

/-- Score-gradient L1 bound for attention pattern terms. -/
def attnScoreGradBound (seqLen modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound : Rat) : Rat :=
  let scale := invSqrtUpperBound headDim
  (seqLen : Rat) * scale *
    ((2 : Rat) * (modelDim : Rat) * ln1OutMaxAbs * wqBound * wkBound)

theorem attnScoreGradBound_def (seqLen modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound : Rat) :
    attnScoreGradBound seqLen modelDim headDim ln1OutMaxAbs wqBound wkBound =
      let scale := invSqrtUpperBound headDim
      (seqLen : Rat) * scale *
        ((2 : Rat) * (modelDim : Rat) * ln1OutMaxAbs * wqBound * wkBound) := rfl

/-- Pattern-term coefficient bound from value and score-gradient bounds. -/
def attnPatternCoeffBound (seqLen modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound valueCoeff : Rat) : Rat :=
  let inputL1 := (modelDim : Rat) * ln1OutMaxAbs
  (seqLen : Rat) *
    attnScoreGradBound seqLen modelDim headDim ln1OutMaxAbs wqBound wkBound *
      (inputL1 * valueCoeff)

theorem attnPatternCoeffBound_def (seqLen modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound valueCoeff : Rat) :
    attnPatternCoeffBound seqLen modelDim headDim ln1OutMaxAbs wqBound wkBound valueCoeff =
      let inputL1 := (modelDim : Rat) * ln1OutMaxAbs
      (seqLen : Rat) *
        attnScoreGradBound seqLen modelDim headDim ln1OutMaxAbs wqBound wkBound *
          (inputL1 * valueCoeff) := rfl

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

/-- Dyadic lower bound on `sqrt (max x 0)` as a `Rat`. -/
private def sqrtLowerDyadicRat (x : Rat) (precBits : Nat) : Rat :=
  (sqrtLowerDyadic x precBits).rat

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
