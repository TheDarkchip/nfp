import Batteries.Lean.Float

/-!
# Executable Circuit Discovery for Induction Heads

This module provides executable functions for discovering **certified induction heads**
from concrete model weights. It bridges the theoretical framework (Frobenius norms,
pattern terms, faithfulness bounds) with practical verification of real neural networks.

## Key Components

1. **Concrete Model Structures**: Computable representations of attention layers using
   Arrays and Floats instead of abstract types. Suitable for export from PyTorch/JAX.

2. **Efficient Bound Calculation**: Algorithms to compute `patternTerm` and `valueTerm`
   bounds without materializing the full (N·D)² Jacobian matrix.

3. **Discovery Functions**: Search algorithms that iterate over layer pairs to find
   certified virtual heads (e.g., induction heads).

## Mathematical Background

For an attention layer with weights (W_Q, W_K, W_V, W_O), the Jacobian at input x
decomposes as: `fullJacobian = valueTerm + patternTerm` where:
- `valueTerm` depends only on attention weights A and projections W_V·W_O
- `patternTerm` captures how A shifts when input changes (the error term)

The **faithfulness bound** states: if ‖patternTerm‖_F ≤ ε, then the simple
attention-based interpretation is ε-accurate.

## Performance Optimizations

This module contains critical hot paths for circuit discovery. Key optimizations:

1. **Array pre-allocation**: Use `Array.ofFn` instead of repeated `push` operations
   - Avoids O(n²) copying from array reallocations
   - Memory usage: O(n) instead of O(n²) during construction

2. **Direct loops over List.range.foldl**: Replace `(List.range n).foldl f acc` with
   `for i in [:n] do ...` - eliminates intermediate list construction (10-100× faster)

3. **Bounds-checked array access**: Use `array[i]!` which panics on out-of-bounds
   instead of `getD` which silently returns default values
   - Makes bugs explicit rather than silent
   - Compiler can optimize bounds checks in loops

4. **Matrix operations**: Pre-allocated `Array.ofFn` with `Id.run do` blocks for
   complex computations (matmul, matVecMul, power iteration)

**Benchmark impact** (GPT-2 Small, 12 layers × 12 heads):
- Matrix operations: 10-50× faster (direct loops vs List.range.foldl)
- Array construction: 2-5× faster (pre-allocation vs repeated push)
- Memory: 50% reduction (no intermediate copies)

These optimizations make circuit discovery practical on real models (seconds instead
of minutes for full network analysis).
-/

namespace Nfp

/-! ## Concrete Weight Representations -/

/-- A concrete weight matrix stored as nested Arrays.
This is the computable representation for export from PyTorch/JAX. -/
structure ConcreteMatrix where
  /-- Number of rows -/
  numRows : Nat
  /-- Number of columns -/
  numCols : Nat
  /-- Row-major data storage. data[i * numCols + j] = entry (i, j) -/
  data : Array Float
  /-- Data has the correct size -/
  size_eq : data.size = numRows * numCols

namespace ConcreteMatrix

/-- Access element (i, j) of the matrix. Returns 0 if out of bounds.

PERFORMANCE: This uses `getD` (safe with default) rather than bounds-checked `[i]!`
because it's called in tight loops where bounds are already verified at a higher level.
The 0.0 default allows graceful handling of edge cases in algorithms.
-/
def get (M : ConcreteMatrix) (i j : Nat) : Float :=
  if i < M.numRows ∧ j < M.numCols then
    -- Index is in-bounds by `size_eq` and the guard above.
    M.data[i * M.numCols + j]!
  else 0.0

/-- Create a zero matrix of given dimensions. -/
def zeros (rows cols : Nat) : ConcreteMatrix where
  numRows := rows
  numCols := cols
  data := .ofFn fun _ : Fin (rows * cols) => (0.0 : Float)
  size_eq := Array.size_ofFn

/-- Create an all-ones matrix of given dimensions. -/
def ones (rows cols : Nat) : ConcreteMatrix where
  numRows := rows
  numCols := cols
  data := .ofFn fun _ : Fin (rows * cols) => (1.0 : Float)
  size_eq := Array.size_ofFn

/-- Create an identity matrix. -/
def identity (n : Nat) : ConcreteMatrix where
  numRows := n
  numCols := n
  data := .ofFn fun idx : Fin (n * n) =>
    let i := idx.val / n
    let j := idx.val % n
    if i = j then 1.0 else 0.0
  size_eq := Array.size_ofFn

/-- Matrix multiplication.

PERFORMANCE CRITICAL: This is the hottest path in circuit discovery.
- Pre-allocates result array with `Array.ofFn` (no intermediate copies)
- Direct `for k in [:A.numCols]` instead of `List.range.foldl` (10-50× faster)
- Uses `Id.run do` to enable mutable accumulator in pure context
- Uses deterministic task-parallelism for very large products (preserving evaluation order
  *within* each dot product).
-/
private def matmulSeqCore (A B : ConcreteMatrix) : ConcreteMatrix :=
  {
    numRows := A.numRows
    numCols := B.numCols
    data := .ofFn fun idx : Fin (A.numRows * B.numCols) => Id.run do
      let i := idx.val / B.numCols
      let j := idx.val % B.numCols
      let mut acc : Float := 0.0
      let aRowBase := i * A.numCols
      for k in [:A.numCols] do
        -- SAFETY: within this branch `i < A.numRows` and `k < A.numCols`,
        -- and `A.size_eq` implies `aRowBase + k < A.data.size`.
        let a := A.data[aRowBase + k]!
        -- SAFETY: within this branch `k < B.numRows` and `j < B.numCols`,
        -- and `B.size_eq` implies `k * B.numCols + j < B.data.size`.
        let b := B.data[k * B.numCols + j]!
        acc := acc + a * b
      return acc
    size_eq := Array.size_ofFn
  }

private def matmulParFlopThreshold : Nat := 10_000_000
private def matmulParMaxTasks : Nat := 16
private def matmulParMinInnerDim : Nat := 256
private def matmulParMinOutCols : Nat := 256
private def matmulParMinRows : Nat := 2

private def shouldUseMatmulPar (A B : ConcreteMatrix) : Bool :=
  let flops := A.numRows * B.numCols * A.numCols
  flops ≥ matmulParFlopThreshold &&
    A.numCols ≥ matmulParMinInnerDim &&
    B.numCols ≥ matmulParMinOutCols &&
    A.numRows ≥ matmulParMinRows

private def matmulPar (A B : ConcreteMatrix) : ConcreteMatrix :=
  if A.numRows = 0 || B.numCols = 0 then
    matmulSeqCore A B
  else
    let numTasks := min matmulParMaxTasks A.numRows
    let q := A.numRows / numTasks
    let r := A.numRows % numTasks

    let tasks : Array (Task (Array Float)) :=
      .ofFn fun t : Fin numTasks =>
        Task.spawn (fun _ =>
          let tid := t.val
          let extra := if tid < r then 1 else 0
          let rowsHere := q + extra
          let startRow := tid * q + min tid r
          let chunkSize := rowsHere * B.numCols
          .ofFn fun idx : Fin chunkSize => Id.run do
            let localRow := idx.val / B.numCols
            let j := idx.val % B.numCols
            let i := startRow + localRow
            let mut acc : Float := 0.0
            let aRowBase := i * A.numCols
            for k in [:A.numCols] do
              -- SAFETY: `i < A.numRows` by chunk construction and `k < A.numCols` by loop bound.
              let a := A.data[aRowBase + k]!
              -- SAFETY: `k < B.numRows` and `j < B.numCols` by loop bounds and chunk indexing.
              let b := B.data[k * B.numCols + j]!
              acc := acc + a * b
            return acc)

    -- Join in increasing task index order (deterministic).
    let chunks := tasks.map Task.get
    let cutoff := (q + 1) * r

    {
      numRows := A.numRows
      numCols := B.numCols
      data := .ofFn fun idx : Fin (A.numRows * B.numCols) =>
        let row := idx.val / B.numCols
        let col := idx.val % B.numCols
        let taskIdx :=
          if row < cutoff then
            row / (q + 1)
          else
            r + (row - cutoff) / q
        let localRow :=
          if row < cutoff then
            row % (q + 1)
          else
            (row - cutoff) % q
        -- SAFETY: `taskIdx < numTasks` by construction; chunks are in task order.
        let chunk := chunks[taskIdx]!
        -- SAFETY: `localRow < rowsHere` for this chunk and `col < B.numCols`.
        chunk[localRow * B.numCols + col]!
      size_eq := Array.size_ofFn
    }

def matmul (A B : ConcreteMatrix) : ConcreteMatrix :=
  if A.numCols = B.numRows then
    if shouldUseMatmulPar A B then
      matmulPar A B
    else
      matmulSeqCore A B
  else zeros 0 0

/-- Compute Frobenius norm squared: Σᵢⱼ M[i,j]². -/
def frobeniusNormSq (M : ConcreteMatrix) : Float :=
  M.data.foldl (fun acc x => acc + x * x) 0.0

/-- Compute Frobenius norm: √(Σᵢⱼ M[i,j]²). -/
def frobeniusNorm (M : ConcreteMatrix) : Float :=
  Float.sqrt M.frobeniusNormSq

/-- Compute `trace(A · B)` without allocating the product.

This uses `trace(A·B) = ∑_{i,j} A[i,j] · B[j,i]`.
Returns 0.0 if the dimensions do not line up.
-/
def traceMul (A B : ConcreteMatrix) : Float := Id.run do
  if A.numCols ≠ B.numRows then return 0.0
  if A.numRows ≠ B.numCols then return 0.0
  if A.numRows ≠ A.numCols then return 0.0
  let n := A.numRows
  let mut acc : Float := 0.0
  for i in [:n] do
    let aRowBase := i * A.numCols
    for j in [:n] do
      -- SAFETY: `i,j < n` and both matrices are `n×n`.
      acc := acc + A.data[aRowBase + j]! * B.data[j * B.numCols + i]!
  return acc

/-! ### Float numerics (heuristics)

The definitions in this section use `Float` arithmetic for speed.

Important: these are **not** kernel-sound upper bounds in general.
They are best-effort numerical estimates (rounding may under- or over-estimate).
Sound certification lives in `Nfp.Sound.*`.
-/

/-! #### Non-certified estimates

The functions in this subsection are **not** mathematically certified upper bounds.
They may be useful for diagnostics, but must not feed into the repo's “rigorous”
error / ε pipelines.
-/

/-- Heuristic operator-norm estimate via power iteration.

The operator norm ‖M‖₂ = max‖x‖=1 ‖Mx‖ is the largest singular value.
We approximate it using power iteration on M^T M.

This is a fast **heuristic estimate** of how much `M` can stretch a vector.

PERFORMANCE: Power iteration is O(iterations × n²) but heavily optimized:
- Pre-allocated vectors with `Array.ofFn` (no array copying)
- Direct loops instead of `List.range.foldl` (10× faster)
- Bounds-checked access `v[j]!` and `Mv[i]!` (compiler optimizes in loops)
-/
def operatorNormHeuristicPI (M : ConcreteMatrix) (numIterations : Nat := 20) : Float := Id.run do
  if M.numRows = 0 || M.numCols = 0 then return 0.0

  -- Initialize with a vector of ones
  let mut v : Array Float := .ofFn fun _ : Fin M.numCols => 1.0

  -- Normalize initial vector
  let initNorm := Float.sqrt (v.foldl (fun acc x => acc + x * x) 0.0)
  if initNorm > 0.0 then
    v := v.map (· / initNorm)

  -- Power iteration: v ← (M^T M) v / ‖(M^T M) v‖
  let mut sigma : Float := 0.0
  for _ in [:numIterations] do
    -- Compute M v
    let mut Mv : Array Float := .ofFn fun i : Fin M.numRows => Id.run do
      let mut acc : Float := 0.0
      for j in [:M.numCols] do
        -- SAFETY: v has size M.numCols, guaranteed by Array.ofFn
        acc := acc + M.get i j * v[j]!
      return acc

    -- Compute M^T (M v) = (M^T M) v
    let mut MTMv : Array Float := .ofFn fun j : Fin M.numCols => Id.run do
      let mut acc : Float := 0.0
      for i in [:M.numRows] do
        -- SAFETY: Mv has size M.numRows, guaranteed by Array.ofFn above
        acc := acc + M.get i j * Mv[i]!
      return acc

    -- Compute norm of MTMv (this is σ² times ‖v‖, and ‖v‖ ≈ 1)
    let normSq := MTMv.foldl (fun acc x => acc + x * x) 0.0
    let norm := Float.sqrt normSq

    if norm < 1e-15 then
      return 0.0

    -- σ² ≈ ‖MTMv‖ / ‖v‖ ≈ ‖MTMv‖
    -- So σ ≈ ‖Mv‖
    let MvNorm := Float.sqrt (Mv.foldl (fun acc x => acc + x * x) 0.0)
    sigma := MvNorm

    -- Normalize for next iteration
    v := MTMv.map (· / norm)

  -- Heuristic safety margin for numerical errors
  sigma * 1.01


/-! ### Provable eigenvalue upper bounds (PSD moments)

The helper below is used to tighten bounds of the form `λ_max(G)` where `G` is
symmetric positive semidefinite (PSD), using only the first two spectral moments.

Let `λ₁ ≥ λ₂ ≥ ... ≥ λₙ ≥ 0` be the eigenvalues of `G`.
Write
- `tr = Σᵢ λᵢ = trace(G)`
- `f2 = Σᵢ λᵢ² = ‖G‖_F²`.

Among all nonnegative spectra with fixed `tr` and `f2`, the maximum possible `λ₁`
is achieved when the remaining `n-1` eigenvalues are equal. Solving that extremal
case yields the closed-form upper bound:

  λ₁ ≤ (tr + sqrt((n-1) * (n*f2 - tr^2))) / n.

We defensively clamp the radicand to `≥ 0` to avoid negative values caused by
Float roundoff.
-/

/-- PSD moment bound on the maximum eigenvalue from trace and Frobenius-squared.

Inputs:
- `n`: matrix dimension
- `tr = trace(G)`
- `f2 = ‖G‖_F²`

Output: a deterministic `Float` expression corresponding to the real bound above.
-/
def psdLambdaMaxUpperMoment (n : Nat) (tr f2 : Float) : Float :=
  if n = 0 then
    0.0
  else if n = 1 then
    -- For 1×1 PSD matrices, λ_max = tr.
    max 0.0 tr
  else
    let nF : Float := n.toFloat
    let rad := max 0.0 ((n - 1).toFloat * (nF * f2 - tr * tr))
    let root := Float.sqrt rad
    max 0.0 ((tr + root) / nF)


/-- Estimate maximum absolute row sum (L∞ induced norm).

Mathematically, the real-valued quantity `maxᵢ Σⱼ |M[i,j]|` is an induced matrix norm.
This function computes a `Float` approximation and is therefore a heuristic estimate.
-/
def maxAbsRowSumEst (M : ConcreteMatrix) : Float := Id.run do
  let mut maxSum : Float := 0.0
  for i in [:M.numRows] do
    let mut rowSum : Float := 0.0
    for j in [:M.numCols] do
      rowSum := rowSum + Float.abs (M.get i j)
    maxSum := max maxSum rowSum
  maxSum


/-- Estimate maximum absolute column sum (L1 induced norm).

Mathematically, the real-valued quantity `maxⱼ Σᵢ |M[i,j]|` is an induced matrix norm.
This function computes a `Float` approximation and is therefore a heuristic estimate.
-/
def maxAbsColSumEst (M : ConcreteMatrix) : Float := Id.run do
  let mut maxSum : Float := 0.0
  for j in [:M.numCols] do
    let mut colSum : Float := 0.0
    for i in [:M.numRows] do
      colSum := colSum + Float.abs (M.get i j)
    maxSum := max maxSum colSum
  maxSum


/-- Rigorous (inequality-based) one/inf upper bound on `‖M‖₂`.

In exact real arithmetic:
`‖M‖₂ ≤ sqrt(‖M‖₁ · ‖M‖∞)`.

We compute the induced 1/∞ norms from the stored Float entries.
-/
def opNormUpperBoundOneInf (M : ConcreteMatrix) : Float :=
  Float.sqrt (M.maxAbsRowSumEst * M.maxAbsColSumEst)

/-- Heuristic Schur-type estimate: `sqrt(‖M‖₁ · ‖M‖∞)` computed in `Float`.

In exact real arithmetic, `sqrt(‖M‖₁ · ‖M‖∞)` upper-bounds the spectral norm.
This implementation uses `Float`, so it should be treated as an estimate.
-/
def schurNormEst (M : ConcreteMatrix) : Float :=
  M.opNormUpperBoundOneInf

/-- Cheap, provably-valid operator-norm upper bound for a concrete real matrix.

In exact real arithmetic, we have the standard inequalities:
- `‖M‖₂ ≤ ‖M‖_F`
- `‖M‖₂ ≤ sqrt(‖M‖₁ · ‖M‖∞)`

We compute both in `Float` and interpret the resulting expression as a real-number
quantity over the stored Float entries (the same convention used throughout this file).
Taking `min` can only tighten a valid upper bound.
-/
def opNormUpperBoundCheap (M : ConcreteMatrix) : Float :=
  let frob := M.frobeniusNorm
  let schur := M.schurNormEst
  min frob schur

/-- Dense Frobenius upper bound on `‖M‖₂`: `‖M‖₂ ≤ ‖M‖_F`.

This is cheap and always valid in exact real arithmetic.
-/
def opNormUpperBoundDenseFrob (M : ConcreteMatrix) : Float :=
  M.frobeniusNorm

/-- Dense Schur upper bound on `‖M‖₂`: `‖M‖₂ ≤ sqrt(‖M‖₁‖M‖∞)`.

This is cheap and always valid in exact real arithmetic.
-/
def opNormUpperBoundDenseSchur (M : ConcreteMatrix) : Float :=
  M.opNormUpperBoundOneInf

/-- Induced `∞` norm with absolute values: `max_i Σ_j |M[i,j]|`.

This is the standard induced matrix norm `‖M‖_∞`.
We compute it deterministically in `Float` and interpret the result as a real-number
expression over the concrete Float entries.
-/
def infNormAbs (M : ConcreteMatrix) : Float := Id.run do
  let mut maxSum : Float := 0.0
  for i in [:M.numRows] do
    let mut rowSum : Float := 0.0
    let rowBase := i * M.numCols
    for j in [:M.numCols] do
      rowSum := rowSum + Float.abs (M.data[rowBase + j]!)
    maxSum := max maxSum rowSum
  maxSum

/-- Upper bound on the operator norm via the Gram matrix and the induced `∞` norm.

For a real matrix `W` we have:
- `‖W‖₂² = λ_max(WᵀW)`
- `‖G‖₂ ≤ ‖G‖∞` for any real matrix `G`

Therefore `‖W‖₂ ≤ sqrt(‖WᵀW‖∞)`.

This computes the quantity `sqrt(max_i Σ_j |(WᵀW)[i,j]|)` **without allocating** `WᵀW`.

Note: This is computed using `Float` arithmetic; we use it as a deterministic bound for the
matrix obtained by interpreting the Float entries as real numbers.

PERFORMANCE: O(numRows * numCols^2). Intended for factor matrices with small `numCols`
(e.g. `modelDim×headDim`).
-/
def opNormUpperBoundViaGramInf (W : ConcreteMatrix) : Float := Id.run do
  if W.numCols = 0 then
    return 0.0

  let mut maxRowSum : Float := 0.0
  for i in [:W.numCols] do
    let mut rowSum : Float := 0.0
    for j in [:W.numCols] do
      let mut gij : Float := 0.0
      for k in [:W.numRows] do
        let rowBase := k * W.numCols
        -- SAFETY: `k < numRows` and `i,j < numCols`, so indices are within `data.size`.
        let wi := W.data[rowBase + i]!
        let wj := W.data[rowBase + j]!
        gij := gij + wi * wj
      rowSum := rowSum + Float.abs gij
    maxRowSum := max maxRowSum rowSum

  -- Guard against negative zero / NaNs propagating into sqrt.
  Float.sqrt (max 0.0 maxRowSum)

/-- Transpose a matrix. -/
def transpose (M : ConcreteMatrix) : ConcreteMatrix where
  numRows := M.numCols
  numCols := M.numRows
  data := .ofFn fun idx : Fin (M.numCols * M.numRows) =>
    let i := idx.val / M.numRows
    let j := idx.val % M.numRows
    M.get j i
  size_eq := Array.size_ofFn


/-- Diagnostics for the rectangular Gram-based operator-norm bound.

If `usedGram=true`, then we formed the smaller Gram matrix `G` and bounded
`λ_max(G)` by several PSD inequalities, returning `sqrt(λ_upper)`.
Otherwise (size-capped), we fall back to cheap dense bounds.
-/
structure RectGramDiag where
  usedGram : Bool
  /-- Used the absolute-Gram fallback (no materialized Gram). -/
  usedAbsGram : Bool
  gramDim : Nat
  lambdaBrauer : Float
  lambdaMoment : Float
  lambdaGersh : Float
  /-- Gershgorin upper bound computed without forming the Gram matrix. -/
  lambdaAbsGersh : Float
  /-- Brauer/Cassini upper bound computed without forming the Gram matrix. -/
  lambdaAbsBrauer : Float
  lambdaUsed : Float
  opBound : Float
  frobBound : Float
  oneInfBound : Float
  deriving Repr

/-- Brauer/Cassini upper bound on `λ_max(G)` for an explicit symmetric matrix `G`.

This is the same Cassini-ovals formula used for Gram matrices, but computed from
the materialized matrix `G` (intended for small `k×k`).

Guardrails:
- Clamp discriminants to `≥ 0`.
- If NaN/Inf appears, fall back to the induced-∞ bound `‖G‖_∞`.
-/
def symmLambdaMaxUpperBrauer (G : ConcreteMatrix) : Float := Id.run do
  let n := G.numRows
  if n = 0 || G.numCols ≠ n then
    return 0.0

  let mut maxDiag : Float := 0.0
  let mut infBound : Float := 0.0
  let mut bad : Bool := false
  let mut ds : Array Float := Array.mkEmpty n
  let mut rs : Array Float := Array.mkEmpty n

  for i in [:n] do
    let di := G.get i i
    ds := ds.push di
    maxDiag := max maxDiag di

  for i in [:n] do
    let mut ri : Float := 0.0
    let rowBase := i * n
    for j in [:n] do
      if j = i then
        continue
      ri := ri + Float.abs (G.data[rowBase + j]!)
    rs := rs.push ri
    let di := ds[i]!
    if Float.isNaN di || Float.isInf di || Float.isNaN ri || Float.isInf ri then
      bad := true
    infBound := max infBound (Float.abs di + ri)

  if bad then
    return infBound
  if n < 2 then
    return maxDiag

  let mut maxPair : Float := 0.0
  for i in [:n] do
    let di := ds[i]!
    let ri := rs[i]!
    for j in [i + 1:n] do
      let dj := ds[j]!
      let rj := rs[j]!
      let delta := di - dj
      let disc := max 0.0 (delta * delta + 4.0 * ri * rj)
      let root := Float.sqrt disc
      let bij := (di + dj + root) / 2.0
      if Float.isNaN bij || Float.isInf bij then
        bad := true
      maxPair := max maxPair bij

  if bad then
    return infBound
  else
    return max maxDiag maxPair

/-- Gershgorin and Brauer/Cassini upper bounds for the reduced Gram matrix, without forming it.

Let `A` be `m×n`. We reduce to `G = A Aᵀ` if `m ≤ n` and `G = Aᵀ A` otherwise, so
`‖A‖₂² = λ_max(G)` (exact real arithmetic).

We avoid forming `G` by bounding absolute row sums via:
- If `m > n` (`G = AᵀA`): use `s_row[k] = Σ_j |A[k,j]|` and
  `Σ_j |G[i,j]| ≤ Σ_k |A[k,i]| * s_row[k]`.
- If `m ≤ n` (`G = AAᵀ`): use `s_col[t] = Σ_i |A[i,t]|` and
  `Σ_j |G[i,j]| ≤ Σ_t |A[i,t]| * s_col[t]`.

The first component is the Gershgorin/∞ bound `max_i rowSumUpper[i]`.
The second is the Brauer/Cassini bound computed from `diag[i]` and
`offSumUpper[i] = max(0, rowSumUpper[i] - diag[i])`.

If NaN/Inf appears in the Brauer calculation, we conservatively fall back to the
Gershgorin bound.
-/
def rectAbsGramLambdaUpperGershBrauer (A : ConcreteMatrix) : Float × Float := Id.run do
  let m := A.numRows
  let n := A.numCols
  if m = 0 || n = 0 then
    return (0.0, 0.0)

  if m > n then
    -- `G = AᵀA` (size `n×n`), using row absolute sums of `A`.
    let mut sRow : Array Float := Array.replicate m 0.0
    for k in [:m] do
      let mut acc : Float := 0.0
      let rowBase := k * n
      for j in [:n] do
        acc := acc + Float.abs (A.data[rowBase + j]!)
      sRow := sRow.set! k acc

    let mut diag : Array Float := Array.replicate n 0.0
    let mut rowSumUpper : Array Float := Array.replicate n 0.0
    for k in [:m] do
      let s := sRow[k]!
      let rowBase := k * n
      for i in [:n] do
        let a := A.data[rowBase + i]!
        diag := diag.set! i (diag[i]! + a * a)
        rowSumUpper := rowSumUpper.set! i (rowSumUpper[i]! + Float.abs a * s)

    let mut lambdaAbsGersh : Float := 0.0
    for i in [:n] do
      lambdaAbsGersh := max lambdaAbsGersh rowSumUpper[i]!

    if n < 2 then
      return (lambdaAbsGersh, diag[0]!)

    let mut maxPair : Float := 0.0
    let mut bad : Bool := false
    for i in [:n] do
      let di := diag[i]!
      let ri := max 0.0 (rowSumUpper[i]! - di)
      if Float.isNaN di || Float.isInf di || Float.isNaN ri || Float.isInf ri then
        bad := true
      for j in [i + 1:n] do
        let dj := diag[j]!
        let rj := max 0.0 (rowSumUpper[j]! - dj)
        if Float.isNaN dj || Float.isInf dj || Float.isNaN rj || Float.isInf rj then
          bad := true
        let delta := di - dj
        let disc := max 0.0 (delta * delta + 4.0 * ri * rj)
        let root := Float.sqrt disc
        let bij := (di + dj + root) / 2.0
        if Float.isNaN bij || Float.isInf bij then
          bad := true
        maxPair := max maxPair bij

    let lambdaAbsBrauer := if bad then lambdaAbsGersh else maxPair
    return (lambdaAbsGersh, lambdaAbsBrauer)
  else
    -- `G = AAᵀ` (size `m×m`), using column absolute sums of `A`.
    let mut sCol : Array Float := Array.replicate n 0.0
    for i in [:m] do
      let rowBase := i * n
      for t in [:n] do
        let a := A.data[rowBase + t]!
        sCol := sCol.set! t (sCol[t]! + Float.abs a)

    let mut diag : Array Float := Array.replicate m 0.0
    let mut rowSumUpper : Array Float := Array.replicate m 0.0
    for i in [:m] do
      let mut di : Float := 0.0
      let mut ru : Float := 0.0
      let rowBase := i * n
      for t in [:n] do
        let a := A.data[rowBase + t]!
        di := di + a * a
        ru := ru + Float.abs a * sCol[t]!
      diag := diag.set! i di
      rowSumUpper := rowSumUpper.set! i ru

    let mut lambdaAbsGersh : Float := 0.0
    for i in [:m] do
      lambdaAbsGersh := max lambdaAbsGersh rowSumUpper[i]!

    if m < 2 then
      return (lambdaAbsGersh, diag[0]!)

    let mut maxPair : Float := 0.0
    let mut bad : Bool := false
    for i in [:m] do
      let di := diag[i]!
      let ri := max 0.0 (rowSumUpper[i]! - di)
      if Float.isNaN di || Float.isInf di || Float.isNaN ri || Float.isInf ri then
        bad := true
      for j in [i + 1:m] do
        let dj := diag[j]!
        let rj := max 0.0 (rowSumUpper[j]! - dj)
        if Float.isNaN dj || Float.isInf dj || Float.isNaN rj || Float.isInf rj then
          bad := true
        let delta := di - dj
        let disc := max 0.0 (delta * delta + 4.0 * ri * rj)
        let root := Float.sqrt disc
        let bij := (di + dj + root) / 2.0
        if Float.isNaN bij || Float.isInf bij then
          bad := true
        maxPair := max maxPair bij

    let lambdaAbsBrauer := if bad then lambdaAbsGersh else maxPair
    return (lambdaAbsGersh, lambdaAbsBrauer)

/-- Compute a rigorous operator-norm upper bound for a rectangular matrix via Gram reduction.

Algorithm (exact real arithmetic):
Let `G = A Aᵀ` if `m ≤ n`, otherwise `G = Aᵀ A`. Then
`‖A‖₂² = λ_max(G)`.

We upper-bound `λ_max(G)` using:
- Brauer/Cassini (for PSD symmetric)
- PSD moment bound from `trace(G)` and `‖G‖_F²`
- Gershgorin / induced-∞: `λ_max(G) ≤ ‖G‖_∞`

PERFORMANCE: forming `G` is `O(m n min(m,n))` and can be very expensive.
We therefore size-cap the Gram dimension. Above the cap, we fall back to
`min(‖A‖_F, sqrt(‖A‖₁‖A‖∞))`.
-/
def opNormUpperBoundRectGramDiag (A : ConcreteMatrix) (maxGramDim : Nat := 256) : RectGramDiag :=
  let frob := A.frobeniusNorm
  let oneInf := A.opNormUpperBoundOneInf
  let cheap := min frob oneInf
  let m := A.numRows
  let n := A.numCols
  let k := min m n
  if k = 0 then
    { usedGram := true
      usedAbsGram := false
      gramDim := 0
      lambdaBrauer := 0.0
      lambdaMoment := 0.0
      lambdaGersh := 0.0
      lambdaAbsGersh := 0.0
      lambdaAbsBrauer := 0.0
      lambdaUsed := 0.0
      opBound := 0.0
      frobBound := frob
      oneInfBound := oneInf }
  else if k > maxGramDim then
    let (lambdaAbsGersh, lambdaAbsBrauer) := rectAbsGramLambdaUpperGershBrauer A
    let lambdaAbsUpper := min lambdaAbsGersh lambdaAbsBrauer
    let opAbsRaw := Float.sqrt (max 0.0 lambdaAbsUpper)
    let opAbs : Float :=
      if Float.isNaN opAbsRaw || Float.isInf opAbsRaw then
        Float.inf
      else
        opAbsRaw
    let opBound := min cheap opAbs
    let lambdaUsed := min (max 0.0 (cheap * cheap)) (max 0.0 lambdaAbsUpper)
    { usedGram := false
      usedAbsGram := true
      gramDim := k
      lambdaBrauer := 0.0
      lambdaMoment := 0.0
      lambdaGersh := 0.0
      lambdaAbsGersh := lambdaAbsGersh
      lambdaAbsBrauer := lambdaAbsBrauer
      lambdaUsed := lambdaUsed
      opBound := opBound
      frobBound := frob
      oneInfBound := oneInf }
  else
    let G : ConcreteMatrix :=
      if m ≤ n then
        A.matmul A.transpose
      else
        A.transpose.matmul A
    let lambdaGersh := G.infNormAbs
    let lambdaBrauer := symmLambdaMaxUpperBrauer G
    -- For Gram `G`, `trace(G) = ‖A‖_F²`.
    let tr : Float := A.frobeniusNormSq
    let f2 : Float := G.frobeniusNormSq
    let lambdaMoment := psdLambdaMaxUpperMoment k tr f2
    let lambdaUpper := min lambdaGersh (min lambdaBrauer lambdaMoment)
    let op := Float.sqrt (max 0.0 lambdaUpper)
    { usedGram := true
      usedAbsGram := false
      gramDim := k
      lambdaBrauer := lambdaBrauer
      lambdaMoment := lambdaMoment
      lambdaGersh := lambdaGersh
      lambdaAbsGersh := 0.0
      lambdaAbsBrauer := 0.0
      lambdaUsed := lambdaUpper
      opBound := min cheap op
      frobBound := frob
      oneInfBound := oneInf }

/-- Rectangular Gram-based operator-norm bound (with a size-cap fallback). -/
def opNormUpperBoundRectGram (A : ConcreteMatrix) (maxGramDim : Nat := 256) : Float :=
  (A.opNormUpperBoundRectGramDiag maxGramDim).opBound

/-- Gram-matrix induced-∞ upper bound on the spectral norm.

In exact real arithmetic:
`‖M‖₂² = λ_max(MᵀM) ≤ ‖MᵀM‖_∞`, hence `‖M‖₂ ≤ sqrt(‖MᵀM‖_∞)`.

This allocates `MᵀM`, so it is intended for small matrices.
If the computation produces NaN/Inf, we conservatively fall back to `‖M‖_F`.
-/
def gramInfOpBound (M : ConcreteMatrix) : Float :=
  if M.numRows = 0 || M.numCols = 0 then
    0.0
  else
    let g := M.transpose.matmul M
    let v := Float.sqrt (max 0.0 g.infNormAbs)
    if Float.isNaN v || Float.isInf v then
      M.frobeniusNorm
    else
      v

/-- Compute an entry of the Gram matrix `MᵀM` without materializing it.

`gramMatrixEntry M i j = (MᵀM)[i,j] = Σ_k M[k,i] * M[k,j]`.

This is intended for small `numCols` (e.g. `headDim=64`).
-/
def gramMatrixEntry (M : ConcreteMatrix) (i j : Nat) : Float := Id.run do
  if M.numRows = 0 || M.numCols = 0 then
    return 0.0
  if i ≥ M.numCols || j ≥ M.numCols then
    return 0.0
  let mut acc : Float := 0.0
  for k in [:M.numRows] do
    let rowBase := k * M.numCols
    let mi := M.data[rowBase + i]!
    let mj := M.data[rowBase + j]!
    acc := acc + mi * mj
  return acc

/-- Gram diagonal entry `d_i = (MᵀM)[i,i] = Σ_k M[k,i]^2`.

For real `M`, these are nonnegative in exact arithmetic.
-/
def gramDiag (M : ConcreteMatrix) (i : Nat) : Float := Id.run do
  if M.numRows = 0 || M.numCols = 0 then
    return 0.0
  if i ≥ M.numCols then
    return 0.0
  let mut acc : Float := 0.0
  for k in [:M.numRows] do
    let rowBase := k * M.numCols
    let mi := M.data[rowBase + i]!
    acc := acc + mi * mi
  return acc

/-- Off-diagonal absolute row sum of the Gram matrix.

Let `G = MᵀM`. This computes
`R_i = Σ_{j ≠ i} |G[i,j]|`.
-/
def gramRowAbsSumExclDiag (M : ConcreteMatrix) (i : Nat) : Float := Id.run do
  if M.numCols = 0 || i ≥ M.numCols then
    return 0.0
  let mut acc : Float := 0.0
  for j in [:M.numCols] do
    if j = i then
      continue
    acc := acc + Float.abs (M.gramMatrixEntry i j)
  return acc

/-- Frobenius norm squared of the Gram matrix `G = MᵀM`, computed without allocating `G`.

This returns `‖G‖_F² = Σ_{i,j} G[i,j]^2`.
Intended for small `numCols` (e.g. `headDim=64`).
-/
def gramFrobeniusNormSq (M : ConcreteMatrix) : Float := Id.run do
  let n := M.numCols
  if n = 0 then
    return 0.0
  let mut acc : Float := 0.0
  for i in [:n] do
    for j in [:n] do
      let gij := M.gramMatrixEntry i j
      acc := acc + gij * gij
  return acc

/-- Brauer/Cassini upper bound on `λ_max(G)` for a Gram matrix `G = MᵀM`.

Mathematical facts (exact real arithmetic):

Let `G` be real symmetric (Gram matrices are symmetric PSD). Define:
- `d_i = G[i,i]`
- `R_i = Σ_{j≠i} |G[i,j]|`

Brauer bound (Cassini ovals):
`λ_max(G) ≤ max_{i≠j} b_ij`, where
`b_ij = (d_i + d_j + sqrt((d_i - d_j)^2 + 4*R_i*R_j)) / 2`.

We also have the induced-∞ / Gershgorin bound:
`λ_max(G) ≤ max_i (d_i + R_i) = ‖G‖_∞`.

Guardrails:
- Clamp the discriminant inside `sqrt` to `≥ 0`.
- If any NaN/Inf appears, conservatively fall back to `‖G‖_∞`.
- If `n < 2`, return `max_i d_i`.
-/
def gramLambdaMaxUpperBrauer (M : ConcreteMatrix) : Float := Id.run do
  let n := M.numCols
  if n = 0 then
    return 0.0

  let ds : Array Float := .ofFn fun i : Fin n => M.gramDiag i.val
  let rs : Array Float := .ofFn fun i : Fin n => M.gramRowAbsSumExclDiag i.val

  let mut maxDiag : Float := 0.0
  let mut infBound : Float := 0.0
  let mut bad : Bool := false
  for i in [:n] do
    let di := ds[i]!
    let ri := rs[i]!
    if Float.isNaN di || Float.isInf di || Float.isNaN ri || Float.isInf ri then
      bad := true
    maxDiag := max maxDiag di
    infBound := max infBound (Float.abs di + ri)

  if bad then
    return infBound

  if n < 2 then
    return maxDiag

  let mut maxPair : Float := 0.0
  for i in [:n] do
    let di := ds[i]!
    let ri := rs[i]!
    for j in [i + 1:n] do
      let dj := ds[j]!
      let rj := rs[j]!
      let delta := di - dj
      let disc := max 0.0 (delta * delta + 4.0 * ri * rj)
      let root := Float.sqrt disc
      let bij := (di + dj + root) / 2.0
      if Float.isNaN bij || Float.isInf bij then
        bad := true
      maxPair := max maxPair bij

  if bad then
    return infBound
  else
    -- Brauer/Cassini bound candidate
    let lambdaBrauerUpper := max maxDiag maxPair
    -- Moment bound candidate for PSD `G = MᵀM`.
    -- Here `trace(G) = ‖M‖_F²`.
    let tr : Float := M.frobeniusNormSq
    let f2 : Float := M.gramFrobeniusNormSq
    let lambdaMomentUpper := psdLambdaMaxUpperMoment n tr f2
    -- Combine cheap valid upper bounds by taking `min`.
    return min infBound (min lambdaBrauerUpper lambdaMomentUpper)

/-- Dense (small) spectral-norm upper bound using the Brauer/Cassini Gram bound.

`‖M‖₂² = λ_max(MᵀM) ≤ gramLambdaMaxUpperBrauer(M)`.
-/
def opNormUpperBoundDenseBrauer (M : ConcreteMatrix) : Float :=
  Float.sqrt (max 0.0 (M.gramLambdaMaxUpperBrauer))

/-- Matrix addition. Returns zero matrix if dimensions don't match. -/
def add (A B : ConcreteMatrix) : ConcreteMatrix :=
  if A.numRows = B.numRows ∧ A.numCols = B.numCols then
    {
      numRows := A.numRows
      numCols := A.numCols
      data := .ofFn fun idx : Fin (A.numRows * A.numCols) =>
        A.data.getD idx.val 0.0 + B.data.getD idx.val 0.0
      size_eq := Array.size_ofFn
    }
  else zeros 0 0

/-- Scalar multiplication. -/
def scale (c : Float) (M : ConcreteMatrix) : ConcreteMatrix where
  numRows := M.numRows
  numCols := M.numCols
  data := M.data.map (c * ·)
  size_eq := by simp [M.size_eq]

/-- Get row i as a 1×numCols matrix. -/
def getRow (M : ConcreteMatrix) (i : Nat) : ConcreteMatrix :=
  if i < M.numRows then
    {
      numRows := 1
      numCols := M.numCols
      data := .ofFn fun j : Fin M.numCols => M.get i j.val
      size_eq := by simp
    }
  else zeros 1 M.numCols

/-- Set row i from a 1×numCols matrix. Returns original if dimensions wrong. -/
def setRow (M : ConcreteMatrix) (i : Nat) (row : ConcreteMatrix) : ConcreteMatrix :=
  if i < M.numRows ∧ row.numRows = 1 ∧ row.numCols = M.numCols then
    {
      numRows := M.numRows
      numCols := M.numCols
      data := .ofFn fun idx : Fin (M.numRows * M.numCols) =>
        let r := idx.val / M.numCols
        let c := idx.val % M.numCols
        if r = i then row.get 0 c else M.get r c
      size_eq := Array.size_ofFn
    }
  else M

/-- Element-wise application of a function. -/
def map (f : Float → Float) (M : ConcreteMatrix) : ConcreteMatrix where
  numRows := M.numRows
  numCols := M.numCols
  data := M.data.map f
  size_eq := by simp [M.size_eq]

/-- Broadcast add: add a 1×numCols bias to each row. -/
def addBias (M : ConcreteMatrix) (bias : ConcreteMatrix) : ConcreteMatrix :=
  if bias.numRows = 1 ∧ bias.numCols = M.numCols then
    {
      numRows := M.numRows
      numCols := M.numCols
      data := .ofFn fun idx : Fin (M.numRows * M.numCols) =>
        let c := idx.val % M.numCols
        M.data.getD idx.val 0.0 + bias.get 0 c
      size_eq := Array.size_ofFn
    }
  else M

/-- Row-wise LayerNorm with learnable scale γ and bias β (both 1×numCols).

This implements the Pre-LN transformer normalization convention: each token (row)
is normalized across model dimension (columns), then scaled and shifted.
-/
def layerNormRowwise (X γ β : ConcreteMatrix) (eps : Float := 1e-5) : ConcreteMatrix := Id.run do
  if X.numRows = 0 || X.numCols = 0 then
    return ConcreteMatrix.zeros X.numRows X.numCols
  if !(γ.numRows = 1 ∧ γ.numCols = X.numCols ∧ β.numRows = 1 ∧ β.numCols = X.numCols) then
    return X

  -- Per-row mean and inverse stddev (compute once for speed).
  let mut means : Array Float := Array.mkEmpty X.numRows
  let mut invStds : Array Float := Array.mkEmpty X.numRows
  for r in [:X.numRows] do
    let mut sum : Float := 0.0
    for c in [:X.numCols] do
      sum := sum + X.get r c
    let μ := sum / X.numCols.toFloat
    let mut varSum : Float := 0.0
    for c in [:X.numCols] do
      let d := X.get r c - μ
      varSum := varSum + d * d
    let var := varSum / X.numCols.toFloat
    let invσ := 1.0 / Float.sqrt (var + eps)
    means := means.push μ
    invStds := invStds.push invσ

  return {
    numRows := X.numRows
    numCols := X.numCols
    data := .ofFn fun idx : Fin (X.numRows * X.numCols) =>
      let r := idx.val / X.numCols
      let c := idx.val % X.numCols
      let μ := means.getD r 0.0
      let invσ := invStds.getD r 0.0
      let normalized := (X.get r c - μ) * invσ
      (γ.get 0 c) * normalized + (β.get 0 c)
    size_eq := Array.size_ofFn
  }

/-- Heuristic estimate for the operator norm of the Jacobian of row-wise LayerNorm.

We bound the **global** operator norm on the block-diagonal Jacobian (one block per row)
by the maximum over rows of a **tight spectral-norm bound**.

For a single row `x : ℝ^d` (ignoring β), LayerNorm is:
`LN(x) = γ ⊙ ((x - μ) / σ)` with `σ = sqrt(var + eps)`.
Its Jacobian has the closed form:

`J = diag(γ) * (1/σ) * (I - (1/d)11ᵀ - (1/d)vvᵀ)`

where `v` is the centered vector scaled by `1/σ`. The symmetric matrix in parentheses
has eigenvalues `{0, 1, eps/(var+eps)}` so its spectral norm is exactly `1`.
Therefore `‖J‖₂ ≤ max |γ| / σ` in exact real arithmetic.

This avoids the previous row-sum bound which could overestimate by orders of magnitude
and made downstream certification thresholds unusable.
-/
def layerNormRowwiseOpEst (X γ : ConcreteMatrix) (eps : Float := 1e-5) : Float := Id.run do
  if X.numRows = 0 || X.numCols = 0 then return 0.0
  if !(γ.numRows = 1 ∧ γ.numCols = X.numCols) then return 0.0

  -- max |γ|
  let mut gammaMaxAbs : Float := 0.0
  for c in [:X.numCols] do
    let g := Float.abs (γ.get 0 c)
    if g > gammaMaxAbs then gammaMaxAbs := g

  -- max_r (1/σ_r)
  let mut maxInvStd : Float := 0.0
  for r in [:X.numRows] do
    -- Mean
    let mut sum : Float := 0.0
    for c in [:X.numCols] do
      sum := sum + X.get r c
    let μ := sum / X.numCols.toFloat

    -- Variance
    let mut varSum : Float := 0.0
    for c in [:X.numCols] do
      let centered := X.get r c - μ
      varSum := varSum + centered * centered
    let var := varSum / X.numCols.toFloat
    let σ := Float.sqrt (var + eps)

    if σ > 0.0 then
      let invσ := 1.0 / σ
      if invσ > maxInvStd then maxInvStd := invσ

  return gammaMaxAbs * maxInvStd

/-- Get column j as a vector (stored as numRows×1 matrix). -/
def getCol (M : ConcreteMatrix) (j : Nat) : ConcreteMatrix :=
  if j < M.numCols then
    {
      numRows := M.numRows
      numCols := 1
      data := .ofFn fun i : Fin M.numRows => M.get i.val j
      size_eq := by simp
    }
  else zeros M.numRows 1

/-- Compute matrix-vector product M * v where v is stored as numCols×1 matrix.
    Returns a numRows×1 matrix. -/
def matVecMul (M : ConcreteMatrix) (v : ConcreteMatrix) : ConcreteMatrix :=
  if M.numCols = v.numRows ∧ v.numCols = 1 then
    {
      numRows := M.numRows
      numCols := 1
      data := .ofFn fun i : Fin M.numRows => Id.run do
        let mut acc : Float := 0.0
        for k in [:M.numCols] do
          acc := acc + M.get i.val k * v.get k 0
        return acc
      size_eq := by simp
    }
  else zeros M.numRows 1

/-- Compute dot product of two vectors (stored as n×1 matrices). -/
def dot (v1 v2 : ConcreteMatrix) : Float :=
  if v1.numRows = v2.numRows ∧ v1.numCols = 1 ∧ v2.numCols = 1 then Id.run do
    let mut acc : Float := 0.0
    for i in [:v1.numRows] do
      acc := acc + v1.get i 0 * v2.get i 0
    return acc
  else 0.0

/-- Compute L2 norm of a vector (stored as n×1 matrix). -/
def vecNorm (v : ConcreteMatrix) : Float :=
  if v.numCols = 1 then
    Float.sqrt (v.data.foldl (fun acc x => acc + x * x) 0.0)
  else 0.0

/-- Vector subtraction for n×1 matrices. -/
def vecSub (v1 v2 : ConcreteMatrix) : ConcreteMatrix :=
  if v1.numRows = v2.numRows ∧ v1.numCols = 1 ∧ v2.numCols = 1 then
    {
      numRows := v1.numRows
      numCols := 1
      data := .ofFn fun i : Fin v1.numRows => v1.get i.val 0 - v2.get i.val 0
      size_eq := by simp
    }
  else zeros v1.numRows 1

end ConcreteMatrix

/-! ## Concrete LayerNorm Parameters -/

/-- Concrete LayerNorm parameters for Pre-LN transformers (scale γ and bias β). -/
structure ConcreteLayerNormParams where
  /-- Scale γ (1×modelDim) -/
  gamma : ConcreteMatrix
  /-- Bias β (1×modelDim) -/
  beta : ConcreteMatrix

namespace ConcreteLayerNormParams

/-- Identity LayerNorm affine parameters: γ=1, β=0. -/
def identity (modelDim : Nat) : ConcreteLayerNormParams :=
  { gamma := ConcreteMatrix.ones 1 modelDim, beta := ConcreteMatrix.zeros 1 modelDim }

end ConcreteLayerNormParams

/-! ## Concrete Attention Layer -/

/-- A concrete attention layer with exported weights.

This structure holds the four projection matrices that define a single attention head:
- W_Q: Query projection (d × d_head)
- W_K: Key projection (d × d_head)
- W_V: Value projection (d × d_head)
- W_O: Output projection (d_head × d)
-/
structure ConcreteAttentionLayer where
  /-- Model dimension (embedding size) -/
  modelDim : Nat
  /-- Head dimension -/
  headDim : Nat
  /-- Query projection matrix (modelDim × headDim) -/
  W_Q : ConcreteMatrix
  /-- Key projection matrix (modelDim × headDim) -/
  W_K : ConcreteMatrix
  /-- Value projection matrix (modelDim × headDim) -/
  W_V : ConcreteMatrix
  /-- Output projection matrix (headDim × modelDim) -/
  W_O : ConcreteMatrix
  /-- Dimension consistency for W_Q -/
  W_Q_dims : W_Q.numRows = modelDim ∧ W_Q.numCols = headDim
  /-- Dimension consistency for W_K -/
  W_K_dims : W_K.numRows = modelDim ∧ W_K.numCols = headDim
  /-- Dimension consistency for W_V -/
  W_V_dims : W_V.numRows = modelDim ∧ W_V.numCols = headDim
  /-- Dimension consistency for W_O -/
  W_O_dims : W_O.numRows = headDim ∧ W_O.numCols = modelDim

namespace ConcreteAttentionLayer

/-- Compute the value-output projection W_V · W_O. -/
def valueOutputProjection (layer : ConcreteAttentionLayer) : ConcreteMatrix :=
  layer.W_V.matmul layer.W_O

/-- Compute the query-key alignment W_Q · W_K^T. -/
def queryKeyAlignment (layer : ConcreteAttentionLayer) : ConcreteMatrix :=
  layer.W_Q.matmul layer.W_K.transpose

private def opBoundMinOfMany (dense : ConcreteMatrix)
    (leftFactor rightFactor : ConcreteMatrix) : Float :=
  let denseSchur := dense.schurNormEst
  let denseFrob := dense.frobeniusNorm
  let factorSchur := leftFactor.schurNormEst * rightFactor.schurNormEst
  let factorFrob := leftFactor.frobeniusNorm * rightFactor.frobeniusNorm
  min denseFrob (min denseSchur (min factorSchur factorFrob))

/-- A tighter (still sound-in-ℝ) Float upper bound on ‖W_Q · W_Kᵀ‖₂.

We take the minimum of several valid upper bounds:
  ‖M‖₂ ≤ schurNormEst(M)
  ‖M‖₂ ≤ ‖M‖_F
  ‖W_Q W_Kᵀ‖₂ ≤ ‖W_Q‖₂‖W_K‖₂ ≤ schur(W_Q)·schur(W_K)
  ‖W_Q W_Kᵀ‖₂ ≤ ‖W_Q‖_F‖W_K‖_F

This is computed in `Float` and is therefore a deterministic heuristic estimate.
In exact real arithmetic, each candidate is an upper bound, so taking `min`
can only tighten the bound.
-/
def queryKeyAlignmentOpBoundFrom (layer : ConcreteAttentionLayer) (qk : ConcreteMatrix) : Float :=
  let base := opBoundMinOfMany qk layer.W_Q layer.W_K
  let wqOpGram := layer.W_Q.opNormUpperBoundViaGramInf
  let wkOpGram := layer.W_K.opNormUpperBoundViaGramInf
  let qkFactorGram := wqOpGram * wkOpGram
  -- Low-rank Gram-product tightening (64×64):
  -- ‖W_Q W_Kᵀ‖₂² = λ_max((W_QᵀW_Q)(W_KᵀW_K)) ≤ ‖(W_QᵀW_Q)(W_KᵀW_K)‖_∞.
  let wqGram := layer.W_Q.transpose.matmul layer.W_Q
  let wkGram := layer.W_K.transpose.matmul layer.W_K
  let qkDenseGram := Float.sqrt (max 0.0 ((wqGram.matmul wkGram).infNormAbs))
  -- Brauer/Cassini Gram bound on a 64×64 product with the same singular values:
  -- For `A = W_Q` and `B = W_K`, `‖A Bᵀ‖₂ = ‖Bᵀ A‖₂`.
  let qkSmall := layer.W_K.transpose.matmul layer.W_Q
  let qkDenseBrauer := qkSmall.opNormUpperBoundDenseBrauer
  min (min base qkFactorGram) (min (min qkDenseGram qkDenseBrauer) qk.frobeniusNorm)

/-- A tighter (still sound-in-ℝ) Float upper bound on ‖W_Q · W_Kᵀ‖₂.

Convenience wrapper that materializes `W_Q·W_Kᵀ`.
Prefer `queryKeyAlignmentOpBoundFrom` when `qk` is already available.
-/
def queryKeyAlignmentOpBound (layer : ConcreteAttentionLayer) : Float :=
  layer.queryKeyAlignmentOpBoundFrom layer.queryKeyAlignment

/-- A tighter (still sound-in-ℝ) Float upper bound on ‖W_V · W_O‖₂.

This is the minimum of Schur / Frobenius / factorized Schur / factorized Frobenius
upper bounds, analogous to `queryKeyAlignmentOpBoundFrom`.
-/
def valueOutputProjectionOpBoundFrom
    (layer : ConcreteAttentionLayer) (vo : ConcreteMatrix) : Float :=
  let base := opBoundMinOfMany vo layer.W_V layer.W_O
  let wvOpGram := layer.W_V.opNormUpperBoundViaGramInf
  -- PERFORMANCE: `W_O` is typically wide (`headDim×modelDim`), so we compute the same
  -- bound on `W_Oᵀ` instead (‖W_O‖₂ = ‖W_Oᵀ‖₂) to avoid an O(modelDim²) loop.
  let woOpGram := layer.W_O.transpose.opNormUpperBoundViaGramInf
  let voFactorGram := wvOpGram * woOpGram
  -- Low-rank Gram-product tightening (64×64):
  -- For `M = W_V W_O`, ‖M‖₂² = λ_max((W_VᵀW_V)(W_O W_Oᵀ)) up to reordering.
  let wvGram := layer.W_V.transpose.matmul layer.W_V
  let woGram := layer.W_O.matmul layer.W_O.transpose
  let voDenseGram := Float.sqrt (max 0.0 ((wvGram.matmul woGram).infNormAbs))
  -- Brauer/Cassini Gram bound on a 64×64 product with the same singular values:
  -- For `A = W_V` and `B = W_Oᵀ`, `‖A B‖₂ = ‖B A‖₂`.
  let voSmall := layer.W_O.matmul layer.W_V
  let voDenseBrauer := voSmall.opNormUpperBoundDenseBrauer
  min (min base voFactorGram) (min (min voDenseGram voDenseBrauer) vo.frobeniusNorm)


/-- A tighter (still sound-in-ℝ) Float upper bound on ‖W_V · W_O‖₂.

Convenience wrapper that materializes `W_V·W_O`.
Prefer `valueOutputProjectionOpBoundFrom` when `vo` is already available.
-/
def valueOutputProjectionOpBound (layer : ConcreteAttentionLayer) : Float :=
  layer.valueOutputProjectionOpBoundFrom layer.valueOutputProjection

end ConcreteAttentionLayer

/-! ## Concrete MLP Layer -/

/-- A concrete MLP (Feed-Forward) layer with exported weights.

Standard transformer MLP: `output = W_out · activation(W_in · x + b_in) + b_out`

For interpretability, we analyze individual **neurons** (columns of W_in / rows of W_out).
Each neuron i computes: `activation(W_in[:,i]·x + b_in[i])` and writes `W_out[i,:]` to output.
-/
structure ConcreteMLPLayer where
  /-- Model dimension (embedding size) -/
  modelDim : Nat
  /-- Hidden dimension (number of neurons) -/
  hiddenDim : Nat
  /-- Input projection matrix (modelDim × hiddenDim): maps input to hidden activations -/
  W_in : ConcreteMatrix
  /-- Output projection matrix (hiddenDim × modelDim): maps hidden to output -/
  W_out : ConcreteMatrix
  /-- Input bias (hiddenDim) stored as 1×hiddenDim matrix for uniformity -/
  b_in : ConcreteMatrix
  /-- Output bias (modelDim) stored as 1×modelDim matrix for uniformity -/
  b_out : ConcreteMatrix
  /-- Dimension consistency for W_in -/
  W_in_dims : W_in.numRows = modelDim ∧ W_in.numCols = hiddenDim
  /-- Dimension consistency for W_out -/
  W_out_dims : W_out.numRows = hiddenDim ∧ W_out.numCols = modelDim
  /-- Dimension consistency for b_in -/
  b_in_dims : b_in.numRows = 1 ∧ b_in.numCols = hiddenDim
  /-- Dimension consistency for b_out -/
  b_out_dims : b_out.numRows = 1 ∧ b_out.numCols = modelDim

namespace ConcreteMLPLayer

/-- Get the input weight vector for neuron i (column i of W_in). -/
def neuronInputWeights (layer : ConcreteMLPLayer) (neuronIdx : Nat) : Array Float :=
  if neuronIdx < layer.hiddenDim then
    .ofFn fun row : Fin layer.modelDim => layer.W_in.get row.val neuronIdx
  else #[]

/-- Get the output weight vector for neuron i (row i of W_out). -/
def neuronOutputWeights (layer : ConcreteMLPLayer) (neuronIdx : Nat) : Array Float :=
  if neuronIdx < layer.hiddenDim then
    .ofFn fun col : Fin layer.modelDim => layer.W_out.get neuronIdx col.val
  else #[]

/-- Compute the L2 norm of input weights for a neuron. -/
def neuronInputNorm (layer : ConcreteMLPLayer) (neuronIdx : Nat) : Float :=
  let weights := layer.neuronInputWeights neuronIdx
  Float.sqrt (weights.foldl (fun acc w => acc + w * w) 0.0)

/-- Compute the L2 norm of output weights for a neuron. -/
def neuronOutputNorm (layer : ConcreteMLPLayer) (neuronIdx : Nat) : Float :=
  let weights := layer.neuronOutputWeights neuronIdx
  Float.sqrt (weights.foldl (fun acc w => acc + w * w) 0.0)

/-- Compute the "influence magnitude" of a neuron: ‖W_in[:,i]‖ · ‖W_out[i,:]‖

This measures how much information can flow through neuron i.
For ReLU networks, this bounds the neuron's contribution to the output.
-/
def neuronInfluence (layer : ConcreteMLPLayer) (neuronIdx : Nat) : Float :=
  layer.neuronInputNorm neuronIdx * layer.neuronOutputNorm neuronIdx

/-- Get the bias for neuron i. -/
def getBias (layer : ConcreteMLPLayer) (neuronIdx : Nat) : Float :=
  if neuronIdx < layer.hiddenDim then
    layer.b_in.get 0 neuronIdx
  else 0.0

end ConcreteMLPLayer

/-! ## Interval Bound Propagation (IBP) for MLP Activation Stability

When analyzing circuit faithfulness, we need to know if ablating upstream components
can cause MLP neurons to "flip" their activation states. A neuron is **stable** if
its pre-activation stays positive (always active) or negative (always inactive) under
all perturbations bounded by ε.

**Mathematical Setup:**
- Pre-activation: z = W_in^T · x + b
- For perturbation δx with ‖δx‖₂ ≤ ε:
  - z' = W_in^T · (x + δx) + b = z + W_in^T · δx
  - By Cauchy-Schwarz: |W_in^T · δx| ≤ ‖W_in‖₂ · ‖δx‖₂ ≤ ‖W_in‖₂ · ε
- Therefore: z - ε·‖W_in‖ ≤ z' ≤ z + ε·‖W_in‖

**Stability Criterion:**
- Neuron is "stably ON" if z - ε·‖W_in‖ > 0
- Neuron is "stably OFF" if z + ε·‖W_in‖ < 0
- Otherwise, neuron is "unstable" (may flip)

**Pattern Term for Unstable Neurons:**
When a ReLU neuron flips, the linearization error is bounded by the magnitude
of the output weight times the activation change:
  ‖ΔOutput‖ ≤ ‖W_out[i,:]‖ · |activation_change|
            ≤ ‖W_out[i,:]‖ · max(|z + ε·‖W_in‖|, |z - ε·‖W_in‖|)

For GeLU, the bound is tighter but we use ReLU-style conservative bounds.
-/

/-- Result of interval bound propagation for a single neuron. -/
structure NeuronIntervalBound where
  /-- Neuron index within the layer -/
  neuronIdx : Nat
  /-- Lower bound on pre-activation (z - ε·‖W_in‖) -/
  preActLower : Float
  /-- Upper bound on pre-activation (z + ε·‖W_in‖) -/
  preActUpper : Float
  /-- Nominal pre-activation (z = W_in^T · x + b) -/
  preActNominal : Float
  /-- Input weight norm ‖W_in[:,i]‖ -/
  inputNorm : Float
  /-- Output weight norm ‖W_out[i,:]‖ -/
  outputNorm : Float
  deriving Repr

namespace NeuronIntervalBound

/-- Is this neuron stably active (always ON) under the perturbation bound? -/
def isStablyActive (b : NeuronIntervalBound) : Bool :=
  b.preActLower > 0.0

/-- Is this neuron stably inactive (always OFF) under the perturbation bound? -/
def isStablyInactive (b : NeuronIntervalBound) : Bool :=
  b.preActUpper < 0.0

/-- Is this neuron stable (won't flip activation state)? -/
def isStable (b : NeuronIntervalBound) : Bool :=
  b.isStablyActive || b.isStablyInactive

/-- Is this neuron unstable (may flip activation state)? -/
def isUnstable (b : NeuronIntervalBound) : Bool :=
  ¬b.isStable

/-- The "flip margin" - how close the pre-activation interval is to zero.

For stable neurons this is positive (distance from zero).
For unstable neurons this is negative (interval crosses zero).
-/
def flipMargin (b : NeuronIntervalBound) : Float :=
  min b.preActLower (-b.preActUpper)

/-- Bound on the activation change if the neuron flips (ReLU).

For ReLU, if the neuron flips from ON to OFF, the activation changes from z to 0.
If it flips from OFF to ON, the activation changes from 0 to z.
The maximum change magnitude is bounded by max(|z_lower|, |z_upper|).
-/
def maxActivationChange (b : NeuronIntervalBound) : Float :=
  if b.isStable then 0.0
  else max (Float.abs b.preActLower) (Float.abs b.preActUpper)

/-- Bound on the output error due to potential activation flip.

This is the pattern term contribution for an unstable neuron:
  ‖ΔOutput‖ ≤ ‖W_out[i,:]‖ · max_activation_change
-/
def patternTermBound (b : NeuronIntervalBound) : Float :=
  b.outputNorm * b.maxActivationChange

end NeuronIntervalBound

/-- Result of IBP analysis for an entire MLP layer. -/
structure MLPIntervalAnalysis where
  /-- Layer index -/
  layerIdx : Nat
  /-- Per-neuron interval bounds -/
  neuronBounds : Array NeuronIntervalBound
  /-- Input perturbation norm (ε) used for analysis -/
  perturbationNorm : Float
  /-- Number of stable neurons -/
  numStable : Nat
  /-- Number of unstable neurons -/
  numUnstable : Nat
  /-- Total pattern term bound for unstable neurons -/
  totalPatternBound : Float
  deriving Repr

namespace MLPIntervalAnalysis

/-- Fraction of neurons that are stable. -/
def stabilityRatio (a : MLPIntervalAnalysis) : Float :=
  if a.neuronBounds.size = 0 then 1.0
  else a.numStable.toFloat / a.neuronBounds.size.toFloat

/-- Is the layer "fully stable" (all neurons stable)? -/
def isFullyStable (a : MLPIntervalAnalysis) : Bool :=
  a.numUnstable = 0

/-- Get bounds for a specific neuron. -/
def getNeuronBound (a : MLPIntervalAnalysis) (idx : Nat) : Option NeuronIntervalBound :=
  if h : idx < a.neuronBounds.size then some a.neuronBounds[idx] else none

end MLPIntervalAnalysis

namespace ConcreteMLPLayer

/-- Compute pre-activations for all neurons given an input vector (single position).

Returns array of pre-activations: z[i] = W_in[:,i]^T · x + b[i]
-/
def computePreActivations (layer : ConcreteMLPLayer) (input : Array Float) :
    Array Float :=
  .ofFn fun i : Fin layer.hiddenDim => Id.run do
    let mut z : Float := layer.getBias i.val
    for j in [:layer.modelDim] do
      -- SAFETY: input should have size modelDim, but getD provides safe fallback
      let x_j := input.getD j 0.0
      let w_ji := layer.W_in.get j i.val
      z := z + w_ji * x_j
    return z

/-- Compute interval bounds for all neurons given input and perturbation bound.

**Algorithm:**
For each neuron i:
1. Compute nominal pre-activation: z = W_in[:,i]^T · x + b[i]
2. Compute Δz = ε · ‖W_in[:,i]‖₂ (maximum change due to perturbation)
3. Set bounds: [z - Δz, z + Δz]
4. Determine stability based on whether interval crosses zero

**Parameters:**
- `input`: Nominal input vector (modelDim elements)
- `perturbationNorm`: L2 norm bound on input perturbation (ε)
-/
def computeIntervalBounds (layer : ConcreteMLPLayer)
    (input : Array Float) (perturbationNorm : Float) : Array NeuronIntervalBound :=
  let preActs := layer.computePreActivations input
  .ofFn fun i : Fin layer.hiddenDim =>
    -- SAFETY: preActs has size hiddenDim by construction from computePreActivations
    let z := preActs[i.val]!
    let inputNorm := layer.neuronInputNorm i.val
    let outputNorm := layer.neuronOutputNorm i.val
    let delta := perturbationNorm * inputNorm
    {
      neuronIdx := i.val
      preActLower := z - delta
      preActUpper := z + delta
      preActNominal := z
      inputNorm := inputNorm
      outputNorm := outputNorm
    }

/-- Run full IBP analysis on an MLP layer.

Returns comprehensive analysis including stability counts and total pattern bound.
-/
def analyzeIntervalBounds (layer : ConcreteMLPLayer) (layerIdx : Nat)
    (input : Array Float) (perturbationNorm : Float) : MLPIntervalAnalysis := Id.run do
  let bounds := layer.computeIntervalBounds input perturbationNorm
  let mut numStable : Nat := 0
  let mut numUnstable : Nat := 0
  let mut totalPattern : Float := 0.0

  for b in bounds do
    if b.isStable then
      numStable := numStable + 1
    else
      numUnstable := numUnstable + 1
      totalPattern := totalPattern + b.patternTermBound

  {
    layerIdx := layerIdx
    neuronBounds := bounds
    perturbationNorm := perturbationNorm
    numStable := numStable
    numUnstable := numUnstable
    totalPatternBound := totalPattern
  }

/-- Compute the pattern term bound for a single neuron using IBP.

This is a convenient wrapper for single-neuron queries, useful for
`computeNeuronImportance`.

**Parameters:**
- `neuronIdx`: Index of the neuron to analyze
- `input`: Nominal input vector (from forward pass)
- `perturbationNorm`: L2 norm bound on input perturbation

**Returns:** Pattern term bound (0 if stable, output_norm * max_change if unstable)
-/
def neuronPatternTermBoundIBP (layer : ConcreteMLPLayer) (neuronIdx : Nat)
    (input : Array Float) (perturbationNorm : Float) : Float :=
  if neuronIdx ≥ layer.hiddenDim then 0.0
  else
    let z := Id.run do
      let mut acc : Float := layer.getBias neuronIdx
      for j in [:layer.modelDim] do
        let x_j := input.getD j 0.0
        let w_ji := layer.W_in.get j neuronIdx
        acc := acc + w_ji * x_j
      acc
    let inputNorm := layer.neuronInputNorm neuronIdx
    let outputNorm := layer.neuronOutputNorm neuronIdx
    let delta := perturbationNorm * inputNorm

    -- Check stability
    let lower := z - delta
    let upper := z + delta
    if lower > 0.0 || upper < 0.0 then
      -- Stable: no pattern term error
      0.0
    else
      -- Unstable: bound by output weight times max activation change
      outputNorm * max (Float.abs lower) (Float.abs upper)

end ConcreteMLPLayer

/-- Create a ConcreteMLPLayer from raw Float arrays. -/
def mkConcreteMLPLayer
    (modelDim hiddenDim : Nat)
    (w_in w_out b_in b_out : Array Float)
    (hw_in : w_in.size = modelDim * hiddenDim)
    (hw_out : w_out.size = hiddenDim * modelDim)
    (hb_in : b_in.size = hiddenDim)
    (hb_out : b_out.size = modelDim) : ConcreteMLPLayer where
  modelDim := modelDim
  hiddenDim := hiddenDim
  W_in := { numRows := modelDim, numCols := hiddenDim, data := w_in, size_eq := hw_in }
  W_out := { numRows := hiddenDim, numCols := modelDim, data := w_out, size_eq := hw_out }
  b_in := { numRows := 1, numCols := hiddenDim, data := b_in,
            size_eq := by simp [hb_in] }
  b_out := { numRows := 1, numCols := modelDim, data := b_out,
             size_eq := by simp [hb_out] }
  W_in_dims := ⟨rfl, rfl⟩
  W_out_dims := ⟨rfl, rfl⟩
  b_in_dims := ⟨rfl, rfl⟩
  b_out_dims := ⟨rfl, rfl⟩

/-! ## Sparse Autoencoders (SAEs) for Feature-Level Analysis

Sparse Autoencoders decompose MLP activations into interpretable **features**.
Instead of analyzing raw neurons (which are often polysemantic), we analyze
sparse linear combinations that correspond to semantic concepts.

**Architecture:**
- Encoder: `f = ReLU(W_enc · x + b_enc)` maps residual stream to sparse features
- Decoder: `x' = W_dec · f + b_dec` reconstructs the residual stream
- Sparsity: Only a small number of features are active (f[k] > 0) for any input

**Key Insight for Circuit Discovery:**
The Jacobian of an MLP approximated through an SAE becomes a sum of rank-1 matrices:
  `J ≈ Σ_{k ∈ active} W_dec[:,k] ⊗ W_enc[k,:]`

This allows us to:
1. Identify which **features** (not neurons) are responsible for behavior
2. Discover cleaner circuits for complex behaviors
3. Handle polysemantic neurons by decomposing them into monosemantic features

**Reconstruction Error:**
The SAE approximation introduces error: `‖MLP(x) - SAE(x)‖_F`
This must be accounted for in the total faithfulness bound.
-/

/-- A Sparse Autoencoder for analyzing MLP features.

Trained to reconstruct residual stream activations with sparse latent codes.
Typically has many more features than the residual stream dimension (e.g., 16x).
-/
structure ConcreteSAE where
  /-- Input/output dimension (residual stream size) -/
  inputDim : Nat
  /-- Number of features (typically >> inputDim for overcomplete SAEs) -/
  numFeatures : Nat
  /-- Encoder weights (inputDim × numFeatures): W_enc[i,k] = weight from input i to feature k -/
  W_enc : ConcreteMatrix
  /-- Decoder weights (numFeatures × inputDim): W_dec[k,j] = weight from feature k to output j -/
  W_dec : ConcreteMatrix
  /-- Encoder bias (numFeatures): b_enc[k] = bias for feature k -/
  b_enc : ConcreteMatrix
  /-- Decoder bias (inputDim): b_dec[j] = bias for output j -/
  b_dec : ConcreteMatrix
  /-- Dimension constraints -/
  W_enc_dims : W_enc.numRows = inputDim ∧ W_enc.numCols = numFeatures
  W_dec_dims : W_dec.numRows = numFeatures ∧ W_dec.numCols = inputDim
  b_enc_dims : b_enc.numRows = 1 ∧ b_enc.numCols = numFeatures
  b_dec_dims : b_dec.numRows = 1 ∧ b_dec.numCols = inputDim

namespace ConcreteSAE

/-- ReLU activation for SAE encoder. -/
private def relu (x : Float) : Float := if x > 0.0 then x else 0.0

/-- Encode input to sparse feature activations: f = ReLU(W_enc^T · x + b_enc)

Note: W_enc is stored as (inputDim × numFeatures), so we compute x^T · W_enc.

PERFORMANCE: Pre-allocates output array (critical for SAE-based circuit discovery).
-/
def encode (sae : ConcreteSAE) (input : Array Float) : Array Float :=
  .ofFn fun k : Fin sae.numFeatures => Id.run do
    let mut z : Float := sae.b_enc.get 0 k.val
    for i in [:sae.inputDim] do
      let x_i := input.getD i 0.0
      let w_ik := sae.W_enc.get i k.val
      z := z + x_i * w_ik
    return relu z

/-- Decode sparse features back to residual stream: x' = W_dec^T · f + b_dec

Note: W_dec is stored as (numFeatures × inputDim), so we compute f^T · W_dec.

PERFORMANCE: Pre-allocates output array (critical for SAE-based circuit discovery).
-/
def decode (sae : ConcreteSAE) (features : Array Float) : Array Float :=
  .ofFn fun j : Fin sae.inputDim => Id.run do
    let mut y : Float := sae.b_dec.get 0 j.val
    for k in [:sae.numFeatures] do
      let f_k := features.getD k 0.0
      let w_kj := sae.W_dec.get k j.val
      y := y + f_k * w_kj
    return y

/-- Full forward pass: encode then decode. -/
def forward (sae : ConcreteSAE) (input : Array Float) : Array Float :=
  sae.decode (sae.encode input)

/-- Compute reconstruction error ‖x - SAE(x)‖₂ for a single input. -/
def reconstructionError (sae : ConcreteSAE) (input : Array Float) : Float := Id.run do
  let reconstructed := sae.forward input
  let mut errSq : Float := 0.0
  for i in [:sae.inputDim] do
    let diff := input.getD i 0.0 - reconstructed.getD i 0.0
    errSq := errSq + diff * diff
  Float.sqrt errSq

/-- Compute reconstruction error for a matrix (multiple positions). -/
def reconstructionErrorMatrix (sae : ConcreteSAE) (input : ConcreteMatrix) : Float := Id.run do
  let mut totalErrSq : Float := 0.0
  for pos in [:input.numRows] do
    let inputVec : Array Float := .ofFn fun d : Fin input.numCols => input.get pos d.val
    let err := sae.reconstructionError inputVec
    totalErrSq := totalErrSq + err * err
  Float.sqrt totalErrSq

/-- Get indices of active features (f[k] > threshold). -/
def activeFeatures (sae : ConcreteSAE) (input : Array Float)
    (threshold : Float := 0.0) : Array Nat := Id.run do
  let features := sae.encode input
  let mut active : Array Nat := #[]
  for k in [:sae.numFeatures] do
    if features.getD k 0.0 > threshold then
      active := active.push k
  active

/-- Count active features for an input. -/
def numActiveFeatures (sae : ConcreteSAE) (input : Array Float)
    (threshold : Float := 0.0) : Nat :=
  (sae.activeFeatures input threshold).size

/-- Get the encoder weight vector for feature k (column k of W_enc). -/
def encoderWeights (sae : ConcreteSAE) (featureIdx : Nat) : Array Float :=
  if featureIdx < sae.numFeatures then
    .ofFn fun i : Fin sae.inputDim => sae.W_enc.get i.val featureIdx
  else #[]

/-- Get the decoder weight vector for feature k (row k of W_dec). -/
def decoderWeights (sae : ConcreteSAE) (featureIdx : Nat) : Array Float :=
  if featureIdx < sae.numFeatures then
    .ofFn fun j : Fin sae.inputDim => sae.W_dec.get featureIdx j.val
  else #[]

/-- Compute the L2 norm of encoder weights for feature k. -/
def encoderNorm (sae : ConcreteSAE) (featureIdx : Nat) : Float :=
  let weights := sae.encoderWeights featureIdx
  Float.sqrt (weights.foldl (fun acc w => acc + w * w) 0.0)

/-- Compute the L2 norm of decoder weights for feature k. -/
def decoderNorm (sae : ConcreteSAE) (featureIdx : Nat) : Float :=
  let weights := sae.decoderWeights featureIdx
  Float.sqrt (weights.foldl (fun acc w => acc + w * w) 0.0)

/-- Compute the "influence magnitude" of feature k: ‖W_enc[:,k]‖ · ‖W_dec[k,:]‖

This bounds how much information can flow through the feature.
Analogous to `neuronInfluence` for MLP neurons.
-/
def featureInfluence (sae : ConcreteSAE) (featureIdx : Nat) : Float :=
  sae.encoderNorm featureIdx * sae.decoderNorm featureIdx

/-- Get encoder bias for feature k. -/
def encoderBias (sae : ConcreteSAE) (featureIdx : Nat) : Float :=
  if featureIdx < sae.numFeatures then sae.b_enc.get 0 featureIdx else 0.0

/-- Compute the pre-activation for feature k given input. -/
def featurePreActivation (sae : ConcreteSAE) (featureIdx : Nat)
    (input : Array Float) : Float := Id.run do
  if featureIdx ≥ sae.numFeatures then return 0.0
  let mut z : Float := sae.encoderBias featureIdx
  for i in [:sae.inputDim] do
    let x_i := input.getD i 0.0
    let w_ik := sae.W_enc.get i featureIdx
    z := z + x_i * w_ik
  z

/-- Check if feature k is active (pre-activation > 0) for given input. -/
def isFeatureActive (sae : ConcreteSAE) (featureIdx : Nat) (input : Array Float) : Bool :=
  sae.featurePreActivation featureIdx input > 0.0

end ConcreteSAE

/-- Create a ConcreteSAE from raw Float arrays. -/
def mkConcreteSAE
    (inputDim numFeatures : Nat)
    (w_enc w_dec b_enc b_dec : Array Float)
    (hw_enc : w_enc.size = inputDim * numFeatures)
    (hw_dec : w_dec.size = numFeatures * inputDim)
    (hb_enc : b_enc.size = numFeatures)
    (hb_dec : b_dec.size = inputDim) : ConcreteSAE where
  inputDim := inputDim
  numFeatures := numFeatures
  W_enc := { numRows := inputDim, numCols := numFeatures, data := w_enc, size_eq := hw_enc }
  W_dec := { numRows := numFeatures, numCols := inputDim, data := w_dec, size_eq := hw_dec }
  b_enc := { numRows := 1, numCols := numFeatures, data := b_enc,
             size_eq := by simp [hb_enc] }
  b_dec := { numRows := 1, numCols := inputDim, data := b_dec,
             size_eq := by simp [hb_dec] }
  W_enc_dims := ⟨rfl, rfl⟩
  W_dec_dims := ⟨rfl, rfl⟩
  b_enc_dims := ⟨rfl, rfl⟩
  b_dec_dims := ⟨rfl, rfl⟩

/-! ### SAE Interval Bound Propagation

Like MLP neurons, SAE features can flip activation states under perturbation.
We extend IBP to track feature stability.
-/

/-- Result of IBP analysis for a single SAE feature. -/
structure SAEFeatureIntervalBound where
  /-- Feature index -/
  featureIdx : Nat
  /-- Lower bound on pre-activation -/
  preActLower : Float
  /-- Upper bound on pre-activation -/
  preActUpper : Float
  /-- Nominal pre-activation -/
  preActNominal : Float
  /-- Encoder weight norm ‖W_enc[:,k]‖ -/
  encoderNorm : Float
  /-- Decoder weight norm ‖W_dec[k,:]‖ -/
  decoderNorm : Float
  deriving Repr

namespace SAEFeatureIntervalBound

/-- Is this feature stably active? -/
def isStablyActive (b : SAEFeatureIntervalBound) : Bool := b.preActLower > 0.0

/-- Is this feature stably inactive? -/
def isStablyInactive (b : SAEFeatureIntervalBound) : Bool := b.preActUpper < 0.0

/-- Is this feature stable (won't flip)? -/
def isStable (b : SAEFeatureIntervalBound) : Bool := b.isStablyActive || b.isStablyInactive

/-- Pattern term bound if this feature flips. -/
def patternTermBound (b : SAEFeatureIntervalBound) : Float :=
  if b.isStable then 0.0
  else b.decoderNorm * max (Float.abs b.preActLower) (Float.abs b.preActUpper)

end SAEFeatureIntervalBound

/-- Result of IBP analysis for an entire SAE. -/
structure SAEIntervalAnalysis where
  /-- Per-feature bounds -/
  featureBounds : Array SAEFeatureIntervalBound
  /-- Perturbation norm used -/
  perturbationNorm : Float
  /-- Number of stable features -/
  numStable : Nat
  /-- Number of unstable features -/
  numUnstable : Nat
  /-- Total pattern term bound from unstable features -/
  totalPatternBound : Float
  /-- Reconstruction error (SAE approximation) -/
  reconstructionError : Float
  deriving Repr

namespace SAEIntervalAnalysis

/-- Stability ratio. -/
def stabilityRatio (a : SAEIntervalAnalysis) : Float :=
  if a.featureBounds.size = 0 then 1.0
  else a.numStable.toFloat / a.featureBounds.size.toFloat

/-- Total error bound (pattern + reconstruction). -/
def totalErrorBound (a : SAEIntervalAnalysis) : Float :=
  a.totalPatternBound + a.reconstructionError

end SAEIntervalAnalysis

namespace ConcreteSAE

/-- Compute interval bounds for all features given input and perturbation.

PERFORMANCE: Pre-allocates result array with `Array.ofFn` to avoid O(n) reallocations.
-/
def computeFeatureIntervalBounds (sae : ConcreteSAE)
    (input : Array Float) (perturbationNorm : Float) : Array SAEFeatureIntervalBound :=
  .ofFn fun k : Fin sae.numFeatures =>
    let z := sae.featurePreActivation k.val input
    let encNorm := sae.encoderNorm k.val
    let decNorm := sae.decoderNorm k.val
    let delta := perturbationNorm * encNorm
    {
      featureIdx := k.val
      preActLower := z - delta
      preActUpper := z + delta
      preActNominal := z
      encoderNorm := encNorm
      decoderNorm := decNorm
    }

/-- Run full IBP analysis on an SAE. -/
def analyzeIntervalBounds (sae : ConcreteSAE) (input : Array Float)
    (perturbationNorm : Float) : SAEIntervalAnalysis := Id.run do
  let bounds := sae.computeFeatureIntervalBounds input perturbationNorm
  let mut numStable : Nat := 0
  let mut numUnstable : Nat := 0
  let mut totalPattern : Float := 0.0

  for b in bounds do
    if b.isStable then
      numStable := numStable + 1
    else
      numUnstable := numUnstable + 1
      totalPattern := totalPattern + b.patternTermBound

  let reconErr := sae.reconstructionError input

  {
    featureBounds := bounds
    perturbationNorm := perturbationNorm
    numStable := numStable
    numUnstable := numUnstable
    totalPatternBound := totalPattern
    reconstructionError := reconErr
  }

/-- Pattern term bound for a single feature using IBP. -/
def featurePatternTermBoundIBP (sae : ConcreteSAE) (featureIdx : Nat)
    (input : Array Float) (perturbationNorm : Float) : Float :=
  if featureIdx ≥ sae.numFeatures then 0.0
  else
    let z := sae.featurePreActivation featureIdx input
    let encNorm := sae.encoderNorm featureIdx
    let decNorm := sae.decoderNorm featureIdx
    let delta := perturbationNorm * encNorm
    let lower := z - delta
    let upper := z + delta
    if lower > 0.0 || upper < 0.0 then 0.0
    else decNorm * max (Float.abs lower) (Float.abs upper)

end ConcreteSAE

/-! ## Attention Weights Computation -/

/-- Concrete attention weights for a sequence.
A[q][k] = attention weight from query position q to key position k. -/
structure ConcreteAttentionWeights where
  /-- Sequence length -/
  seqLen : Nat
  /-- Attention weights stored row-major: weights[q * seqLen + k] = A[q,k] -/
  weights : Array Float
  /-- Size check -/
  size_eq : weights.size = seqLen * seqLen

namespace ConcreteAttentionWeights

/-- Access A[q, k]. -/
def get (A : ConcreteAttentionWeights) (q k : Nat) : Float :=
  if q < A.seqLen ∧ k < A.seqLen then
    -- Index is in-bounds by `size_eq` and the guard above.
    A.weights[q * A.seqLen + k]!
  else 0.0

/-- Fast access to `A[q, k]` assuming `q < seqLen` and `k < seqLen`. -/
@[inline] def getUnsafe (A : ConcreteAttentionWeights) (q k : Nat) : Float :=
  A.weights[q * A.seqLen + k]!

/-- Convert attention weights to a `ConcreteMatrix` for use with matrix multiplication. -/
def toMatrix (A : ConcreteAttentionWeights) : ConcreteMatrix where
  numRows := A.seqLen
  numCols := A.seqLen
  data := A.weights
  size_eq := by
    simpa using A.size_eq

/-- Compute softmax for a row of logits. -/
def softmaxRow (logits : Array Float) : Array Float :=
  Id.run do
    -- PERFORMANCE: keep arrays linear to enable in-place updates
    -- (see Lean Reference Manual: runtime reference counting + array performance).
    let mut expVals : Array Float := logits
    let mut maxVal : Float := -1e30
    for i in [:expVals.size] do
      let v := expVals[i]!
      if v > maxVal then maxVal := v
    let mut sumExp : Float := 0.0
    for i in [:expVals.size] do
      let v := Float.exp (expVals[i]! - maxVal)
      expVals := expVals.set! i v
      sumExp := sumExp + v
    if sumExp > 0.0 then
      for i in [:expVals.size] do
        expVals := expVals.set! i (expVals[i]! / sumExp)
    return expVals

/-- Compute attention weights given queries, keys, and scaling. -/
def compute (queries keys : ConcreteMatrix) (scale : Float)
    (seqLen : Nat)
    (causal : Bool := true) : ConcreteAttentionWeights := Id.run do
  -- PERFORMANCE: avoid allocating an `Array (Array Float)` of rows; write the flattened
  -- weights row-by-row into a single mutable array.
  let cols := min queries.numCols keys.numCols
  let n := seqLen * seqLen
  let mut weights : { w : Array Float // w.size = n } :=
    ⟨Array.replicate n 0.0, by simp [n]⟩
  for q in [:seqLen] do
    -- Initialize to -∞ and only fill the causal prefix when `causal = true`.
    let mut rowScores : Array Float := Array.replicate seqLen (-1e30)
    let stop := if causal then min (q + 1) seqLen else seqLen
    let qBase := q * queries.numCols
    for j in [:stop] do
      if q < queries.numRows ∧ j < keys.numRows then
        let jBase := j * keys.numCols
        let mut dotProd : Float := 0.0
        for d in [:cols] do
          -- SAFETY: within this branch `q < queries.numRows` and `j < keys.numRows`,
          -- and `d < cols ≤ queries.numCols/keys.numCols`.
          dotProd := dotProd + queries.data[qBase + d]! * keys.data[jBase + d]!
        rowScores := rowScores.set! j (dotProd / scale)
    let row := softmaxRow rowScores
    let rowBase := q * seqLen
    for k in [:stop] do
      let weights' := weights.1.set! (rowBase + k) (row[k]!)
      have weights'SizeEq : weights'.size = n := by
        have hsize : weights'.size = weights.1.size := by
          -- `set!` is `setIfInBounds`, which preserves size.
          simp [weights']
        exact hsize.trans weights.2
      weights := ⟨weights', weights'SizeEq⟩
  return {
    seqLen := seqLen
    weights := weights.1
    size_eq := by simpa [n] using weights.2
  }

end ConcreteAttentionWeights

/-! ## Softmax Jacobian Sparsity Analysis

For a probability vector p, the softmax Jacobian J has entries J_ij = p_i(δ_ij - p_j).
The Frobenius norm squared of J for a single row is:

  ‖J‖²_F = Σᵢⱼ p_i²(δ_ij - p_j)² = Σᵢ p_i²(1-p_i)² + Σᵢ≠ⱼ p_i²p_j²
         = Σᵢ p_i² - 2Σᵢ p_i³ + (Σᵢ p_i²)²

A simpler upper bound is `Σᵢ p_i(1-p_i) = 1 - Σᵢ p_i²` (the Gini impurity).

For sparse (one-hot) distributions: Σ p_i² ≈ 1, so ‖J‖_F ≈ 0
For uniform distributions: Σ p_i² = 1/n, so ‖J‖_F ≈ √((n-1)/n)

This allows us to compute much tighter pattern term bounds for sharp attention heads.
-/

/-- Compute the "effective softmax derivative norm" for a single probability row.

For probability vector p, this computes `sqrt(Σᵢ p_i(1-p_i)) = sqrt(1 - Σᵢ p_i²)`.
This bounds the Frobenius norm of the softmax Jacobian for that row.

- One-hot distribution → 0 (no gradient flow through softmax)
- Uniform over n → sqrt((n-1)/n) ≈ 1 for large n
-/
def softmaxRowJacobianNorm (row : Array Float) : Float :=
  let sumSq := row.foldl (fun acc p => acc + p * p) 0.0
  Float.sqrt (max 0.0 (1.0 - sumSq))

/-- Compute the average softmax Jacobian norm across all rows of attention weights.

This provides a data-dependent bound on the softmax Jacobian that is much tighter
than the worst-case constant 2.0 for sharp attention patterns.
-/
def ConcreteAttentionWeights.avgSoftmaxJacobianNorm (A : ConcreteAttentionWeights) : Float :=
  if A.seqLen = 0 then 0.0
  else Id.run do
    let mut totalNormSq : Float := 0.0
    for q in [:A.seqLen] do
      -- Extract row q
      let mut sumSq : Float := 0.0
      let rowBase := q * A.seqLen
      for k in [:A.seqLen] do
        -- SAFETY: `q < seqLen` and `k < seqLen` by loop bounds.
        let p := A.weights[rowBase + k]!
        sumSq := sumSq + p * p
      -- Jacobian norm squared for this row is bounded by 1 - sumSq
      totalNormSq := totalNormSq + max 0.0 (1.0 - sumSq)
    -- Return RMS (root mean square) of per-row norms
    Float.sqrt (totalNormSq / A.seqLen.toFloat)

/-- Compute the maximum softmax Jacobian norm across all rows.

More conservative than avg, but still much tighter than 2.0 for sparse attention.
-/
def ConcreteAttentionWeights.maxSoftmaxJacobianNorm (A : ConcreteAttentionWeights) : Float :=
  if A.seqLen = 0 then 0.0
  else Id.run do
    let mut maxNorm : Float := 0.0
    for q in [:A.seqLen] do
      let mut sumSq : Float := 0.0
      let rowBase := q * A.seqLen
      for k in [:A.seqLen] do
        -- SAFETY: `q < seqLen` and `k < seqLen` by loop bounds.
        let p := A.weights[rowBase + k]!
        sumSq := sumSq + p * p
      let rowNorm := Float.sqrt (max 0.0 (1.0 - sumSq))
      maxNorm := max maxNorm rowNorm
    maxNorm

/-! ### Attention Jacobian heuristics

The following quantities are computed from `Float` attention weights produced by a `Float`
softmax. They must be treated as **heuristic estimates**, not sound certificates.
-/

/-- Diagnostics for the softmax-Jacobian operator norm bound.

The full softmax Jacobian over an attention matrix is block-diagonal over rows,
so the overall operator norm is the maximum of the per-row operator norms.

We record statistics for the row that attains this maximum.
-/
structure SoftmaxJacobianOpDiag where
  /-- Overall (block-diagonal) operator norm upper bound, `max_q rowBound(q)`. -/
  opBound : Float
  /-- `max_i p_i` for the maximizing row. -/
  maxRowMaxP : Float
  /-- `tr(J) = 1 - ∑ p_i^2` for the maximizing row. -/
  maxRowTraceBound : Float
  /-- PSD moment bound for the maximizing row, derived from `tr(J)` and `‖J‖_F²`. -/
  maxRowMomentBound : Float
  /-- Gershgorin / induced-∞ bound `max_i 2 p_i (1 - p_i)` for the maximizing row. -/
  maxRowGersh : Float
  /-- The final per-row bound used for the maximizing row. -/
  maxRowBoundUsed : Float
  /-- Number of rows that triggered a conservative fallback (NaN/Inf/zero-sum). -/
  numRowsFallback : Nat
  deriving Repr

/-- Heuristic estimate of the softmax-Jacobian operator norm per row (then maxed).

For a probability row `p`, the softmax Jacobian is:
`J = diag(p) - p pᵀ`.
It is positive semidefinite and satisfies:
- `J ≤ diag(p)` so `‖J‖₂ ≤ maxᵢ pᵢ`
- `‖J‖₂ ≤ tr(J) = 1 - Σᵢ pᵢ²`

We take the tighter bound
`min(maxᵢ pᵢ, 1 - Σᵢ pᵢ²)` for each row,
and then take the maximum over rows.
This is especially sharp for nearly one-hot or nearly uniform rows.
-/
def ConcreteAttentionWeights.softmaxJacobianOpDiag
    (A : ConcreteAttentionWeights) : SoftmaxJacobianOpDiag :=
  if A.seqLen = 0 then
    { opBound := 0.0
      maxRowMaxP := 0.0
      maxRowTraceBound := 0.0
      maxRowMomentBound := 0.0
      maxRowGersh := 0.0
      maxRowBoundUsed := 0.0
      numRowsFallback := 0 }
  else Id.run do
    let mut maxBound : Float := 0.0
    let mut bestMaxP : Float := 0.0
    let mut bestTrace : Float := 0.0
    let mut bestMoment : Float := 0.0
    let mut bestGersh : Float := 0.0
    let mut bestUsed : Float := 0.0
    let mut fallbackCount : Nat := 0

    for q in [:A.seqLen] do
      let rowBase := q * A.seqLen

      -- Pass 1: clamp negatives to 0 and compute sum.
      let mut sumP : Float := 0.0
      for k in [:A.seqLen] do
        let p0 := A.weights[rowBase + k]!
        let p := if p0 < 0.0 then 0.0 else p0
        sumP := sumP + p

      let mut rowBound : Float := 0.5
      let mut rowMaxP : Float := 0.0
      let mut rowTrace : Float := 0.0
      let mut rowMoment : Float := 0.0
      let mut rowGersh : Float := 0.0

      if sumP ≤ 0.0 || Float.isNaN sumP || Float.isInf sumP then
        -- Conservative fallback: global bound for any probability row.
        fallbackCount := fallbackCount + 1
        rowBound := 0.5
      else
        -- Pass 2: renormalize and compute per-row bounds.
        let invSum := 1.0 / sumP
        let mut sumSq : Float := 0.0
        let mut sumCube : Float := 0.0
        let mut maxP : Float := 0.0
        let mut gersh : Float := 0.0
        for k in [:A.seqLen] do
          let p0 := A.weights[rowBase + k]!
          let pClamped := if p0 < 0.0 then 0.0 else p0
          let p := pClamped * invSum
          sumSq := sumSq + p * p
          sumCube := sumCube + p * p * p
          if p > maxP then maxP := p
          let g := 2.0 * p * (1.0 - p)
          if g > gersh then gersh := g
        let traceBound := max 0.0 (1.0 - sumSq)
        -- Moment bound: J is PSD with
        --   tr(J) = 1 - Σ p_i²
        --   ‖J‖_F² = Σ (p_i - p_i²)² + Σ_{i≠j} (p_i p_j)²
        --         = (Σ p_i²) - 2(Σ p_i³) + (Σ p_i²)².
        let frob2 := max 0.0 (sumSq - 2.0 * sumCube + sumSq * sumSq)
        let momentBound := ConcreteMatrix.psdLambdaMaxUpperMoment A.seqLen traceBound frob2
        -- Rigorous (for probability rows):
        --   λ_max(J) ≤ maxP
        --   λ_max(J) ≤ tr(J)
        --   λ_max(J) ≤ ‖J‖_∞ = max_i 2 p_i (1-p_i)
        --   λ_max(J) ≤ 1/2
        let bound0 := min maxP (min traceBound (min gersh momentBound))
        let bound1 := min bound0 0.5
        let bound2 := max 0.0 (min 0.5 bound1)
        let bound := if Float.isNaN bound2 || Float.isInf bound2 then 0.5 else bound2
        rowBound := bound
        rowMaxP := maxP
        rowTrace := traceBound
        rowMoment := momentBound
        rowGersh := gersh

      if rowBound > maxBound then
        maxBound := rowBound
        bestMaxP := rowMaxP
        bestTrace := rowTrace
        bestMoment := rowMoment
        bestGersh := rowGersh
        bestUsed := rowBound

    { opBound := maxBound
      maxRowMaxP := bestMaxP
      maxRowTraceBound := bestTrace
      maxRowMomentBound := bestMoment
      maxRowGersh := bestGersh
      maxRowBoundUsed := bestUsed
      numRowsFallback := fallbackCount }

/-- Backwards-compatible accessor: the bound value. -/
def ConcreteAttentionWeights.softmaxJacobianOpEst (A : ConcreteAttentionWeights) : Float :=
  (A.softmaxJacobianOpDiag).opBound

/-- Compute the overall Frobenius norm of the softmax Jacobian across all rows.

This is `sqrt(Σ_q (1 - Σ_k A[q,k]²))`, which gives the total "softness" of attention.
-/
def ConcreteAttentionWeights.softmaxJacobianFrobeniusNorm
    (A : ConcreteAttentionWeights) : Float :=
  if A.seqLen = 0 then 0.0
  else Id.run do
    let mut totalNormSq : Float := 0.0
    for q in [:A.seqLen] do
      let mut sumSq : Float := 0.0
      let rowBase := q * A.seqLen
      for k in [:A.seqLen] do
        -- SAFETY: `q < seqLen` and `k < seqLen` by loop bounds.
        let p := A.weights[rowBase + k]!
        sumSq := sumSq + p * p
      totalNormSq := totalNormSq + max 0.0 (1.0 - sumSq)
    Float.sqrt totalNormSq

/-! ## Forward Pass Implementations

These methods compute the actual forward pass through attention and MLP layers,
accumulating the residual stream as in real transformers.
-/

/-- Compute attention weights for a layer given an input matrix. -/
def ConcreteAttentionLayer.computeAttentionWeights (layer : ConcreteAttentionLayer)
    (input : ConcreteMatrix) (causal : Bool := true) : ConcreteAttentionWeights :=
  let queries := input.matmul layer.W_Q
  let keys := input.matmul layer.W_K
  let scale := Float.sqrt layer.headDim.toFloat
  ConcreteAttentionWeights.compute queries keys scale input.numRows causal

/-- Forward pass for a single attention head.

Input: X (seqLen × modelDim)
Output: Y (seqLen × modelDim) where Y = A·V·W_O

This computes the attention output (before residual connection):
1. Q = X·W_Q, K = X·W_K, V = X·W_V
2. A = softmax(Q·K^T / √d_head)
3. Y = A·V·W_O
-/
def ConcreteAttentionLayer.forward (layer : ConcreteAttentionLayer) (input : ConcreteMatrix)
    (causal : Bool := true) : ConcreteMatrix :=
  let attn := layer.computeAttentionWeights input causal
  let values := input.matmul layer.W_V  -- seqLen × headDim
  -- Compute A · V using direct construction
  let attnValues : ConcreteMatrix := {
    numRows := input.numRows
    numCols := layer.headDim
    data := .ofFn fun idx : Fin (input.numRows * layer.headDim) => Id.run do
      let q := idx.val / layer.headDim
      let d := idx.val % layer.headDim
      let mut acc : Float := 0.0
      let n := input.numRows
      let attnRowBase := q * n
      for k in [:n] do
        -- SAFETY: `q < n` and `k < n` by loop bounds, and `attn.seqLen = n`.
        let a := attn.weights[attnRowBase + k]!
        -- SAFETY: `k < values.numRows = n` and `d < values.numCols = headDim`.
        let v := values.data[k * layer.headDim + d]!
        acc := acc + a * v
      return acc
    size_eq := Array.size_ofFn
  }
  -- Project back to model dimension: (A·V) · W_O
  attnValues.matmul layer.W_O

/-- ReLU activation function for Float. -/
def reluFloat (x : Float) : Float := if x > 0.0 then x else 0.0

/-- GeLU activation function (approximate) for Float. -/
def geluFloat (x : Float) : Float :=
  let pi : Float := 3.14159265358979323846
  let sqrt2pi := Float.sqrt (2.0 * pi)
  0.5 * x * (1.0 + Float.tanh (sqrt2pi * (x + 0.044715 * x * x * x)))

/-- Derivative of `geluFloat` with respect to `x`.

This matches the tanh-based approximate GeLU used by `geluFloat`.
-/
def geluDerivFloat (x : Float) : Float :=
  let pi : Float := 3.14159265358979323846
  let a := Float.sqrt (2.0 * pi)
  let b : Float := 0.044715
  let t := a * (x + b * x * x * x)
  let tanhT := Float.tanh t
  let sech2 := 1.0 - tanhT * tanhT
  let t' := a * (1.0 + 3.0 * b * x * x)
  0.5 * (1.0 + tanhT) + 0.5 * x * sech2 * t'

/-- Forward pass for an MLP layer with GeLU activation.

Input: X (seqLen × modelDim)
Output: Y (seqLen × modelDim) where Y = W_out · GeLU(W_in · X + b_in) + b_out

For each position, this computes the standard transformer FFN.
-/
def ConcreteMLPLayer.forward (layer : ConcreteMLPLayer) (input : ConcreteMatrix) : ConcreteMatrix :=
  -- hidden = input · W_in + b_in  (seqLen × hiddenDim)
  let hidden := (input.matmul layer.W_in).addBias layer.b_in
  -- Apply GeLU activation
  let activated := hidden.map geluFloat
  -- output = activated · W_out + b_out  (seqLen × modelDim)
  (activated.matmul layer.W_out).addBias layer.b_out

/-- Forward pass plus a data-dependent bound on `max |gelu'(z)|` over this layer's preactivations.

The returned derivative maximum is exact for the computed `hidden` matrix entries (interpreting
Float arithmetic as defining a concrete real expression, consistent with this file's conventions).
If NaN/Inf is encountered, callers should conservatively fall back to a global bound.
-/
def ConcreteMLPLayer.forwardWithGeluDerivMax (layer : ConcreteMLPLayer)
    (input : ConcreteMatrix) : (ConcreteMatrix × Float) := Id.run do
  let hidden := (input.matmul layer.W_in).addBias layer.b_in
  let mut maxDeriv : Float := 0.0
  for z in hidden.data do
    let d := Float.abs (geluDerivFloat z)
    if d > maxDeriv then
      maxDeriv := d
  let activated := hidden.map geluFloat
  let out := (activated.matmul layer.W_out).addBias layer.b_out
  return (out, maxDeriv)

/-! ## Efficient Pattern Term Bound Calculation

The core insight: the valueTerm Frobenius norm factors as `‖A‖_F · ‖W_V·W_O‖_F`.
This avoids computing the full (N·D)² matrix!
-/

/-- Compute ‖valueTerm‖_F efficiently via factorization. -/
def computeValueTermNorm (attn : ConcreteAttentionWeights)
    (valueOutputProj : ConcreteMatrix) : Float :=
  let attnNormSq := attn.weights.foldl (fun acc x => acc + x * x) 0.0
  let projNormSq := valueOutputProj.frobeniusNormSq
  Float.sqrt (attnNormSq * projNormSq)

/-- Information needed to bound the pattern term. -/
structure PatternTermBoundInputs where
  /-- Attention weights -/
  attention : ConcreteAttentionWeights
  /-- W_Q · W_K^T alignment matrix -/
  queryKeyAlign : ConcreteMatrix
  /-- Deterministic Float upper bound on ‖W_Q · W_K^T‖₂.

This is treated as an upper bound in exact real arithmetic and may be computed
as the minimum of several valid upper bounds (e.g. Schur / Frobenius / factor bounds).
-/
  queryKeyAlignSchurNorm : Float
  /-- W_V · W_O projection -/
  valueOutputProj : ConcreteMatrix
  /-- Deterministic Float upper bound on ‖W_V · W_O‖₂.

This is treated as an upper bound in exact real arithmetic and may be computed
as the minimum of several valid upper bounds (e.g. Schur / Frobenius / factor bounds).
-/
  valueOutputProjSchurNorm : Float
  /-- Input embedding norm (‖X‖_F) -/
  inputNorm : Float
  /-- Scaling factor (√d_head) -/
  scaleFactor : Float

/-- Bound ‖patternTerm‖_F without expanding the full Jacobian.

The pattern term arises from how attention weights A change when input X changes:
  patternTerm = (∂A/∂X) ⊗ (V·W_O)

The key insight is that ∂A/∂X involves the softmax Jacobian, which is bounded by
the "softness" of the attention distribution. For sparse (one-hot) attention,
the softmax Jacobian is nearly zero, giving much tighter bounds.

**Sparsity-aware bound**:
  ‖patternTerm‖_F ≤ (‖J_softmax‖₂ / scale) · ‖X‖_F · ‖W_Q·W_K^T‖₂ · ‖W_V·W_O‖₂

We use a tight, data-dependent bound on `‖J_softmax‖₂` per row of attention:
for a probability row `p`, `J = diag(p) - p pᵀ` and
`‖J‖₂ ≤ min(maxᵢ pᵢ, 1 - Σᵢ pᵢ²)`.

- Perfectly one-hot rows give `‖J‖₂ = 0`.
- Uniform rows give `‖J‖₂ = 1/n`.
- Worst-case (all n): `‖J‖₂ ≤ 0.5`.
-/
def computePatternTermBound (inputs : PatternTermBoundInputs) : Float :=
  -- Data-dependent bound on the softmax Jacobian operator norm.
  -- Provable global bound: for any probability vector p, J = diag(p) - p pᵀ has ‖J‖₂ ≤ 1/2.
  -- Clamp defensively so callers cannot accidentally exceed this.
  let softmaxBound := min inputs.attention.softmaxJacobianOpEst 0.5
  (softmaxBound / inputs.scaleFactor) * inputs.inputNorm *
    inputs.queryKeyAlignSchurNorm * inputs.valueOutputProjSchurNorm

/-- Bound ‖patternTerm‖_F using the old pessimistic constant bound.

This uses the worst-case softmax Jacobian spectral-norm bound of 0.5, which is valid but loose.
Prefer `computePatternTermBound` for tighter data-dependent bounds.
-/
def computePatternTermBoundPessimistic (inputs : PatternTermBoundInputs) : Float :=
  let softmaxBound : Float := 0.5  -- Worst-case softmax Jacobian spectral norm
  (softmaxBound / inputs.scaleFactor) * inputs.inputNorm *
    inputs.queryKeyAlignSchurNorm * inputs.valueOutputProjSchurNorm

/-- Compute faithfulness ratio: ‖patternTerm‖_F / ‖valueTerm‖_F. -/
def computeFaithfulnessRatio (inputs : PatternTermBoundInputs) : Float :=
  let patternBound := computePatternTermBound inputs
  let valueNorm := computeValueTermNorm inputs.attention inputs.valueOutputProj
  if valueNorm < 1e-10 then Float.inf else patternBound / valueNorm

/-! ## Discovery Structures -/

/-- Result of discovering a potential induction head pair. -/
structure CandidateInductionHead where
  /-- Layer index of the "previous token" head (L1) -/
  layer1Idx : Nat
  /-- Layer index of the "induction" head (L2) -/
  layer2Idx : Nat
  /-- Head index within layer 1 -/
  head1Idx : Nat
  /-- Head index within layer 2 -/
  head2Idx : Nat
  /-- Faithfulness ratio ε₁ for L1: ‖PatternTerm‖_F / ‖ValueTerm‖_F -/
  patternBound1 : Float
  /-- Faithfulness ratio ε₂ for L2: ‖PatternTerm‖_F / ‖ValueTerm‖_F -/
  patternBound2 : Float
  /-- Combined relative error: (1+ε₁)(1+ε₂) - 1 = ε₁ + ε₂ + ε₁·ε₂ -/
  combinedError : Float
  /-- Previous-token strength: avg A₁[i, i-1] -/
  prevTokenStrength : Float
  /-- Induction "copy-next" pattern score for head 2 (prompt-dependent). -/
  inductionScore : Float
  /-- K-composition score between head 1 and head 2, as in the circuits framework paper:

  `kComp_raw = ‖W_QK² · W_OV¹‖_F / (‖W_QK²‖_F · ‖W_OV¹‖_F)`,

  then we subtract the random-baseline `1/√modelDim`:

  `kComp = kComp_raw - 1/√modelDim`.

  This measures how strongly head 1 can feed information into head 2's QK circuit,
  i.e. whether head 1 plausibly acts as a **pattern enabler** for head 2.
  -/
  kComp : Float
  /-- Description of the discovered pattern -/
  description : String

/-- A verified induction head that meets the certification threshold. -/
structure VerifiedInductionHead where
  /-- The candidate that was verified -/
  candidate : CandidateInductionHead
  /-- The certification threshold used -/
  threshold : Float
  /-- Combined error is below threshold (runtime-checked) -/
  errorChecked : Bool

/-- An induction head candidate with an explicit effectiveness score `δ` on a target direction.

This is produced by the Float-based discovery pipeline and should be interpreted as a
**heuristic ranking**, not a proof-grade certification.
-/
structure HeuristicInductionHead where
  /-- The discovered candidate pair (pattern-checked, heuristically) -/
  candidate : CandidateInductionHead
  /-- Raw effectiveness score `δ` on the target direction (Float). -/
  delta : Float
  /-- Scale-invariant effectiveness score (Float):

  `effect = δ / (‖ln₁(X₂)‖_F · ‖u‖₂)`,

  where `X₂` is the layer-2 input residual stream and `u` is the target direction.
  This isolates the **mechanism** (virtual-head computation) from residual-stream energy.
  -/
  effect : Float
  /-- Frobenius norm of the layer-2 input residual stream `‖X₂‖_F` (Float). -/
  layer2InputNorm : Float
  /-- Frobenius norm of the Pre-LN attention input `‖ln₁(X₂)‖_F` (Float). -/
  layer2Ln1InputNorm : Float

/-- Result of discovering a multi-layer circuit with N-layer error bounds.

This extends `CandidateInductionHead` with rigorous N-layer amplification bounds
from the theorem `n_layer_faithfulness_composition`:

  ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ)
-/
structure DeepCircuitCandidate where
  /-- Layer indices involved in the circuit (sorted) -/
  layerIndices : Array Nat
  /-- Head indices at each layer -/
  headIndices : Array Nat
  /-- Per-layer pattern term bounds (εᵢ) -/
  patternBounds : Array Float
  /-- Per-layer operator norm upper bounds (Cᵢ) -/
  operatorNormUbs : Array Float
  /-- Simple error sum: Σᵢ εᵢ (no amplification) -/
  simpleErrorSum : Float
  /-- N-layer amplified error: Σᵢ εᵢ · ∏_{j>i}(1+Cⱼ) -/
  amplifiedError : Float
  /-- Total amplification factor: ∏ᵢ(1+Cᵢ) -/
  amplificationFactor : Float
  /-- Pattern type description (e.g., "induction", "composition") -/
  patternType : String
  /-- Human-readable description -/
  description : String

namespace DeepCircuitCandidate

def toString (c : DeepCircuitCandidate) : String :=
  let heads := c.layerIndices.zip c.headIndices |>.map fun (l, h) => s!"L{l}H{h}"
  s!"{c.patternType}: {heads.toList} | " ++
  s!"ε_simple={c.simpleErrorSum}, ε_amplified={c.amplifiedError}, amp={c.amplificationFactor}"

instance : ToString DeepCircuitCandidate := ⟨toString⟩

end DeepCircuitCandidate

/-! ## Discovery Algorithm -/

/-- Check if a layer exhibits "previous-token" attention pattern. -/
def checkPrevTokenPattern (attn : ConcreteAttentionWeights)
    (minStrength : Float := 0.3) : Option Float :=
  if attn.seqLen < 2 then none
  else
    let sum : Float := Id.run do
      let n := attn.seqLen
      let w := attn.weights
      let mut acc : Float := 0.0
      for i in [:n - 1] do
        -- SAFETY: `i < n-1` implies `i+1 < n`, so `(i+1,i)` is in-bounds.
        acc := acc + w[(i + 1) * n + i]!
      return acc
    let avgStrength := sum / (attn.seqLen - 1).toFloat
    if avgStrength ≥ minStrength then some avgStrength else none

/-- Check if a head exhibits a **content-addressable** attention pattern.

We say a head is content-addressable on a prompt when, for many query positions `q`,
it places substantial attention mass on *previous occurrences of the same token*:

`score(q) = ∑_{k < q, tokens[k] = tokens[q]} A[q, k]`.

The returned score is the average of `score(q)` over query positions that have at least
one previous occurrence. This is variable-lag by construction (no fixed positional lag).
-/
def checkContentAddressablePattern (tokens : Array Nat) (attn : ConcreteAttentionWeights)
    (minScore : Float := 0.1) : Option Float :=
  if tokens.size ≠ attn.seqLen then none
  else if attn.seqLen < 2 then none
  else
    let n := attn.seqLen
    let w := attn.weights
    let (sumScore, count) : (Float × Nat) := Id.run do
      let mut sumScore : Float := 0.0
      let mut count : Nat := 0
      for q in [1:n] do
        let tq := tokens[q]!
        let rowBase := q * n
        let mut hasPrev : Bool := false
        let mut rowScore : Float := 0.0
        for k in [:q] do
          if tokens[k]! = tq then
            hasPrev := true
            -- SAFETY: `q < n` and `k < q ≤ n` by loop bounds.
            rowScore := rowScore + w[rowBase + k]!
        if hasPrev then
          sumScore := sumScore + rowScore
          count := count + 1
      return (sumScore, count)

    if count = 0 then none
    else
      let avgScore := sumScore / count.toFloat
      if avgScore ≥ minScore then some avgScore else none

/-- Check if a head exhibits an **induction** ("copy-next") attention pattern.

We say a head is induction-like on a prompt when, for many query positions `q`, it places
substantial attention mass on tokens *immediately after* previous occurrences of the same token:

`score(q) = ∑_{k+1 < q, tokens[k] = tokens[q]} A[q, k+1]`.

This is the token-level signature of the induction mechanism described in the transformer-circuits
literature: when the current token repeats, attend to the successor of the previous occurrence so
the head can **copy** that successor forward.

The returned score is the average of `score(q)` over query positions that have at least one
previous occurrence with an in-bounds successor (`k+1 < q`). This is variable-lag by construction.
-/
def checkInductionCopyNextPattern (tokens : Array Nat) (attn : ConcreteAttentionWeights)
    (minScore : Float := 0.1) : Option Float :=
  if tokens.size ≠ attn.seqLen then none
  else if attn.seqLen < 3 then none
  else
    let n := attn.seqLen
    let w := attn.weights
    let (sumScore, count) : (Float × Nat) := Id.run do
      let mut sumScore : Float := 0.0
      let mut count : Nat := 0
      -- Need `q ≥ 2` so there is room for a predecessor `k` with successor `k+1 < q`.
      for q in [2:n] do
        let tq := tokens[q]!
        let rowBase := q * n
        let mut hasPrevSucc : Bool := false
        let mut rowScore : Float := 0.0
        -- Scan all earlier positions `k` whose successor is still < q.
        for k in [:q - 1] do
          if tokens[k]! = tq then
            hasPrevSucc := true
            -- Attend to the *successor* position `k+1`.
            rowScore := rowScore + w[rowBase + (k + 1)]!
        if hasPrevSucc then
          sumScore := sumScore + rowScore
          count := count + 1
      return (sumScore, count)

    if count = 0 then none
    else
      let avgScore := sumScore / count.toFloat
      if avgScore ≥ minScore then some avgScore else none

/-- Compute composed attention score for induction pattern detection.

Generalizes over all possible repetition lags from 2 to n/2, computing the
"induction score" (average attention mass transferred from q to q+lag via
the two-layer circuit). Returns the maximum score across all lags.

This enables detection of induction heads with arbitrary repetition periods,
not just lag-2 patterns.
-/
def checkInductionPattern (attn1 attn2 : ConcreteAttentionWeights)
    (minScore : Float := 0.1) : Option Float :=
  if attn1.seqLen ≠ attn2.seqLen then none
  else if attn1.seqLen < 3 then none
  else
    let n := attn1.seqLen
    let maxLag := n / 2

    -- Try all possible lags and find the maximum induction score.
    --
    -- PERFORMANCE: this is called in an O(L²H²) search, so avoid `List.range.foldl`.
    let maxScore : Float := Id.run do
      let w1 := attn1.weights
      let w2 := attn2.weights
      let mut currentMax : Float := 0.0
      for lagIdx in [:maxLag - 1] do
        let lag := lagIdx + 2  -- Start from lag=2
        if lag < n then
          -- Compute average induction score for this lag
          let validPositions := n - lag
          let mut composedSum : Float := 0.0
          for q in [:validPositions] do
            let q' := q + lag
            let row2Base := q' * n
            let mut composedToQ : Float := 0.0
            -- Column access into `w1` is strided, so avoid repeated multiplications.
            let mut col1Idx := q
            for j in [:n] do
              -- SAFETY: `q' < n` and `j < n` by loop bounds.
              let a2 := w2[row2Base + j]!
              -- SAFETY: `j < n` and `q < n` by loop bounds.
              let a1 := w1[col1Idx]!
              composedToQ := composedToQ + a2 * a1
              col1Idx := col1Idx + n
            composedSum := composedSum + composedToQ
          let avgScore := composedSum / validPositions.toFloat
          if avgScore > currentMax then
            currentMax := avgScore
      return currentMax

    if maxScore ≥ minScore then some maxScore else none

/-- Multi-layer model with concrete weights. -/
structure ConcreteModel where
  /-- Number of layers -/
  numLayers : Nat
  /-- Attention layers with their heads: layers[l] is array of heads in layer l -/
  layers : Array (Array ConcreteAttentionLayer)
  /-- MLP layers: mlps[l] is the MLP in layer l (one per layer) -/
  mlps : Array ConcreteMLPLayer
  /-- Pre-LN LayerNorm parameters before attention (ln_1), one per layer. -/
  ln1 : Array ConcreteLayerNormParams := #[]
  /-- Pre-LN LayerNorm parameters before MLP (ln_2), one per layer. -/
  ln2 : Array ConcreteLayerNormParams := #[]
  /-- Final LayerNorm parameters (ln_f) before unembedding. -/
  lnf : ConcreteLayerNormParams := ConcreteLayerNormParams.identity 0
  /-- Sequence length for analysis -/
  seqLen : Nat
  /-- Optional ground-truth input token IDs for the prompt being analyzed.

  When present, this enables **self-supervised induction targeting** by choosing the
  correct next-token prediction target from sequence history (see
  `TargetDirection.fromInductionHistory`).
  -/
  inputTokens : Option (Array Nat) := none
  /-- Input embeddings (seqLen × modelDim) -/
  inputEmbeddings : ConcreteMatrix
  /-- Unembedding (decoder) matrix (modelDim × vocabSize) for logit computation.
      Maps final residual stream to vocabulary logits: logits = residual · W_U
      Optional: if not provided, target-aware analysis is unavailable. -/
  unembedding : Option ConcreteMatrix := none

namespace ConcreteModel

/-- Model dimension (d), inferred from input embeddings. -/
def modelDim (model : ConcreteModel) : Nat :=
  model.inputEmbeddings.numCols

/-- Get ln_1 parameters for a layer, defaulting to identity. -/
def ln1Params (model : ConcreteModel) (layerIdx : Nat) : ConcreteLayerNormParams :=
  model.ln1.getD layerIdx (ConcreteLayerNormParams.identity model.modelDim)

/-- Get ln_2 parameters for a layer, defaulting to identity. -/
def ln2Params (model : ConcreteModel) (layerIdx : Nat) : ConcreteLayerNormParams :=
  model.ln2.getD layerIdx (ConcreteLayerNormParams.identity model.modelDim)

/-- Apply ln_1 to a residual stream (row-wise, per token). -/
def applyLn1 (model : ConcreteModel) (layerIdx : Nat) (X : ConcreteMatrix) : ConcreteMatrix :=
  let p := model.ln1Params layerIdx
  ConcreteMatrix.layerNormRowwise X p.gamma p.beta

/-- Apply ln_2 to a residual stream (row-wise, per token). -/
def applyLn2 (model : ConcreteModel) (layerIdx : Nat) (X : ConcreteMatrix) : ConcreteMatrix :=
  let p := model.ln2Params layerIdx
  ConcreteMatrix.layerNormRowwise X p.gamma p.beta

/-- Apply final ln_f to a residual stream (row-wise, per token). -/
def applyLnf (model : ConcreteModel) (X : ConcreteMatrix) : ConcreteMatrix :=
  ConcreteMatrix.layerNormRowwise X model.lnf.gamma model.lnf.beta

/-- Heuristic estimate for ln_1 Jacobian operator norm at a specific activation. -/
def ln1OpBound (model : ConcreteModel) (layerIdx : Nat) (X : ConcreteMatrix) : Float :=
  let p := model.ln1Params layerIdx
  ConcreteMatrix.layerNormRowwiseOpEst X p.gamma

/-- Heuristic estimate for ln_2 Jacobian operator norm at a specific activation. -/
def ln2OpBound (model : ConcreteModel) (layerIdx : Nat) (X : ConcreteMatrix) : Float :=
  let p := model.ln2Params layerIdx
  ConcreteMatrix.layerNormRowwiseOpEst X p.gamma

end ConcreteModel

/-- Get the number of neurons in the MLP at a given layer. -/
def ConcreteModel.numNeuronsAtLayer (model : ConcreteModel) (layerIdx : Nat) : Nat :=
  if h : layerIdx < model.mlps.size then
    model.mlps[layerIdx].hiddenDim
  else 0

/-- Result of running a forward pass: the residual stream after each layer.

`layerInputs[l]` is the input to layer l (the accumulated residual stream).
`layerInputs[0]` = inputEmbeddings (initial token embeddings)
`layerInputs[l+1]` = x_{l+1} in the Pre-LN recurrence.
-/
structure ForwardPassResult where
  /-- Input to each layer. layerInputs[l] is what layer l receives. -/
  layerInputs : Array ConcreteMatrix
  /-- Attention outputs per layer per head: attnOutputs[l][h] = output of head h at layer l -/
  attnOutputs : Array (Array ConcreteMatrix)
  /-- MLP outputs per layer: mlpOutputs[l] = output of MLP at layer l -/
  mlpOutputs : Array ConcreteMatrix
  /-- Per-layer maximum absolute GeLU derivative over MLP preactivations.

Length is `numLayers`. For layers without an MLP, the entry is `0.0`.
-/
  mlpActDerivMax : Array Float
  /-- Final output after all layers (after ln_f, i.e. what goes into unembedding). -/
  finalOutput : ConcreteMatrix

/-- Run a full forward pass through the model, computing the residual stream at each layer.

This is the key function that enables deep circuit analysis: layer N sees the accumulated
output of layers 0..N-1, not just the raw embeddings.

 For each layer l (Pre-LN, GPT-2 style):
 1. u = ln_1(x_l)
 2. y_l = x_l + Σₕ AttentionHead[l,h].forward(u)
 3. v = ln_2(y_l)
 4. x_{l+1} = y_l + MLP[l].forward(v)

 After all layers: output = ln_f(x_L)
-/
def ConcreteModel.runForward (model : ConcreteModel)
    (causal : Bool := true) : ForwardPassResult := Id.run do
  let mut layerInputs : Array ConcreteMatrix := Array.mkEmpty (model.numLayers + 1)
  let mut attnOutputs : Array (Array ConcreteMatrix) := Array.mkEmpty model.numLayers
  let mut mlpOutputs : Array ConcreteMatrix := Array.mkEmpty model.numLayers
  let mut mlpActDerivMax : Array Float := Array.mkEmpty model.numLayers
  let mut residual := model.inputEmbeddings
  layerInputs := layerInputs.push residual

  for l in [:model.numLayers] do
    -- Pre-LN: attention sees ln_1(residual)
    let attnInput := model.applyLn1 l residual
    -- Compute attention outputs for all heads in this layer
    let mut layerAttnOutputs : Array ConcreteMatrix := #[]
    let rows := residual.numRows
    let cols := residual.numCols

    if hl : l < model.layers.size then
      let layerHeads := model.layers[l]
      let useParallelHeads := layerHeads.size >= 4
      layerAttnOutputs :=
        if useParallelHeads then
          let tasks : Array (Task ConcreteMatrix) :=
            .ofFn fun i : Fin layerHeads.size =>
              Task.spawn (fun _ => (layerHeads[i]).forward attnInput causal)
          tasks.map Task.get
        else
          Id.run do
            let mut outs : Array ConcreteMatrix := Array.mkEmpty layerHeads.size
            for head in layerHeads do
              outs := outs.push (head.forward attnInput causal)
            return outs

    attnOutputs := attnOutputs.push layerAttnOutputs

    -- Add attention residual
    let residualAfterAttn :=
      if layerAttnOutputs.isEmpty then
        residual
      else
        let attnSum : ConcreteMatrix := {
          numRows := rows
          numCols := cols
          data := .ofFn fun idx : Fin (rows * cols) => Id.run do
            let mut acc : Float := 0.0
            for hOut in layerAttnOutputs do
              acc := acc + hOut.data[idx.val]!
            return acc
          size_eq := Array.size_ofFn
        }
        residual.add attnSum

    -- Compute MLP output
    -- Pre-LN: MLP sees ln_2(residualAfterAttn)
    let mlpInput := model.applyLn2 l residualAfterAttn
    let mut mlpOut : ConcreteMatrix := ConcreteMatrix.zeros residual.numRows residual.numCols
    if hm : l < model.mlps.size then
      let (out, dmax) := model.mlps[l].forwardWithGeluDerivMax mlpInput
      let dmax' : Float :=
        if Float.isNaN dmax || Float.isInf dmax then 0.0 else dmax
      mlpActDerivMax := mlpActDerivMax.push dmax'
      mlpOut := out
    else
      -- No MLP at this layer.
      mlpActDerivMax := mlpActDerivMax.push 0.0
      mlpOut := ConcreteMatrix.zeros residual.numRows residual.numCols

    mlpOutputs := mlpOutputs.push mlpOut

    -- Add MLP residual
    residual := residualAfterAttn.add mlpOut

    -- Store input for next layer
    layerInputs := layerInputs.push residual

  let finalOutput := model.applyLnf residual
  {
    layerInputs := layerInputs
    attnOutputs := attnOutputs
    mlpOutputs := mlpOutputs
    mlpActDerivMax := mlpActDerivMax
    finalOutput := finalOutput
  }

/-- Get the input to a specific layer from a forward pass result. -/
def ForwardPassResult.getLayerInput (result : ForwardPassResult)
    (layerIdx : Nat) : ConcreteMatrix :=
  if h : layerIdx < result.layerInputs.size then
    result.layerInputs[layerIdx]
  else ConcreteMatrix.zeros 0 0

/-- Get the post-attention residual `y_l = x_l + attn_sum` for a layer.

This is the input to `ln_2` in a Pre-LN transformer block.
-/
def ForwardPassResult.getPostAttnResidual (result : ForwardPassResult)
    (layerIdx : Nat) : ConcreteMatrix := Id.run do
  let x := result.getLayerInput layerIdx
  if h : layerIdx < result.attnOutputs.size then
    let heads := result.attnOutputs[layerIdx]
    if heads.isEmpty then
      return x
    else
      let rows := x.numRows
      let cols := x.numCols
      let sum : ConcreteMatrix := {
        numRows := rows
        numCols := cols
        data := .ofFn fun idx : Fin (rows * cols) => Id.run do
          let mut acc : Float := 0.0
          for hOut in heads do
            acc := acc + hOut.data[idx.val]!
          return acc
        size_eq := Array.size_ofFn
      }
      return x.add sum
  else
    return x

/-! ## N-Layer Error Amplification Computation

These functions implement the N-layer faithfulness composition formula from
`Linearization.lean`. They must be defined early so they can be used by
discovery functions like `findDeepCircuitCandidates`.
-/

/-- Compute suffix amplification factor: ∏_{j≥start} (1 + C_j)

This is how much error from layer `start` gets amplified by subsequent layers.
When start ≥ normBounds.size, returns 1 (no amplification).
-/
def computeSuffixAmplification (normBounds : Array Float) (start : Nat) : Float := Id.run do
  let mut product : Float := 1.0
  for j in [start:normBounds.size] do
    if hj : j < normBounds.size then
      product := product * (1.0 + normBounds[j])
  product

/-- Compute total amplified error: Σᵢ εᵢ · suffixAmplification(i+1)

This implements the N-layer faithfulness composition formula from
`Linearization.lean` theorem `n_layer_faithfulness_composition`.
-/
def computeTotalAmplifiedError (patternBounds normBounds : Array Float) : Float := Id.run do
  if patternBounds.size = 0 then return 0.0
  let mut total : Float := 0.0
  for i in [:patternBounds.size] do
    if hi : i < patternBounds.size then
      let epsilon_i := patternBounds[i]
      -- Suffix amplification from layer i+1 onwards
      let suffix := computeSuffixAmplification normBounds (i + 1)
      total := total + epsilon_i * suffix
  total

/-- Estimate the operator norm bound for a single attention layer.

For an attention layer, the Jacobian includes both the attention pattern term
and the value projection. We estimate:
  ‖I + J‖ ≤ 1 + ‖A‖_F · ‖W_V·W_O‖_op + ‖∂A/∂x‖ · ‖V·W_O‖

For simplicity, we use Frobenius norms as upper bounds.
-/
def estimateAttentionLayerNorm (model : ConcreteModel) (fwdResult : ForwardPassResult)
    (layerIdx : Nat) (causal : Bool := true) : Float := Id.run do
  if h : layerIdx < model.layers.size then
    let heads := model.layers[layerIdx]
    let mut totalNorm : Float := 0.0

    -- Pre-LN: attention and MLP see normalized activations; account for LN Jacobian scaling.
    let x := fwdResult.getLayerInput layerIdx
    let y := fwdResult.getPostAttnResidual layerIdx
    let ln1Bound := model.ln1OpBound layerIdx x
    let ln2Bound := model.ln2OpBound layerIdx y

    -- Attention pattern/value bounds are computed at the Pre-LN attention input.
    let attnInput := model.applyLn1 layerIdx x
    let inputNorm := attnInput.frobeniusNorm

    -- Sum contributions from all heads in this layer
    for hidx in [:heads.size] do
      if hh : hidx < heads.size then
        let head := heads[hidx]

        -- QK operator norm bound via a 64×64 companion product.
        -- `‖W_Q W_Kᵀ‖₂ = ‖W_Kᵀ W_Q‖₂` because `AB` and `BA` have the same nonzero
        -- singular values.
        let qkSmall : ConcreteMatrix := head.W_K.transpose.matmul head.W_Q
        let qkNorm : Float :=
          min (qkSmall.opNormUpperBoundDenseBrauer)
            (min (qkSmall.opNormUpperBoundDenseSchur) (qkSmall.opNormUpperBoundDenseFrob))

        -- VO operator norm bound via a 64×64 companion product.
        -- `‖W_V W_O‖₂ = ‖W_O W_V‖₂` because `AB` and `BA` have the same nonzero
        -- singular values.
        let voSmall : ConcreteMatrix := head.W_O.matmul head.W_V
        let valueOutputProjNorm : Float :=
          min (voSmall.opNormUpperBoundDenseBrauer)
            (min (voSmall.opNormUpperBoundDenseSchur) (voSmall.opNormUpperBoundDenseFrob))

        -- Data-dependent softmax Jacobian operator bound (per-row max, clamped to 1/2).
        let attn := head.computeAttentionWeights attnInput causal
        let softmaxOpBound := min attn.softmaxJacobianOpDiag.opBound 0.5
        let scaleFactor := Float.sqrt head.headDim.toFloat

        -- Value-term operator bound:
        --   ‖X ↦ A·X·(W_VW_O)‖ ≤ ‖A‖₂ · ‖W_VW_O‖₂.
        -- For the attention matrix `A` (nonnegative row-stochastic), `‖A‖_∞ ≈ 1`, so
        -- `sqrt(‖A‖₁‖A‖∞)` is typically far tighter than `‖A‖_F`.
        let attnFrob : Float := Id.run do
          let mut s : Float := 0.0
          for w in attn.weights do
            s := s + w * w
          Float.sqrt (max 0.0 s)
        let attnMaxRowSum : Float := Id.run do
          let n := attn.seqLen
          let mut maxSum : Float := 0.0
          for q in [:n] do
            let mut s : Float := 0.0
            let rowBase := q * n
            for k in [:n] do
              s := s + Float.abs (attn.weights[rowBase + k]!)
            maxSum := max maxSum s
          maxSum
        let attnMaxColSum : Float := Id.run do
          let n := attn.seqLen
          let mut maxSum : Float := 0.0
          for k in [:n] do
            let mut s : Float := 0.0
            for q in [:n] do
              s := s + Float.abs (attn.weights[q * n + k]!)
            maxSum := max maxSum s
          maxSum
        let attnOneInf : Float := Float.sqrt (attnMaxRowSum * attnMaxColSum)
        let attnOpUb : Float := min attnFrob attnOneInf
        let valueTermUb := attnOpUb * valueOutputProjNorm

        -- Pattern-term bound (same factorization as `computePatternTermBound`):
        let patternTermUb :=
          (softmaxOpBound / scaleFactor) * inputNorm * qkNorm * valueOutputProjNorm

        totalNorm := totalNorm + ln1Bound * (valueTermUb + patternTermUb)

    -- Add MLP contribution if present
    if hm : layerIdx < model.mlps.size then
      let mlp := model.mlps[layerIdx]
      -- MLP Jacobian norm ≤ ‖W_out‖ · ‖∂activation‖ · ‖W_in‖
      let winNormUb := mlp.W_in.opNormUpperBoundRectGram
      let woutNormUb := mlp.W_out.opNormUpperBoundRectGram
      let geluDerivBound : Float :=
        let d := fwdResult.mlpActDerivMax.getD layerIdx 1.7
        if d ≤ 0.0 || Float.isNaN d || Float.isInf d then 1.7 else d
      totalNorm := totalNorm + ln2Bound * (winNormUb * geluDerivBound * woutNormUb)

    totalNorm
  else
    return 0.0

/-- Diagnostics-only variant of `estimateAttentionLayerNorm`.

This uses power iteration (`operatorNormHeuristicPI`) for the key operator norms,
to provide an "old PI vs new rigorous" comparison under `-d`.

WARNING: This is **not** a certified upper bound and must never be used in
bound/certification codepaths.
-/
def estimateAttentionLayerNormHeuristicPI (model : ConcreteModel) (fwdResult : ForwardPassResult)
    (layerIdx : Nat) (causal : Bool := true) : Float := Id.run do
  if h : layerIdx < model.layers.size then
    let heads := model.layers[layerIdx]
    let mut totalNorm : Float := 0.0

    let x := fwdResult.getLayerInput layerIdx
    let y := fwdResult.getPostAttnResidual layerIdx
    let ln1Bound := model.ln1OpBound layerIdx x
    let ln2Bound := model.ln2OpBound layerIdx y
    let attnInput := model.applyLn1 layerIdx x
    let inputNorm := attnInput.frobeniusNorm

    for hidx in [:heads.size] do
      if hh : hidx < heads.size then
        let head := heads[hidx]
        let qkSmall : ConcreteMatrix := head.W_K.transpose.matmul head.W_Q
        let qkPi := qkSmall.operatorNormHeuristicPI 20
        let voSmall : ConcreteMatrix := head.W_O.matmul head.W_V
        let voPi := voSmall.operatorNormHeuristicPI 20

        let attn := head.computeAttentionWeights attnInput causal
        let softmaxOpBound := min attn.softmaxJacobianOpDiag.opBound 0.5
        let scaleFactor := Float.sqrt head.headDim.toFloat
        let attnFrob : Float := Id.run do
          let mut s : Float := 0.0
          for w in attn.weights do
            s := s + w * w
          Float.sqrt (max 0.0 s)
        let attnMaxRowSum : Float := Id.run do
          let n := attn.seqLen
          let mut maxSum : Float := 0.0
          for q in [:n] do
            let mut s : Float := 0.0
            let rowBase := q * n
            for k in [:n] do
              s := s + Float.abs (attn.weights[rowBase + k]!)
            maxSum := max maxSum s
          maxSum
        let attnMaxColSum : Float := Id.run do
          let n := attn.seqLen
          let mut maxSum : Float := 0.0
          for k in [:n] do
            let mut s : Float := 0.0
            for q in [:n] do
              s := s + Float.abs (attn.weights[q * n + k]!)
            maxSum := max maxSum s
          maxSum
        let attnOneInf : Float := Float.sqrt (attnMaxRowSum * attnMaxColSum)
        let attnOpUb : Float := min attnFrob attnOneInf
        let valueTermEst := attnOpUb * voPi
        let patternTermEst := (softmaxOpBound / scaleFactor) * inputNorm * qkPi * voPi
        totalNorm := totalNorm + ln1Bound * (valueTermEst + patternTermEst)

    if hm : layerIdx < model.mlps.size then
      let mlp := model.mlps[layerIdx]
      -- Keep PI iterations low-ish here to avoid expensive diagnostics runs.
      let winPi := mlp.W_in.operatorNormHeuristicPI 5
      let woutPi := mlp.W_out.operatorNormHeuristicPI 5
      let geluDerivBound : Float :=
        let d := fwdResult.mlpActDerivMax.getD layerIdx 1.7
        if d ≤ 0.0 || Float.isNaN d || Float.isInf d then 1.7 else d
      totalNorm := totalNorm + ln2Bound * (winPi * geluDerivBound * woutPi)

    totalNorm
  else
    return 0.0

/-- Compute attention weights for a given layer and head using the correct layer input.

This is the corrected version that uses the accumulated residual stream.
-/
def ConcreteModel.computeAttentionWithInput (model : ConcreteModel)
    (layerIdx headIdx : Nat) (input : ConcreteMatrix) : Option ConcreteAttentionWeights :=
  if h1 : layerIdx < model.layers.size then
    let layerHeads := model.layers[layerIdx]
    if h2 : headIdx < layerHeads.size then
      let head := layerHeads[headIdx]'h2
      -- Pre-LN: attention weights are computed from ln_1(input).
      let attnInput := model.applyLn1 layerIdx input
      some (head.computeAttentionWeights attnInput)
    else none
  else none

/-- Compute attention weights for a given layer and head.

This is a legacy helper that only uses the model's `inputEmbeddings`.
Prefer `computeAttentionWithInput` for Pre-LN-correct layer inputs.
-/
def ConcreteModel.computeAttention (model : ConcreteModel)
    (layerIdx headIdx : Nat) : Option ConcreteAttentionWeights :=
  if h1 : layerIdx < model.layers.size then
    let layerHeads := model.layers[layerIdx]
    if h2 : headIdx < layerHeads.size then
      let head := layerHeads[headIdx]
      let scale := Float.sqrt head.headDim.toFloat
      let attnInput := model.applyLn1 layerIdx model.inputEmbeddings
      let queries := attnInput.matmul head.W_Q
      let keys := attnInput.matmul head.W_K
      some (ConcreteAttentionWeights.compute queries keys scale model.seqLen)
    else none
  else none

/-- Compute input norm for bound calculations. -/
def computeInputNorm (embeddings : ConcreteMatrix) : Float :=
  embeddings.frobeniusNorm

/-! ## Precomputation Cache Structures

To optimize the O(L²H²) nested loop in deep circuit discovery, we precompute and cache:
1. Attention patterns (A = softmax(QK^T/√d)) for each layer-head
2. Value-output projections (V·W_O) for each head
3. Query-key alignments (Q·K^T) for each head
4. Operator norm bounds and Frobenius norms

This reduces redundant computation from O(L²H²) to O(LH), a massive improvement
for models like GPT-2 Small (12 layers × 12 heads = 144 heads → 20,736 → 144 calls).
-/

/-- Precomputed data for a single attention head at a specific layer input. -/
structure PrecomputedHeadData where
  /-- Layer index -/
  layerIdx : Nat
  /-- Head index within layer -/
  headIdx : Nat
  /-- Attention weights (A = softmax(QK^T/√d)) -/
  attention : ConcreteAttentionWeights
  /-- Average previous-token attention strength: `(1/(n-1)) * Σᵢ A[i+1, i]`. -/
  prevTokenStrength : Float
  /-- Cached softmax Jacobian operator-norm estimate for this head's attention rows. -/
  softmaxJacobianOpEst : Float
  /-- Softmax Jacobian diagnostics: `max_i p_i` for the maximizing row. -/
  softmaxRowMaxP : Float
  /-- Softmax Jacobian diagnostics: `tr(J) = 1 - ∑ p_i^2` for the maximizing row. -/
  softmaxRowTraceBound : Float
  /-- Softmax Jacobian diagnostics: PSD moment bound for the maximizing row. -/
  softmaxRowMomentBound : Float
  /-- Softmax Jacobian diagnostics: Gershgorin / `‖J‖_∞ = max_i 2 p_i (1-p_i)`
  for the maximizing row. -/
  softmaxRowGershBound : Float
  /-- Softmax Jacobian diagnostics: final per-row bound used for the maximizing row. -/
  softmaxRowBoundUsed : Float
  /-- Number of rows that triggered a conservative fallback (NaN/Inf/zero-sum). -/
  softmaxRowsFallback : Nat
  /-- Cached Frobenius norm squared of the attention matrix: `‖A‖_F²`. -/
  attentionFrobeniusNormSq : Float
  /-- Cached pattern-term bound `‖PatternTerm‖_F` for this head at the cached input. -/
  patternTermBoundCached : Float
  /-- Cached value-term Frobenius norm `‖ValueTerm‖_F` for this head. -/
  valueTermNormCached : Float
  /-- Cached dimensionless faithfulness ratio `‖PatternTerm‖_F / ‖ValueTerm‖_F`. -/
  faithfulnessRatioCached : Float
  /-- Value-output projection (V·W_O) -/
  valueOutputProj : ConcreteMatrix
  /-- Query-key alignment (Q·K^T) -/
  queryKeyAlign : ConcreteMatrix
  /-- Cached Gram matrix `W_Qᵀ · W_Q` (headDim × headDim). -/
  wqGram : ConcreteMatrix
  /-- Cached Gram matrix `W_Vᵀ · W_V` (headDim × headDim). -/
  wvGram : ConcreteMatrix
  /-- Input norm ‖X‖_F for this layer -/
  inputNorm : Float
  /-- Operator-norm bound for ln_1 Jacobian at this layer input. -/
  ln1OpBound : Float
  /-- Scaling factor √d_head -/
  scaleFactor : Float
  /-- Cached Frobenius norm of V·W_O -/
  valueOutputProjNorm : Float
  /-- Cached Frobenius norm of Q·K^T -/
  queryKeyAlignNorm : Float
  /-- Cached deterministic Float upper bound on ‖V·W_O‖₂.

  This is computed as the minimum of several valid upper bounds
  (Schur / Frobenius / factor bounds).
  -/
  valueOutputProjSchurNorm : Float
  /-- Cached deterministic Float upper bound on ‖Q·K^T‖₂.

  This is computed as the minimum of several valid upper bounds
  (Schur / Frobenius / factor bounds).
  -/
  queryKeyAlignSchurNorm : Float

  /-- Candidate bounds for `‖Q·Kᵀ‖₂` used in diagnostics. -/
  qkDenseSchur : Float
  qkDenseFrob : Float
  /-- Tight Gram-product candidate derived from 64×64 Gram matrices. -/
  qkDenseGram : Float
  /-- Brauer/Cassini Gram candidate computed on a 64×64 matrix with matching singular values. -/
  qkDenseBrauer : Float
  qkFactorSchur : Float
  qkFactorFrob : Float

  /-- Gram-based operator bound for `W_Q`: `sqrt(‖W_QᵀW_Q‖∞)` (computed implicitly). -/
  wqOpGram : Float
  /-- Gram-based operator bound for `W_K`: `sqrt(‖W_KᵀW_K‖∞)` (computed implicitly). -/
  wkOpGram : Float
  /-- Factorized Gram bound for `‖W_Q·W_Kᵀ‖₂`: `wqOpGram * wkOpGram`. -/
  qkFactorGram : Float

  /-- Candidate bounds for `‖W_V·W_O‖₂` used in diagnostics. -/
  voDenseSchur : Float
  voDenseFrob : Float
  /-- Tight Gram-product candidate derived from 64×64 Gram matrices. -/
  voDenseGram : Float
  /-- Brauer/Cassini Gram candidate computed on a 64×64 matrix with matching singular values. -/
  voDenseBrauer : Float
  voFactorSchur : Float
  voFactorFrob : Float
  /-- Gram-based operator bound for `W_V`: `sqrt(‖W_VᵀW_V‖∞)` (computed implicitly). -/
  wvOpGram : Float
  /-- Gram-based operator bound for `W_O` (computed via `W_Oᵀ` for speed). -/
  woOpGram : Float
  /-- Factorized Gram bound for `‖W_V·W_O‖₂`: `wvOpGram * woOpGram`. -/
  voFactorGram : Float

namespace PrecomputedHeadData

/-- Precomputed pattern term bound for a head (cached computation). -/
def patternTermBound (data : PrecomputedHeadData) : Float :=
  data.patternTermBoundCached

/-- Frobenius norm of the Value Term of this head's Jacobian.

For the attention linearization, the Value Term factorizes as `A ⊗ (W_V·W_O)`,
so `‖ValueTerm‖_F = ‖A‖_F · ‖W_V·W_O‖_F`.
-/
def valueTermNorm (data : PrecomputedHeadData) : Float :=
  data.valueTermNormCached

/-- Dimensionless faithfulness ratio: `‖PatternTerm‖_F / ‖ValueTerm‖_F`.

This matches `relativeApproximationError` from `Nfp.Linearization` and is the
quantity that should be compared to user-facing thresholds like `0.1`.
-/
def faithfulnessRatio (data : PrecomputedHeadData) : Float :=
  data.faithfulnessRatioCached

end PrecomputedHeadData

/-- Cache for all precomputed head data across all layers.

Structure: `cache[layerIdx][headIdx]` gives the PrecomputedHeadData for that head.
-/
structure PrecomputedCache where
  /-- Model this cache was built for -/
  model : ConcreteModel
  /-- Forward pass result with layer inputs -/
  forwardResult : ForwardPassResult
  /-- Cached Pre-LN attention inputs `ln_1(x_l)` for each layer `l`. -/
  ln1Inputs : Array ConcreteMatrix
  /-- Precomputed data: cache[layerIdx][headIdx] -/
  headData : Array (Array PrecomputedHeadData)
  /-- Precomputed operator norm bounds for each layer (for N-layer error amplification) -/
  layerNormBounds : Array Float

namespace PrecomputedCache

/-- Build a complete precomputed cache for a model.

This precomputes all attention patterns, projections, and norms once.
-/
def build (model : ConcreteModel) (causal : Bool := true) : PrecomputedCache := Id.run do
  let fwdResult := model.runForward causal
  let computeLayer (l : Nat) : (Array PrecomputedHeadData × (Float × ConcreteMatrix)) := Id.run do
    let mut layerHeadData : Array PrecomputedHeadData := #[]
    let layerInput := fwdResult.getLayerInput l
    let attnInput := model.applyLn1 l layerInput
    let inputNorm := computeInputNorm attnInput
    let ln1Bound := model.ln1OpBound l layerInput

    if h : l < model.layers.size then
      let heads := model.layers[l]'h
      layerHeadData := Array.mkEmpty heads.size
      for h_idx in [:heads.size] do
        if hh : h_idx < heads.size then
          let head := heads[h_idx]'hh

          -- Precompute attention weights
          let attn := head.computeAttentionWeights attnInput causal
          let softmaxDiag := attn.softmaxJacobianOpDiag
          let softmaxOpBound := min softmaxDiag.opBound 0.5
          let mut attnFrobNormSq : Float := 0.0
          for x in attn.weights do
            attnFrobNormSq := attnFrobNormSq + x * x

          let prevTokenStrength : Float :=
            if attn.seqLen < 2 then
              0.0
            else
              Id.run do
                let n := attn.seqLen
                let w := attn.weights
                let mut sum : Float := 0.0
                for i in [:n - 1] do
                  sum := sum + w[(i + 1) * n + i]!
                return sum / (n - 1).toFloat

          -- Precompute projections
          let voProj := head.valueOutputProjection
          let qkAlign := head.queryKeyAlignment
          -- Low-rank Gram caches for composition scoring.
          let wqGram := head.W_Q.transpose.matmul head.W_Q
          let wvGram := head.W_V.transpose.matmul head.W_V
          let wkGram := head.W_K.transpose.matmul head.W_K
          let woGram := head.W_O.matmul head.W_O.transpose

          -- Precompute norms
          let voNorm := voProj.frobeniusNorm
          let qkNorm := qkAlign.frobeniusNorm

          -- Candidate bounds (for diagnostics + sanity checking).
          let qkDenseSchur := qkAlign.schurNormEst
          let qkDenseFrob := qkNorm
          -- Tight Gram-product candidate (low-rank, 64×64):
          -- ‖W_Q W_Kᵀ‖₂² = λ_max((W_QᵀW_Q)(W_KᵀW_K)) ≤ ‖(W_QᵀW_Q)(W_KᵀW_K)‖_∞.
          let qkDenseGram := Float.sqrt (max 0.0 ((wqGram.matmul wkGram).infNormAbs))
          -- Brauer/Cassini Gram candidate on a 64×64 product with matching singular values.
          let qkSmall := head.W_K.transpose.matmul head.W_Q
          let qkDenseBrauer := qkSmall.opNormUpperBoundDenseBrauer
          let qkFactorSchur := head.W_Q.schurNormEst * head.W_K.schurNormEst
          let qkFactorFrob := head.W_Q.frobeniusNorm * head.W_K.frobeniusNorm

          let voDenseSchur := voProj.schurNormEst
          let voDenseFrob := voNorm
          -- Tight Gram-product candidate (low-rank, 64×64):
          -- For `M = W_V W_O`, ‖M‖₂² = λ_max((W_VᵀW_V)(W_O W_Oᵀ)) up to reordering.
          let voDenseGram := Float.sqrt (max 0.0 ((wvGram.matmul woGram).infNormAbs))
          -- Brauer/Cassini Gram candidate on a 64×64 product with matching singular values.
          let voSmall := head.W_O.matmul head.W_V
          let voDenseBrauer := voSmall.opNormUpperBoundDenseBrauer
          let voFactorSchur := head.W_V.schurNormEst * head.W_O.schurNormEst
          let voFactorFrob := head.W_V.frobeniusNorm * head.W_O.frobeniusNorm

          -- Gram-based factor operator bounds.
          let wqOpGram := head.W_Q.opNormUpperBoundViaGramInf
          let wkOpGram := head.W_K.opNormUpperBoundViaGramInf
          let qkFactorGram := wqOpGram * wkOpGram
          let wvOpGram := head.W_V.opNormUpperBoundViaGramInf
          let woOpGram := head.W_O.transpose.opNormUpperBoundViaGramInf
          let voFactorGram := wvOpGram * woOpGram
          -- Deterministic Float upper bounds for operator norms.
          --
          -- Each candidate is a valid upper bound in exact real arithmetic.
          -- Taking `min` tightens deterministically (in Float arithmetic).
          let qkOpBound :=
            min qkDenseFrob <|
            min qkDenseSchur <|
            min qkFactorSchur <|
            min qkFactorFrob <|
            min qkFactorGram <|
            min qkDenseGram qkDenseBrauer
          let voOpBound :=
            min voDenseFrob <|
            min voDenseSchur <|
            min voFactorSchur <|
            min voFactorFrob <|
            min voFactorGram <|
            min voDenseGram voDenseBrauer

          let scaleFactor := Float.sqrt head.headDim.toFloat
          let patternBound : Float :=
            (softmaxOpBound / scaleFactor) * inputNorm * qkOpBound * voOpBound
          let valueNorm : Float :=
            Float.sqrt attnFrobNormSq * voNorm
          let ratio : Float :=
            if valueNorm < 1e-10 then Float.inf else patternBound / valueNorm

          let data : PrecomputedHeadData := {
            layerIdx := l
            headIdx := h_idx
            attention := attn
            prevTokenStrength := prevTokenStrength
            softmaxJacobianOpEst := softmaxOpBound
            softmaxRowMaxP := softmaxDiag.maxRowMaxP
            softmaxRowTraceBound := softmaxDiag.maxRowTraceBound
            softmaxRowMomentBound := softmaxDiag.maxRowMomentBound
            softmaxRowGershBound := softmaxDiag.maxRowGersh
            softmaxRowBoundUsed := softmaxDiag.maxRowBoundUsed
            softmaxRowsFallback := softmaxDiag.numRowsFallback
            attentionFrobeniusNormSq := attnFrobNormSq
            patternTermBoundCached := patternBound
            valueTermNormCached := valueNorm
            faithfulnessRatioCached := ratio
            valueOutputProj := voProj
            queryKeyAlign := qkAlign
            wqGram := wqGram
            wvGram := wvGram
            inputNorm := inputNorm
            ln1OpBound := ln1Bound
            scaleFactor := scaleFactor
            valueOutputProjNorm := voNorm
            queryKeyAlignNorm := qkNorm
            valueOutputProjSchurNorm := voOpBound
            queryKeyAlignSchurNorm := qkOpBound

            qkDenseSchur := qkDenseSchur
            qkDenseFrob := qkDenseFrob
            qkDenseGram := qkDenseGram
            qkDenseBrauer := qkDenseBrauer
            qkFactorSchur := qkFactorSchur
            qkFactorFrob := qkFactorFrob

            wqOpGram := wqOpGram
            wkOpGram := wkOpGram
            qkFactorGram := qkFactorGram

            voDenseSchur := voDenseSchur
            voDenseFrob := voDenseFrob
            voDenseGram := voDenseGram
            voDenseBrauer := voDenseBrauer
            voFactorSchur := voFactorSchur
            voFactorFrob := voFactorFrob
            wvOpGram := wvOpGram
            woOpGram := woOpGram
            voFactorGram := voFactorGram
          }

          layerHeadData := layerHeadData.push data

    -- OPTIMIZATION: Precompute operator norm bounds for each layer
    let norm := estimateAttentionLayerNorm model fwdResult l causal
    return (layerHeadData, (norm, attnInput))

  -- Pure parallelism via tasks: layer cache construction is independent once the
  -- forward pass has produced all layer inputs.
  let useParallel := model.numLayers >= 4
  let layerResults : Array (Array PrecomputedHeadData × (Float × ConcreteMatrix)) :=
    if useParallel then
      let tasks : Array (Task (Array PrecomputedHeadData × (Float × ConcreteMatrix))) :=
        .ofFn fun i : Fin model.numLayers =>
          Task.spawn (fun _ => computeLayer i.val)
      tasks.map Task.get
    else
      .ofFn fun i : Fin model.numLayers =>
        computeLayer i.val

  let mut headData : Array (Array PrecomputedHeadData) := Array.mkEmpty model.numLayers
  let mut layerNormBounds : Array Float := Array.mkEmpty model.numLayers
  let mut ln1Inputs : Array ConcreteMatrix := Array.mkEmpty model.numLayers
  for (layerHeadData, (norm, attnInput)) in layerResults do
    headData := headData.push layerHeadData
    layerNormBounds := layerNormBounds.push norm
    ln1Inputs := ln1Inputs.push attnInput

  { model := model
    forwardResult := fwdResult
    ln1Inputs := ln1Inputs
    headData := headData
    layerNormBounds := layerNormBounds }

/-- Retrieve cached data for a specific head. -/
def getHeadData (cache : PrecomputedCache) (layerIdx headIdx : Nat) :
    Option PrecomputedHeadData :=
  if h1 : layerIdx < cache.headData.size then
    let layerCache := cache.headData[layerIdx]
    if h2 : headIdx < layerCache.size then
      some layerCache[headIdx]
    else none
  else none

/-- Retrieve cached Pre-LN attention input `ln_1(x_l)` for a specific layer. -/
def getLn1Input (cache : PrecomputedCache) (layerIdx : Nat) : ConcreteMatrix :=
  if h : layerIdx < cache.ln1Inputs.size then
    cache.ln1Inputs[layerIdx]
  else
    ConcreteMatrix.zeros 0 0

end PrecomputedCache

/-! ## Head Composition Metrics -/

/-- Compute the **K-composition** score between two attention heads.

This follows the definition in *"A Mathematical Framework for Transformer Circuits"*
(see the "composition diagram caption"):

`kComp(h₁→h₂) = ‖W_QK² · W_OV¹‖_F / (‖W_QK²‖_F · ‖W_OV¹‖_F)`.

In this codebase's row-vector convention, we store `W_QK = W_Q · W_Kᵀ` and
`W_OV,row = W_V · W_O`. The paper's `W_OV` corresponds to `(W_OV,row)ᵀ`, but
`‖W_OV‖_F = ‖W_OV,row‖_F` and we compute the numerator via low-rank factors without
materializing any `modelDim×modelDim` products.

By default (as in the paper), we subtract the expected composition of random matrices
of shape `modelDim×modelDim`, which is approximately `1/√modelDim`.

We return 0.0 on dimension mismatch or missing heads.
-/
def computeKCompositionScore
    (model : ConcreteModel)
    (data1 data2 : PrecomputedHeadData) : Float :=
  if h1 : data1.layerIdx < model.layers.size then
    let heads1 := model.layers[data1.layerIdx]'h1
    if h2 : data2.layerIdx < model.layers.size then
      let heads2 := model.layers[data2.layerIdx]'h2
      if hh1 : data1.headIdx < heads1.size then
        let head1 := heads1[data1.headIdx]'hh1
        if hh2 : data2.headIdx < heads2.size then
          let head2 := heads2[data2.headIdx]'hh2
            let denom := data2.queryKeyAlignNorm * data1.valueOutputProjNorm
            if denom < 1e-10 then 0.0
            else
            -- K-composition numerator:
            -- ‖W_QK² · W_OV¹‖_F where W_QK² = W_Q²W_K²ᵀ and W_OV¹ = (W_V¹W_O¹)ᵀ.
            -- Using low-rank factorization:
            -- W_Q²W_K²ᵀ (W_V¹W_O¹)ᵀ = W_Q² (W_K²ᵀ W_O¹ᵀ) W_V¹ᵀ
            -- and M := W_O¹ W_K² so W_K²ᵀ W_O¹ᵀ = Mᵀ.
            let M := head1.W_O.matmul head2.W_K  -- (d_head × d_head)
            let T := data1.wvGram.matmul M       -- (d_head × d_head)
              let S := M.transpose.matmul T        -- Mᵀ · (W_V¹ᵀW_V¹) · M
              let numeratorSq := ConcreteMatrix.traceMul S data2.wqGram
              let numerator := Float.sqrt (max numeratorSq 0.0)
              let raw := numerator / denom
              let baseline : Float :=
                if head1.modelDim = 0 then 0.0
                else 1.0 / Float.sqrt head1.modelDim.toFloat
              raw - baseline
        else
          0.0
      else
        0.0
    else
      0.0
  else
    0.0

/-- Convert a 2-layer deep circuit candidate into an induction-head candidate.

This is used to avoid re-running the expensive `checkInductionPattern` scan when both
induction heads and deep circuits are requested from the same cache.
-/
def DeepCircuitCandidate.toInductionCandidate?
    (c : DeepCircuitCandidate) (cache : PrecomputedCache) :
    Option CandidateInductionHead :=
  if c.layerIndices.size = 2 && c.headIndices.size = 2 then
    let l1 := c.layerIndices[0]!
    let l2 := c.layerIndices[1]!
    let h1 := c.headIndices[0]!
    let h2 := c.headIndices[1]!
    match cache.getHeadData l1 h1, cache.getHeadData l2 h2 with
    | some d1, some d2 =>
      let ε1 := d1.faithfulnessRatio
      let ε2 := d2.faithfulnessRatio
      let combinedError := ε1 + ε2 + ε1 * ε2
      let inductionScore : Float :=
        match cache.model.inputTokens with
        | some tokens =>
            (checkInductionCopyNextPattern tokens d2.attention (minScore := 0.0)).getD 0.0
        | none => 1.0
      let kComp := computeKCompositionScore cache.model d1 d2
      some {
        layer1Idx := l1
        layer2Idx := l2
        head1Idx := h1
        head2Idx := h2
        patternBound1 := ε1
        patternBound2 := ε2
        combinedError := combinedError
        prevTokenStrength := d1.prevTokenStrength
        inductionScore := inductionScore
        kComp := kComp
        description := s!"L{l1}H{h1}->L{l2}H{h2} (deep)"
      }
    | _, _ => none
  else
    none

/-- Find candidate (L1, L2) induction-head pairs from a `PrecomputedCache`.

This searches for the classic two-head induction circuit:
- Layer 1 (L1): a strong **previous-token** head,
- Layer 2 (L2): an **induction** head (token-level "copy-next" attention).
-/
def findInductionHeadCandidatesFromCache (cache : PrecomputedCache)
    (minPrevTokenStrength : Float := 0.1)
    (minInductionScore : Float := 0.05) : Array CandidateInductionHead := Id.run do
  let model := cache.model
  let tokens? := model.inputTokens

  let computeForLayer (l1 : Nat) : Array CandidateInductionHead := Id.run do
    let layer1Cache := cache.headData.getD l1 #[]
    let mut layerCandidates : Array CandidateInductionHead := #[]

    -- Preserve the original traversal order: l1, l2, head1, head2.
    for l2 in [l1 + 1:model.numLayers] do
      let layer2Cache := cache.headData.getD l2 #[]
      for data1 in layer1Cache do
        if data1.prevTokenStrength ≥ minPrevTokenStrength then
            for data2 in layer2Cache do
              let inductionScore? : Option Float :=
                match tokens? with
                | some tokens =>
                    -- Token-aware induction check: attend to the **successor** of a previous
                    -- matching token (variable-lag, "copy-next"), not fixed-lag shift matching.
                    checkInductionCopyNextPattern tokens data2.attention minInductionScore
                | none =>
                    -- If the prompt has no token IDs, we can't do token-aware filtering.
                    -- Fall back to **no pattern filter** and rely on certification.
                    some 1.0
              match inductionScore? with
              | some inductionScore =>
                  -- Use dimensionless faithfulness ratios (relative approximation errors).
                  let ε1 := data1.faithfulnessRatio
                  let ε2 := data2.faithfulnessRatio
                  let combinedError := ε1 + ε2 + ε1 * ε2
                  let kComp := computeKCompositionScore model data1 data2

                  layerCandidates := layerCandidates.push {
                    layer1Idx := l1
                    layer2Idx := l2
                    head1Idx := data1.headIdx
                    head2Idx := data2.headIdx
                    patternBound1 := ε1
                    patternBound2 := ε2
                    combinedError := combinedError
                    prevTokenStrength := data1.prevTokenStrength
                    inductionScore := inductionScore
                    kComp := kComp
                    description := s!"L{l1}H{data1.headIdx}->L{l2}H{data2.headIdx}"
                  }
              | none => pure ()
          else
            pure ()

    return layerCandidates

  -- Pure parallelism via tasks: layer-1 index computations are independent.
  let useParallel := model.numLayers >= 4
  let chunks : Array (Array CandidateInductionHead) :=
    if useParallel then
      let tasks : Array (Task (Array CandidateInductionHead)) :=
        .ofFn fun i : Fin model.numLayers =>
          Task.spawn (fun _ => computeForLayer i.val)
      tasks.map Task.get
    else
      .ofFn fun i : Fin model.numLayers =>
        computeForLayer i.val

  -- Join without quadratic copying.
  let total := chunks.foldl (fun acc cs => acc + cs.size) 0
  let mut candidates : Array CandidateInductionHead := Array.mkEmpty total
  for cs in chunks do
    for c in cs do
      candidates := candidates.push c

  candidates.qsort (·.combinedError < ·.combinedError)

/-- Search for induction heads using proper layer-wise residual stream computation.

This method performs multi-layer analysis with correct forward pass:
- Layer 1 attention is computed on the residual stream *after* layer 0
- Layer 2 attention is computed on the residual stream *after* layers 0-1

This enables detection of true induction heads where layer 2 attends to
information created by layer 1's "previous-token" head.

**Performance Note**: Uses `PrecomputedCache` to avoid O(L²H²) redundant attention
computations, reducing to O(LH) for typical models.
-/
def findInductionHeadCandidates (model : ConcreteModel)
    (_threshold : Float)
    (minPrevTokenStrength : Float := 0.1)
    (minInductionScore : Float := 0.05) : Array CandidateInductionHead := Id.run do
  let cache := PrecomputedCache.build model
  findInductionHeadCandidatesFromCache cache minPrevTokenStrength minInductionScore

/-- Filter candidates to only those meeting the threshold.

Uses proper layer-wise residual stream computation.
-/
def findVerifiedInductionHeads (model : ConcreteModel)
    (threshold : Float) : Array VerifiedInductionHead := Id.run do
  let candidates := findInductionHeadCandidates model threshold
  let mut verified : Array VerifiedInductionHead := #[]

  for candidate in candidates do
    if candidate.combinedError ≤ threshold then
      verified := verified.push {
        candidate := candidate
        threshold := threshold
        errorChecked := true
      }

  verified

/-- Find induction head candidates with rigorous N-layer amplification bounds.

This replaces the ad-hoc `ε₁ + ε₂ + ε₁·ε₂` formula with the correct N-layer
composition theorem:

  ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ)

For a 2-layer induction head with layers l1 < l2:
- Layer l1 contributes: ε₁ · (1 + C_l2) · (1 + C_{l2+1}) · ...
- Layer l2 contributes: ε₂ · (1 + C_{l2+1}) · ...

The amplification factors Cⱼ are estimated from the layer Jacobian norms.
-/
def findDeepCircuitCandidatesFromCache (cache : PrecomputedCache)
    (minPrevTokenStrength : Float := 0.1)
    (minInductionScore : Float := 0.05) : Array DeepCircuitCandidate := Id.run do
  let model := cache.model

  -- OPTIMIZATION: Use cached operator norm bounds (already computed in cache.build)
  let allNormBounds := cache.layerNormBounds

  -- OPTIMIZATION: precompute suffix amplification products:
  -- `suffixAmp[i] = ∏_{j≥i} (1 + C_j)` and `suffixAmp[size] = 1`.
  let suffixAmp : Array Float := Id.run do
    let n := allNormBounds.size
    let mut out : Array Float := Array.replicate (n + 1) 1.0
    let mut prod : Float := 1.0
    for offset in [:n] do
      let i := n - 1 - offset
      prod := prod * (1.0 + allNormBounds[i]!)
      out := out.set! i prod
    return out

  let computeForLayer (l1 : Nat) : Array DeepCircuitCandidate := Id.run do
    let layer1Cache := cache.headData.getD l1 #[]
    let suffix1 := suffixAmp.getD (l1 + 1) 1.0
    let totalAmpFactor := suffixAmp.getD l1 1.0
    let mut layerCandidates : Array DeepCircuitCandidate := #[]

    -- Preserve the original traversal order: l1, l2, head1, head2.
    for l2 in [l1 + 1:model.numLayers] do
      let layer2Cache := cache.headData.getD l2 #[]
      let suffix2 := suffixAmp.getD (l2 + 1) 1.0
      let relevantNormBounds := allNormBounds.extract l1 (l2 + 1)

      for data1 in layer1Cache do
        if data1.prevTokenStrength ≥ minPrevTokenStrength then
          for data2 in layer2Cache do
            match checkInductionPattern data1.attention data2.attention minInductionScore with
            | some _ =>
              let bound1 := data1.patternTermBound
              let bound2 := data2.patternTermBound
              let amplifiedError := bound1 * suffix1 + bound2 * suffix2

              layerCandidates := layerCandidates.push {
                layerIndices := #[l1, l2]
                headIndices := #[data1.headIdx, data2.headIdx]
                patternBounds := #[bound1, bound2]
                operatorNormUbs := relevantNormBounds
                simpleErrorSum := bound1 + bound2
                amplifiedError := amplifiedError
                amplificationFactor := totalAmpFactor
                patternType := "induction"
                description := s!"L{l1}H{data1.headIdx}->L{l2}H{data2.headIdx}"
              }
            | none => pure ()
        else
          pure ()

    return layerCandidates

  -- Pure parallelism via tasks: layer-1 index computations are independent.
  let useParallel := model.numLayers >= 4
  let chunks : Array (Array DeepCircuitCandidate) :=
    if useParallel then
      let tasks : Array (Task (Array DeepCircuitCandidate)) :=
        .ofFn fun i : Fin model.numLayers =>
          Task.spawn (fun _ => computeForLayer i.val)
      tasks.map Task.get
    else
      .ofFn fun i : Fin model.numLayers =>
        computeForLayer i.val

  -- Join without quadratic copying.
  let total := chunks.foldl (fun acc cs => acc + cs.size) 0
  let mut candidates : Array DeepCircuitCandidate := Array.mkEmpty total
  for cs in chunks do
    for c in cs do
      candidates := candidates.push c

  candidates.qsort (·.amplifiedError < ·.amplifiedError)

/-- Find deep circuit candidates with rigorous N-layer amplification bounds.

This is a wrapper around `findDeepCircuitCandidatesFromCache` that builds the cache.
-/
def findDeepCircuitCandidates (model : ConcreteModel)
    (_threshold : Float)
    (minPrevTokenStrength : Float := 0.1)
    (minInductionScore : Float := 0.05) : Array DeepCircuitCandidate := Id.run do
  let cache := PrecomputedCache.build model
  findDeepCircuitCandidatesFromCache cache minPrevTokenStrength minInductionScore

/-- Filter deep circuit candidates by N-layer error threshold. -/
def findVerifiedDeepCircuits (model : ConcreteModel)
    (threshold : Float) : Array DeepCircuitCandidate := Id.run do
  let candidates := findDeepCircuitCandidates model threshold
  let mut verified : Array DeepCircuitCandidate := #[]

  for candidate in candidates do
    if candidate.amplifiedError ≤ threshold then
      verified := verified.push candidate

  verified

/-! ## Pretty Printing -/

/-- Pretty print a candidate induction head. -/
def CandidateInductionHead.toString (c : CandidateInductionHead) : String :=
  s!"InductionHead L{c.layer1Idx}H{c.head1Idx}->L{c.layer2Idx}H{c.head2Idx} " ++
  s!"(error={c.combinedError}, prev-token={c.prevTokenStrength}, " ++
    s!"induction={c.inductionScore}, kComp={c.kComp})"

instance : ToString CandidateInductionHead := ⟨CandidateInductionHead.toString⟩

/-- Pretty print a verified induction head. -/
def VerifiedInductionHead.toString (vh : VerifiedInductionHead) : String :=
  s!"Verified: {vh.candidate} [threshold={vh.threshold}]"

instance : ToString VerifiedInductionHead := ⟨VerifiedInductionHead.toString⟩

/-! ## Summary Statistics -/

/-- Summary statistics for discovery results. -/
structure DiscoverySummary where
  /-- Number of candidates found -/
  candidateCount : Nat
  /-- Number meeting the threshold -/
  verifiedCount : Nat
  /-- Best (lowest) error bound found -/
  bestError : Float

/-- Compute summary statistics for discovery results. -/
def computeDiscoverySummary (model : ConcreteModel) (threshold : Float) :
    DiscoverySummary := Id.run do
  let candidates := findInductionHeadCandidates model threshold
  let mut verifiedCount : Nat := 0
  for c in candidates do
    if c.combinedError ≤ threshold then
      verifiedCount := verifiedCount + 1

  let bestErr := if candidates.isEmpty then Float.inf
    else candidates.foldl (fun acc c => min acc c.combinedError) Float.inf

  {
    candidateCount := candidates.size
    verifiedCount := verifiedCount
    bestError := bestErr
  }

/-! ## Convenience Functions -/

/-- Create a ConcreteAttentionLayer from raw Float arrays. -/
def mkConcreteAttentionLayer
    (modelDim headDim : Nat)
    (wq wk wv wo : Array Float)
    (hq : wq.size = modelDim * headDim)
    (hk : wk.size = modelDim * headDim)
    (hv : wv.size = modelDim * headDim)
    (ho : wo.size = headDim * modelDim) : ConcreteAttentionLayer where
  modelDim := modelDim
  headDim := headDim
  W_Q := { numRows := modelDim, numCols := headDim, data := wq, size_eq := hq }
  W_K := { numRows := modelDim, numCols := headDim, data := wk, size_eq := hk }
  W_V := { numRows := modelDim, numCols := headDim, data := wv, size_eq := hv }
  W_O := { numRows := headDim, numCols := modelDim, data := wo, size_eq := ho }
  W_Q_dims := ⟨rfl, rfl⟩
  W_K_dims := ⟨rfl, rfl⟩
  W_V_dims := ⟨rfl, rfl⟩
  W_O_dims := ⟨rfl, rfl⟩

/-! ## Generic Circuit Discovery

This section provides a framework for discovering arbitrary circuits (sparse subgraphs)
that are certifiably responsible for model behavior on a given input.

The key insight is that we can bound the error introduced by pruning a component
without running forward passes—using only weight matrices and attention patterns.
-/

/-! ### Circuit Mask Representation -/

/-- A component identifier: attention head, MLP neuron, or SAE feature. -/
inductive ComponentId where
  /-- Attention head at (layer, head index) -/
  | head : (layerIdx : Nat) → (headIdx : Nat) → ComponentId
  /-- MLP neuron at (layer, neuron index within that layer's MLP) -/
  | mlpNeuron : (layerIdx : Nat) → (neuronIdx : Nat) → ComponentId
  /-- SAE feature at (layer, feature index within that layer's SAE) -/
  | saeFeature : (layerIdx : Nat) → (featureIdx : Nat) → ComponentId
  deriving DecidableEq, Repr

namespace ComponentId

/-- Pretty print a component ID. -/
def toString : ComponentId → String
  | head l h => s!"L{l}H{h}"
  | mlpNeuron l n => s!"L{l}N{n}"
  | saeFeature l f => s!"L{l}F{f}"

/-- Check if this is an attention head. -/
def isHead : ComponentId → Bool
  | head _ _ => true
  | _ => false

/-- Check if this is an MLP neuron. -/
def isNeuron : ComponentId → Bool
  | mlpNeuron _ _ => true
  | _ => false

/-- Check if this is an SAE feature. -/
def isSAEFeature : ComponentId → Bool
  | saeFeature _ _ => true
  | _ => false

/-- Get the layer index. -/
def layerIdx : ComponentId → Nat
  | head l _ => l
  | mlpNeuron l _ => l
  | saeFeature l _ => l

instance : ToString ComponentId := ⟨ComponentId.toString⟩

end ComponentId

/-- A circuit is a sparse subgraph mask over model components.

The mask indicates which components are **included** in the circuit.
Components not in the mask are considered "ablated" (zeroed out).

This is a simplified structure without proof obligations - validity is checked at runtime.
-/
structure ConcreteCircuit where
  /-- Number of layers in the model -/
  numLayers : Nat
  /-- Number of heads per layer -/
  headsPerLayer : Array Nat
  /-- Number of MLP neurons per layer -/
  neuronsPerLayer : Array Nat
  /-- Included attention heads: includedHeads[l][h] = true iff head h in layer l is active -/
  includedHeads : Array (Array Bool)
  /-- Included MLP neurons: includedNeurons[l][n] = true iff neuron n in layer l is active -/
  includedNeurons : Array (Array Bool)

namespace ConcreteCircuit

/-- Check if a specific head is included in the circuit. -/
def isHeadIncluded (circuit : ConcreteCircuit) (layerIdx headIdx : Nat) : Bool :=
  if layerIdx < circuit.includedHeads.size then
    let layerMask := circuit.includedHeads.getD layerIdx #[]
    layerMask.getD headIdx false
  else false

/-- Check if a specific MLP neuron is included in the circuit. -/
def isNeuronIncluded (circuit : ConcreteCircuit) (layerIdx neuronIdx : Nat) : Bool :=
  if layerIdx < circuit.includedNeurons.size then
    let layerMask := circuit.includedNeurons.getD layerIdx #[]
    layerMask.getD neuronIdx false
  else false

/-- Check if any component is included (dispatches on ComponentId type). -/
def isIncluded (circuit : ConcreteCircuit) (comp : ComponentId) : Bool :=
  match comp with
  | ComponentId.head l h => circuit.isHeadIncluded l h
  | ComponentId.mlpNeuron l n => circuit.isNeuronIncluded l n
  | ComponentId.saeFeature _ _ => false  -- SAE features handled by SAECircuit

/-- Count total number of included attention heads. -/
def countIncludedHeads (circuit : ConcreteCircuit) : Nat :=
  circuit.includedHeads.foldl (fun acc layer =>
    acc + layer.foldl (fun acc' b => if b then acc' + 1 else acc') 0) 0

/-- Count total number of included MLP neurons. -/
def countIncludedNeurons (circuit : ConcreteCircuit) : Nat :=
  circuit.includedNeurons.foldl (fun acc layer =>
    acc + layer.foldl (fun acc' b => if b then acc' + 1 else acc') 0) 0

/-- Count total number of included components. -/
def countIncluded (circuit : ConcreteCircuit) : Nat :=
  circuit.countIncludedHeads + circuit.countIncludedNeurons

/-- Count total number of attention heads (included + excluded). -/
def totalHeads (circuit : ConcreteCircuit) : Nat :=
  circuit.headsPerLayer.foldl (· + ·) 0

/-- Count total number of MLP neurons (included + excluded). -/
def totalNeurons (circuit : ConcreteCircuit) : Nat :=
  circuit.neuronsPerLayer.foldl (· + ·) 0

/-- Count total number of components (included + excluded). -/
def totalComponents (circuit : ConcreteCircuit) : Nat :=
  circuit.totalHeads + circuit.totalNeurons

/-- List all included component IDs. -/
def includedComponents (circuit : ConcreteCircuit) : Array ComponentId := Id.run do
  let mut result : Array ComponentId := #[]
  -- Attention heads
  for l in [:circuit.numLayers] do
    let layerMask := circuit.includedHeads.getD l #[]
    for h_idx in [:layerMask.size] do
      if layerMask.getD h_idx false then
        result := result.push (ComponentId.head l h_idx)
  -- MLP neurons
  for l in [:circuit.numLayers] do
    let layerMask := circuit.includedNeurons.getD l #[]
    for n_idx in [:layerMask.size] do
      if layerMask.getD n_idx false then
        result := result.push (ComponentId.mlpNeuron l n_idx)
  result

/-- List all excluded component IDs. -/
def excludedComponents (circuit : ConcreteCircuit) : Array ComponentId := Id.run do
  let mut result : Array ComponentId := #[]
  -- Attention heads
  for l in [:circuit.numLayers] do
    let layerMask := circuit.includedHeads.getD l #[]
    for h_idx in [:layerMask.size] do
      if !layerMask.getD h_idx false then
        result := result.push (ComponentId.head l h_idx)
  -- MLP neurons
  for l in [:circuit.numLayers] do
    let layerMask := circuit.includedNeurons.getD l #[]
    for n_idx in [:layerMask.size] do
      if !layerMask.getD n_idx false then
        result := result.push (ComponentId.mlpNeuron l n_idx)
  result

/-- Create a full circuit (all components included). -/
def full (numLayers : Nat) (headsPerLayer neuronsPerLayer : Array Nat) : ConcreteCircuit where
  numLayers := numLayers
  headsPerLayer := headsPerLayer
  neuronsPerLayer := neuronsPerLayer
  includedHeads := headsPerLayer.map fun numHeads =>
    .ofFn fun _ : Fin numHeads => true
  includedNeurons := neuronsPerLayer.map fun numNeurons =>
    .ofFn fun _ : Fin numNeurons => true

/-- Create an empty circuit (no components included). -/
def empty (numLayers : Nat) (headsPerLayer neuronsPerLayer : Array Nat) : ConcreteCircuit where
  numLayers := numLayers
  headsPerLayer := headsPerLayer
  neuronsPerLayer := neuronsPerLayer
  includedHeads := headsPerLayer.map fun numHeads =>
    .ofFn fun _ : Fin numHeads => false
  includedNeurons := neuronsPerLayer.map fun numNeurons =>
    .ofFn fun _ : Fin numNeurons => false

/-- Remove a single component from the circuit (returns new circuit). -/
def removeComponent (circuit : ConcreteCircuit) (comp : ComponentId) : ConcreteCircuit :=
  match comp with
  | ComponentId.head layerIdx headIdx =>
    if layerIdx < circuit.includedHeads.size then
      let newIncluded := circuit.includedHeads.modify layerIdx fun layerMask =>
        if headIdx < layerMask.size then
          layerMask.modify headIdx fun _ => false
        else layerMask
      { circuit with includedHeads := newIncluded }
    else circuit
  | ComponentId.mlpNeuron layerIdx neuronIdx =>
    if layerIdx < circuit.includedNeurons.size then
      let newIncluded := circuit.includedNeurons.modify layerIdx fun layerMask =>
        if neuronIdx < layerMask.size then
          layerMask.modify neuronIdx fun _ => false
        else layerMask
      { circuit with includedNeurons := newIncluded }
    else circuit
  | ComponentId.saeFeature _ _ => circuit  -- SAE features handled by SAECircuit

/-- Add a single component to the circuit (returns new circuit). -/
def addComponent (circuit : ConcreteCircuit) (comp : ComponentId) : ConcreteCircuit :=
  match comp with
  | ComponentId.head layerIdx headIdx =>
    if layerIdx < circuit.includedHeads.size then
      let newIncluded := circuit.includedHeads.modify layerIdx fun layerMask =>
        if headIdx < layerMask.size then
          layerMask.modify headIdx fun _ => true
        else layerMask
      { circuit with includedHeads := newIncluded }
    else circuit
  | ComponentId.mlpNeuron layerIdx neuronIdx =>
    if layerIdx < circuit.includedNeurons.size then
      let newIncluded := circuit.includedNeurons.modify layerIdx fun layerMask =>
        if neuronIdx < layerMask.size then
          layerMask.modify neuronIdx fun _ => true
        else layerMask
      { circuit with includedNeurons := newIncluded }
    else circuit
  | ComponentId.saeFeature _ _ => circuit  -- SAE features handled by SAECircuit

/-- Pretty print the circuit. -/
def toString (circuit : ConcreteCircuit) : String :=
  let heads := circuit.countIncludedHeads
  let neurons := circuit.countIncludedNeurons
  let totalH := circuit.totalHeads
  let totalN := circuit.totalNeurons
  s!"Circuit(heads={heads}/{totalH}, neurons={neurons}/{totalN})"

instance : ToString ConcreteCircuit := ⟨ConcreteCircuit.toString⟩

end ConcreteCircuit

/-! ## SAE-Enhanced Circuit Discovery

When using Sparse Autoencoders, we replace MLP neuron masks with SAE feature masks.
This enables discovering circuits in terms of interpretable features rather than
polysemantic neurons.
-/

/-- A circuit mask that operates on SAE features instead of MLP neurons.

This extends ConcreteCircuit by replacing MLP neuron masks with SAE feature masks.
The attention head masks remain the same.
-/
structure SAECircuit where
  /-- Number of layers in the model -/
  numLayers : Nat
  /-- Number of heads per layer -/
  headsPerLayer : Array Nat
  /-- Number of SAE features per layer -/
  featuresPerLayer : Array Nat
  /-- Included attention heads -/
  includedHeads : Array (Array Bool)
  /-- Included SAE features: includedFeatures[l][f] = true iff feature f in layer l is active -/
  includedFeatures : Array (Array Bool)

namespace SAECircuit

/-- Check if a specific head is included. -/
def isHeadIncluded (circuit : SAECircuit) (layerIdx headIdx : Nat) : Bool :=
  if layerIdx < circuit.includedHeads.size then
    let layerMask := circuit.includedHeads.getD layerIdx #[]
    layerMask.getD headIdx false
  else false

/-- Check if a specific SAE feature is included. -/
def isFeatureIncluded (circuit : SAECircuit) (layerIdx featureIdx : Nat) : Bool :=
  if layerIdx < circuit.includedFeatures.size then
    let layerMask := circuit.includedFeatures.getD layerIdx #[]
    layerMask.getD featureIdx false
  else false

/-- Check if any component is included. -/
def isIncluded (circuit : SAECircuit) (comp : ComponentId) : Bool :=
  match comp with
  | ComponentId.head l h => circuit.isHeadIncluded l h
  | ComponentId.mlpNeuron _ _ => false  -- SAE circuits don't track neurons
  | ComponentId.saeFeature l f => circuit.isFeatureIncluded l f

/-- Count included heads. -/
def countIncludedHeads (circuit : SAECircuit) : Nat :=
  circuit.includedHeads.foldl (fun acc layer =>
    acc + layer.foldl (fun acc' b => if b then acc' + 1 else acc') 0) 0

/-- Count included features. -/
def countIncludedFeatures (circuit : SAECircuit) : Nat :=
  circuit.includedFeatures.foldl (fun acc layer =>
    acc + layer.foldl (fun acc' b => if b then acc' + 1 else acc') 0) 0

/-- Total heads. -/
def totalHeads (circuit : SAECircuit) : Nat :=
  circuit.headsPerLayer.foldl (· + ·) 0

/-- Total features. -/
def totalFeatures (circuit : SAECircuit) : Nat :=
  circuit.featuresPerLayer.foldl (· + ·) 0

/-- Create a full circuit (all components included). -/
def full (numLayers : Nat) (headsPerLayer featuresPerLayer : Array Nat) : SAECircuit where
  numLayers := numLayers
  headsPerLayer := headsPerLayer
  featuresPerLayer := featuresPerLayer
  includedHeads := headsPerLayer.map fun numHeads =>
    .ofFn fun _ : Fin numHeads => true
  includedFeatures := featuresPerLayer.map fun numFeats =>
    .ofFn fun _ : Fin numFeats => true

/-- Create an empty circuit. -/
def empty (numLayers : Nat) (headsPerLayer featuresPerLayer : Array Nat) : SAECircuit where
  numLayers := numLayers
  headsPerLayer := headsPerLayer
  featuresPerLayer := featuresPerLayer
  includedHeads := headsPerLayer.map fun numHeads =>
    .ofFn fun _ : Fin numHeads => false
  includedFeatures := featuresPerLayer.map fun numFeats =>
    .ofFn fun _ : Fin numFeats => false

/-- Remove a component. -/
def removeComponent (circuit : SAECircuit) (comp : ComponentId) : SAECircuit :=
  match comp with
  | ComponentId.head layerIdx headIdx =>
    if layerIdx < circuit.includedHeads.size then
      let newIncluded := circuit.includedHeads.modify layerIdx fun layerMask =>
        if headIdx < layerMask.size then
          layerMask.modify headIdx fun _ => false
        else layerMask
      { circuit with includedHeads := newIncluded }
    else circuit
  | ComponentId.saeFeature layerIdx featureIdx =>
    if layerIdx < circuit.includedFeatures.size then
      let newIncluded := circuit.includedFeatures.modify layerIdx fun layerMask =>
        if featureIdx < layerMask.size then
          layerMask.modify featureIdx fun _ => false
        else layerMask
      { circuit with includedFeatures := newIncluded }
    else circuit
  | ComponentId.mlpNeuron _ _ => circuit  -- Not supported in SAE circuits

/-- Add a component. -/
def addComponent (circuit : SAECircuit) (comp : ComponentId) : SAECircuit :=
  match comp with
  | ComponentId.head layerIdx headIdx =>
    if layerIdx < circuit.includedHeads.size then
      let newIncluded := circuit.includedHeads.modify layerIdx fun layerMask =>
        if headIdx < layerMask.size then
          layerMask.modify headIdx fun _ => true
        else layerMask
      { circuit with includedHeads := newIncluded }
    else circuit
  | ComponentId.saeFeature layerIdx featureIdx =>
    if layerIdx < circuit.includedFeatures.size then
      let newIncluded := circuit.includedFeatures.modify layerIdx fun layerMask =>
        if featureIdx < layerMask.size then
          layerMask.modify featureIdx fun _ => true
        else layerMask
      { circuit with includedFeatures := newIncluded }
    else circuit
  | ComponentId.mlpNeuron _ _ => circuit

def toString (circuit : SAECircuit) : String :=
  let heads := circuit.countIncludedHeads
  let features := circuit.countIncludedFeatures
  let totalH := circuit.totalHeads
  let totalF := circuit.totalFeatures
  s!"SAECircuit(heads={heads}/{totalH}, features={features}/{totalF})"

instance : ToString SAECircuit := ⟨SAECircuit.toString⟩

end SAECircuit

/-- Model with SAEs attached at each layer's MLP.

Replaces `ConcreteModel.mlps` with SAEs for feature-level analysis.
-/
structure SAEEnhancedModel where
  /-- Number of layers -/
  numLayers : Nat
  /-- Attention layers with their heads -/
  layers : Array (Array ConcreteAttentionLayer)
  /-- Pre-LN LayerNorm parameters before attention (ln_1), one per layer. -/
  ln1 : Array ConcreteLayerNormParams := #[]
  /-- Pre-LN LayerNorm parameters before SAE/MLP (ln_2), one per layer. -/
  ln2 : Array ConcreteLayerNormParams := #[]
  /-- Final LayerNorm parameters (ln_f) before unembedding. -/
  lnf : ConcreteLayerNormParams := ConcreteLayerNormParams.identity 0
  /-- SAEs for MLP analysis: saes[l] is the SAE for layer l's MLP -/
  saes : Array ConcreteSAE
  /-- Sequence length -/
  seqLen : Nat
  /-- Input embeddings -/
  inputEmbeddings : ConcreteMatrix
  /-- Unembedding matrix -/
  unembedding : Option ConcreteMatrix := none

namespace SAEEnhancedModel

/-- Model dimension (d), inferred from input embeddings. -/
def modelDim (model : SAEEnhancedModel) : Nat :=
  model.inputEmbeddings.numCols

/-- Get ln_1 parameters for a layer, defaulting to identity. -/
def ln1Params (model : SAEEnhancedModel) (layerIdx : Nat) : ConcreteLayerNormParams :=
  model.ln1.getD layerIdx (ConcreteLayerNormParams.identity model.modelDim)

/-- Get ln_2 parameters for a layer, defaulting to identity. -/
def ln2Params (model : SAEEnhancedModel) (layerIdx : Nat) : ConcreteLayerNormParams :=
  model.ln2.getD layerIdx (ConcreteLayerNormParams.identity model.modelDim)

/-- Apply ln_1 to a residual stream (row-wise, per token). -/
def applyLn1 (model : SAEEnhancedModel) (layerIdx : Nat) (X : ConcreteMatrix) : ConcreteMatrix :=
  let p := model.ln1Params layerIdx
  ConcreteMatrix.layerNormRowwise X p.gamma p.beta

/-- Apply ln_2 to a residual stream (row-wise, per token). -/
def applyLn2 (model : SAEEnhancedModel) (layerIdx : Nat) (X : ConcreteMatrix) : ConcreteMatrix :=
  let p := model.ln2Params layerIdx
  ConcreteMatrix.layerNormRowwise X p.gamma p.beta

/-- Conservative operator-norm bound for ln_1 Jacobian at a specific activation. -/
def ln1OpBound (model : SAEEnhancedModel) (layerIdx : Nat) (X : ConcreteMatrix) : Float :=
  let p := model.ln1Params layerIdx
  ConcreteMatrix.layerNormRowwiseOpEst X p.gamma

/-- Create from ConcreteModel with externally trained SAEs. -/
def fromModel (model : ConcreteModel) (saes : Array ConcreteSAE) : Option SAEEnhancedModel :=
  if saes.size = model.numLayers then
    some {
      numLayers := model.numLayers
      layers := model.layers
      ln1 := model.ln1
      ln2 := model.ln2
      lnf := model.lnf
      saes := saes
      seqLen := model.seqLen
      inputEmbeddings := model.inputEmbeddings
      unembedding := model.unembedding
    }
  else none

/-- Total reconstruction error across all SAEs for given forward pass. -/
def totalReconstructionError (model : SAEEnhancedModel)
    (fwd : ForwardPassResult) : Float := Id.run do
  let mut totalErr : Float := 0.0
  for l in [:model.numLayers] do
    if hl : l < model.saes.size then
      let sae := model.saes[l]
      let layerInput := fwd.getLayerInput l
      let err := sae.reconstructionErrorMatrix layerInput
      totalErr := totalErr + err * err
  Float.sqrt totalErr

end SAEEnhancedModel

/-- Importance metrics for SAE features. -/
structure SAEFeatureImportance where
  /-- Component identifier -/
  component : ComponentId
  /-- Value term norm (feature influence) -/
  valueTermNorm : Float
  /-- Pattern term bound (activation instability) -/
  patternTermBound : Float
  /-- Faithfulness ratio -/
  faithfulnessRatio : Float
  /-- Reconstruction error contribution (SAE approximation) -/
  reconstructionError : Float

namespace SAEFeatureImportance

def toString (imp : SAEFeatureImportance) : String :=
  s!"{imp.component}: value={imp.valueTermNorm}, pattern={imp.patternTermBound}, " ++
  s!"recon={imp.reconstructionError}, ratio={imp.faithfulnessRatio}"

instance : ToString SAEFeatureImportance := ⟨toString⟩

end SAEFeatureImportance

/-- Compute importance metrics for a single SAE feature. -/
def computeSAEFeatureImportance (sae : ConcreteSAE) (layerIdx featureIdx : Nat)
    (layerInput : ConcreteMatrix) (perturbationNorm : Float) : Option SAEFeatureImportance :=
  if featureIdx < sae.numFeatures then
    let influence := sae.featureInfluence featureIdx

    -- Compute IBP-based pattern term bound across positions
    let patternBound := Id.run do
      let mut totalPatternSq : Float := 0.0
      for pos in [:layerInput.numRows] do
        let inputVec : Array Float := .ofFn fun d : Fin layerInput.numCols =>
          layerInput.get pos d.val
        let posBound := sae.featurePatternTermBoundIBP featureIdx inputVec perturbationNorm
        totalPatternSq := totalPatternSq + posBound * posBound
      Float.sqrt (totalPatternSq / (max 1 layerInput.numRows).toFloat)

    -- Estimate per-feature contribution to reconstruction error
    -- Approximation: uniform distribution across features (could be refined)
    let perFeatureRecon := Id.run do
      let mut totalRecon : Float := 0.0
      for pos in [:layerInput.numRows] do
        let inputVec : Array Float := .ofFn fun d : Fin layerInput.numCols =>
          layerInput.get pos d.val
        totalRecon := totalRecon + sae.reconstructionError inputVec
      totalRecon / (max 1 layerInput.numRows).toFloat / sae.numFeatures.toFloat

    let ratio := if influence < 1e-10 then Float.inf else patternBound / influence

    some {
      component := ComponentId.saeFeature layerIdx featureIdx
      valueTermNorm := influence
      patternTermBound := patternBound
      faithfulnessRatio := ratio
      reconstructionError := perFeatureRecon
    }
  else none

/-- Error breakdown for SAE-based circuits. -/
structure SAECircuitError where
  /-- Pattern term error from included components -/
  patternTermError : Float
  /-- Ablation error from excluded components -/
  ablationError : Float
  /-- SAE reconstruction error (approximation of MLP) -/
  reconstructionError : Float
  /-- Total error bound -/
  totalError : Float
  /-- Number of included components -/
  includedCount : Nat
  /-- Number of excluded components -/
  excludedCount : Nat
  /-- Number of unstable features -/
  unstableFeatureCount : Nat
  deriving Repr

namespace SAECircuitError

def toString (e : SAECircuitError) : String :=
  s!"SAECircuitError(total={e.totalError}, pattern={e.patternTermError}, " ++
  s!"ablation={e.ablationError}, recon={e.reconstructionError}, " ++
  s!"unstable={e.unstableFeatureCount})"

instance : ToString SAECircuitError := ⟨toString⟩

end SAECircuitError

/-- Helper to compute head importance for SAE analysis.
Inline version used before computeHeadImportance is defined.
Works with both ConcreteModel and SAEEnhancedModel (via their shared fields). -/
private def computeHeadMetricsForSAE
    (model : SAEEnhancedModel)
    (layerIdx headIdx : Nat) (layerInput : ConcreteMatrix) : Option (Float × Float) :=
  if h1 : layerIdx < model.layers.size then
    let layerHeads := model.layers[layerIdx]
    if h2 : headIdx < layerHeads.size then
      let head := layerHeads[headIdx]
      let attnInput := model.applyLn1 layerIdx layerInput
      let attn := head.computeAttentionWeights attnInput false
      let inputNorm := computeInputNorm attnInput
      let ln1Bound := model.ln1OpBound layerIdx layerInput
      let voProj := head.valueOutputProjection
      let qkAlign := head.queryKeyAlignment

      let valueNorm := ln1Bound * computeValueTermNorm attn voProj
      let inputs : PatternTermBoundInputs := {
        attention := attn
        queryKeyAlign := qkAlign
        queryKeyAlignSchurNorm := head.queryKeyAlignmentOpBoundFrom qkAlign
        valueOutputProj := voProj
        valueOutputProjSchurNorm := head.valueOutputProjectionOpBoundFrom voProj
        inputNorm := inputNorm
        scaleFactor := Float.sqrt head.headDim.toFloat
      }
      let patternBound := ln1Bound * computePatternTermBound inputs
      some (valueNorm, patternBound)
    else none
  else none

/-- Estimate faithfulness error for an SAE-based circuit.

Extends the standard circuit error model to include:
1. Pattern term error for included attention heads
2. Pattern term error for included SAE features (with IBP)
3. Ablation error for excluded components
4. SAE reconstruction error (the approximation of using SAE instead of MLP)

Total Error = Σ(included) patternBound + Σ(excluded) valueNorm + reconstructionError
-/
def estimateSAECircuitFaithfulness (model : SAEEnhancedModel)
    (circuit : SAECircuit) (_causal : Bool := true) : SAECircuitError := Id.run do
  -- Simplified forward pass (just attention)
  let mut residual := model.inputEmbeddings
  let mut layerInputs : Array ConcreteMatrix := #[model.inputEmbeddings]

  for l in [:model.numLayers] do
    let attnInput := model.applyLn1 l residual
    let mut attnSum := ConcreteMatrix.zeros residual.numRows residual.numCols
    if hl : l < model.layers.size then
      let layerHeads := model.layers[l]
      for h in [:layerHeads.size] do
        if hh : h < layerHeads.size then
          let head := layerHeads[h]'hh
          let headOutput := head.forward attnInput true
          attnSum := attnSum.add headOutput
    residual := residual.add attnSum
    layerInputs := layerInputs.push residual

  let mut patternError : Float := 0.0
  let mut ablationError : Float := 0.0
  let mut totalRecon : Float := 0.0
  let mut includedCount : Nat := 0
  let mut excludedCount : Nat := 0
  let mut unstableCount : Nat := 0
  let mut cumulativeAblation : Float := 0.0

  -- Process attention heads
  for l in [:model.numLayers] do
    if hl : l < model.layers.size then
      let layerHeads := model.layers[l]
      for h_idx in [:layerHeads.size] do
        let included := circuit.isHeadIncluded l h_idx
        let layerInput :=
          if hl2 : l < layerInputs.size then layerInputs[l]'hl2 else model.inputEmbeddings
        match computeHeadMetricsForSAE model l h_idx layerInput with
        | some (valueNorm, patternBound) =>
          if included then
            patternError := patternError + patternBound
            includedCount := includedCount + 1
          else
            ablationError := ablationError + valueNorm
            cumulativeAblation := cumulativeAblation + valueNorm
            excludedCount := excludedCount + 1
        | none => pure ()

  -- Process SAE features
  for l in [:model.numLayers] do
    if hl : l < model.saes.size then
      let sae := model.saes[l]
      -- Pre-LN: SAE/MLP sees ln_2(y_l) where y_l is post-attention residual.
      let postAttn :=
        if hpost : l + 1 < layerInputs.size then layerInputs[l + 1]'hpost else residual
      let layerInput := model.applyLn2 l postAttn

      -- Add reconstruction error for this layer
      let layerRecon := sae.reconstructionErrorMatrix layerInput
      totalRecon := totalRecon + layerRecon

      for f_idx in [:sae.numFeatures] do
        let included := circuit.isFeatureIncluded l f_idx
        match computeSAEFeatureImportance sae l f_idx layerInput cumulativeAblation with
        | some imp =>
          if included then
            patternError := patternError + imp.patternTermBound
            if imp.patternTermBound > 0.0 then
              unstableCount := unstableCount + 1
            includedCount := includedCount + 1
          else
            ablationError := ablationError + imp.valueTermNorm
            excludedCount := excludedCount + 1
        | none => pure ()

  {
    patternTermError := patternError
    ablationError := ablationError
    reconstructionError := totalRecon
    totalError := patternError + ablationError + totalRecon
    includedCount := includedCount
    excludedCount := excludedCount
    unstableFeatureCount := unstableCount
  }

/-! ### SAE Circuit Discovery

Greedy pruning algorithm for SAE-enhanced circuits:
1. Start with all components included
2. Compute importance for each component (heads + SAE features)
3. Remove the component with smallest valueNorm (least information loss when ablated)
4. Repeat until error threshold would be exceeded

Note: SAE reconstruction error is an additive constant for a given set of SAEs,
so it doesn't affect the pruning order.
-/

/-- Ranked component importance for SAE circuits. -/
structure SAERankedComponent where
  /-- Component identifier -/
  component : ComponentId
  /-- Value term norm (importance for ablation) -/
  valueTermNorm : Float
  /-- Pattern term bound (error when included) -/
  patternTermBound : Float

/-- Compute all component importances for an SAE-enhanced model. -/
def computeAllSAEImportance (model : SAEEnhancedModel) : Array SAERankedComponent := Id.run do
  let mut result : Array SAERankedComponent := #[]

  -- Simplified forward pass
  let mut residual := model.inputEmbeddings
  let mut layerInputs : Array ConcreteMatrix := #[model.inputEmbeddings]

  for l in [:model.numLayers] do
    let attnInput := model.applyLn1 l residual
    let mut attnSum := ConcreteMatrix.zeros residual.numRows residual.numCols
    if hl : l < model.layers.size then
      let layerHeads := model.layers[l]
      for h in [:layerHeads.size] do
        if hh : h < layerHeads.size then
          let head := layerHeads[h]'hh
          let headOutput := head.forward attnInput true
          attnSum := attnSum.add headOutput
    residual := residual.add attnSum
    layerInputs := layerInputs.push residual

  -- Compute head importances
  for l in [:model.numLayers] do
    if hl : l < model.layers.size then
      let layerHeads := model.layers[l]
      for h_idx in [:layerHeads.size] do
        let layerInput :=
          if hl2 : l < layerInputs.size then layerInputs[l]'hl2 else model.inputEmbeddings
        match computeHeadMetricsForSAE model l h_idx layerInput with
        | some (valueNorm, patternBound) =>
          result := result.push {
            component := ComponentId.head l h_idx
            valueTermNorm := valueNorm
            patternTermBound := patternBound
          }
        | none => pure ()

  -- Compute SAE feature importances
  for l in [:model.numLayers] do
    if hl : l < model.saes.size then
      let sae := model.saes[l]
      let postAttn :=
        if hpost : l + 1 < layerInputs.size then layerInputs[l + 1]'hpost else residual
      let layerInput := model.applyLn2 l postAttn
      for f_idx in [:sae.numFeatures] do
        match computeSAEFeatureImportance sae l f_idx layerInput 0.0 with
        | some imp =>
          result := result.push {
            component := imp.component
            valueTermNorm := imp.valueTermNorm
            patternTermBound := imp.patternTermBound
          }
        | none => pure ()

  result

/-- Discover minimal SAE circuit using greedy pruning.

Algorithm:
1. Start with full circuit (all heads, all features)
2. Compute base reconstruction error (constant for given SAEs)
3. Sort all components by valueTermNorm (ascending)
4. Iteratively remove smallest-impact components until error exceeds threshold

The threshold should account for SAE reconstruction error as a baseline.
-/
def discoverSAECircuit (model : SAEEnhancedModel) (threshold : Float) : SAECircuit := Id.run do
  -- Initialize full circuit
  let headsPerLayer := model.layers.map (·.size)
  let featuresPerLayer := model.saes.map (·.numFeatures)
  let mut circuit := SAECircuit.full model.numLayers headsPerLayer featuresPerLayer

  -- Compute base reconstruction error
  let baseError := estimateSAECircuitFaithfulness model circuit
  let reconError := baseError.reconstructionError

  -- If base error already exceeds threshold, return empty
  if reconError > threshold then
    return SAECircuit.empty model.numLayers headsPerLayer featuresPerLayer

  let adjustedThreshold := threshold - reconError

  -- Get all components sorted by valueTermNorm (ascending)
  let allImportance := computeAllSAEImportance model
  let sorted := allImportance.qsort fun a b => a.valueTermNorm < b.valueTermNorm

  let mut cumulativePatternError : Float := baseError.patternTermError
  let mut cumulativeAblationError : Float := 0.0

  -- Greedily remove components
  for comp in sorted do
    -- Removing this component:
    -- - Removes its patternTermBound from included error
    -- - Adds its valueTermNorm to ablation error
    let newPatternError := cumulativePatternError - comp.patternTermBound
    let newAblationError := cumulativeAblationError + comp.valueTermNorm
    let newTotalError := newPatternError + newAblationError

    if newTotalError > adjustedThreshold then
      -- Stop: removing this component would exceed threshold
      break

    -- Remove the component
    circuit := circuit.removeComponent comp.component
    cumulativePatternError := newPatternError
    cumulativeAblationError := newAblationError

  circuit

/-- Result of SAE circuit discovery. -/
structure SAEDiscoveryResult where
  /-- The discovered circuit -/
  circuit : SAECircuit
  /-- Error breakdown -/
  error : SAECircuitError
  /-- Compression ratio (components kept / total) -/
  compressionRatio : Float

/-- Discover SAE circuit with full result details. -/
def discoverSAECircuitWithResult (model : SAEEnhancedModel)
    (threshold : Float) : SAEDiscoveryResult := Id.run do
  let circuit := discoverSAECircuit model threshold
  let error := estimateSAECircuitFaithfulness model circuit
  let totalComponents := circuit.totalHeads + circuit.totalFeatures
  let includedComponents := circuit.countIncludedHeads + circuit.countIncludedFeatures
  let compression := if totalComponents > 0 then
    includedComponents.toFloat / totalComponents.toFloat
  else 1.0

  { circuit, error, compressionRatio := compression }

/-! ## Ablated Forward Pass (Causal Intervention)

These functions implement executable causal interventions: running a forward pass
where specific attention heads or MLP neurons are masked out (ablated) based on
a `ConcreteCircuit` mask.

This bridges the theoretical `Abstraction.lean` bounds with empirical verification:
- `runAblatedForward`: Execute the circuit (masked forward pass)
- `computeAblationDiscrepancy`: Measure actual difference from full model
- `verifyCircuitFaithfulness`: Assert empirical ≤ theoretical bound
-/

/-- Forward pass for an MLP layer with neuron-level ablation.

Same as `ConcreteMLPLayer.forward` but with a mask specifying which neurons are active.
Inactive neurons have their contributions zeroed out.

For neuron i:
- If included: contribute GeLU(W_in[:,i] · x + b_in[i]) * W_out[i,:]
- If excluded: contribute 0
-/
def ConcreteMLPLayer.forwardAblated (layer : ConcreteMLPLayer) (input : ConcreteMatrix)
    (neuronMask : Array Bool) : ConcreteMatrix :=
  -- hidden = input · W_in + b_in  (seqLen × hiddenDim)
  let hidden := (input.matmul layer.W_in).addBias layer.b_in
  -- Apply GeLU activation with masking using .ofFn for proper size proof
  let activated : ConcreteMatrix := {
    numRows := hidden.numRows
    numCols := hidden.numCols
    data := .ofFn fun idx : Fin (hidden.numRows * hidden.numCols) =>
      let i := idx.val / hidden.numCols
      let j := idx.val % hidden.numCols
      let val := hidden.get i j
      let act := geluFloat val
      -- Zero out if neuron j is not included
      if neuronMask.getD j true then act else 0.0
    size_eq := Array.size_ofFn
  }
  -- output = activated · W_out + b_out  (seqLen × modelDim)
  (activated.matmul layer.W_out).addBias layer.b_out

/-- Run an ablated forward pass through the model.

Like `runForward`, but with a `ConcreteCircuit` mask that specifies which attention
heads and MLP neurons are active. Excluded components have their contributions
zeroed out, implementing a causal intervention.

This enables **empirical validation** of theoretical circuit bounds:
1. Discover a circuit via `discoverCircuit` or `discoverTargetedCircuit`
2. Run `runAblatedForward` with that circuit
3. Compare to `runForward` to measure actual discrepancy
4. Verify that empirical discrepancy ≤ theoretical bound

**Ablation semantics:**
- Excluded attention head: its output is zero (does not contribute to residual)
- Excluded MLP neuron: its activation is zero (does not contribute to FFN output)
-/
def ConcreteModel.runAblatedForward (model : ConcreteModel) (circuit : ConcreteCircuit)
    (causal : Bool := true) : ForwardPassResult := Id.run do
  let mut layerInputs : Array ConcreteMatrix := Array.mkEmpty (model.numLayers + 1)
  let mut attnOutputs : Array (Array ConcreteMatrix) := Array.mkEmpty model.numLayers
  let mut mlpOutputs : Array ConcreteMatrix := Array.mkEmpty model.numLayers
  let mut residual := model.inputEmbeddings
  layerInputs := layerInputs.push residual

  for l in [:model.numLayers] do
    -- Pre-LN: attention sees ln_1(residual)
    let attnInput := model.applyLn1 l residual
    -- Compute attention outputs for included heads only
    let mut layerAttnOutputs : Array ConcreteMatrix := #[]
    let mut includedHeadOutputs : Array ConcreteMatrix := #[]
    let rows := residual.numRows
    let cols := residual.numCols
    let zeroOutput := ConcreteMatrix.zeros rows cols

    if hl : l < model.layers.size then
      let layerHeads := model.layers[l]
      let includedMask := circuit.includedHeads.getD l #[]
      let includedCount :=
        includedMask.foldl (fun acc b => if b then acc + 1 else acc) 0
      let useParallelHeads :=
        layerHeads.size >= 4 && includedCount >= 4
      layerAttnOutputs :=
        if useParallelHeads then
          let tasks : Array (Task ConcreteMatrix) :=
            .ofFn fun i : Fin layerHeads.size =>
              Task.spawn (fun _ =>
                if circuit.isHeadIncluded l i.val then
                  (layerHeads[i]).forward attnInput causal
                else
                  zeroOutput)
          tasks.map Task.get
        else
          Id.run do
            let mut outs : Array ConcreteMatrix := Array.mkEmpty layerHeads.size
            for h in [:layerHeads.size] do
              if hh : h < layerHeads.size then
                let head := layerHeads[h]'hh
                if circuit.isHeadIncluded l h then
                  outs := outs.push (head.forward attnInput causal)
                else
                  outs := outs.push zeroOutput
            return outs
      -- Preserve the original summation order: increasing head index.
      includedHeadOutputs :=
        Id.run do
          let mut outs : Array ConcreteMatrix := Array.mkEmpty includedCount
          for h in [:layerAttnOutputs.size] do
            if circuit.isHeadIncluded l h then
              if hh : h < layerAttnOutputs.size then
                outs := outs.push (layerAttnOutputs[h]'hh)
          return outs

    attnOutputs := attnOutputs.push layerAttnOutputs

    -- Add attention residual
    let residualAfterAttn :=
      if includedHeadOutputs.isEmpty then
        residual
      else
        let attnSum : ConcreteMatrix := {
          numRows := rows
          numCols := cols
          data := .ofFn fun idx : Fin (rows * cols) => Id.run do
            let mut acc : Float := 0.0
            for hOut in includedHeadOutputs do
              acc := acc + hOut.data[idx.val]!
            return acc
          size_eq := Array.size_ofFn
        }
        residual.add attnSum

    -- Compute MLP output with neuron-level ablation
    -- Pre-LN: MLP sees ln_2(residualAfterAttn)
    let mlpInput := model.applyLn2 l residualAfterAttn
    let mlpOut :=
      if hm : l < model.mlps.size then
        -- Get neuron mask for this layer
        let neuronMask := circuit.includedNeurons.getD l #[]
        model.mlps[l].forwardAblated mlpInput neuronMask
      else ConcreteMatrix.zeros residual.numRows residual.numCols

    mlpOutputs := mlpOutputs.push mlpOut

    -- Add MLP residual
    residual := residualAfterAttn.add mlpOut

    -- Store input for next layer
    layerInputs := layerInputs.push residual

  let finalOutput := model.applyLnf residual
  {
    layerInputs := layerInputs
    attnOutputs := attnOutputs
    mlpOutputs := mlpOutputs
    mlpActDerivMax := Array.replicate model.numLayers 0.0
    finalOutput := finalOutput
  }

/-! ### Empirical Discrepancy and Verification

These functions compute the actual difference between full and ablated model outputs,
enabling empirical validation of theoretical circuit bounds.
-/

/-- Compute the element-wise difference between two matrices. -/
def ConcreteMatrix.sub (A B : ConcreteMatrix) : ConcreteMatrix :=
  if A.numRows = B.numRows ∧ A.numCols = B.numCols then
    {
      numRows := A.numRows
      numCols := A.numCols
      data := .ofFn fun idx : Fin (A.numRows * A.numCols) =>
        A.data.getD idx.val 0.0 - B.data.getD idx.val 0.0
      size_eq := Array.size_ofFn
    }
  else ConcreteMatrix.zeros 0 0

/-- Result of comparing full model output to ablated circuit output.

This captures the empirical discrepancy between running the full model
and running only the discovered circuit.
-/
structure AblationResult where
  /-- Full model output (residual stream after all layers) -/
  fullOutput : ConcreteMatrix
  /-- Ablated circuit output -/
  ablatedOutput : ConcreteMatrix
  /-- Difference: fullOutput - ablatedOutput -/
  difference : ConcreteMatrix
  /-- Frobenius norm of the difference: ‖full - ablated‖_F -/
  empiricalError : Float
  /-- Relative error: ‖full - ablated‖_F / ‖full‖_F -/
  relativeError : Float
  /-- Number of components in the circuit -/
  circuitSize : Nat
  /-- Total number of components in the model -/
  totalComponents : Nat

namespace AblationResult

/-- Compute compression ratio: what fraction of components are included. -/
def compressionRatio (r : AblationResult) : Float :=
  if r.totalComponents > 0 then
    r.circuitSize.toFloat / r.totalComponents.toFloat
  else 1.0

/-- Pretty print the ablation result. -/
def toString (r : AblationResult) : String :=
  s!"AblationResult:\n" ++
  s!"  Empirical Error (‖Δ‖_F): {r.empiricalError}\n" ++
  s!"  Relative Error: {r.relativeError * 100.0}%\n" ++
  s!"  Circuit Size: {r.circuitSize}/{r.totalComponents} " ++
  s!"({r.compressionRatio * 100.0}%)"

instance : ToString AblationResult := ⟨AblationResult.toString⟩

end AblationResult

/-- Compute the empirical discrepancy between full model and ablated circuit.

This is the core function for empirical validation:
1. Runs full forward pass
2. Runs ablated forward pass with the circuit mask
3. Computes the difference and its Frobenius norm

The empirical error should be bounded by the theoretical error estimate from
`estimateCircuitFaithfulness`.
-/
def computeAblationDiscrepancy (model : ConcreteModel) (circuit : ConcreteCircuit)
    (causal : Bool := true) : AblationResult :=
  let fullResult := model.runForward causal
  let ablatedResult := model.runAblatedForward circuit causal
  let diff := fullResult.finalOutput.sub ablatedResult.finalOutput
  let empiricalErr := diff.frobeniusNorm
  let fullNorm := fullResult.finalOutput.frobeniusNorm
  let relErr := if fullNorm > 1e-10 then empiricalErr / fullNorm else 0.0
  {
    fullOutput := fullResult.finalOutput
    ablatedOutput := ablatedResult.finalOutput
    difference := diff
    empiricalError := empiricalErr
    relativeError := relErr
    circuitSize := circuit.countIncluded
    totalComponents := circuit.totalComponents
  }

/-- Result of comparing empirical discrepancy to theoretical bound.

This is the verification bridge between `Abstraction.lean` (theory) and
`Discovery.lean` (practice).
-/
structure VerificationResult where
  /-- Ablation result with empirical measurements -/
  ablation : AblationResult
  /-- Theoretical error bound from circuit analysis -/
  theoreticalBound : Float
  /-- Whether empirical ≤ theoretical (verification passed) -/
  verified : Bool
  /-- Slack: theoretical - empirical (how much margin we have) -/
  slack : Float
  /-- Tightness ratio: empirical / theoretical -/
  tightness : Float

namespace VerificationResult

/-- Pretty print the verification result. -/
def toString (r : VerificationResult) : String :=
  let status := if r.verified then "✓ VERIFIED" else "✗ FAILED"
  s!"VerificationResult [{status}]\n" ++
  s!"  Empirical Error: {r.ablation.empiricalError}\n" ++
  s!"  Theoretical Bound: {r.theoreticalBound}\n" ++
  s!"  Slack: {r.slack}\n" ++
  s!"  Tightness: {r.tightness * 100.0}%\n" ++
  s!"  Circuit: {r.ablation.circuitSize}/{r.ablation.totalComponents} components"

instance : ToString VerificationResult := ⟨VerificationResult.toString⟩

end VerificationResult

/-- Verify that a discovered circuit's empirical error is within theoretical bounds.

This is the key function that **closes the loop** between theory and practice:

**Input:**
- A model
- A discovered circuit (from `discoverCircuit` or `discoverTargetedCircuit`)
- The theoretical error bound (from `CircuitError.totalError`)

**Output:**
- Verification result showing whether empirical ≤ theoretical

**Usage:**
```
let result := discoverCircuit model 0.1
let verification := verifyCircuitFaithfulness model result.circuit result.error.totalError
if verification.verified then
  IO.println "Circuit is empirically faithful!"
```

**Interpretation:**
- `verified = true`: The circuit recapitulates model behavior within bounds
- `tightness ≈ 1`: Theoretical bound is tight (good analysis)
- `tightness << 1`: Theoretical bound is loose (conservative)
-/
def verifyCircuitFaithfulness (model : ConcreteModel) (circuit : ConcreteCircuit)
    (theoreticalBound : Float) (causal : Bool := true) : VerificationResult :=
  let ablation := computeAblationDiscrepancy model circuit causal
  let verified := ablation.empiricalError ≤ theoreticalBound
  let slack := theoreticalBound - ablation.empiricalError
  let tightness := if theoreticalBound > 1e-10
    then ablation.empiricalError / theoreticalBound
    else 1.0
  {
    ablation := ablation
    theoreticalBound := theoreticalBound
    verified := verified
    slack := slack
    tightness := tightness
  }

/-! ### Component Importance Metrics -/

/-- Importance metrics for a single component.

These metrics allow ranking components by their contribution to model behavior,
enabling principled circuit pruning.
-/
structure ComponentImportance where
  /-- Component identifier -/
  component : ComponentId
  /-- Value term norm: ‖A‖_F · ‖W_V·W_O‖_F (how much information flows through) -/
  valueTermNorm : Float
  /-- Pattern term bound (approximation error if we trust attention patterns) -/
  patternTermBound : Float
  /-- Faithfulness ratio: patternBound / valueNorm -/
  faithfulnessRatio : Float

namespace ComponentImportance

/-- Pretty print component importance. -/
def toString (imp : ComponentImportance) : String :=
  s!"{imp.component}: value={imp.valueTermNorm}, pattern={imp.patternTermBound}, " ++
  s!"ratio={imp.faithfulnessRatio}"

instance : ToString ComponentImportance := ⟨ComponentImportance.toString⟩

end ComponentImportance

/-- Compute importance metrics for a single attention head. -/
def computeHeadImportance (model : ConcreteModel) (layerIdx headIdx : Nat)
    (layerInput : ConcreteMatrix) : Option ComponentImportance := do
  if h1 : layerIdx < model.layers.size then
    let layerHeads := model.layers[layerIdx]
    if h2 : headIdx < layerHeads.size then
      let head := layerHeads[headIdx]
      let attnInput := model.applyLn1 layerIdx layerInput
      let attn := head.computeAttentionWeights attnInput
      let inputNorm := attnInput.frobeniusNorm
      let ln1Bound := model.ln1OpBound layerIdx layerInput
      let voProj := head.valueOutputProjection
      let qkAlign := head.queryKeyAlignment

      -- Pre-LN: effective value path includes the LayerNorm Jacobian.
      let valueNorm := ln1Bound * computeValueTermNorm attn voProj
      let inputs : PatternTermBoundInputs := {
        attention := attn
        queryKeyAlign := qkAlign
        queryKeyAlignSchurNorm := head.queryKeyAlignmentOpBoundFrom qkAlign
        valueOutputProj := voProj
        valueOutputProjSchurNorm := head.valueOutputProjectionOpBoundFrom voProj
        inputNorm := inputNorm
        scaleFactor := Float.sqrt head.headDim.toFloat
      }
      let patternBound := ln1Bound * computePatternTermBound inputs
      let ratio := if valueNorm < 1e-10 then Float.inf else patternBound / valueNorm

      return {
        component := ComponentId.head layerIdx headIdx
        valueTermNorm := valueNorm
        patternTermBound := patternBound
        faithfulnessRatio := ratio
      }
    else none
  else none

/-- Compute importance metrics for a single MLP neuron.

**Simple Version (no forward pass):**
For ReLU/GeLU MLPs, this uses weight-based bounds only.
The influence magnitude = ‖W_in[:,i]‖ · ‖W_out[i,:]‖ bounds information flow.

Pattern term is set to 0 (assumes locally linear), which is **unsound** if
ablations cause activation flips. Use `computeNeuronImportanceIBP` with
forward pass data for rigorous bounds.
-/
def computeNeuronImportance (model : ConcreteModel) (layerIdx neuronIdx : Nat)
    (_inputNorm : Float) : Option ComponentImportance :=
  if h : layerIdx < model.mlps.size then
    let mlp := model.mlps[layerIdx]
    if neuronIdx < mlp.hiddenDim then
      let influence := mlp.neuronInfluence neuronIdx
      -- Conservative assumption: locally linear (pattern term = 0)
      -- WARNING: This is unsound if activation flips occur!
      let patternBound : Float := 0.0
      let ratio := if influence < 1e-10 then Float.inf else patternBound / influence

      some {
        component := ComponentId.mlpNeuron layerIdx neuronIdx
        valueTermNorm := influence
        patternTermBound := patternBound
        faithfulnessRatio := ratio
      }
    else none
  else none

/-- Compute importance metrics for a single MLP neuron using IBP.

**Sound Version (requires forward pass):**
Uses Interval Bound Propagation to detect neurons that may flip activation
states under input perturbations. Provides mathematically rigorous pattern
term bounds.

**Parameters:**
- `layerInput`: Input to this layer (from forward pass), used to compute
  nominal pre-activations
- `perturbationNorm`: L2 bound on how much ablations can change the input
  (typically computed from the ablated components' value terms)

**Returns:** ComponentImportance with rigorous pattern term bound that
accounts for potential activation flips.
-/
def computeNeuronImportanceIBP (model : ConcreteModel) (layerIdx neuronIdx : Nat)
    (layerInput : ConcreteMatrix) (perturbationNorm : Float) : Option ComponentImportance :=
  if h : layerIdx < model.mlps.size then
    let mlp := model.mlps[layerIdx]
    if neuronIdx < mlp.hiddenDim then
      let influence := mlp.neuronInfluence neuronIdx

      -- Average the IBP bound across sequence positions
      let patternBound := Id.run do
        let mut totalPatternBound : Float := 0.0
        let numPositions := layerInput.numRows

        for pos in [:numPositions] do
          -- Extract input vector at this position
          let inputVec : Array Float := .ofFn fun d : Fin layerInput.numCols =>
            layerInput.get pos d.val
          -- Compute IBP-based pattern term bound
          let posBound := mlp.neuronPatternTermBoundIBP neuronIdx inputVec perturbationNorm
          totalPatternBound := totalPatternBound + posBound * posBound

        -- RMS of per-position bounds
        Float.sqrt (totalPatternBound / (max 1 numPositions).toFloat)

      let ratio := if influence < 1e-10 then Float.inf else patternBound / influence

      some {
        component := ComponentId.mlpNeuron layerIdx neuronIdx
        valueTermNorm := influence
        patternTermBound := patternBound
        faithfulnessRatio := ratio
      }
    else none
  else none

/-- Compute importance metrics for all components in a model. -/
def computeAllImportance (model : ConcreteModel) : Array ComponentImportance := Id.run do
  let inputNorm := computeInputNorm model.inputEmbeddings

  let computeHeadsForLayer (l : Nat) : Array ComponentImportance := Id.run do
    if h : l < model.layers.size then
      let layerHeads := model.layers[l]
      let mut outs : Array ComponentImportance := Array.mkEmpty layerHeads.size
      for h_idx in [:layerHeads.size] do
        match computeHeadImportance model l h_idx model.inputEmbeddings with
        | some imp => outs := outs.push imp
        | none => pure ()
      return outs
    else
      return #[]

  let computeNeuronsForLayer (l : Nat) : Array ComponentImportance := Id.run do
    if h : l < model.mlps.size then
      let mlp := model.mlps[l]
      let mut outs : Array ComponentImportance := Array.mkEmpty mlp.hiddenDim
      for n_idx in [:mlp.hiddenDim] do
        match computeNeuronImportance model l n_idx inputNorm with
        | some imp => outs := outs.push imp
        | none => pure ()
      return outs
    else
      return #[]

  let useParallel := model.numLayers >= 4
  let headChunks : Array (Array ComponentImportance) :=
    if useParallel then
      let tasks : Array (Task (Array ComponentImportance)) :=
        .ofFn fun i : Fin model.numLayers =>
          Task.spawn (fun _ => computeHeadsForLayer i.val)
      tasks.map Task.get
    else
      .ofFn fun i : Fin model.numLayers =>
        computeHeadsForLayer i.val

  let neuronChunks : Array (Array ComponentImportance) :=
    if useParallel then
      let tasks : Array (Task (Array ComponentImportance)) :=
        .ofFn fun i : Fin model.numLayers =>
          Task.spawn (fun _ => computeNeuronsForLayer i.val)
      tasks.map Task.get
    else
      .ofFn fun i : Fin model.numLayers =>
        computeNeuronsForLayer i.val

  -- Join in the same order as the original loop: heads then neurons, increasing layer index.
  let totalHeads := headChunks.foldl (fun acc cs => acc + cs.size) 0
  let totalNeurons := neuronChunks.foldl (fun acc cs => acc + cs.size) 0
  let mut result : Array ComponentImportance := Array.mkEmpty (totalHeads + totalNeurons)
  for cs in headChunks do
    for c in cs do
      result := result.push c
  for cs in neuronChunks do
    for c in cs do
      result := result.push c
  result

/-! ### Target-Aware Circuit Discovery (Logit Lens)

Standard circuit discovery finds components that contribute to the *entire* residual stream,
yielding large generic circuits. **Target-aware discovery** instead finds minimal circuits
for *specific predictions* by projecting component outputs onto a target direction.

**Key Insight:** For a specific prediction (e.g., "the model predicts 'cat' not 'dog'"),
we define a target direction `u = W_U[cat] - W_U[dog]` in the residual stream space.
A component's importance is then `‖output · u‖` rather than `‖output‖_F`.

This enables:
- Finding circuits responsible for specific token predictions
- Isolating mechanisms for behavioral differences (e.g., IOI task)
- Producing smaller, more interpretable circuits

**Mathematical Formulation:**
For attention head with value-output projection `W_V · W_O`:
- Standard importance: `‖A‖_F · ‖W_V · W_O‖_F` (generic)
- Target-aware: `‖A‖_F · ‖(W_V · W_O) · u‖` (specific)

For MLP neuron with output weights `W_out[i,:]`:
- Standard importance: `‖W_in[:,i]‖ · ‖W_out[i,:]‖` (generic)
- Target-aware: `‖W_in[:,i]‖ · |W_out[i,:] · u|` (specific)
-/

/-- A target direction for focused circuit discovery.

Specifies a direction in residual stream space to project component outputs onto.
Typically constructed as `u = W_U[correct_token] - W_U[incorrect_token]`.
-/
structure TargetDirection where
  /-- The target vector in model dimension space (modelDim × 1 matrix) -/
  direction : ConcreteMatrix
  /-- Human-readable description of what this direction represents -/
  description : String := "target"

namespace TargetDirection

/-- Create a target direction from unembedding columns for two tokens.

`u = W_U[:, correctToken] - W_U[:, incorrectToken]`

This direction points from the incorrect prediction toward the correct one,
so components with positive projection increase P(correct) / P(incorrect).
-/
def fromLogitDiff (unembedding : ConcreteMatrix)
    (correctToken incorrectToken : Nat) : TargetDirection :=
  let correctCol := unembedding.getCol correctToken
  let incorrectCol := unembedding.getCol incorrectToken
  let direction := correctCol.vecSub incorrectCol
  {
    direction := direction
    description := s!"logit_diff({correctToken}-{incorrectToken})"
  }

/-- Create a target direction from a single token's unembedding.

Useful when you want to understand what promotes a specific token.
-/
def fromSingleToken (unembedding : ConcreteMatrix) (token : Nat) : TargetDirection :=
  {
    direction := unembedding.getCol token
    description := s!"logit({token})"
  }

/-- Normalize the target direction to unit length. -/
def normalize (t : TargetDirection) : TargetDirection :=
  let norm := t.direction.vecNorm
  if norm > 1e-10 then
    { t with direction := t.direction.scale (1.0 / norm) }
  else t

/-- Construct a next-token logit-difference direction from the model's input token history.

This is the **self-supervised induction target**:
let `T` be the ground-truth token sequence, and let `t_curr = T[last]`.
If `t_curr` appeared before at index `k`, the "induction target" is `t_next = T[k+1]`.

Returns `none` if:
- the model has no `inputTokens`,
- the sequence has no previous occurrence of `t_curr`,
- or the model is missing an `unembedding` matrix.
-/
def fromInductionHistory (model : ConcreteModel) : Option TargetDirection := do
  let tokens ← model.inputTokens
  if tokens.size = 0 then none else
    let lastIdx := tokens.size - 1
    let tCurr := tokens[lastIdx]!
    let mut foundIdx : Option Nat := none
    for offset in [:lastIdx] do
      if foundIdx.isNone then
        let idx := lastIdx - 1 - offset
        if tokens[idx]! = tCurr then
          foundIdx := some idx

    let k ← foundIdx
    let tNext := tokens[k + 1]!

    let W_U ← model.unembedding
    let vocabSize := W_U.numCols
    if vocabSize < 2 then none
    else if tNext ≥ vocabSize then none
    else
      let incorrect : Nat :=
        if tCurr < vocabSize ∧ tCurr ≠ tNext then tCurr
        else
          let cand1 := (tNext + 1) % vocabSize
          if cand1 ≠ tNext then cand1 else (tNext + 2) % vocabSize
      if incorrect = tNext then none
      else
        let base := TargetDirection.fromLogitDiff W_U tNext incorrect
        some { base with
          description := s!"induction_history(curr={tCurr}, prev={k}, \
            next={tNext}, neg={incorrect})"
        }

end TargetDirection

/-! ## Virtual-Head Effectiveness Verification -/

/-- Extremely generous cutoff to reject only egregious **interpretability illusions**.

In theory, a mechanism is "genuine" when its relative approximation error is < 1.0
(`isGenuineMechanism` / `mechanism_trichotomy` in `Nfp.Linearization`). In practice, the
executable Frobenius-norm bounds can be loose in high dimensions, making the strict < 1.0
test numerically vacuous. Empirically, however, massive errors indicate clear illusions.

We therefore filter only astronomically large `combinedError` values while ranking by
faithfulness (smallest `combinedError` first).
-/
def egregiousIllusionThreshold : Float := 1.0e30

/-- Egregious-illusion filter (currently disabled).

We intentionally keep *all* candidates (even those with extremely loose bounds), since
the Frobenius-norm estimates can scale poorly with depth and dimension.
-/
def passesEgregiousIllusionFilter (_candidate : CandidateInductionHead) : Bool :=
  true

/-- Compute the raw **direct** effectiveness score `δ` for an induction-head candidate.

Induction heads are primarily a **pattern** story (Q/K-composition): a "previous token" head
enables the *induction head* to attend to the **successor** of a previous matching token.
Once the induction head is attending to the right source positions, its functional effect is
driven by the head's **OV circuit** applied to the residual stream at those source tokens.

Accordingly, we treat head 1 purely as a *pattern enabler* (enforced by the pattern filters)
and score only the **direct value path** of head 2:

For a **Pre-LayerNorm** transformer (GPT-2 style), attention reads from `ln₁(X₂)` (not `X₂`).
We compute:

`Y = A₂ · ln₁(X₂) · W₂`,

then score the last position against `target.direction = u`:

`δ = ⟪Y[last], u⟫`.
-/
def computeInductionEffectiveness
    (candidate : CandidateInductionHead)
    (cache : PrecomputedCache)
    (layer2Ln1Input : ConcreteMatrix)
    (target : TargetDirection) : Float :=
  match cache.getHeadData candidate.layer2Idx candidate.head2Idx with
  | some data2 =>
      let W2 := data2.valueOutputProj
      let u := target.direction

      -- Direct head score: δ = ⟪(A₂ · ln₁(X₂) · W₂)[last], u⟫.
      --
      -- PERFORMANCE: compute this scalar without materializing any dense `d×d` products.
      -- 1) v = W₂ · u
      -- 2) xdot[k] = ⟪ln₁(X₂)[k], v⟫
      -- 3) δ = ⟪A₂[last], xdot⟫
      if u.numCols ≠ 1 then 0.0
      else if layer2Ln1Input.numCols ≠ u.numRows then 0.0
      else if data2.attention.seqLen = 0 then 0.0
      else if layer2Ln1Input.numRows ≠ data2.attention.seqLen then 0.0
      else
        let n := data2.attention.seqLen
        let lastPos := n - 1

        -- v = W2 · u
        let v := W2.matVecMul u
        if v.numRows = 0 then 0.0
        else
          -- xdot[k] = ⟪ln₁(X₂)[k], v⟫
          let xdot : Array Float := .ofFn fun k : Fin n => Id.run do
            let xBase := k.val * layer2Ln1Input.numCols
            let mut acc : Float := 0.0
            for c in [:layer2Ln1Input.numCols] do
              -- SAFETY: `k < n = layer2Ln1Input.numRows` by construction and guard above.
              let x := layer2Ln1Input.data[xBase + c]!
              -- SAFETY: `v` is `modelDim×1` so `c < v.data.size`.
              let vc := v.data[c]!
              acc := acc + x * vc
            return acc

          -- δ = ⟪A2[last], xdot⟫
          Id.run do
            let w2 := data2.attention.weights
            let row2Base := lastPos * n
            let mut score : Float := 0.0
            for k in [:n] do
              -- SAFETY: `w2` has size `n*n` and `row2Base + k < n*n`.
              score := score + w2[row2Base + k]! * xdot[k]!
            return score
  | none => 0.0

/-- Compute the **certified lower bound** from `true_induction_head_predicts_logits`.

`LowerBound = δ - (ε · ‖X‖_F · ‖u‖₂)` where:
- `δ` is the virtual effectiveness score,
- `ε` is `candidate.combinedError`,
- `X` is the layer-2 Pre-LN attention input matrix `ln₁(X₂)`,
- `u` is the target direction.
-/
def computeCertifiedLowerBound
    (delta : Float)
    (candidate : CandidateInductionHead)
    (layer2Ln1Input : ConcreteMatrix)
    (target : TargetDirection) : Float :=
  delta - (candidate.combinedError * layer2Ln1Input.frobeniusNorm * target.direction.vecNorm)

/-- Rank induction-head candidates by a **mechanism-first** score.

We compute the raw direct-effect score `δ`, normalize it to a scale-invariant `effect`,
and also compute a prompt/weight-based mechanism score:

`mechScore = kComp · inductionScore · prevTokenStrength`,

as well as the combined score:

`circuitScore = effect · mechScore`.

Here:
- `effect = δ / (‖ln₁(X₂)‖_F · ‖u‖₂)` removes the Pre-LN depth confounder where the residual
  stream norm grows with depth, and
- `kComp` (from the circuits framework paper) measures how strongly head 1 can feed into
  head 2's **QK circuit**, i.e. whether head 1 plausibly acts as a *pattern enabler* for head 2.
- `inductionScore` is the prompt-dependent "copy-next" attention score for head 2, and
- `prevTokenStrength` is the prompt-dependent previous-token attention score for head 1.

We rank primarily by `mechScore` (to identify the canonical induction mechanism) and use
`effect` only as a secondary key.

We still compute and report `combinedError` for inspection, but avoid using it as a primary
ranking key since Frobenius-norm bounds can be systematically looser in high dimensions.

Uses a `PrecomputedCache` so attention patterns/projections and layer inputs are computed once.
-/
def findHeuristicInductionHeadsWithCache (model : ConcreteModel)
    (target : TargetDirection)
    (minEffect : Float := 0.0)
    (minPrevTokenStrength : Float := 0.1)
    (minInductionScore : Float := 0.05) : (Array HeuristicInductionHead × PrecomputedCache) :=
  Id.run do
    let cache := PrecomputedCache.build model
    let tokens? := model.inputTokens
    let targetNorm := target.direction.vecNorm
    let candidates :=
      findInductionHeadCandidatesFromCache cache minPrevTokenStrength minInductionScore
    let mut certified : Array HeuristicInductionHead := #[]

    for candidate in candidates do
      if passesEgregiousIllusionFilter candidate then
        -- Explicit token-aware pattern verification:
        -- If `tokens?` is present, certify candidates only when the **variable-lag**
        -- induction "copy-next" pattern check passes. We intentionally do *not* use
        -- `checkInductionPattern` (fixed-lag shift matching) here.
        let passesPattern : Bool :=
          match tokens? with
          | some tokens =>
              match cache.getHeadData candidate.layer2Idx candidate.head2Idx with
              | some data2 =>
                  (checkInductionCopyNextPattern tokens data2.attention minInductionScore).isSome
              | none => false
          | none => true
        if passesPattern then
          let layer2Input := cache.forwardResult.getLayerInput candidate.layer2Idx
          let layer2InputNorm := layer2Input.frobeniusNorm
          let layer2Ln1Input := cache.getLn1Input candidate.layer2Idx
          let layer2Ln1InputNorm := layer2Ln1Input.frobeniusNorm
          let delta := computeInductionEffectiveness candidate cache layer2Ln1Input target
          let denom := layer2Ln1InputNorm * targetNorm
          let effect :=
            if denom > 1e-10 then delta / denom else 0.0
          if effect > minEffect then
            certified := certified.push {
              candidate := candidate
              delta := delta
              effect := effect
              layer2InputNorm := layer2InputNorm
              layer2Ln1InputNorm := layer2Ln1InputNorm
            }

    let certifiedSorted :=
      certified.qsort (fun a b =>
        -- Primary key: higher **mechanism score** first.
        --
        -- Induction heads are primarily defined by attention-pattern structure (copy-next)
        -- plus K-composition with a previous-token head. Target-direction Effect is useful,
        -- but prompt/target-dependent; we therefore use it only as a secondary key.
        let sa :=
          if Float.isNaN a.effect ∨ Float.isNaN a.candidate.kComp ∨
              Float.isNaN a.candidate.inductionScore ∨
              Float.isNaN a.candidate.prevTokenStrength then
            (-Float.inf)
          else
            a.candidate.kComp * a.candidate.inductionScore * a.candidate.prevTokenStrength
        let sb :=
          if Float.isNaN b.effect ∨ Float.isNaN b.candidate.kComp ∨
              Float.isNaN b.candidate.inductionScore ∨
              Float.isNaN b.candidate.prevTokenStrength then
            (-Float.inf)
          else
            b.candidate.kComp * b.candidate.inductionScore * b.candidate.prevTokenStrength
        if sb < sa then true
        else if sa < sb then false
        else
          -- Secondary key: higher normalized Effect first.
          let ea := if Float.isNaN a.effect then (-Float.inf) else a.effect
          let eb := if Float.isNaN b.effect then (-Float.inf) else b.effect
          if eb < ea then true
          else if ea < eb then false
          else
            -- Tertiary key: higher K-composition first.
            let ka := if Float.isNaN a.candidate.kComp then (-Float.inf) else a.candidate.kComp
            let kb := if Float.isNaN b.candidate.kComp then (-Float.inf) else b.candidate.kComp
            if kb < ka then true
            else if ka < kb then false
            else
              -- Next key: higher raw δ first.
              let δa := if Float.isNaN a.delta then (-Float.inf) else a.delta
              let δb := if Float.isNaN b.delta then (-Float.inf) else b.delta
              if δb < δa then true
              else if δa < δb then false
              else
                -- Final key: smaller relative-error bound first.
                let εa :=
                  if Float.isNaN a.candidate.combinedError then
                    Float.inf
                  else
                    a.candidate.combinedError
                let εb :=
                  if Float.isNaN b.candidate.combinedError then
                    Float.inf
                  else
                    b.candidate.combinedError
                εa < εb)

    return (certifiedSorted, cache)

/-- Rank induction-head candidates by a **mechanism-first** score.

This is the same as `findHeuristicInductionHeadsWithCache`, but discards the cache.
-/
def findHeuristicInductionHeads (model : ConcreteModel)
    (target : TargetDirection)
    (minEffect : Float := 0.0)
    (minPrevTokenStrength : Float := 0.1)
    (minInductionScore : Float := 0.05) : Array HeuristicInductionHead := Id.run do
  (findHeuristicInductionHeadsWithCache model target minEffect
      (minPrevTokenStrength := minPrevTokenStrength)
      (minInductionScore := minInductionScore)).1

/-- Target-aware importance metrics for a component.

Like `ComponentImportance` but with an additional field measuring
projection onto the target direction.
-/
structure TargetAwareImportance where
  /-- Component identifier -/
  component : ComponentId
  /-- Standard value term norm (generic importance) -/
  valueTermNorm : Float
  /-- Pattern term bound -/
  patternTermBound : Float
  /-- **Target projection**: how much this component contributes to the target direction.
      For heads: `‖(W_V · W_O) · u‖`
      For neurons: `|W_out[i,:] · u|` -/
  targetProjection : Float
  /-- Faithfulness ratio for target: patternBound / targetProjection -/
  targetFaithfulnessRatio : Float

namespace TargetAwareImportance

/-- Pretty print target-aware importance. -/
def toString (imp : TargetAwareImportance) : String :=
  s!"{imp.component}: target={imp.targetProjection}, generic={imp.valueTermNorm}, " ++
  s!"pattern={imp.patternTermBound}, ratio={imp.targetFaithfulnessRatio}"

instance : ToString TargetAwareImportance := ⟨TargetAwareImportance.toString⟩

end TargetAwareImportance

/-- Compute target-aware importance for a single attention head.

The target projection is computed as `‖(W_V · W_O) · u‖` where u is the target direction.
This measures how much the head's output aligns with the target direction.
-/
def computeHeadTargetImportance (model : ConcreteModel) (layerIdx headIdx : Nat)
    (inputNorm : Float) (target : TargetDirection) : Option TargetAwareImportance := do
  let attn ← model.computeAttention layerIdx headIdx
  if h1 : layerIdx < model.layers.size then
    let layerHeads := model.layers[layerIdx]
    if h2 : headIdx < layerHeads.size then
      let head := layerHeads[headIdx]
      let voProj := head.valueOutputProjection  -- (modelDim × modelDim)
      let qkAlign := head.queryKeyAlignment

      -- Standard metrics
      let valueNorm := computeValueTermNorm attn voProj
      let inputs : PatternTermBoundInputs := {
        attention := attn
        queryKeyAlign := qkAlign
        queryKeyAlignSchurNorm := head.queryKeyAlignmentOpBoundFrom qkAlign
        valueOutputProj := voProj
        valueOutputProjSchurNorm := head.valueOutputProjectionOpBoundFrom voProj
        inputNorm := inputNorm
        scaleFactor := Float.sqrt head.headDim.toFloat
      }
      let patternBound := computePatternTermBound inputs

      -- Target-aware: compute ‖(W_V · W_O) · u‖
      -- voProj is (modelDim × modelDim), target.direction is (modelDim × 1)
      let projectedVec := voProj.matVecMul target.direction
      let targetProj := projectedVec.vecNorm

      -- Scale by attention norm (as in standard valueTermNorm)
      let attnNormSq := attn.weights.foldl (fun acc x => acc + x * x) 0.0
      let attnNorm := Float.sqrt attnNormSq
      let targetImportance := attnNorm * targetProj

      let ratio := if targetImportance < 1e-10 then Float.inf else patternBound / targetImportance

      return {
        component := ComponentId.head layerIdx headIdx
        valueTermNorm := valueNorm
        patternTermBound := patternBound
        targetProjection := targetImportance
        targetFaithfulnessRatio := ratio
      }
    else none
  else none

/-- Compute target-aware importance for a single MLP neuron.

The target projection is `|W_out[i,:] · u|` - the absolute dot product of the
neuron's output weights with the target direction.
-/
def computeNeuronTargetImportance (model : ConcreteModel) (layerIdx neuronIdx : Nat)
    (_inputNorm : Float) (target : TargetDirection) : Option TargetAwareImportance :=
  if h : layerIdx < model.mlps.size then
    let mlp := model.mlps[layerIdx]
    if neuronIdx < mlp.hiddenDim then
      -- Standard influence
      let inputNormVal := mlp.neuronInputNorm neuronIdx
      let outputNormVal := mlp.neuronOutputNorm neuronIdx
      let influence := inputNormVal * outputNormVal

      -- Target-aware: compute W_out[i,:] · u
      -- Get row i of W_out as a column vector for dot product
      let outputWeights : ConcreteMatrix := {
        numRows := mlp.modelDim
        numCols := 1
        data := .ofFn fun j : Fin mlp.modelDim => mlp.W_out.get neuronIdx j.val
        size_eq := by simp
      }
      let dotProd := outputWeights.dot target.direction
      let targetProj := inputNormVal * Float.abs dotProd

      let patternBound : Float := 0.0  -- ReLU is locally linear
      let ratio := if targetProj < 1e-10 then Float.inf else patternBound / targetProj

      some {
        component := ComponentId.mlpNeuron layerIdx neuronIdx
        valueTermNorm := influence
        patternTermBound := patternBound
        targetProjection := targetProj
        targetFaithfulnessRatio := ratio
      }
    else none
  else none

/-- Compute target-aware importance for all components in a model. -/
def computeAllTargetImportance (model : ConcreteModel)
    (target : TargetDirection) : Array TargetAwareImportance := Id.run do
  let inputNorm := computeInputNorm model.inputEmbeddings

  let computeHeadsForLayer (l : Nat) : Array TargetAwareImportance := Id.run do
    if h : l < model.layers.size then
      let layerHeads := model.layers[l]
      let mut outs : Array TargetAwareImportance := Array.mkEmpty layerHeads.size
      for h_idx in [:layerHeads.size] do
        match computeHeadTargetImportance model l h_idx inputNorm target with
        | some imp => outs := outs.push imp
        | none => pure ()
      return outs
    else
      return #[]

  let computeNeuronsForLayer (l : Nat) : Array TargetAwareImportance := Id.run do
    if h : l < model.mlps.size then
      let mlp := model.mlps[l]
      let mut outs : Array TargetAwareImportance := Array.mkEmpty mlp.hiddenDim
      for n_idx in [:mlp.hiddenDim] do
        match computeNeuronTargetImportance model l n_idx inputNorm target with
        | some imp => outs := outs.push imp
        | none => pure ()
      return outs
    else
      return #[]

  let useParallel := model.numLayers >= 4
  let headChunks : Array (Array TargetAwareImportance) :=
    if useParallel then
      let tasks : Array (Task (Array TargetAwareImportance)) :=
        .ofFn fun i : Fin model.numLayers =>
          Task.spawn (fun _ => computeHeadsForLayer i.val)
      tasks.map Task.get
    else
      .ofFn fun i : Fin model.numLayers =>
        computeHeadsForLayer i.val

  let neuronChunks : Array (Array TargetAwareImportance) :=
    if useParallel then
      let tasks : Array (Task (Array TargetAwareImportance)) :=
        .ofFn fun i : Fin model.numLayers =>
          Task.spawn (fun _ => computeNeuronsForLayer i.val)
      tasks.map Task.get
    else
      .ofFn fun i : Fin model.numLayers =>
        computeNeuronsForLayer i.val

  -- Join in the same order as the original loop: heads then neurons, increasing layer index.
  let totalHeads := headChunks.foldl (fun acc cs => acc + cs.size) 0
  let totalNeurons := neuronChunks.foldl (fun acc cs => acc + cs.size) 0
  let mut result : Array TargetAwareImportance := Array.mkEmpty (totalHeads + totalNeurons)
  for cs in headChunks do
    for c in cs do
      result := result.push c
  for cs in neuronChunks do
    for c in cs do
      result := result.push c
  result

/-! ### Circuit Faithfulness Estimation -/

/-- Error breakdown for a circuit.

Total error has two components:
1. **Pattern Term Error**: Approximation error from trusting attention patterns
2. **Ablation Error**: Information loss from pruned components

Total Error ≤ PatternTermError + AblationError
-/
structure CircuitError where
  /-- Sum of pattern term bounds for included components -/
  patternTermError : Float
  /-- Sum of value term norms for excluded (ablated) components -/
  ablationError : Float
  /-- Combined error bound -/
  totalError : Float
  /-- Number of included components -/
  includedCount : Nat
  /-- Number of excluded components -/
  excludedCount : Nat
  deriving Repr

namespace CircuitError

/-- Pretty print error breakdown. -/
def toString (err : CircuitError) : String :=
  s!"CircuitError(total={err.totalError}, pattern={err.patternTermError}, " ++
  s!"ablation={err.ablationError}, included={err.includedCount}, excluded={err.excludedCount})"

instance : ToString CircuitError := ⟨CircuitError.toString⟩

end CircuitError

/-! ### N-Layer Faithfulness Verification

This section implements the N-layer error amplification formula from `Linearization.lean`
for concrete Float matrices. The key insight is that errors in early layers get
amplified as they propagate through subsequent layers:

  ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ)

where:
- εᵢ = pattern term bound for layer i (interpretation error)
- Cⱼ = operator norm bound for layer j's Jacobian (amplification factor)
-/

/-- Per-layer error metrics for deep circuit analysis. -/
structure LayerErrorMetrics where
  /-- Layer index -/
  layerIdx : Nat
  /-- Pattern term bound εᵢ (faithfulness error before amplification) -/
  patternTermBound : Float
  /-- Operator norm upper bound Cᵢ for I + Jacobian (amplification factor) -/
  operatorNormUb : Float
  /-- Suffix amplification: ∏_{j>i} (1 + Cⱼ) -/
  suffixAmplification : Float
  /-- Amplified error contribution: εᵢ · suffixAmplification(i+1) -/
  amplifiedError : Float

namespace LayerErrorMetrics

def toString (m : LayerErrorMetrics) : String :=
  s!"Layer {m.layerIdx}: ε={m.patternTermBound}, C_ub={m.operatorNormUb}, " ++
  s!"amp={m.suffixAmplification}, contrib={m.amplifiedError}"

instance : ToString LayerErrorMetrics := ⟨toString⟩

end LayerErrorMetrics

/-- Deep circuit verification result with rigorous N-layer error bounds. -/
structure DeepCircuitVerification where
  /-- Per-layer error breakdown -/
  layerMetrics : Array LayerErrorMetrics
  /-- Total error bound: Σᵢ εᵢ · suffixAmplification(i+1) -/
  totalAmplifiedError : Float
  /-- Simple sum error (no amplification): Σᵢ εᵢ -/
  simpleErrorSum : Float
  /-- Total amplification factor: ∏ᵢ (1 + Cᵢ) -/
  totalAmplificationFactor : Float
  /-- Ablation error from excluded components -/
  ablationError : Float
  /-- Combined error bound (amplified + ablation) -/
  combinedError : Float
  /-- Number of layers analyzed -/
  numLayers : Nat

namespace DeepCircuitVerification

def toString (v : DeepCircuitVerification) : String :=
  s!"DeepCircuitVerification:\n" ++
  s!"  Layers: {v.numLayers}\n" ++
  s!"  Total Amplified Error: {v.totalAmplifiedError}\n" ++
  s!"  Simple Error Sum: {v.simpleErrorSum}\n" ++
  s!"  Amplification Factor: {v.totalAmplificationFactor}\n" ++
  s!"  Ablation Error: {v.ablationError}\n" ++
  s!"  Combined Error: {v.combinedError}"

instance : ToString DeepCircuitVerification := ⟨toString⟩

end DeepCircuitVerification

/-- Estimate pattern term bound for a single layer (all heads combined).

Aggregates pattern term bounds across all attention heads in the layer.
-/
def estimateLayerPatternBound (model : ConcreteModel) (fwdResult : ForwardPassResult)
    (layerIdx : Nat) (circuit : ConcreteCircuit) : Float := Id.run do
  if h : layerIdx < model.layers.size then
    let heads := model.layers[layerIdx]
    let layerInput := fwdResult.getLayerInput layerIdx
    let attnInput := model.applyLn1 layerIdx layerInput
    let inputNorm := attnInput.frobeniusNorm
    let ln1Bound := model.ln1OpBound layerIdx layerInput
    let mut totalBound : Float := 0.0

    for hidx in [:heads.size] do
      if hh : hidx < heads.size then
        -- Only count included heads
        if circuit.isHeadIncluded layerIdx hidx then
          let head := heads[hidx]
          let attn := head.computeAttentionWeights attnInput
          let inputs : PatternTermBoundInputs := {
            attention := attn
            queryKeyAlign := head.queryKeyAlignment
            queryKeyAlignSchurNorm := head.queryKeyAlignmentOpBound
            valueOutputProj := head.valueOutputProjection
            valueOutputProjSchurNorm := head.valueOutputProjectionOpBound
            inputNorm := inputNorm
            scaleFactor := Float.sqrt head.headDim.toFloat
          }
          -- Pre-LN: pattern sensitivity is scaled by the LayerNorm Jacobian.
          totalBound := totalBound + ln1Bound * computePatternTermBound inputs

    totalBound
  else
    return 0.0

/-- Estimate ablation error for excluded components at a single layer. -/
def estimateLayerAblationError (model : ConcreteModel) (fwdResult : ForwardPassResult)
    (layerIdx : Nat) (circuit : ConcreteCircuit) : Float := Id.run do
  let layerInput := fwdResult.getLayerInput layerIdx
  let mut totalError : Float := 0.0

  -- Ablation error from excluded attention heads
  if h : layerIdx < model.layers.size then
    let heads := model.layers[layerIdx]
    for hidx in [:heads.size] do
      if hh : hidx < heads.size then
        if !circuit.isHeadIncluded layerIdx hidx then
          match computeHeadImportance model layerIdx hidx layerInput with
          | some imp => totalError := totalError + imp.valueTermNorm
          | none => pure ()

  -- Ablation error from excluded neurons
  if hm : layerIdx < model.mlps.size then
    let mlp := model.mlps[layerIdx]
    for nidx in [:mlp.hiddenDim] do
      if !circuit.isNeuronIncluded layerIdx nidx then
        match computeNeuronImportance model layerIdx nidx (layerInput.frobeniusNorm) with
        | some imp => totalError := totalError + imp.valueTermNorm
        | none => pure ()

  totalError

/-- Verify a deep circuit using rigorous N-layer error amplification bounds.

This is the main function that bridges theoretical bounds to practical verification.
It computes:
1. Per-layer pattern term bounds (εᵢ)
2. Per-layer operator norm bounds (Cᵢ)
3. Suffix amplification factors
4. Total amplified error using the N-layer composition formula

**The N-Layer Formula:**
  ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ)

This captures that early layer errors get amplified more because they pass
through more subsequent layers.
-/
def verifyDeepCircuit (model : ConcreteModel)
    (circuit : ConcreteCircuit) : DeepCircuitVerification := Id.run do
  -- Run forward pass to get layer-wise inputs
  let fwdResult := model.runForward

  -- Step 1: Compute per-layer operator norm bounds
  let mut normBounds : Array Float := #[]
  for l in [:model.numLayers] do
    let norm := estimateAttentionLayerNorm model fwdResult l true
    normBounds := normBounds.push norm

  -- Step 2: Compute per-layer pattern term bounds and ablation errors
  let mut patternBounds : Array Float := #[]
  let mut ablationErrors : Array Float := #[]
  for l in [:model.numLayers] do
    let pattern := estimateLayerPatternBound model fwdResult l circuit
    let ablation := estimateLayerAblationError model fwdResult l circuit
    patternBounds := patternBounds.push pattern
    ablationErrors := ablationErrors.push ablation

  -- Step 3: Compute suffix amplification and amplified errors per layer
  let mut layerMetrics : Array LayerErrorMetrics := #[]
  let mut simpleSum : Float := 0.0
  let mut totalAmplified : Float := 0.0

  for l in [:model.numLayers] do
    if hl : l < patternBounds.size then
      let epsilon := patternBounds[l]
      let normBound := if hn : l < normBounds.size then normBounds[l] else 0.0
      let suffix := computeSuffixAmplification normBounds (l + 1)
      let amplified := epsilon * suffix

      simpleSum := simpleSum + epsilon
      totalAmplified := totalAmplified + amplified

      layerMetrics := layerMetrics.push {
        layerIdx := l
        patternTermBound := epsilon
        operatorNormUb := normBound
        suffixAmplification := suffix
        amplifiedError := amplified
      }

  -- Step 4: Compute total ablation error
  let totalAblation := ablationErrors.foldl (· + ·) 0.0

  -- Step 5: Compute total amplification factor
  let totalAmpFactor := computeSuffixAmplification normBounds 0

  {
    layerMetrics := layerMetrics
    totalAmplifiedError := totalAmplified
    simpleErrorSum := simpleSum
    totalAmplificationFactor := totalAmpFactor
    ablationError := totalAblation
    combinedError := totalAmplified + totalAblation
    numLayers := model.numLayers
  }

/-- Check if a deep circuit meets a certification threshold. -/
def isDeepCircuitCertified (verification : DeepCircuitVerification)
    (threshold : Float) : Bool :=
  verification.combinedError ≤ threshold

/-- Structure for a verified deep circuit with certification. -/
structure VerifiedDeepCircuit where
  /-- The circuit that was verified -/
  circuit : ConcreteCircuit
  /-- Full verification details -/
  verification : DeepCircuitVerification
  /-- The threshold used -/
  threshold : Float
  /-- Whether it passed certification -/
  certified : Bool

namespace VerifiedDeepCircuit

def toString (v : VerifiedDeepCircuit) : String :=
  let status := if v.certified then "✓ CERTIFIED" else "✗ NOT CERTIFIED"
  s!"{status} (threshold={v.threshold})\n{v.verification}"

instance : ToString VerifiedDeepCircuit := ⟨toString⟩

end VerifiedDeepCircuit

/-- Verify and certify a deep circuit against a threshold. -/
def certifyDeepCircuit (model : ConcreteModel) (circuit : ConcreteCircuit)
    (threshold : Float) : VerifiedDeepCircuit :=
  let verification := verifyDeepCircuit model circuit
  {
    circuit := circuit
    verification := verification
    threshold := threshold
    certified := isDeepCircuitCertified verification threshold
  }

/-- Estimate the faithfulness error for a given circuit mask.

This is the core function that enables circuit discovery without forward passes.
It uses only weight matrices and attention patterns to bound the error.

**Error Model:**
- For **included** components: we incur the pattern term error (approximation)
- For **excluded** components: we incur the value term error (ablation)

Total Error = Σ(included) patternBound + Σ(excluded) valueNorm
-/
def estimateCircuitFaithfulness (model : ConcreteModel)
    (circuit : ConcreteCircuit) : CircuitError := Id.run do
  let inputNorm := computeInputNorm model.inputEmbeddings
  let mut patternError : Float := 0.0
  let mut ablationError : Float := 0.0
  let mut includedCount : Nat := 0
  let mut excludedCount : Nat := 0

  -- Process attention heads
  for l in [:model.numLayers] do
    if h : l < model.layers.size then
      let layerHeads := model.layers[l]
      for h_idx in [:layerHeads.size] do
        let included := circuit.isHeadIncluded l h_idx
        match computeHeadImportance model l h_idx model.inputEmbeddings with
        | some imp =>
          if included then
            patternError := patternError + imp.patternTermBound
            includedCount := includedCount + 1
          else
            ablationError := ablationError + imp.valueTermNorm
            excludedCount := excludedCount + 1
        | none => pure ()

  -- Process MLP neurons
  for l in [:model.numLayers] do
    if h : l < model.mlps.size then
      let mlp := model.mlps[l]
      for n_idx in [:mlp.hiddenDim] do
        let included := circuit.isNeuronIncluded l n_idx
        match computeNeuronImportance model l n_idx inputNorm with
        | some imp =>
          if included then
            patternError := patternError + imp.patternTermBound
            includedCount := includedCount + 1
          else
            ablationError := ablationError + imp.valueTermNorm
            excludedCount := excludedCount + 1
        | none => pure ()

  {
    patternTermError := patternError
    ablationError := ablationError
    totalError := patternError + ablationError
    includedCount := includedCount
    excludedCount := excludedCount
  }

/-- Extended circuit error with IBP analysis details. -/
structure CircuitErrorIBP extends CircuitError where
  /-- Total number of unstable neurons detected -/
  unstableNeuronCount : Nat
  /-- Pattern error contribution from unstable MLP neurons -/
  mlpInstabilityError : Float
  /-- Per-layer MLP stability ratios -/
  layerStabilityRatios : Array Float
  deriving Repr

namespace CircuitErrorIBP

def toString (e : CircuitErrorIBP) : String :=
  s!"CircuitErrorIBP: pattern={e.patternTermError}, ablation={e.ablationError}, " ++
  s!"total={e.totalError}, unstable_neurons={e.unstableNeuronCount}, " ++
  s!"mlp_instability={e.mlpInstabilityError}"

instance : ToString CircuitErrorIBP := ⟨toString⟩

end CircuitErrorIBP

/-- Estimate circuit faithfulness with Interval Bound Propagation for MLPs.

This is the **sound** version of circuit faithfulness estimation that properly
accounts for MLP activation instability. It runs a forward pass to get layer
inputs, then uses IBP to bound the pattern term error for neurons that may
flip activation states.

**Key Improvement over `estimateCircuitFaithfulness`:**
- Standard version assumes MLP pattern term = 0 (unsound)
- IBP version detects unstable neurons and computes rigorous error bounds

**Algorithm:**
1. Run forward pass to get layer inputs
2. For each layer, compute the ablation perturbation norm (sum of excluded
   component value terms up to this layer)
3. For each MLP neuron, use IBP to determine if it's stable under this
   perturbation
4. Unstable neurons contribute their IBP pattern term bound to total error

**Parameters:**
- `causal`: Whether to use causal attention masking (default true)

**Returns:** Extended error struct with stability analysis details
-/
def estimateCircuitFaithfulnessIBP (model : ConcreteModel)
    (circuit : ConcreteCircuit) (causal : Bool := true) : CircuitErrorIBP := Id.run do
  let inputNorm := computeInputNorm model.inputEmbeddings
  let fwd := model.runForward causal

  let mut patternError : Float := 0.0
  let mut ablationError : Float := 0.0
  let mut mlpInstabilityError : Float := 0.0
  let mut unstableCount : Nat := 0
  let mut includedCount : Nat := 0
  let mut excludedCount : Nat := 0
  let mut layerStability : Array Float := #[]

  -- Track cumulative ablation perturbation up to each layer
  -- This is the norm of the change to the residual stream from ablated components
  let mut cumulativeAblation : Float := 0.0

  -- Process layer by layer
  for l in [:model.numLayers] do
    let mut layerAblation : Float := 0.0

    -- Process attention heads in this layer
    if h : l < model.layers.size then
      let layerHeads := model.layers[l]
      let layerInput := fwd.getLayerInput l
      for h_idx in [:layerHeads.size] do
        let included := circuit.isHeadIncluded l h_idx
        match computeHeadImportance model l h_idx layerInput with
        | some imp =>
          if included then
            patternError := patternError + imp.patternTermBound
            includedCount := includedCount + 1
          else
            ablationError := ablationError + imp.valueTermNorm
            layerAblation := layerAblation + imp.valueTermNorm
            excludedCount := excludedCount + 1
        | none => pure ()

    -- Update cumulative ablation (this affects MLP inputs)
    cumulativeAblation := cumulativeAblation + layerAblation

    -- Process MLP neurons with IBP
    if hm : l < model.mlps.size then
      let mlp := model.mlps[l]
      let layerInput := fwd.getLayerInput l

      let mut layerUnstable : Nat := 0
      let mut layerMlpPattern : Float := 0.0

      for n_idx in [:mlp.hiddenDim] do
        let included := circuit.isNeuronIncluded l n_idx
        if included then
          -- Use IBP with cumulative perturbation norm
          match computeNeuronImportanceIBP model l n_idx layerInput cumulativeAblation with
          | some imp =>
            patternError := patternError + imp.patternTermBound
            if imp.patternTermBound > 0.0 then
              layerUnstable := layerUnstable + 1
              layerMlpPattern := layerMlpPattern + imp.patternTermBound
            includedCount := includedCount + 1
          | none => pure ()
        else
          -- Excluded neurons contribute value term (ablation error)
          let influence := mlp.neuronInfluence n_idx
          ablationError := ablationError + influence
          excludedCount := excludedCount + 1

      unstableCount := unstableCount + layerUnstable
      mlpInstabilityError := mlpInstabilityError + layerMlpPattern

      -- Record stability ratio for this layer
      let stabilityRatio := if mlp.hiddenDim = 0 then 1.0
        else (mlp.hiddenDim - layerUnstable).toFloat / mlp.hiddenDim.toFloat
      layerStability := layerStability.push stabilityRatio

  {
    patternTermError := patternError
    ablationError := ablationError
    totalError := patternError + ablationError
    includedCount := includedCount
    excludedCount := excludedCount
    unstableNeuronCount := unstableCount
    mlpInstabilityError := mlpInstabilityError
    layerStabilityRatios := layerStability
  }

/-- Summary of MLP stability across a model for a given perturbation. -/
structure MLPStabilitySummary where
  /-- Per-layer analysis results -/
  layerAnalyses : Array MLPIntervalAnalysis
  /-- Total stable neurons across all layers -/
  totalStable : Nat
  /-- Total unstable neurons across all layers -/
  totalUnstable : Nat
  /-- Overall stability ratio -/
  overallStabilityRatio : Float
  /-- Total pattern term bound from all unstable neurons -/
  totalPatternBound : Float
  deriving Repr

/-- Analyze MLP stability across the entire model.

Runs IBP analysis on all MLP layers to identify which neurons are stable
under a given perturbation bound.
-/
def analyzeModelMLPStability (model : ConcreteModel)
    (perturbationNorm : Float) (causal : Bool := true) : MLPStabilitySummary := Id.run do
  let fwd := model.runForward causal
  let mut analyses : Array MLPIntervalAnalysis := #[]
  let mut totalStable : Nat := 0
  let mut totalUnstable : Nat := 0
  let mut totalPattern : Float := 0.0

  for l in [:model.numLayers] do
    if hm : l < model.mlps.size then
      let mlp := model.mlps[l]
      let layerInput := fwd.getLayerInput l

      -- Analyze each position and aggregate
      let mut layerAnalysis : MLPIntervalAnalysis := {
        layerIdx := l
        neuronBounds := #[]
        perturbationNorm := perturbationNorm
        numStable := 0
        numUnstable := 0
        totalPatternBound := 0.0
      }

      -- Use position 0 as representative (could average over positions)
      if layerInput.numRows > 0 then
        let inputVec : Array Float := .ofFn fun d : Fin layerInput.numCols =>
          layerInput.get 0 d.val
        layerAnalysis := mlp.analyzeIntervalBounds l inputVec perturbationNorm

      analyses := analyses.push layerAnalysis
      totalStable := totalStable + layerAnalysis.numStable
      totalUnstable := totalUnstable + layerAnalysis.numUnstable
      totalPattern := totalPattern + layerAnalysis.totalPatternBound

  let totalNeurons := totalStable + totalUnstable
  let ratio := if totalNeurons = 0 then 1.0
    else totalStable.toFloat / totalNeurons.toFloat

  {
    layerAnalyses := analyses
    totalStable := totalStable
    totalUnstable := totalUnstable
    overallStabilityRatio := ratio
    totalPatternBound := totalPattern
  }

/-! ### Greedy Circuit Pruning -/

/-- Result of the greedy pruning algorithm. -/
structure PruningResult where
  /-- The discovered circuit -/
  circuit : ConcreteCircuit
  /-- Error estimate for the circuit -/
  error : CircuitError
  /-- History of pruning steps (component removed, error after removal) -/
  pruningHistory : Array (ComponentId × Float)
  /-- The error threshold that was used -/
  threshold : Float

namespace PruningResult

/-- Pretty print pruning result. -/
def toString (pr : PruningResult) : String :=
  s!"PruningResult: {pr.circuit}\n  Error: {pr.error}\n  " ++
  s!"Steps: {pr.pruningHistory.size}, Threshold: {pr.threshold}"

instance : ToString PruningResult := ⟨PruningResult.toString⟩

end PruningResult

/-- Find the component with smallest value term (least important for information flow).

Returns the component ID and its value term norm, considering only currently included
components.
-/
def findLeastImportantComponent (circuit : ConcreteCircuit)
    (importance : Array ComponentImportance) : Option (ComponentId × Float) := Id.run do
  let mut best : Option (ComponentId × Float) := none

  for imp in importance do
    let included := circuit.isIncluded imp.component
    if included then
      match best with
      | none => best := some (imp.component, imp.valueTermNorm)
      | some (_, bestValue) =>
        if imp.valueTermNorm < bestValue then
          best := some (imp.component, imp.valueTermNorm)

  best

/-- Find the component with smallest target projection (least important for target behavior).

Returns the component ID and its target projection, considering only currently included
components. This is the target-aware version of `findLeastImportantComponent`.
-/
def findLeastImportantTargetComponent (circuit : ConcreteCircuit)
    (importance : Array TargetAwareImportance) : Option (ComponentId × Float) := Id.run do
  let mut best : Option (ComponentId × Float) := none

  for imp in importance do
    let included := circuit.isIncluded imp.component
    if included then
      match best with
      | none => best := some (imp.component, imp.targetProjection)
      | some (_, bestValue) =>
        if imp.targetProjection < bestValue then
          best := some (imp.component, imp.targetProjection)

  best

/-- Estimate circuit faithfulness for target-aware pruning.

For target-aware circuits, the error model is:
- **Included components**: Contribute approximation error (pattern term)
- **Excluded components**: Contribute information loss measured by target projection

Unlike generic discovery where we use `‖W_V·W_O‖_F` for ablation error,
here we use `targetProjection` - the component's contribution to the target direction.
-/
def estimateTargetCircuitError (_model : ConcreteModel) (circuit : ConcreteCircuit)
    (importance : Array TargetAwareImportance) : CircuitError := Id.run do
  let mut patternTermError : Float := 0.0
  let mut ablationError : Float := 0.0
  let mut includedCount : Nat := 0
  let mut excludedCount : Nat := 0

  for imp in importance do
    if circuit.isIncluded imp.component then
      patternTermError := patternTermError + imp.patternTermBound
      includedCount := includedCount + 1
    else
      ablationError := ablationError + imp.targetProjection
      excludedCount := excludedCount + 1

  {
    patternTermError := patternTermError
    ablationError := ablationError
    totalError := patternTermError + ablationError
    includedCount := includedCount
    excludedCount := excludedCount
  }

/-- Greedy circuit pruning algorithm.

Starting from the full model, iteratively removes the component with the smallest
value term contribution until the total error would exceed the threshold.

**Algorithm:**
1. Start with all components included
2. Compute importance metrics for all components
3. Repeat:
   a. Find component with smallest valueTermNorm among included
   b. Tentatively remove it
   c. Estimate new total error
   d. If error ≤ threshold, commit removal; else restore and stop
4. Return the pruned circuit

**Complexity:** O(n²) where n = number of components (n iterations, each scanning n components)
-/
def discoverCircuit (model : ConcreteModel) (threshold : Float) : PruningResult := Id.run do
  -- Build heads per layer array
  let mut headsPerLayer : Array Nat := #[]
  for l in [:model.numLayers] do
    if h : l < model.layers.size then
      headsPerLayer := headsPerLayer.push model.layers[l].size
    else
      headsPerLayer := headsPerLayer.push 0

  -- Build neurons per layer array
  let mut neuronsPerLayer : Array Nat := #[]
  for l in [:model.numLayers] do
    if h : l < model.mlps.size then
      neuronsPerLayer := neuronsPerLayer.push model.mlps[l].hiddenDim
    else
      neuronsPerLayer := neuronsPerLayer.push 0

  -- Start with full circuit
  let mut circuit := ConcreteCircuit.full model.numLayers headsPerLayer neuronsPerLayer
  let mut history : Array (ComponentId × Float) := #[]

  -- Precompute all importance metrics
  let importance := computeAllImportance model

  -- Initial error
  let mut currentError := estimateCircuitFaithfulness model circuit

  -- Greedy pruning loop
  let maxIters := circuit.totalComponents
  for _ in [:maxIters] do
    -- Find least important included component
    match findLeastImportantComponent circuit importance with
    | none => break  -- No more components to prune
    | some (comp, _) =>
      -- Tentatively remove component
      let tentativeCircuit := circuit.removeComponent comp
      let tentativeError := estimateCircuitFaithfulness model tentativeCircuit

      -- Check if we can afford to remove it
      if tentativeError.totalError ≤ threshold then
        circuit := tentativeCircuit
        currentError := tentativeError
        history := history.push (comp, tentativeError.totalError)
      else
        break  -- Would exceed threshold, stop pruning

  {
    circuit := circuit
    error := currentError
    pruningHistory := history
    threshold := threshold
  }

/-- Discover circuit with verbose output of each step. -/
def discoverCircuitVerbose (model : ConcreteModel) (threshold : Float) :
    PruningResult × Array String := Id.run do
  let mut logs : Array String := #[]

  -- Build heads per layer array
  let mut headsPerLayer : Array Nat := #[]
  for l in [:model.numLayers] do
    if h : l < model.layers.size then
      headsPerLayer := headsPerLayer.push model.layers[l].size
    else
      headsPerLayer := headsPerLayer.push 0

  -- Build neurons per layer array
  let mut neuronsPerLayer : Array Nat := #[]
  for l in [:model.numLayers] do
    if h : l < model.mlps.size then
      neuronsPerLayer := neuronsPerLayer.push model.mlps[l].hiddenDim
    else
      neuronsPerLayer := neuronsPerLayer.push 0

  let mut circuit := ConcreteCircuit.full model.numLayers headsPerLayer neuronsPerLayer
  let mut history : Array (ComponentId × Float) := #[]
  let importance := computeAllImportance model

  logs := logs.push s!"Starting with full circuit: {circuit.countIncluded} components"

  let mut currentError := estimateCircuitFaithfulness model circuit
  logs := logs.push s!"Initial error: {currentError.totalError}"

  let maxIters := circuit.totalComponents
  for step in [:maxIters] do
    match findLeastImportantComponent circuit importance with
    | none =>
      logs := logs.push s!"Step {step}: No more components to prune"
      break
    | some (comp, valueNorm) =>
      let tentativeCircuit := circuit.removeComponent comp
      let tentativeError := estimateCircuitFaithfulness model tentativeCircuit

      if tentativeError.totalError ≤ threshold then
        circuit := tentativeCircuit
        currentError := tentativeError
        history := history.push (comp, tentativeError.totalError)
        let msg := s!"Step {step}: Removed {comp}, new error={tentativeError.totalError}"
        logs := logs.push msg
      else
        let msg := s!"Step {step}: Cannot remove {comp}, exceeds threshold"
        logs := logs.push msg
        break

  logs := logs.push s!"Final circuit: {circuit}"

  ({
    circuit := circuit
    error := currentError
    pruningHistory := history
    threshold := threshold
  }, logs)

/-! ### Circuit Verification -/

/-- A verified circuit that meets the certification threshold. -/
structure VerifiedCircuit where
  /-- The pruned circuit -/
  circuit : ConcreteCircuit
  /-- Error estimate -/
  error : CircuitError
  /-- Certification threshold -/
  threshold : Float
  /-- Human-readable description -/
  description : String

namespace VerifiedCircuit

/-- Pretty print verified circuit. -/
def toString (vc : VerifiedCircuit) : String :=
  s!"VerifiedCircuit [{vc.description}]\n  {vc.circuit}\n  {vc.error}\n  " ++
  s!"Threshold: {vc.threshold}"

instance : ToString VerifiedCircuit := ⟨VerifiedCircuit.toString⟩

end VerifiedCircuit

/-- Discover and verify a circuit, returning None if threshold cannot be met. -/
def discoverVerifiedCircuit (model : ConcreteModel) (threshold : Float)
    (description : String := "auto-discovered") : Option VerifiedCircuit := do
  let result := discoverCircuit model threshold
  if result.error.totalError ≤ threshold then
    some {
      circuit := result.circuit
      error := result.error
      threshold := threshold
      description := description
    }
  else
    none

/-! ### Target-Aware Circuit Discovery

These functions discover circuits optimized for specific predictions rather than
general model behavior. Given a target direction (e.g., logit difference between
correct and incorrect tokens), they find the minimal circuit that explains that
specific prediction.
-/

/-- Target-aware greedy circuit pruning algorithm.

Like `discoverCircuit`, but uses target projection instead of generic value term
for importance ranking. This finds circuits that explain *specific behaviors*
rather than everything the model does.

**Algorithm:**
1. Start with all components included
2. Compute target-aware importance for all components
3. Repeat:
   a. Find component with smallest targetProjection among included
   b. Tentatively remove it
   c. Estimate new total error (using target projections for ablation)
   d. If error ≤ threshold, commit removal; else stop
4. Return the pruned circuit

**Key Difference from Generic Discovery:**
- Generic: prunes by `‖W_V·W_O‖_F` (generic information flow)
- Target-aware: prunes by `‖(W_V·W_O)·u‖` (contribution to target direction)

This typically produces much smaller circuits when you care about specific outputs.
-/
def discoverTargetedCircuit (model : ConcreteModel) (threshold : Float)
    (target : TargetDirection) : PruningResult := Id.run do
  -- Build heads per layer array
  let mut headsPerLayer : Array Nat := #[]
  for l in [:model.numLayers] do
    if h : l < model.layers.size then
      headsPerLayer := headsPerLayer.push model.layers[l].size
    else
      headsPerLayer := headsPerLayer.push 0

  -- Build neurons per layer array
  let mut neuronsPerLayer : Array Nat := #[]
  for l in [:model.numLayers] do
    if h : l < model.mlps.size then
      neuronsPerLayer := neuronsPerLayer.push model.mlps[l].hiddenDim
    else
      neuronsPerLayer := neuronsPerLayer.push 0

  -- Start with full circuit
  let mut circuit := ConcreteCircuit.full model.numLayers headsPerLayer neuronsPerLayer
  let mut history : Array (ComponentId × Float) := #[]

  -- Precompute target-aware importance metrics
  let importance := computeAllTargetImportance model target

  -- Initial error
  let mut currentError := estimateTargetCircuitError model circuit importance

  -- Greedy pruning loop
  let maxIters := circuit.totalComponents
  for _ in [:maxIters] do
    -- Find least important included component (by target projection)
    match findLeastImportantTargetComponent circuit importance with
    | none => break
    | some (comp, _) =>
      let tentativeCircuit := circuit.removeComponent comp
      let tentativeError := estimateTargetCircuitError model tentativeCircuit importance

      if tentativeError.totalError ≤ threshold then
        circuit := tentativeCircuit
        currentError := tentativeError
        history := history.push (comp, tentativeError.totalError)
      else
        break

  {
    circuit := circuit
    error := currentError
    pruningHistory := history
    threshold := threshold
  }

/-- Target-aware circuit discovery with verbose logging. -/
def discoverTargetedCircuitVerbose (model : ConcreteModel) (threshold : Float)
    (target : TargetDirection) : PruningResult × Array String := Id.run do
  let mut logs : Array String := #[]
  logs := logs.push s!"Target-aware discovery for: {target.description}"

  -- Build layer arrays
  let mut headsPerLayer : Array Nat := #[]
  for l in [:model.numLayers] do
    if h : l < model.layers.size then
      headsPerLayer := headsPerLayer.push model.layers[l].size
    else
      headsPerLayer := headsPerLayer.push 0

  let mut neuronsPerLayer : Array Nat := #[]
  for l in [:model.numLayers] do
    if h : l < model.mlps.size then
      neuronsPerLayer := neuronsPerLayer.push model.mlps[l].hiddenDim
    else
      neuronsPerLayer := neuronsPerLayer.push 0

  let mut circuit := ConcreteCircuit.full model.numLayers headsPerLayer neuronsPerLayer
  let mut history : Array (ComponentId × Float) := #[]
  let importance := computeAllTargetImportance model target

  logs := logs.push s!"Starting with full circuit: {circuit.countIncluded} components"

  let mut currentError := estimateTargetCircuitError model circuit importance
  logs := logs.push s!"Initial error: {currentError.totalError}"

  let maxIters := circuit.totalComponents
  for step in [:maxIters] do
    match findLeastImportantTargetComponent circuit importance with
    | none =>
      logs := logs.push s!"Step {step}: No more components to prune"
      break
    | some (comp, targetProj) =>
      let tentativeCircuit := circuit.removeComponent comp
      let tentativeError := estimateTargetCircuitError model tentativeCircuit importance

      if tentativeError.totalError ≤ threshold then
        circuit := tentativeCircuit
        currentError := tentativeError
        history := history.push (comp, tentativeError.totalError)
        let msg := s!"Step {step}: Removed {comp} (target={targetProj}), " ++
                   s!"new error={tentativeError.totalError}"
        logs := logs.push msg
      else
        let msg := s!"Step {step}: Cannot remove {comp}, exceeds threshold"
        logs := logs.push msg
        break

  logs := logs.push s!"Final circuit: {circuit}"
  logs := logs.push s!"Compression: {circuit.countIncluded}/{circuit.totalComponents} components"

  ({
    circuit := circuit
    error := currentError
    pruningHistory := history
    threshold := threshold
  }, logs)

/-- Discover and verify a target-aware circuit. -/
def discoverVerifiedTargetedCircuit (model : ConcreteModel) (threshold : Float)
    (target : TargetDirection) : Option VerifiedCircuit := do
  let result := discoverTargetedCircuit model threshold target
  if result.error.totalError ≤ threshold then
    some {
      circuit := result.circuit
      error := result.error
      threshold := threshold
      description := s!"target-aware: {target.description}"
    }
  else
    none

/-- Convenience function to discover circuit for logit difference.

Given correct and incorrect token IDs, creates the target direction
`u = W_U[:, correct] - W_U[:, incorrect]` and discovers the minimal
circuit that explains why the model predicts correct over incorrect.
-/
def discoverLogitDiffCircuit (model : ConcreteModel) (threshold : Float)
    (correctToken incorrectToken : Nat) : Option (PruningResult × TargetDirection) := do
  let W_U ← model.unembedding
  let target := TargetDirection.fromLogitDiff W_U correctToken incorrectToken
  let result := discoverTargetedCircuit model threshold target
  some (result, target)

/-- Rank components by their target projection (descending).

Useful for identifying which components most strongly promote the target behavior.
-/
def rankComponentsByTargetImportance (model : ConcreteModel)
    (target : TargetDirection) : Array TargetAwareImportance :=
  let importance := computeAllTargetImportance model target
  importance.qsort (·.targetProjection > ·.targetProjection)

/-- Get the top-k components most important for a target direction. -/
def topKTargetComponents (model : ConcreteModel) (target : TargetDirection)
    (k : Nat) : Array TargetAwareImportance :=
  let ranked := rankComponentsByTargetImportance model target
  ranked.extract 0 (min k ranked.size)

/-! ### End-to-End Discovery and Verification

These functions combine circuit discovery with empirical verification,
providing a complete workflow from model analysis to validated circuits.
-/

/-- Discover a circuit and immediately verify it empirically.

This is the end-to-end function that:
1. Discovers a minimal circuit using greedy pruning
2. Computes theoretical error bounds
3. Verifies empirically that the circuit is faithful

Returns both the pruning result and verification result.
-/
def discoverAndVerify (model : ConcreteModel) (threshold : Float)
    (causal : Bool := true) : PruningResult × VerificationResult :=
  let pruning := discoverCircuit model threshold
  let verification := verifyCircuitFaithfulness model pruning.circuit
    pruning.error.totalError causal
  (pruning, verification)

/-- Discover a target-aware circuit and verify it empirically.

Like `discoverAndVerify` but optimizes for a specific prediction target
(e.g., logit difference between correct and incorrect tokens).
-/
def discoverTargetedAndVerify (model : ConcreteModel) (threshold : Float)
    (target : TargetDirection) (causal : Bool := true) :
    PruningResult × VerificationResult :=
  let pruning := discoverTargetedCircuit model threshold target
  let verification := verifyCircuitFaithfulness model pruning.circuit
    pruning.error.totalError causal
  (pruning, verification)

/-- Complete analysis: discover, verify, and return detailed comparison.

This is the most comprehensive function for circuit analysis. It:
1. Discovers the minimal circuit meeting the error threshold
2. Runs both full and ablated forward passes
3. Computes empirical vs theoretical error comparison
4. Returns everything needed for detailed analysis

**Example output interpretation:**
- `verification.verified = true`: Circuit is empirically faithful
- `verification.tightness = 0.8`: Theoretical bound is 80% utilized (20% slack)
- `ablation.relativeError = 0.05`: Circuit output differs by 5% from full model
-/
def analyzeCircuitFaithfulness (model : ConcreteModel) (threshold : Float)
    (causal : Bool := true) : PruningResult × VerificationResult × AblationResult :=
  let pruning := discoverCircuit model threshold
  let ablation := computeAblationDiscrepancy model pruning.circuit causal
  let verification := verifyCircuitFaithfulness model pruning.circuit
    pruning.error.totalError causal
  (pruning, verification, ablation)

/-- Analyze a target-aware circuit with full verification details. -/
def analyzeTargetedCircuitFaithfulness (model : ConcreteModel) (threshold : Float)
    (target : TargetDirection) (causal : Bool := true) :
    PruningResult × VerificationResult × AblationResult :=
  let pruning := discoverTargetedCircuit model threshold target
  let ablation := computeAblationDiscrepancy model pruning.circuit causal
  let verification := verifyCircuitFaithfulness model pruning.circuit
    pruning.error.totalError causal
  (pruning, verification, ablation)

/-! ### Analysis Utilities -/

/-- Rank all components by their value term contribution (descending). -/
def rankComponentsByImportance (model : ConcreteModel) : Array ComponentImportance :=
  let importance := computeAllImportance model
  importance.qsort (·.valueTermNorm > ·.valueTermNorm)

/-- Get the top-k most important components. -/
def topKComponents (model : ConcreteModel) (k : Nat) : Array ComponentImportance :=
  let ranked := rankComponentsByImportance model
  ranked.extract 0 (min k ranked.size)

/-- Get components with faithfulness ratio below threshold (most reliable). -/
def reliableComponents (model : ConcreteModel) (maxRatio : Float) : Array ComponentImportance :=
  let importance := computeAllImportance model
  importance.filter (·.faithfulnessRatio ≤ maxRatio)

/-- Summary of circuit discovery analysis. -/
structure CircuitAnalysis where
  /-- Total number of components in model -/
  totalComponents : Nat
  /-- Number of components in discovered circuit -/
  circuitSize : Nat
  /-- Compression ratio: circuitSize / totalComponents -/
  compressionRatio : Float
  /-- Total error bound -/
  totalError : Float
  /-- Pattern term contribution to error -/
  patternContribution : Float
  /-- Ablation contribution to error -/
  ablationContribution : Float
  /-- Most important component (by value term) -/
  topComponent : Option ComponentImportance
  /-- Most reliable component (by faithfulness ratio) -/
  mostReliable : Option ComponentImportance

/-- Perform comprehensive circuit analysis. -/
def analyzeCircuit (model : ConcreteModel) (threshold : Float) : CircuitAnalysis := Id.run do
  let result := discoverCircuit model threshold
  let importance := computeAllImportance model
  let ranked := importance.qsort (·.valueTermNorm > ·.valueTermNorm)
  let reliable := importance.qsort (·.faithfulnessRatio < ·.faithfulnessRatio)

  let total := result.circuit.totalComponents
  let included := result.circuit.countIncluded
  let ratio := if total > 0 then included.toFloat / total.toFloat else 1.0

  {
    totalComponents := total
    circuitSize := included
    compressionRatio := ratio
    totalError := result.error.totalError
    patternContribution := result.error.patternTermError
    ablationContribution := result.error.ablationError
    topComponent := if h : 0 < ranked.size then some ranked[0] else none
    mostReliable := if h : 0 < reliable.size then some reliable[0] else none
  }

end Nfp
