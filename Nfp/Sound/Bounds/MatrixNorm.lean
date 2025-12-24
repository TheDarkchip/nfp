-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Nfp.Sound.Bounds.Basic

namespace Nfp.Sound

open scoped BigOperators

/-!
# Matrix norm helpers (Rat, row-sum)

Utilities for row-sum operator norm bounds on finite matrices.
-/

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

end Nfp.Sound
