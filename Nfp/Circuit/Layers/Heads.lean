-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Logic.Equiv.Fin.Basic
public import Mathlib.Logic.Equiv.Prod
public import Nfp.Circuit.Layers.Reshape

/-!
Head split/merge combinators for transformer-style shapes.
-/

public section

namespace Nfp

namespace Circuit

namespace Layers

universe u v u_in u_out

variable {Node : Type u} [Fintype Node] [DecidableEq Node]
variable {Val : Type v}
variable {Batch : Type u}

/-- Split a hidden index into `(heads, headDim)` using `Fin` product equivalence. -/
def headSplitEquiv (heads dim : Nat) :
    Fin (heads * dim) ≃ Fin heads × Fin dim :=
  (finProdFinEquiv (m := heads) (n := dim)).symm

/-- Merge `(heads, headDim)` back into a hidden index. -/
def headMergeEquiv (heads dim : Nat) :
    Fin heads × Fin dim ≃ Fin (heads * dim) :=
  finProdFinEquiv (m := heads) (n := dim)

/-- Split the hidden dimension inside a batched index. -/
def batchHeadSplitEquiv (Batch : Type u) (heads dim : Nat) :
    Batch × Fin (heads * dim) ≃ Batch × Fin heads × Fin dim :=
  _root_.Equiv.prodCongr (_root_.Equiv.refl Batch) (headSplitEquiv heads dim)

/-- Merge the head and head-dimension inside a batched index. -/
def batchHeadMergeEquiv (Batch : Type u) (heads dim : Nat) :
    Batch × Fin heads × Fin dim ≃ Batch × Fin (heads * dim) :=
  (batchHeadSplitEquiv Batch heads dim).symm

/-- Split heads on the input labels of a typed circuit. -/
def splitHeadsInput {Output : Type u_out} (heads dim : Nat)
    (T : TypedCircuit Node Val (Batch × Fin (heads * dim)) Output) :
    TypedCircuit Node Val (Batch × Fin heads × Fin dim) Output :=
  mapInterface T (batchHeadSplitEquiv Batch heads dim) (_root_.Equiv.refl Output)

/-- Split heads on the output labels of a typed circuit. -/
def splitHeadsOutput {Input : Type u_in} (heads dim : Nat)
    (T : TypedCircuit Node Val Input (Batch × Fin (heads * dim))) :
    TypedCircuit Node Val Input (Batch × Fin heads × Fin dim) :=
  mapInterface T (_root_.Equiv.refl Input) (batchHeadSplitEquiv Batch heads dim)

/-- Split heads on both input and output labels. -/
def splitHeads (heads dim : Nat)
    (T : TypedCircuit Node Val (Batch × Fin (heads * dim)) (Batch × Fin (heads * dim))) :
    TypedCircuit Node Val (Batch × Fin heads × Fin dim) (Batch × Fin heads × Fin dim) :=
  mapInterface T (batchHeadSplitEquiv Batch heads dim) (batchHeadSplitEquiv Batch heads dim)

/-- Merge heads on the input labels of a typed circuit. -/
def mergeHeadsInput {Output : Type u_out} (heads dim : Nat)
    (T : TypedCircuit Node Val (Batch × Fin heads × Fin dim) Output) :
    TypedCircuit Node Val (Batch × Fin (heads * dim)) Output :=
  mapInterface T (batchHeadMergeEquiv Batch heads dim) (_root_.Equiv.refl Output)

/-- Merge heads on the output labels of a typed circuit. -/
def mergeHeadsOutput {Input : Type u_in} (heads dim : Nat)
    (T : TypedCircuit Node Val Input (Batch × Fin heads × Fin dim)) :
    TypedCircuit Node Val Input (Batch × Fin (heads * dim)) :=
  mapInterface T (_root_.Equiv.refl Input) (batchHeadMergeEquiv Batch heads dim)

/-- Merge heads on both input and output labels. -/
def mergeHeads (heads dim : Nat)
    (T : TypedCircuit Node Val (Batch × Fin heads × Fin dim) (Batch × Fin heads × Fin dim)) :
    TypedCircuit Node Val (Batch × Fin (heads * dim)) (Batch × Fin (heads * dim)) :=
  mapInterface T (batchHeadMergeEquiv Batch heads dim) (batchHeadMergeEquiv Batch heads dim)

end Layers

end Circuit

end Nfp
