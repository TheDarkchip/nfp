-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Layers.Heads
import Nfp.Circuit.Layers.Tensor

/-!
QKV and output projection wiring for attention layers.
-/

namespace Nfp

namespace Circuit

namespace Layers

universe v

variable {Batch : Type} [Fintype Batch] [DecidableEq Batch]
variable {Val : Type v} [NonUnitalNonAssocSemiring Val]

/-- Hidden dimension type for a `heads × dim` factorization. -/
abbrev Hidden (heads dim : Nat) : Type := Fin (heads * dim)

/-- Node type for Q/K/V and output projection layers. -/
abbrev QkvNode (Batch : Type) (heads dim : Nat) : Type :=
  BatchedLinearNode Batch (Hidden heads dim) (Hidden heads dim)

/-- Q projection with head-split output. -/
def qProj (heads dim : Nat) (Wq : Matrix (Hidden heads dim) (Hidden heads dim) Val) :
    TypedCircuit (QkvNode Batch heads dim) Val (Batch × Hidden heads dim)
      (Batch × Fin heads × Fin dim) :=
  splitHeadsOutput (Batch := Batch) heads dim
    (batchedLinearTyped (Batch := Batch)
      (Row := Hidden heads dim) (Col := Hidden heads dim) Wq)

/-- K projection with head-split output. -/
def kProj (heads dim : Nat) (Wk : Matrix (Hidden heads dim) (Hidden heads dim) Val) :
    TypedCircuit (QkvNode Batch heads dim) Val (Batch × Hidden heads dim)
      (Batch × Fin heads × Fin dim) :=
  splitHeadsOutput (Batch := Batch) heads dim
    (batchedLinearTyped (Batch := Batch)
      (Row := Hidden heads dim) (Col := Hidden heads dim) Wk)

/-- V projection with head-split output. -/
def vProj (heads dim : Nat) (Wv : Matrix (Hidden heads dim) (Hidden heads dim) Val) :
    TypedCircuit (QkvNode Batch heads dim) Val (Batch × Hidden heads dim)
      (Batch × Fin heads × Fin dim) :=
  splitHeadsOutput (Batch := Batch) heads dim
    (batchedLinearTyped (Batch := Batch)
      (Row := Hidden heads dim) (Col := Hidden heads dim) Wv)

/-- Output projection with head-merged input. -/
def outProj (heads dim : Nat) (Wo : Matrix (Hidden heads dim) (Hidden heads dim) Val) :
    TypedCircuit (QkvNode Batch heads dim) Val (Batch × Fin heads × Fin dim)
      (Batch × Hidden heads dim) :=
  splitHeadsInput (Batch := Batch) heads dim
    (batchedLinearTyped (Batch := Batch)
      (Row := Hidden heads dim) (Col := Hidden heads dim) Wo)

end Layers

end Circuit

end Nfp
