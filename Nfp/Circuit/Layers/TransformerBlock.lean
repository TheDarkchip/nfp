-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Compose
import Nfp.Circuit.Layers.Attention

/-!
Transformer block wiring built from sequential composition and residual links.
-/

namespace Nfp

namespace Circuit

namespace Layers

universe u v u_in

variable {Val : Type v} [Add Val]
variable {Input : Type u_in} [Fintype Input] [DecidableEq Input]

variable {NodeLn1 : Type u} [Fintype NodeLn1] [DecidableEq NodeLn1]
variable {NodeAttn : Type u} [Fintype NodeAttn] [DecidableEq NodeAttn]
variable {NodeLn2 : Type u} [Fintype NodeLn2] [DecidableEq NodeLn2]
variable {NodeMlp : Type u} [Fintype NodeMlp] [DecidableEq NodeMlp]

/-- Node type for the attention subpath (LN1 ∘ attention). -/
abbrev AttnPathNode (NodeLn1 NodeAttn : Type u) : Type u :=
  NodeLn1 ⊕ NodeAttn

/-- Node type for the attention residual wrapper. -/
abbrev AttnResidualNode (NodeLn1 NodeAttn : Type u) (Input : Type u_in) :
    Type (max u u_in) :=
  (AttnPathNode NodeLn1 NodeAttn) ⊕ Input

/-- Node type for the MLP subpath (LN2 ∘ MLP). -/
abbrev MlpPathNode (NodeLn2 NodeMlp : Type u) : Type u :=
  NodeLn2 ⊕ NodeMlp

/-- Node type for the MLP residual wrapper. -/
abbrev MlpResidualNode (NodeLn2 NodeMlp : Type u) (Input : Type u_in) :
    Type (max u u_in) :=
  (MlpPathNode NodeLn2 NodeMlp) ⊕ Input

/-- Node type for a full transformer block. -/
abbrev TransformerBlockNode (NodeLn1 NodeAttn NodeLn2 NodeMlp : Type u) (Input : Type u_in) :
    Type (max u u_in) :=
  (AttnResidualNode NodeLn1 NodeAttn Input) ⊕ (MlpResidualNode NodeLn2 NodeMlp Input)

/-- Compose a GPT-style transformer block from LN/attention/MLP circuits. -/
def transformerBlock
    (ln1 : TypedCircuit NodeLn1 Val Input Input)
    (attn : TypedCircuit NodeAttn Val Input Input)
    (ln2 : TypedCircuit NodeLn2 Val Input Input)
    (mlp : TypedCircuit NodeMlp Val Input Input) :
    TypedCircuit (TransformerBlockNode NodeLn1 NodeAttn NodeLn2 NodeMlp Input) Val Input Input :=
  let attnPath := TypedCircuit.seq ln1 attn
  let attnRes := TypedCircuit.residual attnPath
  let mlpPath := TypedCircuit.seq ln2 mlp
  let mlpRes := TypedCircuit.residual mlpPath
  TypedCircuit.seq attnRes mlpRes

/-- Token hidden state type for GPT-2 style blocks. -/
abbrev BlockInput (Batch : Type) (seq heads dim : Nat) : Type :=
  Batch × Fin seq × Hidden heads dim

/-- Transformer block specialized to GPT-style hidden states. -/
def gpt2Block {Batch : Type} [Fintype Batch] [DecidableEq Batch] {seq heads dim : Nat}
    (ln1 : TypedCircuit NodeLn1 Val (BlockInput Batch seq heads dim)
      (BlockInput Batch seq heads dim))
    (attn : TypedCircuit NodeAttn Val (BlockInput Batch seq heads dim)
      (BlockInput Batch seq heads dim))
    (ln2 : TypedCircuit NodeLn2 Val (BlockInput Batch seq heads dim)
      (BlockInput Batch seq heads dim))
    (mlp : TypedCircuit NodeMlp Val (BlockInput Batch seq heads dim)
      (BlockInput Batch seq heads dim)) :
    TypedCircuit (TransformerBlockNode NodeLn1 NodeAttn NodeLn2 NodeMlp
      (BlockInput Batch seq heads dim)) Val (BlockInput Batch seq heads dim)
        (BlockInput Batch seq heads dim) :=
  transformerBlock ln1 attn ln2 mlp

end Layers

end Circuit

end Nfp
