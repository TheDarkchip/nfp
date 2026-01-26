-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Core.Basic
public import Nfp.Circuit.Cert.ValueRange
public import Nfp.Linear.FinFold

/-!
Exact GPT-2 slices for induction certification.

This module holds token embeddings, unembedding rows, head projection weights,
and per-layer MLP/LayerNorm parameters used to define `InductionHeadInputs`
and bound computations.
-/

public section

namespace Nfp

namespace Model

open Nfp.Circuit

/-- Token indices describing a logit-diff direction (target minus negative). -/
structure DirectionTokens (vocab : Nat) where
  /-- Target token index. -/
  target : Fin vocab
  /-- Negative token index. -/
  negative : Fin vocab

/-- Convert `DirectionTokens` to a `DirectionSpec`. -/
def DirectionTokens.spec {vocab : Nat} (dir : DirectionTokens vocab) : DirectionSpec :=
  { target := dir.target.val, negative := dir.negative.val }

/-- Direction vector in model space from unembedding rows. -/
def DirectionTokens.directionVec {vocab dModel : Nat} (dir : DirectionTokens vocab)
    (wunembed : Fin vocab → Fin dModel → Rat) : Fin dModel → Rat :=
  fun d => wunembed dir.target d - wunembed dir.negative d

/-- Unembedding logit for a token (Rat-valued). -/
def unembedLogit {vocab dModel : Nat}
    (wunembed : Fin vocab → Fin dModel → Rat) (h : Fin dModel → Rat) (tok : Fin vocab) : Rat :=
  Linear.dotFin dModel (fun d => wunembed tok d) h

/-- Unembedding logit for a token (Real-valued). -/
def unembedLogitReal {vocab dModel : Nat}
    (wunembed : Fin vocab → Fin dModel → Rat) (h : Fin dModel → Real) (tok : Fin vocab) : Real :=
  dotProduct (fun d => (wunembed tok d : Real)) h

/-- Logit-diff equals the direction dot product of the hidden state. -/
theorem logitDiffReal_eq_direction_dot {vocab dModel : Nat}
    (wunembed : Fin vocab → Fin dModel → Rat) (dir : DirectionTokens vocab)
    (h : Fin dModel → Real) :
    dotProduct (fun d => (dir.directionVec wunembed d : Real)) h =
      unembedLogitReal wunembed h dir.target - unembedLogitReal wunembed h dir.negative := by
  classical
  simp [DirectionTokens.directionVec, unembedLogitReal, dotProduct, sub_mul,
    Finset.sum_sub_distrib]

/-- Exact GPT-2 head slice needed to build induction-head inputs. -/
structure Gpt2HeadSlice (seq dModel dHead vocab : Nat) where
  /-- Softmax scale factor (e.g. `1/8` for head dim 64). -/
  scale : Rat
  /-- Token ids for the prompt. -/
  tokens : Fin seq → Fin vocab
  /-- Token embedding matrix. -/
  wte : Fin vocab → Fin dModel → Rat
  /-- Unembedding matrix (token logits). -/
  wunembed : Fin vocab → Fin dModel → Rat
  /-- Positional embedding matrix. -/
  wpe : Fin seq → Fin dModel → Rat
  /-- Query projection weights. -/
  wq : Fin dModel → Fin dHead → Rat
  /-- Query projection bias. -/
  bq : Fin dHead → Rat
  /-- Key projection weights. -/
  wk : Fin dModel → Fin dHead → Rat
  /-- Key projection bias. -/
  bk : Fin dHead → Rat
  /-- Value projection weights. -/
  wv : Fin dModel → Fin dHead → Rat
  /-- Value projection bias. -/
  bv : Fin dHead → Rat
  /-- Output projection weights for this head slice. -/
  wo : Fin dModel → Fin dHead → Rat
  /-- Attention output bias (shared across heads). -/
  attnBias : Fin dModel → Rat
  /-- LayerNorm epsilon for the attention input. -/
  lnEps : Rat
  /-- LayerNorm scale for the attention input. -/
  ln1Gamma : Fin dModel → Rat
  /-- LayerNorm bias for the attention input. -/
  ln1Beta : Fin dModel → Rat
  /-- Direction tokens for logit-diff certification. -/
  direction : DirectionTokens vocab

/-- Exact per-head attention weights and biases. -/
structure Gpt2HeadWeights (dModel dHead : Nat) where
  /-- Query projection weights. -/
  wq : Fin dModel → Fin dHead → Rat
  /-- Query projection bias. -/
  bq : Fin dHead → Rat
  /-- Key projection weights. -/
  wk : Fin dModel → Fin dHead → Rat
  /-- Key projection bias. -/
  bk : Fin dHead → Rat
  /-- Value projection weights. -/
  wv : Fin dModel → Fin dHead → Rat
  /-- Value projection bias. -/
  bv : Fin dHead → Rat
  /-- Output projection weights for this head slice. -/
  wo : Fin dModel → Fin dHead → Rat

/-- Exact GPT-2 layer slice with MLP and LayerNorm parameters. -/
structure Gpt2LayerSlice (dModel hidden : Nat) where
  /-- Attention output bias (shared across heads). -/
  attnBias : Fin dModel → Rat
  /-- MLP input projection weights. -/
  mlpWIn : Fin dModel → Fin hidden → Rat
  /-- MLP input projection bias. -/
  mlpBIn : Fin hidden → Rat
  /-- MLP output projection weights. -/
  mlpWOut : Fin hidden → Fin dModel → Rat
  /-- MLP output projection bias. -/
  mlpBOut : Fin dModel → Rat
  /-- LayerNorm scale for the attention input. -/
  ln1Gamma : Fin dModel → Rat
  /-- LayerNorm bias for the attention input. -/
  ln1Beta : Fin dModel → Rat
  /-- LayerNorm scale for the MLP input. -/
  ln2Gamma : Fin dModel → Rat
  /-- LayerNorm bias for the MLP input. -/
  ln2Beta : Fin dModel → Rat

/-- Final LayerNorm parameters applied before unembedding. -/
structure Gpt2FinalLayerNorm (dModel : Nat) where
  /-- LayerNorm scale. -/
  gamma : Fin dModel → Rat
  /-- LayerNorm bias. -/
  beta : Fin dModel → Rat

/-- Token-plus-position embeddings for a GPT-2 head slice. -/
def Gpt2HeadSlice.embed {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) :
    Fin seq → Fin dModel → Rat :=
  fun q d => slice.wte (slice.tokens q) d + slice.wpe q d

/-- Direction vector in model space for a GPT-2 head slice. -/
def Gpt2HeadSlice.directionVec {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) : Fin dModel → Rat :=
  slice.direction.directionVec slice.wunembed

end Model

end Nfp
