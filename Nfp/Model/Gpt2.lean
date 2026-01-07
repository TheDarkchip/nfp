-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Core.Basic
import Nfp.Circuit.Cert.ValueRange

/-!
Exact GPT-2 slices for induction certification and downstream bounds.

This module holds token embeddings, head projection weights, and per-layer
MLP/LayerNorm parameters needed to build `InductionHeadInputs` and downstream
bound computations.
-/

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

/-- Exact GPT-2 head slice needed to build induction-head inputs. -/
structure Gpt2HeadSlice (seq dModel dHead vocab : Nat) where
  /-- Softmax scale factor (e.g. `1/8` for head dim 64). -/
  scale : Dyadic
  /-- Token ids for the prompt. -/
  tokens : Fin seq → Fin vocab
  /-- Token embedding matrix. -/
  wte : Fin vocab → Fin dModel → Dyadic
  /-- Positional embedding matrix. -/
  wpe : Fin seq → Fin dModel → Dyadic
  /-- Query projection weights. -/
  wq : Fin dModel → Fin dHead → Dyadic
  /-- Query projection bias. -/
  bq : Fin dHead → Dyadic
  /-- Key projection weights. -/
  wk : Fin dModel → Fin dHead → Dyadic
  /-- Key projection bias. -/
  bk : Fin dHead → Dyadic
  /-- Value projection weights. -/
  wv : Fin dModel → Fin dHead → Dyadic
  /-- Value projection bias. -/
  bv : Fin dHead → Dyadic
  /-- Output projection weights for this head slice. -/
  wo : Fin dModel → Fin dHead → Dyadic
  /-- Attention output bias (shared across heads). -/
  attnBias : Fin dModel → Dyadic
  /-- LayerNorm epsilon for the attention input. -/
  lnEps : Dyadic
  /-- LayerNorm scale for the attention input. -/
  ln1Gamma : Fin dModel → Dyadic
  /-- LayerNorm bias for the attention input. -/
  ln1Beta : Fin dModel → Dyadic
  /-- Direction tokens for logit-diff certification. -/
  direction : DirectionTokens vocab

/-- Exact per-head attention weights and biases. -/
structure Gpt2HeadWeights (dModel dHead : Nat) where
  /-- Query projection weights. -/
  wq : Fin dModel → Fin dHead → Dyadic
  /-- Query projection bias. -/
  bq : Fin dHead → Dyadic
  /-- Key projection weights. -/
  wk : Fin dModel → Fin dHead → Dyadic
  /-- Key projection bias. -/
  bk : Fin dHead → Dyadic
  /-- Value projection weights. -/
  wv : Fin dModel → Fin dHead → Dyadic
  /-- Value projection bias. -/
  bv : Fin dHead → Dyadic
  /-- Output projection weights for this head slice. -/
  wo : Fin dModel → Fin dHead → Dyadic

/-- Exact GPT-2 layer slice with MLP and LayerNorm parameters. -/
structure Gpt2LayerSlice (dModel hidden : Nat) where
  /-- Attention output bias (shared across heads). -/
  attnBias : Fin dModel → Dyadic
  /-- MLP input projection weights. -/
  mlpWIn : Fin dModel → Fin hidden → Dyadic
  /-- MLP input projection bias. -/
  mlpBIn : Fin hidden → Dyadic
  /-- MLP output projection weights. -/
  mlpWOut : Fin hidden → Fin dModel → Dyadic
  /-- MLP output projection bias. -/
  mlpBOut : Fin dModel → Dyadic
  /-- LayerNorm scale for the attention input. -/
  ln1Gamma : Fin dModel → Dyadic
  /-- LayerNorm bias for the attention input. -/
  ln1Beta : Fin dModel → Dyadic
  /-- LayerNorm scale for the MLP input. -/
  ln2Gamma : Fin dModel → Dyadic
  /-- LayerNorm bias for the MLP input. -/
  ln2Beta : Fin dModel → Dyadic

/-- Final LayerNorm parameters applied before unembedding. -/
structure Gpt2FinalLayerNorm (dModel : Nat) where
  /-- LayerNorm scale. -/
  gamma : Fin dModel → Dyadic
  /-- LayerNorm bias. -/
  beta : Fin dModel → Dyadic

/-- Token-plus-position embeddings for a GPT-2 head slice. -/
def Gpt2HeadSlice.embed {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) :
    Fin seq → Fin dModel → Dyadic :=
  fun q d => slice.wte (slice.tokens q) d + slice.wpe q d

/-- Direction vector in model space for a GPT-2 head slice. -/
def Gpt2HeadSlice.directionVec {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) : Fin dModel → Dyadic :=
  fun d => slice.wte slice.direction.target d - slice.wte slice.direction.negative d

end Model

end Nfp
