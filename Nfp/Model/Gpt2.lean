-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Rat
import Nfp.Circuit.Cert.ValueRange

/-!
Exact GPT-2 head-slice data for induction certification.

This module holds the precise token embeddings, position embeddings, and head
projection weights needed to build `InductionHeadInputs` for a single head.
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
  scale : Rat
  /-- Token ids for the prompt. -/
  tokens : Fin seq → Fin vocab
  /-- Token embedding matrix. -/
  wte : Fin vocab → Fin dModel → Rat
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

/-- Token-plus-position embeddings for a GPT-2 head slice. -/
def Gpt2HeadSlice.embed {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) :
    Fin seq → Fin dModel → Rat :=
  fun q d => slice.wte (slice.tokens q) d + slice.wpe q d

/-- Direction vector in model space for a GPT-2 head slice. -/
def Gpt2HeadSlice.directionVec {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) : Fin dModel → Rat :=
  fun d => slice.wte slice.direction.target d - slice.wte slice.direction.negative d

end Model

end Nfp
