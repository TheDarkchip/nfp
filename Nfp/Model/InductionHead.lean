-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Basic
public import Nfp.Core.Basic
public import Nfp.Circuit.Cert.ValueRange

/-!
Exact inputs for induction-head scoring and value-direction computations.

These structures store exact rational inputs (pre-LN residual stream and weights)
for a single attention head. They are intended to be consumed by sound builders.
-/

public section

namespace Nfp

namespace Model

open Nfp.Circuit

/-- Exact head inputs for induction certification. -/
structure InductionHeadInputs (seq dModel dHead : Nat) where
  /-- Softmax scale factor (e.g. `1/8` for GPT-2-small head dim 64). -/
  scale : Rat
  /-- Active queries for which bounds are required. -/
  active : Finset (Fin seq)
  /-- `prev` selector for induction-style attention. -/
  prev : Fin seq → Fin seq
  /-- Pre-LN residual stream for the sequence. -/
  embed : Fin seq → Fin dModel → Rat
  /-- LayerNorm epsilon used before attention. -/
  lnEps : Rat
  /-- LayerNorm scale for pre-attention normalization. -/
  ln1Gamma : Fin dModel → Rat
  /-- LayerNorm bias for pre-attention normalization. -/
  ln1Beta : Fin dModel → Rat
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
  /-- Output projection weights (head slice). -/
  wo : Fin dModel → Fin dHead → Rat
  /-- Attention output bias (shared across heads). -/
  attnBias : Fin dModel → Rat
  /-- Whether to apply a causal mask to attention scores. -/
  maskCausal : Bool
  /-- Score value for masked entries (e.g. `-10000` for GPT-2 causal masking). -/
  maskValue : Rat
  /-- Logit-diff direction metadata. -/
  directionSpec : DirectionSpec
  /-- Logit-diff direction vector in model space. -/
  direction : Fin dModel → Rat

/--
Construct induction-head inputs from a pre-LN residual stream.

This makes the canonical use explicit: `embed` stores the pre-LN residual stream
that LayerNorm and value bounds act on, not necessarily raw token embeddings.
-/
abbrev InductionHeadInputs.ofPreLnResidual {seq dModel dHead : Nat}
    (scale : Rat) (active : Finset (Fin seq)) (prev : Fin seq → Fin seq)
    (preLn : Fin seq → Fin dModel → Rat)
    (lnEps : Rat) (ln1Gamma : Fin dModel → Rat) (ln1Beta : Fin dModel → Rat)
    (wq : Fin dModel → Fin dHead → Rat) (bq : Fin dHead → Rat)
    (wk : Fin dModel → Fin dHead → Rat) (bk : Fin dHead → Rat)
    (wv : Fin dModel → Fin dHead → Rat) (bv : Fin dHead → Rat)
    (wo : Fin dModel → Fin dHead → Rat) (attnBias : Fin dModel → Rat)
    (maskCausal : Bool) (maskValue : Rat)
    (directionSpec : DirectionSpec) (direction : Fin dModel → Rat) :
    InductionHeadInputs seq dModel dHead :=
  { scale := scale
    active := active
    prev := prev
    embed := preLn
    lnEps := lnEps
    ln1Gamma := ln1Gamma
    ln1Beta := ln1Beta
    wq := wq
    bq := bq
    wk := wk
    bk := bk
    wv := wv
    bv := bv
    wo := wo
    attnBias := attnBias
    maskCausal := maskCausal
    maskValue := maskValue
    directionSpec := directionSpec
    direction := direction }

end Model

end Nfp
