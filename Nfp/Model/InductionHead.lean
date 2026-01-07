-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Basic
import Nfp.Core.Basic
import Nfp.Circuit.Cert.ValueRange

/-!
Exact inputs for induction-head scoring and value-direction computations.

These structures store exact dyadic inputs (embeddings and weights) for a
single attention head. They are intended to be consumed by sound builders.
-/

namespace Nfp

namespace Model

open Nfp.Circuit

/-- Exact head inputs for induction certification. -/
structure InductionHeadInputs (seq dModel dHead : Nat) where
  /-- Softmax scale factor (e.g. `1/8` for GPT-2-small head dim 64). -/
  scale : Dyadic
  /-- Active queries for which bounds are required. -/
  active : Finset (Fin seq)
  /-- `prev` selector for induction-style attention. -/
  prev : Fin seq → Fin seq
  /-- Token embeddings for the sequence. -/
  embed : Fin seq → Fin dModel → Dyadic
  /-- LayerNorm epsilon used before attention. -/
  lnEps : Dyadic
  /-- LayerNorm scale for pre-attention normalization. -/
  ln1Gamma : Fin dModel → Dyadic
  /-- LayerNorm bias for pre-attention normalization. -/
  ln1Beta : Fin dModel → Dyadic
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
  /-- Output projection weights (head slice). -/
  wo : Fin dModel → Fin dHead → Dyadic
  /-- Attention output bias (shared across heads). -/
  attnBias : Fin dModel → Dyadic
  /-- Whether to apply a causal mask to attention scores. -/
  maskCausal : Bool
  /-- Score value for masked entries (e.g. `-10000` for GPT-2 causal masking). -/
  maskValue : Dyadic
  /-- Logit-diff direction metadata. -/
  directionSpec : DirectionSpec
  /-- Logit-diff direction vector in model space. -/
  direction : Fin dModel → Dyadic

end Model

end Nfp
