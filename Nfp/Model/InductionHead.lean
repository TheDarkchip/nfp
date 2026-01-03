-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.Order.Ring.Rat
import Nfp.Circuit.Cert.ValueRange

/-!
Exact inputs for induction-head scoring and value-direction computations.

These structures store exact rational inputs (embeddings and weights) for a
single attention head. They are intended to be consumed by sound builders.
-/

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
  /-- Token embeddings for the sequence. -/
  embed : Fin seq → Fin dModel → Rat
  /-- Query projection weights. -/
  wq : Fin dModel → Fin dHead → Rat
  /-- Key projection weights. -/
  wk : Fin dModel → Fin dHead → Rat
  /-- Value projection weights. -/
  wv : Fin dModel → Fin dHead → Rat
  /-- Output projection weights (head slice). -/
  wo : Fin dModel → Fin dHead → Rat
  /-- Logit-diff direction metadata. -/
  directionSpec : DirectionSpec
  /-- Logit-diff direction vector in model space. -/
  direction : Fin dModel → Rat

end Model

end Nfp
