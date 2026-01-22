-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Nfp.Core.Basic
public import Nfp.Linear.FinFold

/-!
Exact score computation from post-LayerNorm residuals and head slices.

These definitions provide a trusted, explicit formula for QK logits
using rational inputs.
-/

public section

namespace Nfp

namespace Sound

namespace Induction

open scoped BigOperators

open Nfp.Linear

variable {seq dModel dHead : Nat}

/-- Query projection from post-LN residuals (Rat-valued). -/
def qRatOfSlice (resid : Fin seq → Fin dModel → Rat)
    (wq : Fin dModel → Fin dHead → Rat) (bq : Fin dHead → Rat) :
    Fin seq → Fin dHead → Rat :=
  fun q d => dotFin dModel (fun i => resid q i) (fun i => wq i d) + bq d

/-- Unfolding lemma for `qRatOfSlice`. -/
theorem qRatOfSlice_def (resid : Fin seq → Fin dModel → Rat)
    (wq : Fin dModel → Fin dHead → Rat) (bq : Fin dHead → Rat)
    (q : Fin seq) (d : Fin dHead) :
    qRatOfSlice resid wq bq q d =
      dotFin dModel (fun i => resid q i) (fun i => wq i d) + bq d := by
  rfl

/-- Key projection from post-LN residuals (Rat-valued). -/
def kRatOfSlice (resid : Fin seq → Fin dModel → Rat)
    (wk : Fin dModel → Fin dHead → Rat) (bk : Fin dHead → Rat) :
    Fin seq → Fin dHead → Rat :=
  fun k d => dotFin dModel (fun i => resid k i) (fun i => wk i d) + bk d

/-- Unfolding lemma for `kRatOfSlice`. -/
theorem kRatOfSlice_def (resid : Fin seq → Fin dModel → Rat)
    (wk : Fin dModel → Fin dHead → Rat) (bk : Fin dHead → Rat)
    (k : Fin seq) (d : Fin dHead) :
    kRatOfSlice resid wk bk k d =
      dotFin dModel (fun i => resid k i) (fun i => wk i d) + bk d := by
  rfl

/-- Unmasked attention score for a head slice (Rat-valued). -/
def baseScoreRatOfSlice (scale : Rat)
    (resid : Fin seq → Fin dModel → Rat)
    (wq : Fin dModel → Fin dHead → Rat) (bq : Fin dHead → Rat)
    (wk : Fin dModel → Fin dHead → Rat) (bk : Fin dHead → Rat) :
    Fin seq → Fin seq → Rat :=
  fun q k =>
    let qv := qRatOfSlice (seq := seq) (dModel := dModel) (dHead := dHead) resid wq bq q
    let kv := kRatOfSlice (seq := seq) (dModel := dModel) (dHead := dHead) resid wk bk k
    scale * dotFin dHead qv kv

/-- Unfolding lemma for `baseScoreRatOfSlice`. -/
theorem baseScoreRatOfSlice_def (scale : Rat)
    (resid : Fin seq → Fin dModel → Rat)
    (wq : Fin dModel → Fin dHead → Rat) (bq : Fin dHead → Rat)
    (wk : Fin dModel → Fin dHead → Rat) (bk : Fin dHead → Rat)
    (q k : Fin seq) :
    baseScoreRatOfSlice (seq := seq) (dModel := dModel) (dHead := dHead)
        scale resid wq bq wk bk q k =
      let qv :=
        qRatOfSlice (seq := seq) (dModel := dModel) (dHead := dHead) resid wq bq q
      let kv :=
        kRatOfSlice (seq := seq) (dModel := dModel) (dHead := dHead) resid wk bk k
      scale * dotFin dHead qv kv := by
  rfl

/-- Attention scores from a head slice with optional causal masking (Rat-valued). -/
def scoresRatOfSlice (scale : Rat) (maskValue : Rat) (maskCausal : Bool)
    (resid : Fin seq → Fin dModel → Rat)
    (wq : Fin dModel → Fin dHead → Rat) (bq : Fin dHead → Rat)
    (wk : Fin dModel → Fin dHead → Rat) (bk : Fin dHead → Rat) :
    Fin seq → Fin seq → Rat :=
  fun q k =>
    let base :=
      baseScoreRatOfSlice (seq := seq) (dModel := dModel) (dHead := dHead)
        scale resid wq bq wk bk q k
    if maskCausal then
      if k ≤ q then
        base
      else
        maskValue
    else
      base

/-- Unfolding lemma for `scoresRatOfSlice`. -/
theorem scoresRatOfSlice_def (scale : Rat) (maskValue : Rat) (maskCausal : Bool)
    (resid : Fin seq → Fin dModel → Rat)
    (wq : Fin dModel → Fin dHead → Rat) (bq : Fin dHead → Rat)
    (wk : Fin dModel → Fin dHead → Rat) (bk : Fin dHead → Rat)
    (q k : Fin seq) :
    scoresRatOfSlice (seq := seq) (dModel := dModel) (dHead := dHead)
        scale maskValue maskCausal resid wq bq wk bk q k =
      let base :=
        baseScoreRatOfSlice (seq := seq) (dModel := dModel) (dHead := dHead)
          scale resid wq bq wk bk q k
      if maskCausal then
        if k ≤ q then
          base
        else
          maskValue
      else
        base := by
  rfl

end Induction

end Sound

end Nfp
