-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Vector.Defs
import Nfp.Circuit.Layers.Induction
import Nfp.Circuit.Layers.Softmax
import Nfp.Core.Basic
import Nfp.Model.InductionHead
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Linear.FinFold

/-!
Core definitions for induction-head certificates.

These definitions are shared across induction certificate builders and checkers.
-/

namespace Nfp

namespace Sound

open scoped BigOperators

open Nfp.Circuit
open Nfp.Sound.Bounds

variable {seq : Nat}

/-- Cached direction head for head inputs. -/
def dirHeadVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector Rat dHead :=
  Vector.ofFn (fun d : Fin dHead =>
    Linear.dotFin dModel (fun j => inputs.wo j d) (fun j => inputs.direction j))

/-- Real-valued LayerNorm outputs for head inputs. -/
noncomputable def lnRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dModel → Real :=
  fun q =>
    Bounds.layerNormReal inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)

/-- Unfolding lemma for `lnRealOfInputs`. -/
theorem lnRealOfInputs_def {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) (q : Fin seq) (i : Fin dModel) :
    lnRealOfInputs inputs q i =
      Bounds.layerNormReal inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q) i := rfl

/-- Real-valued query projections for head inputs. -/
noncomputable def qRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dHead → Real :=
  fun q d =>
    dotProduct (fun j => (inputs.wq j d : Real)) (lnRealOfInputs inputs q) + (inputs.bq d : Real)

/-- Unfolding lemma for `qRealOfInputs`. -/
theorem qRealOfInputs_def {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) (q : Fin seq) (d : Fin dHead) :
    qRealOfInputs inputs q d =
      dotProduct (fun j => (inputs.wq j d : Real)) (lnRealOfInputs inputs q) +
        (inputs.bq d : Real) := rfl

/-- Real-valued key projections for head inputs. -/
noncomputable def kRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dHead → Real :=
  fun q d =>
    dotProduct (fun j => (inputs.wk j d : Real)) (lnRealOfInputs inputs q) + (inputs.bk d : Real)

/-- Unfolding lemma for `kRealOfInputs`. -/
theorem kRealOfInputs_def {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) (q : Fin seq) (d : Fin dHead) :
    kRealOfInputs inputs q d =
      dotProduct (fun j => (inputs.wk j d : Real)) (lnRealOfInputs inputs q) +
        (inputs.bk d : Real) := rfl

/-- Real-valued value projections for head inputs. -/
noncomputable def vRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dHead → Real :=
  fun q d =>
    dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs q) + (inputs.bv d : Real)

/-- Unfolding lemma for `vRealOfInputs`. -/
theorem vRealOfInputs_def {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) (q : Fin seq) (d : Fin dHead) :
    vRealOfInputs inputs q d =
      dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs q) +
        (inputs.bv d : Real) := rfl

/-- Real-valued attention scores for head inputs. -/
noncomputable def scoresRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin seq → Real :=
  fun q k =>
    let base :=
      (inputs.scale : Real) *
        dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d)
    if inputs.maskCausal then
      if k ≤ q then
        base
      else
        (inputs.maskValue : Real)
    else
      base

/-- Unfolding lemma for `scoresRealOfInputs`. -/
theorem scoresRealOfInputs_def {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) (q k : Fin seq) :
    scoresRealOfInputs inputs q k =
      let base :=
        (inputs.scale : Real) *
          dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d)
      if inputs.maskCausal then
        if k ≤ q then
          base
        else
          (inputs.maskValue : Real)
      else
        base := rfl

/-- Real-valued per-key head outputs in model space. -/
noncomputable def headValueRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dModel → Real :=
  fun k i =>
    dotProduct (fun d => (inputs.wo i d : Real)) (fun d => vRealOfInputs inputs k d)

/-- Unfolding lemma for `headValueRealOfInputs`. -/
theorem headValueRealOfInputs_def {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) (k : Fin seq) (i : Fin dModel) :
    headValueRealOfInputs inputs k i =
      dotProduct (fun d => (inputs.wo i d : Real)) (fun d => vRealOfInputs inputs k d) := rfl

/-- Real-valued direction scores for head inputs. -/
noncomputable def valsRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Real :=
  let dirHead : Fin dHead → Real := fun d => (dirHeadVecOfInputs inputs).get d
  fun k => dotProduct dirHead (fun d => vRealOfInputs inputs k d)

/-- Unfolding lemma for `valsRealOfInputs`. -/
theorem valsRealOfInputs_def {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) (k : Fin seq) :
    valsRealOfInputs inputs k =
      let dirHead : Fin dHead → Real := fun d => (dirHeadVecOfInputs inputs).get d
      dotProduct dirHead (fun d => vRealOfInputs inputs k d) := rfl

/-- Interval data for direction values. -/
structure ValueInterval (seq : Nat) where
  /-- Lower bound for values. -/
  lo : Rat
  /-- Upper bound for values. -/
  hi : Rat
  /-- Lower bounds on per-key values. -/
  valsLo : Fin seq → Rat
  /-- Upper bounds on per-key values. -/
  valsHi : Fin seq → Rat
  /-- Optional logit-diff direction metadata (ignored by the checker). -/
  direction : Option DirectionSpec

/-- Soundness predicate for direction-value interval data. -/
structure ValueIntervalBounds {seq : Nat} (vals : Fin seq → Real)
    (c : ValueInterval seq) : Prop where
  /-- Interval endpoints are ordered. -/
  lo_le_hi : c.lo ≤ c.hi
  /-- `lo` is below every lower bound. -/
  lo_le_valsLo : ∀ k, (c.lo : Real) ≤ (c.valsLo k : Real)
  /-- Bounds sandwich the real values. -/
  vals_bounds :
    ∀ k, (c.valsLo k : Real) ≤ vals k ∧ vals k ≤ (c.valsHi k : Real)
  /-- `hi` is above every upper bound. -/
  valsHi_le_hi : ∀ k, (c.valsHi k : Real) ≤ (c.hi : Real)

/-- Split-budget knobs for sign-splitting bounds in induction-head certificates. -/
structure InductionHeadSplitConfig where
  /-- Split budget for query dims. -/
  splitBudgetQ : Nat
  /-- Split budget for key dims. -/
  splitBudgetK : Nat
  /-- Split budget for base diff dims. -/
  splitBudgetDiffBase : Nat
  /-- Split budget for refined diff dims. -/
  splitBudgetDiffRefined : Nat

/-- Default split budgets for induction-head sign-splitting bounds. -/
def defaultInductionHeadSplitConfig : InductionHeadSplitConfig :=
  { splitBudgetQ := 2
    splitBudgetK := 2
    splitBudgetDiffBase := 0
    splitBudgetDiffRefined := 12 }

/-- Unfolding lemma for `defaultInductionHeadSplitConfig`. -/
theorem defaultInductionHeadSplitConfig_def :
    defaultInductionHeadSplitConfig =
      { splitBudgetQ := 2
        splitBudgetK := 2
        splitBudgetDiffBase := 0
        splitBudgetDiffRefined := 12 } := rfl

/-- Sound induction-certificate payload built from exact head inputs. -/
structure InductionHeadCert (seq : Nat) where
  /-- Weight tolerance. -/
  eps : Rat
  /-- Per-query weight tolerance derived from local margins. -/
  epsAt : Fin seq → Rat
  /-- Per-key weight bounds derived from score gaps. -/
  weightBoundAt : Fin seq → Fin seq → Rat
  /-- Score margin used to justify the weight tolerance. -/
  margin : Rat
  /-- Active queries for which bounds are required. -/
  active : Finset (Fin seq)
  /-- `prev` selector for induction-style attention. -/
  prev : Fin seq → Fin seq
  /-- Value-interval certificate for the direction values. -/
  values : ValueInterval seq

/-- Soundness predicate for `InductionHeadCert`. -/
structure InductionHeadCertSound [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) (c : InductionHeadCert seq) : Prop where
  /-- Softmax weights respect the derived margin bounds. -/
  softmax_bounds :
    Layers.SoftmaxMarginBoundsOn (Val := Real) (c.eps : Real) (c.margin : Real)
      (fun q => q ∈ c.active) c.prev
      (scoresRealOfInputs inputs)
      (fun q k => Circuit.softmax (scoresRealOfInputs inputs q) k)
  /-- Per-query one-hot bounds derived from local margins. -/
  oneHot_bounds_at :
    ∀ q, q ∈ c.active →
      Layers.OneHotApproxBoundsOnActive (Val := Real) (c.epsAt q : Real)
        (fun q' => q' = q) c.prev
        (fun q' k => Circuit.softmax (scoresRealOfInputs inputs q') k)
  /-- Per-key weight bounds derived from local score gaps. -/
  weight_bounds_at :
    ∀ q, q ∈ c.active → ∀ k, k ≠ c.prev q →
      Circuit.softmax (scoresRealOfInputs inputs q) k ≤ (c.weightBoundAt q k : Real)
  /-- Interval bounds hold for the direction values. -/
  value_bounds :
    ValueIntervalBounds (vals := valsRealOfInputs inputs) c.values

end Sound

end Nfp
