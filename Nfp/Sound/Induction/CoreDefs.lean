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
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector Dyadic dHead :=
  Vector.ofFn (fun d : Fin dHead =>
    Linear.dotFin dModel (fun j => inputs.wo j d) (fun j => inputs.direction j))

/-- Real-valued LayerNorm outputs for head inputs. -/
noncomputable def lnRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dModel → Real :=
  fun q =>
    Bounds.layerNormReal inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)

/-- Real-valued query projections for head inputs. -/
noncomputable def qRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dHead → Real :=
  fun q d =>
    dotProduct (fun j => (inputs.wq j d : Real)) (lnRealOfInputs inputs q) + (inputs.bq d : Real)

/-- Real-valued key projections for head inputs. -/
noncomputable def kRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dHead → Real :=
  fun q d =>
    dotProduct (fun j => (inputs.wk j d : Real)) (lnRealOfInputs inputs q) + (inputs.bk d : Real)

/-- Real-valued value projections for head inputs. -/
noncomputable def vRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dHead → Real :=
  fun q d =>
    dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs q) + (inputs.bv d : Real)

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

/-- Real-valued per-key head outputs in model space. -/
noncomputable def headValueRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dModel → Real :=
  fun k i =>
    dotProduct (fun d => (inputs.wo i d : Real)) (fun d => vRealOfInputs inputs k d)

/-- Real-valued direction scores for head inputs. -/
noncomputable def valsRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Real :=
  let dirHead : Fin dHead → Real := fun d => (dirHeadVecOfInputs inputs).get d
  fun k => dotProduct dirHead (fun d => vRealOfInputs inputs k d)

/-- Interval data for direction values. -/
structure ValueInterval (seq : Nat) where
  /-- Lower bound for values. -/
  lo : Dyadic
  /-- Upper bound for values. -/
  hi : Dyadic
  /-- Lower bounds on per-key values. -/
  valsLo : Fin seq → Dyadic
  /-- Upper bounds on per-key values. -/
  valsHi : Fin seq → Dyadic
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

/-- Sound induction-certificate payload built from exact head inputs. -/
structure InductionHeadCert (seq : Nat) where
  /-- Weight tolerance. -/
  eps : Dyadic
  /-- Per-query weight tolerance derived from local margins. -/
  epsAt : Fin seq → Dyadic
  /-- Score margin used to justify the weight tolerance. -/
  margin : Dyadic
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
  /-- Interval bounds hold for the direction values. -/
  value_bounds :
    ValueIntervalBounds (vals := valsRealOfInputs inputs) c.values

end Sound

end Nfp
