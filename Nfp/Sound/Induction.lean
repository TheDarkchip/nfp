-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Data.Vector.Defs
import Nfp.Circuit.Cert.ResidualInterval
import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange
import Nfp.Circuit.Layers.Softmax
import Nfp.Model.InductionHead
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Induction.OneHot
import Nfp.Sound.Linear.FinFold

/-!
Sound builders for induction certificates.

These builders recompute certificate bounds inside Lean from exact inputs and
return proof-carrying results. The head-input path derives softmax tolerances
from score margins rather than trusting external weight dumps.
-/

namespace Nfp

namespace Sound

open scoped BigOperators

open Nfp.Circuit
open Nfp.Sound.Bounds

variable {seq : Nat}

/-- Cached direction head for head inputs. -/
private def dirHeadVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector Rat dHead :=
  Vector.ofFn (fun d : Fin dHead =>
    Linear.dotFin dModel (fun j => inputs.wo j d) (fun j => inputs.direction j))

/-- Real-valued LayerNorm outputs for head inputs. -/
private noncomputable def lnRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dModel → Real :=
  fun q =>
    Bounds.layerNormReal inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)

/-- Real-valued query projections for head inputs. -/
private noncomputable def qRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dHead → Real :=
  fun q d =>
    dotProduct (fun j => (inputs.wq j d : Real)) (lnRealOfInputs inputs q) + (inputs.bq d : Real)

/-- Real-valued key projections for head inputs. -/
private noncomputable def kRealOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin seq → Fin dHead → Real :=
  fun q d =>
    dotProduct (fun j => (inputs.wk j d : Real)) (lnRealOfInputs inputs q) + (inputs.bk d : Real)

/-- Real-valued value projections for head inputs. -/
private noncomputable def vRealOfInputs {seq dModel dHead : Nat}
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
private noncomputable def headValueRealOfInputs {seq dModel dHead : Nat}
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

/-- Sound induction-certificate payload built from exact head inputs. -/
structure InductionHeadCert (seq : Nat) where
  /-- Weight tolerance. -/
  eps : Rat
  /-- Per-query weight tolerance derived from local margins. -/
  epsAt : Fin seq → Rat
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
  /-- Interval bounds hold for the direction values. -/
  value_bounds :
    ValueIntervalBounds (vals := valsRealOfInputs inputs) c.values

/-- Build and certify a softmax-margin certificate from exact scores/weights. -/
def buildSoftmaxMarginCert? [NeZero seq]
    (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (scores : Fin seq → Fin seq → Rat)
    (weights : Fin seq → Fin seq → Rat) :
    Option {c : SoftmaxMarginCert seq // checkSoftmaxMarginCert c = true} := by
  classical
  let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (prev q)
  let epsAt : Fin seq → Rat := fun q =>
    let other := otherKeys q
    let maxOther :=
      if h : other.Nonempty then
        other.sup' h (fun k => weights q k)
      else
        (0 : Rat)
    let deficit := (1 : Rat) - weights q (prev q)
    max maxOther deficit
  let marginAt : Fin seq → Rat := fun q =>
    let other := otherKeys q
    if h : other.Nonempty then
      other.inf' h (fun k => scores q (prev q) - scores q k)
    else
      (0 : Rat)
  let eps :=
    if h : active.Nonempty then
      active.sup' h epsAt
    else
      (0 : Rat)
  let margin :=
    if h : active.Nonempty then
      active.inf' h marginAt
    else
      (0 : Rat)
  let cert : SoftmaxMarginCert seq :=
    { eps := eps
      margin := margin
      active := active
      prev := prev
      scores := scores
      weights := weights }
  if h : checkSoftmaxMarginCert cert = true then
    exact some ⟨cert, h⟩
  else
    exact none

/-- Build and certify a value-range certificate from exact values. -/
def buildValueRangeCert? [NeZero seq]
    (vals : Fin seq → Rat)
    (direction : Option DirectionSpec) :
    Option {c : ValueRangeCert seq // checkValueRangeCert c = true} := by
  classical
  let _ : Nonempty (Fin seq) := by
    refine ⟨⟨0, ?_⟩⟩
    exact Nat.pos_of_ne_zero (NeZero.ne seq)
  let univ : Finset (Fin seq) := Finset.univ
  let hnonempty : univ.Nonempty := Finset.univ_nonempty
  let lo := univ.inf' hnonempty vals
  let hi := univ.sup' hnonempty vals
  let cert : ValueRangeCert seq :=
    { lo := lo
      hi := hi
      vals := vals
      direction := direction }
  if h : checkValueRangeCert cert = true then
    exact some ⟨cert, h⟩
  else
    exact none

/-- Build and certify induction certificates from exact head inputs. -/
def buildInductionCertFromHead? [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option {c : InductionHeadCert seq // InductionHeadCertSound inputs c} := by
  classical
  by_cases hEps : 0 < inputs.lnEps
  · by_cases hmodel : dModel = 0
    · exact none
    · let lnBounds : Fin seq → (Fin dModel → Rat) × (Fin dModel → Rat) := fun q =>
        Bounds.layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
      let lnLo : Fin seq → Fin dModel → Rat := fun q => (lnBounds q).1
      let lnHi : Fin seq → Fin dModel → Rat := fun q => (lnBounds q).2
      let qLo : Fin seq → Fin dHead → Rat := fun q d =>
        dotIntervalLower (fun j => inputs.wq j d) (lnLo q) (lnHi q) + inputs.bq d
      let qHi : Fin seq → Fin dHead → Rat := fun q d =>
        dotIntervalUpper (fun j => inputs.wq j d) (lnLo q) (lnHi q) + inputs.bq d
      let kLo : Fin seq → Fin dHead → Rat := fun q d =>
        dotIntervalLower (fun j => inputs.wk j d) (lnLo q) (lnHi q) + inputs.bk d
      let kHi : Fin seq → Fin dHead → Rat := fun q d =>
        dotIntervalUpper (fun j => inputs.wk j d) (lnLo q) (lnHi q) + inputs.bk d
      let vLo : Fin seq → Fin dHead → Rat := fun q d =>
        dotIntervalLower (fun j => inputs.wv j d) (lnLo q) (lnHi q) + inputs.bv d
      let vHi : Fin seq → Fin dHead → Rat := fun q d =>
        dotIntervalUpper (fun j => inputs.wv j d) (lnLo q) (lnHi q) + inputs.bv d
      let qAbs : Fin seq → Fin dHead → Rat := fun q d =>
        max |qLo q d| |qHi q d|
      let kAbs : Fin seq → Fin dHead → Rat := fun q d =>
        max |kLo q d| |kHi q d|
      let masked : Fin seq → Fin seq → Prop := fun q k =>
        inputs.maskCausal ∧ q < k
      let dotAbs : Fin seq → Fin seq → Rat := fun q k =>
        dotProduct (fun d => qAbs q d) (fun d => kAbs k d)
      let scoreBaseAbs : Fin seq → Fin seq → Rat := fun q k =>
        |inputs.scale| * dotAbs q k
      let scoreAbs : Fin seq → Fin seq → Rat := fun q k =>
        if masked q k then |inputs.maskValue| else scoreBaseAbs q k
      let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
        if masked q k then inputs.maskValue else -scoreBaseAbs q k
      let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
        if masked q k then inputs.maskValue else scoreBaseAbs q k
      let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
        (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
      let marginAt : Fin seq → Rat := fun q =>
        let other := otherKeys q
        if h : other.Nonempty then
          other.inf' h (fun k => scoreLo q (inputs.prev q) - scoreHi q k)
        else
          (0 : Rat)
      let epsAt : Fin seq → Rat := fun q =>
        if marginAt q < 0 then
          (1 : Rat)
        else
          (seq - 1 : Rat) / (1 + marginAt q)
      let margin : Rat :=
        if h : inputs.active.Nonempty then
          inputs.active.inf' h marginAt
        else
          (0 : Rat)
      let eps : Rat :=
        if margin < 0 then
          (1 : Rat)
        else
          (seq - 1 : Rat) / (1 + margin)
      have hseq : (1 : Nat) ≤ seq :=
        Nat.succ_le_iff.mpr (Nat.pos_of_ne_zero (NeZero.ne seq))
      let dirHeadVec := dirHeadVecOfInputs inputs
      let dirHead : Fin dHead → Rat := fun d => dirHeadVec.get d
      let valsLo : Fin seq → Rat := fun k =>
        dotIntervalLower dirHead (vLo k) (vHi k)
      let valsHi : Fin seq → Rat := fun k =>
        dotIntervalUpper dirHead (vLo k) (vHi k)
      let univ : Finset (Fin seq) := Finset.univ
      have hnonempty : univ.Nonempty := by simp [univ]
      let lo := univ.inf' hnonempty valsLo
      let hi := univ.sup' hnonempty valsHi
      let valCert : ValueInterval seq :=
        { lo := lo
          hi := hi
          valsLo := valsLo
          valsHi := valsHi
          direction := some inputs.directionSpec }
      let cert : InductionHeadCert seq :=
        { eps := eps
          epsAt := epsAt
          margin := margin
          active := inputs.active
          prev := inputs.prev
          values := valCert }
      have hln_bounds :
          ∀ q i, (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
            lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
        intro q i
        have hln :=
          Bounds.layerNormBounds_spec (eps := inputs.lnEps)
            (gamma := inputs.ln1Gamma) (beta := inputs.ln1Beta)
            (x := inputs.embed q) hmodel hEps
        simpa [lnBounds, lnLo, lnHi, lnRealOfInputs] using hln i
      have hq_bounds :
          ∀ q d, (qLo q d : Real) ≤ qRealOfInputs inputs q d ∧
            qRealOfInputs inputs q d ≤ (qHi q d : Real) := by
        intro q d
        have hln := hln_bounds q
        have hlo : ∀ j, (lnLo q j : Real) ≤ lnRealOfInputs inputs q j := fun j => (hln j).1
        have hhi : ∀ j, lnRealOfInputs inputs q j ≤ (lnHi q j : Real) := fun j => (hln j).2
        have hlow :=
          dotIntervalLower_le_dotProduct_real (v := fun j => inputs.wq j d)
            (lo := lnLo q) (hi := lnHi q) (x := lnRealOfInputs inputs q) hlo hhi
        have hhigh :=
          dotProduct_le_dotIntervalUpper_real (v := fun j => inputs.wq j d)
            (lo := lnLo q) (hi := lnHi q) (x := lnRealOfInputs inputs q) hlo hhi
        have hlow' := add_le_add_right hlow (inputs.bq d : Real)
        have hhigh' := add_le_add_right hhigh (inputs.bq d : Real)
        constructor
        · simpa [qLo, qRealOfInputs, Rat.cast_add] using hlow'
        · simpa [qHi, qRealOfInputs, Rat.cast_add] using hhigh'
      have hk_bounds :
          ∀ q d, (kLo q d : Real) ≤ kRealOfInputs inputs q d ∧
            kRealOfInputs inputs q d ≤ (kHi q d : Real) := by
        intro q d
        have hln := hln_bounds q
        have hlo : ∀ j, (lnLo q j : Real) ≤ lnRealOfInputs inputs q j := fun j => (hln j).1
        have hhi : ∀ j, lnRealOfInputs inputs q j ≤ (lnHi q j : Real) := fun j => (hln j).2
        have hlow :=
          dotIntervalLower_le_dotProduct_real (v := fun j => inputs.wk j d)
            (lo := lnLo q) (hi := lnHi q) (x := lnRealOfInputs inputs q) hlo hhi
        have hhigh :=
          dotProduct_le_dotIntervalUpper_real (v := fun j => inputs.wk j d)
            (lo := lnLo q) (hi := lnHi q) (x := lnRealOfInputs inputs q) hlo hhi
        have hlow' := add_le_add_right hlow (inputs.bk d : Real)
        have hhigh' := add_le_add_right hhigh (inputs.bk d : Real)
        constructor
        · simpa [kLo, kRealOfInputs, Rat.cast_add] using hlow'
        · simpa [kHi, kRealOfInputs, Rat.cast_add] using hhigh'
      have hv_bounds :
          ∀ q d, (vLo q d : Real) ≤ vRealOfInputs inputs q d ∧
            vRealOfInputs inputs q d ≤ (vHi q d : Real) := by
        intro q d
        have hln := hln_bounds q
        have hlo : ∀ j, (lnLo q j : Real) ≤ lnRealOfInputs inputs q j := fun j => (hln j).1
        have hhi : ∀ j, lnRealOfInputs inputs q j ≤ (lnHi q j : Real) := fun j => (hln j).2
        have hlow :=
          dotIntervalLower_le_dotProduct_real (v := fun j => inputs.wv j d)
            (lo := lnLo q) (hi := lnHi q) (x := lnRealOfInputs inputs q) hlo hhi
        have hhigh :=
          dotProduct_le_dotIntervalUpper_real (v := fun j => inputs.wv j d)
            (lo := lnLo q) (hi := lnHi q) (x := lnRealOfInputs inputs q) hlo hhi
        have hlow' := add_le_add_right hlow (inputs.bv d : Real)
        have hhigh' := add_le_add_right hhigh (inputs.bv d : Real)
        constructor
        · simpa [vLo, vRealOfInputs, Rat.cast_add] using hlow'
        · simpa [vHi, vRealOfInputs, Rat.cast_add] using hhigh'
      have hscore_bounds :
          ∀ q k, (scoreLo q k : Real) ≤ scoresRealOfInputs inputs q k ∧
            scoresRealOfInputs inputs q k ≤ (scoreHi q k : Real) := by
        intro q k
        let scoresReal := scoresRealOfInputs inputs
        let base :=
          (inputs.scale : Real) *
            dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d)
        have hq_abs : ∀ d, |qRealOfInputs inputs q d| ≤ (qAbs q d : Real) := by
          intro d
          have hq := hq_bounds q d
          have h := abs_le_max_abs_abs_of_interval_real hq.1 hq.2
          simpa [qAbs] using h
        have hk_abs : ∀ d, |kRealOfInputs inputs k d| ≤ (kAbs k d : Real) := by
          intro d
          have hk := hk_bounds k d
          have h := abs_le_max_abs_abs_of_interval_real hk.1 hk.2
          simpa [kAbs] using h
        have hdot_abs :
            |dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d)| ≤
              (dotAbs q k : Real) := by
          have hsum :
              |∑ d, qRealOfInputs inputs q d * kRealOfInputs inputs k d| ≤
                ∑ d, |qRealOfInputs inputs q d * kRealOfInputs inputs k d| := by
            simpa [dotProduct] using
              (Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset (Fin dHead)))
                (f := fun d => qRealOfInputs inputs q d * kRealOfInputs inputs k d))
          have hterm :
              ∀ d,
                |qRealOfInputs inputs q d * kRealOfInputs inputs k d| ≤
                  (qAbs q d : Real) * (kAbs k d : Real) := by
            intro d
            have hq := hq_abs d
            have hk := hk_abs d
            have hqnonneg : 0 ≤ (qAbs q d : Real) := by
              have hqnonneg' : 0 ≤ qAbs q d := by
                have h1 : 0 ≤ |qLo q d| := abs_nonneg (qLo q d)
                exact le_trans h1 (le_max_left _ _)
              exact_mod_cast hqnonneg'
            calc
              |qRealOfInputs inputs q d * kRealOfInputs inputs k d| =
                  |qRealOfInputs inputs q d| * |kRealOfInputs inputs k d| := by
                    simp [abs_mul]
              _ ≤ (qAbs q d : Real) * (kAbs k d : Real) :=
                mul_le_mul hq hk (abs_nonneg _) hqnonneg
          have hsum_le :
              ∑ d, |qRealOfInputs inputs q d * kRealOfInputs inputs k d| ≤
                ∑ d, (qAbs q d : Real) * (kAbs k d : Real) := by
            refine Finset.sum_le_sum ?_
            intro d _
            exact hterm d
          have hcast :
              (dotAbs q k : Real) =
                ∑ d, (qAbs q d : Real) * (kAbs k d : Real) := by
            simp [dotAbs, dotProduct]
          have hfinal := hsum.trans (hsum_le.trans_eq hcast.symm)
          simpa [dotProduct] using hfinal
        have hscale_abs : 0 ≤ (|inputs.scale| : Real) := by
          exact_mod_cast (abs_nonneg (inputs.scale))
        have hbase_abs :
            |base| ≤ (scoreBaseAbs q k : Real) := by
          have hdot_abs' := hdot_abs
          have hmul :
              |base| =
                (|inputs.scale| : Real) *
                  |dotProduct (fun d => qRealOfInputs inputs q d)
                      (fun d => kRealOfInputs inputs k d)| := by
            simp [base, abs_mul]
          have hmul_le :
              (|inputs.scale| : Real) *
                  |dotProduct (fun d => qRealOfInputs inputs q d)
                      (fun d => kRealOfInputs inputs k d)| ≤
                (|inputs.scale| : Real) * (dotAbs q k : Real) := by
            exact mul_le_mul_of_nonneg_left hdot_abs' hscale_abs
          simpa [scoreBaseAbs, hmul] using hmul_le
        by_cases hcausal : inputs.maskCausal
        · by_cases hle : k ≤ q
          · have hnot : ¬ q < k := not_lt_of_ge hle
            have hscore_eq : scoresReal q k = base := by
              simp [scoresReal, scoresRealOfInputs, hcausal, hle, base]
            have hscore_abs' : |scoresReal q k| ≤ (scoreBaseAbs q k : Real) := by
              simpa [hscore_eq] using hbase_abs
            have hscore_abs :
                |scoresReal q k| ≤ (scoreAbs q k : Real) := by
              simpa [scoreAbs, masked, hcausal, hnot] using hscore_abs'
            have hscore_bounds := (abs_le).1 hscore_abs
            constructor
            · simpa [scoresReal, scoreLo, scoreAbs, masked, hcausal, hnot]
                using hscore_bounds.1
            · simpa [scoresReal, scoreHi, scoreAbs, masked, hcausal, hnot]
                using hscore_bounds.2
          · have hlt : q < k := lt_of_not_ge hle
            constructor
            · simp [scoresRealOfInputs, scoreLo, masked, hcausal, hle, hlt]
            · simp [scoresRealOfInputs, scoreHi, masked, hcausal, hle, hlt]
        · have hscore_eq : scoresReal q k = base := by
            simp [scoresReal, scoresRealOfInputs, hcausal, base]
          have hscore_abs' : |scoresReal q k| ≤ (scoreBaseAbs q k : Real) := by
            simpa [hscore_eq] using hbase_abs
          have hscore_abs :
              |scoresReal q k| ≤ (scoreAbs q k : Real) := by
            simpa [scoreAbs, masked, hcausal] using hscore_abs'
          have hscore_bounds := (abs_le).1 hscore_abs
          constructor
          · simpa [scoresReal, scoreLo, scoreAbs, masked, hcausal]
              using hscore_bounds.1
          · simpa [scoresReal, scoreHi, scoreAbs, masked, hcausal]
              using hscore_bounds.2
      let scoresReal := scoresRealOfInputs inputs
      let weights : Fin seq → Fin seq → Real := fun q k =>
        Circuit.softmax (scoresReal q) k
      let others : Fin seq → Finset (Fin seq) := fun q =>
        (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
      have hscore_margin_real :
          ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
            scoresReal q k + (margin : Real) ≤ scoresReal q (inputs.prev q) := by
        intro q hq k hk
        by_cases hactive : inputs.active.Nonempty
        · have hmargin_le : margin ≤ marginAt q := by
            have hle : margin ≤ inputs.active.inf' hactive marginAt := by
              simp [margin, hactive]
            have hle_all :=
              (Finset.le_inf'_iff (s := inputs.active) (H := hactive) (f := marginAt)
                (a := margin)).1 hle
            exact hle_all q hq
          have hother : (otherKeys q).Nonempty := ⟨k, by simp [otherKeys, hk]⟩
          have hgap_le :
              marginAt q ≤ scoreLo q (inputs.prev q) - scoreHi q k := by
            have hle : marginAt q ≤
                (otherKeys q).inf' hother
                  (fun k => scoreLo q (inputs.prev q) - scoreHi q k) := by
              simp [marginAt, hother]
            have hle_all :=
              (Finset.le_inf'_iff (s := otherKeys q) (H := hother)
                (f := fun k => scoreLo q (inputs.prev q) - scoreHi q k)
                (a := marginAt q)).1 hle
            exact hle_all k (by simp [otherKeys, hk])
          have hgap : margin ≤ scoreLo q (inputs.prev q) - scoreHi q k :=
            le_trans hmargin_le hgap_le
          have hgap_real : (margin : Real) ≤
              (scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real) := by
            have hgap_real' :
                (margin : Real) ≤ ((scoreLo q (inputs.prev q) - scoreHi q k : Rat) : Real) :=
              (Rat.cast_le (K := Real)).2 hgap
            simpa [Rat.cast_sub] using hgap_real'
          have hk_bounds := hscore_bounds q k
          have hprev_bounds := hscore_bounds q (inputs.prev q)
          have h1 :
              scoresReal q k + (margin : Real) ≤
                scoresReal q k + ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) := by
            exact (add_le_add_iff_left (scoresReal q k)).2 hgap_real
          have h2 :
              scoresReal q k + ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) ≤
                (scoreLo q (inputs.prev q) : Real) := by
            have hscore_le' :
                scoresReal q k + ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) ≤
                  (scoreHi q k : Real) +
                    ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) := by
              exact (add_le_add_iff_right
                ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real))).2 hk_bounds.2
            calc
              scoresReal q k + ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) ≤
                  (scoreHi q k : Real) +
                    ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) := by
                exact hscore_le'
              _ = (scoreLo q (inputs.prev q) : Real) := by
                exact add_sub_cancel (scoreHi q k : Real) (scoreLo q (inputs.prev q) : Real)
          have h3 :
              scoresReal q k + (margin : Real) ≤ (scoreLo q (inputs.prev q) : Real) :=
            h1.trans h2
          exact h3.trans hprev_bounds.1
        · exact (hactive ⟨q, hq⟩).elim
      have hsoftmax_bounds :
          Layers.SoftmaxMarginBoundsOn (Val := Real) (eps : Real) (margin : Real)
            (fun q => q ∈ inputs.active) inputs.prev scoresReal weights := by
        classical
        refine
          { score_margin := ?_
            nonneg := ?_
            sum_one := ?_
            prev_large := ?_
            other_le := ?_ }
        · intro q hq k hk
          exact hscore_margin_real q hq k hk
        · intro q _ k
          simpa [weights] using
            (Circuit.softmax_nonneg (scores := scoresReal q) k)
        · intro q _
          simpa [weights] using
            (Circuit.softmax_sum_one (scores := scoresReal q))
        · intro q hq
          have hsum_others_le : (∑ k ∈ others q, weights q k) ≤ (eps : Real) := by
            by_cases hneg : margin < 0
            · have heps : (eps : Real) = 1 := by
                simp [eps, hneg]
              have hsubset : others q ⊆ (Finset.univ : Finset (Fin seq)) := by
                intro k hk
                simp
              have hnonneg :
                  ∀ k ∈ (Finset.univ : Finset (Fin seq)), 0 ≤ weights q k := by
                intro k _
                simpa [weights] using
                  (Circuit.softmax_nonneg (scores := scoresReal q) k)
              have hsum_le :
                  (∑ k ∈ others q, weights q k) ≤
                    ∑ k ∈ (Finset.univ : Finset (Fin seq)), weights q k :=
                Finset.sum_le_sum_of_subset_of_nonneg hsubset (by
                  intro k hk _; exact hnonneg k hk)
              have hsum_one : (∑ k, weights q k) = 1 := by
                simpa [weights] using
                  (Circuit.softmax_sum_one (scores := scoresReal q))
              have hsum_le' : (∑ k ∈ others q, weights q k) ≤ 1 := by
                simpa [hsum_one] using hsum_le
              simpa [heps] using hsum_le'
            · have hnonneg : 0 ≤ margin := le_of_not_gt hneg
              have hnonneg_real : 0 ≤ (margin : Real) := by
                exact (Rat.cast_nonneg (K := Real)).2 hnonneg
              have hbound :
                  ∀ k ∈ others q,
                    weights q k ≤ (1 + (margin : Real))⁻¹ := by
                intro k hk
                have hkne : k ≠ inputs.prev q := (Finset.mem_erase.mp hk).1
                have hscore := hscore_margin_real q hq k hkne
                simpa [weights] using
                  (Circuit.softmax_other_le_inv_one_add (scores := scoresReal q)
                    (prev := inputs.prev q) (k := k) (m := (margin : Real))
                    hnonneg_real hscore)
              have hsum_le :
                  (∑ k ∈ others q, weights q k) ≤
                    ∑ k ∈ others q, (1 + (margin : Real))⁻¹ :=
                Finset.sum_le_sum hbound
              have hsum_const :
                  (∑ k ∈ others q, (1 + (margin : Real))⁻¹) =
                    (others q).card * (1 + (margin : Real))⁻¹ := by
                simp
              have hcard : (others q).card = seq - 1 := by
                simp [others, Finset.card_erase_of_mem]
              have hsum_le' :
                  (∑ k ∈ others q, weights q k) ≤
                    (seq - 1 : Real) * (1 + (margin : Real))⁻¹ := by
                have hsum_le'' := hsum_le.trans_eq hsum_const
                have hsum_le''' := hsum_le''
                simp only [hcard, Nat.cast_sub hseq, Nat.cast_one] at hsum_le'''
                exact hsum_le'''
              have heps :
                  (eps : Real) = (seq - 1 : Real) * (1 + (margin : Real))⁻¹ := by
                simp [eps, hneg, Rat.cast_add, div_eq_mul_inv]
              simpa [heps] using hsum_le'
          have hsum_eq :
              weights q (inputs.prev q) + ∑ k ∈ others q, weights q k = 1 := by
            have hsum' :
                weights q (inputs.prev q) + ∑ k ∈ others q, weights q k =
                  ∑ k, weights q k := by
              simp [others]
            have hsum_one : (∑ k, weights q k) = 1 := by
              simpa [weights] using
                (Circuit.softmax_sum_one (scores := scoresReal q))
            calc
              weights q (inputs.prev q) + ∑ k ∈ others q, weights q k =
                  ∑ k, weights q k := hsum'
              _ = 1 := hsum_one
          have hsum_le' :
              weights q (inputs.prev q) + ∑ k ∈ others q, weights q k ≤
                weights q (inputs.prev q) + (eps : Real) := by
            have hsum_le'' := add_le_add_left hsum_others_le (weights q (inputs.prev q))
            simpa [add_comm, add_left_comm, add_assoc] using hsum_le''
          have hprev :
              1 ≤ weights q (inputs.prev q) + (eps : Real) := by
            simpa [hsum_eq] using hsum_le'
          exact hprev
        · intro q hq k hk
          have hsum_others_le : (∑ j ∈ others q, weights q j) ≤ (eps : Real) := by
            by_cases hneg : margin < 0
            · have heps : (eps : Real) = 1 := by
                simp [eps, hneg]
              have hsubset : others q ⊆ (Finset.univ : Finset (Fin seq)) := by
                intro j hj
                simp
              have hnonneg :
                  ∀ j ∈ (Finset.univ : Finset (Fin seq)), 0 ≤ weights q j := by
                intro j _
                simpa [weights] using
                  (Circuit.softmax_nonneg (scores := scoresReal q) j)
              have hsum_le :
                  (∑ j ∈ others q, weights q j) ≤
                    ∑ j ∈ (Finset.univ : Finset (Fin seq)), weights q j :=
                Finset.sum_le_sum_of_subset_of_nonneg hsubset (by
                  intro j hj _; exact hnonneg j hj)
              have hsum_one : (∑ j, weights q j) = 1 := by
                simpa [weights] using
                  (Circuit.softmax_sum_one (scores := scoresReal q))
              have hsum_le' : (∑ j ∈ others q, weights q j) ≤ 1 := by
                simpa [hsum_one] using hsum_le
              simpa [heps] using hsum_le'
            · have hnonneg : 0 ≤ margin := le_of_not_gt hneg
              have hnonneg_real : 0 ≤ (margin : Real) := by
                exact (Rat.cast_nonneg (K := Real)).2 hnonneg
              have hbound :
                  ∀ j ∈ others q,
                    weights q j ≤ (1 + (margin : Real))⁻¹ := by
                intro j hj
                have hjne : j ≠ inputs.prev q := (Finset.mem_erase.mp hj).1
                have hscore := hscore_margin_real q hq j hjne
                simpa [weights] using
                  (Circuit.softmax_other_le_inv_one_add (scores := scoresReal q)
                    (prev := inputs.prev q) (k := j) (m := (margin : Real))
                    hnonneg_real hscore)
              have hsum_le :
                  (∑ j ∈ others q, weights q j) ≤
                    ∑ j ∈ others q, (1 + (margin : Real))⁻¹ :=
                Finset.sum_le_sum hbound
              have hsum_const :
                  (∑ j ∈ others q, (1 + (margin : Real))⁻¹) =
                    (others q).card * (1 + (margin : Real))⁻¹ := by
                simp
              have hcard : (others q).card = seq - 1 := by
                simp [others, Finset.card_erase_of_mem]
              have hsum_le' :
                  (∑ j ∈ others q, weights q j) ≤
                    (seq - 1 : Real) * (1 + (margin : Real))⁻¹ := by
                have hsum_le'' := hsum_le.trans_eq hsum_const
                have hsum_le''' := hsum_le''
                simp only [hcard, Nat.cast_sub hseq, Nat.cast_one] at hsum_le'''
                exact hsum_le'''
              have heps :
                  (eps : Real) = (seq - 1 : Real) * (1 + (margin : Real))⁻¹ := by
                simp [eps, hneg, Rat.cast_add, div_eq_mul_inv]
              simpa [heps] using hsum_le'
          have hk' : k ∈ others q := by
            simp [others, hk]
          have hnonneg :
              ∀ j ∈ others q, 0 ≤ weights q j := by
            intro j _
            simpa [weights] using
              (Circuit.softmax_nonneg (scores := scoresReal q) j)
          have hle :
              weights q k ≤ ∑ j ∈ others q, weights q j := by
            have h := Finset.single_le_sum hnonneg hk'
            simpa using h
          exact hle.trans hsum_others_le
      have hscore_margin_real_at :
          ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
            scoresReal q k + (marginAt q : Real) ≤ scoresReal q (inputs.prev q) := by
        intro q hq k hk
        let other := otherKeys q
        have hother : other.Nonempty := by
          refine ⟨k, ?_⟩
          simp [other, otherKeys, hk]
        have hgap_le :
            marginAt q ≤ scoreLo q (inputs.prev q) - scoreHi q k := by
          have hkmem : k ∈ other := by
            simp [other, otherKeys, hk]
          have hle :
              other.inf' hother (fun k => scoreLo q (inputs.prev q) - scoreHi q k) ≤
                scoreLo q (inputs.prev q) - scoreHi q k := by
            exact (Finset.inf'_le (s := other) (f := fun k =>
              scoreLo q (inputs.prev q) - scoreHi q k) (b := k) hkmem)
          have hmarginAt :
              marginAt q =
                other.inf' hother (fun k => scoreLo q (inputs.prev q) - scoreHi q k) := by
            simp [marginAt, hother, other]
          simpa [hmarginAt] using hle
        have hgap_real :
            (marginAt q : Real) ≤
              (scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real) := by
          have hgap_real' :
              (marginAt q : Real) ≤
                ((scoreLo q (inputs.prev q) - scoreHi q k : Rat) : Real) :=
            (Rat.cast_le (K := Real)).2 hgap_le
          simpa [Rat.cast_sub] using hgap_real'
        have hk_bounds := hscore_bounds q k
        have hprev_bounds := hscore_bounds q (inputs.prev q)
        have h1 :
            scoresReal q k + (marginAt q : Real) ≤
              (scoreHi q k : Real) + (marginAt q : Real) := by
          have h1' := add_le_add_right hk_bounds.2 (marginAt q : Real)
          simpa [scoresReal] using h1'
        have h2 :
            (scoreHi q k : Real) + (marginAt q : Real) ≤
              (scoreLo q (inputs.prev q) : Real) := by
          have hgap_real' :
              (scoreHi q k : Real) + (marginAt q : Real) ≤
                (scoreHi q k : Real) +
                  ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) := by
            exact add_le_add_right hgap_real (scoreHi q k : Real)
          have hgap_real'' :
              (scoreHi q k : Real) +
                  ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) =
                (scoreLo q (inputs.prev q) : Real) := by
            calc
              (scoreHi q k : Real) +
                  ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) =
                  ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) +
                    (scoreHi q k : Real) := by
                exact add_comm _ _
              _ = (scoreLo q (inputs.prev q) : Real) := by
                exact sub_add_cancel (scoreLo q (inputs.prev q) : Real) (scoreHi q k : Real)
          exact hgap_real'.trans (le_of_eq hgap_real'')
        have h3 :
            scoresReal q k + (marginAt q : Real) ≤
              (scoreLo q (inputs.prev q) : Real) := h1.trans h2
        exact h3.trans hprev_bounds.1
      have hepsAt :
          ∀ q, epsAt q =
            if marginAt q < 0 then (1 : Rat) else (seq - 1 : Rat) / (1 + marginAt q) := by
        intro q
        simp [epsAt]
      have oneHot_bounds_at :
          ∀ q, q ∈ inputs.active →
            Layers.OneHotApproxBoundsOnActive (Val := Real) (epsAt q : Real)
              (fun q' => q' = q) inputs.prev weights := by
        intro q hq
        exact
          Sound.oneHot_bounds_at_of_marginAt
            (active := inputs.active)
            (prev := inputs.prev)
            (scoresReal := scoresReal)
            (marginAt := marginAt)
            (epsAt := epsAt)
            (hepsAt := hepsAt)
            (hseq := hseq)
            (hscore_margin_real_at := hscore_margin_real_at)
            q hq
      have hvals_bounds :
          ValueIntervalBounds (vals := valsRealOfInputs inputs) valCert := by
        refine
          { lo_le_hi := ?_
            lo_le_valsLo := ?_
            vals_bounds := ?_
            valsHi_le_hi := ?_ }
        · rcases (Finset.univ_nonempty : univ.Nonempty) with ⟨k0, hk0⟩
          have hmem0 : k0 ∈ univ := hk0
          have hlo : (valCert.lo : Real) ≤ (valCert.valsLo k0 : Real) := by
            have hloRat : valCert.lo ≤ valCert.valsLo k0 := by
              change lo ≤ valsLo k0
              dsimp [lo]
              refine (Finset.inf'_le_iff (s := univ) (H := hnonempty)
                (f := valsLo) (a := valsLo k0)).2 ?_
              refine ⟨k0, hmem0, ?_⟩
              exact le_rfl
            exact (Rat.cast_le (K := Real)).2 hloRat
          have hvals :
              (valCert.valsLo k0 : Real) ≤ valsRealOfInputs inputs k0 ∧
                valsRealOfInputs inputs k0 ≤ (valCert.valsHi k0 : Real) := by
            have hv := hv_bounds k0
            have hlo' : ∀ d, (vLo k0 d : Real) ≤ vRealOfInputs inputs k0 d := fun d => (hv d).1
            have hhi' : ∀ d, vRealOfInputs inputs k0 d ≤ (vHi k0 d : Real) := fun d => (hv d).2
            have hlow :=
              dotIntervalLower_le_dotProduct_real (v := dirHead)
                (lo := vLo k0) (hi := vHi k0)
                (x := fun d => vRealOfInputs inputs k0 d) hlo' hhi'
            have hhigh :=
              dotProduct_le_dotIntervalUpper_real (v := dirHead)
                (lo := vLo k0) (hi := vHi k0)
                (x := fun d => vRealOfInputs inputs k0 d) hlo' hhi'
            have hlow' :
                (valCert.valsLo k0 : Real) ≤ valsRealOfInputs inputs k0 := by
              simpa [valsLo, valCert, dirHead, valsRealOfInputs] using hlow
            have hhigh' :
                valsRealOfInputs inputs k0 ≤ (valCert.valsHi k0 : Real) := by
              simpa [valsHi, valCert, dirHead, valsRealOfInputs] using hhigh
            exact ⟨hlow', hhigh'⟩
          have hhi : (valCert.valsHi k0 : Real) ≤ (valCert.hi : Real) := by
            have hhiRat : valCert.valsHi k0 ≤ valCert.hi := by
              change valsHi k0 ≤ hi
              dsimp [hi]
              refine (Finset.le_sup'_iff (s := univ) (H := hnonempty)
                (f := valsHi) (a := valsHi k0)).2 ?_
              exact ⟨k0, ⟨hmem0, le_rfl⟩⟩
            exact (Rat.cast_le (K := Real)).2 hhiRat
          have hreal :
              (valCert.lo : Real) ≤ (valCert.hi : Real) :=
            le_trans hlo (le_trans hvals.1 (le_trans hvals.2 hhi))
          exact (Rat.cast_le (K := Real)).1 hreal
        · intro k
          have hmem : k ∈ univ := by simp [univ]
          have hloRat : valCert.lo ≤ valCert.valsLo k := by
            change lo ≤ valsLo k
            dsimp [lo]
            refine (Finset.inf'_le_iff (s := univ) (H := hnonempty)
              (f := valsLo) (a := valsLo k)).2 ?_
            refine ⟨k, hmem, ?_⟩
            exact le_rfl
          exact (Rat.cast_le (K := Real)).2 hloRat
        · intro k
          have hv := hv_bounds k
          have hlo' : ∀ d, (vLo k d : Real) ≤ vRealOfInputs inputs k d := fun d => (hv d).1
          have hhi' : ∀ d, vRealOfInputs inputs k d ≤ (vHi k d : Real) := fun d => (hv d).2
          have hlow :=
            dotIntervalLower_le_dotProduct_real (v := dirHead)
              (lo := vLo k) (hi := vHi k)
              (x := fun d => vRealOfInputs inputs k d) hlo' hhi'
          have hhigh :=
            dotProduct_le_dotIntervalUpper_real (v := dirHead)
              (lo := vLo k) (hi := vHi k)
              (x := fun d => vRealOfInputs inputs k d) hlo' hhi'
          have hlow' :
              (valCert.valsLo k : Real) ≤ valsRealOfInputs inputs k := by
            simpa [valsLo, valCert, dirHead, valsRealOfInputs] using hlow
          have hhigh' :
              valsRealOfInputs inputs k ≤ (valCert.valsHi k : Real) := by
            simpa [valsHi, valCert, dirHead, valsRealOfInputs] using hhigh
          exact ⟨hlow', hhigh'⟩
        · intro k
          have hmem : k ∈ univ := by simp [univ]
          have hhiRat : valCert.valsHi k ≤ valCert.hi := by
            change valsHi k ≤ hi
            dsimp [hi]
            refine (Finset.le_sup'_iff (s := univ) (H := hnonempty)
              (f := valsHi) (a := valsHi k)).2 ?_
            refine ⟨k, hmem, ?_⟩
            exact le_rfl
          exact (Rat.cast_le (K := Real)).2 hhiRat
      exact some ⟨cert,
        { softmax_bounds := hsoftmax_bounds
          oneHot_bounds_at := oneHot_bounds_at
          value_bounds := hvals_bounds }⟩
  · exact none

section HeadOutputInterval

variable {seq dModel dHead : Nat}

noncomputable section

/-- Real-valued head output using explicit score inputs. -/
def headOutputWithScores (scores : Fin seq → Fin seq → Real)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scores q) k
  let vals : Fin seq → Real := fun k => headValueRealOfInputs inputs k i
  dotProduct (weights q) vals

/-- Unfolding lemma for `headOutputWithScores`. -/
theorem headOutputWithScores_def (scores : Fin seq → Fin seq → Real)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutputWithScores scores inputs q i =
      let weights : Fin seq → Fin seq → Real := fun q k =>
        Circuit.softmax (scores q) k
      let vals : Fin seq → Real := fun k => headValueRealOfInputs inputs k i
      dotProduct (weights q) vals := rfl

/-- Real-valued head output for a query and model dimension. -/
def headOutput (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  headOutputWithScores (scoresRealOfInputs inputs) inputs q i

/-- Unfolding lemma for `headOutput`. -/
theorem headOutput_def (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutput inputs q i =
      headOutputWithScores (scoresRealOfInputs inputs) inputs q i := rfl

/-- Soundness predicate for head-output interval bounds. -/
structure HeadOutputIntervalSound [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (active : Finset (Fin seq))
    (c : Circuit.ResidualIntervalCert dModel) : Prop where
  /-- Interval bounds are ordered coordinatewise. -/
  bounds : Circuit.ResidualIntervalBounds c
  /-- Active-query outputs lie inside the interval bounds. -/
  output_mem :
    ∀ q, q ∈ active → ∀ i,
      (c.lo i : Real) ≤ headOutput inputs q i ∧
        headOutput inputs q i ≤ (c.hi i : Real)

/-- Certified head-output interval data for a specific active set. -/
structure HeadOutputIntervalResult [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) where
  /-- Active queries covered by the interval bounds. -/
  active : Finset (Fin seq)
  /-- Residual-interval certificate for head outputs. -/
  cert : Circuit.ResidualIntervalCert dModel
  /-- Soundness proof for the interval bounds. -/
  sound : HeadOutputIntervalSound inputs active cert

/-- Build residual-interval bounds for head outputs on active queries. -/
def buildHeadOutputIntervalFromHead? [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (HeadOutputIntervalResult inputs) := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      by_cases hEps : 0 < inputs.lnEps
      · by_cases hmodel : dModel = 0
        · exact none
        · cases hbuild : buildInductionCertFromHead? inputs with
          | none => exact none
          | some certWithProof =>
              rcases certWithProof with ⟨cert, hcert⟩
              let lnBounds : Fin (Nat.succ n) → (Fin dModel → Rat) × (Fin dModel → Rat) := fun q =>
                Bounds.layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
              let lnLo : Fin (Nat.succ n) → Fin dModel → Rat := fun q => (lnBounds q).1
              let lnHi : Fin (Nat.succ n) → Fin dModel → Rat := fun q => (lnBounds q).2
              let vLo : Fin (Nat.succ n) → Fin dHead → Rat := fun q d =>
                dotIntervalLower (fun j => inputs.wv j d) (lnLo q) (lnHi q) + inputs.bv d
              let vHi : Fin (Nat.succ n) → Fin dHead → Rat := fun q d =>
                dotIntervalUpper (fun j => inputs.wv j d) (lnLo q) (lnHi q) + inputs.bv d
              let headValueLo : Fin (Nat.succ n) → Fin dModel → Rat := fun k i =>
                dotIntervalLower (fun d => inputs.wo i d) (vLo k) (vHi k)
              let headValueHi : Fin (Nat.succ n) → Fin dModel → Rat := fun k i =>
                dotIntervalUpper (fun d => inputs.wo i d) (vLo k) (vHi k)
              have hln_bounds :
                  ∀ q i, (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
                    lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
                intro q i
                have hln :=
                  Bounds.layerNormBounds_spec (eps := inputs.lnEps)
                    (gamma := inputs.ln1Gamma) (beta := inputs.ln1Beta)
                    (x := inputs.embed q) hmodel hEps
                simpa [lnBounds, lnLo, lnHi, lnRealOfInputs] using hln i
              have hv_bounds :
                  ∀ q d, (vLo q d : Real) ≤ vRealOfInputs inputs q d ∧
                    vRealOfInputs inputs q d ≤ (vHi q d : Real) := by
                intro q d
                have hln := hln_bounds q
                have hlo : ∀ j, (lnLo q j : Real) ≤ lnRealOfInputs inputs q j := fun j => (hln j).1
                have hhi : ∀ j, lnRealOfInputs inputs q j ≤ (lnHi q j : Real) := fun j => (hln j).2
                have hlow :=
                  dotIntervalLower_le_dotProduct_real (v := fun j => inputs.wv j d)
                    (lo := lnLo q) (hi := lnHi q) (x := lnRealOfInputs inputs q) hlo hhi
                have hhigh :=
                  dotProduct_le_dotIntervalUpper_real (v := fun j => inputs.wv j d)
                    (lo := lnLo q) (hi := lnHi q) (x := lnRealOfInputs inputs q) hlo hhi
                have hlow' := add_le_add_right hlow (inputs.bv d : Real)
                have hhigh' := add_le_add_right hhigh (inputs.bv d : Real)
                constructor
                · simpa [vLo, vRealOfInputs, Rat.cast_add] using hlow'
                · simpa [vHi, vRealOfInputs, Rat.cast_add] using hhigh'
              have hhead_bounds :
                  ∀ k i, (headValueLo k i : Real) ≤ headValueRealOfInputs inputs k i ∧
                    headValueRealOfInputs inputs k i ≤ (headValueHi k i : Real) := by
                intro k i
                have hv := hv_bounds k
                have hlo : ∀ d, (vLo k d : Real) ≤ vRealOfInputs inputs k d := fun d => (hv d).1
                have hhi : ∀ d, vRealOfInputs inputs k d ≤ (vHi k d : Real) := fun d => (hv d).2
                have hlow :=
                  dotIntervalLower_le_dotProduct_real (v := fun d => inputs.wo i d)
                    (lo := vLo k) (hi := vHi k)
                    (x := fun d => vRealOfInputs inputs k d) hlo hhi
                have hhigh :=
                  dotProduct_le_dotIntervalUpper_real (v := fun d => inputs.wo i d)
                    (lo := vLo k) (hi := vHi k)
                    (x := fun d => vRealOfInputs inputs k d) hlo hhi
                constructor
                · simpa [headValueLo, headValueRealOfInputs] using hlow
                · simpa [headValueHi, headValueRealOfInputs] using hhigh
              let scoresReal : Fin (Nat.succ n) → Fin (Nat.succ n) → Real :=
                scoresRealOfInputs inputs
              let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := fun q k =>
                Circuit.softmax (scoresReal q) k
              let activeSet : Finset (Fin (Nat.succ n)) := cert.active
              let univ : Finset (Fin (Nat.succ n)) := Finset.univ
              have huniv : univ.Nonempty := by simp [univ]
              let loVal : Fin dModel → Rat := fun i =>
                univ.inf' huniv (fun k => headValueLo k i)
              let hiVal : Fin dModel → Rat := fun i =>
                univ.sup' huniv (fun k => headValueHi k i)
              have hvalsBoundsReal :
                  ∀ i, Layers.ValueRangeBounds (Val := Real)
                    (loVal i : Real) (hiVal i : Real)
                    (fun k => headValueRealOfInputs inputs k i) := by
                intro i
                refine { lo_le_hi := ?_, lo_le := ?_, le_hi := ?_ }
                · rcases (Finset.univ_nonempty : univ.Nonempty) with ⟨k0, hk0⟩
                  have hmem0 : k0 ∈ univ := hk0
                  have hloRat : loVal i ≤ headValueLo k0 i := by
                    change loVal i ≤ headValueLo k0 i
                    dsimp [loVal]
                    refine (Finset.inf'_le_iff (s := univ) (H := huniv)
                      (f := fun k => headValueLo k i) (a := headValueLo k0 i)).2 ?_
                    refine ⟨k0, hmem0, ?_⟩
                    exact le_rfl
                  have hhiRat : headValueHi k0 i ≤ hiVal i := by
                    change headValueHi k0 i ≤ hiVal i
                    dsimp [hiVal]
                    refine (Finset.le_sup'_iff (s := univ) (H := huniv)
                      (f := fun k => headValueHi k i) (a := headValueHi k0 i)).2 ?_
                    exact ⟨k0, ⟨hmem0, le_rfl⟩⟩
                  have hbounds := hhead_bounds k0 i
                  have hreal :
                      (loVal i : Real) ≤ (hiVal i : Real) :=
                    le_trans ((Rat.cast_le (K := Real)).2 hloRat)
                      (le_trans hbounds.1 (le_trans hbounds.2 ((Rat.cast_le (K := Real)).2 hhiRat)))
                  exact hreal
                · intro k
                  have hmem : k ∈ univ := by simp [univ]
                  have hloRat : loVal i ≤ headValueLo k i := by
                    change loVal i ≤ headValueLo k i
                    dsimp [loVal]
                    refine (Finset.inf'_le_iff (s := univ) (H := huniv)
                      (f := fun k => headValueLo k i) (a := headValueLo k i)).2 ?_
                    refine ⟨k, hmem, ?_⟩
                    exact le_rfl
                  have hbounds := hhead_bounds k i
                  exact (Rat.cast_le (K := Real)).2 hloRat |>.trans hbounds.1
                · intro k
                  have hmem : k ∈ univ := by simp [univ]
                  have hhiRat : headValueHi k i ≤ hiVal i := by
                    change headValueHi k i ≤ hiVal i
                    dsimp [hiVal]
                    refine (Finset.le_sup'_iff (s := univ) (H := huniv)
                      (f := fun k => headValueHi k i) (a := headValueHi k i)).2 ?_
                    exact ⟨k, ⟨hmem, le_rfl⟩⟩
                  have hbounds := hhead_bounds k i
                  exact hbounds.2.trans ((Rat.cast_le (K := Real)).2 hhiRat)
              have hsoftmax :
                  Layers.SoftmaxMarginBoundsOn (Val := Real) (cert.eps : Real) (cert.margin : Real)
                    (fun q => q ∈ activeSet) cert.prev scoresReal weights := by
                simpa [scoresReal, weights, activeSet] using hcert.softmax_bounds
              have hweights :
                  Layers.OneHotApproxBoundsOnActive (Val := Real) (cert.eps : Real)
                    (fun q => q ∈ activeSet) cert.prev weights :=
                Layers.oneHotApproxBoundsOnActive_of_softmaxMargin
                  (Val := Real)
                  (ε := (cert.eps : Real))
                  (margin := (cert.margin : Real))
                  (active := fun q => q ∈ activeSet)
                  (prev := cert.prev)
                  (scores := scoresReal)
                  (weights := weights)
                  hsoftmax
              have happrox :
                  ∀ i, Layers.InductionSpecApproxOn (Val := Real) (n := n)
                    ((cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)))
                    (fun q => q ∈ activeSet) cert.prev
                    (fun q => dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i))
                    (fun k => headValueRealOfInputs inputs k i) := by
                intro i
                exact
                  Layers.inductionSpecApproxOn_of_oneHotApprox_valueRange
                    (Val := Real)
                    (n := n)
                    (ε := (cert.eps : Real))
                    (lo := (loVal i : Real))
                    (hi := (hiVal i : Real))
                    (active := fun q => q ∈ activeSet)
                    (prev := cert.prev)
                    (weights := weights)
                    (vals := fun k => headValueRealOfInputs inputs k i)
                    (hweights := hweights)
                    (hvals := hvalsBoundsReal i)
              let delta : Fin dModel → Rat := fun i => hiVal i - loVal i
              let boundLoRat : Fin (Nat.succ n) → Fin dModel → Rat := fun q i =>
                headValueLo (cert.prev q) i - cert.eps * delta i
              let boundHiRat : Fin (Nat.succ n) → Fin dModel → Rat := fun q i =>
                headValueHi (cert.prev q) i + cert.eps * delta i
              let loOut : Fin dModel → Rat := fun i =>
                if h : activeSet.Nonempty then
                  activeSet.inf' h (fun q => boundLoRat q i)
                else
                  0
              let hiOut : Fin dModel → Rat := fun i =>
                if h : activeSet.Nonempty then
                  activeSet.sup' h (fun q => boundHiRat q i)
                else
                  0
              have hout :
                  ∀ q, q ∈ activeSet → ∀ i,
                    (loOut i : Real) ≤ headOutput inputs q i ∧
                      headOutput inputs q i ≤ (hiOut i : Real) := by
                intro q hq i
                have hactive : activeSet.Nonempty := ⟨q, hq⟩
                have hspec := (happrox i) q hq
                have hout_def :
                    headOutput inputs q i =
                      dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) := by
                  simp [headOutput, headOutputWithScores, scoresReal, weights]
                have hprev_bounds := hhead_bounds (cert.prev q) i
                have hupper :
                    headOutput inputs q i ≤ (boundHiRat q i : Real) := by
                  have hupper' :
                      dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) ≤
                        headValueRealOfInputs inputs (cert.prev q) i +
                          (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) := by
                    exact hspec.1
                  have hupper'' :
                      dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) ≤
                        (headValueHi (cert.prev q) i : Real) +
                          (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) := by
                    have hprev_bounds' :=
                      (add_le_add_iff_right
                          ((cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)))).2
                        hprev_bounds.2
                    exact le_trans hupper' hprev_bounds'
                  simpa
                    [hout_def, boundHiRat, delta, Rat.cast_add, Rat.cast_mul, Rat.cast_sub] using
                    hupper''
                have hlower :
                    (boundLoRat q i : Real) ≤ headOutput inputs q i := by
                  have hlower' :
                      (headValueRealOfInputs inputs (cert.prev q) i : Real) -
                          (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) ≤
                        dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) := by
                    exact (sub_le_iff_le_add).2 hspec.2
                  have hlower'' :
                      (headValueLo (cert.prev q) i : Real) -
                          (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) ≤
                        dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) := by
                    exact le_trans (sub_le_sub_right hprev_bounds.1
                      ((cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)))) hlower'
                  simpa [hout_def, boundLoRat, delta, Rat.cast_mul, Rat.cast_sub] using
                    hlower''
                have hlo :
                    (loOut i : Real) ≤ (boundLoRat q i : Real) := by
                  have hloRat : loOut i ≤ boundLoRat q i := by
                    simpa [loOut, hactive] using
                      (Finset.inf'_le (s := activeSet) (f := fun q => boundLoRat q i) (b := q) hq)
                  exact (Rat.cast_le (K := Real)).2 hloRat
                have hhi :
                    (boundHiRat q i : Real) ≤ (hiOut i : Real) := by
                  have hhiRat : boundHiRat q i ≤ hiOut i := by
                    simpa [hiOut, hactive] using
                      (Finset.le_sup' (s := activeSet) (f := fun q => boundHiRat q i) (b := q) hq)
                  exact (Rat.cast_le (K := Real)).2 hhiRat
                exact ⟨le_trans hlo hlower, le_trans hupper hhi⟩
              have hbounds : Circuit.ResidualIntervalBounds { lo := loOut, hi := hiOut } := by
                refine { lo_le_hi := ?_ }
                intro i
                by_cases hactive : activeSet.Nonempty
                · rcases hactive with ⟨q, hq⟩
                  have hout_i := hout q hq i
                  have hleReal : (loOut i : Real) ≤ (hiOut i : Real) :=
                    le_trans hout_i.1 hout_i.2
                  exact (Rat.cast_le (K := Real)).1 hleReal
                · simp [loOut, hiOut, hactive]
              let certOut : Circuit.ResidualIntervalCert dModel := { lo := loOut, hi := hiOut }
              exact some
                { active := activeSet
                  cert := certOut
                  sound :=
                    { bounds := hbounds
                      output_mem := by
                        intro q hq i
                        exact hout q hq i } }
      · exact none

end

end HeadOutputInterval

end Sound

end Nfp
