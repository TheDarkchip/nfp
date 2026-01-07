-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Nfp.Core.Basic
import Mathlib.Data.Finset.Lattice.Fold
import Nfp.Circuit.Cert.ResidualInterval
import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange
import Nfp.Sound.Bounds.Attention
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Induction.CoreDefs
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

/-- Build and certify a softmax-margin certificate from exact scores/weights. -/
def buildSoftmaxMarginCert? [NeZero seq]
    (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (scores : Fin seq → Fin seq → Dyadic)
    (weights : Fin seq → Fin seq → Dyadic) :
    Option {c : SoftmaxMarginCert seq // checkSoftmaxMarginCert c = true} := by
  classical
  let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (prev q)
  let epsAt : Fin seq → Dyadic := fun q =>
    let other := otherKeys q
    let maxOther :=
      if h : other.Nonempty then
        other.sup' h (fun k => weights q k)
      else
        (0 : Dyadic)
    let deficit := (1 : Dyadic) - weights q (prev q)
    max maxOther deficit
  let marginAt : Fin seq → Dyadic := fun q =>
    let other := otherKeys q
    if h : other.Nonempty then
      other.inf' h (fun k => scores q (prev q) - scores q k)
    else
      (0 : Dyadic)
  let eps :=
    if h : active.Nonempty then
      active.sup' h epsAt
    else
      (0 : Dyadic)
  let margin :=
    if h : active.Nonempty then
      active.inf' h marginAt
    else
      (0 : Dyadic)
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
    (vals : Fin seq → Dyadic)
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

/-- Build induction certificates from exact head inputs (core computation). -/
def buildInductionCertFromHeadCore? [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (InductionHeadCert seq) := by
  classical
  by_cases hEps : 0 < inputs.lnEps
  · by_cases hSqrt : 0 < sqrtLower inputs.lnEps
    · by_cases hmodel : dModel = 0
      · exact none
      · by_cases hactive : inputs.active.Nonempty
        · let lnBounds := Bounds.cacheBoundPair2 (fun q =>
            Bounds.layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q))
          let lnLo : Fin seq → Fin dModel → Dyadic := lnBounds.1
          let lnHi : Fin seq → Fin dModel → Dyadic := lnBounds.2
          let qLo :=
            Bounds.cacheBound2 (fun q d =>
              Bounds.dotIntervalLowerCachedDyadic (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
                inputs.bq d)
          let qHi :=
            Bounds.cacheBound2 (fun q d =>
              Bounds.dotIntervalUpperCachedDyadic (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
                inputs.bq d)
          let kLo :=
            Bounds.cacheBound2 (fun q d =>
              Bounds.dotIntervalLowerCachedDyadic (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
                inputs.bk d)
          let kHi :=
            Bounds.cacheBound2 (fun q d =>
              Bounds.dotIntervalUpperCachedDyadic (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
                inputs.bk d)
          let vLo :=
            Bounds.cacheBound2 (fun q d =>
              Bounds.dotIntervalLowerCachedDyadic (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
                inputs.bv d)
          let vHi :=
            Bounds.cacheBound2 (fun q d =>
              Bounds.dotIntervalUpperCachedDyadic (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
                inputs.bv d)
          let qAbs := Bounds.cacheBound2 (fun q d => max |qLo q d| |qHi q d|)
          let kAbs := Bounds.cacheBound2 (fun q d => max |kLo q d| |kHi q d|)
          let kAbsMax : Fin dHead → Dyadic := fun d =>
            let univ : Finset (Fin seq) := Finset.univ
            have hnonempty : univ.Nonempty := Finset.univ_nonempty
            univ.sup' hnonempty (fun k => kAbs k d)
          let masked : Fin seq → Fin seq → Prop := fun q k =>
            inputs.maskCausal = true ∧ q < k
          let dotAbs : Fin seq → Fin seq → Dyadic := fun q k =>
            Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d)
          let scoreBaseAbs : Fin seq → Fin seq → Dyadic := fun q k =>
            |inputs.scale| * dotAbs q k
          let scoreAbs : Fin seq → Fin seq → Dyadic := fun q k =>
            if masked q k then |inputs.maskValue| else scoreBaseAbs q k
          let scoreLo : Fin seq → Fin seq → Dyadic := fun q k =>
            if masked q k then inputs.maskValue else -scoreBaseAbs q k
          let scoreHi : Fin seq → Fin seq → Dyadic := fun q k =>
            if masked q k then inputs.maskValue else scoreBaseAbs q k
          let dotAbsUpper : Fin seq → Dyadic := fun q =>
            Linear.dotFin dHead (fun d => qAbs q d) kAbsMax
          let scoreHiUpper : Fin seq → Dyadic := fun q =>
            max inputs.maskValue (|inputs.scale| * dotAbsUpper q)
          let fastGap : Fin seq → Dyadic := fun q =>
            let prev := inputs.prev q
            scoreLo q prev - scoreHiUpper q
          let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
            (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
          let maskedKeys : Fin seq → Finset (Fin seq) := fun q =>
            if inputs.maskCausal = true then
              (otherKeys q).filter (fun k => q < k)
            else
              (∅ : Finset (Fin seq))
          let unmaskedKeys : Fin seq → Finset (Fin seq) := fun q =>
            (otherKeys q) \ (maskedKeys q)
          let maskedGap : Fin seq → Dyadic := fun q =>
            scoreLo q (inputs.prev q) - inputs.maskValue
          let marginAtRaw : Fin seq → Dyadic := fun q =>
            let fast := fastGap q
            if fast < 0 then
              let other := unmaskedKeys q
              let maskedSet := maskedKeys q
              if hunmasked : other.Nonempty then
                let unmaskedMin := other.inf' hunmasked (fun k =>
                  scoreLo q (inputs.prev q) - scoreHi q k)
                if maskedSet.Nonempty then
                  min unmaskedMin (maskedGap q)
                else
                  unmaskedMin
              else
                if maskedSet.Nonempty then
                  maskedGap q
                else
                  (0 : Dyadic)
            else
              fast
          let marginAt : Fin seq → Dyadic := fun q =>
            if q ∈ inputs.active then
              marginAtRaw q
            else
              (0 : Dyadic)
          let epsAt : Fin seq → Dyadic := fun q =>
            if marginAt q < 0 then
              (1 : Dyadic)
            else
              dyadicDivUp (seq - 1) (1 + marginAt q)
          let margin : Dyadic := inputs.active.inf' hactive marginAt
          let eps : Dyadic :=
            if margin < 0 then
              (1 : Dyadic)
            else
              dyadicDivUp (seq - 1) (1 + margin)
          let dirHeadVec := dirHeadVecOfInputs inputs
          let dirHead : Fin dHead → Dyadic := fun d => dirHeadVec.get d
          let valsLo :=
            Bounds.cacheBound (fun k =>
              Bounds.dotIntervalLowerCachedDyadic dirHead (vLo k) (vHi k))
          let valsHi :=
            Bounds.cacheBound (fun k =>
              Bounds.dotIntervalUpperCachedDyadic dirHead (vLo k) (vHi k))
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
          exact some cert
        · exact none
    · exact none
  · exact none

set_option maxHeartbeats 1000000 in
-- Large softmax/interval proof expands many bounds; bump heartbeats to avoid timeouts.
/-- Soundness for `buildInductionCertFromHeadCore?`. -/
theorem buildInductionCertFromHeadCore?_sound [NeZero seq] {dModel dHead : Nat}
      (inputs : Model.InductionHeadInputs seq dModel dHead) (c : InductionHeadCert seq)
      (hcore : buildInductionCertFromHeadCore? inputs = some c) :
      InductionHeadCertSound inputs c := by
    classical
    by_cases hEps : 0 < inputs.lnEps
    · by_cases hSqrt : 0 < sqrtLower inputs.lnEps
      · by_cases hmodel : dModel = 0
        · have : False := by
            simp [buildInductionCertFromHeadCore?, hEps, hSqrt, hmodel] at hcore
          exact this.elim
        · by_cases hactive : inputs.active.Nonempty
          · let lnBounds := Bounds.cacheBoundPair2 (fun q =>
              Bounds.layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q))
            let lnLo : Fin seq → Fin dModel → Dyadic := lnBounds.1
            let lnHi : Fin seq → Fin dModel → Dyadic := lnBounds.2
            let qLo :=
              Bounds.cacheBound2 (fun q d =>
                Bounds.dotIntervalLowerCachedDyadic (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
                  inputs.bq d)
            let qHi :=
              Bounds.cacheBound2 (fun q d =>
                Bounds.dotIntervalUpperCachedDyadic (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
                  inputs.bq d)
            let kLo :=
              Bounds.cacheBound2 (fun q d =>
                Bounds.dotIntervalLowerCachedDyadic (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
                  inputs.bk d)
            let kHi :=
              Bounds.cacheBound2 (fun q d =>
                Bounds.dotIntervalUpperCachedDyadic (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
                  inputs.bk d)
            let vLo :=
              Bounds.cacheBound2 (fun q d =>
                Bounds.dotIntervalLowerCachedDyadic (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
                  inputs.bv d)
            let vHi :=
              Bounds.cacheBound2 (fun q d =>
                Bounds.dotIntervalUpperCachedDyadic (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
                  inputs.bv d)
            let qAbs :=
              Bounds.cacheBound2 (fun q d => max |qLo q d| |qHi q d|)
            let kAbs :=
              Bounds.cacheBound2 (fun q d => max |kLo q d| |kHi q d|)
            let kAbsMax : Fin dHead → Dyadic := fun d =>
              let univ : Finset (Fin seq) := Finset.univ
              have hnonempty : univ.Nonempty := Finset.univ_nonempty
              univ.sup' hnonempty (fun k => kAbs k d)
            let masked : Fin seq → Fin seq → Prop := fun q k =>
              inputs.maskCausal = true ∧ q < k
            let dotAbs : Fin seq → Fin seq → Dyadic := fun q k =>
              Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d)
            let scoreBaseAbs : Fin seq → Fin seq → Dyadic := fun q k =>
              |inputs.scale| * dotAbs q k
            let scoreAbs : Fin seq → Fin seq → Dyadic := fun q k =>
              if masked q k then |inputs.maskValue| else scoreBaseAbs q k
            let scoreLo : Fin seq → Fin seq → Dyadic := fun q k =>
              if masked q k then inputs.maskValue else -scoreBaseAbs q k
            let scoreHi : Fin seq → Fin seq → Dyadic := fun q k =>
              if masked q k then inputs.maskValue else scoreBaseAbs q k
            let dotAbsUpper : Fin seq → Dyadic := fun q =>
              Linear.dotFin dHead (fun d => qAbs q d) kAbsMax
            let scoreHiUpper : Fin seq → Dyadic := fun q =>
              max inputs.maskValue (|inputs.scale| * dotAbsUpper q)
            let fastGap : Fin seq → Dyadic := fun q =>
              let prev := inputs.prev q
              scoreLo q prev - scoreHiUpper q
            let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
              (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
            let maskedKeys : Fin seq → Finset (Fin seq) := fun q =>
              if inputs.maskCausal = true then
                (otherKeys q).filter (fun k => q < k)
              else
                (∅ : Finset (Fin seq))
            let unmaskedKeys : Fin seq → Finset (Fin seq) := fun q =>
              (otherKeys q) \ (maskedKeys q)
            let maskedGap : Fin seq → Dyadic := fun q =>
              scoreLo q (inputs.prev q) - inputs.maskValue
            let marginAtRaw : Fin seq → Dyadic := fun q =>
              let fast := fastGap q
              if fast < 0 then
                let other := unmaskedKeys q
                let maskedSet := maskedKeys q
                if hunmasked : other.Nonempty then
                  let unmaskedMin := other.inf' hunmasked (fun k =>
                    scoreLo q (inputs.prev q) - scoreHi q k)
                  if maskedSet.Nonempty then
                    min unmaskedMin (maskedGap q)
                  else
                    unmaskedMin
                else
                  if maskedSet.Nonempty then
                    maskedGap q
                  else
                    (0 : Dyadic)
              else
                fast
            let marginAt : Fin seq → Dyadic := fun q =>
              if q ∈ inputs.active then
                marginAtRaw q
              else
                (0 : Dyadic)
            let epsAt : Fin seq → Dyadic := fun q =>
              if marginAt q < 0 then
                (1 : Dyadic)
              else
                dyadicDivUp (seq - 1) (1 + marginAt q)
            let margin : Dyadic := inputs.active.inf' hactive marginAt
            let eps : Dyadic :=
              if margin < 0 then
                (1 : Dyadic)
              else
                dyadicDivUp (seq - 1) (1 + margin)
            have hseq : (1 : Nat) ≤ seq :=
              Nat.succ_le_iff.mpr (Nat.pos_of_ne_zero (NeZero.ne seq))
            let dirHeadVec := dirHeadVecOfInputs inputs
            let dirHead : Fin dHead → Dyadic := fun d => dirHeadVec.get d
            let valsLo :=
              Bounds.cacheBound (fun k =>
                Bounds.dotIntervalLowerCachedDyadic dirHead (vLo k) (vHi k))
            let valsHi :=
              Bounds.cacheBound (fun k =>
                Bounds.dotIntervalUpperCachedDyadic dirHead (vLo k) (vHi k))
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
            have hcore' : some cert = some c := by
              simpa
                [buildInductionCertFromHeadCore?, hEps, hSqrt, hmodel, hactive, lnBounds, lnLo,
                  lnHi, qLo, qHi, kLo, kHi, vLo, vHi, qAbs, kAbs, kAbsMax, masked, dotAbs,
                  scoreBaseAbs, scoreAbs, scoreLo, scoreHi, dotAbsUpper, scoreHiUpper, fastGap,
                  otherKeys, maskedKeys, unmaskedKeys, maskedGap, marginAt, marginAtRaw, epsAt,
                  margin, eps, dirHeadVec, dirHead, valsLo, valsHi, univ, lo, hi, valCert, cert]
                using hcore
            have hc : c = cert := by
              simpa using (Option.some.inj hcore').symm
            subst hc
            have hln_bounds :
                ∀ q i, (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
                  lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
              intro q i
              have hln :=
                Bounds.layerNormBounds_spec (eps := inputs.lnEps)
                  (gamma := inputs.ln1Gamma) (beta := inputs.ln1Beta)
                  (x := inputs.embed q) hmodel hEps hSqrt
              simpa [lnBounds, lnLo, lnHi, lnRealOfInputs,
                Bounds.cacheBoundPair2_apply_left, Bounds.cacheBoundPair2_apply_right]
                using hln i
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
              · simpa [qLo, qRealOfInputs, Bounds.cacheBound2_apply,
                  Bounds.dotIntervalLowerCachedRat_eq] using hlow'
              · simpa [qHi, qRealOfInputs, Bounds.cacheBound2_apply,
                  Bounds.dotIntervalUpperCachedRat_eq] using hhigh'
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
              · simpa [kLo, kRealOfInputs, Bounds.cacheBound2_apply,
                  Bounds.dotIntervalLowerCachedRat_eq] using hlow'
              · simpa [kHi, kRealOfInputs, Bounds.cacheBound2_apply,
                  Bounds.dotIntervalUpperCachedRat_eq] using hhigh'
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
              · simpa [vLo, vRealOfInputs, Bounds.cacheBound2_apply,
                  Bounds.dotIntervalLowerCachedRat_eq] using hlow'
              · simpa [vHi, vRealOfInputs, Bounds.cacheBound2_apply,
                  Bounds.dotIntervalUpperCachedRat_eq] using hhigh'
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
                simpa [qAbs, Bounds.cacheBound2_apply] using h
              have hk_abs : ∀ d, |kRealOfInputs inputs k d| ≤ (kAbs k d : Real) := by
                intro d
                have hk := hk_bounds k d
                have h := abs_le_max_abs_abs_of_interval_real hk.1 hk.2
                simpa [kAbs, Bounds.cacheBound2_apply] using h
              have hdot_abs :
                  |dotProduct (fun d => qRealOfInputs inputs q d)
                      (fun d => kRealOfInputs inputs k d)| ≤
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
                      have hmax : |qLo q d| ≤ qAbs q d := by
                        simp [qAbs, Bounds.cacheBound2_apply]
                      exact le_trans (abs_nonneg _) hmax
                    exact dyadicToReal_nonneg_of_nonneg hqnonneg'
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
                  have hsum :
                      ((∑ d, qAbs q d * kAbs k d : Dyadic) : Real) =
                        ∑ d, ((qAbs q d * kAbs k d : Dyadic) : Real) :=
                    Linear.dyadicToReal_sum_univ (f := fun d => qAbs q d * kAbs k d)
                  have hsum' :
                      ∑ d, ((qAbs q d * kAbs k d : Dyadic) : Real) =
                        ∑ d, (qAbs q d : Real) * (kAbs k d : Real) := by
                    refine Finset.sum_congr rfl ?_
                    intro d _
                    simp [dyadicToReal_mul]
                  have hfinal := hsum.trans hsum'
                  simpa [dotAbs, Linear.dotFin_eq_dotProduct, dotProduct] using hfinal
                have hfinal := hsum.trans (hsum_le.trans_eq hcast.symm)
                simpa [dotProduct] using hfinal
              have hscale_abs : 0 ≤ (|inputs.scale| : Real) := by
                exact abs_nonneg (dyadicToReal inputs.scale)
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
                    simpa [scoreAbs, masked, hcausal, hnot]
                      using hscore_abs'
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
            have hdotAbs_le : ∀ q k, dotAbs q k ≤ dotAbsUpper q := by
              intro q k
              classical
              have hnonneg : ∀ d, 0 ≤ qAbs q d := by
                intro d
                have h0 : 0 ≤ |qLo q d| := abs_nonneg _
                have hle : |qLo q d| ≤ qAbs q d := by
                  simp [qAbs, Bounds.cacheBound2_apply]
                exact le_trans h0 hle
              have hterm : ∀ d, qAbs q d * kAbs k d ≤ qAbs q d * kAbsMax d := by
                intro d
                have hmem : k ∈ (Finset.univ : Finset (Fin seq)) := by simp
                have hkabs : kAbs k d ≤ kAbsMax d := by
                  simpa [kAbsMax] using
                    (Finset.le_sup' (s := (Finset.univ : Finset (Fin seq)))
                      (f := fun k => kAbs k d) (b := k) hmem)
                exact mul_le_mul_of_nonneg_left hkabs (hnonneg d)
              have hsum : (∑ d, qAbs q d * kAbs k d) ≤ ∑ d, qAbs q d * kAbsMax d := by
                refine Finset.sum_le_sum ?_
                intro d _
                exact hterm d
              simpa [dotAbs, dotAbsUpper, Linear.dotFin_eq_dotProduct, dotProduct] using hsum
            have hscoreHi_le : ∀ q k, scoreHi q k ≤ scoreHiUpper q := by
              intro q k
              by_cases hmask : masked q k
              · simp [scoreHi, scoreHiUpper, hmask]
              · have hdot := hdotAbs_le q k
                have hmul : |inputs.scale| * dotAbs q k ≤ |inputs.scale| * dotAbsUpper q := by
                  exact mul_le_mul_of_nonneg_left hdot (abs_nonneg _)
                calc
                  scoreHi q k = |inputs.scale| * dotAbs q k := by
                    simp [scoreHi, scoreBaseAbs, hmask]
                  _ ≤ |inputs.scale| * dotAbsUpper q := hmul
                  _ ≤ max inputs.maskValue (|inputs.scale| * dotAbsUpper q) := by
                    exact le_max_right _ _
            let weights : Fin seq → Fin seq → Real := fun q k =>
              Circuit.softmax (scoresReal q) k
            let others : Fin seq → Finset (Fin seq) := fun q =>
              (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
            have hscore_margin_real :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  scoresReal q k + (margin : Real) ≤ scoresReal q (inputs.prev q) := by
              intro q hq k hk
              have hmargin_le : margin ≤ marginAt q := by
                have hle : margin ≤ inputs.active.inf' hactive marginAt := by
                  simp [margin]
                have hle_all :=
                  (Finset.le_inf'_iff (s := inputs.active) (H := hactive) (f := marginAt)
                    (a := margin)).1 hle
                exact hle_all q hq
              have hgap_le :
                  marginAt q ≤ scoreLo q (inputs.prev q) - scoreHi q k := by
                by_cases hfast : fastGap q < 0
                · by_cases hmask : k ∈ maskedKeys q
                  · have hmask_nonempty : (maskedKeys q).Nonempty := ⟨k, hmask⟩
                    have hmargin_eq : marginAt q = marginAtRaw q := by
                      simp [marginAt, hq]
                    have hraw_le : marginAtRaw q ≤ maskedGap q := by
                      by_cases hunmasked : (unmaskedKeys q).Nonempty
                      · have hraw_eq :
                            marginAtRaw q =
                              let unmaskedMin := (unmaskedKeys q).inf' hunmasked
                                (fun k => scoreLo q (inputs.prev q) - scoreHi q k)
                              min unmaskedMin (maskedGap q) := by
                            simp [marginAtRaw, hfast, hunmasked, hmask_nonempty]
                        simp [hraw_eq]
                      · have hraw_eq : marginAtRaw q = maskedGap q := by
                          simp [marginAtRaw, hfast, hunmasked, hmask_nonempty]
                        simp [hraw_eq]
                    have hcausal : inputs.maskCausal = true := by
                      by_contra hcausal
                      simp [maskedKeys, hcausal] at hmask
                    have hmem :
                        k ∈ (otherKeys q).filter (fun k => q < k) := by
                      simpa [maskedKeys, hcausal] using hmask
                    have hlt : q < k := (Finset.mem_filter.mp hmem).2
                    have hmask_prop : masked q k := ⟨hcausal, hlt⟩
                    have hmask_score : scoreHi q k = inputs.maskValue := by
                      simp [scoreHi, hmask_prop]
                    have hgap : marginAt q ≤ scoreLo q (inputs.prev q) - inputs.maskValue := by
                      simpa [hmargin_eq] using hraw_le
                    simpa [maskedGap, hmask_score] using hgap
                  · have hmem : k ∈ unmaskedKeys q := by
                      have hother_mem : k ∈ otherKeys q := by simp [otherKeys, hk]
                      simp [unmaskedKeys, hother_mem, hmask]
                    have hunmasked : (unmaskedKeys q).Nonempty := ⟨k, hmem⟩
                    have hmargin_eq : marginAt q = marginAtRaw q := by
                      simp [marginAt, hq]
                    have hraw_le : marginAtRaw q ≤
                        (unmaskedKeys q).inf' hunmasked
                          (fun k => scoreLo q (inputs.prev q) - scoreHi q k) := by
                      let unmaskedMin :=
                        (unmaskedKeys q).inf' hunmasked
                          (fun k => scoreLo q (inputs.prev q) - scoreHi q k)
                      by_cases hmask_nonempty : (maskedKeys q).Nonempty
                      · have hraw_eq : marginAtRaw q = min unmaskedMin (maskedGap q) := by
                          simp [marginAtRaw, hfast, hunmasked, hmask_nonempty, unmaskedMin,
                            unmaskedKeys, maskedKeys]
                        have hmin_le : marginAtRaw q ≤ unmaskedMin := by
                          rw [hraw_eq]
                          exact min_le_left _ _
                        exact hmin_le
                      · simp [marginAtRaw, hfast, hunmasked, hmask_nonempty, unmaskedKeys,
                          maskedKeys]
                    have hle_all :=
                      (Finset.le_inf'_iff (s := unmaskedKeys q) (H := hunmasked)
                        (f := fun k => scoreLo q (inputs.prev q) - scoreHi q k)
                        (a := marginAtRaw q)).1 hraw_le
                    have hle := hle_all k hmem
                    simpa [hmargin_eq] using hle
                · have hgap_fast : fastGap q ≤ scoreLo q (inputs.prev q) - scoreHi q k := by
                    have hle_score : scoreHi q k ≤ scoreHiUpper q := hscoreHi_le q k
                    have hle_sub :
                        scoreLo q (inputs.prev q) - scoreHiUpper q ≤
                          scoreLo q (inputs.prev q) - scoreHi q k :=
                      sub_le_sub_left hle_score (scoreLo q (inputs.prev q))
                    simpa [fastGap] using hle_sub
                  have hmargin_eq : marginAt q = fastGap q := by
                    simp [marginAt, marginAtRaw, hq, hfast]
                  simpa [hmargin_eq] using hgap_fast
              have hgap : margin ≤ scoreLo q (inputs.prev q) - scoreHi q k :=
                le_trans hmargin_le hgap_le
              have hgap_real : (margin : Real) ≤
                  (scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real) := by
                have hgap_real' :
                    (margin : Real) ≤ ((scoreLo q (inputs.prev q) - scoreHi q k : Dyadic) : Real) :=
                  dyadicToReal_le_of_le hgap
                simpa [dyadicToReal_sub] using hgap_real'
              have hk_bounds := hscore_bounds q k
              have hprev_bounds := hscore_bounds q (inputs.prev q)
              have h1 :
                  scoresReal q k + (margin : Real) ≤
                    scoresReal q k +
                      ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) := by
                exact (add_le_add_iff_left (scoresReal q k)).2 hgap_real
              have h2 :
                  scoresReal q k +
                      ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) ≤
                    (scoreLo q (inputs.prev q) : Real) := by
                have hscore_le' :
                    scoresReal q k +
                        ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) ≤
                      (scoreHi q k : Real) +
                        ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) := by
                  exact (add_le_add_iff_right
                    ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real))).2 hk_bounds.2
                calc
                  scoresReal q k +
                      ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) ≤
                    (scoreHi q k : Real) +
                      ((scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real)) := by
                    exact hscore_le'
                  _ = (scoreLo q (inputs.prev q) : Real) := by
                    exact add_sub_cancel (scoreHi q k : Real) (scoreLo q (inputs.prev q) : Real)
              have h3 :
                  scoresReal q k + (margin : Real) ≤ (scoreLo q (inputs.prev q) : Real) :=
                h1.trans h2
              exact h3.trans hprev_bounds.1
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
                      exact dyadicToReal_nonneg_of_nonneg hnonneg
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
                    have hden : (1 + margin) ≠ 0 := by
                      have hnonneg_real : 0 ≤ (margin : Real) :=
                        dyadicToReal_nonneg_of_nonneg hnonneg
                      have hpos_real : (0 : Real) < 1 + (margin : Real) := by
                        linarith
                      have hpos_real' : dyadicToReal 0 < dyadicToReal (1 + margin) := by
                        simpa [dyadicToReal_add] using hpos_real
                      have hpos : (0 : Dyadic) < 1 + margin :=
                        (dyadicToReal_lt_iff).1 hpos_real'
                      exact ne_of_gt hpos
                    have heps :
                        (seq - 1 : Real) * (1 + (margin : Real))⁻¹ ≤ (eps : Real) := by
                      have hrat' := dyadicDivUp_ge_real (seq - 1) (1 + margin) hden
                      simpa [eps, hneg, dyadicToReal, Dyadic.toRat_add, Dyadic.toRat_natCast,
                        Rat.cast_div, Rat.cast_add, Rat.cast_natCast, div_eq_mul_inv] using hrat'
                    exact le_trans hsum_le' heps
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
                      exact dyadicToReal_nonneg_of_nonneg hnonneg
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
                    have hden : (1 + margin) ≠ 0 := by
                      have hnonneg_real : 0 ≤ (margin : Real) :=
                        dyadicToReal_nonneg_of_nonneg hnonneg
                      have hpos_real : (0 : Real) < 1 + (margin : Real) := by
                        linarith
                      have hpos_real' : dyadicToReal 0 < dyadicToReal (1 + margin) := by
                        simpa [dyadicToReal_add] using hpos_real
                      have hpos : (0 : Dyadic) < 1 + margin :=
                        (dyadicToReal_lt_iff).1 hpos_real'
                      exact ne_of_gt hpos
                    have heps :
                        (seq - 1 : Real) * (1 + (margin : Real))⁻¹ ≤ (eps : Real) := by
                      have hrat' := dyadicDivUp_ge_real (seq - 1) (1 + margin) hden
                      simpa [eps, hneg, dyadicToReal, Dyadic.toRat_add, Dyadic.toRat_natCast,
                        Rat.cast_div, Rat.cast_add, Rat.cast_natCast, div_eq_mul_inv] using hrat'
                    exact le_trans hsum_le' heps
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
              have hgap_le :
                  marginAt q ≤ scoreLo q (inputs.prev q) - scoreHi q k := by
                by_cases hfast : fastGap q < 0
                · by_cases hmask : k ∈ maskedKeys q
                  · have hmask_nonempty : (maskedKeys q).Nonempty := ⟨k, hmask⟩
                    have hmargin_eq : marginAt q = marginAtRaw q := by
                      simp [marginAt, hq]
                    have hraw_le : marginAtRaw q ≤ maskedGap q := by
                      by_cases hunmasked : (unmaskedKeys q).Nonempty
                      · have hraw_eq :
                            marginAtRaw q =
                              let unmaskedMin := (unmaskedKeys q).inf' hunmasked
                                (fun k => scoreLo q (inputs.prev q) - scoreHi q k)
                              min unmaskedMin (maskedGap q) := by
                            simp [marginAtRaw, hfast, hunmasked, hmask_nonempty]
                        simp [hraw_eq]
                      · have hraw_eq : marginAtRaw q = maskedGap q := by
                          simp [marginAtRaw, hfast, hunmasked, hmask_nonempty]
                        simp [hraw_eq]
                    have hcausal : inputs.maskCausal = true := by
                      by_contra hcausal
                      simp [maskedKeys, hcausal] at hmask
                    have hmem :
                        k ∈ (otherKeys q).filter (fun k => q < k) := by
                      simpa [maskedKeys, hcausal] using hmask
                    have hlt : q < k := (Finset.mem_filter.mp hmem).2
                    have hmask_prop : masked q k := ⟨hcausal, hlt⟩
                    have hmask_score : scoreHi q k = inputs.maskValue := by
                      simp [scoreHi, hmask_prop]
                    have hgap : marginAt q ≤ scoreLo q (inputs.prev q) - inputs.maskValue := by
                      simpa [hmargin_eq] using hraw_le
                    simpa [maskedGap, hmask_score] using hgap
                  · have hmem : k ∈ unmaskedKeys q := by
                      have hother_mem : k ∈ otherKeys q := by simp [otherKeys, hk]
                      simp [unmaskedKeys, hother_mem, hmask]
                    have hunmasked : (unmaskedKeys q).Nonempty := ⟨k, hmem⟩
                    have hmargin_eq : marginAt q = marginAtRaw q := by
                      simp [marginAt, hq]
                    have hraw_le : marginAtRaw q ≤
                        (unmaskedKeys q).inf' hunmasked
                          (fun k => scoreLo q (inputs.prev q) - scoreHi q k) := by
                      let unmaskedMin :=
                        (unmaskedKeys q).inf' hunmasked
                          (fun k => scoreLo q (inputs.prev q) - scoreHi q k)
                      by_cases hmask_nonempty : (maskedKeys q).Nonempty
                      · have hraw_eq : marginAtRaw q = min unmaskedMin (maskedGap q) := by
                          simp [marginAtRaw, hfast, hunmasked, hmask_nonempty, unmaskedMin,
                            unmaskedKeys, maskedKeys]
                        rw [hraw_eq]
                        exact min_le_left _ _
                      · simp [marginAtRaw, hfast, hunmasked, hmask_nonempty, unmaskedKeys,
                          maskedKeys]
                    have hle_all :=
                      (Finset.le_inf'_iff (s := unmaskedKeys q) (H := hunmasked)
                        (f := fun k => scoreLo q (inputs.prev q) - scoreHi q k)
                        (a := marginAtRaw q)).1 hraw_le
                    have hle := hle_all k hmem
                    simpa [hmargin_eq] using hle
                · have hgap_fast : fastGap q ≤ scoreLo q (inputs.prev q) - scoreHi q k := by
                    have hle_score : scoreHi q k ≤ scoreHiUpper q := hscoreHi_le q k
                    have hle_sub :
                        scoreLo q (inputs.prev q) - scoreHiUpper q ≤
                          scoreLo q (inputs.prev q) - scoreHi q k :=
                      sub_le_sub_left hle_score (scoreLo q (inputs.prev q))
                    simpa [fastGap] using hle_sub
                  have hmargin_eq : marginAt q = fastGap q := by
                    simp [marginAt, marginAtRaw, hq, hfast]
                  simpa [hmargin_eq] using hgap_fast
              have hgap_real :
                  (marginAt q : Real) ≤
                    (scoreLo q (inputs.prev q) : Real) - (scoreHi q k : Real) := by
                have hgap_real' :
                    (marginAt q : Real) ≤
                      ((scoreLo q (inputs.prev q) - scoreHi q k : Dyadic) : Real) :=
                  dyadicToReal_le_of_le hgap_le
                simpa [dyadicToReal_sub] using hgap_real'
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
                  if marginAt q < 0 then
                    (1 : Dyadic)
                  else
                    dyadicDivUp (seq - 1) (1 + marginAt q) := by
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
                  have hloDyadic : valCert.lo ≤ valCert.valsLo k0 := by
                    change lo ≤ valsLo k0
                    dsimp [lo]
                    refine (Finset.inf'_le_iff (s := univ) (H := hnonempty)
                      (f := valsLo) (a := valsLo k0)).2 ?_
                    refine ⟨k0, hmem0, ?_⟩
                    exact le_rfl
                  exact dyadicToReal_le_of_le hloDyadic
                have hvals :
                    (valCert.valsLo k0 : Real) ≤ valsRealOfInputs inputs k0 ∧
                      valsRealOfInputs inputs k0 ≤ (valCert.valsHi k0 : Real) := by
                  have hv := hv_bounds k0
                  have hlo' : ∀ d, (vLo k0 d : Real) ≤ vRealOfInputs inputs k0 d := fun d =>
                    (hv d).1
                  have hhi' : ∀ d, vRealOfInputs inputs k0 d ≤ (vHi k0 d : Real) := fun d =>
                    (hv d).2
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
                    simpa [valsLo, valCert, dirHead, valsRealOfInputs,
                      Bounds.cacheBound_apply, Bounds.dotIntervalLowerCachedRat_eq] using hlow
                  have hhigh' :
                      valsRealOfInputs inputs k0 ≤ (valCert.valsHi k0 : Real) := by
                    simpa [valsHi, valCert, dirHead, valsRealOfInputs,
                      Bounds.cacheBound_apply, Bounds.dotIntervalUpperCachedRat_eq] using hhigh
                  exact ⟨hlow', hhigh'⟩
                have hhi : (valCert.valsHi k0 : Real) ≤ (valCert.hi : Real) := by
                  have hhiDyadic : valCert.valsHi k0 ≤ valCert.hi := by
                    change valsHi k0 ≤ hi
                    dsimp [hi]
                    refine (Finset.le_sup'_iff (s := univ) (H := hnonempty)
                      (f := valsHi) (a := valsHi k0)).2 ?_
                    exact ⟨k0, ⟨hmem0, le_rfl⟩⟩
                  exact dyadicToReal_le_of_le hhiDyadic
                have hreal :
                    (valCert.lo : Real) ≤ (valCert.hi : Real) :=
                  le_trans hlo (le_trans hvals.1 (le_trans hvals.2 hhi))
                exact (dyadicToReal_le_iff (x := valCert.lo) (y := valCert.hi)).1 hreal
              · intro k
                have hmem : k ∈ univ := by simp [univ]
                have hloDyadic : valCert.lo ≤ valCert.valsLo k := by
                  change lo ≤ valsLo k
                  dsimp [lo]
                  refine (Finset.inf'_le_iff (s := univ) (H := hnonempty)
                    (f := valsLo) (a := valsLo k)).2 ?_
                  refine ⟨k, hmem, ?_⟩
                  exact le_rfl
                exact dyadicToReal_le_of_le hloDyadic
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
                  simpa [valsLo, valCert, dirHead, valsRealOfInputs,
                    Bounds.cacheBound_apply, Bounds.dotIntervalLowerCachedRat_eq] using hlow
                have hhigh' :
                    valsRealOfInputs inputs k ≤ (valCert.valsHi k : Real) := by
                  simpa [valsHi, valCert, dirHead, valsRealOfInputs,
                    Bounds.cacheBound_apply, Bounds.dotIntervalUpperCachedRat_eq] using hhigh
                exact ⟨hlow', hhigh'⟩
              · intro k
                have hmem : k ∈ univ := by simp [univ]
                have hhiDyadic : valCert.valsHi k ≤ valCert.hi := by
                  change valsHi k ≤ hi
                  dsimp [hi]
                  refine (Finset.le_sup'_iff (s := univ) (H := hnonempty)
                    (f := valsHi) (a := valsHi k)).2 ?_
                  refine ⟨k, hmem, ?_⟩
                  exact le_rfl
                exact dyadicToReal_le_of_le hhiDyadic
            exact
              { softmax_bounds := hsoftmax_bounds
                oneHot_bounds_at := oneHot_bounds_at
                value_bounds := hvals_bounds }
          · have : False := by
              simp [buildInductionCertFromHeadCore?, hEps, hSqrt, hmodel, hactive] at hcore
            exact this.elim
      · have : False := by
          simp [buildInductionCertFromHeadCore?, hEps, hSqrt] at hcore
        exact this.elim
    · have : False := by
        simp [buildInductionCertFromHeadCore?, hEps] at hcore
      exact this.elim
  
end Sound

end Nfp
