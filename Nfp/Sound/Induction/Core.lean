-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Nfp.Core.Basic
import Mathlib.Data.Finset.Lattice.Fold
import Nfp.Circuit.Cert.ResidualInterval
import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange
import Nfp.Sound.Bounds.Attention
import Nfp.Sound.Bounds.Cache
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
          let lnLo : Fin seq → Fin dModel → Rat := lnBounds.1
          let lnHi : Fin seq → Fin dModel → Rat := lnBounds.2
          let lnAbsMaxTask : Fin seq → Rat :=
            Bounds.cacheBoundTask (fun q =>
              Bounds.intervalAbsBound (lnLo q) (lnHi q))
          let lnAbsMaxArr : Array Rat :=
            Array.ofFn (fun q : Fin seq => lnAbsMaxTask q)
          let lnAbsMax : Fin seq → Rat := fun q =>
            lnAbsMaxArr[q.1]'(by
              have hsize : lnAbsMaxArr.size = seq := by
                simp [lnAbsMaxArr]
              simp [hsize])
          let lnAbsMaxMax : Rat :=
            let univ : Finset (Fin seq) := Finset.univ
            have hnonempty : univ.Nonempty := Finset.univ_nonempty
            univ.sup' hnonempty (fun q => lnAbsMax q)
          let qAbsRowTasks : Array (Task { row : Array Rat // row.size = dHead }) :=
            Array.ofFn (fun q : Fin seq =>
              Task.spawn (fun _ =>
                ⟨Array.ofFn (fun d : Fin dHead =>
                    Bounds.dotIntervalAbsBound (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
                      |inputs.bq d|),
                  by simp⟩))
          let qAbsBaseArr : Array { row : Array Rat // row.size = dHead } :=
            Array.ofFn (fun q : Fin seq =>
              (qAbsRowTasks[q.1]'(by
                have hsize : qAbsRowTasks.size = seq := by
                  simp [qAbsRowTasks]
                simp [hsize])).get)
          let qAbsBase : Fin seq → Fin dHead → Rat := fun q d =>
            let row := qAbsBaseArr[q.1]'(by
              have hsize : qAbsBaseArr.size = seq := by
                simp [qAbsBaseArr]
              simp [hsize])
            row.1[d.1]'(by
              simp [row.2])
          let kAbsRowTasks : Array (Task { row : Array Rat // row.size = dHead }) :=
            Array.ofFn (fun q : Fin seq =>
              Task.spawn (fun _ =>
                ⟨Array.ofFn (fun d : Fin dHead =>
                    Bounds.dotIntervalAbsBound (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
                      |inputs.bk d|),
                  by simp⟩))
          let kAbsBaseArr : Array { row : Array Rat // row.size = dHead } :=
            Array.ofFn (fun q : Fin seq =>
              (kAbsRowTasks[q.1]'(by
                have hsize : kAbsRowTasks.size = seq := by
                  simp [kAbsRowTasks]
                simp [hsize])).get)
          let kAbsBase : Fin seq → Fin dHead → Rat := fun q d =>
            let row := kAbsBaseArr[q.1]'(by
              have hsize : kAbsBaseArr.size = seq := by
                simp [kAbsBaseArr]
              simp [hsize])
            row.1[d.1]'(by
              simp [row.2])
          let qLo : Fin seq → Fin dHead → Rat := fun q d => -qAbsBase q d
          let qHi : Fin seq → Fin dHead → Rat := fun q d => qAbsBase q d
          let kLo : Fin seq → Fin dHead → Rat := fun q d => -kAbsBase q d
          let kHi : Fin seq → Fin dHead → Rat := fun q d => kAbsBase q d
          let qAbs : Fin seq → Fin dHead → Rat := qAbsBase
          let kAbs : Fin seq → Fin dHead → Rat := kAbsBase
          let masked : Fin seq → Fin seq → Prop := fun q k =>
            inputs.maskCausal = true ∧ q < k
          let dotAbs :=
            Bounds.cacheBound2Task (fun q k =>
              Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d))
          let scoreBaseAbs : Fin seq → Fin seq → Rat := fun q k =>
            |inputs.scale| * dotAbs q k
          let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
            if masked q k then inputs.maskValue else -scoreBaseAbs q k
          let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
            if masked q k then inputs.maskValue else scoreBaseAbs q k
          let scoreLoPrev : Fin seq → Rat := fun q =>
            scoreLo q (inputs.prev q)
          let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
            (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
          let marginAt : Fin seq → Rat := fun q =>
            if hq : q ∈ inputs.active then
              let other := otherKeys q
              if h : other.Nonempty then
                other.inf' h (fun k => scoreLoPrev q - scoreHi q k)
              else
                (0 : Rat)
            else
              (0 : Rat)
          let epsAt : Fin seq → Rat := fun q =>
            if marginAt q < 0 then
              (1 : Rat)
            else
              ratDivUp (seq - 1) (1 + marginAt q)
          let margin : Rat :=
            if h : inputs.active.Nonempty then
              inputs.active.inf' h marginAt
            else
              (0 : Rat)
          let eps : Rat :=
            if margin < 0 then
              (1 : Rat)
            else
              ratDivUp (seq - 1) (1 + margin)
          let dirHeadVec := dirHeadVecOfInputs inputs
          let dirHead : Fin dHead → Rat := fun d => dirHeadVec.get d
          let wvDir : Fin dModel → Rat :=
            Bounds.cacheBoundTask (fun j =>
              Linear.dotFin dHead dirHead (fun d => inputs.wv j d))
          let bDir : Rat :=
            Linear.dotFin dHead dirHead (fun d => inputs.bv d)
          let valsAbsBase : Rat :=
            Linear.sumFin dModel (fun j => |wvDir j|) * lnAbsMaxMax
          let valsLoBase := bDir - valsAbsBase
          let valsHiBase := bDir + valsAbsBase
          let valsLo : Fin seq → Rat := fun _ => valsLoBase
          let valsHi : Fin seq → Rat := fun _ => valsHiBase
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
            let lnLo : Fin seq → Fin dModel → Rat := lnBounds.1
            let lnHi : Fin seq → Fin dModel → Rat := lnBounds.2
            let lnAbsMaxTask : Fin seq → Rat :=
              Bounds.cacheBoundTask (fun q =>
                Bounds.intervalAbsBound (lnLo q) (lnHi q))
            let lnAbsMaxArr : Array Rat :=
              Array.ofFn (fun q : Fin seq => lnAbsMaxTask q)
            let lnAbsMax : Fin seq → Rat := fun q =>
              lnAbsMaxArr[q.1]'(by
                have hsize : lnAbsMaxArr.size = seq := by
                  simp [lnAbsMaxArr]
                simp [hsize])
            let lnAbsMaxMax : Rat :=
              let univ : Finset (Fin seq) := Finset.univ
              have hnonempty : univ.Nonempty := Finset.univ_nonempty
              univ.sup' hnonempty (fun q => lnAbsMax q)
            let qAbsRowTasks : Array (Task { row : Array Rat // row.size = dHead }) :=
              Array.ofFn (fun q : Fin seq =>
                Task.spawn (fun _ =>
                  ⟨Array.ofFn (fun d : Fin dHead =>
                      Bounds.dotIntervalAbsBound (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
                        |inputs.bq d|),
                    by simp⟩))
            let qAbsBaseArr : Array { row : Array Rat // row.size = dHead } :=
              Array.ofFn (fun q : Fin seq =>
                (qAbsRowTasks[q.1]'(by
                  have hsize : qAbsRowTasks.size = seq := by
                    simp [qAbsRowTasks]
                  simp [hsize])).get)
            let qAbsBase : Fin seq → Fin dHead → Rat := fun q d =>
              let row := qAbsBaseArr[q.1]'(by
                have hsize : qAbsBaseArr.size = seq := by
                  simp [qAbsBaseArr]
                simp [hsize])
              row.1[d.1]'(by
                simp [row.2])
            let kAbsRowTasks : Array (Task { row : Array Rat // row.size = dHead }) :=
              Array.ofFn (fun q : Fin seq =>
                Task.spawn (fun _ =>
                  ⟨Array.ofFn (fun d : Fin dHead =>
                      Bounds.dotIntervalAbsBound (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
                        |inputs.bk d|),
                    by simp⟩))
            let kAbsBaseArr : Array { row : Array Rat // row.size = dHead } :=
              Array.ofFn (fun q : Fin seq =>
                (kAbsRowTasks[q.1]'(by
                  have hsize : kAbsRowTasks.size = seq := by
                    simp [kAbsRowTasks]
                  simp [hsize])).get)
            let kAbsBase : Fin seq → Fin dHead → Rat := fun q d =>
              let row := kAbsBaseArr[q.1]'(by
                have hsize : kAbsBaseArr.size = seq := by
                  simp [kAbsBaseArr]
                simp [hsize])
              row.1[d.1]'(by
                simp [row.2])
            let qLo : Fin seq → Fin dHead → Rat := fun q d => -qAbsBase q d
            let qHi : Fin seq → Fin dHead → Rat := fun q d => qAbsBase q d
            let kLo : Fin seq → Fin dHead → Rat := fun q d => -kAbsBase q d
            let kHi : Fin seq → Fin dHead → Rat := fun q d => kAbsBase q d
            let qAbs : Fin seq → Fin dHead → Rat := qAbsBase
            let kAbs : Fin seq → Fin dHead → Rat := kAbsBase
            let masked : Fin seq → Fin seq → Prop := fun q k =>
              inputs.maskCausal = true ∧ q < k
            let dotAbs :=
              Bounds.cacheBound2Task (fun q k =>
                Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d))
            let scoreBaseAbs : Fin seq → Fin seq → Rat := fun q k =>
              |inputs.scale| * dotAbs q k
            let scoreAbs : Fin seq → Fin seq → Rat := fun q k =>
              if masked q k then |inputs.maskValue| else scoreBaseAbs q k
            let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
              if masked q k then inputs.maskValue else -scoreBaseAbs q k
            let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
              if masked q k then inputs.maskValue else scoreBaseAbs q k
            let scoreLoPrev : Fin seq → Rat := fun q =>
              scoreLo q (inputs.prev q)
            let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
              (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
            let marginAt : Fin seq → Rat := fun q =>
              if hq : q ∈ inputs.active then
                let other := otherKeys q
                if h : other.Nonempty then
                  other.inf' h (fun k => scoreLoPrev q - scoreHi q k)
                else
                  (0 : Rat)
              else
                (0 : Rat)
            let epsAt : Fin seq → Rat := fun q =>
              if marginAt q < 0 then
                (1 : Rat)
              else
                ratDivUp (seq - 1) (1 + marginAt q)
            let margin : Rat :=
              if h : inputs.active.Nonempty then
                inputs.active.inf' h marginAt
              else
                (0 : Rat)
            let eps : Rat :=
              if margin < 0 then
                (1 : Rat)
              else
                ratDivUp (seq - 1) (1 + margin)
            have hseq : (1 : Nat) ≤ seq :=
              Nat.succ_le_iff.mpr (Nat.pos_of_ne_zero (NeZero.ne seq))
            let dirHeadVec := dirHeadVecOfInputs inputs
            let dirHead : Fin dHead → Rat := fun d => dirHeadVec.get d
            let wvDir : Fin dModel → Rat :=
              Bounds.cacheBoundTask (fun j =>
                Linear.dotFin dHead dirHead (fun d => inputs.wv j d))
            let bDir : Rat :=
              Linear.dotFin dHead dirHead (fun d => inputs.bv d)
            let valsAbsBase : Rat :=
              Linear.sumFin dModel (fun j => |wvDir j|) * lnAbsMaxMax
            let valsLoBase := bDir - valsAbsBase
            let valsHiBase := bDir + valsAbsBase
            let valsLo : Fin seq → Rat := fun _ => valsLoBase
            let valsHi : Fin seq → Rat := fun _ => valsHiBase
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
                [buildInductionCertFromHeadCore?, hEps, hSqrt, hmodel, hactive, lnBounds,
                  lnLo, lnHi, lnAbsMaxTask, lnAbsMaxArr, lnAbsMax, lnAbsMaxMax,
                  qAbsRowTasks, qAbsBaseArr, qAbsBase, kAbsRowTasks, kAbsBaseArr, kAbsBase,
                  qLo, qHi, kLo, kHi, qAbs, kAbs, masked, dotAbs, scoreBaseAbs, scoreLo,
                  scoreHi, scoreLoPrev, otherKeys, marginAt, epsAt, margin, eps, dirHeadVec,
                  dirHead, wvDir, bDir, valsAbsBase, valsLoBase, valsHiBase, valsLo,
                  valsHi, univ, lo, hi, valCert, cert, Task.spawn, Bounds.cacheBoundTask_apply,
                  Bounds.cacheBound2Task_apply, Array.getElem_ofFn]
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
            have hln_abs : ∀ q j, |lnRealOfInputs inputs q j| ≤ (lnAbsMax q : Real) := by
              intro q j
              have hln := hln_bounds q
              have h :=
                Bounds.abs_le_intervalAbsBound_real (lo := lnLo q) (hi := lnHi q)
                  (x := lnRealOfInputs inputs q) (hlo := fun j => (hln j).1)
                  (hhi := fun j => (hln j).2) j
              simpa [lnAbsMax, lnAbsMaxArr, lnAbsMaxTask, Bounds.cacheBoundTask_apply,
                Array.getElem_ofFn] using h
            have hln_abs_max : ∀ q, lnAbsMax q ≤ lnAbsMaxMax := by
              intro q
              have hnonempty : (Finset.univ : Finset (Fin seq)).Nonempty :=
                Finset.univ_nonempty
              have hmem : q ∈ (Finset.univ : Finset (Fin seq)) := by simp
              simpa [lnAbsMaxMax] using
                (Finset.le_sup'_iff (s := (Finset.univ : Finset (Fin seq)))
                  (H := hnonempty) (f := fun q => lnAbsMax q) (a := lnAbsMax q)).2
                  ⟨q, hmem, le_rfl⟩
            have hdot_abs_bound :
                ∀ (v : Fin dModel → Rat) (q : Fin seq),
                  |dotProduct (fun j => (v j : Real)) (lnRealOfInputs inputs q)| ≤
                    (Bounds.dotIntervalAbsBound v (lnLo q) (lnHi q) : Real) := by
              intro v q
              have hln := hln_bounds q
              have hlo : ∀ j, (lnLo q j : Real) ≤ lnRealOfInputs inputs q j := fun j =>
                (hln j).1
              have hhi : ∀ j, lnRealOfInputs inputs q j ≤ (lnHi q j : Real) := fun j =>
                (hln j).2
              simpa using
                (Bounds.abs_dotProduct_le_dotIntervalAbsBound_real
                  (v := v) (lo := lnLo q) (hi := lnHi q)
                  (x := lnRealOfInputs inputs q) hlo hhi)
            have hdot_abs_bound_sum :
                ∀ (v : Fin dModel → Rat) (q : Fin seq),
                  |dotProduct (fun j => (v j : Real)) (lnRealOfInputs inputs q)| ≤
                    (Linear.sumFin dModel (fun j => |v j|) : Real) * (lnAbsMax q : Real) := by
              intro v q
              have hsum :
                  |∑ j, (v j : Real) * lnRealOfInputs inputs q j| ≤
                    ∑ j, |(v j : Real) * lnRealOfInputs inputs q j| := by
                simpa [dotProduct] using
                  (Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset (Fin dModel)))
                    (f := fun j => (v j : Real) * lnRealOfInputs inputs q j))
              have hterm :
                  ∀ j, |(v j : Real) * lnRealOfInputs inputs q j| ≤
                    (|v j| : Real) * (lnAbsMax q : Real) := by
                intro j
                have hln := hln_abs q j
                have hnonneg : 0 ≤ (|v j| : Real) := by
                  exact abs_nonneg _
                calc
                  |(v j : Real) * lnRealOfInputs inputs q j| =
                      |(v j : Real)| * |lnRealOfInputs inputs q j| := by
                        simp [abs_mul]
                  _ ≤ (|v j| : Real) * (lnAbsMax q : Real) :=
                    mul_le_mul_of_nonneg_left hln hnonneg
              have hsum_le :
                  ∑ j, |(v j : Real) * lnRealOfInputs inputs q j| ≤
                    ∑ j, (|v j| : Real) * (lnAbsMax q : Real) := by
                refine Finset.sum_le_sum ?_
                intro j _
                exact hterm j
              have hsum_mul :
                  ∑ j, (|v j| : Real) * (lnAbsMax q : Real) =
                    (∑ j, (|v j| : Real)) * (lnAbsMax q : Real) := by
                symm
                simpa using
                  (Finset.sum_mul (s := (Finset.univ : Finset (Fin dModel)))
                    (f := fun j => (|v j| : Real)) (a := (lnAbsMax q : Real)))
              have hsum_cast :
                  (Linear.sumFin dModel (fun j => |v j|) : Real) = ∑ j, (|v j| : Real) := by
                simpa [ratToReal] using
                  (Linear.ratToReal_sumFin (f := fun j => |v j|))
              have hsum_eq :
                  ∑ j, (|v j| : Real) * (lnAbsMax q : Real) =
                    (Linear.sumFin dModel (fun j => |v j|) : Real) * (lnAbsMax q : Real) := by
                calc
                  ∑ j, (|v j| : Real) * (lnAbsMax q : Real)
                      = (∑ j, (|v j| : Real)) * (lnAbsMax q : Real) := hsum_mul
                  _ = (Linear.sumFin dModel (fun j => |v j|) : Real) * (lnAbsMax q : Real) := by
                    simp [hsum_cast]
              have hfinal := hsum.trans (hsum_le.trans_eq hsum_eq)
              simpa [dotProduct] using hfinal
            have hq_bounds :
                ∀ q d, (qLo q d : Real) ≤ qRealOfInputs inputs q d ∧
                  qRealOfInputs inputs q d ≤ (qHi q d : Real) := by
              intro q d
              have hdot := hdot_abs_bound (fun j => inputs.wq j d) q
              have hq_abs :
                  |qRealOfInputs inputs q d| ≤ (qAbsBase q d : Real) := by
                have hsum :
                    |qRealOfInputs inputs q d| ≤
                      (Bounds.dotIntervalAbsBound (fun j => inputs.wq j d) (lnLo q) (lnHi q) :
                          Real) + |(inputs.bq d : Real)| := by
                  calc
                    |qRealOfInputs inputs q d|
                        = |dotProduct (fun j => (inputs.wq j d : Real))
                            (lnRealOfInputs inputs q) + (inputs.bq d : Real)| := by
                          simp [qRealOfInputs]
                    _ ≤ |dotProduct (fun j => (inputs.wq j d : Real))
                            (lnRealOfInputs inputs q)| + |(inputs.bq d : Real)| := by
                          exact
                            (abs_add_le (a := dotProduct (fun j => (inputs.wq j d : Real))
                              (lnRealOfInputs inputs q)) (b := (inputs.bq d : Real)))
                    _ ≤ (Bounds.dotIntervalAbsBound (fun j => inputs.wq j d) (lnLo q) (lnHi q) :
                          Real) + |(inputs.bq d : Real)| := by
                          exact add_le_add hdot (le_rfl)
                have hsum' :
                    |qRealOfInputs inputs q d| ≤ (qAbsBase q d : Real) := by
                  simpa [qAbsBase, qAbsBaseArr, qAbsRowTasks, lnLo, lnHi, Task.spawn,
                    Bounds.dotIntervalAbsBound, ratToReal_add, ratToReal_abs]
                    using hsum
                exact hsum'
              have hq_bounds := (abs_le).1 hq_abs
              constructor
              · simpa [qLo] using hq_bounds.1
              · simpa [qHi] using hq_bounds.2
            have hk_bounds :
                ∀ q d, (kLo q d : Real) ≤ kRealOfInputs inputs q d ∧
                  kRealOfInputs inputs q d ≤ (kHi q d : Real) := by
              intro q d
              have hdot := hdot_abs_bound (fun j => inputs.wk j d) q
              have hk_abs :
                  |kRealOfInputs inputs q d| ≤ (kAbsBase q d : Real) := by
                have hsum :
                    |kRealOfInputs inputs q d| ≤
                      (Bounds.dotIntervalAbsBound (fun j => inputs.wk j d) (lnLo q) (lnHi q) :
                          Real) + |(inputs.bk d : Real)| := by
                  calc
                    |kRealOfInputs inputs q d|
                        = |dotProduct (fun j => (inputs.wk j d : Real))
                            (lnRealOfInputs inputs q) + (inputs.bk d : Real)| := by
                          simp [kRealOfInputs]
                    _ ≤ |dotProduct (fun j => (inputs.wk j d : Real))
                            (lnRealOfInputs inputs q)| + |(inputs.bk d : Real)| := by
                          exact
                            (abs_add_le (a := dotProduct (fun j => (inputs.wk j d : Real))
                              (lnRealOfInputs inputs q)) (b := (inputs.bk d : Real)))
                    _ ≤ (Bounds.dotIntervalAbsBound (fun j => inputs.wk j d) (lnLo q) (lnHi q) :
                          Real) + |(inputs.bk d : Real)| := by
                          exact add_le_add hdot (le_rfl)
                have hsum' :
                    |kRealOfInputs inputs q d| ≤ (kAbsBase q d : Real) := by
                  simpa [kAbsBase, kAbsBaseArr, kAbsRowTasks, lnLo, lnHi, Task.spawn,
                    Bounds.dotIntervalAbsBound, ratToReal_add, ratToReal_abs]
                    using hsum
                exact hsum'
              have hk_bounds := (abs_le).1 hk_abs
              constructor
              · simpa [kLo] using hk_bounds.1
              · simpa [kHi] using hk_bounds.2
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
                have hq' :
                    -(qAbsBase q d : Real) ≤ qRealOfInputs inputs q d ∧
                      qRealOfInputs inputs q d ≤ (qAbsBase q d : Real) := by
                  simpa [qLo, qHi] using hq
                have h := (abs_le).2 hq'
                simpa [qAbs, qAbsBase] using h
              have hk_abs : ∀ d, |kRealOfInputs inputs k d| ≤ (kAbs k d : Real) := by
                intro d
                have hk := hk_bounds k d
                have hk' :
                    -(kAbsBase k d : Real) ≤ kRealOfInputs inputs k d ∧
                      kRealOfInputs inputs k d ≤ (kAbsBase k d : Real) := by
                  simpa [kLo, kHi] using hk
                have h := (abs_le).2 hk'
                simpa [kAbs, kAbsBase] using h
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
                    have hdot_nonneg :
                        0 ≤ Bounds.dotIntervalAbsBound
                          (fun j => inputs.wq j d) (lnLo q) (lnHi q) := by
                      have hleft :
                          0 ≤ |Bounds.dotIntervalLower (fun j => inputs.wq j d)
                            (lnLo q) (lnHi q)| := by
                        exact abs_nonneg _
                      exact le_trans hleft (le_max_left _ _)
                    have hbq_nonneg : 0 ≤ |inputs.bq d| := abs_nonneg _
                    have hsum_nonneg :
                        0 ≤ Bounds.dotIntervalAbsBound
                          (fun j => inputs.wq j d) (lnLo q) (lnHi q) + |inputs.bq d| := by
                      exact add_nonneg hdot_nonneg hbq_nonneg
                    have hqnonneg' : 0 ≤ qAbs q d := by
                      simpa [qAbs, qAbsBase, qAbsBaseArr, qAbsRowTasks, lnLo, lnHi,
                        Task.spawn, Bounds.dotIntervalAbsBound] using hsum_nonneg
                    exact ratToReal_nonneg_of_nonneg hqnonneg'
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
                      ((∑ d, qAbs q d * kAbs k d : Rat) : Real) =
                        ∑ d, ((qAbs q d * kAbs k d : Rat) : Real) := by
                    have h := Linear.ratToReal_sum_univ (f := fun d => qAbs q d * kAbs k d)
                    dsimp [ratToReal] at h
                    exact h
                  have hsum' :
                      ∑ d, ((qAbs q d * kAbs k d : Rat) : Real) =
                        ∑ d, (qAbs q d : Real) * (kAbs k d : Real) := by
                    refine Finset.sum_congr rfl ?_
                    intro d _
                    simp
                  have hfinal := hsum.trans hsum'
                  calc
                    (dotAbs q k : Real)
                        = ((∑ d, qAbs q d * kAbs k d : Rat) : Real) := by
                          simp [dotAbs, Bounds.cacheBound2Task_apply,
                            Linear.dotFin_eq_dotProduct, dotProduct]
                    _ = ∑ d, (qAbs q d : Real) * (kAbs k d : Real) := hfinal
                have hfinal := hsum.trans (hsum_le.trans_eq hcast.symm)
                simpa [dotProduct] using hfinal
              have hscale_abs : 0 ≤ (|inputs.scale| : Real) := by
                exact abs_nonneg (ratToReal inputs.scale)
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
            have hmarginAt_le :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  marginAt q ≤ scoreLoPrev q - scoreHi q k := by
              intro q hq k hk
              have hmem : k ∈ otherKeys q := by
                simp [otherKeys, hk]
              have hnonempty : (otherKeys q).Nonempty := ⟨k, hmem⟩
              have hle :
                  (otherKeys q).inf' hnonempty (fun k => scoreLoPrev q - scoreHi q k) ≤
                    scoreLoPrev q - scoreHi q k := by
                exact
                  (Finset.inf'_le_iff (s := otherKeys q) (H := hnonempty)
                    (f := fun k => scoreLoPrev q - scoreHi q k)
                    (a := scoreLoPrev q - scoreHi q k)).2
                    ⟨k, hmem, le_rfl⟩
              simpa [marginAt, hq, hnonempty] using hle
            let weights : Fin seq → Fin seq → Real := fun q k =>
              Circuit.softmax (scoresReal q) k
            let others : Fin seq → Finset (Fin seq) := fun q =>
              (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
            have hscore_margin_real_at :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  scoresReal q k + (marginAt q : Real) ≤ scoresReal q (inputs.prev q) := by
              intro q hq k hk
              have hmargin_le : marginAt q ≤ scoreLoPrev q - scoreHi q k :=
                hmarginAt_le q hq k hk
              have hmargin_le_real :
                  (marginAt q : Real) ≤ (scoreLoPrev q : Real) - (scoreHi q k : Real) :=
                by
                  simpa [ratToReal_sub] using (ratToReal_le_of_le hmargin_le)
              have hscore_hi : scoresReal q k ≤ (scoreHi q k : Real) :=
                (hscore_bounds q k).2
              have hscore_prev : (scoreLoPrev q : Real) ≤ scoresReal q (inputs.prev q) := by
                have hprev_bounds := hscore_bounds q (inputs.prev q)
                simpa [scoreLoPrev] using hprev_bounds.1
              have hscore_diff : scoresReal q k - (scoreHi q k : Real) ≤ 0 := by
                have h := sub_le_sub_right hscore_hi (scoreHi q k : Real)
                simpa using h
              have hsum_le' :
                  (scoreLoPrev q : Real) - (scoreHi q k : Real) + scoresReal q k ≤
                    (scoreLoPrev q : Real) := by
                have hsub :
                    (scoreLoPrev q : Real) - (scoreHi q k : Real) ≤
                      (scoreLoPrev q : Real) - scoresReal q k :=
                  sub_le_sub_left hscore_hi (scoreLoPrev q : Real)
                have hsum_le'' := add_le_add_left hsub (scoresReal q k)
                have hsum_le''' :
                    (scoreLoPrev q : Real) - (scoreHi q k : Real) + scoresReal q k ≤
                      (scoreLoPrev q : Real) - scoresReal q k + scoresReal q k := by
                  simpa [add_comm, add_left_comm, add_assoc] using hsum_le''
                calc
                  (scoreLoPrev q : Real) - (scoreHi q k : Real) + scoresReal q k
                      ≤ (scoreLoPrev q : Real) - scoresReal q k + scoresReal q k := hsum_le'''
                  _ = (scoreLoPrev q : Real) := by
                    simp [sub_add_cancel]
              have hgap :
                  scoresReal q k + (marginAt q : Real) ≤ (scoreLoPrev q : Real) := by
                have hstep := add_le_add_left hmargin_le_real (scoresReal q k)
                have hstep' :
                    scoresReal q k + (marginAt q : Real) ≤
                      (scoreLoPrev q : Real) - (scoreHi q k : Real) + scoresReal q k := by
                  simpa [add_comm, add_left_comm, add_assoc] using hstep
                exact hstep'.trans hsum_le'
              exact hgap.trans hscore_prev
            have hscore_margin_real :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  scoresReal q k + (margin : Real) ≤ scoresReal q (inputs.prev q) := by
              intro q hq k hk
              have hmargin_le : margin ≤ marginAt q := by
                have hmem : q ∈ inputs.active := hq
                have hnonempty : inputs.active.Nonempty := hactive
                have hle :=
                  (Finset.inf'_le_iff (s := inputs.active) (H := hnonempty)
                    (f := marginAt) (a := marginAt q)).2 ⟨q, hmem, le_rfl⟩
                simpa [margin, hnonempty] using hle
              have hmargin_le_real : (margin : Real) ≤ (marginAt q : Real) :=
                ratToReal_le_of_le hmargin_le
              have hscore := hscore_margin_real_at q hq k hk
              have hscore' :
                  (marginAt q : Real) + scoresReal q k ≤ scoresReal q (inputs.prev q) := by
                simpa [add_comm, add_left_comm, add_assoc] using hscore
              have hstep := add_le_add_left hmargin_le_real (scoresReal q k)
              have hstep' :
                  scoresReal q k + (margin : Real) ≤ (marginAt q : Real) + scoresReal q k := by
                simpa [add_comm, add_left_comm, add_assoc] using hstep
              exact hstep'.trans hscore'
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
                      exact ratToReal_nonneg_of_nonneg hnonneg
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
                    have hpos : (0 : Rat) < 1 + margin := by
                      have hone : (0 : Rat) < 1 := by
                        exact zero_lt_one
                      have hle : (1 : Rat) ≤ 1 + margin := by
                        exact le_add_of_nonneg_right hnonneg
                      exact lt_of_lt_of_le hone hle
                    have hden : (1 + margin) ≠ 0 := by
                      exact ne_of_gt hpos
                    have hrat' := ratDivUp_ge_real (seq - 1) (1 + margin) hden
                    have heps :
                        (seq - 1 : Real) * (1 + (margin : Real))⁻¹ ≤ (eps : Real) := by
                      simpa [eps, hneg, ratToReal, Rat.cast_div, Rat.cast_add,
                        Rat.cast_natCast, div_eq_mul_inv] using hrat'
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
                  have hsum_le''' := hsum_le''
                  rw [add_comm (∑ k ∈ others q, weights q k)
                    (weights q (inputs.prev q))] at hsum_le'''
                  rw [add_comm (eps : Real) (weights q (inputs.prev q))] at hsum_le'''
                  exact hsum_le'''
                have hprev :
                    1 ≤ weights q (inputs.prev q) + (eps : Real) := by
                  have hsum_le'' := hsum_le'
                  rw [hsum_eq] at hsum_le''
                  exact hsum_le''
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
                      exact ratToReal_nonneg_of_nonneg hnonneg
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
                    have hpos : (0 : Rat) < 1 + margin := by
                      have hone : (0 : Rat) < 1 := by
                        exact zero_lt_one
                      have hle : (1 : Rat) ≤ 1 + margin := by
                        exact le_add_of_nonneg_right hnonneg
                      exact lt_of_lt_of_le hone hle
                    have hden : (1 + margin) ≠ 0 := by
                      exact ne_of_gt hpos
                    have hrat' := ratDivUp_ge_real (seq - 1) (1 + margin) hden
                    have heps :
                        (seq - 1 : Real) * (1 + (margin : Real))⁻¹ ≤ (eps : Real) := by
                      simpa [eps, hneg, ratToReal, Rat.cast_div, Rat.cast_add,
                        Rat.cast_natCast, div_eq_mul_inv] using hrat'
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
            have hepsAt :
                ∀ q, epsAt q =
                  if marginAt q < 0 then
                    (1 : Rat)
                  else
                    ratDivUp (seq - 1) (1 + marginAt q) := by
              intro q
              rfl
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
            have hdir_wv :
                ∀ j, (wvDir j : Real) = ∑ d, (dirHead d : Real) * (inputs.wv j d : Real) := by
              intro j
              have hsum :
                  ((∑ d, dirHead d * inputs.wv j d : Rat) : Real) =
                    ∑ d, ((dirHead d * inputs.wv j d : Rat) : Real) := by
                have h :=
                  Linear.ratToReal_sum_univ (f := fun d => dirHead d * inputs.wv j d)
                dsimp [ratToReal] at h
                exact h
              have hsum' :
                  ∑ d, ((dirHead d * inputs.wv j d : Rat) : Real) =
                    ∑ d, (dirHead d : Real) * (inputs.wv j d : Real) := by
                refine Finset.sum_congr rfl ?_
                intro d _
                simp
              have hfinal := hsum.trans hsum'
              calc
                (wvDir j : Real)
                    = ((∑ d, dirHead d * inputs.wv j d : Rat) : Real) := by
                      simp [wvDir, Bounds.cacheBoundTask_apply, Linear.dotFin_eq_dotProduct,
                        dotProduct]
                _ = ∑ d, (dirHead d : Real) * (inputs.wv j d : Real) := hfinal
            have hdir_bv :
                (bDir : Real) = ∑ d, (dirHead d : Real) * (inputs.bv d : Real) := by
              have hsum :
                  ((∑ d, dirHead d * inputs.bv d : Rat) : Real) =
                    ∑ d, ((dirHead d * inputs.bv d : Rat) : Real) := by
                have h :=
                  Linear.ratToReal_sum_univ (f := fun d => dirHead d * inputs.bv d)
                dsimp [ratToReal] at h
                exact h
              have hsum' :
                  ∑ d, ((dirHead d * inputs.bv d : Rat) : Real) =
                    ∑ d, (dirHead d : Real) * (inputs.bv d : Real) := by
                refine Finset.sum_congr rfl ?_
                intro d _
                simp
              have hfinal := hsum.trans hsum'
              calc
                (bDir : Real)
                    = ((∑ d, dirHead d * inputs.bv d : Rat) : Real) := by
                      simp [bDir, Linear.dotFin_eq_dotProduct, dotProduct]
                _ = ∑ d, (dirHead d : Real) * (inputs.bv d : Real) := hfinal
            have hvals_eq :
                ∀ k,
                  valsRealOfInputs inputs k =
                    dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) +
                      (bDir : Real) := by
              intro k
              classical
              have hdot_add :
                  dotProduct (fun d => (dirHead d : Real))
                      (fun d =>
                        dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs k) +
                          (inputs.bv d : Real)) =
                    dotProduct (fun d => (dirHead d : Real))
                        (fun d => dotProduct (fun j => (inputs.wv j d : Real))
                          (lnRealOfInputs inputs k)) +
                      dotProduct (fun d => (dirHead d : Real)) (fun d => (inputs.bv d : Real)) := by
                simp [dotProduct, mul_add, Finset.sum_add_distrib]
              have hdot_wv :
                  dotProduct (fun d => (dirHead d : Real))
                      (fun d => dotProduct (fun j => (inputs.wv j d : Real))
                        (lnRealOfInputs inputs k)) =
                    dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) := by
                classical
                calc
                  dotProduct (fun d => (dirHead d : Real))
                      (fun d => dotProduct (fun j => (inputs.wv j d : Real))
                        (lnRealOfInputs inputs k)) =
                    ∑ d, (dirHead d : Real) * ∑ j,
                      (inputs.wv j d : Real) * lnRealOfInputs inputs k j := by
                      simp [dotProduct]
                  _ = ∑ d, ∑ j,
                        (dirHead d : Real) *
                          ((inputs.wv j d : Real) * lnRealOfInputs inputs k j) := by
                      simp [Finset.mul_sum]
                  _ = ∑ j, ∑ d,
                        (dirHead d : Real) *
                          ((inputs.wv j d : Real) * lnRealOfInputs inputs k j) := by
                      simpa using
                        (Finset.sum_comm (s := (Finset.univ : Finset (Fin dHead)))
                          (t := (Finset.univ : Finset (Fin dModel)))
                          (f := fun d j =>
                            (dirHead d : Real) *
                              ((inputs.wv j d : Real) * lnRealOfInputs inputs k j)))
                  _ = ∑ j, (∑ d, (dirHead d : Real) * (inputs.wv j d : Real)) *
                        lnRealOfInputs inputs k j := by
                      refine Finset.sum_congr rfl ?_
                      intro j _
                      have hsum :
                          (∑ d, (dirHead d : Real) * (inputs.wv j d : Real)) *
                              lnRealOfInputs inputs k j =
                            ∑ d, (dirHead d : Real) * (inputs.wv j d : Real) *
                              lnRealOfInputs inputs k j := by
                        simp [Finset.sum_mul, mul_assoc]
                      simpa [mul_assoc] using hsum.symm
                  _ = ∑ j, (wvDir j : Real) * lnRealOfInputs inputs k j := by
                      refine Finset.sum_congr rfl ?_
                      intro j _
                      simp [hdir_wv j]
                  _ = dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) := by
                      simp [dotProduct]
              calc
                valsRealOfInputs inputs k =
                    dotProduct (fun d => (dirHead d : Real))
                      (fun d =>
                        dotProduct (fun j => (inputs.wv j d : Real))
                          (lnRealOfInputs inputs k) +
                          (inputs.bv d : Real)) := by
                  simp [valsRealOfInputs, vRealOfInputs, dirHeadVec, dirHead]
                _ =
                    dotProduct (fun d => (dirHead d : Real))
                        (fun d => dotProduct (fun j => (inputs.wv j d : Real))
                          (lnRealOfInputs inputs k)) +
                      dotProduct (fun d => (dirHead d : Real))
                        (fun d => (inputs.bv d : Real)) := hdot_add
                _ =
                    dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) +
                      dotProduct (fun d => (dirHead d : Real))
                        (fun d => (inputs.bv d : Real)) := by
                  simp [hdot_wv]
                _ =
                    dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) +
                      (bDir : Real) := by
                  have hb :
                      dotProduct (fun d => (dirHead d : Real))
                          (fun d => (inputs.bv d : Real)) =
                        (bDir : Real) := by
                    have hb : (dotProduct (fun d => (dirHead d : Real))
                        (fun d => (inputs.bv d : Real)) : Real) = (bDir : Real) := by
                      calc
                        dotProduct (fun d => (dirHead d : Real)) (fun d => (inputs.bv d : Real))
                            = ∑ d, (dirHead d : Real) * (inputs.bv d : Real) := by
                              simp [dotProduct]
                        _ = (bDir : Real) := hdir_bv.symm
                    exact hb
                  simp [hb]
            have hvals_bounds_at :
                ∀ k,
                  (valCert.valsLo k : Real) ≤ valsRealOfInputs inputs k ∧
                    valsRealOfInputs inputs k ≤ (valCert.valsHi k : Real) := by
              intro k
              have hdot_abs :
                  |dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k)| ≤
                    (valsAbsBase : Real) := by
                have hdot := hdot_abs_bound_sum (fun j => wvDir j) k
                have hln_max_real :
                    (lnAbsMax k : Real) ≤ (lnAbsMaxMax : Real) :=
                  ratToReal_le_of_le (hln_abs_max k)
                have hsum_nonneg : 0 ≤ (Linear.sumFin dModel (fun j => |wvDir j|) : Real) := by
                  have hsum_nonneg' : 0 ≤ (Linear.sumFin dModel (fun j => |wvDir j|) : Rat) := by
                    have hsum_nonneg'' : 0 ≤ ∑ j, |wvDir j| := by
                      refine Finset.sum_nonneg ?_
                      intro j _
                      exact abs_nonneg _
                    simpa [Linear.sumFin_eq_sum_univ] using hsum_nonneg''
                  exact ratToReal_nonneg_of_nonneg hsum_nonneg'
                have hmul :
                    (Linear.sumFin dModel (fun j => |wvDir j|) : Real) * (lnAbsMax k : Real) ≤
                      (Linear.sumFin dModel (fun j => |wvDir j|) : Real) * (lnAbsMaxMax : Real) :=
                  mul_le_mul_of_nonneg_left hln_max_real hsum_nonneg
                have hfinal := hdot.trans hmul
                simpa [valsAbsBase, ratToReal_mul] using hfinal
              have hdot_bounds := (abs_le).1 hdot_abs
              have hlow' := add_le_add_right hdot_bounds.1 (bDir : Real)
              have hhigh' := add_le_add_right hdot_bounds.2 (bDir : Real)
              have hlow :
                  (valCert.valsLo k : Real) ≤ valsRealOfInputs inputs k := by
                simpa [valCert, valsLo, valsLoBase, valsAbsBase, hvals_eq k, ratToReal_sub,
                  sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using hlow'
              have hhigh :
                  valsRealOfInputs inputs k ≤ (valCert.valsHi k : Real) := by
                simpa [valCert, valsHi, valsHiBase, valsAbsBase, hvals_eq k, ratToReal_add,
                  add_comm, add_left_comm, add_assoc] using hhigh'
              exact ⟨hlow, hhigh⟩
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
                  exact ratToReal_le_of_le hloRat
                have hvals :
                    (valCert.valsLo k0 : Real) ≤ valsRealOfInputs inputs k0 ∧
                      valsRealOfInputs inputs k0 ≤ (valCert.valsHi k0 : Real) := by
                  exact hvals_bounds_at k0
                have hhi : (valCert.valsHi k0 : Real) ≤ (valCert.hi : Real) := by
                  have hhiRat : valCert.valsHi k0 ≤ valCert.hi := by
                    change valsHi k0 ≤ hi
                    dsimp [hi]
                    refine (Finset.le_sup'_iff (s := univ) (H := hnonempty)
                      (f := valsHi) (a := valsHi k0)).2 ?_
                    exact ⟨k0, ⟨hmem0, le_rfl⟩⟩
                  exact ratToReal_le_of_le hhiRat
                have hreal :
                    (valCert.lo : Real) ≤ (valCert.hi : Real) :=
                  le_trans hlo (le_trans hvals.1 (le_trans hvals.2 hhi))
                exact (ratToReal_le_iff (x := valCert.lo) (y := valCert.hi)).1 hreal
              · intro k
                have hmem : k ∈ univ := by simp [univ]
                have hloRat : valCert.lo ≤ valCert.valsLo k := by
                  change lo ≤ valsLo k
                  dsimp [lo]
                  refine (Finset.inf'_le_iff (s := univ) (H := hnonempty)
                    (f := valsLo) (a := valsLo k)).2 ?_
                  refine ⟨k, hmem, ?_⟩
                  exact le_rfl
                exact ratToReal_le_of_le hloRat
              · intro k
                exact hvals_bounds_at k
              · intro k
                have hmem : k ∈ univ := by simp [univ]
                have hhiRat : valCert.valsHi k ≤ valCert.hi := by
                  change valsHi k ≤ hi
                  dsimp [hi]
                  refine (Finset.le_sup'_iff (s := univ) (H := hnonempty)
                    (f := valsHi) (a := valsHi k)).2 ?_
                  refine ⟨k, hmem, ?_⟩
                  exact le_rfl
                exact ratToReal_le_of_le hhiRat
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
