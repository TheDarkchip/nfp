-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Core.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Vector.Defs
import Nfp.Model.InductionHead
import Nfp.Sound.Bounds.Attention
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Linear.FinFold

/-!
Helper bounds for head-induction certificate construction.

These are pure precomputations that are useful for profiling and staging.
-/

namespace Nfp

namespace Sound

open Nfp.Sound.Bounds

variable {seq : Nat}

/-- Cached direction head for head inputs. -/
private def dirHeadVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector Dyadic dHead :=
  Vector.ofFn (fun d : Fin dHead =>
    Linear.dotFin dModel (fun j => inputs.wo j d) (fun j => inputs.direction j))

/-- LayerNorm bounds used by the induction-head builder. -/
def headLnBounds [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    (Fin seq → Fin dModel → Dyadic) × (Fin seq → Fin dModel → Dyadic) :=
  Bounds.cacheBoundPair2 (fun q =>
    Bounds.layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q))

theorem headLnBounds_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    headLnBounds inputs =
      Bounds.cacheBoundPair2 (fun q =>
        Bounds.layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)) := rfl

/-- Q/K/V bounds used by the induction-head builder. -/
structure HeadQKVBounds (seq dModel dHead : Nat) where
  /-- Q lower bounds. -/
  qLo : Fin seq → Fin dHead → Dyadic
  /-- Q upper bounds. -/
  qHi : Fin seq → Fin dHead → Dyadic
  /-- K lower bounds. -/
  kLo : Fin seq → Fin dHead → Dyadic
  /-- K upper bounds. -/
  kHi : Fin seq → Fin dHead → Dyadic
  /-- V lower bounds. -/
  vLo : Fin seq → Fin dHead → Dyadic
  /-- V upper bounds. -/
  vHi : Fin seq → Fin dHead → Dyadic
  /-- Q absolute bounds. -/
  qAbs : Fin seq → Fin dHead → Dyadic
  /-- K absolute bounds. -/
  kAbs : Fin seq → Fin dHead → Dyadic

/-- Compute Q/K/V bounds from LayerNorm bounds. -/
def headQKVBounds [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lnLo lnHi : Fin seq → Fin dModel → Dyadic) :
    HeadQKVBounds seq dModel dHead :=
  let qLo :=
    Bounds.cacheBound2TaskElem (fun q d =>
      Bounds.dotIntervalLowerUnnorm (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
        inputs.bq d)
  let qHi :=
    Bounds.cacheBound2TaskElem (fun q d =>
      Bounds.dotIntervalUpperUnnorm (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
        inputs.bq d)
  let kLo :=
    Bounds.cacheBound2TaskElem (fun q d =>
      Bounds.dotIntervalLowerUnnorm (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
        inputs.bk d)
  let kHi :=
    Bounds.cacheBound2TaskElem (fun q d =>
      Bounds.dotIntervalUpperUnnorm (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
        inputs.bk d)
  let vLo :=
    Bounds.cacheBound2TaskElem (fun q d =>
      Bounds.dotIntervalLowerUnnorm (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
        inputs.bv d)
  let vHi :=
    Bounds.cacheBound2TaskElem (fun q d =>
      Bounds.dotIntervalUpperUnnorm (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
        inputs.bv d)
  let qAbs :=
    Bounds.cacheBound2TaskElem (fun q d => max |qLo q d| |qHi q d|)
  let kAbs :=
    Bounds.cacheBound2TaskElem (fun q d => max |kLo q d| |kHi q d|)
  { qLo := qLo
    qHi := qHi
    kLo := kLo
    kHi := kHi
    vLo := vLo
    vHi := vHi
    qAbs := qAbs
    kAbs := kAbs }

theorem headQKVBounds_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lnLo lnHi : Fin seq → Fin dModel → Dyadic) :
    headQKVBounds inputs lnLo lnHi =
      let qLo :=
        Bounds.cacheBound2TaskElem (fun q d =>
          Bounds.dotIntervalLowerUnnorm (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
            inputs.bq d)
      let qHi :=
        Bounds.cacheBound2TaskElem (fun q d =>
          Bounds.dotIntervalUpperUnnorm (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
            inputs.bq d)
      let kLo :=
        Bounds.cacheBound2TaskElem (fun q d =>
          Bounds.dotIntervalLowerUnnorm (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
            inputs.bk d)
      let kHi :=
        Bounds.cacheBound2TaskElem (fun q d =>
          Bounds.dotIntervalUpperUnnorm (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
            inputs.bk d)
      let vLo :=
        Bounds.cacheBound2TaskElem (fun q d =>
          Bounds.dotIntervalLowerUnnorm (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
            inputs.bv d)
      let vHi :=
        Bounds.cacheBound2TaskElem (fun q d =>
          Bounds.dotIntervalUpperUnnorm (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
            inputs.bv d)
      let qAbs :=
        Bounds.cacheBound2TaskElem (fun q d => max |qLo q d| |qHi q d|)
      let kAbs :=
        Bounds.cacheBound2TaskElem (fun q d => max |kLo q d| |kHi q d|)
      { qLo := qLo
        qHi := qHi
        kLo := kLo
        kHi := kHi
        vLo := vLo
        vHi := vHi
        qAbs := qAbs
        kAbs := kAbs } := rfl

/-- Score and margin bounds used by the induction-head builder. -/
structure HeadScoreBounds (seq dModel dHead : Nat) where
  /-- Absolute dot-product bound. -/
  dotAbs : Fin seq → Fin seq → Dyadic
  /-- Base score absolute bound. -/
  scoreBaseAbs : Fin seq → Fin seq → Dyadic
  /-- Score absolute bound with causal masking. -/
  scoreAbs : Fin seq → Fin seq → Dyadic
  /-- Score lower bound. -/
  scoreLo : Fin seq → Fin seq → Dyadic
  /-- Score upper bound. -/
  scoreHi : Fin seq → Fin seq → Dyadic
  /-- Margin per query. -/
  marginAt : Fin seq → Dyadic
  /-- Epsilon per query. -/
  epsAt : Fin seq → Dyadic
  /-- Global margin. -/
  margin : Dyadic
  /-- Global epsilon. -/
  eps : Dyadic

/-- Compute score and margin bounds from cached score lower/upper bounds. -/
def headScoreBoundsFromCaches [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dotAbs : Fin seq → Fin seq → Dyadic)
    (scoreLo scoreHi : Fin seq → Fin seq → Dyadic) :
    HeadScoreBounds seq dModel dHead :=
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let scoreBaseAbs : Fin seq → Fin seq → Dyadic := fun q k =>
    |inputs.scale| * dotAbs q k
  let scoreAbs : Fin seq → Fin seq → Dyadic := fun q k =>
    if masked q k then |inputs.maskValue| else scoreBaseAbs q k
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
  let marginTasks : Array (Task Dyadic) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        if q ∈ inputs.active then
          let other := unmaskedKeys q
          let masked := maskedKeys q
          if hunmasked : other.Nonempty then
            let unmaskedMin := other.inf' hunmasked (fun k =>
              scoreLo q (inputs.prev q) - scoreHi q k)
            if hmasked : masked.Nonempty then
              min unmaskedMin (maskedGap q)
            else
              unmaskedMin
          else
            if hmasked : masked.Nonempty then
              maskedGap q
            else
              (0 : Dyadic)
        else
          (0 : Dyadic)))
  let marginAt : Fin seq → Dyadic := fun q =>
    (marginTasks[q.1]'(by
      simp [marginTasks, q.isLt])).get
  let epsTasks : Array (Task Dyadic) :=
    Array.ofFn (fun q : Fin seq =>
      (marginTasks[q.1]'(by
        simp [marginTasks, q.isLt])).map (fun m =>
          if m < 0 then
            (1 : Dyadic)
          else
            dyadicDivUp (seq - 1) (1 + m)))
  let epsAt : Fin seq → Dyadic := fun q =>
    (epsTasks[q.1]'(by
      simp [epsTasks, q.isLt])).get
  let margin : Dyadic :=
    if h : inputs.active.Nonempty then
      inputs.active.inf' h marginAt
    else
      (0 : Dyadic)
  let eps : Dyadic :=
    if margin < 0 then
      (1 : Dyadic)
    else
      dyadicDivUp (seq - 1) (1 + margin)
  { dotAbs := dotAbs
    scoreBaseAbs := scoreBaseAbs
    scoreAbs := scoreAbs
    scoreLo := scoreLo
    scoreHi := scoreHi
    marginAt := marginAt
    epsAt := epsAt
    margin := margin
    eps := eps }

/-- Compute score and margin bounds from dot-product absolute bounds. -/
def headScoreBoundsFromDotAbs [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dotAbs : Fin seq → Fin seq → Dyadic) : HeadScoreBounds seq dModel dHead :=
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let dotAbsRowTasks : Array (Task { row : Array Dyadic // row.size = seq }) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        ⟨Array.ofFn (fun k : Fin seq => dotAbs q k), by simp⟩))
  let scoreRowTasks : Array (Task (Array Dyadic × Array Dyadic)) :=
    Array.ofFn (fun q : Fin seq =>
      (dotAbsRowTasks[q.1]'(by
        simp [dotAbsRowTasks, q.isLt])).map (fun row =>
          let rowArr := row.1
          let scoreBaseAt : Fin seq → Dyadic := fun k =>
            |inputs.scale| * rowArr.getD k.1 0
          let loRow := Array.ofFn (fun k : Fin seq =>
            if masked q k then inputs.maskValue else -scoreBaseAt k)
          let hiRow := Array.ofFn (fun k : Fin seq =>
            if masked q k then inputs.maskValue else scoreBaseAt k)
          (loRow, hiRow)))
  let scoreLoCached : Fin seq → Fin seq → Dyadic := fun q k =>
    let rowPair := (scoreRowTasks[q.1]'(by
      simp [scoreRowTasks, q.isLt])).get
    rowPair.1.getD k.1 0
  let scoreHiCached : Fin seq → Fin seq → Dyadic := fun q k =>
    let rowPair := (scoreRowTasks[q.1]'(by
      simp [scoreRowTasks, q.isLt])).get
    rowPair.2.getD k.1 0
  headScoreBoundsFromCaches inputs dotAbs scoreLoCached scoreHiCached

theorem headScoreBoundsFromDotAbs_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dotAbs : Fin seq → Fin seq → Dyadic) :
  headScoreBoundsFromDotAbs inputs dotAbs =
      let masked : Fin seq → Fin seq → Prop := fun q k =>
        inputs.maskCausal = true ∧ q < k
      let dotAbsRowTasks : Array (Task { row : Array Dyadic // row.size = seq }) :=
        Array.ofFn (fun q : Fin seq =>
          Task.spawn (fun _ =>
            ⟨Array.ofFn (fun k : Fin seq => dotAbs q k), by simp⟩))
      let scoreRowTasks : Array (Task (Array Dyadic × Array Dyadic)) :=
        Array.ofFn (fun q : Fin seq =>
          (dotAbsRowTasks[q.1]'(by
            simp [dotAbsRowTasks, q.isLt])).map (fun row =>
              let rowArr := row.1
              let scoreBaseAt : Fin seq → Dyadic := fun k =>
                |inputs.scale| * rowArr.getD k.1 0
              let loRow := Array.ofFn (fun k : Fin seq =>
                if masked q k then inputs.maskValue else -scoreBaseAt k)
              let hiRow := Array.ofFn (fun k : Fin seq =>
                if masked q k then inputs.maskValue else scoreBaseAt k)
              (loRow, hiRow)))
      let scoreLoCached : Fin seq → Fin seq → Dyadic := fun q k =>
        let rowPair := (scoreRowTasks[q.1]'(by
          simp [scoreRowTasks, q.isLt])).get
        rowPair.1.getD k.1 0
      let scoreHiCached : Fin seq → Fin seq → Dyadic := fun q k =>
        let rowPair := (scoreRowTasks[q.1]'(by
          simp [scoreRowTasks, q.isLt])).get
        rowPair.2.getD k.1 0
      headScoreBoundsFromCaches inputs dotAbs scoreLoCached scoreHiCached := rfl

/-- Compute score and margin bounds from Q/K absolute bounds. -/
def headScoreBounds [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qAbs kAbs : Fin seq → Fin dHead → Dyadic) :
    HeadScoreBounds seq dModel dHead :=
  headScoreBoundsFromDotAbs inputs (fun q k =>
    Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d))

theorem headScoreBounds_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qAbs kAbs : Fin seq → Fin dHead → Dyadic) :
    headScoreBounds inputs qAbs kAbs =
      let masked : Fin seq → Fin seq → Prop := fun q k =>
        inputs.maskCausal = true ∧ q < k
      let dotAbs : Fin seq → Fin seq → Dyadic := fun q k =>
        Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d)
      let dotAbsRowTasks : Array (Task { row : Array Dyadic // row.size = seq }) :=
        Array.ofFn (fun q : Fin seq =>
          Task.spawn (fun _ =>
            ⟨Array.ofFn (fun k : Fin seq => dotAbs q k), by simp⟩))
      let scoreRowTasks : Array (Task (Array Dyadic × Array Dyadic)) :=
        Array.ofFn (fun q : Fin seq =>
          (dotAbsRowTasks[q.1]'(by
            simp [dotAbsRowTasks, q.isLt])).map (fun row =>
              let rowArr := row.1
              let scoreBaseAt : Fin seq → Dyadic := fun k =>
                |inputs.scale| * rowArr.getD k.1 0
              let loRow := Array.ofFn (fun k : Fin seq =>
                if masked q k then inputs.maskValue else -scoreBaseAt k)
              let hiRow := Array.ofFn (fun k : Fin seq =>
                if masked q k then inputs.maskValue else scoreBaseAt k)
              (loRow, hiRow)))
      let scoreLoCached : Fin seq → Fin seq → Dyadic := fun q k =>
        let rowPair := (scoreRowTasks[q.1]'(by
          simp [scoreRowTasks, q.isLt])).get
        rowPair.1.getD k.1 0
      let scoreHiCached : Fin seq → Fin seq → Dyadic := fun q k =>
        let rowPair := (scoreRowTasks[q.1]'(by
          simp [scoreRowTasks, q.isLt])).get
        rowPair.2.getD k.1 0
      headScoreBoundsFromCaches inputs dotAbs scoreLoCached scoreHiCached := rfl

/-- Value bounds used by the induction-head builder. -/
structure HeadValueBounds (seq dModel dHead : Nat) where
  /-- Value lower bounds. -/
  valsLo : Fin seq → Dyadic
  /-- Value upper bounds. -/
  valsHi : Fin seq → Dyadic
  /-- Global value lower bound. -/
  lo : Dyadic
  /-- Global value upper bound. -/
  hi : Dyadic

/-- Cached direction vector for value bounds. -/
def headValueDirHead {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin dHead → Dyadic :=
  let dirHeadVec := dirHeadVecOfInputs inputs
  fun d => dirHeadVec.get d

theorem headValueDirHead_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    headValueDirHead inputs =
      let dirHeadVec := dirHeadVecOfInputs inputs
      fun d => dirHeadVec.get d := rfl

/-- Cached lower value bounds from V intervals. -/
def headValueValsLo {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) : Fin seq → Dyadic :=
  let dirHead := headValueDirHead inputs
  Bounds.cacheBound (fun k =>
    Bounds.dotIntervalLowerCachedDyadic dirHead (vLo k) (vHi k))

theorem headValueValsLo_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueValsLo inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      Bounds.cacheBound (fun k =>
        Bounds.dotIntervalLowerCachedDyadic dirHead (vLo k) (vHi k)) := rfl

/-- Cached lower value bounds from V intervals using a common-denominator sum. -/
def headValueValsLoCommonDen {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) : Fin seq → Dyadic :=
  let dirHead := headValueDirHead inputs
  Bounds.cacheBound (fun k =>
    Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k))

theorem headValueValsLoCommonDen_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueValsLoCommonDen inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      Bounds.cacheBound (fun k =>
        Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k)) := rfl

theorem headValueValsLoCommonDen_eq {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueValsLoCommonDen inputs vLo vHi = headValueValsLo inputs vLo vHi := by
  classical
  funext k
  simp [headValueValsLoCommonDen, headValueValsLo, Bounds.cacheBound_apply,
    Bounds.dotIntervalLowerCommonDen_eq, Bounds.dotIntervalLowerCachedRat_eq]

/-- Cached upper value bounds from V intervals. -/
def headValueValsHi {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) : Fin seq → Dyadic :=
  let dirHead := headValueDirHead inputs
  Bounds.cacheBound (fun k =>
    Bounds.dotIntervalUpperCachedDyadic dirHead (vLo k) (vHi k))

theorem headValueValsHi_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueValsHi inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      Bounds.cacheBound (fun k =>
        Bounds.dotIntervalUpperCachedDyadic dirHead (vLo k) (vHi k)) := rfl

/-- Cached upper value bounds from V intervals using a common-denominator sum. -/
def headValueValsHiCommonDen {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) : Fin seq → Dyadic :=
  let dirHead := headValueDirHead inputs
  Bounds.cacheBound (fun k =>
    Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k))

theorem headValueValsHiCommonDen_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueValsHiCommonDen inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      Bounds.cacheBound (fun k =>
        Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k)) := rfl

theorem headValueValsHiCommonDen_eq {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueValsHiCommonDen inputs vLo vHi = headValueValsHi inputs vLo vHi := by
  classical
  funext k
  simp [headValueValsHiCommonDen, headValueValsHi, Bounds.cacheBound_apply,
    Bounds.dotIntervalUpperCommonDen_eq, Bounds.dotIntervalUpperCachedRat_eq]

/-- Global lower value bound from cached per-key values. -/
def headValueLo [NeZero seq] (valsLo : Fin seq → Dyadic) : Dyadic :=
  let univ : Finset (Fin seq) := Finset.univ
  have hnonempty : univ.Nonempty := by simp [univ]
  univ.inf' hnonempty valsLo

theorem headValueLo_spec [NeZero seq] (valsLo : Fin seq → Dyadic) :
    headValueLo valsLo =
      let univ : Finset (Fin seq) := Finset.univ
      have hnonempty : univ.Nonempty := by simp [univ]
      univ.inf' hnonempty valsLo := rfl

/-- Global upper value bound from cached per-key values. -/
def headValueHi [NeZero seq] (valsHi : Fin seq → Dyadic) : Dyadic :=
  let univ : Finset (Fin seq) := Finset.univ
  have hnonempty : univ.Nonempty := by simp [univ]
  univ.sup' hnonempty valsHi

theorem headValueHi_spec [NeZero seq] (valsHi : Fin seq → Dyadic) :
    headValueHi valsHi =
      let univ : Finset (Fin seq) := Finset.univ
      have hnonempty : univ.Nonempty := by simp [univ]
      univ.sup' hnonempty valsHi := rfl

/-- Compute value bounds from V interval bounds. -/
def headValueBounds [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    HeadValueBounds seq dModel dHead :=
  let valsLo := headValueValsLo inputs vLo vHi
  let valsHi := headValueValsHi inputs vLo vHi
  let lo := headValueLo valsLo
  let hi := headValueHi valsHi
  { valsLo := valsLo, valsHi := valsHi, lo := lo, hi := hi }

theorem headValueBounds_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueBounds inputs vLo vHi =
      let valsLo := headValueValsLo inputs vLo vHi
      let valsHi := headValueValsHi inputs vLo vHi
      let lo := headValueLo valsLo
      let hi := headValueHi valsHi
      { valsLo := valsLo, valsHi := valsHi, lo := lo, hi := hi } := rfl

/-- Compute value bounds from V interval bounds using a common-denominator sum. -/
def headValueBoundsCommonDen [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    HeadValueBounds seq dModel dHead :=
  let valsLo := headValueValsLoCommonDen inputs vLo vHi
  let valsHi := headValueValsHiCommonDen inputs vLo vHi
  let lo := headValueLo valsLo
  let hi := headValueHi valsHi
  { valsLo := valsLo, valsHi := valsHi, lo := lo, hi := hi }

theorem headValueBoundsCommonDen_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueBoundsCommonDen inputs vLo vHi =
      let valsLo := headValueValsLoCommonDen inputs vLo vHi
      let valsHi := headValueValsHiCommonDen inputs vLo vHi
      let lo := headValueLo valsLo
      let hi := headValueHi valsHi
      { valsLo := valsLo, valsHi := valsHi, lo := lo, hi := hi } := rfl

theorem headValueBoundsCommonDen_eq [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Dyadic) :
    headValueBoundsCommonDen inputs vLo vHi = headValueBounds inputs vLo vHi := by
  classical
  simp [headValueBoundsCommonDen, headValueBounds, headValueValsLoCommonDen_eq,
    headValueValsHiCommonDen_eq]

end Sound

end Nfp
