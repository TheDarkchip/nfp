-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Core.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Range
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

private def taskMin (t1 t2 : Task Rat) : Task Rat :=
  Task.bind t1 (fun v1 => Task.map (fun v2 => min v1 v2) t2)

private def taskMax (t1 t2 : Task Rat) : Task Rat :=
  Task.bind t1 (fun v1 => Task.map (fun v2 => max v1 v2) t2)

/-! Helpers for reducing cached arrays without extra allocation. -/

/-- Reduce an array of rational bounds to its minimum (defaulting to `0` on empty arrays). -/
private def reduceMinArray (arr : Array Rat) : Rat :=
  let init := arr.getD 0 (0 : Rat)
  arr.foldl (fun acc x => min acc x) init

/-- Reduce an array of rational bounds to its maximum (defaulting to `0` on empty arrays). -/
private def reduceMaxArray (arr : Array Rat) : Rat :=
  let init := arr.getD 0 (0 : Rat)
  arr.foldl (fun acc x => max acc x) init

/-- Reduce a `Fin seq`-indexed function in parallel using chunked tasks. -/
private def reduceFnTask [NeZero seq] (vals : Fin seq → Rat)
    (combine : Rat → Rat → Rat) (combineTask : Task Rat → Task Rat → Task Rat) : Task Rat :=
  let n := seq
  if n = 0 then
    Task.pure (0 : Rat)
  else
    let chunkSize : Nat := 256
    let chunks : Nat := (n + chunkSize - 1) / chunkSize
    let hpos : 0 < seq := Nat.pos_of_ne_zero (by simpa using (NeZero.ne (n := seq)))
    let defaultIdx : Fin seq := ⟨0, hpos⟩
    let idxs : Array (Fin seq) := Array.ofFn (fun i : Fin seq => i)
    let defaultTask : Task Rat := Task.pure (0 : Rat)
    let chunkTasks : Array (Task Rat) :=
      Array.ofFn (fun c : Fin chunks =>
        Task.spawn (fun _ =>
          let start := c.val * chunkSize
          let stop := Nat.min n (start + chunkSize)
          let init := vals (idxs.getD start defaultIdx)
          if stop ≤ start + 1 then
            init
          else
            let rest := (List.range (stop - start - 1)).map (fun i => start + i + 1)
            rest.foldl (fun acc i => combine acc (vals (idxs.getD i defaultIdx))) init))
    let init := chunkTasks.getD 0 defaultTask
    let rest := (List.range (chunkTasks.size - 1)).map (fun i => i + 1)
    rest.foldl (fun acc i => combineTask acc (chunkTasks.getD i defaultTask)) init

/-- Unfold `reduceFnTask` to its chunked-task definition. -/
theorem reduceFnTask_spec [NeZero seq] (vals : Fin seq → Rat)
    (combine : Rat → Rat → Rat) (combineTask : Task Rat → Task Rat → Task Rat) :
    reduceFnTask (seq := seq) vals combine combineTask =
      let n := seq
      if n = 0 then
        Task.pure (0 : Rat)
      else
        let chunkSize : Nat := 256
        let chunks : Nat := (n + chunkSize - 1) / chunkSize
        let hpos : 0 < seq := Nat.pos_of_ne_zero (by simpa using (NeZero.ne (n := seq)))
        let defaultIdx : Fin seq := ⟨0, hpos⟩
        let idxs : Array (Fin seq) := Array.ofFn (fun i : Fin seq => i)
        let defaultTask : Task Rat := Task.pure (0 : Rat)
        let chunkTasks : Array (Task Rat) :=
          Array.ofFn (fun c : Fin chunks =>
            Task.spawn (fun _ =>
              let start := c.val * chunkSize
              let stop := Nat.min n (start + chunkSize)
              let init := vals (idxs.getD start defaultIdx)
              if stop ≤ start + 1 then
                init
              else
                let rest := (List.range (stop - start - 1)).map (fun i => start + i + 1)
                rest.foldl (fun acc i => combine acc (vals (idxs.getD i defaultIdx))) init))
        let init := chunkTasks.getD 0 defaultTask
        let rest := (List.range (chunkTasks.size - 1)).map (fun i => i + 1)
        rest.foldl (fun acc i => combineTask acc (chunkTasks.getD i defaultTask)) init := rfl

private def reduceMinFnTask [NeZero seq] (vals : Fin seq → Rat) : Task Rat :=
  reduceFnTask vals min taskMin

private def reduceMaxFnTask [NeZero seq] (vals : Fin seq → Rat) : Task Rat :=
  reduceFnTask vals max taskMax

/-- Cached direction head for head inputs. -/
private def dirHeadVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector Rat dHead :=
  Vector.ofFn (fun d : Fin dHead =>
    Linear.dotFin dModel (fun j => inputs.wo j d) (fun j => inputs.direction j))

/-- LayerNorm bounds used by the induction-head builder. -/
def headLnBounds [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    (Fin seq → Fin dModel → Rat) × (Fin seq → Fin dModel → Rat) :=
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
  qLo : Fin seq → Fin dHead → Rat
  /-- Q upper bounds. -/
  qHi : Fin seq → Fin dHead → Rat
  /-- K lower bounds. -/
  kLo : Fin seq → Fin dHead → Rat
  /-- K upper bounds. -/
  kHi : Fin seq → Fin dHead → Rat
  /-- V lower bounds. -/
  vLo : Fin seq → Fin dHead → Rat
  /-- V upper bounds. -/
  vHi : Fin seq → Fin dHead → Rat
  /-- Q absolute bounds. -/
  qAbs : Fin seq → Fin dHead → Rat
  /-- K absolute bounds. -/
  kAbs : Fin seq → Fin dHead → Rat

/-- Compute Q/K/V bounds from LayerNorm bounds. -/
def headQKVBounds [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lnLo lnHi : Fin seq → Fin dModel → Rat) :
    HeadQKVBounds seq dModel dHead :=
  let qLo :=
    Bounds.cacheBound2 (fun q d =>
      Bounds.dotIntervalLowerUnnorm (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
        inputs.bq d)
  let qHi :=
    Bounds.cacheBound2 (fun q d =>
      Bounds.dotIntervalUpperUnnorm (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
        inputs.bq d)
  let kLo :=
    Bounds.cacheBound2 (fun q d =>
      Bounds.dotIntervalLowerUnnorm (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
        inputs.bk d)
  let kHi :=
    Bounds.cacheBound2 (fun q d =>
      Bounds.dotIntervalUpperUnnorm (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
        inputs.bk d)
  let vLo :=
    Bounds.cacheBound2 (fun q d =>
      Bounds.dotIntervalLowerUnnorm (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
        inputs.bv d)
  let vHi :=
    Bounds.cacheBound2 (fun q d =>
      Bounds.dotIntervalUpperUnnorm (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
        inputs.bv d)
  let qAbs :=
    Bounds.cacheBound2 (fun q d => max |qLo q d| |qHi q d|)
  let kAbs :=
    Bounds.cacheBound2 (fun q d => max |kLo q d| |kHi q d|)
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
    (lnLo lnHi : Fin seq → Fin dModel → Rat) :
    headQKVBounds inputs lnLo lnHi =
      let qLo :=
        Bounds.cacheBound2 (fun q d =>
          Bounds.dotIntervalLowerUnnorm (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
            inputs.bq d)
      let qHi :=
        Bounds.cacheBound2 (fun q d =>
          Bounds.dotIntervalUpperUnnorm (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
            inputs.bq d)
      let kLo :=
        Bounds.cacheBound2 (fun q d =>
          Bounds.dotIntervalLowerUnnorm (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
            inputs.bk d)
      let kHi :=
        Bounds.cacheBound2 (fun q d =>
          Bounds.dotIntervalUpperUnnorm (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
            inputs.bk d)
      let vLo :=
        Bounds.cacheBound2 (fun q d =>
          Bounds.dotIntervalLowerUnnorm (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
            inputs.bv d)
      let vHi :=
        Bounds.cacheBound2 (fun q d =>
          Bounds.dotIntervalUpperUnnorm (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
            inputs.bv d)
      let qAbs :=
        Bounds.cacheBound2 (fun q d => max |qLo q d| |qHi q d|)
      let kAbs :=
        Bounds.cacheBound2 (fun q d => max |kLo q d| |kHi q d|)
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
  dotAbs : Fin seq → Fin seq → Rat
  /-- Base score absolute bound. -/
  scoreBaseAbs : Fin seq → Fin seq → Rat
  /-- Score absolute bound with causal masking. -/
  scoreAbs : Fin seq → Fin seq → Rat
  /-- Score lower bound. -/
  scoreLo : Fin seq → Fin seq → Rat
  /-- Score upper bound. -/
  scoreHi : Fin seq → Fin seq → Rat
  /-- Margin per query. -/
  marginAt : Fin seq → Rat
  /-- Epsilon per query. -/
  epsAt : Fin seq → Rat
  /-- Global margin. -/
  margin : Rat
  /-- Global epsilon. -/
  eps : Rat

/-- Compute score and margin bounds from cached score lower/upper bounds. -/
def headScoreBoundsFromCaches [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dotAbs : Fin seq → Fin seq → Rat)
    (scoreLo scoreHi : Fin seq → Fin seq → Rat) :
    HeadScoreBounds seq dModel dHead :=
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let scoreBaseAbs : Fin seq → Fin seq → Rat := fun q k =>
    |inputs.scale| * dotAbs q k
  let scoreAbs : Fin seq → Fin seq → Rat := fun q k =>
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
  let maskedGap : Fin seq → Rat := fun q =>
    scoreLo q (inputs.prev q) - inputs.maskValue
  let marginTasks : Array (Task Rat) :=
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
              (0 : Rat)
        else
          (0 : Rat)))
  let marginAt : Fin seq → Rat := fun q =>
    (marginTasks[q.1]'(by
      simp [marginTasks, q.isLt])).get
  let epsTasks : Array (Task Rat) :=
    Array.ofFn (fun q : Fin seq =>
      (marginTasks[q.1]'(by
        simp [marginTasks, q.isLt])).map (fun m =>
          if m < 0 then
            (1 : Rat)
          else
            ratDivUp (seq - 1) (1 + m)))
  let epsAt : Fin seq → Rat := fun q =>
    (epsTasks[q.1]'(by
      simp [epsTasks, q.isLt])).get
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
    (dotAbs : Fin seq → Fin seq → Rat) : HeadScoreBounds seq dModel dHead :=
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let dotAbsRowTasks : Array (Task { row : Array Rat // row.size = seq }) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        ⟨Array.ofFn (fun k : Fin seq => dotAbs q k), by simp⟩))
  let scaleAbs : Rat := |inputs.scale|
  let scoreLoCached : Fin seq → Fin seq → Rat := fun q k =>
    let row := (dotAbsRowTasks[q.1]'(by
      simp [dotAbsRowTasks, q.isLt])).get
    let base := scaleAbs * row.1.getD k.1 0
    if masked q k then inputs.maskValue else -base
  let scoreHiCached : Fin seq → Fin seq → Rat := fun q k =>
    let row := (dotAbsRowTasks[q.1]'(by
      simp [dotAbsRowTasks, q.isLt])).get
    let base := scaleAbs * row.1.getD k.1 0
    if masked q k then inputs.maskValue else base
  let marginAtRaw : Fin seq → Rat := fun q =>
    let row := (dotAbsRowTasks[q.1]'(by
      simp [dotAbsRowTasks, q.isLt])).get
    if q ∈ inputs.active then
      let rowArr := row.1
      let prev := inputs.prev q
      let dotAbsPrev := rowArr.getD prev.1 0
      if masked q prev then
        let scoreLoPrev := inputs.maskValue
        let scoreHiAt : Fin seq → Rat := fun k =>
          if masked q k then inputs.maskValue else scaleAbs * rowArr.getD k.1 0
        let maskedGap := scoreLoPrev - inputs.maskValue
        let step :
            (Option Rat × Bool) → Fin seq → (Option Rat × Bool) :=
          fun acc k =>
            if k = prev then
              acc
            else if masked q k then
              (acc.1, true)
            else
              let v := scoreLoPrev - scoreHiAt k
              match acc.1 with
              | none => (some v, acc.2)
              | some cur => (some (min cur v), acc.2)
        let acc := Linear.foldlFin seq step (none, false)
        match acc.1, acc.2 with
        | some unmaskedMin, true => min unmaskedMin maskedGap
        | some unmaskedMin, false => unmaskedMin
        | none, true => maskedGap
        | none, false => (0 : Rat)
      else
        let scoreLoPrev := -(scaleAbs * dotAbsPrev)
        let maskedGap := scoreLoPrev - inputs.maskValue
        let step :
            (Option Rat × Bool) → Fin seq → (Option Rat × Bool) :=
          fun acc k =>
            if k = prev then
              acc
            else if masked q k then
              (acc.1, true)
            else
              let raw := -(dotAbsPrev + rowArr.getD k.1 0)
              match acc.1 with
              | none => (some raw, acc.2)
              | some cur => (some (min cur raw), acc.2)
        let acc := Linear.foldlFin seq step (none, false)
        match acc.1, acc.2 with
        | some unmaskedMin, true => min (scaleAbs * unmaskedMin) maskedGap
        | some unmaskedMin, false => scaleAbs * unmaskedMin
        | none, true => maskedGap
        | none, false => (0 : Rat)
    else
      (0 : Rat)
  let marginAtCached := Bounds.cacheBoundThunk marginAtRaw
  let marginAt : Fin seq → Rat := fun q =>
    marginAtCached q
  let epsAtRaw : Fin seq → Rat := fun q =>
    let m := marginAt q
    if m < 0 then
      (1 : Rat)
    else
      ratDivUp (seq - 1) (1 + m)
  let epsAtCached := Bounds.cacheBoundThunk epsAtRaw
  let epsAt : Fin seq → Rat := fun q =>
    epsAtCached q
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
  { dotAbs := dotAbs
    scoreBaseAbs := fun q k => |inputs.scale| * dotAbs q k
    scoreAbs := fun q k =>
      if masked q k then |inputs.maskValue| else |inputs.scale| * dotAbs q k
    scoreLo := scoreLoCached
    scoreHi := scoreHiCached
    marginAt := marginAt
    epsAt := epsAt
    margin := margin
    eps := eps }

theorem headScoreBoundsFromDotAbs_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dotAbs : Fin seq → Fin seq → Rat) :
  headScoreBoundsFromDotAbs inputs dotAbs =
      let masked : Fin seq → Fin seq → Prop := fun q k =>
        inputs.maskCausal = true ∧ q < k
      let dotAbsRowTasks : Array (Task { row : Array Rat // row.size = seq }) :=
        Array.ofFn (fun q : Fin seq =>
          Task.spawn (fun _ =>
            ⟨Array.ofFn (fun k : Fin seq => dotAbs q k), by simp⟩))
      let scaleAbs : Rat := |inputs.scale|
      let scoreLoCached : Fin seq → Fin seq → Rat := fun q k =>
        let row := (dotAbsRowTasks[q.1]'(by
          simp [dotAbsRowTasks, q.isLt])).get
        let base := scaleAbs * row.1.getD k.1 0
        if masked q k then inputs.maskValue else -base
      let scoreHiCached : Fin seq → Fin seq → Rat := fun q k =>
        let row := (dotAbsRowTasks[q.1]'(by
          simp [dotAbsRowTasks, q.isLt])).get
        let base := scaleAbs * row.1.getD k.1 0
        if masked q k then inputs.maskValue else base
      let marginAtRaw : Fin seq → Rat := fun q =>
        let row := (dotAbsRowTasks[q.1]'(by
          simp [dotAbsRowTasks, q.isLt])).get
        if q ∈ inputs.active then
          let rowArr := row.1
          let prev := inputs.prev q
          let dotAbsPrev := rowArr.getD prev.1 0
          if masked q prev then
            let scoreLoPrev := inputs.maskValue
            let scoreHiAt : Fin seq → Rat := fun k =>
              if masked q k then inputs.maskValue else scaleAbs * rowArr.getD k.1 0
            let maskedGap := scoreLoPrev - inputs.maskValue
            let step :
                (Option Rat × Bool) → Fin seq → (Option Rat × Bool) :=
              fun acc k =>
                if k = prev then
                  acc
                else if masked q k then
                  (acc.1, true)
                else
                  let v := scoreLoPrev - scoreHiAt k
                  match acc.1 with
                  | none => (some v, acc.2)
                  | some cur => (some (min cur v), acc.2)
            let acc := Linear.foldlFin seq step (none, false)
            match acc.1, acc.2 with
            | some unmaskedMin, true => min unmaskedMin maskedGap
            | some unmaskedMin, false => unmaskedMin
            | none, true => maskedGap
            | none, false => (0 : Rat)
          else
            let scoreLoPrev := -(scaleAbs * dotAbsPrev)
            let maskedGap := scoreLoPrev - inputs.maskValue
            let step :
                (Option Rat × Bool) → Fin seq → (Option Rat × Bool) :=
              fun acc k =>
                if k = prev then
                  acc
                else if masked q k then
                  (acc.1, true)
                else
                  let raw := -(dotAbsPrev + rowArr.getD k.1 0)
                  match acc.1 with
                  | none => (some raw, acc.2)
                  | some cur => (some (min cur raw), acc.2)
            let acc := Linear.foldlFin seq step (none, false)
            match acc.1, acc.2 with
            | some unmaskedMin, true => min (scaleAbs * unmaskedMin) maskedGap
            | some unmaskedMin, false => scaleAbs * unmaskedMin
            | none, true => maskedGap
            | none, false => (0 : Rat)
        else
          (0 : Rat)
      let marginAtCached := Bounds.cacheBoundThunk marginAtRaw
      let marginAt : Fin seq → Rat := fun q =>
        marginAtCached q
      let epsAtRaw : Fin seq → Rat := fun q =>
        let m := marginAt q
        if m < 0 then
          (1 : Rat)
        else
          ratDivUp (seq - 1) (1 + m)
      let epsAtCached := Bounds.cacheBoundThunk epsAtRaw
      let epsAt : Fin seq → Rat := fun q =>
        epsAtCached q
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
      { dotAbs := dotAbs
        scoreBaseAbs := fun q k => |inputs.scale| * dotAbs q k
        scoreAbs := fun q k =>
          if masked q k then |inputs.maskValue| else |inputs.scale| * dotAbs q k
        scoreLo := scoreLoCached
        scoreHi := scoreHiCached
        marginAt := marginAt
        epsAt := epsAt
        margin := margin
        eps := eps } := rfl

/-- Compute score and margin bounds from Q/K interval bounds. -/
def headScoreBoundsFromIntervals [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qLo qHi kLo kHi : Fin seq → Fin dHead → Rat) :
    HeadScoreBounds seq dModel dHead :=
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let dotRowTasks : Array (Task { row : Array (Rat × Rat) // row.size = seq }) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        ⟨Array.ofFn (fun k : Fin seq =>
            dotIntervalLowerUpper2CommonDen (fun d => qLo q d) (fun d => qHi q d)
              (fun d => kLo k d) (fun d => kHi k d)),
          by simp⟩))
  let dotLo : Fin seq → Fin seq → Rat := fun q k =>
    let row := (dotRowTasks[q.1]'(by
      simp [dotRowTasks, q.isLt])).get
    let entry := row.1[k.1]'(by
      simp [row.2, k.isLt])
    entry.1
  let dotHi : Fin seq → Fin seq → Rat := fun q k =>
    let row := (dotRowTasks[q.1]'(by
      simp [dotRowTasks, q.isLt])).get
    let entry := row.1[k.1]'(by
      simp [row.2, k.isLt])
    entry.2
  let dotAbs : Fin seq → Fin seq → Rat := fun q k => max |dotLo q k| |dotHi q k|
  let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      if 0 ≤ inputs.scale then
        inputs.scale * dotLo q k
      else
        inputs.scale * dotHi q k
  let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      if 0 ≤ inputs.scale then
        inputs.scale * dotHi q k
      else
        inputs.scale * dotLo q k
  headScoreBoundsFromCaches inputs dotAbs scoreLo scoreHi

theorem headScoreBoundsFromIntervals_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qLo qHi kLo kHi : Fin seq → Fin dHead → Rat) :
    headScoreBoundsFromIntervals inputs qLo qHi kLo kHi =
      let masked : Fin seq → Fin seq → Prop := fun q k =>
        inputs.maskCausal = true ∧ q < k
      let dotRowTasks : Array (Task { row : Array (Rat × Rat) // row.size = seq }) :=
        Array.ofFn (fun q : Fin seq =>
          Task.spawn (fun _ =>
            ⟨Array.ofFn (fun k : Fin seq =>
                dotIntervalLowerUpper2CommonDen (fun d => qLo q d) (fun d => qHi q d)
                  (fun d => kLo k d) (fun d => kHi k d)),
              by simp⟩))
      let dotLo : Fin seq → Fin seq → Rat := fun q k =>
        let row := (dotRowTasks[q.1]'(by
          simp [dotRowTasks, q.isLt])).get
        let entry := row.1[k.1]'(by
          simp [row.2, k.isLt])
        entry.1
      let dotHi : Fin seq → Fin seq → Rat := fun q k =>
        let row := (dotRowTasks[q.1]'(by
          simp [dotRowTasks, q.isLt])).get
        let entry := row.1[k.1]'(by
          simp [row.2, k.isLt])
        entry.2
      let dotAbs : Fin seq → Fin seq → Rat := fun q k => max |dotLo q k| |dotHi q k|
      let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
        if masked q k then
          inputs.maskValue
        else
          if 0 ≤ inputs.scale then
            inputs.scale * dotLo q k
          else
            inputs.scale * dotHi q k
      let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
        if masked q k then
          inputs.maskValue
        else
          if 0 ≤ inputs.scale then
            inputs.scale * dotHi q k
          else
            inputs.scale * dotLo q k
      headScoreBoundsFromCaches inputs dotAbs scoreLo scoreHi := rfl

/-- Compute score and margin bounds from Q/K absolute bounds. -/
def headScoreBounds [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qAbs kAbs : Fin seq → Fin dHead → Rat) :
    HeadScoreBounds seq dModel dHead :=
  headScoreBoundsFromDotAbs inputs (fun q k =>
    Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d))

theorem headScoreBounds_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qAbs kAbs : Fin seq → Fin dHead → Rat) :
    headScoreBounds inputs qAbs kAbs =
      let masked : Fin seq → Fin seq → Prop := fun q k =>
        inputs.maskCausal = true ∧ q < k
      let dotAbs : Fin seq → Fin seq → Rat := fun q k =>
        Linear.dotFin dHead (fun d => qAbs q d) (fun d => kAbs k d)
      let dotAbsRowTasks : Array (Task { row : Array Rat // row.size = seq }) :=
        Array.ofFn (fun q : Fin seq =>
          Task.spawn (fun _ =>
            ⟨Array.ofFn (fun k : Fin seq => dotAbs q k), by simp⟩))
      let scaleAbs : Rat := |inputs.scale|
      let scoreLoCached : Fin seq → Fin seq → Rat := fun q k =>
        let row := (dotAbsRowTasks[q.1]'(by
          simp [dotAbsRowTasks, q.isLt])).get
        let base := scaleAbs * row.1.getD k.1 0
        if masked q k then inputs.maskValue else -base
      let scoreHiCached : Fin seq → Fin seq → Rat := fun q k =>
        let row := (dotAbsRowTasks[q.1]'(by
          simp [dotAbsRowTasks, q.isLt])).get
        let base := scaleAbs * row.1.getD k.1 0
        if masked q k then inputs.maskValue else base
      let marginAtRaw : Fin seq → Rat := fun q =>
        let row := (dotAbsRowTasks[q.1]'(by
          simp [dotAbsRowTasks, q.isLt])).get
        if q ∈ inputs.active then
          let rowArr := row.1
          let prev := inputs.prev q
          let dotAbsPrev := rowArr.getD prev.1 0
          if masked q prev then
            let scoreLoPrev := inputs.maskValue
            let scoreHiAt : Fin seq → Rat := fun k =>
              if masked q k then inputs.maskValue else scaleAbs * rowArr.getD k.1 0
            let maskedGap := scoreLoPrev - inputs.maskValue
            let step :
                (Option Rat × Bool) → Fin seq → (Option Rat × Bool) :=
              fun acc k =>
                if k = prev then
                  acc
                else if masked q k then
                  (acc.1, true)
                else
                  let v := scoreLoPrev - scoreHiAt k
                  match acc.1 with
                  | none => (some v, acc.2)
                  | some cur => (some (min cur v), acc.2)
            let acc := Linear.foldlFin seq step (none, false)
            match acc.1, acc.2 with
            | some unmaskedMin, true => min unmaskedMin maskedGap
            | some unmaskedMin, false => unmaskedMin
            | none, true => maskedGap
            | none, false => (0 : Rat)
          else
            let scoreLoPrev := -(scaleAbs * dotAbsPrev)
            let maskedGap := scoreLoPrev - inputs.maskValue
            let step :
                (Option Rat × Bool) → Fin seq → (Option Rat × Bool) :=
              fun acc k =>
                if k = prev then
                  acc
                else if masked q k then
                  (acc.1, true)
                else
                  let raw := -(dotAbsPrev + rowArr.getD k.1 0)
                  match acc.1 with
                  | none => (some raw, acc.2)
                  | some cur => (some (min cur raw), acc.2)
            let acc := Linear.foldlFin seq step (none, false)
            match acc.1, acc.2 with
            | some unmaskedMin, true => min (scaleAbs * unmaskedMin) maskedGap
            | some unmaskedMin, false => scaleAbs * unmaskedMin
            | none, true => maskedGap
            | none, false => (0 : Rat)
        else
          (0 : Rat)
      let marginAtCached := Bounds.cacheBoundThunk marginAtRaw
      let marginAt : Fin seq → Rat := fun q =>
        marginAtCached q
      let epsAtRaw : Fin seq → Rat := fun q =>
        let m := marginAt q
        if m < 0 then
          (1 : Rat)
        else
          ratDivUp (seq - 1) (1 + m)
      let epsAtCached := Bounds.cacheBoundThunk epsAtRaw
      let epsAt : Fin seq → Rat := fun q =>
        epsAtCached q
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
      { dotAbs := dotAbs
        scoreBaseAbs := fun q k => |inputs.scale| * dotAbs q k
        scoreAbs := fun q k =>
          if masked q k then |inputs.maskValue| else |inputs.scale| * dotAbs q k
        scoreLo := scoreLoCached
        scoreHi := scoreHiCached
        marginAt := marginAt
        epsAt := epsAt
        margin := margin
        eps := eps } := rfl

/-- Value bounds used by the induction-head builder. -/
structure HeadValueBounds (seq dModel dHead : Nat) where
  /-- Value lower bounds. -/
  valsLo : Fin seq → Rat
  /-- Value upper bounds. -/
  valsHi : Fin seq → Rat
  /-- Global value lower bound. -/
  lo : Rat
  /-- Global value upper bound. -/
  hi : Rat

/-- Cached direction vector for value bounds. -/
def headValueDirHead {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Fin dHead → Rat :=
  let dirHeadVec := dirHeadVecOfInputs inputs
  fun d => dirHeadVec.get d

theorem headValueDirHead_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    headValueDirHead inputs =
      let dirHeadVec := dirHeadVecOfInputs inputs
      fun d => dirHeadVec.get d := rfl

/-- Cached lower value bounds from V intervals. -/
def headValueValsLoArray {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) : Array Rat :=
  let dirHead := headValueDirHead inputs
  Array.ofFn (fun k =>
    Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k))

/-- Unfold `headValueValsLoArray` to its `Array.ofFn` definition. -/
theorem headValueValsLoArray_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsLoArray inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      Array.ofFn (fun k =>
        Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k)) := rfl

/-- Cached lower value bounds from V intervals. -/
def headValueValsLo {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) : Fin seq → Rat :=
  let arr := headValueValsLoArray inputs vLo vHi
  fun k => arr.getD k.1 (0 : Rat)

theorem headValueValsLo_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsLo inputs vLo vHi =
      let arr := headValueValsLoArray inputs vLo vHi
      fun k => arr.getD k.1 (0 : Rat) := rfl

/-- Cached lower value bounds from V intervals using a common-denominator sum. -/
def headValueValsLoCommonDenArray {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) : Array Rat :=
  headValueValsLoArray inputs vLo vHi

/-- Unfold `headValueValsLoCommonDenArray` to its `Array.ofFn` definition. -/
theorem headValueValsLoCommonDenArray_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsLoCommonDenArray inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      Array.ofFn (fun k =>
        Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k)) := rfl

/-- Cached lower value bounds from V intervals using a common-denominator sum. -/
def headValueValsLoCommonDen {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) : Fin seq → Rat :=
  let arr := headValueValsLoCommonDenArray inputs vLo vHi
  fun k => arr.getD k.1 (0 : Rat)

theorem headValueValsLoCommonDen_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsLoCommonDen inputs vLo vHi =
      let arr := headValueValsLoCommonDenArray inputs vLo vHi
      fun k => arr.getD k.1 (0 : Rat) := rfl

/-- Common-denominator lower bounds agree with cached rational bounds pointwise. -/
theorem headValueValsLoCommonDenArray_eq {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsLoCommonDenArray inputs vLo vHi = headValueValsLoArray inputs vLo vHi := by
  rfl

theorem headValueValsLoCommonDen_eq {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsLoCommonDen inputs vLo vHi = headValueValsLo inputs vLo vHi := by
  funext k
  simp [headValueValsLoCommonDen, headValueValsLo, headValueValsLoCommonDenArray_eq]

/-- Cached upper value bounds from V intervals. -/
def headValueValsHiArray {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) : Array Rat :=
  let dirHead := headValueDirHead inputs
  Array.ofFn (fun k =>
    Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k))

/-- Unfold `headValueValsHiArray` to its `Array.ofFn` definition. -/
theorem headValueValsHiArray_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsHiArray inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      Array.ofFn (fun k =>
        Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k)) := rfl

/-- Cached upper value bounds from V intervals. -/
def headValueValsHi {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) : Fin seq → Rat :=
  let arr := headValueValsHiArray inputs vLo vHi
  fun k => arr.getD k.1 (0 : Rat)

theorem headValueValsHi_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsHi inputs vLo vHi =
      let arr := headValueValsHiArray inputs vLo vHi
      fun k => arr.getD k.1 (0 : Rat) := rfl

/-- Cached upper value bounds from V intervals using a common-denominator sum. -/
def headValueValsHiCommonDenArray {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) : Array Rat :=
  headValueValsHiArray inputs vLo vHi

/-- Unfold `headValueValsHiCommonDenArray` to its `Array.ofFn` definition. -/
theorem headValueValsHiCommonDenArray_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsHiCommonDenArray inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      Array.ofFn (fun k =>
        Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k)) := rfl

/-- Cached upper value bounds from V intervals using a common-denominator sum. -/
def headValueValsHiCommonDen {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) : Fin seq → Rat :=
  let arr := headValueValsHiCommonDenArray inputs vLo vHi
  fun k => arr.getD k.1 (0 : Rat)

theorem headValueValsHiCommonDen_spec {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsHiCommonDen inputs vLo vHi =
      let arr := headValueValsHiCommonDenArray inputs vLo vHi
      fun k => arr.getD k.1 (0 : Rat) := rfl

/-- Common-denominator upper bounds agree with cached rational bounds pointwise. -/
theorem headValueValsHiCommonDenArray_eq {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsHiCommonDenArray inputs vLo vHi = headValueValsHiArray inputs vLo vHi := by
  rfl

theorem headValueValsHiCommonDen_eq {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueValsHiCommonDen inputs vLo vHi = headValueValsHi inputs vLo vHi := by
  funext k
  simp [headValueValsHiCommonDen, headValueValsHi, headValueValsHiCommonDenArray_eq]

/-- Global lower value bound from an array of per-key values. -/
def headValueLoArray (valsLo : Array Rat) : Rat :=
  reduceMinArray valsLo

/-- Unfold `headValueLoArray` to its reduction helper. -/
theorem headValueLoArray_spec (valsLo : Array Rat) :
    headValueLoArray valsLo = reduceMinArray valsLo := rfl

/-- Global lower value bound from cached per-key values. -/
def headValueLo [NeZero seq] (valsLo : Fin seq → Rat) : Rat :=
  headValueLoArray (Array.ofFn valsLo)

theorem headValueLo_spec [NeZero seq] (valsLo : Fin seq → Rat) :
    headValueLo valsLo = headValueLoArray (Array.ofFn valsLo) := rfl

/-- Task wrapper for `headValueLo`. -/
def headValueLoTask [NeZero seq] (valsLo : Fin seq → Rat) : Task Rat :=
  reduceMinFnTask valsLo

theorem headValueLoTask_spec [NeZero seq] (valsLo : Fin seq → Rat) :
    headValueLoTask valsLo = reduceMinFnTask valsLo := rfl

/-- Global upper value bound from an array of per-key values. -/
def headValueHiArray (valsHi : Array Rat) : Rat :=
  reduceMaxArray valsHi

/-- Unfold `headValueHiArray` to its reduction helper. -/
theorem headValueHiArray_spec (valsHi : Array Rat) :
    headValueHiArray valsHi = reduceMaxArray valsHi := rfl

/-- Global upper value bound from cached per-key values. -/
def headValueHi [NeZero seq] (valsHi : Fin seq → Rat) : Rat :=
  headValueHiArray (Array.ofFn valsHi)

theorem headValueHi_spec [NeZero seq] (valsHi : Fin seq → Rat) :
    headValueHi valsHi = headValueHiArray (Array.ofFn valsHi) := rfl

/-- Task wrapper for `headValueHi`. -/
def headValueHiTask [NeZero seq] (valsHi : Fin seq → Rat) : Task Rat :=
  reduceMaxFnTask valsHi

theorem headValueHiTask_spec [NeZero seq] (valsHi : Fin seq → Rat) :
    headValueHiTask valsHi = reduceMaxFnTask valsHi := rfl

/-- Build `HeadValueBounds` from precomputed arrays. -/
private def headValueBoundsOfArrays {seq dModel dHead : Nat}
    (valsLoArr valsHiArr : Array Rat) : HeadValueBounds seq dModel dHead :=
  let valsLo : Fin seq → Rat := fun k => valsLoArr.getD k.1 (0 : Rat)
  let valsHi : Fin seq → Rat := fun k => valsHiArr.getD k.1 (0 : Rat)
  let lo := headValueLoArray valsLoArr
  let hi := headValueHiArray valsHiArr
  { valsLo := valsLo, valsHi := valsHi, lo := lo, hi := hi }

/-- Build a cached bounds array in parallel from a per-key computation. -/
private def buildBoundArrayTask [NeZero seq] (f : Fin seq → Rat) : Task (Array Rat) :=
  let n := seq
  let chunkSize : Nat := 64
  let chunks : Nat := (n + chunkSize - 1) / chunkSize
  let hpos : 0 < seq := Nat.pos_of_ne_zero (by simpa using (NeZero.ne (n := seq)))
  let defaultIdx : Fin seq := ⟨0, hpos⟩
  let idxs : Array (Fin seq) := Array.ofFn (fun i : Fin seq => i)
  let chunkTasks : List (Task (Array Rat)) :=
    (List.range chunks).map (fun c =>
      Task.spawn (fun _ =>
        let start := c * chunkSize
        let stop := Nat.min n (start + chunkSize)
        let vals :=
          (List.range (stop - start)).map (fun i =>
            f (idxs.getD (start + i) defaultIdx))
        vals.toArray))
  Task.mapList (fun xs => xs.foldl (fun acc arr => acc ++ arr) #[]) chunkTasks

/-- Compute value bounds from V interval bounds. -/
def headValueBounds [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    HeadValueBounds seq dModel dHead :=
  let valsLoArr := headValueValsLoArray inputs vLo vHi
  let valsHiArr := headValueValsHiArray inputs vLo vHi
  headValueBoundsOfArrays valsLoArr valsHiArr

theorem headValueBounds_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueBounds inputs vLo vHi =
      let valsLoArr := headValueValsLoArray inputs vLo vHi
      let valsHiArr := headValueValsHiArray inputs vLo vHi
      headValueBoundsOfArrays valsLoArr valsHiArr := rfl

/-- Compute value bounds from V interval bounds in parallel. -/
def headValueBoundsTask [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    Task (HeadValueBounds seq dModel dHead) :=
  let dirHead := headValueDirHead inputs
  let valsLoTask := buildBoundArrayTask (fun k =>
    Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k))
  let valsHiTask := buildBoundArrayTask (fun k =>
    Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k))
  Task.bind valsLoTask (fun valsLoArr =>
    Task.map (fun valsHiArr => headValueBoundsOfArrays valsLoArr valsHiArr) valsHiTask)

/-- Unfold `headValueBoundsTask` to its task graph. -/
theorem headValueBoundsTask_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueBoundsTask inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      let valsLoTask := buildBoundArrayTask (fun k =>
        Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k))
      let valsHiTask := buildBoundArrayTask (fun k =>
        Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k))
      Task.bind valsLoTask (fun valsLoArr =>
        Task.map (fun valsHiArr => headValueBoundsOfArrays valsLoArr valsHiArr) valsHiTask) := rfl

/-- Compute value bounds from V interval bounds using a common-denominator sum. -/
def headValueBoundsCommonDen [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    HeadValueBounds seq dModel dHead :=
  let valsLoArr := headValueValsLoCommonDenArray inputs vLo vHi
  let valsHiArr := headValueValsHiCommonDenArray inputs vLo vHi
  headValueBoundsOfArrays valsLoArr valsHiArr

theorem headValueBoundsCommonDen_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueBoundsCommonDen inputs vLo vHi =
      let valsLoArr := headValueValsLoCommonDenArray inputs vLo vHi
      let valsHiArr := headValueValsHiCommonDenArray inputs vLo vHi
      headValueBoundsOfArrays valsLoArr valsHiArr := rfl

/-- Compute value bounds from V intervals using a common-denominator sum in parallel. -/
def headValueBoundsCommonDenTask [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    Task (HeadValueBounds seq dModel dHead) :=
  let dirHead := headValueDirHead inputs
  let valsLoTask := buildBoundArrayTask (fun k =>
    Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k))
  let valsHiTask := buildBoundArrayTask (fun k =>
    Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k))
  Task.bind valsLoTask (fun valsLoArr =>
    Task.map (fun valsHiArr => headValueBoundsOfArrays valsLoArr valsHiArr) valsHiTask)

/-- Unfold `headValueBoundsCommonDenTask` to its task graph. -/
theorem headValueBoundsCommonDenTask_spec [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueBoundsCommonDenTask inputs vLo vHi =
      let dirHead := headValueDirHead inputs
      let valsLoTask := buildBoundArrayTask (fun k =>
        Bounds.dotIntervalLowerCommonDen dirHead (vLo k) (vHi k))
      let valsHiTask := buildBoundArrayTask (fun k =>
        Bounds.dotIntervalUpperCommonDen dirHead (vLo k) (vHi k))
      Task.bind valsLoTask (fun valsLoArr =>
        Task.map (fun valsHiArr => headValueBoundsOfArrays valsLoArr valsHiArr) valsHiTask) := rfl

theorem headValueBoundsCommonDen_eq [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vLo vHi : Fin seq → Fin dHead → Rat) :
    headValueBoundsCommonDen inputs vLo vHi = headValueBounds inputs vLo vHi := by
  classical
  simp [headValueBoundsCommonDen, headValueBounds, headValueValsLoCommonDenArray_eq,
    headValueValsHiCommonDenArray_eq]

end Sound

end Nfp
