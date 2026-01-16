-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.List.Range
public import Nfp.IO.HeadScore
public import Nfp.Model.InductionHead
public import Nfp.Sound.Induction.HeadBounds
public import Nfp.Sound.Induction.LogitDiff

/-!
Small IO helpers for profiling slow phases.
-/

public section

namespace Nfp

namespace IO

open Sound

/-- Current monotonic time in microseconds. -/
def monoUsNow : IO Nat := do
  let t ← IO.monoNanosNow
  return t / 1000

/-! Timing configuration -/

/-- Runtime configuration for timing output. -/
structure TimingConfig where
  /-- Optional stdout override for timing output. -/
  stdout? : Option Bool
  /-- Optional heartbeat interval override (ms). -/
  heartbeatMs? : Option UInt32
  deriving Inhabited

/-- Mutable timing configuration (overrides environment defaults). -/
initialize timingConfig : IO.Ref TimingConfig ←
  IO.mkRef { stdout? := none, heartbeatMs? := none }

/-- Enable or disable timing stdout output. -/
def setTimingStdout (enabled : Bool) : IO Unit := do
  timingConfig.modify (fun cfg => { cfg with stdout? := some enabled })

/-- Override the heartbeat interval (ms). -/
def setTimingHeartbeatMs (ms : UInt32) : IO Unit := do
  timingConfig.modify (fun cfg => { cfg with heartbeatMs? := some ms })

/-- Resolve whether timing output should be printed. -/
def timingStdoutEnabled : IO Bool := do
  let cfg ← timingConfig.get
  match cfg.stdout? with
  | some enabled => return enabled
  | none =>
      match (← IO.getEnv "NFP_TIMING_STDOUT") with
      | some "1" => return true
      | some "true" => return true
      | some "yes" => return true
      | _ => return false

/-- Resolve the heartbeat interval (ms), respecting overrides. -/
def timingHeartbeatMs : IO UInt32 := do
  let cfg ← timingConfig.get
  match cfg.heartbeatMs? with
  | some ms => return ms
  | none =>
      let defaultMs : Nat := 0
      let ms :=
        (← IO.getEnv "NFP_TIMING_HEARTBEAT_MS").bind String.toNat? |>.getD defaultMs
      return UInt32.ofNat ms

/-- Resolve the heartbeat interval (ms) for long-running induction cert builds. -/
def heartbeatMs : IO UInt32 :=
  timingHeartbeatMs

/-- Print a timing line only when stdout timing is enabled. -/
def timingPrint (line : String) : IO Unit := do
  if (← timingStdoutEnabled) then
    IO.println line
  else
    pure ()

/-- Flush stdout only when timing output is enabled. -/
def timingFlush : IO Unit := do
  if (← timingStdoutEnabled) then
    let h ← IO.getStdout
    h.flush
  else
    pure ()

/-- Append a timing log line to `NFP_TIMING_LOG` when set. -/
def logTiming (line : String) : IO Unit := do
  match (← IO.getEnv "NFP_TIMING_LOG") with
  | some path =>
      let h ← IO.FS.Handle.mk (System.FilePath.mk path) IO.FS.Mode.append
      h.putStr (line ++ "\n")
      h.flush
  | none => pure ()

/-- Time an IO phase and print the duration when timing output is enabled. -/
def timePhase {α : Type} (label : String) (act : IO α) : IO α := do
  logTiming s!"start: {label}"
  let t0 ← monoUsNow
  let res ← act
  let t1 ← monoUsNow
  logTiming s!"done: {label} {t1 - t0} us"
  timingPrint s!"timing: {label} {t1 - t0} us"
  return res

/-- Time an IO phase supplied as a thunk and print the duration when timing output is enabled. -/
def timePhaseThunk {α : Type} (label : String) (act : Unit → IO α) : IO α := do
  logTiming s!"start: {label}"
  let t0 ← monoUsNow
  let res ← act ()
  let t1 ← monoUsNow
  logTiming s!"done: {label} {t1 - t0} us"
  timingPrint s!"timing: {label} {t1 - t0} us"
  return res

/-- Time a pure thunk and print the duration when timing output is enabled. -/
def timePure {α : Type} (label : String) (f : Unit → α) : IO α := do
  logTiming s!"start: {label}"
  let t0 ← monoUsNow
  let res := f ()
  let t1 ← monoUsNow
  logTiming s!"done: {label} {t1 - t0} us"
  timingPrint s!"timing: {label} {t1 - t0} us"
  return res

/-- Time a pure thunk, printing heartbeat updates while it runs. -/
def timePureWithHeartbeat {α : Type} (label : String) (f : Unit → α) : IO α := do
  let t0 ← monoUsNow
  timingPrint s!"timing: {label} start"
  timingFlush
  let task : Task α := Task.spawn (fun _ => f ())
  let heartbeatMs ← heartbeatMs
  if heartbeatMs ≠ 0 then
    let mut finished := (← IO.hasFinished task)
    while !finished do
      IO.sleep heartbeatMs
      finished := (← IO.hasFinished task)
      if !finished then
        let now ← monoUsNow
        timingPrint s!"timing: {label} running {now - t0} us"
        timingFlush
  let res ← IO.wait task
  let t1 ← monoUsNow
  timingPrint s!"timing: {label} {t1 - t0} us"
  return res

/-- Flush stdout immediately for interleaved timing output. -/
def flushStdout : IO Unit := do
  let h ← IO.getStdout
  h.flush

/-- Force a sample score-gap computation for timing. -/
def timeHeadScoreSampleGap {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (score : Sound.HeadScoreBounds seq dModel dHead) : IO Unit := do
  timingPrint "timing: head score sample gap start"
  timingFlush
  let t0 ← monoUsNow
  match List.finRange seq with
  | [] =>
      timingPrint "timing: head score sample gap skipped (empty seq)"
  | q :: _ =>
      let _ := score.scoreLo q (inputs.prev q)
      let _ := score.scoreHi q (inputs.prev q)
      let _ := score.scoreLo q (inputs.prev q) - score.scoreHi q (inputs.prev q)
      pure ()
  let t1 ← monoUsNow
  timingPrint s!"timing: head score sample gap {t1 - t0} us"
  timingFlush

/-- Force marginAt evaluation over the active list for timing. -/
def timeHeadScoreMarginList {seq dModel dHead : Nat}
    (activeList : List (Fin seq))
    (score : Sound.HeadScoreBounds seq dModel dHead) : IO Unit := do
  timingPrint "timing: head score marginAt list start"
  timingFlush
  let t0 ← monoUsNow
  for q in activeList do
    let _ := score.marginAt q
    pure ()
  let t1 ← monoUsNow
  timingPrint s!"timing: head score marginAt list {t1 - t0} us"
  timingFlush

/-- Force marginAt evaluation without constructing the full score bounds record. -/
def timeHeadScoreMarginRaw {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dotAbs : Fin seq → Fin seq → Rat)
    (activeList : List (Fin seq)) : IO Unit := do
  timingPrint "timing: head score marginRaw list start"
  timingFlush
  let t0 ← monoUsNow
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let scoreBaseAbs : Fin seq → Fin seq → Rat := fun q k =>
    |inputs.scale| * dotAbs q k
  let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      -scoreBaseAbs q k
  let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      scoreBaseAbs q k
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
  let scoreGap : Fin seq → Fin seq → Rat := fun q k =>
    scoreLo q (inputs.prev q) - scoreHi q k
  let marginAtRaw : Fin seq → Rat := fun q =>
    let other := unmaskedKeys q
    let maskedSet := maskedKeys q
    if hunmasked : other.Nonempty then
      let unmaskedMin := other.inf' hunmasked (fun k => scoreGap q k)
      if _hmasked : maskedSet.Nonempty then
        min unmaskedMin (maskedGap q)
      else
        unmaskedMin
    else
      if _hmasked : maskedSet.Nonempty then
        maskedGap q
      else
        (0 : Rat)
  for q in activeList do
    let _ := marginAtRaw q
    pure ()
  let t1 ← monoUsNow
  timingPrint s!"timing: head score marginRaw list {t1 - t0} us"
  timingFlush

/-- Force individual score-bound fields to locate slow evaluations. -/
def timeHeadScoreFieldForces {seq dModel dHead : Nat}
    (score : Sound.HeadScoreBounds seq dModel dHead) : IO Unit := do
  timingPrint "timing: head score field force start"
  timingFlush
  let timeOne (label : String) (f : Unit → IO Unit) : IO Unit := do
    let t0 ← monoUsNow
    f ()
    let t1 ← monoUsNow
    timingPrint s!"timing: head score field {label} {t1 - t0} us"
    timingFlush
  match List.finRange seq with
  | [] =>
      timingPrint "timing: head score field force skipped (empty seq)"
      timingFlush
  | q :: _ =>
      match List.finRange seq with
      | [] =>
          timingPrint "timing: head score field force skipped (empty seq)"
          timingFlush
      | k :: _ =>
          timeOne "scoreBaseAbs" (fun _ => do let _ := score.scoreBaseAbs q k; pure ())
          timeOne "scoreAbs" (fun _ => do let _ := score.scoreAbs q k; pure ())
          timeOne "scoreLo" (fun _ => do let _ := score.scoreLo q k; pure ())
          timeOne "scoreHi" (fun _ => do let _ := score.scoreHi q k; pure ())
          timeOne "marginAt" (fun _ => do let _ := score.marginAt q; pure ())
          timeOne "epsAt" (fun _ => do let _ := score.epsAt q; pure ())
          timeOne "margin" (fun _ => do let _ := score.margin; pure ())
          timeOne "eps" (fun _ => do let _ := score.eps; pure ())
  timingPrint "timing: head score field force done"
  timingFlush

/-- Force a rational to help isolate cached computations. -/
def forceRat (x : Rat) : IO Unit := do
  if x = x then
    pure ()
  else
    pure ()

/-- Report detailed timing for weighted logit-diff components when enabled. -/
def logitDiffProfileEnabled : IO Bool := do
  return (← IO.getEnv "NFP_TIMING_LOGITDIFF_PROFILE").isSome

/-- Profile weighted logit-diff sub-steps when logit-diff profiling is enabled. -/
def profileLogitDiffWeighted {seq : Nat}
    (cert : Sound.InductionHeadCert seq)
    (cache : Sound.LogitDiffCache seq) : IO Unit := do
  if !(← logitDiffProfileEnabled) then
    pure ()
  else
    timingPrint "timing: logit-diff profile start"
    timingFlush
    let valsLoArr ← timePureWithHeartbeat "logit-diff profile: valsLo force" (fun () =>
      Array.ofFn (fun q : Fin seq => cache.valsLo q))
    let weightRows ← timePureWithHeartbeat "logit-diff profile: weightBoundAt force" (fun () =>
      Array.ofFn (fun q : Fin seq =>
        Array.ofFn (fun k : Fin seq => cert.weightBoundAt q k)))
    let valsLo : Fin seq → Rat := fun k =>
      valsLoArr.getD k.1 (0 : Rat)
    let weightBoundAt : Fin seq → Fin seq → Rat := fun q k =>
      let row := weightRows.getD q.1 #[]
      row.getD k.1 (0 : Rat)
    let _ ← timePureWithHeartbeat "logit-diff profile: weighted gap sum" (fun () =>
      Array.ofFn (fun q : Fin seq =>
        let valsLoPrev := valsLo (cert.prev q)
        Linear.sumFin seq (fun k =>
          let diff := valsLoPrev - valsLo k
          weightBoundAt q k * max (0 : Rat) diff)))
    let _ ← timePureWithHeartbeat "logit-diff profile: weighted min" (fun () =>
      let gap : Fin seq → Rat := fun q =>
        let valsLoPrev := valsLo (cert.prev q)
        Linear.sumFin seq (fun k =>
          let diff := valsLoPrev - valsLo k
          weightBoundAt q k * max (0 : Rat) diff)
      let f : Fin seq → Rat := fun q => valsLo (cert.prev q) - gap q
      if h : cert.active.Nonempty then
        let _ := cert.active.inf' h f
        ()
      else
        ())

/-- Profile the core induction-head bounds used by the sound certificate builder. -/
def timeInductionHeadCoreStages {seq dModel dHead : Nat} [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) : IO Unit := do
  timingPrint "timing: core stages start"
  timingFlush
  let lnBounds ← timePureWithHeartbeat "core: ln bounds" (fun () =>
    Sound.headLnBounds inputs)
  let lnLo := lnBounds.1
  let lnHi := lnBounds.2
  let lnAbsMaxTask : Fin seq → Rat :=
    Sound.Bounds.cacheBoundTask (fun q =>
      Sound.Bounds.intervalAbsBound (lnLo q) (lnHi q))
  let lnAbsMaxArr ← timePureWithHeartbeat "core: lnAbsMax force" (fun () =>
    Array.ofFn (fun q : Fin seq => lnAbsMaxTask q))
  let lnAbsMax : Fin seq → Rat := fun q =>
    lnAbsMaxArr.getD q.1 (0 : Rat)
  let lnAbsMaxMax : Rat :=
    let univ : Finset (Fin seq) := Finset.univ
    have hnonempty : univ.Nonempty := by
      simp [univ]
    univ.sup' hnonempty (fun q => lnAbsMax q)
  let qAbsRowTasks : Array (Task (Array Rat)) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        Array.ofFn (fun d : Fin dHead =>
          Sound.Bounds.dotIntervalAbsBound (fun j => inputs.wq j d) (lnLo q) (lnHi q) +
            |inputs.bq d|)))
  let qAbsBaseArr ← timePureWithHeartbeat "core: qAbs base force" (fun () =>
    let defaultTask : Task (Array Rat) := Task.spawn (fun _ => #[])
    Array.ofFn (fun q : Fin seq =>
      (qAbsRowTasks.getD q.1 defaultTask).get))
  let qAbsBase : Fin seq → Fin dHead → Rat := fun q d =>
    let row := qAbsBaseArr.getD q.1 #[]
    row.getD d.1 (0 : Rat)
  let kAbsRowTasks : Array (Task (Array Rat)) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        Array.ofFn (fun d : Fin dHead =>
          Sound.Bounds.dotIntervalAbsBound (fun j => inputs.wk j d) (lnLo q) (lnHi q) +
            |inputs.bk d|)))
  let kAbsBaseArr ← timePureWithHeartbeat "core: kAbs base force" (fun () =>
    let defaultTask : Task (Array Rat) := Task.spawn (fun _ => #[])
    Array.ofFn (fun q : Fin seq =>
      (kAbsRowTasks.getD q.1 defaultTask).get))
  let kAbsBase : Fin seq → Fin dHead → Rat := fun q d =>
    let row := kAbsBaseArr.getD q.1 #[]
    row.getD d.1 (0 : Rat)
  let dotAbs ← timePureWithHeartbeat "core: dotAbs tasks" (fun () =>
    dotAbsFromQKV qAbsBase kAbsBase)
  let _ ← timePureWithHeartbeat "core: dotAbs force" (fun () =>
    match List.finRange seq with
    | [] => (0 : Rat)
    | q :: _ =>
        match List.finRange seq with
        | [] => (0 : Rat)
        | k :: _ => dotAbs q k)
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
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
  let margin ← timePureWithHeartbeat "core: margin" (fun () =>
    if h : inputs.active.Nonempty then
      inputs.active.inf' h marginAt
    else
      (0 : Rat))
  let marginNeg ← timePureWithHeartbeat "core: margin < 0" (fun () =>
    decide (margin < 0))
  let verboseTiming ← IO.getEnv "NFP_TIMING_VERBOSE"
  if verboseTiming.isSome then
    timingPrint s!"timing: core: margin neg={marginNeg}"
  let tEps0 ← monoUsNow
  timingPrint "timing: core: eps start"
  timingFlush
  let eps :=
    if marginNeg then
      (1 : Rat)
    else
      ratDivUp (seq - 1) (1 + margin)
  let tEps1 ← monoUsNow
  timingPrint s!"timing: core: eps {tEps1 - tEps0} us"
  timingFlush
  let _ := marginAt
  let dirHeadVec ← timePureWithHeartbeat "core: dir head vec" (fun () =>
    Sound.dirHeadVecOfInputs inputs)
  let dirHead : Fin dHead → Rat := fun d => dirHeadVec.get d
  let wvDir : Fin dModel → Rat :=
    Sound.Bounds.cacheBoundTask (fun j =>
      Sound.Linear.dotFin dHead dirHead (fun d => inputs.wv j d))
  let _ ← timePureWithHeartbeat "core: wvDir force" (fun () =>
    Array.ofFn (fun j : Fin dModel => wvDir j))
  let bDir ← timePureWithHeartbeat "core: bDir" (fun () =>
    Sound.Linear.dotFin dHead dirHead (fun d => inputs.bv d))
  let valsAbsBase ← timePureWithHeartbeat "core: valsAbsBase" (fun () =>
    Sound.Linear.sumFin dModel (fun j => |wvDir j|) * lnAbsMaxMax)
  let valsLoBase := bDir - valsAbsBase
  let valsHiBase := bDir + valsAbsBase
  let valsLo : Fin seq → Rat := fun _ => valsLoBase
  let valsHi : Fin seq → Rat := fun _ => valsHiBase
  let _ ← timePureWithHeartbeat "core: value bounds" (fun () =>
    let univ : Finset (Fin seq) := Finset.univ
    have hnonempty : univ.Nonempty := by
      simp [univ]
    let lo := univ.inf' hnonempty valsLo
    let hi := univ.sup' hnonempty valsHi
    (lo, hi))
  timingPrint "timing: core stages done"
  timingFlush

end IO

end Nfp
