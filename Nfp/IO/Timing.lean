-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.List.Range
import Nfp.Model.InductionHead
import Nfp.Sound.Induction.HeadBounds

/-!
Small IO helpers for benchmarking task overhead and profiling slow phases.
-/

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

/-- Flush stdout immediately for interleaved timing output. -/
def flushStdout : IO Unit := do
  let h ← IO.getStdout
  h.flush

/-- Measure task spawn/get overhead on this machine. -/
def taskBench (n : Nat) : IO Unit := do
  if n = 0 then
    timingPrint "timing: task bench skipped (n=0)"
    return
  let t0 ← monoUsNow
  let tasks := (List.range n).map (fun _ => Task.spawn (fun _ => ()))
  for t in tasks do
    let _ := t.get
    pure ()
  let t1 ← monoUsNow
  let total := t1 - t0
  let avg := total / n
  timingPrint s!"timing: task bench n={n} total={total} us avg={avg} us"

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

end IO

end Nfp
