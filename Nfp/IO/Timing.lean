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

/-- Append a timing log line to `NFP_TIMING_LOG` when set. -/
def logTiming (line : String) : IO Unit := do
  match (← IO.getEnv "NFP_TIMING_LOG") with
  | some path =>
      let h ← IO.FS.Handle.mk (System.FilePath.mk path) IO.FS.Mode.append
      h.putStr (line ++ "\n")
      h.flush
  | none => pure ()

/-- Time an IO phase and print the duration in microseconds. -/
def timePhase {α : Type} (label : String) (act : IO α) : IO α := do
  logTiming s!"start: {label}"
  let t0 ← monoUsNow
  let res ← act
  let t1 ← monoUsNow
  logTiming s!"done: {label} {t1 - t0} us"
  IO.println s!"timing: {label} {t1 - t0} us"
  return res

/-- Time an IO phase supplied as a thunk and print the duration in microseconds. -/
def timePhaseThunk {α : Type} (label : String) (act : Unit → IO α) : IO α := do
  logTiming s!"start: {label}"
  let t0 ← monoUsNow
  let res ← act ()
  let t1 ← monoUsNow
  logTiming s!"done: {label} {t1 - t0} us"
  IO.println s!"timing: {label} {t1 - t0} us"
  return res

/-- Time a pure thunk and print the duration in microseconds. -/
def timePure {α : Type} (label : String) (f : Unit → α) : IO α := do
  logTiming s!"start: {label}"
  let t0 ← monoUsNow
  let res := f ()
  let t1 ← monoUsNow
  logTiming s!"done: {label} {t1 - t0} us"
  IO.println s!"timing: {label} {t1 - t0} us"
  return res

/-- Flush stdout immediately for interleaved timing output. -/
def flushStdout : IO Unit := do
  let h ← IO.getStdout
  h.flush

/-- Measure task spawn/get overhead on this machine. -/
def taskBench (n : Nat) : IO Unit := do
  if n = 0 then
    IO.println "timing: task bench skipped (n=0)"
    return
  let t0 ← monoUsNow
  let tasks := (List.range n).map (fun _ => Task.spawn (fun _ => ()))
  for t in tasks do
    let _ := t.get
    pure ()
  let t1 ← monoUsNow
  let total := t1 - t0
  let avg := total / n
  IO.println s!"timing: task bench n={n} total={total} us avg={avg} us"

/-- Force a sample score-gap computation for timing. -/
def timeHeadScoreSampleGap {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (score : Sound.HeadScoreBounds seq dModel dHead) : IO Unit := do
  IO.println "timing: head score sample gap start"
  (← IO.getStdout).flush
  let t0 ← monoUsNow
  match List.finRange seq with
  | [] =>
      IO.println "timing: head score sample gap skipped (empty seq)"
  | q :: _ =>
      let _ := score.scoreLo q (inputs.prev q)
      let _ := score.scoreHi q (inputs.prev q)
      let _ := score.scoreLo q (inputs.prev q) - score.scoreHi q (inputs.prev q)
      pure ()
  let t1 ← monoUsNow
  IO.println s!"timing: head score sample gap {t1 - t0} us"
  (← IO.getStdout).flush

/-- Force marginAt evaluation over the active list for timing. -/
def timeHeadScoreMarginList {seq dModel dHead : Nat}
    (activeList : List (Fin seq))
    (score : Sound.HeadScoreBounds seq dModel dHead) : IO Unit := do
  IO.println "timing: head score marginAt list start"
  (← IO.getStdout).flush
  let t0 ← monoUsNow
  for q in activeList do
    let _ := score.marginAt q
    pure ()
  let t1 ← monoUsNow
  IO.println s!"timing: head score marginAt list {t1 - t0} us"
  (← IO.getStdout).flush

/-- Force marginAt evaluation without constructing the full score bounds record. -/
def timeHeadScoreMarginRaw {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dotAbs : Fin seq → Fin seq → Dyadic)
    (activeList : List (Fin seq)) : IO Unit := do
  IO.println "timing: head score marginRaw list start"
  (← IO.getStdout).flush
  let t0 ← monoUsNow
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let scoreBaseAbs : Fin seq → Fin seq → Dyadic := fun q k =>
    |inputs.scale| * dotAbs q k
  let scoreLo : Fin seq → Fin seq → Dyadic := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      -scoreBaseAbs q k
  let scoreHi : Fin seq → Fin seq → Dyadic := fun q k =>
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
  let maskedGap : Fin seq → Dyadic := fun q =>
    scoreLo q (inputs.prev q) - inputs.maskValue
  let scoreGap : Fin seq → Fin seq → Dyadic := fun q k =>
    scoreLo q (inputs.prev q) - scoreHi q k
  let marginAtRaw : Fin seq → Dyadic := fun q =>
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
        (0 : Dyadic)
  for q in activeList do
    let _ := marginAtRaw q
    pure ()
  let t1 ← monoUsNow
  IO.println s!"timing: head score marginRaw list {t1 - t0} us"
  (← IO.getStdout).flush

/-- Force individual score-bound fields to locate slow evaluations. -/
def timeHeadScoreFieldForces {seq dModel dHead : Nat}
    (score : Sound.HeadScoreBounds seq dModel dHead) : IO Unit := do
  IO.println "timing: head score field force start"
  (← IO.getStdout).flush
  let timeOne (label : String) (f : Unit → IO Unit) : IO Unit := do
    let t0 ← monoUsNow
    f ()
    let t1 ← monoUsNow
    IO.println s!"timing: head score field {label} {t1 - t0} us"
    (← IO.getStdout).flush
  match List.finRange seq with
  | [] =>
      IO.println "timing: head score field force skipped (empty seq)"
      (← IO.getStdout).flush
  | q :: _ =>
      match List.finRange seq with
      | [] =>
          IO.println "timing: head score field force skipped (empty seq)"
          (← IO.getStdout).flush
      | k :: _ =>
          timeOne "scoreBaseAbs" (fun _ => do let _ := score.scoreBaseAbs q k; pure ())
          timeOne "scoreAbs" (fun _ => do let _ := score.scoreAbs q k; pure ())
          timeOne "scoreLo" (fun _ => do let _ := score.scoreLo q k; pure ())
          timeOne "scoreHi" (fun _ => do let _ := score.scoreHi q k; pure ())
          timeOne "marginAt" (fun _ => do let _ := score.marginAt q; pure ())
          timeOne "epsAt" (fun _ => do let _ := score.epsAt q; pure ())
          timeOne "margin" (fun _ => do let _ := score.margin; pure ())
          timeOne "eps" (fun _ => do let _ := score.eps; pure ())
  IO.println "timing: head score field force done"
  (← IO.getStdout).flush

end IO

end Nfp
