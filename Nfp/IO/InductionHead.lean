-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.List.Range
import Nfp.IO.Pure
import Nfp.IO.NfptPure
import Nfp.IO.HeadScore
import Nfp.IO.Timing
import Nfp.IO.Util
import Nfp.Circuit.Cert.LogitDiff
import Nfp.Circuit.Cert.ResidualInterval
import Nfp.Sound.Induction
import Nfp.Sound.Induction.HeadBounds
import Nfp.Sound.Induction.LogitDiff
import Nfp.Sound.Linear.FinFold

/-!
IO helpers for induction-head certificate construction.
-/

namespace Nfp

namespace IO

open Nfp.Circuit

private def valueBoundsModeFromEnv : IO (Option Bool) := do
  match (← IO.getEnv "NFP_VALUE_BOUNDS_MODE") with
  | some "common" => return some true
  | some "cached" => return some false
  | _ => return none

/-- Load induction head inputs from disk. -/
def loadInductionHeadInputs (path : System.FilePath) :
    IO (Except String (Sigma (fun seq =>
      Sigma (fun dModel => Sigma (fun dHead => Model.InductionHeadInputs seq dModel dHead))))) := do
  let t0 ← monoUsNow
  let data ← IO.FS.readFile path
  let t1 ← monoUsNow
  IO.println s!"timing: read head input file {t1 - t0} us"
  let t2 ← monoUsNow
  let parsed :=
    match Pure.parseInductionHeadInputs data with
    | Except.error msg => Except.error msg
    | Except.ok v => Except.ok v
  let t3 ← monoUsNow
  IO.println s!"timing: parse head input file {t3 - t2} us"
  return parsed

private def dyadicToString (x : Dyadic) : String :=
  toString x.toRat

private def renderResidualIntervalCert {n : Nat} (c : Circuit.ResidualIntervalCert n) : String :=
  let header := s!"dim {n}"
  let lines :=
    (List.finRange n).foldr (fun i acc =>
      s!"lo {i.val} {dyadicToString (c.lo i)}" ::
        s!"hi {i.val} {dyadicToString (c.hi i)}" :: acc) []
  String.intercalate "\n" (header :: lines)

private def emitResidualIntervalCert {n : Nat} (c : Circuit.ResidualIntervalCert n)
    (outPath? : Option System.FilePath) : IO Unit := do
  let payload := renderResidualIntervalCert c
  match outPath? with
  | some path => IO.FS.writeFile path (payload ++ "\n")
  | none => IO.println payload

private def buildHeadOutputIntervalFromInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (outPath? : Option System.FilePath) : IO UInt32 := do
  match seq with
  | 0 =>
      IO.eprintln "error: seq must be positive"
      return 2
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      match Sound.buildHeadOutputIntervalFromHead? inputs with
      | none =>
          IO.eprintln "error: head output interval rejected"
          return 2
      | some result =>
          emitResidualIntervalCert result.cert outPath?
          if outPath?.isSome then
            let activeCount := result.active.card
            IO.println
              s!"ok: head output interval built (seq={seq}, dim={dModel}, active={activeCount})"
          return 0

private def headScoreBoundsFromDotAbsTimed {seq dModel dHead : Nat} [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (dotAbs : Fin seq → Fin seq → Dyadic) :
    IO (Sound.HeadScoreBounds seq dModel dHead) := do
  let headScoreBoundsFromCachesTimed
      (scoreLo scoreHi : Fin seq → Fin seq → Dyadic) :
      IO (Sound.HeadScoreBounds seq dModel dHead) := do
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
    let marginTasks : { arr : Array (Task Dyadic) // arr.size = seq } ←
      timePhase "head: score margin tasks" <| do
        let arr : Array (Task Dyadic) :=
          Array.ofFn (fun q : Fin seq =>
            Task.spawn (fun _ =>
              if q ∈ inputs.active then
                let other := unmaskedKeys q
                let masked := maskedKeys q
                let prev := inputs.prev q
                let gapTasks : Array (Task Dyadic) :=
                  Array.ofFn (fun k : Fin seq =>
                    Task.spawn (fun _ => scoreLo q prev - scoreHi q k))
                let gap : Fin seq → Dyadic := fun k =>
                  let row := gapTasks[k.1]'(by
                    simp [gapTasks, k.isLt])
                  row.get
                if hunmasked : other.Nonempty then
                  let unmaskedMin := other.inf' hunmasked gap
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
        let hsize : arr.size = seq := by simp [arr]
        pure ⟨arr, hsize⟩
    have hmargin : marginTasks.1.size = seq := marginTasks.2
    let marginAt : Fin seq → Dyadic := fun q =>
      let q' : Fin marginTasks.1.size := Fin.cast hmargin.symm q
      (marginTasks.1[q'.1]'(by exact q'.isLt)).get
    let epsTasks : { arr : Array (Task Dyadic) // arr.size = seq } ←
      timePhase "head: score eps tasks" <| do
        let arr : Array (Task Dyadic) :=
          Array.ofFn (fun q : Fin seq =>
            let q' : Fin marginTasks.1.size := Fin.cast hmargin.symm q
            (marginTasks.1[q'.1]'(by exact q'.isLt)).map (fun m =>
              if m < 0 then
                (1 : Dyadic)
              else
                dyadicDivUp (seq - 1) (1 + m)))
        let hsize : arr.size = seq := by simp [arr]
        pure ⟨arr, hsize⟩
    have heps : epsTasks.1.size = seq := epsTasks.2
    let epsAt : Fin seq → Dyadic := fun q =>
      let q' : Fin epsTasks.1.size := Fin.cast heps.symm q
      (epsTasks.1[q'.1]'(by exact q'.isLt)).get
    let margin ← timePhase "head: score margin reduction" <|
      pure (if h : inputs.active.Nonempty then
        inputs.active.inf' h marginAt
      else
        (0 : Dyadic))
    let eps ← timePhase "head: score eps reduction" <|
      pure (if margin < 0 then
        (1 : Dyadic)
      else
        dyadicDivUp (seq - 1) (1 + margin))
    let result : Sound.HeadScoreBounds seq dModel dHead :=
      { dotAbs := dotAbs
        scoreBaseAbs := scoreBaseAbs
        scoreAbs := scoreAbs
        scoreLo := scoreLo
        scoreHi := scoreHi
        marginAt := marginAt
        epsAt := epsAt
        margin := margin
        eps := eps }
    return result
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let scoreBaseAbs : Fin seq → Fin seq → Dyadic := fun q k =>
    |inputs.scale| * dotAbs q k
  let scoreLoRaw : Fin seq → Fin seq → Dyadic := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      -scoreBaseAbs q k
  let scoreHiRaw : Fin seq → Fin seq → Dyadic := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      scoreBaseAbs q k
  IO.println "timing: head score caches skipped (direct score functions)"
  flushStdout
  let scoreLo : Fin seq → Fin seq → Dyadic := fun q k =>
    if masked q k then inputs.maskValue else -(|inputs.scale| * dotAbs q k)
  let scoreHi : Fin seq → Fin seq → Dyadic := fun q k =>
    if masked q k then inputs.maskValue else |inputs.scale| * dotAbs q k
  headScoreBoundsFromCachesTimed scoreLo scoreHi

private def headScoreBoundsFromQAbsKAbsTimed {seq dModel dHead : Nat} [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qAbs kAbs : Fin seq → Fin dHead → Dyadic)
    (dotAbs : Fin seq → Fin seq → Dyadic) :
    IO (Sound.HeadScoreBounds seq dModel dHead) := do
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
  let maskedKeys : Fin seq → Finset (Fin seq) := fun q =>
    if inputs.maskCausal = true then
      (otherKeys q).filter (fun k => q < k)
    else
      (∅ : Finset (Fin seq))
  let unmaskedKeys : Fin seq → Finset (Fin seq) := fun q =>
    (otherKeys q) \ (maskedKeys q)
  let scoreBaseAbs : Fin seq → Fin seq → Dyadic := fun q k =>
    |inputs.scale| * dotAbs q k
  let scoreLo : Fin seq → Fin seq → Dyadic := fun q k =>
    if masked q k then inputs.maskValue else -scoreBaseAbs q k
  let scoreHi : Fin seq → Fin seq → Dyadic := fun q k =>
    if masked q k then inputs.maskValue else scoreBaseAbs q k
  let kAbsMax : Fin dHead → Dyadic := fun d =>
    let univ : Finset (Fin seq) := Finset.univ
    have hnonempty : univ.Nonempty := Finset.univ_nonempty
    univ.sup' hnonempty (fun k => kAbs k d)
  let dotAbsUpper : Fin seq → Dyadic := fun q =>
    Sound.Linear.dotFin dHead (fun d => qAbs q d) kAbsMax
  let scoreHiUpper : Fin seq → Dyadic := fun q =>
    max inputs.maskValue (|inputs.scale| * dotAbsUpper q)
  let fastGap : Fin seq → Dyadic := fun q =>
    let prev := inputs.prev q
    scoreLo q prev - scoreHiUpper q
  let marginTasks : Array (Task Dyadic) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        if q ∈ inputs.active then
          let fast := fastGap q
          if fast < 0 then
            let other := unmaskedKeys q
            let maskedSet := maskedKeys q
            let exact :=
              if hunmasked : other.Nonempty then
                let unmaskedMin :=
                  other.inf' hunmasked (fun k => scoreLo q (inputs.prev q) - scoreHi q k)
                if maskedSet.Nonempty then
                  min unmaskedMin (scoreLo q (inputs.prev q) - inputs.maskValue)
                else
                  unmaskedMin
              else
                if maskedSet.Nonempty then
                  scoreLo q (inputs.prev q) - inputs.maskValue
                else
                  (0 : Dyadic)
            exact
          else
            fast
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
  let result : Sound.HeadScoreBounds seq dModel dHead :=
    { dotAbs := dotAbs
      scoreBaseAbs := scoreBaseAbs
      scoreAbs := fun q k => if masked q k then |inputs.maskValue| else scoreBaseAbs q k
      scoreLo := scoreLo
      scoreHi := scoreHi
      marginAt := marginAt
      epsAt := epsAt
      margin := margin
      eps := eps }
  return result

private def checkInductionHeadInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (minActive? : Option Nat) (minLogitDiff? : Option Dyadic)
    (minMargin maxEps : Dyadic) : IO UInt32 := do
  match seq with
  | 0 =>
      IO.eprintln "error: seq must be positive"
      return 2
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      logTiming "start: head build induction cert"
      IO.println "timing: head build induction cert start"
      flushStdout
      let verboseTiming ← IO.getEnv "NFP_TIMING_VERBOSE"
      let taskBenchEnv ← IO.getEnv "NFP_TASK_BENCH"
      if taskBenchEnv.isSome then
        let n := taskBenchEnv.bind String.toNat? |>.getD 1000
        Nfp.IO.taskBench n
      if verboseTiming.isSome then
        IO.println s!"timing: head dims seq={seq} dModel={dModel} dHead={dHead}"
        IO.println s!"timing: head active card={inputs.active.card}"
        flushStdout
      let precompute := (← IO.getEnv "NFP_TIMING_PRECOMPUTE").isSome
      if precompute then
        IO.println "timing: head ln bounds start"
        flushStdout
        let lnBounds ← timePure "head: ln bounds" (fun () =>
          Sound.headLnBounds inputs)
        IO.println "timing: head ln bounds done"
        flushStdout
        IO.println "timing: head qkv bounds start"
        flushStdout
        let lnLo := lnBounds.1
        let lnHi := lnBounds.2
        let qkv ← timePure "head: qkv bounds" (fun () =>
          Sound.headQKVBounds inputs lnLo lnHi)
        IO.println "timing: head qkv bounds done"
        flushStdout
        if verboseTiming.isSome then
          IO.println "timing: head qkv abs force start"
          flushStdout
          let tAbs0 ← monoUsNow
          for q in List.finRange seq do
            for d in List.finRange dHead do
              let _ := qkv.qAbs q d
              let _ := qkv.kAbs q d
              pure ()
          let tAbs1 ← monoUsNow
          IO.println s!"timing: head qkv abs force {tAbs1 - tAbs0} us"
          flushStdout
        IO.println "timing: head score/value bounds spawn start"
        flushStdout
        let tSpawn0 ← monoUsNow
        if verboseTiming.isSome then
          IO.println "timing: head score dotAbs tasks start"
          flushStdout
        let dotAbs ← timePure "head: score dotAbs tasks" (fun () =>
          dotAbsFromQKV qkv.qAbs qkv.kAbs)
        if verboseTiming.isSome then
          IO.println "timing: head score dotAbs tasks done"
          flushStdout
        if verboseTiming.isSome then
          IO.println "timing: head score dotAbs force start"
          flushStdout
          let tForce0 ← monoUsNow
          match List.finRange seq with
          | [] =>
              IO.println "timing: head score dotAbs force skipped (empty seq)"
          | q :: _ =>
              match List.finRange seq with
              | [] =>
                  IO.println "timing: head score dotAbs force skipped (empty seq)"
              | k :: _ =>
                  let _ := dotAbs q k
                  pure ()
          let tForce1 ← monoUsNow
          IO.println s!"timing: head score dotAbs force {tForce1 - tForce0} us"
          flushStdout
        let inlineVals := (← IO.getEnv "NFP_TIMING_VALUE_INLINE").isSome
        let valueMode? ← valueBoundsModeFromEnv
        let useCommon := valueMode?.getD false
        let (valsInline?, valsTask?) :=
          if inlineVals then
            let vals :=
              if useCommon then
                Sound.headValueBoundsCommonDen inputs qkv.vLo qkv.vHi
              else
                Sound.headValueBounds inputs qkv.vLo qkv.vHi
            (some vals, none)
          else
            let task := Task.spawn (fun _ =>
              if useCommon then
                Sound.headValueBoundsCommonDen inputs qkv.vLo qkv.vHi
              else
                Sound.headValueBounds inputs qkv.vLo qkv.vHi)
            (none, some task)
        let activeList := (List.finRange seq).filter (fun q => q ∈ inputs.active)
        if verboseTiming.isSome then
          timeHeadScoreMarginRaw inputs dotAbs activeList
        let tSpawn1 ← monoUsNow
        IO.println s!"timing: head score/value bounds spawn {tSpawn1 - tSpawn0} us"
        flushStdout
        let skipScoreBounds := (← IO.getEnv "NFP_TIMING_SKIP_SCORE_BOUNDS").isSome
        let scoreOpt ←
          if skipScoreBounds then
            IO.println "timing: head score bounds skipped"
            pure none
          else
            IO.println "timing: head score bounds from dotAbs start"
            flushStdout
            let fastMargin := (← IO.getEnv "NFP_TIMING_FAST_MARGIN").isSome
            let score ←
              if fastMargin then
                headScoreBoundsFromQAbsKAbsTimed inputs qkv.qAbs qkv.kAbs dotAbs
              else
                headScoreBoundsFromDotAbsTimed inputs dotAbs
            IO.println "timing: head score bounds from dotAbs done"
            flushStdout
            pure (some score)
        match scoreOpt with
        | none => pure ()
        | some score =>
            if verboseTiming.isSome then
              timeHeadScoreSampleGap inputs score
            if verboseTiming.isSome then
              timeHeadScoreMarginList activeList score
            if verboseTiming.isSome then
              timeHeadScoreFieldForces score
            if verboseTiming.isSome then
              IO.println "timing: head score bounds force start"
              flushStdout
              let tScore0 ← monoUsNow
              let _ := score.margin
              let _ := score.eps
              let tScore1 ← monoUsNow
              IO.println s!"timing: head score bounds force {tScore1 - tScore0} us"
              flushStdout
        if verboseTiming.isSome then
          IO.println "timing: head value parts start"
          flushStdout
          IO.println "timing: head value dirHead start"
          flushStdout
          let tDir0 ← monoUsNow
          let dirHead := Sound.headValueDirHead inputs
          match List.finRange dHead with
          | [] =>
              IO.println "timing: head value dirHead forced skipped (empty dHead)"
          | d :: _ =>
              let _ := dirHead d
              pure ()
          let tDir1 ← monoUsNow
          IO.println s!"timing: head value dirHead {tDir1 - tDir0} us"
          flushStdout
          IO.println "timing: head value valsLo start"
          flushStdout
          let tLo0 ← monoUsNow
          let valsLo := Sound.headValueValsLo inputs qkv.vLo qkv.vHi
          match List.finRange seq with
          | [] =>
              IO.println "timing: head value valsLo forced skipped (empty seq)"
          | k :: _ =>
              let _ := valsLo k
              pure ()
          let tLo1 ← monoUsNow
          IO.println s!"timing: head value valsLo {tLo1 - tLo0} us"
          flushStdout
          IO.println "timing: head value valsHi start"
          flushStdout
          let tHi0 ← monoUsNow
          let valsHi := Sound.headValueValsHi inputs qkv.vLo qkv.vHi
          match List.finRange seq with
          | [] =>
              IO.println "timing: head value valsHi forced skipped (empty seq)"
          | k :: _ =>
              let _ := valsHi k
              pure ()
          let tHi1 ← monoUsNow
          IO.println s!"timing: head value valsHi {tHi1 - tHi0} us"
          flushStdout
          IO.println "timing: head value lo start"
          flushStdout
          let tLo2 ← monoUsNow
          let _ := Sound.headValueLo valsLo
          let tLo3 ← monoUsNow
          IO.println s!"timing: head value lo {tLo3 - tLo2} us"
          flushStdout
          IO.println "timing: head value hi start"
          flushStdout
          let tHi2 ← monoUsNow
          let _ := Sound.headValueHi valsHi
          let tHi3 ← monoUsNow
          IO.println s!"timing: head value hi {tHi3 - tHi2} us"
          flushStdout
          IO.println "timing: head value parts done"
          flushStdout
        IO.println "timing: head value bounds start"
        flushStdout
        let tVals0 ← monoUsNow
        let vals ←
          match valsInline?, valsTask? with
          | some vals, _ =>
              timePure "head: value bounds inline" (fun () => vals)
          | none, some valsTask =>
              timePure "head: value bounds wait" (fun () => valsTask.get)
          | none, none =>
              timePure "head: value bounds inline" (fun () =>
                Sound.headValueBounds inputs qkv.vLo qkv.vHi)
        let tVals1 ← monoUsNow
        IO.println s!"timing: head value bounds {tVals1 - tVals0} us"
        flushStdout
      let certOpt :
          Option { c : Sound.InductionHeadCert seq // Sound.InductionHeadCertSound inputs c } ←
        timePure "head: build induction cert" (fun () =>
          match Sound.buildInductionCertFromHead? inputs with
          | none => none
          | some ⟨cert, hcert⟩ =>
              let _ := cert.active.card
              some ⟨cert, hcert⟩)
      IO.println "timing: head build induction cert returned"
      flushStdout
      logTiming "done: head build induction cert"
      match certOpt with
      | none =>
          IO.eprintln "error: head inputs rejected"
          return 2
      | some ⟨cert, _hcert⟩ =>
          IO.println "timing: head active count start"
          flushStdout
          let activeCount := cert.active.card
          IO.println "timing: head active count done"
          flushStdout
          let defaultMinActive := max 1 (seq / 8)
          let minActive := minActive?.getD defaultMinActive
          if activeCount < minActive then
            IO.eprintln
              s!"error: active queries {activeCount} below minimum {minActive}"
            return 2
          if cert.margin < minMargin then
            IO.eprintln
              s!"error: margin {dyadicToString cert.margin} \
              below minimum {dyadicToString minMargin}"
            return 2
          if maxEps < cert.eps then
            IO.eprintln
              s!"error: eps {dyadicToString cert.eps} \
              above maximum {dyadicToString maxEps}"
            return 2
          IO.println "timing: head tol start"
          flushStdout
          let tol := cert.eps * (cert.values.hi - cert.values.lo)
          IO.println "timing: head tol done"
          flushStdout
          logTiming "start: head logit-diff lower bound"
          IO.println "timing: head logit-diff lower bound start"
          flushStdout
          let logitDiffLB? ← timePure "head: logit-diff lower bound" (fun () =>
            Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
              cert.values.lo cert.values.hi cert.values.valsLo)
          logTiming "done: head logit-diff lower bound"
          let effectiveMinLogitDiff :=
            match minLogitDiff? with
            | some v => some v
            | none => some (0 : Dyadic)
          match logitDiffLB? with
          | none =>
              IO.eprintln "error: empty active set for logit-diff bound"
              return 2
          | some logitDiffLB =>
              let violation? : Option Dyadic :=
                match effectiveMinLogitDiff with
                | none => none
                | some minLogitDiff =>
                    if logitDiffLB < minLogitDiff then
                      some minLogitDiff
                    else
                      none
              match violation? with
              | some minLogitDiff =>
                  IO.eprintln
                    s!"error: logitDiffLB {dyadicToString logitDiffLB} \
                    below minimum {dyadicToString minLogitDiff}"
                  return 2
              | none =>
                  IO.println
                    s!"ok: induction bound certified \
                    (seq={seq}, active={activeCount}, \
                    tol={dyadicToString tol}, logitDiffLB={dyadicToString logitDiffLB})"
                  return 0

private def checkInductionHeadInputsNonvacuous {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (minActive? : Option Nat) (minLogitDiff? : Option Dyadic)
    (minMargin maxEps : Dyadic) : IO UInt32 := do
  match seq with
  | 0 =>
      IO.eprintln "error: seq must be positive"
      return 2
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      logTiming "start: head build nonvacuous logit-diff"
      let res : Option (Sound.InductionLogitLowerBoundNonvacuous inputs) ←
        timePure "head: build nonvacuous logit-diff" (fun () =>
          Sound.buildInductionLogitLowerBoundNonvacuous? inputs)
      logTiming "done: head build nonvacuous logit-diff"
      match res with
      | none =>
          IO.eprintln "error: nonvacuous logit-diff construction failed"
          return 2
      | some result =>
          let cert := result.base.cert
          let logitDiffLB := result.base.lb
          let activeCount := cert.active.card
          let defaultMinActive := max 1 (seq / 8)
          let minActive := minActive?.getD defaultMinActive
          if activeCount < minActive then
            IO.eprintln
              s!"error: active queries {activeCount} below minimum {minActive}"
            return 2
          if cert.margin < minMargin then
            IO.eprintln
              s!"error: margin {dyadicToString cert.margin} \
              below minimum {dyadicToString minMargin}"
            return 2
          if maxEps < cert.eps then
            IO.eprintln
              s!"error: eps {dyadicToString cert.eps} above maximum {dyadicToString maxEps}"
            return 2
          match minLogitDiff? with
          | some minLogitDiff =>
              if logitDiffLB < minLogitDiff then
                IO.eprintln
                  s!"error: logitDiffLB {dyadicToString logitDiffLB} \
                  below minimum {dyadicToString minLogitDiff}"
                return 2
          | none => pure ()
          let tol := cert.eps * (cert.values.hi - cert.values.lo)
          IO.println
            s!"ok: nonvacuous induction bound certified \
            (seq={seq}, active={activeCount}, \
            tol={dyadicToString tol}, logitDiffLB={dyadicToString logitDiffLB})"
          return 0

/-- Build and check induction certificates from exact head inputs. -/
def runInductionCertifyHead (inputsPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseDyadicOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseDyadicOpt "min-margin" minMarginStr?
  let maxEps?E := parseDyadicOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? => do
      let minMargin := minMargin?.getD (0 : Dyadic)
      let maxEps := maxEps?.getD (dyadicOfRatDown (Rat.divInt 1 2))
      let parsedInputs ← timePhase "load head inputs" <|
        loadInductionHeadInputs inputsPath
      match parsedInputs with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨_seq, ⟨_dModel, ⟨_dHead, inputs⟩⟩⟩ =>
          checkInductionHeadInputs inputs minActive? minLogitDiff? minMargin maxEps

/-- Build and check a strictly positive induction logit-diff bound from head inputs. -/
def runInductionCertifyHeadNonvacuous (inputsPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseDyadicOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseDyadicOpt "min-margin" minMarginStr?
  let maxEps?E := parseDyadicOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? => do
      let minMargin := minMargin?.getD (0 : Dyadic)
      let maxEps := maxEps?.getD (dyadicOfRatDown (Rat.divInt 1 2))
      let parsedInputs ← timePhase "load head inputs" <|
        loadInductionHeadInputs inputsPath
      match parsedInputs with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨_seq, ⟨_dModel, ⟨_dHead, inputs⟩⟩⟩ =>
          checkInductionHeadInputsNonvacuous inputs minActive? minLogitDiff? minMargin maxEps

/-- Build and check induction certificates from a model binary. -/
def runInductionCertifyHeadModel (modelPath : System.FilePath)
    (layer head dirTarget dirNegative : Nat) (period? : Option Nat)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseDyadicOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseDyadicOpt "min-margin" minMarginStr?
  let maxEps?E := parseDyadicOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? =>
      let minMargin := minMargin?.getD (0 : Dyadic)
      let maxEps := maxEps?.getD (dyadicOfRatDown (Rat.divInt 1 2))
      logTiming "start: read model file"
      IO.println "timing: read model file start"
      flushStdout
      let data ← timePhase "read model file" <| IO.FS.readBinFile modelPath
      let headerE ← timePure "parse model header" (fun () =>
        NfptPure.parseHeader data)
      match headerE with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨header, start⟩ =>
          let inputsE ← timePure "read head inputs" (fun () =>
            NfptPure.readInductionHeadInputs
              data start header layer head dirTarget dirNegative period?)
          match inputsE with
          | Except.error msg =>
              IO.eprintln s!"error: {msg}"
              return 1
          | Except.ok inputs =>
              checkInductionHeadInputs inputs minActive? minLogitDiff? minMargin maxEps

/-- Build and check a strictly positive induction logit-diff bound from a model binary. -/
def runInductionCertifyHeadModelNonvacuous (modelPath : System.FilePath)
    (layer head dirTarget dirNegative : Nat) (period? : Option Nat)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseDyadicOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseDyadicOpt "min-margin" minMarginStr?
  let maxEps?E := parseDyadicOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? =>
      let minMargin := minMargin?.getD (0 : Dyadic)
      let maxEps := maxEps?.getD (dyadicOfRatDown (Rat.divInt 1 2))
      logTiming "start: read model file"
      IO.println "timing: read model file start"
      flushStdout
      let data ← timePhase "read model file" <| IO.FS.readBinFile modelPath
      let headerE ← timePure "parse model header" (fun () =>
        NfptPure.parseHeader data)
      match headerE with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨header, start⟩ =>
          let inputsE ← timePure "read head inputs" (fun () =>
            NfptPure.readInductionHeadInputs
              data start header layer head dirTarget dirNegative period?)
          match inputsE with
          | Except.error msg =>
              IO.eprintln s!"error: {msg}"
              return 1
          | Except.ok inputs =>
              checkInductionHeadInputsNonvacuous inputs minActive? minLogitDiff? minMargin maxEps

/-- Build head-output interval bounds from exact head inputs. -/
def runInductionHeadInterval (inputsPath : System.FilePath)
    (outPath? : Option System.FilePath) : IO UInt32 := do
  let parsedInputs ← loadInductionHeadInputs inputsPath
  match parsedInputs with
  | Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 1
  | Except.ok ⟨_seq, ⟨_dModel, ⟨_dHead, inputs⟩⟩⟩ =>
      buildHeadOutputIntervalFromInputs inputs outPath?

/-- Build head-output interval bounds from a model binary. -/
def runInductionHeadIntervalModel (modelPath : System.FilePath)
    (layer head dirTarget dirNegative : Nat) (period? : Option Nat)
    (outPath? : Option System.FilePath) : IO UInt32 := do
  let data ← IO.FS.readBinFile modelPath
  match NfptPure.parseHeader data with
  | Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 1
  | Except.ok ⟨header, start⟩ =>
      match
        NfptPure.readInductionHeadInputs
          data start header layer head dirTarget dirNegative period?
      with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok inputs =>
          buildHeadOutputIntervalFromInputs inputs outPath?

end IO

end Nfp
