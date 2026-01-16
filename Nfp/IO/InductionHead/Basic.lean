-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.List.Range
public import Nfp.IO.Pure
public import Nfp.IO.NfptPure
public import Nfp.IO.HeadScore
public import Nfp.IO.Timing
public import Nfp.IO.Util
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.Circuit.Cert.ResidualInterval
public import Nfp.Sound.Induction
public import Nfp.Sound.Induction.HeadBounds
public import Nfp.Sound.Induction.LogitDiff
public import Nfp.Sound.Linear.FinFold

/-!
IO helpers for induction-head certificate construction.
-/

public section

namespace Nfp

namespace IO

private def unwrapTaskResult {α : Type} (res : Except IO.Error α) : IO α :=
  match res with
  | .ok a => pure a
  | .error e => throw e

/-- Configure timing output and heartbeat reporting. -/
def configureTiming (timing? : Option Nat) (heartbeatMs? : Option Nat) : IO Unit := do
  match timing? with
  | some v => setTimingStdout (v ≠ 0)
  | none => pure ()
  match heartbeatMs? with
  | some v =>
      setTimingHeartbeatMs (UInt32.ofNat v)
      if timing?.isNone && (v != 0) then
        setTimingStdout true
  | none => pure ()

/-- Translate CLI split-budget options into a split config. -/
def splitConfigFromOptions
    (splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined? : Option Nat) :
    Sound.InductionHeadSplitConfig :=
  let base := Sound.defaultInductionHeadSplitConfig
  { base with
    splitBudgetQ := splitBudgetQ?.getD base.splitBudgetQ
    splitBudgetK := splitBudgetK?.getD base.splitBudgetK
    splitBudgetDiffBase := splitBudgetDiffBase?.getD base.splitBudgetDiffBase
    splitBudgetDiffRefined := splitBudgetDiffRefined?.getD base.splitBudgetDiffRefined }

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
  timingPrint s!"timing: read head input file {t1 - t0} us"
  let t2 ← monoUsNow
  let parsed :=
    match Pure.parseInductionHeadInputs data with
    | Except.error msg => Except.error msg
    | Except.ok v => Except.ok v
  let t3 ← monoUsNow
  timingPrint s!"timing: parse head input file {t3 - t2} us"
  return parsed

/-- Render a rational for logging. -/
def ratToString (x : Rat) : String :=
  toString x

/-- Render an optional rational for logging. -/
def ratOptToString (x : Option Rat) : String :=
  match x with
  | some v => ratToString v
  | none => "none"

/-- Check whether logit-diff debug logging is enabled. -/
def logitDiffDebugEnabled : IO Bool := do
  return (← IO.getEnv "NFP_LOGITDIFF_DEBUG").isSome

/-- Check whether logit-diff debug should exit early after dumping a witness. -/
def logitDiffDebugEarlyExitEnabled : IO Bool := do
  return (← IO.getEnv "NFP_LOGITDIFF_DEBUG_EARLY_EXIT").isSome

/-- Check whether logit-diff refinement debug output is enabled. -/
def logitDiffRefineEnabled : IO Bool := do
  return (← IO.getEnv "NFP_LOGITDIFF_REFINE").isSome

/-- Parse an optional query index for alternative logit-diff bound diagnostics. -/
def logitDiffAltBoundQuery : IO (Option Nat) := do
  match (← IO.getEnv "NFP_LOGITDIFF_ALT_BOUND_Q") with
  | none => return none
  | some txt =>
      match txt.toNat? with
      | some n => return some n
      | none =>
          IO.eprintln s!"warn: invalid NFP_LOGITDIFF_ALT_BOUND_Q={txt}"
          return none

/-- Parse an optional query index for q-only logit-diff diagnostics. -/
def logitDiffQueryOnly : IO (Option Nat) := do
  match (← IO.getEnv "NFP_LOGITDIFF_Q_ONLY") with
  | none => return none
  | some txt =>
      match txt.toNat? with
      | some n => return some n
      | none =>
          IO.eprintln s!"warn: invalid NFP_LOGITDIFF_Q_ONLY={txt}"
          return none

/-- Check whether q-only logit-diff diagnostics should include refined weight bounds. -/
def logitDiffQueryOnlyRefineEnabled : IO Bool := do
  return (← IO.getEnv "NFP_LOGITDIFF_Q_ONLY_REFINE").isSome

/-- Check whether q-only logit-diff diagnostics should include refined value bounds. -/
def logitDiffQueryOnlyValsEnabled : IO Bool := do
  return (← IO.getEnv "NFP_LOGITDIFF_Q_ONLY_VALS").isSome

/-- Check whether q-only logit-diff diagnostics should exit early. -/
def logitDiffQueryOnlyEarlyExitEnabled : IO Bool := do
  return (← IO.getEnv "NFP_LOGITDIFF_Q_ONLY_EARLY_EXIT").isSome

private def renderResidualIntervalCert {n : Nat} (c : Circuit.ResidualIntervalCert n) : String :=
  let header := s!"dim {n}"
  let lines :=
    (List.finRange n).foldr (fun i acc =>
      s!"lo {i.val} {ratToString (c.lo i)}" ::
        s!"hi {i.val} {ratToString (c.hi i)}" :: acc) []
  String.intercalate "\n" (header :: lines)

private def emitResidualIntervalCert {n : Nat} (c : Circuit.ResidualIntervalCert n)
    (outPath? : Option System.FilePath) : IO Unit := do
  let payload := renderResidualIntervalCert c
  match outPath? with
  | some path => IO.FS.writeFile path (payload ++ "\n")
  | none => IO.println payload

/-- Emit q-only logit-diff diagnostics, returning whether early exit was requested. -/
def emitLogitDiffQueryOnly {seq dModel dHead : Nat} [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cfg : Sound.InductionHeadSplitConfig)
    (cache : Sound.InductionHeadCoreCache seq dModel dHead)
    (cert : Sound.InductionHeadCert seq)
    (logitCache : Sound.LogitDiffCache seq) : IO Bool := do
  match (← logitDiffQueryOnly) with
  | none => return false
  | some qNat =>
      if hq : qNat < seq then
        let q : Fin seq := ⟨qNat, hq⟩
        let prev := cert.prev q
        let epsAt : Fin seq → Rat := logitCache.epsAt
        let others : Finset (Fin seq) :=
          (Finset.univ : Finset (Fin seq)).erase prev
        IO.eprintln
          s!"debug: q-only q={qNat} prev={prev.1} \
          epsAt={ratToString (epsAt q)}"
        if (← logitDiffQueryOnlyValsEnabled) then
          let valsLo : Fin seq → Rat := logitCache.valsLo
          let loAt : Rat :=
            if h : others.Nonempty then
              others.inf' h valsLo
            else
              cert.values.lo
          let valsPrevLo := valsLo prev
          let delta := valsPrevLo - loAt
          let gap := epsAt q * max (0 : Rat) delta
          let lbAtQ := valsPrevLo - gap
          IO.eprintln
            s!"debug: q-only loAt={ratToString loAt} \
            valsPrevLo={ratToString valsPrevLo} \
            lbAtQ={ratToString lbAtQ}"
        if (← logitDiffQueryOnlyRefineEnabled) then
          let refineBudget := max 1 cfg.splitBudgetDiffRefined
          let spec := Sound.refineSpecForQueryWithWeightOnes inputs cache q refineBudget
          let weightBoundAt := Sound.weightBoundAtOverlay inputs cache spec
          let epsAtRef := Sound.epsAtOverlay cache weightBoundAt q
          IO.eprintln
            s!"debug: q-only refined budget={refineBudget} \
            epsAt={ratToString epsAtRef}"
          if (← logitDiffQueryOnlyValsEnabled) then
            let valBudget := Sound.refineBudgetBoost refineBudget
            let valKeys := Sound.loAtKeysAt inputs cache q
            let valsLoRef : Fin seq → Rat :=
              Sound.valsLoOverlay inputs cache valBudget valKeys
            let loAtRef : Rat :=
              if h : others.Nonempty then
                others.inf' h valsLoRef
              else
                cert.values.lo
            let valsPrevLoRef := valsLoRef prev
            let deltaRef := valsPrevLoRef - loAtRef
            let gapRef := epsAtRef * max (0 : Rat) deltaRef
            let lbAtQRef := valsPrevLoRef - gapRef
            IO.eprintln
              s!"debug: q-only refined loAt={ratToString loAtRef} \
              valsPrevLo={ratToString valsPrevLoRef} \
              lbAtQ={ratToString lbAtQRef}"
        let earlyExit := (← logitDiffQueryOnlyEarlyExitEnabled) ||
          (← logitDiffDebugEarlyExitEnabled)
        if earlyExit then
          IO.eprintln "debug: early exit requested (q-only)"
          return true
        return false
      else
        IO.eprintln s!"warn: NFP_LOGITDIFF_Q_ONLY={qNat} out of range (seq={seq})"
        return false

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
    (dotAbs : Fin seq → Fin seq → Rat) :
    IO (Sound.HeadScoreBounds seq dModel dHead) := do
  timePure "head: score bounds" (fun () =>
    Sound.headScoreBoundsFromDotAbs inputs dotAbs)

private def headScoreBoundsFromQAbsKAbsTimed {seq dModel dHead : Nat} [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qAbs kAbs : Fin seq → Fin dHead → Rat)
    (dotAbs : Fin seq → Fin seq → Rat) :
    IO (Sound.HeadScoreBounds seq dModel dHead) := do
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let scoreBaseAbs : Fin seq → Fin seq → Rat := fun q k =>
    |inputs.scale| * dotAbs q k
  let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
    if masked q k then inputs.maskValue else -scoreBaseAbs q k
  let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
    if masked q k then inputs.maskValue else scoreBaseAbs q k
  let kAbsMax : Fin dHead → Rat := fun d =>
    let univ : Finset (Fin seq) := Finset.univ
    have hnonempty : univ.Nonempty := Finset.univ_nonempty
    univ.sup' hnonempty (fun k => kAbs k d)
  let dotAbsUpper : Fin seq → Rat := fun q =>
    Sound.Linear.dotFin dHead (fun d => qAbs q d) kAbsMax
  let scoreHiUpper : Fin seq → Rat := fun q =>
    max inputs.maskValue (|inputs.scale| * dotAbsUpper q)
  let marginTasks : Array (Task Rat) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        if q ∈ inputs.active then
          let prev := inputs.prev q
          let scoreLoPrev := scoreLo q prev
          scoreLoPrev - scoreHiUpper q
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
    (cfg : Sound.InductionHeadSplitConfig)
    (minActive? : Option Nat) (minLogitDiff? : Option Rat)
    (minMargin maxEps : Rat) : IO UInt32 := do
  match seq with
  | 0 =>
      IO.eprintln "error: seq must be positive"
      return 2
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      logTiming "start: head build induction cert"
      timingPrint "timing: head build induction cert start"
      timingFlush
      let verboseTiming ← IO.getEnv "NFP_TIMING_VERBOSE"
      if verboseTiming.isSome then
        timingPrint s!"timing: head dims seq={seq} dModel={dModel} dHead={dHead}"
        timingPrint s!"timing: head active card={inputs.active.card}"
        timingFlush
      let precompute := (← IO.getEnv "NFP_TIMING_PRECOMPUTE").isSome
      if precompute then
        timingPrint "timing: head ln bounds start"
        timingFlush
        let lnBounds ← timePure "head: ln bounds" (fun () =>
          Sound.headLnBounds inputs)
        timingPrint "timing: head ln bounds done"
        timingFlush
        timingPrint "timing: head qkv bounds start"
        timingFlush
        let lnLo := lnBounds.1
        let lnHi := lnBounds.2
        let qkv ← timePure "head: qkv bounds" (fun () =>
          Sound.headQKVBounds inputs lnLo lnHi)
        timingPrint "timing: head qkv bounds done"
        timingFlush
        if verboseTiming.isSome then
          timingPrint "timing: head qkv abs force start"
          timingFlush
          let tAbs0 ← monoUsNow
          for q in List.finRange seq do
            for d in List.finRange dHead do
              let _ := qkv.qAbs q d
              let _ := qkv.kAbs q d
              pure ()
          let tAbs1 ← monoUsNow
          timingPrint s!"timing: head qkv abs force {tAbs1 - tAbs0} us"
          timingFlush
        timingPrint "timing: head score/value bounds spawn start"
        timingFlush
        let tSpawn0 ← monoUsNow
        if verboseTiming.isSome then
          timingPrint "timing: head score dotAbs tasks start"
          timingFlush
        let dotAbs ← timePure "head: score dotAbs tasks" (fun () =>
          dotAbsFromQKV qkv.qAbs qkv.kAbs)
        if verboseTiming.isSome then
          timingPrint "timing: head score dotAbs tasks done"
          timingFlush
        if verboseTiming.isSome then
          timingPrint "timing: head score dotAbs force start"
          timingFlush
          let tForce0 ← monoUsNow
          match List.finRange seq with
          | [] =>
              timingPrint "timing: head score dotAbs force skipped (empty seq)"
          | q :: _ =>
              match List.finRange seq with
              | [] =>
                  timingPrint "timing: head score dotAbs force skipped (empty seq)"
              | k :: _ =>
                  let _ := dotAbs q k
                  pure ()
          let tForce1 ← monoUsNow
          timingPrint s!"timing: head score dotAbs force {tForce1 - tForce0} us"
          timingFlush
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
            let task :=
              if useCommon then
                Sound.headValueBoundsCommonDenTask inputs qkv.vLo qkv.vHi
              else
                Sound.headValueBoundsTask inputs qkv.vLo qkv.vHi
            (none, some task)
        let activeList := (List.finRange seq).filter (fun q => q ∈ inputs.active)
        if verboseTiming.isSome then
          timeHeadScoreMarginRaw inputs dotAbs activeList
        let tSpawn1 ← monoUsNow
        timingPrint s!"timing: head score/value bounds spawn {tSpawn1 - tSpawn0} us"
        timingFlush
        let skipScoreBounds := (← IO.getEnv "NFP_TIMING_SKIP_SCORE_BOUNDS").isSome
        let scoreTaskOpt ←
          if skipScoreBounds then
            timingPrint "timing: head score bounds skipped"
            pure none
          else
            timingPrint "timing: head score bounds from dotAbs start"
            timingFlush
            let exactMargin := (← IO.getEnv "NFP_TIMING_EXACT_MARGIN").isSome
            let action :=
              if exactMargin then
                headScoreBoundsFromDotAbsTimed inputs dotAbs
              else
                headScoreBoundsFromQAbsKAbsTimed inputs qkv.qAbs qkv.kAbs dotAbs
            let t ← action.asTask
            pure (some t)
        if verboseTiming.isSome then
          timingPrint "timing: head value parts start"
          timingFlush
          timingPrint "timing: head value dirHead start"
          timingFlush
          let tDir0 ← monoUsNow
          let dirHead := Sound.headValueDirHead inputs
          match List.finRange dHead with
          | [] =>
              timingPrint "timing: head value dirHead forced skipped (empty dHead)"
          | d :: _ =>
              let _ := dirHead d
              pure ()
          let tDir1 ← monoUsNow
          timingPrint s!"timing: head value dirHead {tDir1 - tDir0} us"
          timingFlush
          timingPrint "timing: head value valsLo start"
          timingFlush
          let tLo0 ← monoUsNow
          let valsLo := Sound.headValueValsLo inputs qkv.vLo qkv.vHi
          match List.finRange seq with
          | [] =>
              timingPrint "timing: head value valsLo forced skipped (empty seq)"
          | k :: _ =>
              let _ := valsLo k
              pure ()
          let tLo1 ← monoUsNow
          timingPrint s!"timing: head value valsLo {tLo1 - tLo0} us"
          timingFlush
          timingPrint "timing: head value valsHi start"
          timingFlush
          let tHi0 ← monoUsNow
          let valsHi := Sound.headValueValsHi inputs qkv.vLo qkv.vHi
          match List.finRange seq with
          | [] =>
              timingPrint "timing: head value valsHi forced skipped (empty seq)"
          | k :: _ =>
              let _ := valsHi k
              pure ()
          let tHi1 ← monoUsNow
          timingPrint s!"timing: head value valsHi {tHi1 - tHi0} us"
          timingFlush
          timingPrint "timing: head value lo start"
          timingFlush
          let tLo2 ← monoUsNow
          let _ := Sound.headValueLo valsLo
          let tLo3 ← monoUsNow
          timingPrint s!"timing: head value lo {tLo3 - tLo2} us"
          timingFlush
          timingPrint "timing: head value hi start"
          timingFlush
          let tHi2 ← monoUsNow
          let _ := Sound.headValueHi valsHi
          let tHi3 ← monoUsNow
          timingPrint s!"timing: head value hi {tHi3 - tHi2} us"
          timingFlush
          timingPrint "timing: head value parts done"
          timingFlush
        timingPrint "timing: head value bounds start"
        timingFlush
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
        timingPrint s!"timing: head value bounds {tVals1 - tVals0} us"
        timingFlush
        let scoreOpt ←
          match scoreTaskOpt with
          | none => pure none
          | some scoreTask => do
              let res ← IO.wait scoreTask
              let score ← unwrapTaskResult res
              timingPrint "timing: head score bounds from dotAbs done"
              timingFlush
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
              timingPrint "timing: head score bounds force start"
              timingFlush
              let tScore0 ← monoUsNow
              let _ := score.margin
              let _ := score.eps
              let tScore1 ← monoUsNow
              timingPrint s!"timing: head score bounds force {tScore1 - tScore0} us"
              timingFlush
      let coreStages := (← IO.getEnv "NFP_TIMING_CORE_STAGES").isSome
      let coreStagesOnly := (← IO.getEnv "NFP_TIMING_CORE_STAGES_ONLY").isSome
      if coreStages then
        timeInductionHeadCoreStages inputs
        if coreStagesOnly then
          return 0
      let breakdown := (← IO.getEnv "NFP_TIMING_BREAKDOWN").isSome
      if breakdown then
        let lnBounds ← timePureWithHeartbeat "breakdown: ln bounds" (fun () =>
          Sound.headLnBounds inputs)
        timingPrint "timing: breakdown ln bounds force start"
        timingFlush
        let tLn0 ← monoUsNow
        for q in List.finRange seq do
          for i in List.finRange dModel do
            let _ := lnBounds.1 q i
            let _ := lnBounds.2 q i
            pure ()
        let tLn1 ← monoUsNow
        timingPrint s!"timing: breakdown ln bounds force {tLn1 - tLn0} us"
        timingFlush
        let qkv ← timePureWithHeartbeat "breakdown: qkv bounds" (fun () =>
          Sound.headQKVBounds inputs lnBounds.1 lnBounds.2)
        timingPrint "timing: breakdown qkv bounds force start"
        timingFlush
        let tQkv0 ← monoUsNow
        for q in List.finRange seq do
          for d in List.finRange dHead do
            let _ := qkv.qLo q d
            let _ := qkv.qHi q d
            let _ := qkv.kLo q d
            let _ := qkv.kHi q d
            let _ := qkv.vLo q d
            let _ := qkv.vHi q d
            let _ := qkv.qAbs q d
            let _ := qkv.kAbs q d
            pure ()
        let tQkv1 ← monoUsNow
        timingPrint s!"timing: breakdown qkv bounds force {tQkv1 - tQkv0} us"
        timingFlush
        let dotAbs : Fin seq → Fin seq → Rat := fun q k =>
          Sound.Linear.dotFin dHead (fun d => qkv.qAbs q d) (fun d => qkv.kAbs k d)
        let dotAbsRowTasks :
            Array (Task { row : Array Rat // row.size = seq }) ←
          timePureWithHeartbeat "breakdown: score dotAbs rows" (fun () =>
            Array.ofFn (fun q : Fin seq =>
              Task.spawn (fun _ =>
                ⟨Array.ofFn (fun k : Fin seq => dotAbs q k), by simp⟩)))
        let dotAbsRowDefault : Task { row : Array Rat // row.size = seq } :=
          Task.spawn (fun _ => ⟨Array.ofFn (fun _ : Fin seq => (0 : Rat)), by simp⟩)
        timingPrint "timing: breakdown score dotAbs force start"
        timingFlush
        let tDot0 ← monoUsNow
        for q in List.finRange seq do
          let row := (dotAbsRowTasks.getD q.1 dotAbsRowDefault).get
          let _ := row
          pure ()
        let tDot1 ← monoUsNow
        timingPrint s!"timing: breakdown score dotAbs force {tDot1 - tDot0} us"
        timingFlush
        let masked : Fin seq → Fin seq → Prop := fun q k =>
          inputs.maskCausal = true ∧ q < k
        let scaleAbs : Rat := |inputs.scale|
        let marginAtRaw : Fin seq → Rat := fun q =>
          let row := (dotAbsRowTasks.getD q.1 dotAbsRowDefault).get
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
              let acc := Sound.Linear.foldlFin seq step (none, false)
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
              let acc := Sound.Linear.foldlFin seq step (none, false)
              match acc.1, acc.2 with
              | some unmaskedMin, true => min (scaleAbs * unmaskedMin) maskedGap
              | some unmaskedMin, false => scaleAbs * unmaskedMin
              | none, true => maskedGap
              | none, false => (0 : Rat)
          else
            (0 : Rat)
        let marginAtCached : Fin seq → Rat ←
          timePureWithHeartbeat "breakdown: score margin cache" (fun () =>
            Sound.Bounds.cacheBoundThunk marginAtRaw)
        timingPrint "timing: breakdown score margin force start"
        timingFlush
        let tMargin0 ← monoUsNow
        for q in List.finRange seq do
          let m := marginAtCached q
          forceRat m
          pure ()
        let tMargin1 ← monoUsNow
        timingPrint s!"timing: breakdown score margin force {tMargin1 - tMargin0} us"
        timingFlush
        let epsAtRaw : Fin seq → Rat := fun q =>
          let m := marginAtCached q
          if m < 0 then
            (1 : Rat)
          else
            ratDivUp (seq - 1) (1 + m)
        let epsAtCached : Fin seq → Rat ←
          timePureWithHeartbeat "breakdown: score eps cache" (fun () =>
            Sound.Bounds.cacheBoundThunk epsAtRaw)
        timingPrint "timing: breakdown score eps force start"
        timingFlush
        let tEps0 ← monoUsNow
        for q in List.finRange seq do
          let e := epsAtCached q
          forceRat e
          pure ()
        let tEps1 ← monoUsNow
        timingPrint s!"timing: breakdown score eps force {tEps1 - tEps0} us"
        timingFlush
        let valsLo ← timePureWithHeartbeat "breakdown: value valsLo" (fun () =>
          Sound.headValueValsLo inputs qkv.vLo qkv.vHi)
        timingPrint "timing: breakdown value valsLo force start"
        timingFlush
        let tValsLo0 ← monoUsNow
        for k in List.finRange seq do
          let v := valsLo k
          forceRat v
          pure ()
        let tValsLo1 ← monoUsNow
        timingPrint s!"timing: breakdown value valsLo force {tValsLo1 - tValsLo0} us"
        timingFlush
        let valsHi ← timePureWithHeartbeat "breakdown: value valsHi" (fun () =>
          Sound.headValueValsHi inputs qkv.vLo qkv.vHi)
        timingPrint "timing: breakdown value valsHi force start"
        timingFlush
        let tValsHi0 ← monoUsNow
        for k in List.finRange seq do
          let v := valsHi k
          forceRat v
          pure ()
        let tValsHi1 ← monoUsNow
        timingPrint s!"timing: breakdown value valsHi force {tValsHi1 - tValsHi0} us"
        timingFlush
        let heartbeatMsProgress ← heartbeatMs
        let taskMin (t1 t2 : Task Rat) : Task Rat :=
          Task.bind t1 (fun v1 => Task.map (fun v2 => min v1 v2) t2)
        let taskMax (t1 t2 : Task Rat) : Task Rat :=
          Task.bind t1 (fun v1 => Task.map (fun v2 => max v1 v2) t2)
        let reduceMinTasksWithProgress (tasks : Array (Task Rat)) :
            IO Rat := do
          let n := tasks.size
          if n = 0 then
            pure (0 : Rat)
          else
            let chunkSize : Nat := 16
            let chunks : Nat := (n + chunkSize - 1) / chunkSize
            let defaultTask : Task Rat := Task.pure (0 : Rat)
            let chunkTasks : Array (Task Rat) :=
              Array.ofFn (fun c : Fin chunks =>
                let start := c.val * chunkSize
                let stop := Nat.min n (start + chunkSize)
                let init := tasks.getD start defaultTask
                if stop ≤ start + 1 then
                  init
                else
                  let rest := (List.range (stop - start - 1)).map (fun i => start + i + 1)
                  rest.foldl (fun acc i => taskMin acc (tasks.getD i defaultTask)) init)
            if heartbeatMsProgress ≠ 0 then
              let mut finished := 0
              let mut remaining := chunkTasks.size
              while finished < remaining do
                IO.sleep heartbeatMsProgress
                let mut count := 0
                for t in chunkTasks do
                  if (← IO.hasFinished t) then
                    count := count + 1
                finished := count
                remaining := chunkTasks.size
                if finished < remaining then
                  timingPrint s!"timing: breakdown value lo progress {finished}/{remaining}"
                  timingFlush
            let init := chunkTasks.getD 0 defaultTask
            let rest := (List.range (chunkTasks.size - 1)).map (fun i => i + 1)
            pure ((rest.foldl (fun acc i => taskMin acc (chunkTasks.getD i defaultTask)) init).get)
        let reduceMaxTasksWithProgress (tasks : Array (Task Rat)) :
            IO Rat := do
          let n := tasks.size
          if n = 0 then
            pure (0 : Rat)
          else
            let chunkSize : Nat := 16
            let chunks : Nat := (n + chunkSize - 1) / chunkSize
            let defaultTask : Task Rat := Task.pure (0 : Rat)
            let chunkTasks : Array (Task Rat) :=
              Array.ofFn (fun c : Fin chunks =>
                let start := c.val * chunkSize
                let stop := Nat.min n (start + chunkSize)
                let init := tasks.getD start defaultTask
                if stop ≤ start + 1 then
                  init
                else
                  let rest := (List.range (stop - start - 1)).map (fun i => start + i + 1)
                  rest.foldl (fun acc i => taskMax acc (tasks.getD i defaultTask)) init)
            if heartbeatMsProgress ≠ 0 then
              let mut finished := 0
              let mut remaining := chunkTasks.size
              while finished < remaining do
                IO.sleep heartbeatMsProgress
                let mut count := 0
                for t in chunkTasks do
                  if (← IO.hasFinished t) then
                    count := count + 1
                finished := count
                remaining := chunkTasks.size
                if finished < remaining then
                  timingPrint s!"timing: breakdown value hi progress {finished}/{remaining}"
                  timingFlush
            let init := chunkTasks.getD 0 defaultTask
            let rest := (List.range (chunkTasks.size - 1)).map (fun i => i + 1)
            pure ((rest.foldl (fun acc i => taskMax acc (chunkTasks.getD i defaultTask)) init).get)
        if (← IO.getEnv "NFP_TIMING_TASK_PROGRESS").isSome then
          let tasksLo :=
            (List.finRange seq).map (fun k => Task.spawn (fun _ => valsLo k))
          let tasksHi :=
            (List.finRange seq).map (fun k => Task.spawn (fun _ => valsHi k))
          let _ ← timePureWithHeartbeat "breakdown: value lo progress" (fun () =>
            reduceMinTasksWithProgress tasksLo.toArray)
          let _ ← timePureWithHeartbeat "breakdown: value hi progress" (fun () =>
            reduceMaxTasksWithProgress tasksHi.toArray)
        else
          let loTask := Sound.headValueLoTask valsLo
          let hiTask := Sound.headValueHiTask valsHi
          let heartbeatMs ← heartbeatMs
          let tLo0 ← monoUsNow
          if heartbeatMs ≠ 0 then
            let mut finished := (← IO.hasFinished loTask)
            while !finished do
              IO.sleep heartbeatMs
              finished := (← IO.hasFinished loTask)
              if !finished then
                let now ← monoUsNow
                timingPrint s!"timing: breakdown: value lo running {now - tLo0} us"
                timingFlush
          let lo := loTask.get
          let tLo1 ← monoUsNow
          timingPrint s!"timing: breakdown: value lo {tLo1 - tLo0} us"
          timingFlush
          let tHi0 ← monoUsNow
          if heartbeatMs ≠ 0 then
            let mut finished := (← IO.hasFinished hiTask)
            while !finished do
              IO.sleep heartbeatMs
              finished := (← IO.hasFinished hiTask)
              if !finished then
                let now ← monoUsNow
                timingPrint s!"timing: breakdown: value hi running {now - tHi0} us"
                timingFlush
          let hi := hiTask.get
          let tHi1 ← monoUsNow
          timingPrint s!"timing: breakdown: value hi {tHi1 - tHi0} us"
          timingFlush
          let _ := lo
          let _ := hi
        if (← IO.getEnv "NFP_TIMING_SEQ_REDUCE").isSome then
          let loSeq ← timePureWithHeartbeat "breakdown: value lo seq" (fun () =>
            match List.finRange seq with
            | [] => (0 : Rat)
            | k :: ks =>
                let init := valsLo k
                ks.foldl (fun acc k => min acc (valsLo k)) init)
          let hiSeq ← timePureWithHeartbeat "breakdown: value hi seq" (fun () =>
            match List.finRange seq with
            | [] => (0 : Rat)
            | k :: ks =>
                let init := valsHi k
                ks.foldl (fun acc k => max acc (valsHi k)) init)
          let _ := loSeq
          let _ := hiSeq
      let tCert0 ← monoUsNow
      let certTask :
          Task
            (Option { cache : Sound.InductionHeadCoreCache seq dModel dHead //
              Sound.InductionHeadCertSound inputs cache.cert }) :=
        Task.spawn (prio := Task.Priority.dedicated) (fun _ =>
          match Sound.buildInductionCertFromHeadWithCache? cfg inputs with
          | none => none
          | some ⟨cache, hcert⟩ =>
              let _ := cache.cert.active.card
              some ⟨cache, hcert⟩)
      let heartbeatMs ← heartbeatMs
      if heartbeatMs ≠ 0 then
        let mut finished := (← IO.hasFinished certTask)
        while !finished do
          IO.sleep heartbeatMs
          finished := (← IO.hasFinished certTask)
          if !finished then
            let now ← monoUsNow
            timingPrint s!"timing: head build induction cert running {now - tCert0} us"
            timingFlush
      let certOpt ← IO.wait certTask
      let tCert1 ← monoUsNow
      logTiming s!"done: head build induction cert {tCert1 - tCert0} us"
      timingPrint s!"timing: head build induction cert {tCert1 - tCert0} us"
      timingPrint "timing: head build induction cert returned"
      timingFlush
      match certOpt with
      | none =>
          IO.eprintln "error: head inputs rejected"
          return 2
      | some ⟨cache, _hcert⟩ =>
          let cert := cache.cert
          timingPrint "timing: head active count start"
          timingFlush
          let activeCount := cert.active.card
          timingPrint "timing: head active count done"
          timingFlush
          let defaultMinActive := max 1 (seq / 8)
          let minActive := minActive?.getD defaultMinActive
          if activeCount < minActive then
            IO.eprintln
              s!"error: active queries {activeCount} below minimum {minActive}"
            return 2
          if cert.margin < minMargin then
            IO.eprintln
              s!"error: margin {ratToString cert.margin} \
              below minimum {ratToString minMargin}"
            return 2
          if maxEps < cert.eps then
            IO.eprintln
              s!"error: eps {ratToString cert.eps} \
              above maximum {ratToString maxEps}"
            return 2
          timingPrint "timing: head tol start"
          timingFlush
          let tol := cert.eps * (cert.values.hi - cert.values.lo)
          timingPrint "timing: head tol done"
          timingFlush
          let effectiveMinLogitDiff :=
            match minLogitDiff? with
            | some v => some v
            | none => some (0 : Rat)
          let logitCache := Nfp.Sound.logitDiffCache cert
          let qOnlyExit ←
            emitLogitDiffQueryOnly inputs cfg cache cert logitCache
          if qOnlyExit then
            return 2
          let emitLogitDiffDebug (info : Nfp.Sound.LogitDiffAtLoDebug seq) : IO Unit := do
            IO.eprintln
              s!"debug: logitDiffLB0 witness q={info.q.1}, prev={info.prev.1}"
            IO.eprintln
              s!"debug: eps={ratToString info.eps}, \
              valsPrevLo={ratToString info.valsPrevLo}, \
              loAt={ratToString info.loAt}, \
              lo={ratToString info.lo}"
            IO.eprintln
              s!"debug: valsPrevLoMinusLoAt={ratToString info.valsPrevLoMinusLoAt}, \
              gap={ratToString info.gap}, \
              fAtQ={ratToString (info.valsPrevLo - info.gap)}, \
              lbAtQ={ratToString info.lbAtQ}"
            let weightBoundAt := cert.weightBoundAt
            let step : (Rat × Nat × Rat) → Fin seq → (Rat × Nat × Rat) :=
              fun acc k =>
                if k = info.prev then
                  acc
                else
                  let w := weightBoundAt info.q k
                  let sum := acc.1 + w
                  let ones := if w = (1 : Rat) then acc.2.1 + 1 else acc.2.1
                  let maxW := if w > acc.2.2 then w else acc.2.2
                  (sum, ones, maxW)
            let acc := Sound.Linear.foldlFin seq step (0, 0, 0)
            IO.eprintln
              s!"debug: epsAt={ratToString (cert.epsAt info.q)}, \
              weightSum={ratToString acc.1}, ones={acc.2.1}, \
              maxWeight={ratToString acc.2.2}"
            let valsLo := logitCache.valsLo
            let stepOnes : Array String → Fin seq → Array String :=
              fun acc k =>
                if k = info.prev then
                  acc
                else
                  let w := weightBoundAt info.q k
                  if w = (1 : Rat) then
                    acc.push
                      s!"k={k.1} valsLo={ratToString (valsLo k)}"
                  else
                    acc
            let ones := Sound.Linear.foldlFin seq stepOnes #[]
            let onesMsg :=
              if ones.isEmpty then
                "none"
              else
                String.intercalate ", " ones.toList
            IO.eprintln s!"debug: weightBoundAt=1 keys: {onesMsg}"
            let stepLoAt : Array String → Fin seq → Array String :=
              fun acc k =>
                if k = info.prev then
                  acc
                else if valsLo k = info.loAt then
                  acc.push
                    s!"k={k.1} w={ratToString (weightBoundAt info.q k)}"
                else
                  acc
            let loAtKeys := Sound.Linear.foldlFin seq stepLoAt #[]
            let loAtMsg :=
              if loAtKeys.isEmpty then
                "none"
              else
                String.intercalate ", " loAtKeys.toList
            IO.eprintln s!"debug: loAt keys: {loAtMsg}"
            let scoreLoPrev := cache.scoreLoPrev info.q
            let stepAlt :
                (Rat × Nat × Nat) → Fin seq → (Rat × Nat × Nat) :=
              fun acc k =>
                if k = info.prev then
                  acc
                else
                  let g := scoreLoPrev - cache.scoreHi info.q k
                  let nonneg := if g ≥ (0 : Rat) then acc.2.1 + 1 else acc.2.1
                  let gtNegOne := if g > (-1 : Rat) then acc.2.2 + 1 else acc.2.2
                  let expLB :=
                    if g ≥ (0 : Rat) then
                      (1 : Rat) + g + g * g / (2 : Rat)
                    else
                      max (0 : Rat) ((1 : Rat) + g)
                  let w := (1 : Rat) / ((1 : Rat) + expLB)
                  (acc.1 + w, nonneg, gtNegOne)
            let accAlt := Sound.Linear.foldlFin seq stepAlt (0, 0, 0)
            IO.eprintln
              s!"debug: alt-exp epsAt={ratToString accAlt.1}, \
              g>=0={accAlt.2.1}, g>-1={accAlt.2.2}"
            let stepMin : Option Rat → Fin seq → Option Rat :=
              fun acc k =>
                if k = info.prev then
                  acc
                else
                  let g := scoreLoPrev - cache.scoreHi info.q k
                  match acc with
                  | none => some g
                  | some cur => some (min cur g)
            let minGap := Sound.Linear.foldlFin seq stepMin none
            IO.eprintln s!"debug: alt-exp min(scoreLoPrev-scoreHi)={ratOptToString minGap}"
            if (← logitDiffRefineEnabled) then
              let refineBudget := max 1 cfg.splitBudgetDiffRefined
              let refineKeys := Sound.refineKeysAtWithWeightOnes inputs cache info.q refineBudget
              IO.eprintln
                s!"debug: refine budget={refineBudget}, \
                refineKeys.card={refineKeys.card}"
              let refineSpec :=
                Sound.refineSpecForQueryWithWeightOnes inputs cache info.q refineBudget
              let refinedLB? :=
                Sound.logitDiffLowerBoundRefinedFromCache
                  inputs cache cert logitCache refineSpec
              match refinedLB? with
              | none =>
                  IO.eprintln "debug: refined logitDiffLB0 none"
              | some lb =>
                  IO.eprintln
                    s!"debug: refined logitDiffLB0={ratToString lb}"
          logTiming "start: head logit-diff lower bound"
          timingPrint "timing: head logit-diff lower bound start"
          timingFlush
          profileLogitDiffWeighted cert logitCache
          let altQuery? ← logitDiffAltBoundQuery
          match altQuery? with
          | none => pure ()
          | some qNat =>
              if hq : qNat < seq then
                let q : Fin seq := ⟨qNat, hq⟩
                let prev := cert.prev q
                let scoreLoPrev := cache.scoreLoPrev q
                let stepAlt :
                    (Rat × Nat × Nat) → Fin seq → (Rat × Nat × Nat) :=
                  fun acc k =>
                    if k = prev then
                      acc
                    else
                      let g := scoreLoPrev - cache.scoreHi q k
                      let nonneg := if g ≥ (0 : Rat) then acc.2.1 + 1 else acc.2.1
                      let gtNegOne := if g > (-1 : Rat) then acc.2.2 + 1 else acc.2.2
                      let expLB :=
                        if g ≥ (0 : Rat) then
                          (1 : Rat) + g + g * g / (2 : Rat)
                        else
                          max (0 : Rat) ((1 : Rat) + g)
                      let w := (1 : Rat) / ((1 : Rat) + expLB)
                      (acc.1 + w, nonneg, gtNegOne)
                let accAlt := Sound.Linear.foldlFin seq stepAlt (0, 0, 0)
                let stepMin : Option Rat → Fin seq → Option Rat :=
                  fun acc k =>
                    if k = prev then
                      acc
                    else
                      let g := scoreLoPrev - cache.scoreHi q k
                      match acc with
                      | none => some g
                      | some cur => some (min cur g)
                let minGap := Sound.Linear.foldlFin seq stepMin none
                IO.eprintln
                  s!"debug: alt-exp q={qNat} prev={prev.1} \
                  epsAt={ratToString accAlt.1} \
                  g>=0={accAlt.2.1} g>-1={accAlt.2.2} \
                  minGap={ratOptToString minGap}"
                if (← logitDiffDebugEarlyExitEnabled) then
                  IO.eprintln "debug: early exit requested (alt bound)"
                  return 2
              else
                IO.eprintln
                  s!"warn: NFP_LOGITDIFF_ALT_BOUND_Q={qNat} out of range (seq={seq})"
          let earlyExit? ←
            if (← logitDiffDebugEnabled) && (← logitDiffDebugEarlyExitEnabled) then
              let debug? ← timePureWithHeartbeat
                "head: logit-diff lower bound debug" (fun () =>
                  Nfp.Sound.logitDiffLowerBoundAtLoDebug cert logitCache)
              match debug? with
              | none =>
                  IO.eprintln "debug: logitDiffLB0 witness not found"
              | some ⟨info, _⟩ =>
                  emitLogitDiffDebug info
              IO.eprintln "debug: early exit requested"
              pure (some ())
            else
              pure none
          match earlyExit? with
          | some _ => return 2
          | none => pure ()
          let weightedTask? : Option (Task (Option Rat)) := none
          let logitDiffLB0? ← timePureWithHeartbeat
            "head: logit-diff lower bound unweighted" (fun () =>
              Nfp.Sound.logitDiffLowerBoundRefineOnDemand inputs cache cert logitCache)
          if (← logitDiffDebugEnabled) then
            match logitDiffLB0? with
            | some lb0 =>
                if lb0 ≤ 0 then
                  match Nfp.Sound.logitDiffLowerBoundAtLoDebug cert logitCache with
                  | none =>
                      IO.eprintln "debug: logitDiffLB0 witness not found"
                  | some ⟨info, _⟩ =>
                      emitLogitDiffDebug info
            | none => pure ()
          let needsWeighted : Bool :=
            match logitDiffLB0? with
            | none => true
            | some lb0 =>
                if lb0 ≤ 0 then
                  true
                else
                  match minLogitDiff? with
                  | some minLogitDiff => lb0 < minLogitDiff
                  | none => false
          let logitDiffWeighted? ←
            if needsWeighted then
              match weightedTask? with
              | some task =>
                  timePureWithHeartbeat
                    "head: logit-diff lower bound weighted" (fun () =>
                      task.get)
              | none =>
                  timePureWithHeartbeat
                    "head: logit-diff lower bound weighted" (fun () =>
                      Nfp.Sound.logitDiffLowerBoundWeightedFromCache cert logitCache)
            else
              pure none
          let logitDiffLB? : Option Rat :=
            match logitDiffLB0?, logitDiffWeighted? with
            | some lb0, some lb1 => some (max lb0 lb1)
            | some lb0, none => some lb0
            | none, some lb1 => some lb1
            | none, none => none
          let boundLabel : String :=
            match logitDiffLB0?, logitDiffWeighted? with
            | some _, some _ => "max"
            | none, some _ => "weighted"
            | some _, none => "eps"
            | none, none => "none"
          logTiming "done: head logit-diff lower bound"
          match logitDiffLB? with
          | none =>
              IO.eprintln "error: empty active set for logit-diff bound"
              return 2
          | some logitDiffLB =>
              if logitDiffLB ≤ 0 then
                if (← logitDiffDebugEnabled) then
                  IO.eprintln
                    s!"debug: logitDiffLB0={ratOptToString logitDiffLB0?}, \
                    logitDiffWeighted={ratOptToString logitDiffWeighted?}, \
                    logitDiffLB={ratToString logitDiffLB}, \
                    bound={boundLabel}"
                IO.eprintln
                  s!"error: logitDiffLB {ratToString logitDiffLB} \
                  is not strictly positive"
                return 2
              let violation? : Option Rat :=
                match minLogitDiff? with
                | none => none
                | some minLogitDiff =>
                    if logitDiffLB < minLogitDiff then
                      some minLogitDiff
                    else
                      none
              match violation? with
              | some minLogitDiff =>
                  IO.eprintln
                    s!"error: logitDiffLB {ratToString logitDiffLB} \
                    below minimum {ratToString minLogitDiff}"
                  return 2
              | none => pure ()
              let tol := cert.eps * (cert.values.hi - cert.values.lo)
              IO.println
                s!"ok: nonvacuous induction bound certified \
                (seq={seq}, active={activeCount}, \
                tol={ratToString tol}, logitDiffLB={ratToString logitDiffLB}, \
                bound={boundLabel})"
              return 0
/-- Build and check induction certificates from exact head inputs. -/
def runInductionCertifyHead (inputsPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String)
    (timing? : Option Nat) (heartbeatMs? : Option Nat)
    (splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined? : Option Nat) :
    IO UInt32 := do
  configureTiming timing? heartbeatMs?
  let splitCfg :=
    splitConfigFromOptions splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
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
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let parsedInputs ← timePhase "load head inputs" <|
        loadInductionHeadInputs inputsPath
      match parsedInputs with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨_seq, ⟨_dModel, ⟨_dHead, inputs⟩⟩⟩ =>
          checkInductionHeadInputs inputs splitCfg minActive? minLogitDiff? minMargin maxEps

/-- Build and check induction certificates from a model binary. -/
def runInductionCertifyHeadModel (modelPath : System.FilePath)
    (layer head dirTarget dirNegative : Nat) (period? : Option Nat)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String)
    (timing? : Option Nat) (heartbeatMs? : Option Nat)
    (splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined? : Option Nat) :
    IO UInt32 := do
  configureTiming timing? heartbeatMs?
  let splitCfg :=
    splitConfigFromOptions splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
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
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      logTiming "start: read model file"
      timingPrint "timing: read model file start"
      timingFlush
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
              checkInductionHeadInputs inputs splitCfg minActive? minLogitDiff? minMargin maxEps

/-- Heuristic logit-diff direction derived from prompt tokens. -/
def deriveDirectionFromTokens {seq : Nat} (tokens : Fin seq → Nat) :
    Except String (Nat × Nat) := do
  let tokenArr : Array Nat := Array.ofFn (fun i : Fin seq => tokens i)
  let n := tokenArr.size
  if n < 2 then
    throw "token sequence must have length at least 2"
  let lastTok := tokenArr.getD (n - 1) 0
  let prevIdx? :=
    (List.range (n - 1)).reverse.find? (fun i =>
      tokenArr.getD i lastTok = lastTok)
  let targetTok :=
    match prevIdx? with
    | some i => tokenArr.getD (i + 1) lastTok
    | none => lastTok
  let neg0 := tokenArr.getD (n - 2) lastTok
  let neg :=
    if neg0 = targetTok then
      if lastTok ≠ targetTok then
        lastTok
      else if targetTok ≠ 0 then
        0
      else
        1
    else
      neg0
  return (targetTok, neg)

/-- Build and check induction certificates from a model binary, deriving direction tokens from the
prompt sequence. -/
def runInductionCertifyHeadModelAuto (modelPath : System.FilePath)
    (layer head : Nat) (period? : Option Nat)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String)
    (timing? : Option Nat) (heartbeatMs? : Option Nat)
    (splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined? : Option Nat) :
    IO UInt32 := do
  configureTiming timing? heartbeatMs?
  let splitCfg :=
    splitConfigFromOptions splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
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
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      logTiming "start: read model file"
      timingPrint "timing: read model file start"
      timingFlush
      let data ← timePhase "read model file" <| IO.FS.readBinFile modelPath
      let headerE ← timePure "parse model header" (fun () =>
        NfptPure.parseHeader data)
      match headerE with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨header, start⟩ =>
          let tokensE ← timePure "read prompt tokens" (fun () =>
            NfptPure.readTokens data start header)
          match tokensE with
          | Except.error msg =>
              IO.eprintln s!"error: {msg}"
              return 1
          | Except.ok tokens =>
              match deriveDirectionFromTokens tokens with
              | Except.error msg =>
                  IO.eprintln s!"error: {msg}"
                  return 1
              | Except.ok ⟨dirTarget, dirNegative⟩ =>
                  IO.println
                    s!"info: direction-target={dirTarget} direction-negative={dirNegative}"
                  let inputsE ← timePure "read head inputs" (fun () =>
                    NfptPure.readInductionHeadInputs
                      data start header layer head dirTarget dirNegative period?)
                  match inputsE with
                  | Except.error msg =>
                      IO.eprintln s!"error: {msg}"
                      return 1
                  | Except.ok inputs =>
                      checkInductionHeadInputs inputs splitCfg minActive? minLogitDiff?
                        minMargin maxEps

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
