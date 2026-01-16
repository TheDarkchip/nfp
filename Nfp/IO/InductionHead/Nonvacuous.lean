-- SPDX-License-Identifier: AGPL-3.0-or-later

module

import Nfp.IO.InductionHead.Basic

/-!
IO helpers for nonvacuous induction-head certificate checks.
-/

public section

namespace Nfp

namespace IO

/-- Build and check induction certificates from exact head inputs. -/
private def checkInductionHeadInputsNonvacuous {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cfg : Sound.InductionHeadSplitConfig)
    (minActive? : Option Nat) (minLogitDiff? : Option Rat)
    (minMargin? : Option Rat) (maxEps : Rat) : IO UInt32 := do
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
          let activeCount := cert.active.card
          let defaultMinActive := max 1 (seq / 8)
          let minActive := minActive?.getD defaultMinActive
          if activeCount < minActive then
            IO.eprintln
              s!"error: active queries {activeCount} below minimum {minActive}"
            return 2
          if maxEps < cert.eps then
            IO.eprintln
              s!"error: eps {ratToString cert.eps} above maximum {ratToString maxEps}"
            return 2
          let marginViolation? : Option Rat :=
            match minMargin? with
            | none => none
            | some minMargin =>
                if cert.margin < minMargin then
                  some minMargin
                else
                  none
          match marginViolation? with
          | some minMargin =>
              IO.eprintln
                s!"error: margin {ratToString cert.margin} \
                below minimum {ratToString minMargin}"
              return 2
          | none => pure ()
          logTiming "start: head logit-diff lower bound"
          timingPrint "timing: head logit-diff lower bound start"
          timingFlush
          let logitCache := Nfp.Sound.logitDiffCache cert
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
            if (← logitDiffRefineEnabled) then
              let refineBudget := max 1 cfg.splitBudgetDiffRefined
              let refineKeys := Sound.refineKeysAtWithWeightOnes inputs cache info.q
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
          let profiling ← logitDiffProfileEnabled
          if profiling then
            profileLogitDiffWeighted cert logitCache
          else
            pure ()
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
          let weightedTask? : Option (Task (Option Rat)) :=
            if profiling then
              none
            else
              some (Task.spawn (fun _ =>
                Nfp.Sound.logitDiffLowerBoundWeightedFromCache cert logitCache))
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
                  | some ⟨info, _⟩ => emitLogitDiffDebug info
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

/-- Build and check a strictly positive induction logit-diff bound from head inputs. -/
def runInductionCertifyHeadNonvacuous (inputsPath : System.FilePath)
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
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let parsedInputs ← timePhase "load head inputs" <|
        loadInductionHeadInputs inputsPath
      match parsedInputs with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨_seq, ⟨_dModel, ⟨_dHead, inputs⟩⟩⟩ =>
          checkInductionHeadInputsNonvacuous inputs splitCfg minActive? minLogitDiff?
            minMargin? maxEps

/-- Build and check a strictly positive induction logit-diff bound from a model binary. -/
def runInductionCertifyHeadModelNonvacuous (modelPath : System.FilePath)
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
              checkInductionHeadInputsNonvacuous inputs splitCfg minActive? minLogitDiff?
                minMargin? maxEps

/-- Build and check a strictly positive induction logit-diff bound from a model binary, deriving
direction tokens from the prompt sequence. -/
def runInductionCertifyHeadModelAutoNonvacuous (modelPath : System.FilePath)
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
                      checkInductionHeadInputsNonvacuous inputs splitCfg minActive? minLogitDiff?
                        minMargin? maxEps

end IO

end Nfp
