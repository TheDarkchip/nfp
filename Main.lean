-- SPDX-License-Identifier: AGPL-3.0-or-later

import Cli
import Nfp.IO
import Nfp.Linearization
import Nfp.Sound.Cache
import Nfp.Verification
import Nfp.Sound.IO
import Std.Time.Format

/-!
# NFP CLI: Circuit Verification Command-Line Tool

This is the main entry point for the NFP circuit verification tool.

## Usage

Build the executable:
```bash
lake build nfp
```

List the available subcommands:
```bash
lake exe nfp --help
```

Example invocations (see README for full flag descriptions):
```bash
# Analyze a model and write a report to a file
lake exe nfp analyze model.nfpt --threshold 0.1 --output report.txt

# Search for induction heads with diagnostics enabled
lake exe nfp induction model.nfpt --diagnostics --diagTop 5 --adaptive

# Generate a sound-mode certificate report
lake exe nfp certify model.nfpt --eps 1e-5 --actDeriv 2

# Local (input-dependent) sound-mode certificate report
# (If the model `.nfpt` contains `EMBEDDINGS`, `--input` can be omitted.)
lake exe nfp certify model.nfpt --delta 1/100 --eps 1e-5 --actDeriv 2

# Instantiate RoPE bounds for a specific shape
lake exe nfp rope --seqLen 4 --pairs 8

# Show version
lake exe nfp --version
```
-/

open Cli Nfp

private def fmtFloat (x : Float) : String :=
  toString x

/-! ## Stdout logging -/

private structure StdoutLogCtx where
  handle : IO.FS.Handle
  pathRef : IO.Ref System.FilePath
  pendingRef : IO.Ref Bool
  timestamp : String

initialize stdoutLogCtxRef : IO.Ref (Option StdoutLogCtx) ← IO.mkRef none

private def sanitizeFileComponent (s : String) : String :=
  String.mk <|
    s.toList.map fun c =>
      if c.isAlphanum || c = '_' || c = '-' || c = '.' then c else '_'

private def timestampNowForLog : IO String := do
  let dt ← Std.Time.ZonedDateTime.now
  let dateStr := s!"{dt.toPlainDate}"
  let timeRaw := s!"{dt.toPlainTime}"
  let timeNoFrac := (timeRaw.splitOn ".").getD 0 timeRaw
  let timeStr := timeNoFrac.replace ":" "-"
  return s!"{dateStr}-{timeStr}"

private def openPendingStdoutLog : IO StdoutLogCtx := do
  let logsDir : System.FilePath := "logs"
  IO.FS.createDirAll logsDir
  let ts ← timestampNowForLog
  let path : System.FilePath := logsDir / s!"{ts}_pending.log"
  let h ← IO.FS.Handle.mk path .write
  let pathRef ← IO.mkRef path
  let pendingRef ← IO.mkRef true
  return { handle := h, pathRef := pathRef, pendingRef := pendingRef, timestamp := ts }

private def mkTeeStream (out log : IO.FS.Stream) : IO.FS.Stream :=
  { flush := do out.flush; log.flush
    read := fun n => out.read n
    write := fun b => do out.write b; log.write b
    getLine := out.getLine
    putStr := fun s => do out.putStr s; log.putStr s
    isTty := out.isTty }

private def setStdoutLogName (name : String) : IO Unit := do
  let some ctx ← stdoutLogCtxRef.get | return ()
  let pending ← ctx.pendingRef.get
  if !pending then
    return ()
  let oldPath ← ctx.pathRef.get
  let logsDir : System.FilePath := "logs"
  let safeName := sanitizeFileComponent name
  let newPath : System.FilePath := logsDir / s!"{ctx.timestamp}_{safeName}.log"
  try
    IO.FS.rename oldPath newPath
    ctx.pathRef.set newPath
    ctx.pendingRef.set false
  catch
    | _ =>
      -- If rename fails, keep the pending filename but continue.
      pure ()

private def setStdoutLogNameFromModelPath (modelPath : String) : IO Unit := do
  let p : System.FilePath := modelPath
  let stem := p.fileStem.getD (p.fileName.getD "model")
  setStdoutLogName stem

/-- Check whether a `.nfpt` file contains an `EMBEDDINGS` section before the first `LAYER`.

This is used to decide whether `nfp certify --delta ...` can default to using the model file as
its own input source (so users don't have to pass `--input model.nfpt`). -/
private def hasEmbeddingsBeforeLayers (path : System.FilePath) : IO Bool := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let line ← h.getLine
  if line.isEmpty then
    return false
  let s := line.trim
  if s = "NFP_BINARY_V1" then
    return true

  -- Header: read until blank line (text format only).
  let mut seenHeader : Bool := false
  if s.startsWith "NFP_TEXT" then
    seenHeader := true
  while true do
    let line ← h.getLine
    if line.isEmpty then
      return false
    let s := line.trim
    if !seenHeader then
      if s.startsWith "NFP_TEXT" then
        seenHeader := true
      continue
    if s.isEmpty then
      break

  -- After the header, `EMBEDDINGS` (if present) must appear before any layer payload.
  while true do
    let line ← h.getLine
    if line.isEmpty then
      return false
    let s := line.trim
    if s = "EMBEDDINGS" then
      return true
    if s.startsWith "LAYER" then
      return false
  return false

private def printHeadDiagnostics (label : String) (data : PrecomputedHeadData) : IO Unit := do
  let attnFrob : Float := Float.sqrt data.attentionFrobeniusNormSq
  let patternRecon : Float :=
    (data.softmaxJacobianOpEst / data.scaleFactor) *
      data.inputNorm * data.queryKeyAlignSchurNorm * data.valueOutputProjSchurNorm
  let valueRecon : Float := attnFrob * data.valueOutputProjNorm
  let epsRecon : Float :=
    if valueRecon < 1e-10 then Float.inf else patternRecon / valueRecon

  let patternCached := data.patternTermBoundCached
  let valueCached := data.valueTermNormCached
  let epsCached := data.faithfulnessRatioCached

  IO.println s!"  {label} L{data.layerIdx}H{data.headIdx}:"
  IO.println s!"    softmaxOpBound = {fmtFloat data.softmaxJacobianOpEst}"
  IO.println s!"    softmaxParts  = rowMaxP={fmtFloat data.softmaxRowMaxP}"
  IO.println s!"                 = rowTrace={fmtFloat data.softmaxRowTraceBound}"
  IO.println s!"                 = rowMoment={fmtFloat data.softmaxRowMomentBound}"
  IO.println s!"                 = rowGersh={fmtFloat data.softmaxRowGershBound}"
  IO.println s!"                 = rowUsed={fmtFloat data.softmaxRowBoundUsed}"
  IO.println s!"                 = fallbackRows={data.softmaxRowsFallback}"
  IO.println s!"    scaleFactor    = {fmtFloat data.scaleFactor}"
  IO.println s!"    inputNorm      = {fmtFloat data.inputNorm}"
  IO.println s!"    qkOpBoundUsed  = {fmtFloat data.queryKeyAlignSchurNorm}"
  IO.println s!"    qkFrob         = {fmtFloat data.queryKeyAlignNorm}"
  IO.println s!"    wqOpGram       = {fmtFloat data.wqOpGram}"
  IO.println s!"    wkOpGram       = {fmtFloat data.wkOpGram}"
  IO.println s!"    qkFactorGram   = {fmtFloat data.qkFactorGram}"
  let qkDenseSchurStr : String :=
    match data.diag? with
    | some diag => fmtFloat diag.qkDenseSchur.get
    | none => "n/a"
  IO.println s!"    qkCandidates   = denseSchur={qkDenseSchurStr}"
  IO.println s!"                  = denseFrob={fmtFloat data.qkDenseFrob}"
  IO.println s!"                  = denseGram={fmtFloat data.qkDenseGram}"
  IO.println s!"                  = denseBrauer={fmtFloat data.qkDenseBrauer}"
  let qkBrauerOk : String :=
    if data.qkDenseBrauer ≤ data.qkDenseGram then "true" else "false"
  IO.println s!"                  = denseBrauer≤denseGram={qkBrauerOk}"
  IO.println s!"                  = factorSchur={fmtFloat data.qkFactorSchur}"
  IO.println s!"                  = factorFrob={fmtFloat data.qkFactorFrob}"
  IO.println s!"                  = factorGram={fmtFloat data.qkFactorGram}"
  IO.println s!"    voOpBoundUsed  = {fmtFloat data.valueOutputProjSchurNorm}"
  IO.println s!"    voFrob         = {fmtFloat data.valueOutputProjNorm}"
  IO.println s!"    wvOpGram       = {fmtFloat data.wvOpGram}"
  IO.println s!"    woOpGram       = {fmtFloat data.woOpGram}"
  IO.println s!"    voFactorGram   = {fmtFloat data.voFactorGram}"
  let voDenseSchurStr : String :=
    match data.diag? with
    | some diag => fmtFloat diag.voDenseSchur.get
    | none => "n/a"
  IO.println s!"    voCandidates   = denseSchur={voDenseSchurStr}"
  IO.println s!"                  = denseFrob={fmtFloat data.voDenseFrob}"
  IO.println s!"                  = denseGram={fmtFloat data.voDenseGram}"
  IO.println s!"                  = denseBrauer={fmtFloat data.voDenseBrauer}"
  let voBrauerOk : String :=
    if data.voDenseBrauer ≤ data.voDenseGram then "true" else "false"
  IO.println s!"                  = denseBrauer≤denseGram={voBrauerOk}"
  IO.println s!"                  = factorSchur={fmtFloat data.voFactorSchur}"
  IO.println s!"                  = factorFrob={fmtFloat data.voFactorFrob}"
  IO.println s!"                  = factorGram={fmtFloat data.voFactorGram}"
  IO.println s!"    attnFrob       = {fmtFloat attnFrob}"
  IO.println s!"    patternCached  = {fmtFloat patternCached}"
  IO.println s!"    valueCached    = {fmtFloat valueCached}"
  IO.println s!"    εCached        = {fmtFloat epsCached}"
  IO.println s!"    patternRecon   = {fmtFloat patternRecon}"
  IO.println s!"    valueRecon     = {fmtFloat valueRecon}"
  IO.println s!"    εRecon         = {fmtFloat epsRecon}"
  let qkOk : String :=
    if data.queryKeyAlignSchurNorm ≤ data.queryKeyAlignNorm then "true" else "false"
  let voOk : String :=
    if data.valueOutputProjSchurNorm ≤ data.valueOutputProjNorm then "true" else "false"
  IO.println s!"    checks         = qkUsed≤qkFrob={qkOk}, voUsed≤voFrob={voOk}"
  IO.println s!"    reconDiff      = Δpattern={fmtFloat (patternRecon - patternCached)}"
  IO.println s!"                    Δvalue={fmtFloat (valueRecon - valueCached)}"
  IO.println s!"                    Δε={fmtFloat (epsRecon - epsCached)}"

/-- Run the analyze command - perform circuit analysis. -/
def runAnalyze (p : Parsed) : IO UInt32 := do
  let modelPath := p.positionalArg! "model" |>.as! String
  setStdoutLogNameFromModelPath modelPath
  let thresholdStr := p.flag? "threshold" |>.map (·.as! String) |>.getD "0.1"
  let some threshold := Nfp.parseFloat thresholdStr
    | do
      IO.eprintln s!"Error: Invalid threshold value '{thresholdStr}'"
      return 1
  let outputPath := p.flag? "output" |>.map (·.as! String)
  let verify := p.hasFlag "verify"
  let verbose := p.hasFlag "verbose"

  if verbose then
    IO.println s!"Loading model from {modelPath}..."
    IO.println s!"Threshold: {threshold}"
    if verify then
      IO.println "Mode: Verification (with empirical validation)"
    else
      IO.println "Mode: Analysis (static bounds only)"

  let loadResult ← loadModel ⟨modelPath⟩

  match loadResult with
  | .error msg =>
    IO.eprintln s!"Error loading model: {msg}"
    return 1
  | .ok model0 =>
    let model := model0.trimTrailingZeroEmbeddings
    if verbose && model.seqLen ≠ model0.seqLen then
      IO.println s!"  Trimmed trailing zero embeddings: seqLen {model0.seqLen} -> {model.seqLen}"
    if verbose then
      IO.println s!"✓ Model loaded successfully"
      IO.println s!"  Layers: {model.numLayers}"
      IO.println s!"  Sequence Length: {model.seqLen}"
      let vocabSize :=
        match model.unembedding with
        | some u => u.numCols
        | none => 0
      IO.println s!"  Embedding Vocabulary: {vocabSize}"
      IO.println s!"  Model Dimension: {model.modelDim}"
      IO.println ""
      IO.println "Running analysis..."

    let report ← if verify then
      analyzeAndVerify model modelPath threshold none
    else
      analyzeModel model modelPath threshold

    let reportStr := toString report

    match outputPath with
    | some path =>
      IO.FS.writeFile ⟨path⟩ reportStr
      IO.println s!"Report written to {path}"
    | none =>
      IO.println reportStr

    return 0

/-- Run the induction command - discover induction heads ranked by effectiveness. -/
def runInduction (p : Parsed) : IO UInt32 := do
  let modelPath := p.positionalArg! "model" |>.as! String
  setStdoutLogNameFromModelPath modelPath
  let correctOpt := p.flag? "correct" |>.map (·.as! Nat)
  let incorrectOpt := p.flag? "incorrect" |>.map (·.as! Nat)
  let thresholdStr := p.flag? "threshold" |>.map (·.as! String) |>.getD "0.0"
  let verify := p.hasFlag "verify"
  let verbose := p.hasFlag "verbose"
  let diagnostics := p.hasFlag "diagnostics"
  let adaptive := p.hasFlag "adaptive"
  let targetSlackStr := p.flag? "targetSlack" |>.map (·.as! String) |>.getD "8.0"
  let maxUpgrades := p.flag? "maxUpgrades" |>.map (·.as! Nat) |>.getD 200
  let minRelImproveStr := p.flag? "minRelImprove" |>.map (·.as! String) |>.getD "0.01"
  let krylovSteps := p.flag? "krylovSteps" |>.map (·.as! Nat) |>.getD 4
  let adaptiveScopeStr := p.flag? "adaptiveScope" |>.map (·.as! String) |>.getD "layernorm"
  let diagTop := p.flag? "diagTop" |>.map (·.as! Nat) |>.getD 5
  let some minEffect := Nfp.parseFloat thresholdStr
    | do
      IO.eprintln s!"Error: Invalid threshold value '{thresholdStr}'"
      return 1

  let (targetSlack, minRelImprove, adaptiveScope) ←
    if adaptive then
      let some targetSlack := Nfp.parseFloat targetSlackStr
        | do
          IO.eprintln s!"Error: Invalid --targetSlack '{targetSlackStr}'"
          return 1
      let some minRelImprove := Nfp.parseFloat minRelImproveStr
        | do
          IO.eprintln s!"Error: Invalid --minRelImprove '{minRelImproveStr}'"
          return 1
      let adaptiveScope? : Option Nfp.AdaptiveScope :=
        match adaptiveScopeStr.trim.toLower with
        | "layernorm" => some .layernorm
        | "all" => some .all
        | _ => none
      let some adaptiveScope := adaptiveScope?
        | do
          IO.eprintln <|
            s!"Error: Invalid --adaptiveScope '{adaptiveScopeStr}' " ++
              "(expected layernorm|all)"
          return 1
      pure (targetSlack, minRelImprove, adaptiveScope)
    else
      pure (8.0, 0.01, Nfp.AdaptiveScope.layernorm)

  IO.println "Loading model..."
  let loadResult ← loadModel ⟨modelPath⟩

  match loadResult with
  | .error msg =>
    IO.eprintln s!"Error loading model: {msg}"
    return 1
  | .ok model0 =>
    let model := model0.trimTrailingZeroEmbeddings
    if verbose && model.seqLen ≠ model0.seqLen then
      IO.println s!"  Trimmed trailing zero embeddings: seqLen {model0.seqLen} -> {model.seqLen}"
    match model.unembedding with
    | none =>
      IO.eprintln "Error: Model is missing unembedding matrix (needed for logit directions)."
      return 1
    | some W_U =>
      let target? : Option TargetDirection :=
        match correctOpt, incorrectOpt with
        | some correct, some incorrect =>
            some (TargetDirection.fromLogitDiff W_U correct incorrect)
        | some _, none | none, some _ =>
            none
        | none, none =>
            match model.inputTokens with
            | some _ => TargetDirection.fromInductionHistory model
            | none => some (TargetDirection.fromLogitDiff W_U 0 1)

      let some target := target?
        | do
          if correctOpt.isSome ∨ incorrectOpt.isSome then
            IO.eprintln "Error: Use both --correct and --incorrect (or neither to auto-detect)."
            return 1
          else
            IO.eprintln "No valid induction target could be derived from TOKENS \
              (no prior repetition of last token)."
            IO.eprintln "Hint: pass --correct/--incorrect to override, or export a prompt \
              where the last token repeats."
            return 2

      if correctOpt.isNone ∧ incorrectOpt.isNone ∧ model.inputTokens.isNone then
        IO.println "Warning: No TOKENS section found; using default target logit_diff(0-1)."
        IO.println "Hint: export with TOKENS or pass --correct/--incorrect."

      IO.println s!"Target: {target.description}"
      IO.println s!"Searching for heads with Effect > {minEffect}... (heuristic)"
      IO.println "Ranking: highest mechScore (= kComp·indScore·prevTok) first"
      IO.println "  Tie-break: Effect, kComp, δ, then lowest Error"
      IO.println "  circuitScore = Effect · mechScore"
      IO.println "  Effect = δ / (‖ln₁(X₂)‖_F · ‖u‖₂)"
      IO.println "  kComp_raw = ‖W_QK² · W_OV¹‖_F / (‖W_QK²‖_F · ‖W_OV¹‖_F)"
      IO.println "  kComp = kComp_raw - 1/√modelDim"

      let buildLayerNormBounds := diagnostics && (!adaptive)
      let (heads, cache) :=
        findHeuristicInductionHeadsWithCache model target minEffect (minInductionScore := 0.01)
          (buildLayerNormBounds := buildLayerNormBounds)
          (storeDiagnostics := diagnostics)
      let top := heads.take 50
      IO.println s!"Top Induction Head Pairs by mechScore (top {top.size} of {heads.size})"

      let needSched := adaptive && (verbose || diagnostics)
      let sched? : Option Nfp.AdaptiveSchedulerResult :=
        if needSched then
          let cfg : Nfp.AdaptiveSchedulerConfig :=
            { targetSlack := targetSlack
              maxUpgrades := maxUpgrades
              minRelImprove := minRelImprove
              krylovSteps := krylovSteps
              scope := adaptiveScope
              debugMonotone := diagnostics }
          some (Nfp.runAdaptiveScheduler cache cfg)
        else
          none

      if adaptive && verbose then
        let some sched := sched?
          | pure ()
        IO.println ""
        IO.println "ADAPTIVE SCHEDULER"
        IO.println <|
          s!"  targetSlack={fmtFloat targetSlack}  maxUpgrades={maxUpgrades} " ++
            s!"minRelImprove={fmtFloat minRelImprove}  krylovSteps={krylovSteps} " ++
            s!"scope={adaptiveScopeStr}"
        for s in sched.steps do
          IO.println <|
            match s.kind with
            | .ubTier =>
                s!"  it={s.iter}  L{s.layerIdx}: tier {s.tierFrom}->{s.tierTo}  " ++
                  s!"ub {fmtFloat s.ubBefore}->{fmtFloat s.ubAfter}  " ++
                  s!"lb≈{fmtFloat s.lb} (k={s.kTo})  " ++
                  s!"slack {fmtFloat s.slackBefore}->{fmtFloat s.slackAfter}"
            | .lbSteps =>
                s!"  it={s.iter}  L{s.layerIdx}: lb-steps {s.kFrom}->{s.kTo}  " ++
                  s!"ub {fmtFloat s.ubBefore}  " ++
                  s!"lb≈{fmtFloat s.lb}  slack {fmtFloat s.slackBefore}->{fmtFloat s.slackAfter}"

      for h in top do
        let c := h.candidate
        let mechScore := c.kComp * c.inductionScore * c.prevTokenStrength
        let circuitScore := h.effect * mechScore
        if verbose then
          IO.println <|
            s!"L{c.layer1Idx}H{c.head1Idx} -> L{c.layer2Idx}H{c.head2Idx} | " ++
              s!"Mech: {mechScore} | Circuit: {circuitScore} | " ++
              s!"Effect: {h.effect} | kComp: {c.kComp} | " ++
              s!"indScore: {c.inductionScore} | prevTok: {c.prevTokenStrength} | " ++
              s!"δ: {h.delta} | " ++
              s!"Error: {c.combinedError} | " ++
              s!"‖X₂‖_F: {h.layer2InputNorm} | ‖ln₁(X₂)‖_F: {h.layer2Ln1InputNorm} " ++
              s!"(ε₁={c.patternBound1}, ε₂={c.patternBound2})"
        else
          IO.println <|
            s!"L{c.layer1Idx}H{c.head1Idx} -> L{c.layer2Idx}H{c.head2Idx} | " ++
              s!"Mech: {mechScore} | Effect: {h.effect} | " ++
              s!"kComp: {c.kComp} | " ++
              s!"indScore: {c.inductionScore} | prevTok: {c.prevTokenStrength} | " ++
              s!"Error: {c.combinedError} | " ++
              s!"‖X₂‖_F: {h.layer2InputNorm}"

      if diagnostics then
        IO.println ""
        let diagN := min diagTop top.size
        if adaptive then
          let some sched := sched?
            | do
              IO.eprintln "Error: internal scheduler state missing."
              return 1
          IO.println ""
          IO.println "LAYER NORM DIAGNOSTICS (LOWER vs UPPER)"
          IO.println "  (lb is rigorous lower bound via Krylov steps; ub is rigorous upper bound)"
          for l in [:sched.ub.size] do
            let ub := sched.ub[l]!
            let lb := sched.lb[l]!
            let ratio : Float := if lb > 1e-12 then ub / lb else Float.inf
            let tier := sched.tier[l]!
            let k := sched.lbK.getD l 0
            IO.println <|
              s!"  L{l}: lb≈{fmtFloat lb}  ub={fmtFloat ub}  " ++
                s!"ub/lb={fmtFloat ratio}  tier={tier}  k={k}"
            let x := cache.forwardResult.getLayerInput l
            let y := cache.forwardResult.getPostAttnResidual l
            let ln1p := cache.model.ln1Params l
            let ln2p := cache.model.ln2Params l
            let ln1 := ConcreteMatrix.layerNormRowwiseOpDiag x ln1p.gamma
            let ln2 := ConcreteMatrix.layerNormRowwiseOpDiag y ln2p.gamma
            IO.println <|
              s!"      ln1: op≈{fmtFloat (ln1.gammaMaxAbs * ln1.maxInvStd)} " ++
                s!"(γmax≈{fmtFloat ln1.gammaMaxAbs}, invStdMax≈{fmtFloat ln1.maxInvStd} " ++
                s!"@r={ln1.maxInvStdRow}, varMin≈{fmtFloat ln1.minVar} @r={ln1.minVarRow})"
            IO.println <|
              s!"      ln2: op≈{fmtFloat (ln2.gammaMaxAbs * ln2.maxInvStd)} " ++
                s!"(γmax≈{fmtFloat ln2.gammaMaxAbs}, invStdMax≈{fmtFloat ln2.maxInvStd} " ++
                s!"@r={ln2.maxInvStdRow}, varMin≈{fmtFloat ln2.minVar} @r={ln2.minVarRow})"
            if hm : l < cache.model.mlps.size then
              let mlp := cache.model.mlps[l]'hm
              let y := cache.forwardResult.getPostAttnResidual l
              let ln2Bound := cache.model.ln2OpBound l y
              let (attnPart, maxHeadIdx, maxHeadContrib, maxHeadValue, maxHeadPattern) :
                  (Float × Nat × Float × Float × Float) := Id.run do
                let layerData := cache.headData.getD l #[]
                let mut a : Float := 0.0
                let mut bestIdx : Nat := 0
                let mut best : Float := 0.0
                let mut bestValue : Float := 0.0
                let mut bestPattern : Float := 0.0
                let mut idx : Nat := 0
                for d in layerData do
                  let attnFrob : Float := Float.sqrt (max 0.0 d.attentionFrobeniusNormSq)
                  let attnOpUb : Float := min attnFrob d.attentionOneInfBound
                  let valueTermUb : Float := d.ln1OpBound * (attnOpUb * d.valueOutputProjSchurNorm)
                  let inputs : Nfp.PatternTermBoundInputs := {
                    attention := d.attention
                    inputNorm := d.inputNorm
                    inputOpBound := d.inputOpBound
                    scaleFactor := d.scaleFactor
                    wqOpBound := d.wqOpGram
                    wkOpBound := d.wkOpGram
                    wvOpBound := d.wvOpGram
                    woOpBound := d.woOpGram
                    bqFrob := d.bqFrob
                    bkFrob := d.bkFrob
                    bvFrob := d.bvFrob
                  }
                  let patternTermUb : Float := d.ln1OpBound * (Nfp.computePatternTermBound inputs)
                  let contrib := valueTermUb + patternTermUb
                  a := a + contrib
                  if contrib > best then
                    bestIdx := idx
                    best := contrib
                    bestValue := valueTermUb
                    bestPattern := patternTermUb
                  idx := idx + 1
                (a, bestIdx, best, bestValue, bestPattern)
              let mlpTotal : Float := max 0.0 (ub - attnPart)
              let mlpOnly : Float :=
                if 1.0 + attnPart > 1e-12 then
                  mlpTotal / (1.0 + attnPart)
                else
                  mlpTotal
              let cross : Float := max 0.0 (mlpTotal - mlpOnly)
              IO.println <|
                s!"      contrib: attn≈{fmtFloat attnPart}  mlpOnly≈{fmtFloat mlpOnly}  " ++
                  s!"cross≈{fmtFloat cross}  mlpTotal≈{fmtFloat mlpTotal}  " ++
                  s!"(maxHead=H{maxHeadIdx}≈{fmtFloat maxHeadContrib}, " ++
                  s!"value≈{fmtFloat maxHeadValue}, pattern≈{fmtFloat maxHeadPattern})"
              let mlpJacLegacy : Float :=
                let denom := ln2Bound * (1.0 + attnPart)
                if denom > 1e-12 then
                  mlpTotal / denom
                else
                  Float.nan
              let geluDeriv := cache.forwardResult.getMlpGeluDeriv l
              let diag := computeMLPOpAbsSchurDiag mlp geluDeriv
              let chosen : Float := min diag.absSchur mlpJacLegacy
              IO.println <|
                s!"      mlpDiag(L{l}): absSchur={fmtFloat diag.absSchur}  " ++
                  s!"legacy≈{fmtFloat mlpJacLegacy}  chosen≈{fmtFloat chosen}  " ++
                  s!"dMax≈{fmtFloat diag.dMax}"
        else
          IO.println "LAYER NORM DIAGNOSTICS (PI vs rigorous)"
          IO.println "  (PI is diagnostics-only; rigorous is used for bounds)"
          for l in [:cache.layerNormBounds.size] do
            let ub := cache.layerNormBounds[l]!
            let pi := estimateAttentionLayerNormHeuristicPI cache.model cache.forwardResult l true
            let ratio : Float := if pi > 1e-12 then ub / pi else Float.inf
            IO.println s!"  L{l}: PI≈{fmtFloat pi}  ub={fmtFloat ub}  ub/PI={fmtFloat ratio}"

        -- Rectangular Gram diagnostics for MLP weights (layer 0 only).
        if h0 : 0 < cache.model.mlps.size then
          let mlp0 : ConcreteMLPLayer := cache.model.mlps[0]'h0
          let dIn := mlp0.W_in.opNormUpperBoundRectGramDiag
          let dOut := mlp0.W_out.opNormUpperBoundRectGramDiag
          let chosenMsg (d : _) : String :=
            if d.usedGram then "chosen=signedGram"
            else if d.usedAbsGram then "chosen=absGram"
            else "chosen=cheap"
          let signedGramMsg (d : _) : String :=
            if !d.signedGramEnabled then "signedGram=disabled"
            else if d.gramDim > d.maxGramDimCap then "signedGram=skipped(maxGramDim cap)"
            else if d.skippedGram then "signedGram=skipped(cost guard)"
            else if d.computedGram then "signedGram=computed"
            else "signedGram=not-attempted"
          let absGramMsg (d : _) : String :=
            if !d.computedAbsGram then "absGram=disabled"
            else if d.usedAbsGram then "absGram=chosen"
            else "absGram=computed"
          IO.println ""
          IO.println "RECT-GRAM DIAGNOSTICS (MLP layer 0 weights)"
          IO.println <|
            s!"  W_in:  usedGram={dIn.usedGram}  usedAbsGram={dIn.usedAbsGram}  " ++
              s!"computedGram={dIn.computedGram}  computedAbsGram={dIn.computedAbsGram}  " ++
              s!"skippedGram={dIn.skippedGram}"
          IO.println <|
            s!"        gramDim={dIn.gramDim}  maxGramDimCap={dIn.maxGramDimCap}  " ++
              s!"signedGramEnabled={dIn.signedGramEnabled}"
          IO.println <|
            s!"        gramCost={dIn.gramCost}  gramCostLimit={dIn.gramCostLimit}  " ++
              s!"{chosenMsg dIn}  {signedGramMsg dIn}  {absGramMsg dIn}"
          IO.println <|
            s!"        frob={fmtFloat dIn.frobBound}  oneInf={fmtFloat dIn.oneInfBound} " ++
              s!"opBound={fmtFloat dIn.opBound}"
          IO.println <|
            s!"        λ_abs_gersh={fmtFloat dIn.lambdaAbsGersh} " ++
              s!"λ_abs_brauer={fmtFloat dIn.lambdaAbsBrauer}"
          IO.println s!"        λ_gersh={fmtFloat dIn.lambdaGersh}"
          IO.println s!"        λ_brauer={fmtFloat dIn.lambdaBrauer}"
          IO.println s!"        λ_moment={fmtFloat dIn.lambdaMoment}"
          IO.println s!"        λ_used={fmtFloat dIn.lambdaUsed}"
          IO.println <|
            s!"  W_out: usedGram={dOut.usedGram}  usedAbsGram={dOut.usedAbsGram}  " ++
              s!"computedGram={dOut.computedGram}  computedAbsGram={dOut.computedAbsGram}  " ++
              s!"skippedGram={dOut.skippedGram}"
          IO.println <|
            s!"        gramDim={dOut.gramDim}  maxGramDimCap={dOut.maxGramDimCap}  " ++
              s!"signedGramEnabled={dOut.signedGramEnabled}"
          IO.println <|
            s!"        gramCost={dOut.gramCost}  gramCostLimit={dOut.gramCostLimit}  " ++
              s!"{chosenMsg dOut}  {signedGramMsg dOut}  {absGramMsg dOut}"
          IO.println <|
            s!"        frob={fmtFloat dOut.frobBound}  oneInf={fmtFloat dOut.oneInfBound} " ++
              s!"opBound={fmtFloat dOut.opBound}"
          IO.println <|
            s!"        λ_abs_gersh={fmtFloat dOut.lambdaAbsGersh} " ++
              s!"λ_abs_brauer={fmtFloat dOut.lambdaAbsBrauer}"
          IO.println s!"        λ_gersh={fmtFloat dOut.lambdaGersh}"
          IO.println s!"        λ_brauer={fmtFloat dOut.lambdaBrauer}"
          IO.println s!"        λ_moment={fmtFloat dOut.lambdaMoment}"
          IO.println s!"        λ_used={fmtFloat dOut.lambdaUsed}"
        IO.println ""
        IO.println s!"DIAGNOSTICS (ε decomposition) for top-{diagN} candidates"
        for h in (top.take diagN) do
          let c := h.candidate
          IO.println s!"Candidate L{c.layer1Idx}H{c.head1Idx} -> L{c.layer2Idx}H{c.head2Idx}"
          match cache.getHeadData c.layer1Idx c.head1Idx,
                cache.getHeadData c.layer2Idx c.head2Idx with
          | some d1, some d2 =>
              printHeadDiagnostics "Head1" d1
              printHeadDiagnostics "Head2" d2
              let ε1 := d1.faithfulnessRatioCached
              let ε2 := d2.faithfulnessRatioCached
              let combinedRecon := ε1 + ε2 + ε1 * ε2
              IO.println "  Combined:"
              IO.println s!"    ε1 = {fmtFloat ε1}  ε2 = {fmtFloat ε2}"
              IO.println s!"    combinedError = (1+ε1)(1+ε2)-1 = {fmtFloat combinedRecon}"
              IO.println s!"    combinedErrorCached = {fmtFloat c.combinedError}"
              IO.println s!"    reconDiff = {fmtFloat (combinedRecon - c.combinedError)}"
          | _, _ =>
              IO.println "  (diagnostics unavailable: missing cached head data)"

      if verify then
        IO.println ""
        IO.println "Causal Verification (Head Ablation)"
        IO.println "Metric: Δ = logit(target) - logit(top non-target) at last position"
        IO.println "Impact = Δ_base - Δ_ablated"
        IO.println ""

        let targetToken? : Option Nat :=
          match correctOpt with
          | some t => some t
          | none => inductionTargetTokenFromHistory model

        let some targetToken := targetToken?
          | do
            IO.eprintln "Error: Cannot infer induction target token (need TOKENS or --correct)."
            return 2

        match VerificationContext.build model targetToken {} with
        | .error msg =>
            IO.eprintln s!"Error: {msg}"
            return 2
        | .ok ctx =>
            let verifyTop := heads.take 10
            IO.println s!"Top-{verifyTop.size} ablation checks (ranked by mechScore):"
            IO.println s!"  target={ctx.targetToken} | neg={ctx.negativeToken}"
            IO.println s!"  Δ_base={ctx.baseDelta}"
            IO.println ""
            IO.println "Rank | Candidate | Base Δ | Ablated Δ | Impact (Logits) | RelScore | \
              Control Impact | Axioms Verified?"
            IO.println "-----|-----------|--------|----------|----------------|----------|\
              --------------|----------------"
            let fmtOpt (x : Option Float) : String :=
              match x with
              | some v => toString v
              | none => "undef"
            let mut rank : Nat := 1
            for h in verifyTop do
              let c := h.candidate
              let candHeads : Array HeadRef := #[
                { layerIdx := c.layer1Idx, headIdx := c.head1Idx },
                { layerIdx := c.layer2Idx, headIdx := c.head2Idx }
              ]
              let row := verifyCircuit ctx candHeads
              let axiomsStr :=
                if row.axioms.verified then
                  "yes"
                else
                  let reasons := String.intercalate "; " row.axioms.failures.toList
                  if reasons.isEmpty then "no" else s!"no ({reasons})"
              IO.println s!"{rank} | {row.candidateLabel} | {row.baseDelta} | \
                {fmtOpt row.ablatedDelta} | {fmtOpt row.impact} | {fmtOpt row.relScore} | \
                {fmtOpt row.controlImpact} | {axiomsStr}"
              rank := rank + 1

      return 0

/-- Run the certify command - compute conservative, exact bounds in sound mode. -/
def runCertify (p : Parsed) : IO UInt32 := do
  let modelPath := p.positionalArg! "model" |>.as! String
  setStdoutLogNameFromModelPath modelPath
  let inputPath := p.flag? "input" |>.map (·.as! String)
  let epsStr := p.flag? "eps" |>.map (·.as! String) |>.getD "1e-5"
  let actDerivStr := p.flag? "actDeriv" |>.map (·.as! String) |>.getD "2"
  let deltaFlag? := p.flag? "delta" |>.map (·.as! String)
  let deltaStr := deltaFlag?.getD "0"
  let outputPath := p.flag? "output" |>.map (·.as! String)

  let action : ExceptT String IO Nfp.Sound.ModelCert := do
    let eps ←
      match Nfp.Sound.parseRat epsStr with
      | .ok r => pure r
      | .error e => throw s!"invalid --eps '{epsStr}': {e}"
    let actDeriv ←
      match Nfp.Sound.parseRat actDerivStr with
      | .ok r => pure r
      | .error e => throw s!"invalid --actDeriv '{actDerivStr}': {e}"
    let delta ←
      match Nfp.Sound.parseRat deltaStr with
      | .ok r => pure r
      | .error e => throw s!"invalid --delta '{deltaStr}': {e}"

    /- If `--input` is omitted but `--delta` is provided, try to use `modelPath` as the input file
       (for `.nfpt` exports that embed `EMBEDDINGS` in the same file). This keeps `nfp certify`
       ergonomic without changing the default behavior when `--delta` is absent. -/
    let inputPath? : Option System.FilePath ←
      match inputPath with
      | some s => pure (some ⟨s⟩)
      | none =>
          match deltaFlag? with
          | none => pure none
          | some _ =>
              let hasEmbeddings ←
                hasEmbeddingsBeforeLayers ⟨modelPath⟩
              if hasEmbeddings then
                pure (some ⟨modelPath⟩)
              else
                throw <|
                  "local certification requested via --delta, but the model file has no \
EMBEDDINGS section before the first LAYER; pass --input <input.nfpt> (containing EMBEDDINGS) \
or omit --delta for global certification."
    let cert ← ExceptT.mk <|
      Nfp.Sound.certifyModelFile ⟨modelPath⟩ eps actDeriv
        (inputPath? := inputPath?) (inputDelta := delta)
    pure cert

  match ← action.run with
  | .error msg =>
    IO.eprintln s!"Error: {msg}"
    return 1
  | .ok cert =>
    let s := toString cert
    match outputPath with
    | some path =>
      IO.FS.writeFile ⟨path⟩ s
      IO.println s!"Report written to {path}"
    | none =>
      IO.println s
    return 0

/-- Regression test for SOUND fixed-point cache soundness and consistency.

This is intended for CI and small fixtures. It:
- builds a `.nfpc` cache (if needed),
- checks cache size matches the expected tensor stream length,
- checks the `±1`-ulp rounding envelope on up to `maxTokens` numeric tokens in the text file.
-/
def runSoundCacheCheck (p : Parsed) : IO UInt32 := do
  let modelPath := p.positionalArg! "model" |>.as! String
  let scalePow10 := p.flag? "scalePow10" |>.map (·.as! Nat) |>.getD 9
  let maxTokens := p.flag? "maxTokens" |>.map (·.as! Nat) |>.getD 0
  let modelFp : System.FilePath := ⟨modelPath⟩

  let modelHash ← Nfp.Sound.SoundCache.fnv1a64File modelFp
  let cacheFp := Nfp.Sound.SoundCache.cachePath modelFp modelHash scalePow10

  match (← Nfp.Sound.SoundCache.buildCacheFile modelFp cacheFp scalePow10) with
  | .error e =>
      IO.eprintln s!"Error: cache build failed: {e}"
      return 1
  | .ok _ =>
      let ch ← IO.FS.Handle.mk cacheFp IO.FS.Mode.read
      let hdr ← Nfp.Sound.SoundCache.readHeader ch
      if hdr.modelHash ≠ modelHash then
        IO.eprintln "Error: cache hash mismatch"
        return 1
      match (← Nfp.Sound.SoundCache.checkCacheFileSize cacheFp hdr) with
      | .error e =>
          IO.eprintln s!"Error: {e}"
          return 1
      | .ok _ =>
          match (← Nfp.Sound.SoundCache.checkTextTokenEnvelope modelFp scalePow10 maxTokens) with
          | .error e =>
              IO.eprintln s!"Error: {e}"
              return 1
          | .ok _ =>
              IO.println <|
                "OK: sound cache validated " ++
                  s!"(scalePow10={scalePow10}, maxTokens={maxTokens})"
              return 0

/-- Run the rope command - print a proof-backed RoPE operator norm certificate. -/
def runRoPE (p : Parsed) : IO UInt32 := do
  let seqLen := p.flag? "seqLen" |>.map (·.as! Nat) |>.getD 4
  let pairs := p.flag? "pairs" |>.map (·.as! Nat) |>.getD 8

  match seqLen, pairs with
  | Nat.succ n, Nat.succ m =>
      -- Instantiate the theorem at concrete sizes to ensure the report is proof-backed.
      let θ : Fin (Nat.succ n) → Fin (Nat.succ m) → ℝ := fun _ _ => 0
      have _ :
          Nfp.operatorNormBound
              (n := Fin (Nat.succ n)) (d := Nfp.RoPEDim (Fin (Nat.succ m)))
              (Nfp.ropeJacobian (pos := Fin (Nat.succ n)) (pair := Fin (Nat.succ m)) θ)
            ≤ (2 : ℝ) := by
        simpa using
          (Nfp.rope_operatorNormBound_le_two (pos := Fin (Nat.succ n)) (pair := Fin (Nat.succ m)) θ)

      IO.println "RoPE certificate (static):"
      IO.println s!"  seqLen={seqLen}, pairs={pairs}, dim={2 * pairs}"
      IO.println "  Bound: operatorNormBound(ropeJacobian θ) ≤ 2"
      IO.println "  Meaning: max row-sum of absolute weights (ℓ∞ induced upper bound)."
      IO.println "  Proof: Nfp.rope_operatorNormBound_le_two (uses |sin|,|cos| ≤ 1 from mathlib)."
      return 0
  | _, _ =>
      IO.eprintln "Error: --seqLen and --pairs must be positive."
      return 1

/-- Dump a small slice of a forward pass for cross-implementation sanity checks. -/
def runDump (p : Parsed) : IO UInt32 := do
  let modelPath := p.positionalArg! "model" |>.as! String
  setStdoutLogNameFromModelPath modelPath
  let layer := p.flag? "layer" |>.map (·.as! Nat) |>.getD 0
  let pos := p.flag? "pos" |>.map (·.as! Nat) |>.getD 0
  let take := p.flag? "take" |>.map (·.as! Nat) |>.getD 16
  let kind := p.flag? "kind" |>.map (·.as! String) |>.getD "afterLayer"

  let loadResult ← loadModel ⟨modelPath⟩
  match loadResult with
  | .error msg =>
      IO.eprintln s!"Error loading model: {msg}"
      return 1
  | .ok model0 =>
      let model := model0.trimTrailingZeroEmbeddings
      let fwd := model.runForward true

      let X : ConcreteMatrix :=
        match kind with
        | "embeddings" => model.inputEmbeddings
        | "layerInput" =>
            fwd.getLayerInput layer
        | "postAttn" =>
            fwd.getPostAttnResidual layer
        | "afterLayer" =>
            fwd.getLayerInput (layer + 1)
        | _ =>
            fwd.getLayerInput (layer + 1)

      if X.numRows = 0 || X.numCols = 0 then
        IO.eprintln s!"Error: empty matrix for kind={kind}"
        return 1
      if pos ≥ X.numRows then
        IO.eprintln s!"Error: pos={pos} out of bounds (rows={X.numRows})"
        return 1

      let n := min take X.numCols
      let mut xs : Array Float := Array.mkEmpty n
      let mut sum : Float := 0.0
      let mut sumSq : Float := 0.0
      for j in [:n] do
        let v := X.get pos j
        xs := xs.push v
        sum := sum + v
        sumSq := sumSq + v * v

      IO.println <|
        s!"DUMP kind={kind} layer={layer} pos={pos} take={n} " ++
          s!"rows={X.numRows} cols={X.numCols}"
      IO.println s!"sum={sum} sumSq={sumSq}"
      IO.println (String.intercalate " " (xs.toList.map (fun x => s!"{x}")))
      return 0

/-- The analyze subcommand. -/
def analyzeCmd : Cmd := `[Cli|
  analyze VIA runAnalyze;
  "Analyze a neural network model for circuit discovery and verification."

  FLAGS:
    t, threshold : String; "Error threshold for verification (default: 0.1)"
    o, output : String; "Write report to file instead of stdout"
    verify; "Run empirical verification (requires input in model)"
    v, verbose; "Enable verbose output"

  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The induction subcommand. -/
def inductionCmd : Cmd := `[Cli|
  induction VIA runInduction;
  "Discover induction head pairs ranked by mechScore (kComp·indScore·prevTok)."

  FLAGS:
    c, correct : Nat; "Correct token ID (manual override; requires --incorrect)"
    i, incorrect : Nat; "Incorrect token ID (manual override; requires --correct)"
    t, threshold : String; "Minimum normalized Effect threshold (default: 0.0)"
    verify; "Run causal verification via head ablation on the top-10 candidates"
    v, verbose; "Enable verbose output"
    d, diagnostics; "Print diagnostic breakdown of ε bounds (pattern/value decomposition)"
    diagTop : Nat; "How many top candidates get diagnostics (default: 5)"
    adaptive; "Enable adaptive bound scheduler (rigorous; deterministic)"
    targetSlack : String; "Stop when ub/lb ≤ targetSlack (default: 8.0)"
    maxUpgrades : Nat; "Maximum adaptive upgrades (default: 200)"
    minRelImprove : String; "Stop upgrading a layer if improvement < this fraction (default: 0.01)"
    krylovSteps : Nat; "Krylov steps for LOWER bounds only (default: 4)"
    adaptiveScope : String; "Adaptive scope: layernorm | all (default: layernorm)"

  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The certify subcommand. -/
def certifyCmd : Cmd := `[Cli|
  certify VIA runCertify;
  "SOUND mode: compute conservative bounds using exact Rat arithmetic (no Float trust)."

  FLAGS:
    input : String; "Optional input .nfpt file for local certification (must contain EMBEDDINGS)"
    delta : String; "Input ℓ∞ radius δ for local certification (default: 0; if --input is omitted, \
uses EMBEDDINGS in the model file when present)"
    eps : String; "LayerNorm epsilon (default: 1e-5)"
    actDeriv : String; "Activation derivative bound (default: 2)"
    o, output : String; "Write report to file instead of stdout"

  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The sound-cache-check subcommand (CI regression test). -/
def soundCacheCheckCmd : Cmd := `[Cli|
  sound_cache_check VIA runSoundCacheCheck;
  "Check SOUND fixed-point cache soundness (CI / small fixtures)."

  FLAGS:
    scalePow10 : Nat; "Fixed-point scale exponent p in S=10^p (default: 9)"
    maxTokens : Nat; "Check at most this many numeric tokens (0=all; default: 0)"

  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The rope subcommand. -/
def ropeCmd : Cmd := `[Cli|
  rope VIA runRoPE;
  "Static certificate for RoPE (rotary position embedding) linearization bounds."

  FLAGS:
    seqLen : Nat; "Sequence length (>0) used for instantiation (default: 4)"
    pairs : Nat; "Number of RoPE pairs (>0); dimension is 2*pairs (default: 8)"
]

/-- The dump subcommand. -/
def dumpCmd : Cmd := `[Cli|
  dump VIA runDump;
  "Dump a small forward-pass slice (for PyTorch sanity checking)."

  FLAGS:
    layer : Nat; "Layer index (default: 0)"
    pos : Nat; "Token position / row index (default: 0)"
    take : Nat; "How many columns to dump from the start (default: 16)"
    kind : String; "embeddings | layerInput | postAttn | afterLayer (default: afterLayer)"

  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The main CLI command. -/
def nfpCmd : Cmd := `[Cli|
  nfp NOOP;
  "NFP: Neural Formal Pathways verification toolkit"

  SUBCOMMANDS:
    analyzeCmd;
    inductionCmd;
    certifyCmd;
    soundCacheCheckCmd;
    ropeCmd;
    dumpCmd
]

/-- Main entry point. -/
def main (args : List String) : IO UInt32 := do
  let ctx ← openPendingStdoutLog
  stdoutLogCtxRef.set (some ctx)
  let out ← IO.getStdout
  let log := IO.FS.Stream.ofHandle ctx.handle
  let tee := mkTeeStream out log
  IO.withStdout tee <| do
    try
      if args.contains "--version" then
        setStdoutLogName "version"
        IO.println "nfp version 0.1.0"
        return (0 : UInt32)
      nfpCmd.validate args
    finally
      let pending ← ctx.pendingRef.get
      if pending then
        setStdoutLogName "no_model"
      stdoutLogCtxRef.set none
