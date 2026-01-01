-- SPDX-License-Identifier: AGPL-3.0-or-later

import Cli
import Nfp.IO
import Nfp.Linearization
import Nfp.Untrusted.SoundCacheIO
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
lake exe nfp certify model.nfpt

# Local (input-dependent) sound-mode certificate report
# (NFP_BINARY_V1 always embeds inputs; legacy text requires an EMBEDDINGS section.)
lake exe nfp certify model.nfpt --delta 1/100

# Sound per-head contribution bounds (global or local with --delta)
lake exe nfp head_bounds model.nfpt --delta 1/100

# Sound attention pattern bounds for a single head (binary only)
lake exe nfp head_pattern model.nfpt --layer 0 --head 0 --delta 1/100 --offset -1

# Sound induction head certificate (binary only)
lake exe nfp induction_cert model.nfpt --layer1 0 --head1 0 --layer2 1 --head2 0 \
  --coord 0 --delta 1/100 --offset1 -1 --offset2 0 --keyOffset2 -1 \
  --target 42 --negative 17

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
  s.map fun c =>
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

/-- Write a report to stdout or to a file if an output path is provided. -/
private def writeReport (outputPath? : Option System.FilePath) (report : String) : IO Unit := do
  match outputPath? with
  | some path =>
      IO.FS.writeFile path report
      IO.println s!"Report written to {path}"
  | none =>
      IO.println report

/-- Check whether a `.nfpt` file contains an `EMBEDDINGS` section before the first `LAYER`.

For `NFP_BINARY_V1`, embeddings are always present, so this returns true. This is used to decide
whether `nfp certify --delta ...` can default to using the model file as its own input source
(so users don't have to pass `--input model.nfpt`). -/
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
  IO.println s!"    inputOpBound   = {fmtFloat data.inputOpBound}"
  IO.println s!"    qkOpBoundUsed  = {fmtFloat data.queryKeyAlignSchurNorm}"
  IO.println s!"    qkActFrob(c)   = {fmtFloat data.qkActFrobBound}"
  IO.println s!"    kqActFrob(c)   = {fmtFloat data.kqActFrobBound}"
  IO.println s!"    qkActOp(c)     = {fmtFloat data.qkActOpBound}"
  IO.println s!"    kqActOp(c)     = {fmtFloat data.kqActOpBound}"
  let qkActOpUbStr : String :=
    match data.patternBoundParts? with
    | some parts => fmtFloat parts.qkActOpUb
    | none => "n/a"
  let kqActOpUbStr : String :=
    match data.patternBoundParts? with
    | some parts => fmtFloat parts.kqActOpUb
    | none => "n/a"
  IO.println s!"    qkActOpUbUsed  = {qkActOpUbStr}"
  IO.println s!"    kqActOpUbUsed  = {kqActOpUbStr}"
  IO.println s!"    qkActOpSource  = {data.qkActOpBoundSource}"
  IO.println s!"    kqActOpSource  = {data.kqActOpBoundSource}"
  let vOpUbStr : String :=
    match data.patternBoundParts? with
    | some parts => fmtFloat parts.vOpUb
    | none => "n/a"
  let vOpUbWOStr : String :=
    match data.patternBoundParts? with
    | some parts => fmtFloat parts.vOpUbWO
    | none => "n/a"
  IO.println s!"    vOpUbUsed      = {vOpUbStr}"
  IO.println s!"    vOpUbWOUsed    = {vOpUbWOStr}"
  IO.println s!"    qOpBoundAct    = {fmtFloat data.qOpBoundAct}"
  IO.println s!"    kOpBoundAct    = {fmtFloat data.kOpBoundAct}"
  IO.println s!"    vOpBoundAct    = {fmtFloat data.vOpBoundAct}"
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
  let candFrobStr : String :=
    match data.patternBoundParts? with
    | some parts => fmtFloat parts.candFrob
    | none => "n/a"
  let candOpStr : String :=
    match data.patternBoundParts? with
    | some parts => fmtFloat parts.candOp
    | none => "n/a"
  let candOpWOStr : String :=
    match data.patternBoundParts? with
    | some parts => fmtFloat parts.candOpWO
    | none => "n/a"
  IO.println s!"    candFrob       = {candFrobStr}"
  IO.println s!"    candOp         = {candOpStr}"
  IO.println s!"    candOpWO       = {candOpWOStr}"
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

/-! ## Analyze command helpers -/

private structure AnalyzeArgs where
  modelPath : System.FilePath
  modelPathStr : String
  threshold : Float
  outputPath? : Option System.FilePath
  verify : Bool
  verbose : Bool

private def parseAnalyzeArgs (p : Parsed) : IO (Option AnalyzeArgs) := do
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let thresholdStr := p.flag? "threshold" |>.map (·.as! String) |>.getD "0.1"
  let some threshold := Nfp.parseFloat thresholdStr
    | do
      IO.eprintln s!"Error: Invalid threshold value '{thresholdStr}'"
      return none
  let outputPath? := p.flag? "output" |>.map (·.as! String) |>.map (fun s => ⟨s⟩)
  let verify := p.hasFlag "verify"
  let verbose := p.hasFlag "verbose"
  return some {
    modelPath := ⟨modelPathStr⟩
    modelPathStr := modelPathStr
    threshold := threshold
    outputPath? := outputPath?
    verify := verify
    verbose := verbose
  }

private def runAnalyzeWithArgs (args : AnalyzeArgs) : IO UInt32 := do
  setStdoutLogNameFromModelPath args.modelPathStr
  if args.verbose then
    IO.println s!"Loading model from {args.modelPathStr}..."
    IO.println s!"Threshold: {args.threshold}"
    if args.verify then
      IO.println "Mode: Verification (with empirical validation)"
    else
      IO.println "Mode: Analysis (static bounds only)"
  let loadResult ← loadModel args.modelPath
  match loadResult with
  | .error msg =>
    IO.eprintln s!"Error loading model: {msg}"
    return 1
  | .ok model0 =>
    let model := model0.trimTrailingZeroEmbeddings
    if args.verbose && model.seqLen ≠ model0.seqLen then
      IO.println s!"  Trimmed trailing zero embeddings: seqLen {model0.seqLen} -> {model.seqLen}"
    if args.verbose then
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
    let report ← if args.verify then
      analyzeAndVerify model args.modelPathStr args.threshold none
    else
      analyzeModel model args.modelPathStr args.threshold
    writeReport args.outputPath? (toString report)
    return 0

/-- Run the analyze command - perform circuit analysis. -/
def runAnalyze (p : Parsed) : IO UInt32 := do
  let some args ← parseAnalyzeArgs p
    | return 1
  runAnalyzeWithArgs args

/-! ## Induction command helpers -/

private structure InductionArgs where
  modelPath : System.FilePath
  modelPathStr : String
  correctOpt : Option Nat
  incorrectOpt : Option Nat
  minEffect : Float
  verify : Bool
  verbose : Bool
  diagnostics : Bool
  adaptive : Bool
  targetSlack : Float
  maxUpgrades : Nat
  minRelImprove : Float
  krylovSteps : Nat
  adaptiveScope : Nfp.AdaptiveScope
  adaptiveScopeStr : String
  diagTop : Nat

private def parseInductionArgs (p : Parsed) : IO (Option InductionArgs) := do
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let correctOpt := p.flag? "correct" |>.map (·.as! Nat)
  let incorrectOpt := p.flag? "incorrect" |>.map (·.as! Nat)
  let thresholdStr := p.flag? "threshold" |>.map (·.as! String) |>.getD "0.0"
  let verify := p.hasFlag "verify"
  let verbose := p.hasFlag "verbose"
  let diagnostics := p.hasFlag "diagnostics"
  let adaptive := p.hasFlag "adaptive"
  let targetSlackStr := p.flag? "targetSlack" |>.map (·.as! String) |>.getD "8.0"
  let maxUpgrades := p.flag? "maxUpgrades" |>.map (·.as! Nat) |>.getD 120
  let minRelImproveStr := p.flag? "minRelImprove" |>.map (·.as! String) |>.getD "0.01"
  let krylovSteps := p.flag? "krylovSteps" |>.map (·.as! Nat) |>.getD 2
  let adaptiveScopeStr := p.flag? "adaptiveScope" |>.map (·.as! String) |>.getD "layernorm"
  let diagTop := p.flag? "diagTop" |>.map (·.as! Nat) |>.getD 5
  let some minEffect := Nfp.parseFloat thresholdStr
    | do
      IO.eprintln s!"Error: Invalid threshold value '{thresholdStr}'"
      return none
  let (targetSlack, minRelImprove, adaptiveScope) ←
    if adaptive then
      let some targetSlack := Nfp.parseFloat targetSlackStr
        | do
          IO.eprintln s!"Error: Invalid --targetSlack '{targetSlackStr}'"
          return none
      let some minRelImprove := Nfp.parseFloat minRelImproveStr
        | do
          IO.eprintln s!"Error: Invalid --minRelImprove '{minRelImproveStr}'"
          return none
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
          return none
      pure (targetSlack, minRelImprove, adaptiveScope)
    else
      pure (8.0, 0.01, Nfp.AdaptiveScope.layernorm)
  return some {
    modelPath := ⟨modelPathStr⟩
    modelPathStr := modelPathStr
    correctOpt := correctOpt
    incorrectOpt := incorrectOpt
    minEffect := minEffect
    verify := verify
    verbose := verbose
    diagnostics := diagnostics
    adaptive := adaptive
    targetSlack := targetSlack
    maxUpgrades := maxUpgrades
    minRelImprove := minRelImprove
    krylovSteps := krylovSteps
    adaptiveScope := adaptiveScope
    adaptiveScopeStr := adaptiveScopeStr
    diagTop := diagTop
  }

private def deriveInductionTarget
    (model : ConcreteModel)
    (W_U : ConcreteMatrix)
    (correctOpt incorrectOpt : Option Nat) : Option TargetDirection :=
  match correctOpt, incorrectOpt with
  | some correct, some incorrect =>
      some (TargetDirection.fromLogitDiff W_U correct incorrect)
  | some _, none | none, some _ =>
      none
  | none, none =>
      match model.inputTokens with
      | some _ => TargetDirection.fromInductionHistory model
      | none => some (TargetDirection.fromLogitDiff W_U 0 1)

private def printInductionSearchIntro (minEffect : Float) : IO Unit := do
  IO.println s!"Searching for heads with Effect > {minEffect}... (heuristic)"
  IO.println "Ranking: highest mechScore (= kComp·indScore·prevTok) first"
  IO.println "  Tie-break: Effect, kComp, δ, then lowest Error"
  IO.println "  circuitScore = Effect · mechScore"
  IO.println "  Effect = δ / (‖ln₁(X₂)‖_F · ‖u‖₂)"
  IO.println "  kComp_raw = ‖W_QK² · W_OV¹‖_F / (‖W_QK²‖_F · ‖W_OV¹‖_F)"
  IO.println "  kComp = kComp_raw - 1/√modelDim"

private def printInductionCandidate (verbose : Bool) (h : HeuristicInductionHead) : IO Unit := do
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
        s!"‖X₂‖_F: {h.layer2InputNorm} | " ++
        s!"‖ln₁(X₂)‖_F: {h.layer2Ln1InputNorm} " ++
        s!"(ε₁={c.patternBound1}, ε₂={c.patternBound2})"
  else
    IO.println <|
      s!"L{c.layer1Idx}H{c.head1Idx} -> L{c.layer2Idx}H{c.head2Idx} | " ++
        s!"Mech: {mechScore} | Effect: {h.effect} | " ++
        s!"kComp: {c.kComp} | " ++
        s!"indScore: {c.inductionScore} | prevTok: {c.prevTokenStrength} | " ++
        s!"Error: {c.combinedError} | " ++
        s!"‖X₂‖_F: {h.layer2InputNorm}"

private def printInductionCandidates (heads : Array HeuristicInductionHead) (verbose : Bool) :
    IO (Array HeuristicInductionHead) := do
  let top := heads.take 50
  IO.println s!"Top Induction Head Pairs by mechScore (top {top.size} of {heads.size})"
  for h in top do
    printInductionCandidate verbose h
  return top

private def buildAdaptiveScheduler
    (cache : PrecomputedCache)
    (args : InductionArgs) : Option AdaptiveSchedulerResult :=
  if args.adaptive && (args.verbose || args.diagnostics) then
    let cfg : Nfp.AdaptiveSchedulerConfig :=
      { targetSlack := args.targetSlack
        maxUpgrades := args.maxUpgrades
        minRelImprove := args.minRelImprove
        krylovSteps := args.krylovSteps
        scope := args.adaptiveScope
        debugMonotone := args.diagnostics }
    some (Nfp.runAdaptiveScheduler cache cfg none)
  else
    none

private def printAdaptiveSchedulerSteps
    (sched : AdaptiveSchedulerResult)
    (args : InductionArgs) : IO Unit := do
  IO.println ""
  IO.println "ADAPTIVE SCHEDULER"
  IO.println <|
    s!"  targetSlack={fmtFloat args.targetSlack}  maxUpgrades={args.maxUpgrades} " ++
      s!"minRelImprove={fmtFloat args.minRelImprove}  krylovSteps={args.krylovSteps} " ++
      s!"scope={args.adaptiveScopeStr}"
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

private def printInductionDiagnostics
    (cache : PrecomputedCache)
    (top : Array HeuristicInductionHead)
    (args : InductionArgs)
    (sched? : Option AdaptiveSchedulerResult) : IO (Option UInt32) := do
  IO.println ""
  let diagN := min args.diagTop top.size
  if args.adaptive then
    let some sched := sched?
      | do
        IO.eprintln "Error: internal scheduler state missing."
        return some 1
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
              qFrobBound := d.qFrobBound
              kFrobBound := d.kFrobBound
              vFrobBound := d.vFrobBound
              qOpBoundAct := d.qOpBoundAct
              kOpBoundAct := d.kOpBoundAct
              vOpBoundAct := d.vOpBoundAct
              qkActFrobBound := d.qkActFrobBound
              kqActFrobBound := d.kqActFrobBound
              qkActOpBound := d.qkActOpBound
              kqActOpBound := d.kqActOpBound
              scaleFactor := d.scaleFactor
              wqOpBound := d.wqOpGram
              wkOpBound := d.wkOpGram
              wvOpBound := d.wvOpGram
              woOpBound := d.woOpGram
              voOpBound := d.valueOutputProjSchurNorm
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
  return none

private def runInductionVerification
    (model : ConcreteModel)
    (heads : Array HeuristicInductionHead)
    (correctOpt : Option Nat) : IO (Option UInt32) := do
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
      return some 2
  match VerificationContext.build model targetToken {} with
  | .error msg =>
      IO.eprintln s!"Error: {msg}"
      return some 2
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
      return none

/-- Run the induction command - discover induction heads ranked by effectiveness. -/
def runInduction (p : Parsed) : IO UInt32 := do
  let some args ← parseInductionArgs p
    | return 1
  setStdoutLogNameFromModelPath args.modelPathStr
  IO.println "Loading model..."
  let loadResult ← loadModel args.modelPath
  match loadResult with
  | .error msg =>
    IO.eprintln s!"Error loading model: {msg}"
    return 1
  | .ok model0 =>
    let model := model0.trimTrailingZeroEmbeddings
    if args.verbose && model.seqLen ≠ model0.seqLen then
      IO.println s!"  Trimmed trailing zero embeddings: seqLen {model0.seqLen} -> {model.seqLen}"
    match model.unembedding with
    | none =>
      IO.eprintln "Error: Model is missing unembedding matrix (needed for logit directions)."
      return 1
    | some W_U =>
      let target? := deriveInductionTarget model W_U args.correctOpt args.incorrectOpt
      let some target := target?
        | do
          if args.correctOpt.isSome ∨ args.incorrectOpt.isSome then
            IO.eprintln "Error: Use both --correct and --incorrect (or neither to auto-detect)."
            return 1
          else
            IO.eprintln "No valid induction target could be derived from TOKENS \
              (no prior repetition of last token)."
            IO.eprintln "Hint: pass --correct/--incorrect to override, or export a prompt \
              where the last token repeats."
            return 2
      if args.correctOpt.isNone ∧ args.incorrectOpt.isNone ∧ model.inputTokens.isNone then
        IO.println "Warning: No TOKENS section found; using default target logit_diff(0-1)."
        IO.println "Hint: export with TOKENS or pass --correct/--incorrect."
      IO.println s!"Target: {target.description}"
      printInductionSearchIntro args.minEffect
      let buildLayerNormBounds := args.diagnostics && (!args.adaptive)
      let (heads, cache) ← Nfp.timeIt "induction:search" (fun () =>
        pure <|
          findHeuristicInductionHeadsWithCache model target args.minEffect
            (minInductionScore := 0.01)
            (buildLayerNormBounds := buildLayerNormBounds)
            (storeDiagnostics := args.diagnostics))
      let top ← printInductionCandidates heads args.verbose
      let sched? := buildAdaptiveScheduler cache args
      if args.adaptive && args.verbose then
        match sched? with
        | some sched => printAdaptiveSchedulerSteps sched args
        | none => pure ()
      if args.diagnostics then
        let err? ← printInductionDiagnostics cache top args sched?
        if let some code := err? then
          return code
      if args.verify then
        let err? ← Nfp.timeIt "induction:verify" (fun () =>
          runInductionVerification model heads args.correctOpt)
        if let some code := err? then
          return code
      return 0

/-! ## Bench command helpers -/

private inductive BenchMode
  | analyze
  | induction
  deriving Repr

private def parseBenchMode (s : String) : Option BenchMode :=
  match s.trim.toLower with
  | "analysis" => some .analyze
  | "analyze" => some .analyze
  | "induction" => some .induction
  | "induce" => some .induction
  | _ => none

private structure BenchArgs where
  modelPath : System.FilePath
  modelPathStr : String
  mode : BenchMode
  runs : Nat
  repeatCount : Nat
  threshold : Float
  minEffect : Float
  correctOpt : Option Nat
  incorrectOpt : Option Nat
  verbose : Bool
  breakdown : Bool

private def parseBenchArgs (p : Parsed) : IO (Option BenchArgs) := do
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let modeStr := p.flag? "mode" |>.map (·.as! String) |>.getD "analysis"
  let some mode := parseBenchMode modeStr
    | do
      IO.eprintln s!"Error: Invalid --mode '{modeStr}' (analysis|induction)"
      return none
  let runs := p.flag? "runs" |>.map (·.as! Nat) |>.getD 5
  let repeatCount := p.flag? "repeats" |>.map (·.as! Nat) |>.getD 1
  let thresholdStr := p.flag? "threshold" |>.map (·.as! String) |>.getD "0.1"
  let minEffectStr := p.flag? "minEffect" |>.map (·.as! String) |>.getD "0.0"
  let some threshold := Nfp.parseFloat thresholdStr
    | do
      IO.eprintln s!"Error: Invalid --threshold '{thresholdStr}'"
      return none
  let some minEffect := Nfp.parseFloat minEffectStr
    | do
      IO.eprintln s!"Error: Invalid --minEffect '{minEffectStr}'"
      return none
  let correctOpt := p.flag? "correct" |>.map (·.as! Nat)
  let incorrectOpt := p.flag? "incorrect" |>.map (·.as! Nat)
  let verbose := p.hasFlag "verbose"
  let breakdown := p.hasFlag "breakdown"
  return some {
    modelPath := ⟨modelPathStr⟩
    modelPathStr := modelPathStr
    mode := mode
    runs := runs
    repeatCount := repeatCount
    threshold := threshold
    minEffect := minEffect
    correctOpt := correctOpt
    incorrectOpt := incorrectOpt
    verbose := verbose
    breakdown := breakdown
  }

/-- Core analysis work for benchmarking (no IO). -/
private def benchAnalyzeOnce (model : ConcreteModel) (threshold : Float) : Nat × Nat :=
  let cache := PrecomputedCache.build model
  let deepCircuits := findDeepCircuitCandidatesFromCache cache
  let verifiedDeep := deepCircuits.filter (·.amplifiedError ≤ threshold)
  let inductionHeads :=
    (deepCircuits.filterMap (·.toInductionCandidate? cache)).qsort
      (·.combinedError < ·.combinedError)
  let verifiedHeads := inductionHeads.filter (·.combinedError ≤ threshold)
  (verifiedHeads.size, verifiedDeep.size)

/-- Core induction-head search work for benchmarking (no IO). -/
private def benchInductionOnce (model : ConcreteModel) (target : TargetDirection)
    (minEffect : Float) : Nat :=
  let (heads, _) :=
    findHeuristicInductionHeadsWithCache model target minEffect
      (minInductionScore := 0.01)
      (buildLayerNormBounds := false)
      (storeDiagnostics := false)
  heads.size

private def summarizeBenchTimes (label : String) (times : Array Nat) (repeatCount : Nat) :
    IO Unit := do
  let t0 := times[0]!
  let mut minT := t0
  let mut maxT := t0
  let mut sumT : Nat := 0
  for t in times do
    if t < minT then
      minT := t
    if t > maxT then
      maxT := t
    sumT := sumT + t
  let avgT := sumT / times.size
  IO.println <|
    s!"{label} runs={times.size} repeat={repeatCount} " ++
      s!"min={minT}ms avg={avgT}ms max={maxT}ms"

private def timeNs {α : Type} (action : Unit → IO α) : IO (α × Nat) := do
  let t0 ← IO.monoNanosNow
  let result ← action ()
  let t1 ← IO.monoNanosNow
  let dtNs := t1 - t0
  return (result, dtNs)

private def runBenchWithArgs (args : BenchArgs) : IO UInt32 := do
  if args.runs = 0 then
    IO.eprintln "Error: --runs must be > 0"
    return 1
  if args.repeatCount = 0 then
    IO.eprintln "Error: --repeats must be > 0"
    return 1
  setStdoutLogNameFromModelPath args.modelPathStr
  let loadResult ← loadModel args.modelPath
  let model ←
    match loadResult with
    | .error msg =>
        IO.eprintln s!"Error loading model: {msg}"
        return 1
    | .ok model0 => pure (model0.trimTrailingZeroEmbeddings)
  match args.mode with
  | .analyze =>
      let mut times : Array Nat := Array.mkEmpty args.runs
      let mut lastHeads : Nat := 0
      let mut lastCircuits : Nat := 0
      let mut cacheNsTotal : Nat := 0
      let mut deepNsTotal : Nat := 0
      let mut candNsTotal : Nat := 0
      for i in [:args.runs] do
        let t0 ← IO.monoNanosNow
        if args.breakdown then
          let mut localCacheNs : Nat := 0
          let mut localDeepNs : Nat := 0
          let mut localCandNs : Nat := 0
          for _ in [:args.repeatCount] do
            let (cache, cacheNs) ← timeNs (fun () =>
              pure <| PrecomputedCache.build model)
            let (deepCircuits, deepNs) ← timeNs (fun () =>
              pure <| findDeepCircuitCandidatesFromCache cache)
            let verifiedDeep := deepCircuits.filter (·.amplifiedError ≤ args.threshold)
            let (inductionHeads, candNs) ← timeNs (fun () =>
              pure <|
                (deepCircuits.filterMap (·.toInductionCandidate? cache)).qsort
                  (·.combinedError < ·.combinedError))
            let verifiedHeads := inductionHeads.filter (·.combinedError ≤ args.threshold)
            localCacheNs := localCacheNs + cacheNs
            localDeepNs := localDeepNs + deepNs
            localCandNs := localCandNs + candNs
            lastHeads := verifiedHeads.size
            lastCircuits := verifiedDeep.size
          cacheNsTotal := cacheNsTotal + localCacheNs
          deepNsTotal := deepNsTotal + localDeepNs
          candNsTotal := candNsTotal + localCandNs
        else
          for _ in [:args.repeatCount] do
            let (heads, circuits) := benchAnalyzeOnce model args.threshold
            lastHeads := heads
            lastCircuits := circuits
        let t1 ← IO.monoNanosNow
        let dtMs := (t1 - t0) / 1000000
        times := times.push dtMs
        if args.verbose then
          IO.println s!"run {i + 1}: {dtMs}ms heads={lastHeads} circuits={lastCircuits}"
      summarizeBenchTimes "bench:analysis" times args.repeatCount
      if args.breakdown then
        let runs := args.runs
        let repeatCount := args.repeatCount
        let cacheAvgNs := cacheNsTotal / (runs * repeatCount)
        let deepAvgNs := deepNsTotal / (runs * repeatCount)
        let candAvgNs := candNsTotal / (runs * repeatCount)
        let cacheAvgUs := cacheAvgNs / 1000
        let deepAvgUs := deepAvgNs / 1000
        let candAvgUs := candAvgNs / 1000
        IO.println <|
          s!"bench:analysis cacheAvg={cacheAvgUs}us " ++
            s!"scanAvg={deepAvgUs}us candAvg={candAvgUs}us"
      IO.println <|
        s!"bench:analysis threshold={args.threshold} heads={lastHeads} " ++
          s!"circuits={lastCircuits}"
      return 0
  | .induction =>
      let some W_U := model.unembedding
        | do
          IO.eprintln "Error: Model is missing unembedding matrix (needed for target direction)."
          return 1
      let target? := deriveInductionTarget model W_U args.correctOpt args.incorrectOpt
      let some target := target?
        | do
          IO.eprintln "Error: Use both --correct and --incorrect (or ensure TOKENS are present)."
          return 1
      let mut times : Array Nat := Array.mkEmpty args.runs
      let mut lastHeads : Nat := 0
      for i in [:args.runs] do
        let t0 ← IO.monoNanosNow
        for _ in [:args.repeatCount] do
          let heads := benchInductionOnce model target args.minEffect
          lastHeads := heads
        let t1 ← IO.monoNanosNow
        let dtMs := (t1 - t0) / 1000000
        times := times.push dtMs
        if args.verbose then
          IO.println s!"run {i + 1}: {dtMs}ms heads={lastHeads}"
      summarizeBenchTimes "bench:induction" times args.repeatCount
      IO.println s!"bench:induction minEffect={args.minEffect} heads={lastHeads}"
      return 0

/-- Run the bench command for repeatable performance measurements. -/
def runBench (p : Parsed) : IO UInt32 := do
  let some args ← parseBenchArgs p
    | return 1
  runBenchWithArgs args

/-! ## SOUND command helpers -/

private structure CertifyArgs where
  modelPath : System.FilePath
  modelPathStr : String
  inputPath? : Option System.FilePath
  soundnessBits : Nat
  partitionDepth : Nat
  deltaFlag? : Option String
  deltaStr : String
  softmaxMarginStr : String
  softmaxExpEffort : Nat
  bestMatchMargins : Bool
  targetOffset : Int
  maxSeqLen : Nat
  tightPattern : Bool
  tightPatternLayers : Nat
  perRowPatternLayers : Nat
  causalPattern : Bool
  scalePow10 : Nat
  outputPath? : Option System.FilePath

private def parseCertifyArgs (p : Parsed) : CertifyArgs :=
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let inputPath? := p.flag? "input" |>.map (·.as! String) |>.map (fun s => ⟨s⟩)
  let soundnessBits := p.flag? "soundnessBits" |>.map (·.as! Nat) |>.getD 20
  let partitionDepth := p.flag? "partitionDepth" |>.map (·.as! Nat) |>.getD 0
  let deltaFlag? := p.flag? "delta" |>.map (·.as! String)
  let deltaStr := deltaFlag?.getD "0"
  let softmaxMarginStr := p.flag? "softmaxMargin" |>.map (·.as! String) |>.getD "0"
  let softmaxExpEffort :=
    p.flag? "softmaxExpEffort" |>.map (·.as! Nat) |>.getD Nfp.Sound.defaultSoftmaxExpEffort
  let bestMatchMargins := p.flag? "bestMatchMargins" |>.isSome
  let targetOffset := p.flag? "targetOffset" |>.map (·.as! Int) |>.getD (-1)
  let maxSeqLen := p.flag? "maxSeqLen" |>.map (·.as! Nat) |>.getD 0
  let tightPattern := p.flag? "tightPattern" |>.isSome
  let tightPatternLayers := p.flag? "tightPatternLayers" |>.map (·.as! Nat) |>.getD 1
  let perRowPatternLayers := p.flag? "perRowPatternLayers" |>.map (·.as! Nat) |>.getD 0
  let causalPattern := !p.hasFlag "noncausalPattern"
  let scalePow10 := p.flag? "scalePow10" |>.map (·.as! Nat) |>.getD 9
  let outputPath? := p.flag? "output" |>.map (·.as! String) |>.map (fun s => ⟨s⟩)
  { modelPath := ⟨modelPathStr⟩
    modelPathStr := modelPathStr
    inputPath? := inputPath?
    soundnessBits := soundnessBits
    partitionDepth := partitionDepth
    deltaFlag? := deltaFlag?
    deltaStr := deltaStr
    softmaxMarginStr := softmaxMarginStr
    softmaxExpEffort := softmaxExpEffort
    bestMatchMargins := bestMatchMargins
    targetOffset := targetOffset
    maxSeqLen := maxSeqLen
    tightPattern := tightPattern
    tightPatternLayers := tightPatternLayers
    perRowPatternLayers := perRowPatternLayers
    causalPattern := causalPattern
    scalePow10 := scalePow10
    outputPath? := outputPath? }

private def runCertifyAction (args : CertifyArgs) : ExceptT String IO Nfp.Sound.ModelCert := do
  let delta ←
    match Nfp.Sound.parseRat args.deltaStr with
    | .ok r => pure r
    | .error e => throw s!"invalid --delta '{args.deltaStr}': {e}"
  let softmaxMarginLowerBound ←
    match Nfp.Sound.parseRat args.softmaxMarginStr with
    | .ok r => pure r
    | .error e => throw s!"invalid --softmaxMargin '{args.softmaxMarginStr}': {e}"
  /- If `--input` is omitted but `--delta` is provided, try to use `modelPath` as the input file
     (for `.nfpt` exports that embed `EMBEDDINGS` in the same file). This keeps `nfp certify`
     ergonomic without changing the default behavior when `--delta` is absent. -/
  let inputPath? : Option System.FilePath ←
    match args.inputPath? with
    | some path => pure (some path)
    | none =>
        match args.deltaFlag? with
        | none => pure none
        | some _ =>
            let hasEmbeddings ←
              hasEmbeddingsBeforeLayers args.modelPath
            if hasEmbeddings then
              pure (some args.modelPath)
            else
              throw <|
                "local certification requested via --delta, but the model file has no \
EMBEDDINGS section before the first LAYER (legacy text format). Pass --input <input.nfpt> \
containing EMBEDDINGS or omit --delta for global certification."
  let inputPath? ←
    if args.bestMatchMargins && inputPath?.isNone then
      let hasEmbeddings ← hasEmbeddingsBeforeLayers args.modelPath
      if hasEmbeddings then
        pure (some args.modelPath)
      else
        throw <|
          "best-match margin tightening requires local input with EMBEDDINGS. \
Pass --input <input.nfpt> or use a model file that embeds EMBEDDINGS."
    else
      pure inputPath?
  if args.bestMatchMargins && softmaxMarginLowerBound != 0 then
    throw "best-match margin tightening is incompatible with --softmaxMargin"
  if args.bestMatchMargins then
    let cert ← ExceptT.mk <|
      Nfp.Sound.certifyModelFileBestMatchMargins args.modelPath args.soundnessBits
        (inputPath? := inputPath?) (inputDelta := delta) (partitionDepth := args.partitionDepth)
        (targetOffset := args.targetOffset) (maxSeqLen := args.maxSeqLen)
        (tightPattern := args.tightPattern) (tightPatternLayers := args.tightPatternLayers)
        (perRowPatternLayers := args.perRowPatternLayers) (scalePow10 := args.scalePow10)
        (softmaxExpEffort := args.softmaxExpEffort) (causalPattern := args.causalPattern)
    return cert
  else
    let cert ← ExceptT.mk <|
      Nfp.Sound.certifyModelFile args.modelPath args.soundnessBits
        (inputPath? := inputPath?) (inputDelta := delta) (partitionDepth := args.partitionDepth)
        (softmaxMarginLowerBound := softmaxMarginLowerBound)
        (softmaxExpEffort := args.softmaxExpEffort)
    return cert

private structure HeadBoundsArgs where
  modelPath : System.FilePath
  inputPath? : Option System.FilePath
  deltaFlag? : Option String
  deltaStr : String
  soundnessBits : Nat
  scalePow10 : Nat
  outputPath? : Option System.FilePath

private def parseHeadBoundsArgs (p : Parsed) : HeadBoundsArgs :=
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let inputPath? := p.flag? "input" |>.map (·.as! String) |>.map (fun s => ⟨s⟩)
  let deltaFlag? := p.flag? "delta" |>.map (·.as! String)
  let deltaStr := deltaFlag?.getD "0"
  let soundnessBits := p.flag? "soundnessBits" |>.map (·.as! Nat) |>.getD 20
  let scalePow10 := p.flag? "scalePow10" |>.map (·.as! Nat) |>.getD 9
  let outputPath? := p.flag? "output" |>.map (·.as! String) |>.map (fun s => ⟨s⟩)
  { modelPath := ⟨modelPathStr⟩
    inputPath? := inputPath?
    deltaFlag? := deltaFlag?
    deltaStr := deltaStr
    soundnessBits := soundnessBits
    scalePow10 := scalePow10
    outputPath? := outputPath? }

private def formatHeadBoundsLocal
    (heads : Array Nfp.Sound.HeadLocalContributionCert)
    (delta : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath)
    (modelPath : System.FilePath) : String :=
  let header :=
    "SOUND per-head bounds (local): " ++
      s!"delta={delta}, soundnessBits={soundnessBits}, input={inputPath?.getD modelPath}\n"
  let body :=
    heads.foldl (fun acc h =>
      acc ++
        s!"Layer {h.layerIdx} Head {h.headIdx}: " ++
          s!"ln1MaxAbsGamma={h.ln1MaxAbsGamma}, " ++
          s!"ln1VarLB={h.ln1VarianceLowerBound}, " ++
          s!"ln1Bound={h.ln1Bound}, " ++
          s!"wqOp={h.wqOpBound}, wkOp={h.wkOpBound}, " ++
          s!"qk={h.qkFactorBound}, " ++
          s!"softmaxJacobianNormInfUB={h.softmaxJacobianNormInfUpperBound}, " ++
          s!"wvOp={h.wvOpBound}, woOp={h.woOpBound}, " ++
          s!"attn={h.attnJacBound}\n") ""
  header ++ body

private def formatHeadBoundsGlobal
    (heads : Array Nfp.Sound.HeadContributionCert)
    (scalePow10 : Nat) : String :=
  let header :=
    s!"SOUND per-head bounds (weight-only): scalePow10={scalePow10}\n"
  let body :=
    heads.foldl (fun acc h =>
      acc ++
        s!"Layer {h.layerIdx} Head {h.headIdx}: " ++
          s!"wqOp={h.wqOpBound}, wkOp={h.wkOpBound}, " ++
          s!"wvOp={h.wvOpBound}, woOp={h.woOpBound}, " ++
          s!"qk={h.qkFactorBound}, vo={h.voFactorBound}\n") ""
  header ++ body

private def runHeadBoundsAction (args : HeadBoundsArgs) : ExceptT String IO String := do
  let delta ←
    match Nfp.Sound.parseRat args.deltaStr with
    | .ok r => pure r
    | .error e => throw s!"invalid --delta '{args.deltaStr}': {e}"
  let useLocal := (args.inputPath?.isSome || args.deltaFlag?.isSome)
  if useLocal then
    let inputPath? : Option System.FilePath ←
      match args.inputPath? with
      | some path => pure (some path)
      | none =>
          let hasEmbeddings ←
            hasEmbeddingsBeforeLayers args.modelPath
          if hasEmbeddings then
            pure (some args.modelPath)
          else
            throw <|
              "local head bounds requested via --delta, but the model file has no \
EMBEDDINGS section before the first LAYER (legacy text format). Pass --input <input.nfpt> \
containing EMBEDDINGS or omit --delta for global head bounds."
    let heads ← ExceptT.mk <|
      Nfp.Sound.certifyHeadBoundsLocal args.modelPath
        (inputPath? := inputPath?) (inputDelta := delta) (soundnessBits := args.soundnessBits)
    return formatHeadBoundsLocal heads delta args.soundnessBits inputPath? args.modelPath
  else
    let heads ← ExceptT.mk <|
      Nfp.Sound.certifyHeadBounds args.modelPath (scalePow10 := args.scalePow10)
    return formatHeadBoundsGlobal heads args.scalePow10

private structure HeadPatternArgs where
  modelPath : System.FilePath
  layerIdx : Nat
  headIdx : Nat
  offset : Int
  keyOffset : Int
  soundnessBits : Nat
  softmaxExpEffort : Nat
  tightPatternLayers : Nat
  tightPattern : Bool
  perRowPatternLayers : Nat
  causalPattern : Bool
  bestMatch : Bool
  useAffine : Bool
  sweep : Bool
  queryPos? : Option Nat
  inputPath? : Option System.FilePath
  deltaStr : String
  maxSeqLen : Nat
  outputPath? : Option System.FilePath

private def parseHeadPatternArgs (p : Parsed) : HeadPatternArgs :=
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let layerIdx := p.flag? "layer" |>.map (·.as! Nat) |>.getD 0
  let headIdx := p.flag? "head" |>.map (·.as! Nat) |>.getD 0
  let offset := p.flag? "offset" |>.map (·.as! Int) |>.getD (-1)
  let keyOffset := p.flag? "keyOffset" |>.map (·.as! Int) |>.getD 0
  let soundnessBits := p.flag? "soundnessBits" |>.map (·.as! Nat) |>.getD 20
  let softmaxExpEffort :=
    p.flag? "softmaxExpEffort" |>.map (·.as! Nat)
      |>.getD Nfp.Sound.defaultSoftmaxExpEffort
  let tightPatternLayers? := p.flag? "tightPatternLayers" |>.map (·.as! Nat)
  let tightPatternLayers := tightPatternLayers?.getD 1
  let tightPattern := p.hasFlag "tightPattern" || tightPatternLayers?.isSome
  let perRowPatternLayers := p.flag? "perRowPatternLayers" |>.map (·.as! Nat) |>.getD 0
  let causalPattern := !p.hasFlag "noncausalPattern"
  let bestMatch := p.hasFlag "bestMatch"
  let useAffine := p.hasFlag "affine"
  let sweep := p.hasFlag "sweep"
  let queryPos? := p.flag? "queryPos" |>.map (·.as! Nat)
  let inputPath? := p.flag? "input" |>.map (·.as! String) |>.map (fun s => ⟨s⟩)
  let deltaStr := p.flag? "delta" |>.map (·.as! String) |>.getD "0"
  let maxSeqLen := p.flag? "maxSeqLen" |>.map (·.as! Nat) |>.getD 256
  let outputPath? := p.flag? "output" |>.map (·.as! String) |>.map (fun s => ⟨s⟩)
  { modelPath := ⟨modelPathStr⟩
    layerIdx := layerIdx
    headIdx := headIdx
    offset := offset
    keyOffset := keyOffset
    soundnessBits := soundnessBits
    softmaxExpEffort := softmaxExpEffort
    tightPatternLayers := tightPatternLayers
    tightPattern := tightPattern
    perRowPatternLayers := perRowPatternLayers
    causalPattern := causalPattern
    bestMatch := bestMatch
    useAffine := useAffine
    sweep := sweep
    queryPos? := queryPos?
    inputPath? := inputPath?
    deltaStr := deltaStr
    maxSeqLen := maxSeqLen
    outputPath? := outputPath? }

private def formatHeadPatternBestMatchSweep
    (layerIdx headIdx : Nat)
    (offset : Int)
    (keyOffset : Int)
    (certs : Array Nfp.Sound.HeadBestMatchPatternCert) : String :=
  let header :=
    "SOUND head pattern sweep (best-match): " ++
      s!"layer={layerIdx}, head={headIdx}, offset={offset}, keyOffset={keyOffset}\n"
  let body :=
    certs.foldl (fun acc cert =>
      acc ++
        s!"queryPos={cert.queryPos} targetTok={cert.targetToken} " ++
          s!"marginLB={cert.marginLowerBound} " ++
          s!"weightLB={cert.bestMatchWeightLowerBound}\n") ""
  header ++ body

private def formatHeadPatternBestMatch
    (cert : Nfp.Sound.HeadBestMatchPatternCert) : String :=
  "SOUND head pattern (best-match): " ++
    s!"layer={cert.layerIdx}, head={cert.headIdx}, " ++
      s!"offset={cert.targetOffset}, keyOffset={cert.keyOffset}, " ++
      s!"queryPos={cert.queryPos}\n" ++
    s!"seqLen={cert.seqLen}, targetTok={cert.targetToken}, " ++
      s!"bestMatchLogitLB={cert.bestMatchLogitLowerBound}, " ++
      s!"bestNonmatchLogitUB={cert.bestNonmatchLogitUpperBound}\n" ++
    s!"marginLB={cert.marginLowerBound}, " ++
      s!"bestMatchWeightLB={cert.bestMatchWeightLowerBound}, " ++
      s!"softmaxExpEffort={cert.softmaxExpEffort}\n"

private def formatHeadPatternLocal
    (cert : Nfp.Sound.HeadPatternCert) : String :=
  "SOUND head pattern (local): " ++
    s!"layer={cert.layerIdx}, head={cert.headIdx}, " ++
      s!"offset={cert.targetOffset}, keyOffset={cert.keyOffset}\n" ++
    s!"seqLen={cert.seqLen}, " ++
    s!"targetCountLB={cert.targetCountLowerBound}, " ++
    s!"targetLogitLB={cert.targetLogitLowerBound}, " ++
    s!"otherLogitUB={cert.otherLogitUpperBound}\n" ++
    s!"marginLB={cert.marginLowerBound}, " ++
    s!"targetWeightLB={cert.targetWeightLowerBound}, " ++
    s!"softmaxExpEffort={cert.softmaxExpEffort}\n"

private def runHeadPatternAction (args : HeadPatternArgs) : ExceptT String IO String := do
  let delta ←
    match Nfp.Sound.parseRat args.deltaStr with
    | .ok r => pure r
    | .error e => throw s!"invalid --delta '{args.deltaStr}': {e}"
  let inputPath? : Option System.FilePath ←
    match args.inputPath? with
    | some path => pure (some path)
    | none =>
        let hasEmbeddings ←
          hasEmbeddingsBeforeLayers args.modelPath
        if hasEmbeddings then
          pure (some args.modelPath)
        else
          throw <|
            "head pattern bounds require EMBEDDINGS; pass --input for legacy text models."
  if args.useAffine && !args.bestMatch then
    throw "affine bounds are only supported with --bestMatch"
  if args.useAffine && args.sweep then
    throw "affine sweep is unsupported; use --bestMatch without --sweep"
  if args.bestMatch then
    if args.sweep then
      let certs ← ExceptT.mk <|
        Nfp.Sound.certifyHeadPatternBestMatchLocalSweep args.modelPath args.layerIdx args.headIdx
          (inputPath? := inputPath?) (inputDelta := delta)
          (soundnessBits := args.soundnessBits) (targetOffset := args.offset)
          (keyOffset := args.keyOffset)
          (maxSeqLen := args.maxSeqLen)
          (tightPattern := args.tightPattern)
          (tightPatternLayers := args.tightPatternLayers)
          (perRowPatternLayers := args.perRowPatternLayers)
          (useAffine := args.useAffine)
          (softmaxExpEffort := args.softmaxExpEffort)
          (causalPattern := args.causalPattern)
      return formatHeadPatternBestMatchSweep args.layerIdx args.headIdx args.offset
        args.keyOffset certs
    else
      let cert ← ExceptT.mk <|
        Nfp.Sound.certifyHeadPatternBestMatchLocal args.modelPath args.layerIdx args.headIdx
          (queryPos? := args.queryPos?) (inputPath? := inputPath?)
          (inputDelta := delta) (soundnessBits := args.soundnessBits)
          (targetOffset := args.offset) (keyOffset := args.keyOffset)
          (maxSeqLen := args.maxSeqLen)
          (tightPattern := args.tightPattern) (tightPatternLayers := args.tightPatternLayers)
          (perRowPatternLayers := args.perRowPatternLayers)
          (useAffine := args.useAffine)
          (softmaxExpEffort := args.softmaxExpEffort)
          (causalPattern := args.causalPattern)
      return formatHeadPatternBestMatch cert
  else
    if args.sweep then
      throw "use --sweep with --bestMatch"
    let cert ← ExceptT.mk <|
      Nfp.Sound.certifyHeadPatternLocal args.modelPath args.layerIdx args.headIdx
        (inputPath? := inputPath?) (inputDelta := delta)
        (soundnessBits := args.soundnessBits) (targetOffset := args.offset)
        (keyOffset := args.keyOffset)
        (maxSeqLen := args.maxSeqLen)
        (tightPattern := args.tightPattern) (tightPatternLayers := args.tightPatternLayers)
        (perRowPatternLayers := args.perRowPatternLayers)
        (softmaxExpEffort := args.softmaxExpEffort)
        (causalPattern := args.causalPattern)
    return formatHeadPatternLocal cert

/-- Run the certify command - compute conservative, exact bounds in sound mode. -/
def runCertify (p : Parsed) : IO UInt32 := do
  let args := parseCertifyArgs p
  setStdoutLogNameFromModelPath args.modelPathStr
  match ← (runCertifyAction args).run with
  | .error msg =>
    IO.eprintln s!"Error: {msg}"
    return 1
  | .ok cert =>
    writeReport args.outputPath? (toString cert)
    return 0

/-- Run the head-bounds command - compute per-head contribution bounds in sound mode. -/
def runHeadBounds (p : Parsed) : IO UInt32 := do
  let args := parseHeadBoundsArgs p
  match ← (runHeadBoundsAction args).run with
  | .error msg =>
    IO.eprintln s!"Error: {msg}"
    return 1
  | .ok s =>
    writeReport args.outputPath? s
    return 0

/-- Run the head-pattern command - compute per-head attention pattern bounds in sound mode. -/
def runHeadPattern (p : Parsed) : IO UInt32 := do
  let args := parseHeadPatternArgs p
  match ← (runHeadPatternAction args).run with
  | .error msg =>
    IO.eprintln s!"Error: {msg}"
    return 1
  | .ok s =>
    writeReport args.outputPath? s
    return 0

/-- Parsed arguments for `induction-cert`, with resolved input path and delta. -/
private structure InductionCertArgs where
  modelPath : System.FilePath
  layer1 : Nat
  head1 : Nat
  layer2 : Nat
  head2 : Nat
  coord : Nat
  offset1 : Int
  offset2 : Int
  keyOffset1 : Int
  keyOffset2 : Int
  targetToken? : Option Nat
  negativeToken? : Option Nat
  soundnessBits : Nat
  softmaxExpEffort : Nat
  tightPatternLayers : Nat
  tightPattern : Bool
  perRowPatternLayers : Nat
  iterTighten : Bool
  causalPattern : Bool
  bestMatch : Bool
  useAffine : Bool
  queryPos? : Option Nat
  inputPath? : Option System.FilePath
  delta : Rat
  maxSeqLen : Nat
  scalePow10 : Nat
  outputPath? : Option System.FilePath

/-- Parse and validate `induction-cert` arguments. -/
private def parseInductionCertArgs (p : Parsed) : ExceptT String IO InductionCertArgs := do
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let modelPath : System.FilePath := ⟨modelPathStr⟩
  let layer1 := p.flag? "layer1" |>.map (·.as! Nat) |>.getD 0
  let head1 := p.flag? "head1" |>.map (·.as! Nat) |>.getD 0
  let layer2 := p.flag? "layer2" |>.map (·.as! Nat) |>.getD 1
  let head2 := p.flag? "head2" |>.map (·.as! Nat) |>.getD 0
  let coord := p.flag? "coord" |>.map (·.as! Nat) |>.getD 0
  let offset1 := p.flag? "offset1" |>.map (·.as! Int) |>.getD (-1)
  let offset2 := p.flag? "offset2" |>.map (·.as! Int) |>.getD (-1)
  let keyOffset1 := p.flag? "keyOffset1" |>.map (·.as! Int) |>.getD 0
  let keyOffset2 := p.flag? "keyOffset2" |>.map (·.as! Int) |>.getD 0
  let targetToken := p.flag? "target" |>.map (·.as! Nat)
  let negativeToken := p.flag? "negative" |>.map (·.as! Nat)
  let soundnessBits := p.flag? "soundnessBits" |>.map (·.as! Nat) |>.getD 20
  let softmaxExpEffort :=
    p.flag? "softmaxExpEffort" |>.map (·.as! Nat)
      |>.getD Nfp.Sound.defaultSoftmaxExpEffort
  let tightPatternLayers? := p.flag? "tightPatternLayers" |>.map (·.as! Nat)
  let tightPatternLayers := tightPatternLayers?.getD 1
  let tightPattern := p.hasFlag "tightPattern" || tightPatternLayers?.isSome
  let perRowPatternLayers := p.flag? "perRowPatternLayers" |>.map (·.as! Nat) |>.getD 0
  let iterTighten := p.hasFlag "iterTighten"
  let causalPattern := !p.hasFlag "noncausalPattern"
  let bestMatch := p.hasFlag "bestMatch"
  let useAffine := p.hasFlag "affine"
  let queryPos := p.flag? "queryPos" |>.map (·.as! Nat)
  let inputPath := p.flag? "input" |>.map (·.as! String)
  let deltaStr := p.flag? "delta" |>.map (·.as! String) |>.getD "0"
  let maxSeqLen := p.flag? "maxSeqLen" |>.map (·.as! Nat) |>.getD 256
  let scalePow10 := p.flag? "scalePow10" |>.map (·.as! Nat) |>.getD 9
  let outputPath := p.flag? "output" |>.map (·.as! String)
  let delta ←
    match Nfp.Sound.parseRat deltaStr with
    | .ok r => pure r
    | .error e => throw s!"invalid --delta '{deltaStr}': {e}"
  let inputPath? : Option System.FilePath ←
    match inputPath with
    | some s => pure (some ⟨s⟩)
    | none =>
        let hasEmbeddings ←
          hasEmbeddingsBeforeLayers modelPath
        if hasEmbeddings then
          pure (some modelPath)
        else
          throw <|
            "induction cert requires EMBEDDINGS; pass --input for legacy text models."
  let outputPath? : Option System.FilePath :=
    outputPath.map (fun s => ⟨s⟩)
  return {
    modelPath := modelPath
    layer1 := layer1
    head1 := head1
    layer2 := layer2
    head2 := head2
    coord := coord
    offset1 := offset1
    offset2 := offset2
    keyOffset1 := keyOffset1
    keyOffset2 := keyOffset2
    targetToken? := targetToken
    negativeToken? := negativeToken
    soundnessBits := soundnessBits
    softmaxExpEffort := softmaxExpEffort
    tightPatternLayers := tightPatternLayers
    tightPattern := tightPattern
    perRowPatternLayers := perRowPatternLayers
    iterTighten := iterTighten
    causalPattern := causalPattern
    bestMatch := bestMatch
    useAffine := useAffine
    queryPos? := queryPos
    inputPath? := inputPath?
    delta := delta
    maxSeqLen := maxSeqLen
    scalePow10 := scalePow10
    outputPath? := outputPath?
  }

/-- Format the optional logit-diff line for local induction certs. -/
private def formatInductionLogitLine
    (logit? : Option Nfp.Sound.HeadLogitDiffLowerBoundCert) : String :=
  match logit? with
  | none => ""
  | some logit =>
      s!"logitDiffLB={logit.logitDiffLowerBound} " ++
        s!"targetTok={logit.targetToken} negTok={logit.negativeToken}\n" ++
        s!"logitMatchLB={logit.matchLogitLowerBound} " ++
        s!"logitNonmatchLB={logit.nonmatchLogitLowerBound}\n"

/-- Format the optional logit-diff line for best-match induction certs. -/
private def formatInductionLogitLinePos
    (logit? : Option Nfp.Sound.HeadLogitDiffLowerBoundPosCert) : String :=
  match logit? with
  | none => ""
  | some logit =>
      s!"logitDiffLB={logit.logitDiffLowerBound} " ++
        s!"targetTok={logit.targetToken} negTok={logit.negativeToken}\n" ++
        s!"logitMatchLB={logit.matchLogitLowerBound} " ++
        s!"logitNonmatchLB={logit.nonmatchLogitLowerBound}\n"

/-- Render a best-match induction certificate report. -/
private def formatInductionBestMatch
    (cert : Nfp.Sound.InductionHeadBestMatchSoundCert) : String :=
  let p1 := cert.layer1Pattern
  let p2 := cert.layer2Pattern
  let v := cert.layer2Value
  let logitLine := formatInductionLogitLinePos cert.layer2Logit?
  "SOUND induction cert (best-match):\n" ++
    s!"queryPos={p2.queryPos}\n" ++
    s!"layer1=L{p1.layerIdx} H{p1.headIdx} offset={p1.targetOffset} " ++
      s!"keyOffset={p1.keyOffset} " ++
      s!"targetTok={p1.targetToken} " ++
      s!"marginLB={p1.marginLowerBound} " ++
      s!"weightLB={p1.bestMatchWeightLowerBound} " ++
      s!"softmaxExpEffort={p1.softmaxExpEffort}\n" ++
    s!"layer2=L{p2.layerIdx} H{p2.headIdx} offset={p2.targetOffset} " ++
      s!"keyOffset={p2.keyOffset} " ++
      s!"targetTok={p2.targetToken} " ++
      s!"marginLB={p2.marginLowerBound} " ++
      s!"weightLB={p2.bestMatchWeightLowerBound} " ++
      s!"softmaxExpEffort={p2.softmaxExpEffort}\n" ++
    s!"coord={v.coord} matchCoordLB={v.matchCoordLowerBound} " ++
      s!"nonmatchCoordLB={v.nonmatchCoordLowerBound}\n" ++
    s!"deltaLB={cert.deltaLowerBound}\n" ++
    logitLine

/-- Render a local induction certificate report. -/
private def formatInductionLocal
    (cert : Nfp.Sound.InductionHeadSoundCert) : String :=
  let p1 := cert.layer1Pattern
  let p2 := cert.layer2Pattern
  let v := cert.layer2Value
  let logitLine := formatInductionLogitLine cert.layer2Logit?
  "SOUND induction cert:\n" ++
    s!"layer1=L{p1.layerIdx} H{p1.headIdx} offset={p1.targetOffset} " ++
      s!"keyOffset={p1.keyOffset} " ++
      s!"marginLB={p1.marginLowerBound} weightLB={p1.targetWeightLowerBound} " ++
      s!"softmaxExpEffort={p1.softmaxExpEffort}\n" ++
    s!"layer2=L{p2.layerIdx} H{p2.headIdx} offset={p2.targetOffset} " ++
      s!"keyOffset={p2.keyOffset} " ++
      s!"marginLB={p2.marginLowerBound} weightLB={p2.targetWeightLowerBound} " ++
      s!"softmaxExpEffort={p2.softmaxExpEffort}\n" ++
    s!"coord={v.coord} matchCountLB={p2.targetCountLowerBound} " ++
      s!"matchCoordLB={v.matchCoordLowerBound} " ++
      s!"nonmatchCoordLB={v.nonmatchCoordLowerBound}\n" ++
    s!"deltaLB={cert.deltaLowerBound}\n" ++
    logitLine

/-- Run the induction-cert action and return the report string. -/
private def runInductionCertAction (args : InductionCertArgs) : ExceptT String IO String := do
  if args.useAffine && !args.bestMatch then
    throw "affine bounds are only supported with --bestMatch"
  if args.bestMatch then
    let cert ← ExceptT.mk <|
      Nfp.Sound.certifyInductionSoundBestMatch args.modelPath
        args.layer1 args.head1 args.layer2 args.head2 args.coord
        (queryPos? := args.queryPos?) (inputPath? := args.inputPath?)
        (inputDelta := args.delta) (soundnessBits := args.soundnessBits)
        (offset1 := args.offset1) (offset2 := args.offset2)
        (keyOffset1 := args.keyOffset1) (keyOffset2 := args.keyOffset2)
        (maxSeqLen := args.maxSeqLen)
        (scalePow10 := args.scalePow10)
        (tightPattern := args.tightPattern)
        (tightPatternLayers := args.tightPatternLayers)
        (perRowPatternLayers := args.perRowPatternLayers)
        (useAffine := args.useAffine)
        (iterTighten := args.iterTighten)
        (targetToken? := args.targetToken?) (negativeToken? := args.negativeToken?)
        (softmaxExpEffort := args.softmaxExpEffort)
        (causalPattern := args.causalPattern)
    return formatInductionBestMatch cert
  else
    let cert ← ExceptT.mk <|
      Nfp.Sound.certifyInductionSound args.modelPath
        args.layer1 args.head1 args.layer2 args.head2 args.coord
        (inputPath? := args.inputPath?) (inputDelta := args.delta)
        (soundnessBits := args.soundnessBits)
        (offset1 := args.offset1) (offset2 := args.offset2)
        (keyOffset1 := args.keyOffset1) (keyOffset2 := args.keyOffset2)
        (maxSeqLen := args.maxSeqLen)
        (scalePow10 := args.scalePow10)
        (tightPattern := args.tightPattern) (tightPatternLayers := args.tightPatternLayers)
        (perRowPatternLayers := args.perRowPatternLayers)
        (targetToken? := args.targetToken?) (negativeToken? := args.negativeToken?)
        (softmaxExpEffort := args.softmaxExpEffort)
        (causalPattern := args.causalPattern)
    return formatInductionLocal cert

/-- Run the induction-cert command - compute a sound induction head certificate. -/
def runInductionCert (p : Parsed) : IO UInt32 := do
  let action : ExceptT String IO (String × Option System.FilePath) := do
    let args ← parseInductionCertArgs p
    let report ← runInductionCertAction args
    return (report, args.outputPath?)
  match ← action.run with
  | .error msg =>
    IO.eprintln s!"Error: {msg}"
    return 1
  | .ok (s, outputPath?) =>
    match outputPath? with
    | some path =>
      IO.FS.writeFile path s
      IO.println s!"Report written to {path}"
    | none =>
      IO.println s
    return 0

/-! ## Sound cache check helpers -/

private structure SoundCacheCheckArgs where
  modelPath : System.FilePath
  scalePow10 : Nat
  maxTokens : Nat

private def parseSoundCacheCheckArgs (p : Parsed) : SoundCacheCheckArgs :=
  let modelPath := p.positionalArg! "model" |>.as! String
  let scalePow10 := p.flag? "scalePow10" |>.map (·.as! Nat) |>.getD 9
  let maxTokens := p.flag? "maxTokens" |>.map (·.as! Nat) |>.getD 0
  { modelPath := ⟨modelPath⟩, scalePow10 := scalePow10, maxTokens := maxTokens }

private def runSoundCacheCheckWithArgs (args : SoundCacheCheckArgs) : IO UInt32 := do
  let modelHash ← Nfp.Untrusted.SoundCacheIO.fnv1a64File args.modelPath
  let cacheFp := Nfp.Sound.SoundCache.cachePath args.modelPath modelHash args.scalePow10
  match (← Nfp.Untrusted.SoundCacheIO.buildCacheFile args.modelPath cacheFp args.scalePow10) with
  | .error e =>
      IO.eprintln s!"Error: cache build failed: {e}"
      return 1
  | .ok _ =>
      let ch ← IO.FS.Handle.mk cacheFp IO.FS.Mode.read
      let hdr ← Nfp.Untrusted.SoundCacheIO.readHeader ch
      if hdr.modelHash ≠ modelHash then
        IO.eprintln "Error: cache hash mismatch"
        return 1
      match (← Nfp.Untrusted.SoundCacheIO.checkCacheFileSize cacheFp hdr) with
      | .error e =>
          IO.eprintln s!"Error: {e}"
          return 1
      | .ok _ =>
          match
            (← Nfp.Untrusted.SoundCacheIO.checkTextTokenEnvelope
              args.modelPath args.scalePow10 args.maxTokens) with
          | .error e =>
              IO.eprintln s!"Error: {e}"
              return 1
          | .ok _ =>
              IO.println <|
                "OK: sound cache validated " ++
                  s!"(scalePow10={args.scalePow10}, maxTokens={args.maxTokens})"
              return 0

/-- Regression test for SOUND fixed-point cache soundness and consistency.

This is intended for CI and small fixtures. It:
- builds a `.nfpc` cache (if needed),
- checks cache size matches the expected tensor stream length,
- checks the `±1`-ulp rounding envelope on up to `maxTokens` numeric tokens in the text file.
-/
def runSoundCacheCheck (p : Parsed) : IO UInt32 := do
  let args := parseSoundCacheCheckArgs p
  runSoundCacheCheckWithArgs args

/-! ## Sound cache benchmark helpers -/

private structure SoundCacheBenchArgs where
  modelPath : System.FilePath
  scalePow10 : Nat
  runs : Nat

private def parseSoundCacheBenchArgs (p : Parsed) : SoundCacheBenchArgs :=
  let modelPath := p.positionalArg! "model" |>.as! String
  let scalePow10 := p.flag? "scalePow10" |>.map (·.as! Nat) |>.getD 9
  let runs := p.flag? "runs" |>.map (·.as! Nat) |>.getD 1
  { modelPath := ⟨modelPath⟩, scalePow10 := scalePow10, runs := runs }

private def runSoundCacheBenchWithArgs (args : SoundCacheBenchArgs) : IO UInt32 := do
  if args.runs = 0 then
    IO.eprintln "Error: --runs must be > 0"
    return 1
  let modelHash ← Nfp.Untrusted.SoundCacheIO.fnv1a64File args.modelPath
  let mdata ← args.modelPath.metadata
  let modelSize : UInt64 := mdata.byteSize
  let isBinaryE ← Nfp.Untrusted.SoundCacheIO.isBinaryModelFile args.modelPath
  let isBinary ←
    match isBinaryE with
    | .error e =>
        IO.eprintln s!"Error: {e}"
        return 1
    | .ok b => pure b
  let formatStr := if isBinary then "binary" else "text"
  let mut times : Array Nat := Array.mkEmpty args.runs
  let mut lastBytes : Nat := 0
  for i in [:args.runs] do
    let t0 ← IO.monoNanosNow
    let bytesE ←
      if isBinary then
        Nfp.Untrusted.SoundCacheIO.buildCacheBytesBinary
          args.modelPath args.scalePow10 modelHash modelSize
      else
        Nfp.Untrusted.SoundCacheIO.buildCacheBytesText
          args.modelPath args.scalePow10 modelHash modelSize
    let t1 ← IO.monoNanosNow
    match bytesE with
    | .error e =>
        IO.eprintln s!"Error: {e}"
        return 1
    | .ok bytes =>
        let dtMs := (t1 - t0) / 1000000
        times := times.push dtMs
        lastBytes := bytes.size
        if args.runs > 1 then
          IO.println s!"run {i + 1}: {dtMs}ms"
  let t0 := times[0]!
  let mut minT := t0
  let mut maxT := t0
  let mut sumT : Nat := 0
  for t in times do
    if t < minT then
      minT := t
    if t > maxT then
      maxT := t
    sumT := sumT + t
  let avgT := sumT / times.size
  IO.println s!"cacheBuild format={formatStr} scalePow10={args.scalePow10} bytes={lastBytes}"
  IO.println s!"cacheBuild runs={args.runs} min={minT}ms avg={avgT}ms max={maxT}ms"
  return 0

def runSoundCacheBench (p : Parsed) : IO UInt32 := do
  let args := parseSoundCacheBenchArgs p
  runSoundCacheBenchWithArgs args

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
          (Nfp.rope_operatorNormBound_le_two
            (pos := Fin (Nat.succ n)) (pair := Fin (Nat.succ m)) θ)
      IO.println "RoPE certificate (static):"
      IO.println s!"  seqLen={seqLen}, pairs={pairs}, dim={2 * pairs}"
      IO.println "  Bound: operatorNormBound(ropeJacobian θ) ≤ 2"
      IO.println "  Meaning: max row-sum of absolute weights (ℓ1 induced for row-vectors)."
      IO.println "  Proof: Nfp.rope_operatorNormBound_le_two (uses |sin|,|cos| ≤ 1 from mathlib)."
      return 0
  | _, _ =>
      IO.eprintln "Error: --seqLen and --pairs must be positive."
      return 1

/-! ## Dump command helpers -/

private structure DumpArgs where
  modelPath : System.FilePath
  modelPathStr : String
  layer : Nat
  pos : Nat
  take : Nat
  kind : String

private def parseDumpArgs (p : Parsed) : DumpArgs :=
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let layer := p.flag? "layer" |>.map (·.as! Nat) |>.getD 0
  let pos := p.flag? "pos" |>.map (·.as! Nat) |>.getD 0
  let take := p.flag? "take" |>.map (·.as! Nat) |>.getD 16
  let kind := p.flag? "kind" |>.map (·.as! String) |>.getD "afterLayer"
  { modelPath := ⟨modelPathStr⟩
    modelPathStr := modelPathStr
    layer := layer
    pos := pos
    take := take
    kind := kind }

private def selectDumpMatrix
    (kind : String) (layer : Nat) (model : ConcreteModel) (fwd : ForwardPassResult) :
    ConcreteMatrix :=
  match kind with
  | "embeddings" => model.inputEmbeddings
  | "layerInput" => fwd.getLayerInput layer
  | "postAttn" => fwd.getPostAttnResidual layer
  | "afterLayer" => fwd.getLayerInput (layer + 1)
  | _ => fwd.getLayerInput (layer + 1)

private def collectDumpRow (X : ConcreteMatrix) (pos n : Nat) :
    (Array Float × Float × Float) := Id.run do
  let mut xs : Array Float := Array.mkEmpty n
  let mut sum : Float := 0.0
  let mut sumSq : Float := 0.0
  for j in [:n] do
    let v := X.get pos j
    xs := xs.push v
    sum := sum + v
    sumSq := sumSq + v * v
  return (xs, sum, sumSq)

private def runDumpWithArgs (args : DumpArgs) : IO UInt32 := do
  setStdoutLogNameFromModelPath args.modelPathStr
  let loadResult ← loadModel args.modelPath
  match loadResult with
  | .error msg =>
      IO.eprintln s!"Error loading model: {msg}"
      return 1
  | .ok model0 =>
      let model := model0.trimTrailingZeroEmbeddings
      let fwd := model.runForward true
      let X := selectDumpMatrix args.kind args.layer model fwd
      if X.numRows = 0 || X.numCols = 0 then
        IO.eprintln s!"Error: empty matrix for kind={args.kind}"
        return 1
      if args.pos ≥ X.numRows then
        IO.eprintln s!"Error: pos={args.pos} out of bounds (rows={X.numRows})"
        return 1
      let n := min args.take X.numCols
      let (xs, sum, sumSq) := collectDumpRow X args.pos n
      IO.println <|
        s!"DUMP kind={args.kind} layer={args.layer} pos={args.pos} take={n} " ++
          s!"rows={X.numRows} cols={X.numCols}"
      IO.println s!"sum={sum} sumSq={sumSq}"
      IO.println (String.intercalate " " (xs.toList.map (fun x => s!"{x}")))
      return 0

/-- Dump a small slice of a forward pass for cross-implementation sanity checks. -/
def runDump (p : Parsed) : IO UInt32 := do
  let args := parseDumpArgs p
  runDumpWithArgs args

/-! ## Logit-difference helpers -/

private def logitAt (residual : ConcreteMatrix) (pos : Nat)
    (W_U : ConcreteMatrix) (token : Nat) : Except String Float :=
  if residual.numCols ≠ W_U.numRows then
    .error "dimension mismatch: residual.numCols != W_U.numRows"
  else if pos ≥ residual.numRows then
    .error "position out of range"
  else if token ≥ W_U.numCols then
    .error "token out of range"
  else
    .ok <| Id.run do
      let d := residual.numCols
      let vocab := W_U.numCols
      let rowBase := pos * d
      let mut acc : Float := 0.0
      for k in [:d] do
        acc := acc + residual.data[rowBase + k]! * W_U.data[k * vocab + token]!
      return acc

private def topNonTargetToken (residual : ConcreteMatrix) (pos : Nat)
    (W_U : ConcreteMatrix) (targetToken : Nat) : Except String (Nat × Float) :=
  if residual.numCols ≠ W_U.numRows then
    .error "dimension mismatch: residual.numCols != W_U.numRows"
  else if pos ≥ residual.numRows then
    .error "position out of range"
  else if targetToken ≥ W_U.numCols then
    .error "target token out of range"
  else if W_U.numCols < 2 then
    .error "vocab size too small to select non-target token"
  else
    .ok <| Id.run do
      let d := residual.numCols
      let vocab := W_U.numCols
      let rowBase := pos * d
      let mut bestTok : Nat := 0
      let mut bestLogit : Float := (-Float.inf)
      let mut found : Bool := false
      for tok in [:vocab] do
        if tok ≠ targetToken then
          found := true
          let mut acc : Float := 0.0
          for k in [:d] do
            acc := acc + residual.data[rowBase + k]! * W_U.data[k * vocab + tok]!
          if acc > bestLogit then
            bestTok := tok
            bestLogit := acc
      if found then
        return (bestTok, bestLogit)
      else
        return (0, bestLogit)

private structure LogitDiffArgs where
  modelPath : System.FilePath
  modelPathStr : String
  target : Nat
  negative : Nat
  pos? : Option Nat
  inputPath? : Option System.FilePath
  autoNegative : Bool

private def parseLogitDiffArgs (p : Parsed) : LogitDiffArgs :=
  let modelPathStr := p.positionalArg! "model" |>.as! String
  let target := p.positionalArg! "target" |>.as! Nat
  let negative := p.positionalArg! "negative" |>.as! Nat
  let pos? := p.flag? "pos" |>.map (·.as! Nat)
  let inputPath? := p.flag? "input" |>.map (System.FilePath.mk ∘ (·.as! String))
  let autoNegative := p.hasFlag "autoNegative"
  { modelPath := ⟨modelPathStr⟩
    modelPathStr := modelPathStr
    target := target
    negative := negative
    pos? := pos?
    inputPath? := inputPath?
    autoNegative := autoNegative }

private def runLogitDiff (p : Parsed) : IO UInt32 := do
  let args := parseLogitDiffArgs p
  setStdoutLogNameFromModelPath args.modelPathStr
  let loadResult ← loadModel args.modelPath
  match loadResult with
  | .error msg =>
      IO.eprintln s!"Error loading model: {msg}"
      return 1
  | .ok model0 =>
      let model ←
        match args.inputPath? with
        | none => pure model0
        | some inputPath =>
            match ← loadInputBinary inputPath with
            | .error msg =>
                IO.eprintln s!"Error loading input: {msg}"
                return 1
            | .ok input =>
                if input.modelDim ≠ model0.modelDim then
                  IO.eprintln s!"Input model_dim mismatch ({input.modelDim} != {model0.modelDim})"
                  return 1
                pure {
                  model0 with
                  seqLen := input.seqLen
                  inputTokens := some input.tokens
                  inputEmbeddings := input.embeddings
                }
      match model.unembedding with
      | none =>
          IO.eprintln "Error: Model is missing unembedding matrix (needed for logits)."
          return 1
      | some W_U =>
          if model.seqLen = 0 then
            IO.eprintln "Error: seq_len = 0; cannot compute logits."
            return 1
          let pos := args.pos?.getD (model.seqLen - 1)
          if pos ≥ model.seqLen then
            IO.eprintln s!"Error: pos={pos} out of bounds (seq_len={model.seqLen})"
            return 1
          let fwd := model.runForward true
          let residual := fwd.finalOutput
          let negResult :=
            if args.autoNegative then
              topNonTargetToken residual pos W_U args.target
            else
              match logitAt residual pos W_U args.negative with
              | .ok logit => .ok (args.negative, logit)
              | .error msg => .error msg
          match logitAt residual pos W_U args.target, negResult with
          | .ok targetLogit, .ok (negTok, negLogit) =>
              let diff := targetLogit - negLogit
              IO.println s!"pos={pos} target={args.target} negative={negTok}"
              if args.autoNegative then
                IO.println "negativeSource=topNonTarget"
              IO.println s!"logit(target)={targetLogit}"
              IO.println s!"logit(negative)={negLogit}"
              IO.println s!"logitDiff={diff}"
              return 0
          | .error msg, _ =>
              IO.eprintln s!"Error computing target logit: {msg}"
              return 1
          | _, .error msg =>
              IO.eprintln s!"Error computing negative logit: {msg}"
              return 1

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
    maxUpgrades : Nat; "Maximum adaptive upgrades (default: 120)"
    minRelImprove : String; "Stop upgrading a layer if improvement < this fraction (default: 0.01)"
    krylovSteps : Nat; "Krylov steps for LOWER bounds only (default: 2)"
    adaptiveScope : String; "Adaptive scope: layernorm | all (default: layernorm)"
  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The bench subcommand. -/
def benchCmd : Cmd := `[Cli|
  bench VIA runBench;
  "Run repeatable microbenchmarks on analysis or induction search."
  FLAGS:
    mode : String; "analysis|induction (default: analysis)"
    runs : Nat; "Number of timed runs (default: 5)"
    repeats : Nat; "Repeat inner workload per run (default: 1)"
    t, threshold : String; "Analyze threshold (default: 0.1)"
    minEffect : String; "Induction minEffect (default: 0.0)"
    c, correct : Nat; "Correct token ID (requires --incorrect)"
    i, incorrect : Nat; "Incorrect token ID (requires --correct)"
    v, verbose; "Print per-run timing details"
    breakdown; "Emit per-phase averages (analysis only)"
  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The certify subcommand. -/
def certifyCmd : Cmd := `[Cli|
  certify VIA runCertify;
  "SOUND mode: compute conservative bounds using exact Rat arithmetic (no Float trust). \
LayerNorm epsilon and GeLU kind are read from the model header."
  FLAGS:
    input : String; "Optional input .nfpt file for local certification (must contain EMBEDDINGS \
for legacy text)"
    delta : String; "Input ℓ∞ radius δ for local certification (default: 0; \
if --input is omitted, uses EMBEDDINGS in the model file when present)"
    softmaxMargin : String; "Lower bound on softmax logit margin (default: 0)"
    softmaxExpEffort : Nat; "Effort level for margin-based exp lower bounds (default: 1)"
    bestMatchMargins; "Apply best-match margin tightening (binary + local only)"
    targetOffset : Int; "Token-match offset for best-match margins (default: -1)"
    maxSeqLen : Nat; "Max sequence length for best-match margins (default: 0 uses full seq_len)"
    tightPattern; "Use tighter (slower) pattern bounds for best-match margins"
    tightPatternLayers : Nat; "Number of layers using tight pattern bounds (default: 1)"
    perRowPatternLayers : Nat; "Number of layers using per-row MLP propagation (default: 0)"
    noncausalPattern; "Disable causal-prefix restriction for pattern/value bounds"
    scalePow10 : Nat; "Fixed-point scale exponent for best-match margins (default: 9)"
    soundnessBits : Nat; "Dyadic sqrt precision bits for LayerNorm bounds (default: 20)"
    partitionDepth : Nat; "Partition depth for input splitting (default: 0; >0 scaffold only)"
    o, output : String; "Write report to file instead of stdout"
  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The head-bounds subcommand. -/
def headBoundsCmd : Cmd := `[Cli|
  head_bounds VIA runHeadBounds;
  "SOUND mode: compute per-head contribution bounds. \
LayerNorm epsilon is read from the model header."
  FLAGS:
    input : String; "Optional input .nfpt file for local bounds (must contain EMBEDDINGS \
for legacy text)"
    delta : String; "Input ℓ∞ radius δ for local bounds (default: 0; if --input is omitted, \
uses EMBEDDINGS in the model file when present)"
    soundnessBits : Nat; "Dyadic sqrt precision bits for LayerNorm bounds (default: 20)"
    scalePow10 : Nat; "Fixed-point scale exponent p in S=10^p for global bounds (default: 9)"
    o, output : String; "Write report to file instead of stdout"
  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The head-pattern subcommand. -/
def headPatternCmd : Cmd := `[Cli|
  head_pattern VIA runHeadPattern;
  "SOUND mode: compute per-head attention pattern bounds (binary only). \
LayerNorm epsilon is read from the model header."
  FLAGS:
    layer : Nat; "Layer index (default: 0)"
    head : Nat; "Head index (default: 0)"
    offset : Int; "Token-match offset (default: -1 for previous token, 0 for self)"
    keyOffset : Int; "Key-position token offset (default: 0; use -1 with offset=0 for copy-next)"
    tightPattern; "Use tighter (slower) pattern bounds near the target layer"
    tightPatternLayers : Nat; "Number of layers using tight pattern bounds (default: 1)"
    perRowPatternLayers : Nat; "Number of layers using per-row MLP propagation (default: 0)"
    noncausalPattern; "Disable causal-prefix restriction for pattern/value bounds"
    bestMatch; "Use best-match (single-query) pattern bounds"
    affine; "Use affine Q/K dot bounds for best-match (single-query only)"
    sweep; "Sweep best-match bounds across all valid query positions"
    queryPos : Nat; "Query position for best-match bounds (default: last position)"
    input : String; "Optional input .nfpt file for local bounds (must contain EMBEDDINGS \
for legacy text)"
    delta : String; "Input ℓ∞ radius δ for local bounds (default: 0)"
    soundnessBits : Nat; "Dyadic sqrt precision bits for LayerNorm bounds (default: 20)"
    softmaxExpEffort : Nat; "Effort level for margin-based exp lower bounds (default: 1)"
    maxSeqLen : Nat; "Maximum sequence length to analyze (default: 256)"
    scalePow10 : Nat; "Fixed-point scale exponent for best-match bounds (default: 9)"
    o, output : String; "Write report to file instead of stdout"
  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The induction-cert subcommand. -/
def inductionCertCmd : Cmd := `[Cli|
  induction_cert VIA runInductionCert;
  "SOUND mode: compute a minimal induction head certificate (binary only). \
LayerNorm epsilon is read from the model header."
  FLAGS:
    layer1 : Nat; "Layer index for the previous-token head (default: 0)"
    head1 : Nat; "Head index for the previous-token head (default: 0)"
    layer2 : Nat; "Layer index for the token-match head (default: 1)"
    head2 : Nat; "Head index for the token-match head (default: 0)"
    coord : Nat; "Output coordinate for the value bound (default: 0)"
    offset1 : Int; "Token-match offset for layer1 (default: -1)"
    offset2 : Int; "Token-match offset for layer2 (default: -1)"
    keyOffset1 : Int; "Key-position token offset for layer1 (default: 0)"
    keyOffset2 : Int; "Key-position token offset for layer2 (default: 0; use -1 with \
offset2=0 for copy-next)"
    target : Nat; "Target token ID for logit-diff bound (optional; requires --negative)"
    negative : Nat; "Negative token ID for logit-diff bound (optional; requires --target)"
    tightPattern; "Use tighter (slower) pattern bounds near the target layer"
    tightPatternLayers : Nat; "Number of layers using tight pattern bounds (default: 1)"
    perRowPatternLayers : Nat; "Number of layers using per-row MLP propagation (default: 0)"
    iterTighten; "Iteratively tighten best-match bounds (escalates tight/per-row layers to full)"
    noncausalPattern; "Disable causal-prefix restriction for pattern/value bounds"
    bestMatch; "Use best-match (single-query) pattern bounds"
    affine; "Use affine Q/K dot bounds for best-match"
    queryPos : Nat; "Query position for best-match bounds (default: last position)"
    input : String; "Optional input .nfpt file for local bounds (must contain EMBEDDINGS \
for legacy text)"
    delta : String; "Input ℓ∞ radius δ for local bounds (default: 0)"
    soundnessBits : Nat; "Dyadic sqrt precision bits for LayerNorm bounds (default: 20)"
    softmaxExpEffort : Nat; "Effort level for margin-based exp lower bounds (default: 1)"
    maxSeqLen : Nat; "Maximum sequence length to analyze (default: 256)"
    scalePow10 : Nat; "Fixed-point scale exponent for best-match bounds (default: 9)"
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

/-- The sound-cache-bench subcommand. -/
def soundCacheBenchCmd : Cmd := `[Cli|
  sound_cache_bench VIA runSoundCacheBench;
  "Benchmark SOUND fixed-point cache build (text or binary)."
  FLAGS:
    scalePow10 : Nat; "Fixed-point scale exponent p in S=10^p (default: 9)"
    runs : Nat; "Number of benchmark runs (default: 1)"
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

/-- The logit-diff subcommand. -/
def logitDiffCmd : Cmd := `[Cli|
  logit_diff VIA runLogitDiff;
  "Compute empirical logit-difference for target vs. negative token."
  FLAGS:
    pos : Nat; "Token position (default: last position)"
    input : String; "Optional input .nfpt file with TOKENS + EMBEDDINGS"
    autoNegative; "Use top non-target logit as negative token (ignores provided negative)"
  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
    target : Nat; "Target token ID"
    negative : Nat; "Negative token ID"
]

/-- The main CLI command. -/
def nfpCmd : Cmd := `[Cli|
  nfp NOOP;
  "NFP: Neural Formal Pathways verification toolkit"
  SUBCOMMANDS:
    analyzeCmd;
    inductionCmd;
    benchCmd;
    certifyCmd;
    headBoundsCmd;
    headPatternCmd;
    inductionCertCmd;
    soundCacheCheckCmd;
    soundCacheBenchCmd;
    ropeCmd;
    dumpCmd;
    logitDiffCmd
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
