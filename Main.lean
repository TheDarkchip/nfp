import Cli
import Nfp.IO

/-!
# NFP CLI: Circuit Verification Command-Line Tool

This is the main entry point for the NFP circuit verification tool.

## Usage

Build the executable:
```bash
lake build nfp
```

Run analysis:
```bash
# Basic analysis
nfp analyze model.nfpt

# With threshold
nfp analyze model.nfpt --threshold 0.05

# With output file
nfp analyze model.nfpt --output report.txt

# Verify mode (empirical validation)
nfp analyze model.nfpt --verify

# Verbose output
nfp analyze model.nfpt --verbose

# Show version
nfp --version

# Show help
nfp -h
nfp analyze -h
```
-/

open Cli Nfp

/-- Run the analyze command - perform circuit analysis. -/
def runAnalyze (p : Parsed) : IO UInt32 := do
  let modelPath := p.positionalArg! "model" |>.as! String
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
  | .ok model embeddings =>
    if verbose then
      IO.println s!"✓ Model loaded successfully"
      IO.println s!"  Layers: {model.numLayers}"
      IO.println s!"  Sequence Length: {model.seqLen}"
      IO.println s!"  Embedding Vocabulary: {embeddings.numRows}"
      IO.println s!"  Model Dimension: {embeddings.numCols}"
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

/-- Run the induction command - discover induction heads with certified effectiveness. -/
def runInduction (p : Parsed) : IO UInt32 := do
  let modelPath := p.positionalArg! "model" |>.as! String
  let correctOpt := p.flag? "correct" |>.map (·.as! Nat)
  let incorrectOpt := p.flag? "incorrect" |>.map (·.as! Nat)
  let thresholdStr := p.flag? "threshold" |>.map (·.as! String) |>.getD "0.5"
  let verbose := p.hasFlag "verbose"
  let some threshold := Nfp.parseFloat thresholdStr
    | do
      IO.eprintln s!"Error: Invalid threshold value '{thresholdStr}'"
      return 1

  IO.println "Loading model..."
  let loadResult ← loadModel ⟨modelPath⟩

  match loadResult with
  | .error msg =>
    IO.eprintln s!"Error loading model: {msg}"
    return 1
  | .ok model _embeddings =>
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
      IO.println s!"Searching for heads with effect > {threshold}..."

      let heads := findCertifiedInductionHeads model target threshold
      IO.println s!"Certified Induction Heads (δ > {threshold})"
      IO.println s!"Found {heads.size} certified heads:"

      for h in heads do
        let c := h.candidate
        if verbose then
          IO.println <|
            s!"L{c.layer1Idx}H{c.head1Idx} -> L{c.layer2Idx}H{c.head2Idx} | " ++
              s!"Effectiveness (δ): {h.delta} | Error (ε): {c.combinedError} " ++
              s!"(ε₁={c.patternBound1}, ε₂={c.patternBound2})"
        else
          IO.println <|
            s!"L{c.layer1Idx}H{c.head1Idx} -> L{c.layer2Idx}H{c.head2Idx} | " ++
              s!"Effectiveness (δ): {h.delta} | Error (ε): {c.combinedError}"

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
  "Discover induction head pairs with certified virtual-head effectiveness."

  FLAGS:
    c, correct : Nat; "Correct token ID (manual override; requires --incorrect)"
    i, incorrect : Nat; "Incorrect token ID (manual override; requires --correct)"
    t, threshold : String; "Effectiveness threshold (default: 0.5)"
    v, verbose; "Enable verbose output"

  ARGS:
    model : String; "Path to the model weights file (.nfpt)"
]

/-- The main CLI command. -/
def nfpCmd : Cmd := `[Cli|
  nfp NOOP;
  "NFP: Neural Feature Pathway verification toolkit"

  SUBCOMMANDS:
    analyzeCmd;
    inductionCmd
]

/-- Main entry point. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--version" then
    IO.println "nfp version 0.1.0"
    return 0

  nfpCmd.validate args
