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

/-- The main CLI command. -/
def nfpCmd : Cmd := `[Cli|
  nfp NOOP;
  "NFP: Neural Feature Pathway verification toolkit"

  SUBCOMMANDS:
    analyzeCmd
]

/-- Main entry point. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--version" then
    IO.println "nfp version 0.1.0"
    return 0

  nfpCmd.validate args
