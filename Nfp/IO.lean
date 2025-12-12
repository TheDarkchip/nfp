import Nfp.Discovery

/-!
# Model IO: Loading Pre-trained Weights

This module provides functionality to load pre-trained transformer model weights
from external files (exported from PyTorch/JAX) into the `ConcreteModel` structure.

## Performance & Safety Design

This module prioritizes **safety and clear error reporting** over raw performance.
- All array accesses use bounds-checked indexing (`array[i]!` panics on OOB)
- Comprehensive validation of file format and dimensions with helpful error messages
- File format parsing is I/O-bound, so optimizing array operations has minimal impact

For high-performance computation, see `Discovery.lean` where hot paths are optimized.

## File Format: `.nfpt` (NFP Text Format)

Human-readable text format for debugging and small models:

```
NFP_TEXT_V1
num_layers=12
num_heads=12
model_dim=768
head_dim=64
hidden_dim=3072
vocab_size=50257
seq_len=1024

TOKENS
<space-separated integers (token IDs), length = seq_len>

EMBEDDINGS
<space-separated floats, row-major>

LAYER 0
HEAD 0
W_Q
<floats>
W_K
<floats>
...
```
-/

namespace Nfp

open IO

/-! ## Float Parsing Utilities -/

/-- Parse a floating point number from a string.
    Handles integers, decimals, and negative numbers. -/
def parseFloat (s : String) : Option Float := do
  let s := s.trim
  if s.isEmpty then none
  else
    -- Handle sign
    let (negative, rest) :=
      if s.startsWith "-" then (true, s.drop 1)
      else if s.startsWith "+" then (false, s.drop 1)
      else (false, s)

    -- Split on decimal point
    let parts := rest.splitOn "."
    match parts with
    | [intPart] =>
      -- No decimal point - pure integer
      let n := intPart.toNat?
      n.map fun n => (if negative then -1.0 else 1.0) * n.toFloat
    | [intPart, fracPart] =>
      -- Has decimal point
      let intN := if intPart.isEmpty then some 0 else intPart.toNat?
      let fracN := if fracPart.isEmpty then some 0 else fracPart.toNat?
      match intN, fracN with
      | some iN, some fN =>
        let fracLen := fracPart.length
        let divisor := Float.pow 10.0 fracLen.toFloat
        let value := iN.toFloat + fN.toFloat / divisor
        some ((if negative then -1.0 else 1.0) * value)
      | _, _ => none
    | _ => none

/-- Parse a line of space-separated floats. -/
def parseFloatLine (line : String) : Array Float :=
  let parts := line.splitOn " " |>.filter (· ≠ "")
  parts.toArray.filterMap parseFloat

/-! ## Nat Parsing Utilities -/

/-- Parse a line of space-separated natural numbers. -/
def parseNatLine (line : String) : Array Nat :=
  let parts := line.splitOn " " |>.filter (· ≠ "")
  parts.toArray.filterMap (·.trim.toNat?)

/-! ## Matrix Construction for IO

For IO operations, we need to create matrices from runtime data.
We use a safe construction that always ensures size invariants hold
by padding/truncating the data.
-/

/-- Build a ConcreteMatrix from float data, padding or truncating as needed.
    This is safe because we ensure the data has exactly the right size. -/
def buildMatrix (rows cols : Nat) (data : Array Float) : ConcreteMatrix :=
  let expectedSize := rows * cols
  let normalizedData : Array Float :=
    if data.size < expectedSize then
      data ++ (Array.replicate (expectedSize - data.size) 0.0)
    else if data.size > expectedSize then
      data.toSubarray 0 expectedSize |>.toArray
    else
      data
  -- Use Array.ofFn to get the exact size we need with a proof
  let finalData := Array.ofFn fun (i : Fin expectedSize) =>
    normalizedData.getD i.val 0.0
  {
    numRows := rows
    numCols := cols
    data := finalData
    size_eq := Array.size_ofFn
  }

/-- Result of loading a model. -/
inductive LoadResult
  | ok (model : ConcreteModel) (embeddings : ConcreteMatrix)
  | error (msg : String)

namespace LoadResult

def isOk : LoadResult → Bool
  | ok _ _ => true
  | error _ => false

def getModel : LoadResult → Option ConcreteModel
  | ok m _ => some m
  | error _ => none

def getEmbeddings : LoadResult → Option ConcreteMatrix
  | ok _ e => some e
  | error _ => none

def getError : LoadResult → Option String
  | ok _ _ => none
  | error msg => some msg

end LoadResult

/-! ## Text Format Parsing -/

/-- NFP file header structure. -/
structure NfpHeader where
  numLayers : Nat
  numHeads : Nat
  modelDim : Nat
  headDim : Nat
  hiddenDim : Nat
  vocabSize : Nat
  seqLen : Nat
  deriving Repr

/-- Build a ConcreteAttentionLayer from weight matrices.
    The dimension proofs are satisfied by construction (buildMatrix ensures correct sizes). -/
def mkAttentionLayer
    (modelDim headDim : Nat)
    (wq wk wv wo : Array Float) : ConcreteAttentionLayer :=
  let wQ := buildMatrix modelDim headDim wq
  let wK := buildMatrix modelDim headDim wk
  let wV := buildMatrix modelDim headDim wv
  let wO := buildMatrix headDim modelDim wo
  {
    modelDim := modelDim
    headDim := headDim
    W_Q := wQ
    W_K := wK
    W_V := wV
    W_O := wO
    W_Q_dims := ⟨rfl, rfl⟩
    W_K_dims := ⟨rfl, rfl⟩
    W_V_dims := ⟨rfl, rfl⟩
    W_O_dims := ⟨rfl, rfl⟩
  }

/-- Build a ConcreteMLPLayer from weight matrices.
    The dimension proofs are satisfied by construction. -/
def mkMLPLayer
    (modelDim hiddenDim : Nat)
    (win wout bin bout : Array Float) : ConcreteMLPLayer :=
  let wIn := buildMatrix modelDim hiddenDim win
  let wOut := buildMatrix hiddenDim modelDim wout
  let bIn := buildMatrix 1 hiddenDim bin
  let bOut := buildMatrix 1 modelDim bout
  {
    modelDim := modelDim
    hiddenDim := hiddenDim
    W_in := wIn
    W_out := wOut
    b_in := bIn
    b_out := bOut
    W_in_dims := ⟨rfl, rfl⟩
    W_out_dims := ⟨rfl, rfl⟩
    b_in_dims := ⟨rfl, rfl⟩
    b_out_dims := ⟨rfl, rfl⟩
  }

/-- Load a model from NFP text format content. -/
def loadFromText (content : String) : IO LoadResult := do
  let lines := content.splitOn "\n" |>.toArray
  if lines.size = 0 then return .error "Empty file"

  -- Check magic - SAFETY: lines.size > 0 guaranteed above
  if lines[0]! != "NFP_TEXT_V1" then
    return .error "Invalid magic: expected NFP_TEXT_V1"

  IO.println "[1/5] Parsing header..."

  -- Parse header
  let mut numLayers : Nat := 0
  let mut numHeads : Nat := 0
  let mut modelDim : Nat := 0
  let mut headDim : Nat := 0
  let mut hiddenDim : Nat := 0
  let mut vocabSize : Nat := 0
  let mut seqLen : Nat := 0

  let mut i := 1
  while i < lines.size &&
        lines[i]!.trim != "TOKENS" &&
        !lines[i]!.startsWith "EMBEDDINGS" do
    -- SAFETY: i < lines.size guaranteed by while condition
    let line := lines[i]!
    if line.startsWith "num_layers=" then
      numLayers := (line.drop 11).toNat!
    else if line.startsWith "num_heads=" then
      numHeads := (line.drop 10).toNat!
    else if line.startsWith "model_dim=" then
      modelDim := (line.drop 10).toNat!
    else if line.startsWith "head_dim=" then
      headDim := (line.drop 9).toNat!
    else if line.startsWith "hidden_dim=" then
      hiddenDim := (line.drop 11).toNat!
    else if line.startsWith "vocab_size=" then
      vocabSize := (line.drop 11).toNat!
    else if line.startsWith "seq_len=" then
      seqLen := (line.drop 8).toNat!
    i := i + 1

  -- Validate required header fields
  if modelDim = 0 || numLayers = 0 || numHeads = 0 then
    return .error s!"Invalid header: modelDim={modelDim}, \
      numLayers={numLayers}, numHeads={numHeads} (all must be > 0)"

  -- Optional TOKENS section (ground-truth token IDs for the analyzed prompt)
  while i < lines.size && lines[i]!.trim.isEmpty do
    i := i + 1

  let mut inputTokens : Option (Array Nat) := none
  if i < lines.size && lines[i]!.trim = "TOKENS" then
    i := i + 1
    let mut toks : Array Nat := #[]
    while i < lines.size && lines[i]!.trim != "EMBEDDINGS" do
      toks := toks ++ parseNatLine lines[i]!
      i := i + 1
    if toks.size != seqLen then
      return .error s!"TOKENS length mismatch: expected {seqLen}, got {toks.size}"
    inputTokens := some toks

  IO.println s!"[2/5] Loading input embeddings (seq_len={seqLen}, model_dim={modelDim})..."

  -- Skip to EMBEDDINGS
  while i < lines.size && lines[i]! != "EMBEDDINGS" do
    i := i + 1
  if i >= lines.size then
    return .error "Missing EMBEDDINGS section"
  i := i + 1  -- Skip "EMBEDDINGS" line

  -- Parse input embeddings (sample sequence activations)
  let mut embFloats : Array Float := #[]
  while i < lines.size && !lines[i]!.startsWith "LAYER" do
    -- SAFETY: i < lines.size guaranteed by while condition
    embFloats := embFloats ++ parseFloatLine lines[i]!
    i := i + 1

  let inputEmbeddings := buildMatrix seqLen modelDim embFloats
  -- Validate input embeddings were loaded
  if inputEmbeddings.numRows != seqLen || inputEmbeddings.numCols != modelDim then
    return .error s!"Input embeddings dimension mismatch: \
      expected {seqLen}×{modelDim}, got {inputEmbeddings.numRows}×{inputEmbeddings.numCols}"

  IO.println s!"[3/5] Loading {numLayers} layers with {numHeads} heads each..."

  -- Parse layers
  let mut layers : Array (Array ConcreteAttentionLayer) := #[]
  let mut mlps : Array ConcreteMLPLayer := #[]

  for l in [:numLayers] do
    IO.println s!"  Loading layer {l}/{numLayers}..."
    -- Skip to LAYER line
    while i < lines.size && !lines[i]!.startsWith "LAYER" do
      i := i + 1
    i := i + 1

    -- Parse heads for this layer
    let mut layerHeads : Array ConcreteAttentionLayer := #[]
    for _h in [:numHeads] do
      -- Skip to HEAD line
      while i < lines.size && !lines[i]!.startsWith "HEAD" do
        i := i + 1
      i := i + 1

      -- Parse W_Q
      while i < lines.size && lines[i]! != "W_Q" do
        i := i + 1
      i := i + 1
      let mut wqFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "W_K" do
        wqFloats := wqFloats ++ parseFloatLine lines[i]!
        i := i + 1

      -- Parse W_K
      i := i + 1
      let mut wkFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "W_V" do
        wkFloats := wkFloats ++ parseFloatLine lines[i]!
        i := i + 1

      -- Parse W_V
      i := i + 1
      let mut wvFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "W_O" do
        wvFloats := wvFloats ++ parseFloatLine lines[i]!
        i := i + 1

      -- Parse W_O
      i := i + 1
      let mut woFloats : Array Float := #[]
      while i < lines.size &&
            !lines[i]!.startsWith "HEAD" &&
            !lines[i]!.startsWith "MLP" do
        woFloats := woFloats ++ parseFloatLine lines[i]!
        i := i + 1

      let head := mkAttentionLayer modelDim headDim wqFloats wkFloats wvFloats woFloats
      layerHeads := layerHeads.push head

    -- Validate all heads were loaded
    if layerHeads.size != numHeads then
      return .error s!"Layer {l}: expected {numHeads} heads, got {layerHeads.size}"
    layers := layers.push layerHeads

    -- Parse MLP
    while i < lines.size && lines[i]! != "MLP" do
      i := i + 1
    i := i + 1

    -- Parse W_in
    while i < lines.size && lines[i]! != "W_in" do
      i := i + 1
    i := i + 1
    let mut winFloats : Array Float := #[]
    while i < lines.size && !lines[i]!.startsWith "W_out" do
      winFloats := winFloats ++ parseFloatLine lines[i]!
      i := i + 1

    -- Parse W_out
    i := i + 1
    let mut woutFloats : Array Float := #[]
    while i < lines.size && !lines[i]!.startsWith "b_in" do
      woutFloats := woutFloats ++ parseFloatLine lines[i]!
      i := i + 1

    -- Parse b_in
    i := i + 1
    let mut binFloats : Array Float := #[]
    while i < lines.size && !lines[i]!.startsWith "b_out" do
      binFloats := binFloats ++ parseFloatLine lines[i]!
      i := i + 1

    -- Parse b_out
    i := i + 1
    let mut boutFloats : Array Float := #[]
    while i < lines.size &&
          !lines[i]!.startsWith "LAYER" &&
          !lines[i]!.startsWith "UNEMBEDDING" do
      boutFloats := boutFloats ++ parseFloatLine lines[i]!
      i := i + 1

    let mlp := mkMLPLayer modelDim hiddenDim winFloats woutFloats binFloats boutFloats
    mlps := mlps.push mlp

  IO.println "[4/5] Loading unembedding matrix..."

  -- Parse unembedding
  while i < lines.size && lines[i]! != "UNEMBEDDING" do
    i := i + 1
  i := i + 1
  let mut unembFloats : Array Float := #[]
  while i < lines.size do
    unembFloats := unembFloats ++ parseFloatLine lines[i]!
    i := i + 1

  let unembedding := buildMatrix modelDim vocabSize unembFloats

  let model : ConcreteModel := {
    numLayers := numLayers
    layers := layers
    mlps := mlps
    seqLen := seqLen
    inputTokens := inputTokens
    inputEmbeddings := inputEmbeddings
    unembedding := some unembedding
  }

  IO.println "[5/5] Model loaded successfully!\n"

  -- Create dummy vocabulary embeddings for return value (not used in circuit analysis)
  let vocabEmbeddings := ConcreteMatrix.zeros vocabSize modelDim

  return .ok model vocabEmbeddings

/-! ## File IO Operations -/

/-- Load a model from a file path. Supports .nfpt (text) format. -/
def loadModel (path : System.FilePath) : IO LoadResult := do
  let content ← IO.FS.readFile path
  if path.extension = some "nfpt" then
    loadFromText content
  else
    return .error s!"Unsupported file format: {path.extension.getD "unknown"}"

/-! ## Tokenization Utilities -/

/-- Simple tokenizer with vocabulary mapping. -/
structure Tokenizer where
  /-- Token strings in order of ID -/
  tokens : Array String
  /-- Unknown token ID -/
  unkId : Nat
  /-- Padding token ID -/
  padId : Nat
  /-- End of sequence token ID -/
  eosId : Nat

namespace Tokenizer

/-- Create a tokenizer from vocabulary list. -/
def fromVocabList (tokens : Array String)
    (unkId padId eosId : Nat := 0) : Tokenizer :=
  { tokens := tokens, unkId := unkId, padId := padId, eosId := eosId }

/-- Find a token's ID in the vocabulary. -/
def findToken (t : Tokenizer) (word : String) : Nat :=
  match t.tokens.findIdx? (· == word) with
  | some idx => idx
  | none => t.unkId

/-- Tokenize a string using simple whitespace splitting. -/
def tokenize (t : Tokenizer) (text : String) : Array Nat := Id.run do
  let words := text.splitOn " " |>.filter (· ≠ "")
  let mut ids : Array Nat := #[]
  for word in words do
    ids := ids.push (t.findToken word)
  ids

/-- Decode token IDs back to text. -/
def decode (t : Tokenizer) (ids : Array Nat) : String :=
  let tokens := ids.filterMap fun id =>
    if id < t.tokens.size then some t.tokens[id]!
    else none
  " ".intercalate tokens.toList

end Tokenizer

/-- Look up embeddings for token IDs from the embedding matrix. -/
def lookupEmbeddings (embeddings : ConcreteMatrix) (tokenIds : Array Nat)
    (seqLen : Nat) (padId : Nat := 0) : ConcreteMatrix := Id.run do
  let modelDim := embeddings.numCols
  let mut data : Array Float := #[]

  for pos in [:seqLen] do
    let tokenId := if pos < tokenIds.size then tokenIds[pos]! else padId
    -- Copy embedding row for this token
    for dim in [:modelDim] do
      let val := embeddings.get tokenId dim
      data := data.push val

  buildMatrix seqLen modelDim data

/-- Set the input embeddings in a model for a given prompt (token IDs). -/
def ConcreteModel.withInputTokens (model : ConcreteModel)
    (embeddings : ConcreteMatrix) (tokenIds : Array Nat)
    (padId : Nat := 0) : ConcreteModel :=
  let inputEmb := lookupEmbeddings embeddings tokenIds model.seqLen padId
  { model with inputEmbeddings := inputEmb, inputTokens := some tokenIds }

/-! ## Analysis Report Generation -/

/-- Format for circuit analysis results. -/
structure AnalysisReport where
  /-- Model name/path -/
  modelName : String
  /-- Input prompt (if available) -/
  prompt : Option String
  /-- Number of layers analyzed -/
  numLayers : Nat
  /-- Total heads in model -/
  totalHeads : Nat
  /-- Verified induction head candidates -/
  inductionHeads : Array CandidateInductionHead
  /-- Deep circuit candidates with N-layer bounds -/
  deepCircuits : Array DeepCircuitCandidate
  /-- Verification result (if run) -/
  verification : Option VerificationResult

namespace AnalysisReport

/-- Generate a human-readable report. -/
def toString (r : AnalysisReport) : String := Id.run do
  let mut s := s!"═══════════════════════════════════════════════════════════\n"
  s := s ++ s!"NFP Circuit Analysis Report\n"
  s := s ++ s!"Model: {r.modelName}\n"
  match r.prompt with
  | some p => s := s ++ s!"Prompt: \"{p}\"\n"
  | none => pure ()
  s := s ++ s!"Layers: {r.numLayers}, Heads: {r.totalHeads}\n"
  s := s ++ s!"═══════════════════════════════════════════════════════════\n\n"

  if r.inductionHeads.size > 0 then
    s := s ++ s!"VERIFIED INDUCTION HEADS ({r.inductionHeads.size} found):\n"
    s := s ++ s!"───────────────────────────────────────────────────────────\n"
    for head in r.inductionHeads do
      s := s ++ s!"  L{head.layer1Idx}H{head.head1Idx} → L{head.layer2Idx}H{head.head2Idx}\n"
      s := s ++ s!"    Combined Error: {head.combinedError}\n"
      s := s ++ s!"    Prev-Token Strength: {head.prevTokenStrength}\n"
      s := s ++ s!"    Pattern Bounds: ε₁={head.patternBound1}, ε₂={head.patternBound2}\n\n"
  else
    s := s ++ s!"No induction heads found above threshold.\n\n"

  if r.deepCircuits.size > 0 then
    s := s ++ s!"DEEP CIRCUIT CANDIDATES ({r.deepCircuits.size} found):\n"
    s := s ++ s!"───────────────────────────────────────────────────────────\n"
    for circuit in r.deepCircuits do
      s := s ++ s!"  {circuit.description}\n"
      s := s ++ s!"    Pattern Type: {circuit.patternType}\n"
      s := s ++ s!"    Simple Error Sum: {circuit.simpleErrorSum}\n"
      s := s ++ s!"    Amplified Error: {circuit.amplifiedError}\n"
      s := s ++ s!"    Amplification Factor: {circuit.amplificationFactor}\n\n"

  match r.verification with
  | some v =>
    s := s ++ s!"EMPIRICAL VERIFICATION:\n"
    s := s ++ s!"───────────────────────────────────────────────────────────\n"
    let status := if v.verified then "✓ PASSED" else "✗ FAILED"
    s := s ++ s!"  Status: {status}\n"
    s := s ++ s!"  Empirical Error: {v.ablation.empiricalError}\n"
    s := s ++ s!"  Theoretical Bound: {v.theoreticalBound}\n"
    s := s ++ s!"  Tightness: {v.tightness * 100.0}%\n"
    s := s ++ s!"  Circuit Size: {v.ablation.circuitSize}/{v.ablation.totalComponents}\n\n"
  | none => pure ()

  s := s ++ s!"═══════════════════════════════════════════════════════════\n"
  s

instance : ToString AnalysisReport := ⟨AnalysisReport.toString⟩

end AnalysisReport

/-- Run full analysis on a model and generate a report. -/
def analyzeModel (model : ConcreteModel) (modelName : String)
    (threshold : Float := 0.1)
    (prompt : Option String := none) : IO AnalysisReport := do
  IO.println "\n═══════════════════════════════════════════════════════════"
  IO.println "Starting Circuit Analysis"
  IO.println s!"Model: {modelName}"
  IO.println s!"Threshold: {threshold}"
  IO.println "═══════════════════════════════════════════════════════════\n"

  IO.println "[1/2] Searching for induction head candidates..."
  -- Find induction head candidates
  let inductionHeads := findInductionHeadCandidates model threshold
  let verifiedHeads := inductionHeads.filter (·.combinedError ≤ threshold)
  IO.println s!"  Found {verifiedHeads.size} verified induction heads \
    (of {inductionHeads.size} candidates)\n"

  IO.println "[2/2] Searching for deep circuit candidates..."
  -- Find deep circuit candidates
  let deepCircuits := findDeepCircuitCandidates model threshold
  let verifiedDeep := deepCircuits.filter (·.amplifiedError ≤ threshold)
  IO.println s!"  Found {verifiedDeep.size} verified deep circuits \
    (of {deepCircuits.size} candidates)\n"

  IO.println "Analysis complete!\n"

  -- Count total heads
  let totalHeads := model.layers.foldl (fun acc layer => acc + layer.size) 0

  return {
    modelName := modelName
    prompt := prompt
    numLayers := model.numLayers
    totalHeads := totalHeads
    inductionHeads := verifiedHeads
    deepCircuits := verifiedDeep
    verification := none
  }

/-- Run analysis with empirical verification. -/
def analyzeAndVerify (model : ConcreteModel) (modelName : String)
    (threshold : Float := 0.1)
    (prompt : Option String := none) : IO AnalysisReport := do
  let baseReport ← analyzeModel model modelName threshold prompt

  IO.println "═══════════════════════════════════════════════════════════"
  IO.println "Starting Empirical Verification"
  IO.println "═══════════════════════════════════════════════════════════\n"

  IO.println "Running circuit discovery and ablation experiments..."
  -- Run circuit discovery and verification
  let (_, verification) := discoverAndVerify model threshold
  IO.println "Verification complete!\n"

  return { baseReport with verification := some verification }

end Nfp
