-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.IO.Pure

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

## File Format: `.nfpt` (NFP_BINARY_V1)

Hybrid text header + binary body:

```
NFP_BINARY_V1
num_layers=12
num_heads=12
model_dim=768
head_dim=64
hidden_dim=3072
vocab_size=50257
seq_len=1024
BINARY_START
```

Binary payload (little-endian, row-major, no markers):
1. TOKENS: `seq_len` × Int32
2. EMBEDDINGS: `seq_len` × `model_dim` × Float64
3. For each layer (0..num_layers-1), for each head (0..num_heads-1):
   - W_Q (`model_dim`×`head_dim`), b_Q (`head_dim`)
   - W_K (`model_dim`×`head_dim`), b_K (`head_dim`)
   - W_V (`model_dim`×`head_dim`), b_V (`head_dim`)
   - W_O (`head_dim`×`model_dim`)
4. ATTN_BIAS: `model_dim`
5. MLP: W_in (`model_dim`×`hidden_dim`), b_in (`hidden_dim`),
        W_out (`hidden_dim`×`model_dim`), b_out (`model_dim`)
6. LN1 gamma/beta (`model_dim` each)
7. LN2 gamma/beta (`model_dim` each)
8. LN_F gamma/beta (`model_dim` each)
9. UNEMBEDDING: `model_dim`×`vocab_size`
```
-/

namespace Nfp

open IO

/-- Run an IO action and emit timing when `NFP_TIMING` is set. -/
def timeIt {α : Type} (label : String) (action : Unit → IO α) : IO α := do
  let timingEnabled ← IO.getEnv "NFP_TIMING"
  if timingEnabled.isNone then
    action ()
  else
    let t0 ← IO.monoNanosNow
    let result ← action ()
    let t1 ← IO.monoNanosNow
    let dtMs := (t1 - t0) / 1000000
    IO.eprintln s!"timing:{label} {dtMs}ms"
    return result

/-- Load a model from NFP text format content. -/
def loadFromText (_content : String) : IO LoadResult := do
  return .error "NFP_TEXT format is deprecated; use NFP_BINARY_V1"

/-! ## Binary `.nfpt` loading (NFP_BINARY_V1) -/

private def readLine? (h : IO.FS.Handle) : IO (Option String) := do
  let s ← h.getLine
  if s.isEmpty then
    return none
  else
    return some s

private def readExactly (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  if n = 0 then
    return ByteArray.empty
  let mut out : Array UInt8 := Array.replicate n 0
  let mut off : Nat := 0
  while off < n do
    let chunk ← h.read (USize.ofNat (n - off))
    if chunk.isEmpty then
      throw (IO.userError "unexpected EOF")
    for b in chunk.data do
      out := out.set! off b
      off := off + 1
  return ByteArray.mk out

@[inline] private def u32FromLE (b : ByteArray) (off : Nat) : UInt32 :=
  let b0 := (b[off]!).toUInt32
  let b1 := (b[off + 1]!).toUInt32
  let b2 := (b[off + 2]!).toUInt32
  let b3 := (b[off + 3]!).toUInt32
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

@[inline] private def u64FromLE (b : ByteArray) (off : Nat) : UInt64 :=
  let b0 := (b[off]!).toUInt64
  let b1 := (b[off + 1]!).toUInt64
  let b2 := (b[off + 2]!).toUInt64
  let b3 := (b[off + 3]!).toUInt64
  let b4 := (b[off + 4]!).toUInt64
  let b5 := (b[off + 5]!).toUInt64
  let b6 := (b[off + 6]!).toUInt64
  let b7 := (b[off + 7]!).toUInt64
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24) |||
    (b4 <<< 32) ||| (b5 <<< 40) ||| (b6 <<< 48) ||| (b7 <<< 56)

private def twoPow32 : Int := Int.ofNat (Nat.pow 2 32)

@[inline] private def i32FromLE (b : ByteArray) (off : Nat) : Int :=
  let u := u32FromLE b off
  let half : UInt32 := 0x80000000
  if u < half then
    Int.ofNat u.toNat
  else
    (Int.ofNat u.toNat) - twoPow32

@[inline] private def floatFromLE (b : ByteArray) (off : Nat) : Float :=
  Float.ofBits (u64FromLE b off)

private def readFloatArray (h : IO.FS.Handle) (count : Nat) : IO FloatArray := do
  if count = 0 then
    return FloatArray.empty
  let bytes ← readExactly h (count * 8)
  let mut data : Array Float := Array.replicate count 0.0
  let mut i : Nat := 0
  let mut off : Nat := 0
  while i < count do
    data := data.set! i (floatFromLE bytes off)
    off := off + 8
    i := i + 1
  return .mk data

private def readI32Array (h : IO.FS.Handle) (count : Nat) : IO (Array Nat) := do
  if count = 0 then
    return #[]
  let bytes ← readExactly h (count * 4)
  let mut out : Array Nat := Array.replicate count 0
  let mut i : Nat := 0
  let mut off : Nat := 0
  while i < count do
    let v := i32FromLE bytes off
    if v < 0 then
      throw (IO.userError s!"Negative token id at index {i}")
    out := out.set! i v.toNat
    off := off + 4
    i := i + 1
  return out

/-- Load a model from the `.nfpt` binary format (NFP_BINARY_V1). -/
def loadBinary (h : IO.FS.Handle) : IO LoadResult := do
  try
    let some magicLine ← readLine? h
      | return .error "Empty file"
    let magic := magicLine.trim
    if magic != "NFP_BINARY_V1" then
      return .error "Invalid magic: expected NFP_BINARY_V1"

    IO.println "[1/5] Parsing header..."

    let mut numLayers : Nat := 0
    let mut numHeads : Nat := 0
    let mut modelDim : Nat := 0
    let mut headDim : Nat := 0
    let mut hiddenDim : Nat := 0
    let mut vocabSize : Nat := 0
    let mut seqLen : Nat := 0

    let mut line? ← readLine? h
    while true do
      match line? with
      | none => return .error "Unexpected EOF while reading header"
      | some line =>
          let t := line.trim
          if t = "BINARY_START" then
            break
          if t.startsWith "num_layers=" then
            numLayers := (t.drop 11).toNat!
          else if t.startsWith "num_heads=" then
            numHeads := (t.drop 10).toNat!
          else if t.startsWith "model_dim=" then
            modelDim := (t.drop 10).toNat!
          else if t.startsWith "head_dim=" then
            headDim := (t.drop 9).toNat!
          else if t.startsWith "hidden_dim=" then
            hiddenDim := (t.drop 11).toNat!
          else if t.startsWith "vocab_size=" then
            vocabSize := (t.drop 11).toNat!
          else if t.startsWith "seq_len=" then
            seqLen := (t.drop 8).toNat!
          line? ← readLine? h

    if modelDim = 0 || numLayers = 0 || numHeads = 0 then
      return .error s!"Invalid header: modelDim={modelDim}, numLayers={numLayers},         numHeads={numHeads} (all must be > 0)"
    if headDim = 0 || hiddenDim = 0 || vocabSize = 0 || seqLen = 0 then
      return .error "Invalid header: headDim/hiddenDim/vocabSize/seqLen must be > 0"

    IO.println s!"[2/5] Loading input tokens + embeddings (seq_len={seqLen}, model_dim={modelDim})..."

    let inputTokens : Array Nat ← readI32Array h seqLen
    let embFloats ← readFloatArray h (seqLen * modelDim)
    let inputEmbeddings := buildMatrix seqLen modelDim embFloats.data

    IO.println s!"[3/5] Loading {numLayers} layers with {numHeads} heads each..."

    let mut layers : Array (Array ConcreteAttentionLayer) := Array.mkEmpty numLayers
    let mut attnProjBias : Array ConcreteMatrix := Array.mkEmpty numLayers
    let mut mlps : Array ConcreteMLPLayer := Array.mkEmpty numLayers
    let mut ln1 : Array ConcreteLayerNormParams := Array.mkEmpty numLayers
    let mut ln2 : Array ConcreteLayerNormParams := Array.mkEmpty numLayers

    for l in [:numLayers] do
      IO.println s!"  Loading layer {l}/{numLayers}..."
      let mut layerHeads : Array ConcreteAttentionLayer := Array.mkEmpty numHeads
      for _h in [:numHeads] do
        let wq ← readFloatArray h (modelDim * headDim)
        let bq ← readFloatArray h headDim
        let wk ← readFloatArray h (modelDim * headDim)
        let bk ← readFloatArray h headDim
        let wv ← readFloatArray h (modelDim * headDim)
        let bv ← readFloatArray h headDim
        let wo ← readFloatArray h (headDim * modelDim)
        let head := mkAttentionLayer modelDim headDim wq.data wk.data wv.data wo.data bq.data bk.data bv.data
        layerHeads := layerHeads.push head
      layers := layers.push layerHeads

      let bias ← readFloatArray h modelDim
      attnProjBias := attnProjBias.push (buildMatrix 1 modelDim bias.data)

      let win ← readFloatArray h (modelDim * hiddenDim)
      let bin ← readFloatArray h hiddenDim
      let wout ← readFloatArray h (hiddenDim * modelDim)
      let bout ← readFloatArray h modelDim
      mlps := mlps.push (mkMLPLayer modelDim hiddenDim win.data wout.data bin.data bout.data)

      let ln1Gamma ← readFloatArray h modelDim
      let ln1Beta ← readFloatArray h modelDim
      ln1 := ln1.push {
        gamma := buildMatrix 1 modelDim ln1Gamma.data
        beta := buildMatrix 1 modelDim ln1Beta.data
      }

      let ln2Gamma ← readFloatArray h modelDim
      let ln2Beta ← readFloatArray h modelDim
      ln2 := ln2.push {
        gamma := buildMatrix 1 modelDim ln2Gamma.data
        beta := buildMatrix 1 modelDim ln2Beta.data
      }

    IO.println "[4/5] Loading final layernorm + unembedding..."

    let lnfGamma ← readFloatArray h modelDim
    let lnfBeta ← readFloatArray h modelDim
    let lnf := {
      gamma := buildMatrix 1 modelDim lnfGamma.data
      beta := buildMatrix 1 modelDim lnfBeta.data
    }

    let unembFloats ← readFloatArray h (modelDim * vocabSize)
    let unembedding := buildMatrix modelDim vocabSize unembFloats.data

    let model : ConcreteModel := {
      numLayers := numLayers
      layers := layers
      attnProjBias := attnProjBias
      mlps := mlps
      ln1 := ln1
      ln2 := ln2
      lnf := lnf
      seqLen := seqLen
      inputTokens := some inputTokens
      inputEmbeddings := inputEmbeddings
      unembedding := some unembedding
    }

    IO.println "[5/5] Model loaded successfully!
"
    return .ok model
  catch e =>
    return .error s!"Binary load failed: {e}"

/-- Input tokens + embeddings loaded from a binary `.nfpt` file. -/
structure InputBinary where
  /-- Sequence length parsed from the input header. -/
  seqLen : Nat
  /-- Model dimension parsed from the input header. -/
  modelDim : Nat
  /-- Token IDs parsed from the input file. -/
  tokens : Array Nat
  /-- Input embeddings (seqLen × modelDim). -/
  embeddings : ConcreteMatrix

/-- Load input tokens + embeddings from a binary `.nfpt` file. -/
def loadInputBinary (path : System.FilePath) : IO (Except String InputBinary) := do
  try
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let some magicLine ← readLine? h
      | return .error "Empty input file"
    let magic := magicLine.trim
    if magic != "NFP_BINARY_V1" then
      return .error "Invalid input magic: expected NFP_BINARY_V1"
    let mut seqLen? : Option Nat := none
    let mut modelDim? : Option Nat := none
    let mut line? ← readLine? h
    while true do
      match line? with
      | none => return .error "Unexpected EOF while reading input header"
      | some line =>
          let t := line.trim
          if t = "BINARY_START" then
            break
          if t.startsWith "seq_len=" then
            match (t.drop 8).toNat? with
            | some n => seqLen? := some n
            | none => return .error "Invalid seq_len in input header"
          else if t.startsWith "model_dim=" then
            match (t.drop 10).toNat? with
            | some n => modelDim? := some n
            | none => return .error "Invalid model_dim in input header"
          line? ← readLine? h
    let some seqLen := seqLen?
      | return .error "Missing seq_len in input header"
    let some modelDim := modelDim?
      | return .error "Missing model_dim in input header"
    let tokens ← readI32Array h seqLen
    let embFloats ← readFloatArray h (seqLen * modelDim)
    let embeddings := buildMatrix seqLen modelDim embFloats.data
    return .ok { seqLen := seqLen, modelDim := modelDim, tokens := tokens, embeddings := embeddings }
  catch e =>
    return .error s!"Binary input load failed: {e}"
/-! ## File IO Operations -/

/-- Load a model from a file path. Supports .nfpt (binary) format. -/
def loadModel (path : System.FilePath) : IO LoadResult := do
  if path.extension = some "nfpt" then
    timeIt "io:load-model" (fun () =>
      IO.FS.withFile path .read fun h =>
        loadBinary h)
  else
    return .error s!"Unsupported file format: {path.extension.getD "unknown"}"

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
      s := s ++ s!"    Induction Score: {head.inductionScore}\n"
      s := s ++ s!"    K-Composition: {head.kComp}\n"
      s := s ++ s!"    Faithfulness Ratios: ε₁={head.patternBound1}, ε₂={head.patternBound2}\n\n"
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

  IO.println "[1/2] Building precomputed cache..."
  let cache ← timeIt "analysis:precompute-cache" (fun () =>
    pure <| PrecomputedCache.build model)

  IO.println "[2/2] Searching for deep circuit candidates (shared scan)..."
  -- Find deep circuit candidates (reuse cache)
  let deepCircuits ← timeIt "analysis:deep-circuit-scan" (fun () =>
    pure <| findDeepCircuitCandidatesFromCache cache)
  let verifiedDeep := deepCircuits.filter (·.amplifiedError ≤ threshold)
  IO.println s!"  Found {verifiedDeep.size} verified deep circuits \
    (of {deepCircuits.size} candidates)"

  -- Derive induction-head candidates from the same scan to avoid repeating
  -- the expensive `checkInductionPattern` computation.
  let inductionHeads ← timeIt "analysis:induction-candidates" (fun () =>
    pure <|
      (deepCircuits.filterMap (·.toInductionCandidate? cache)).qsort
        (·.combinedError < ·.combinedError))
  let verifiedHeads := inductionHeads.filter (·.combinedError ≤ threshold)
  IO.println s!"  Found {verifiedHeads.size} verified induction heads \
    (of {inductionHeads.size} candidates)\n"

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
  let (_, verification) ← timeIt "analysis:discover-and-verify" (fun () =>
    pure <| discoverAndVerify model threshold)
  IO.println "Verification complete!\n"

  return { baseReport with verification := some verification }

end Nfp
