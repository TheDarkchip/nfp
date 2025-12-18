-- SPDX-License-Identifier: AGPL-3.0-or-later

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

/-! ## Float Parsing Utilities -/

private def pow10PowTable : Array Float := Id.run do
  -- Precompute `Float.pow 10.0 k` for k=0..308 so we avoid calling `Float.pow` per token.
  let mut out : Array Float := Array.mkEmpty 309
  for k in [:309] do
    out := out.push (Float.pow 10.0 k.toFloat)
  out

private def pow10Pow (n : Nat) : Float :=
  if n < pow10PowTable.size then
    pow10PowTable[n]!
  else
    Float.pow 10.0 n.toFloat

private def parseFloatRange (s : String) (start stop : String.Pos.Raw) : Option Float := Id.run do
  -- This is a faster, allocation-free version of the previous `parseFloat`, but it preserves
  -- the exact Float computation structure (Nat parsing + `Float.pow`) to keep results stable.

  let parseNatRange (s : String) (start stop : String.Pos.Raw) : Option Nat := Id.run do
    let mut p := start
    if p ≥ stop then
      return none
    let mut acc : Nat := 0
    let mut saw : Bool := false
    while p < stop do
      let c := p.get s
      if ('0' ≤ c) && (c ≤ '9') then
        acc := acc * 10 + (c.toNat - '0'.toNat)
        saw := true
        p := p.next s
      else
        return none
    if saw then some acc else none

  let mut p := start
  if p ≥ stop then
    return none

  let mut negative := false
  let c0 := p.get s
  if c0 = '-' then
    negative := true
    p := p.next s
  else if c0 = '+' then
    p := p.next s

  if p ≥ stop then
    return none

  -- Find exponent marker the same way as the old parser: accept exactly one `e` if present,
  -- otherwise accept exactly one `E`.
  let mut ePos : Option String.Pos.Raw := none
  let mut eCount : Nat := 0
  let mut EPos : Option String.Pos.Raw := none
  let mut ECount : Nat := 0
  let mut q := p
  while q < stop do
    let c := q.get s
    if c = 'e' then
      eCount := eCount + 1
      if eCount = 1 then ePos := some q
    else if c = 'E' then
      ECount := ECount + 1
      if ECount = 1 then EPos := some q
    q := q.next s

  let expMarker? : Option String.Pos.Raw :=
    if eCount = 1 then ePos else if ECount = 1 then EPos else none

  let mantEnd : String.Pos.Raw :=
    match expMarker? with
    | some ep => ep
    | none => stop

  -- Find decimal point in mantissa (must be 0 or 1 occurrences).
  let mut dotPos : Option String.Pos.Raw := none
  let mut dotCount : Nat := 0
  let mut r := p
  while r < mantEnd do
    if r.get s = '.' then
      dotCount := dotCount + 1
      if dotCount = 1 then dotPos := some r
    r := r.next s
  if dotCount > 1 then
    return none

  let (intStart, intStop, fracStart?, fracStop) :=
    match dotPos with
    | none => (p, mantEnd, none, mantEnd)
    | some dp => (p, dp, some (dp.next s), mantEnd)

  let intN? : Option Nat :=
    if dotPos.isSome && intStart = intStop then
      some 0
    else
      parseNatRange s intStart intStop

  let fracN? : Option Nat :=
    match fracStart? with
    | none => none
    | some fs =>
        if fs = fracStop then some 0 else parseNatRange s fs fracStop

  let mantissa? : Option Float :=
    match dotPos, intN?, fracN? with
    | none, some iN, _ =>
        some iN.toFloat
    | some _, some iN, some fN =>
        let fracLen := (fracStop.byteIdx - (fracStart?.getD fracStop).byteIdx)
        let divisor := pow10Pow fracLen
        some (iN.toFloat + fN.toFloat / divisor)
    | some _, _, none =>
        -- `.` present but no fractional parse (shouldn't happen), treat as invalid.
        none
    | _, none, _ => none

  let some mantissa := mantissa? | return none

  let value : Float :=
    match expMarker? with
    | none => mantissa
    | some ep =>
        let expStart := ep.next s
        if expStart ≥ stop then
          mantissa
        else
          -- Parse exponent, but if it is malformed, ignore it (old behavior).
          let c := expStart.get s
          let (expNeg, es) :=
            if c = '-' then (true, expStart.next s)
            else if c = '+' then (false, expStart.next s)
            else (false, expStart)
          match parseNatRange s es stop with
          | none => mantissa
          | some eNat =>
              let p10 := pow10Pow eNat
              if expNeg then mantissa / p10 else mantissa * p10

  some (if negative then -value else value)

/-- Parse a floating point number from a string. -/
def parseFloat (s : String) : Option Float := Id.run do
  let s := s.trim
  if s.isEmpty then
    none
  else
    parseFloatRange s 0 s.endPos

private def appendFloatsFromLine (line : String) (acc : Array Float) : Array Float := Id.run do
  let mut out := acc
  let s := line
  let mut p : String.Pos.Raw := 0
  let endPos := s.endPos
  let isWs (c : Char) : Bool :=
    c = ' ' || c = '\t' || c = '\n' || c = '\r'
  while p < endPos do
    while p < endPos && isWs (p.get s) do
      p := p.next s
    let start := p
    while p < endPos && !isWs (p.get s) do
      p := p.next s
    if start < p then
      match parseFloatRange s start p with
      | some x => out := out.push x
      | none => pure ()
  out

private def parseFloatsFromLines (lines : Array String) (cap : Nat := 0) : Array Float :=
  Id.run do
    let mut out : Array Float := Array.mkEmpty cap
    for line in lines do
      out := appendFloatsFromLine line out
    out

private def spawnParseFloats (lines : Array String) (cap : Nat := 0) : Task (Array Float) :=
  Task.spawn (fun _ => parseFloatsFromLines lines cap)

/-- Parse a line of space-separated floats. -/
def parseFloatLine (line : String) : Array Float :=
  appendFloatsFromLine line #[]

/-! ## Nat Parsing Utilities -/

/-- Parse a line of space-separated natural numbers. -/
private def parseNatRange (s : String) (start stop : String.Pos.Raw) : Option Nat := Id.run do
  let mut p := start
  if p ≥ stop then
    return none
  let mut acc : Nat := 0
  let mut saw : Bool := false
  while p < stop do
    let c := p.get s
    if ('0' ≤ c) && (c ≤ '9') then
      acc := acc * 10 + (c.toNat - '0'.toNat)
      saw := true
      p := p.next s
    else
      return none
  if saw then some acc else none

private def appendNatsFromLine (line : String) (acc : Array Nat) : Array Nat := Id.run do
  let mut out := acc
  let s := line
  let mut p : String.Pos.Raw := 0
  let endPos := s.endPos
  let isWs (c : Char) : Bool :=
    c = ' ' || c = '\t' || c = '\n' || c = '\r'
  while p < endPos do
    while p < endPos && isWs (p.get s) do
      p := p.next s
    let start := p
    while p < endPos && !isWs (p.get s) do
      p := p.next s
    if start < p then
      match parseNatRange s start p with
      | some n => out := out.push n
      | none => pure ()
  out

def parseNatLine (line : String) : Array Nat :=
  appendNatsFromLine line #[]

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
  | ok (model : ConcreteModel)
  | error (msg : String)

namespace LoadResult

def isOk : LoadResult → Bool
  | ok _ => true
  | error _ => false

def getModel : LoadResult → Option ConcreteModel
  | ok m => some m
  | error _ => none

def getError : LoadResult → Option String
  | ok _ => none
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
    (wq wk wv wo bq bk bv : Array Float) : ConcreteAttentionLayer :=
  let wQ := buildMatrix modelDim headDim wq
  let bQ := buildMatrix 1 headDim bq
  let wK := buildMatrix modelDim headDim wk
  let bK := buildMatrix 1 headDim bk
  let wV := buildMatrix modelDim headDim wv
  let bV := buildMatrix 1 headDim bv
  let wO := buildMatrix headDim modelDim wo
  {
    modelDim := modelDim
    headDim := headDim
    W_Q := wQ
    b_Q := bQ
    W_K := wK
    b_K := bK
    W_V := wV
    b_V := bV
    W_O := wO
    W_Q_dims := ⟨rfl, rfl⟩
    b_Q_dims := ⟨rfl, rfl⟩
    W_K_dims := ⟨rfl, rfl⟩
    b_K_dims := ⟨rfl, rfl⟩
    W_V_dims := ⟨rfl, rfl⟩
    b_V_dims := ⟨rfl, rfl⟩
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
  let mut out := ByteArray.empty
  while out.size < n do
    let chunk ← h.read (USize.ofNat (n - out.size))
    if chunk.isEmpty then
      throw (IO.userError "unexpected EOF")
    out := out ++ chunk
  return out

private def u32FromLE (b : ByteArray) (off : Nat) : UInt32 :=
  let b0 := (b[off]!).toUInt32
  let b1 := (b[off + 1]!).toUInt32
  let b2 := (b[off + 2]!).toUInt32
  let b3 := (b[off + 3]!).toUInt32
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

private def u64FromLE (b : ByteArray) (off : Nat) : UInt64 :=
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

private def i32FromLE (b : ByteArray) (off : Nat) : Int :=
  let u := u32FromLE b off
  let half : UInt32 := 0x80000000
  if u < half then
    Int.ofNat u.toNat
  else
    let two32 : Int := Int.ofNat (Nat.pow 2 32)
    (Int.ofNat u.toNat) - two32

private def floatFromLE (b : ByteArray) (off : Nat) : Float :=
  Float.ofBits (u64FromLE b off)

private def readFloatArray (h : IO.FS.Handle) (count : Nat) : IO FloatArray := do
  if count = 0 then
    return FloatArray.empty
  let bytes ← readExactly h (count * 8)
  let data := Array.ofFn (fun i : Fin count =>
    floatFromLE bytes (i.val * 8))
  return .mk data

private def readI32Array (h : IO.FS.Handle) (count : Nat) : IO (Array Nat) := do
  if count = 0 then
    return #[]
  let bytes ← readExactly h (count * 4)
  let mut out : Array Nat := Array.mkEmpty count
  for i in [:count] do
    let v := i32FromLE bytes (i * 4)
    if v < 0 then
      throw (IO.userError s!"Negative token id at index {i}")
    out := out.push v.toNat
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

    let mut layers : Array (Array ConcreteAttentionLayer) := #[]
    let mut attnProjBias : Array ConcreteMatrix := #[]
    let mut mlps : Array ConcreteMLPLayer := #[]
    let mut ln1 : Array ConcreteLayerNormParams := #[]
    let mut ln2 : Array ConcreteLayerNormParams := #[]

    for l in [:numLayers] do
      IO.println s!"  Loading layer {l}/{numLayers}..."
      let mut layerHeads : Array ConcreteAttentionLayer := #[]
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
/-! ## File IO Operations -/

/-- Load a model from a file path. Supports .nfpt (binary) format. -/
def loadModel (path : System.FilePath) : IO LoadResult := do
  if path.extension = some "nfpt" then
    IO.FS.withFile path .read fun h =>
      loadBinary h
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
  let cache := PrecomputedCache.build model

  IO.println "[2/2] Searching for deep circuit candidates (shared scan)..."
  -- Find deep circuit candidates (reuse cache)
  let deepCircuits := findDeepCircuitCandidatesFromCache cache
  let verifiedDeep := deepCircuits.filter (·.amplifiedError ≤ threshold)
  IO.println s!"  Found {verifiedDeep.size} verified deep circuits \
    (of {deepCircuits.size} candidates)"

  -- Derive induction-head candidates from the same scan to avoid repeating
  -- the expensive `checkInductionPattern` computation.
  let inductionHeads :=
    (deepCircuits.filterMap (·.toInductionCandidate? cache)).qsort
      (·.combinedError < ·.combinedError)
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
  let (_, verification) := discoverAndVerify model threshold
  IO.println "Verification complete!\n"

  return { baseReport with verification := some verification }

end Nfp
