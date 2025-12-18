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

## File Format: `.nfpt` (NFP Text Format)

Human-readable text format for debugging and small models:

```
NFP_TEXT_V2
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
b_Q
<floats>  -- length = head_dim (optional; default 0)
W_K
<floats>
b_K
<floats>  -- length = head_dim (optional; default 0)
W_V
<floats>
b_V
<floats>  -- length = head_dim (optional; default 0)
...
ATTN_BIAS
<floats>  -- length = model_dim (optional; default 0)

MLP
W_in
<floats>
W_out
<floats>
b_in
<floats>
b_out
<floats>

LN1_GAMMA
<floats>  -- length = model_dim
LN1_BETA
<floats>  -- length = model_dim
LN2_GAMMA
<floats>  -- length = model_dim
LN2_BETA
<floats>  -- length = model_dim

LN_F_GAMMA
<floats>  -- length = model_dim
LN_F_BETA
<floats>  -- length = model_dim

UNEMBEDDING
<floats>  -- model_dim × vocab_size, row-major
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
def loadFromText (content : String) : IO LoadResult := do
  let lines := content.splitOn "\n" |>.toArray
  if lines.size = 0 then return .error "Empty file"

  -- Check magic - SAFETY: lines.size > 0 guaranteed above
  let magic := lines[0]!.trim
  let isV1 := magic = "NFP_TEXT_V1"
  let isV2 := magic = "NFP_TEXT_V2"
  if !(isV1 || isV2) then
    return .error "Invalid magic: expected NFP_TEXT_V1 or NFP_TEXT_V2"

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
      for t in parseNatLine lines[i]! do
        toks := toks.push t
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
    for x in parseFloatLine lines[i]! do
      embFloats := embFloats.push x
    i := i + 1

  let inputEmbeddings := buildMatrix seqLen modelDim embFloats
  -- Validate input embeddings were loaded
  if inputEmbeddings.numRows != seqLen || inputEmbeddings.numCols != modelDim then
    return .error s!"Input embeddings dimension mismatch: \
      expected {seqLen}×{modelDim}, got {inputEmbeddings.numRows}×{inputEmbeddings.numCols}"

  IO.println s!"[3/5] Loading {numLayers} layers with {numHeads} heads each..."

  -- Parse layers
  let mut layers : Array (Array ConcreteAttentionLayer) := #[]
  let mut attnProjBias : Array ConcreteMatrix := #[]
  let mut mlps : Array ConcreteMLPLayer := #[]
  let mut ln1 : Array ConcreteLayerNormParams := #[]
  let mut ln2 : Array ConcreteLayerNormParams := #[]

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
      while i < lines.size &&
            !lines[i]!.startsWith "b_Q" &&
            !lines[i]!.startsWith "W_K" do
        for x in parseFloatLine lines[i]! do
          wqFloats := wqFloats.push x
        i := i + 1
      let mut bqFloats : Array Float := #[]
      if i < lines.size && lines[i]! = "b_Q" then
        i := i + 1
        while i < lines.size && !lines[i]!.startsWith "W_K" do
          for x in parseFloatLine lines[i]! do
            bqFloats := bqFloats.push x
          i := i + 1

      -- Parse W_K
      i := i + 1
      let mut wkFloats : Array Float := #[]
      while i < lines.size &&
            !lines[i]!.startsWith "b_K" &&
            !lines[i]!.startsWith "W_V" do
        for x in parseFloatLine lines[i]! do
          wkFloats := wkFloats.push x
        i := i + 1
      let mut bkFloats : Array Float := #[]
      if i < lines.size && lines[i]! = "b_K" then
        i := i + 1
        while i < lines.size && !lines[i]!.startsWith "W_V" do
          for x in parseFloatLine lines[i]! do
            bkFloats := bkFloats.push x
          i := i + 1

      -- Parse W_V
      i := i + 1
      let mut wvFloats : Array Float := #[]
      while i < lines.size &&
            !lines[i]!.startsWith "b_V" &&
            !lines[i]!.startsWith "W_O" do
        for x in parseFloatLine lines[i]! do
          wvFloats := wvFloats.push x
        i := i + 1
      let mut bvFloats : Array Float := #[]
      if i < lines.size && lines[i]! = "b_V" then
        i := i + 1
        while i < lines.size && !lines[i]!.startsWith "W_O" do
          for x in parseFloatLine lines[i]! do
            bvFloats := bvFloats.push x
          i := i + 1

      -- Parse W_O
      i := i + 1
      let mut woFloats : Array Float := #[]
      while i < lines.size &&
            !lines[i]!.startsWith "HEAD" &&
            !lines[i]!.startsWith "ATTN_BIAS" &&
            !lines[i]!.startsWith "MLP" do
        for x in parseFloatLine lines[i]! do
          woFloats := woFloats.push x
        i := i + 1

      let head := mkAttentionLayer modelDim headDim wqFloats wkFloats wvFloats woFloats bqFloats bkFloats bvFloats
      layerHeads := layerHeads.push head

    -- Validate all heads were loaded
    if layerHeads.size != numHeads then
      return .error s!"Layer {l}: expected {numHeads} heads, got {layerHeads.size}"
    layers := layers.push layerHeads

    -- Optional attention projection bias (c_proj.bias), applied once after combining heads.
    let mut layerAttnBias : ConcreteMatrix := ConcreteMatrix.zeros 1 modelDim
    if i < lines.size && lines[i]! = "ATTN_BIAS" then
      i := i + 1
      let mut biasFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "MLP" do
        for x in parseFloatLine lines[i]! do
          biasFloats := biasFloats.push x
        i := i + 1
      layerAttnBias := buildMatrix 1 modelDim biasFloats
    attnProjBias := attnProjBias.push layerAttnBias

    -- Parse MLP
    while i < lines.size && lines[i]! != "MLP" do
      i := i + 1
    i := i + 1

    -- Parse W_in
    while i < lines.size && lines[i]! != "W_in" do
      i := i + 1
    i := i + 1
    let mut winFloats : Array Float := #[]
    -- Support both MLP orderings:
    -- (1) W_in, W_out, b_in, b_out  (legacy)
    -- (2) W_in, b_in, W_out, b_out  (GPT-2 Conv1D export)
    while i < lines.size &&
          !lines[i]!.startsWith "W_out" &&
          !lines[i]!.startsWith "b_in" do
      for x in parseFloatLine lines[i]! do
        winFloats := winFloats.push x
      i := i + 1

    if i >= lines.size then
      return .error s!"Layer {l}: incomplete MLP section after W_in"

    let mut woutFloats : Array Float := Array.mkEmpty (hiddenDim * modelDim)
    let mut binFloats : Array Float := Array.mkEmpty hiddenDim

    if lines[i]! = "b_in" then
      -- Ordering (2): parse b_in then W_out.
      i := i + 1
      while i < lines.size && !lines[i]!.startsWith "W_out" do
        for x in parseFloatLine lines[i]! do
          binFloats := binFloats.push x
        i := i + 1

      if i >= lines.size then
        return .error s!"Layer {l}: missing W_out after b_in"

      i := i + 1
      while i < lines.size && !lines[i]!.startsWith "b_out" do
        for x in parseFloatLine lines[i]! do
          woutFloats := woutFloats.push x
        i := i + 1
    else
      -- Ordering (1): parse W_out then b_in.
      if lines[i]! != "W_out" then
        return .error s!"Layer {l}: expected W_out or b_in after W_in, got {lines[i]!}"

      i := i + 1
      while i < lines.size && !lines[i]!.startsWith "b_in" do
        for x in parseFloatLine lines[i]! do
          woutFloats := woutFloats.push x
        i := i + 1

      if i >= lines.size then
        return .error s!"Layer {l}: missing b_in after W_out"

      i := i + 1
      while i < lines.size && !lines[i]!.startsWith "b_out" do
        for x in parseFloatLine lines[i]! do
          binFloats := binFloats.push x
        i := i + 1

    -- Parse b_out
    i := i + 1
    let mut boutFloats : Array Float := Array.mkEmpty modelDim
    while i < lines.size &&
          (if isV2 then !lines[i]!.startsWith "LN1_GAMMA"
           else !lines[i]!.startsWith "LAYER" && !lines[i]!.startsWith "UNEMBEDDING") do
      for x in parseFloatLine lines[i]! do
        boutFloats := boutFloats.push x
      i := i + 1

    let mlp := mkMLPLayer modelDim hiddenDim winFloats woutFloats binFloats boutFloats
    mlps := mlps.push mlp

    -- Parse LayerNorm params (Pre-LN)
    if isV2 then
      while i < lines.size && lines[i]!.trim.isEmpty do
        i := i + 1

      -- LN1_GAMMA
      while i < lines.size && lines[i]! != "LN1_GAMMA" do
        i := i + 1
      if i >= lines.size then
        return .error s!"Layer {l}: missing LN1_GAMMA section"
      i := i + 1
      let mut ln1GammaFloats : Array Float := Array.mkEmpty modelDim
      while i < lines.size && !lines[i]!.startsWith "LN1_BETA" do
        for x in parseFloatLine lines[i]! do
          ln1GammaFloats := ln1GammaFloats.push x
        i := i + 1

      -- LN1_BETA
      i := i + 1
      let mut ln1BetaFloats : Array Float := Array.mkEmpty modelDim
      while i < lines.size && !lines[i]!.startsWith "LN2_GAMMA" do
        for x in parseFloatLine lines[i]! do
          ln1BetaFloats := ln1BetaFloats.push x
        i := i + 1

      -- LN2_GAMMA
      i := i + 1
      let mut ln2GammaFloats : Array Float := Array.mkEmpty modelDim
      while i < lines.size && !lines[i]!.startsWith "LN2_BETA" do
        for x in parseFloatLine lines[i]! do
          ln2GammaFloats := ln2GammaFloats.push x
        i := i + 1

      -- LN2_BETA
      i := i + 1
      let mut ln2BetaFloats : Array Float := Array.mkEmpty modelDim
      while i < lines.size &&
            !lines[i]!.startsWith "LAYER" &&
            !lines[i]!.startsWith "LN_F_GAMMA" do
        for x in parseFloatLine lines[i]! do
          ln2BetaFloats := ln2BetaFloats.push x
        i := i + 1

      let ln1Params : ConcreteLayerNormParams := {
        gamma := buildMatrix 1 modelDim ln1GammaFloats
        beta := buildMatrix 1 modelDim ln1BetaFloats
      }
      let ln2Params : ConcreteLayerNormParams := {
        gamma := buildMatrix 1 modelDim ln2GammaFloats
        beta := buildMatrix 1 modelDim ln2BetaFloats
      }
      ln1 := ln1.push ln1Params
      ln2 := ln2.push ln2Params
    else
      -- V1: no LayerNorm params present; treat as identity.
      ln1 := ln1.push (ConcreteLayerNormParams.identity modelDim)
      ln2 := ln2.push (ConcreteLayerNormParams.identity modelDim)

  IO.println "[4/5] Loading unembedding matrix..."

  let mut lnf : ConcreteLayerNormParams := ConcreteLayerNormParams.identity modelDim
  if isV2 then
    -- Parse final LayerNorm params (ln_f)
    while i < lines.size && lines[i]!.trim.isEmpty do
      i := i + 1

    while i < lines.size && lines[i]! != "LN_F_GAMMA" do
      i := i + 1
    if i >= lines.size then
      return .error "Missing LN_F_GAMMA section"
    i := i + 1
    let mut lnfGammaFloats : Array Float := Array.mkEmpty modelDim
    while i < lines.size && !lines[i]!.startsWith "LN_F_BETA" do
      for x in parseFloatLine lines[i]! do
        lnfGammaFloats := lnfGammaFloats.push x
      i := i + 1

    i := i + 1
    let mut lnfBetaFloats : Array Float := Array.mkEmpty modelDim
    while i < lines.size && !lines[i]!.startsWith "UNEMBEDDING" do
      for x in parseFloatLine lines[i]! do
        lnfBetaFloats := lnfBetaFloats.push x
      i := i + 1

    lnf := {
      gamma := buildMatrix 1 modelDim lnfGammaFloats
      beta := buildMatrix 1 modelDim lnfBetaFloats
    }

  -- Parse unembedding
  while i < lines.size && lines[i]! != "UNEMBEDDING" do
    i := i + 1
  i := i + 1
  let mut unembFloats : Array Float := #[]
  while i < lines.size do
    for x in parseFloatLine lines[i]! do
      unembFloats := unembFloats.push x
    i := i + 1

  let unembedding := buildMatrix modelDim vocabSize unembFloats

  let model : ConcreteModel := {
    numLayers := numLayers
    layers := layers
    attnProjBias := attnProjBias
    mlps := mlps
    ln1 := ln1
    ln2 := ln2
    lnf := lnf
    seqLen := seqLen
    inputTokens := inputTokens
    inputEmbeddings := inputEmbeddings
    unembedding := some unembedding
  }

  IO.println "[5/5] Model loaded successfully!\n"
  return .ok model

/-! ## Streaming `.nfpt` loading (faster) -/

private def readLine? (h : IO.FS.Handle) : IO (Option String) := do
  let s ← h.getLine
  if s.isEmpty then
    return none
  else
    return some s

private partial def skipUntil (h : IO.FS.Handle) (line? : Option String) (p : String → Bool) :
    IO (Option String) := do
  match line? with
  | none => return none
  | some line =>
      if p line then
        return line?
      else
        let next ← readLine? h
        skipUntil h next p

/-- Load a model from the `.nfpt` text format using streaming `getLine`.

This avoids `content.splitOn "\n"` (large intermediate allocations) and preserves the existing
parsing semantics by following the same markers/section structure.
-/
def loadFromTextHandle (h : IO.FS.Handle) : IO LoadResult := do
  let some magicLine ← readLine? h
    | return .error "Empty file"
  let magic := magicLine.trim
  let isV1 := magic = "NFP_TEXT_V1"
  let isV2 := magic = "NFP_TEXT_V2"
  if !(isV1 || isV2) then
    return .error "Invalid magic: expected NFP_TEXT_V1 or NFP_TEXT_V2"

  IO.println "[1/5] Parsing header..."

  let mut numLayers : Nat := 0
  let mut numHeads : Nat := 0
  let mut modelDim : Nat := 0
  let mut headDim : Nat := 0
  let mut hiddenDim : Nat := 0
  let mut vocabSize : Nat := 0
  let mut seqLen : Nat := 0

  let mut line? ← readLine? h
  -- Header lines end at TOKENS or EMBEDDINGS.
  while true do
    match line? with
    | none => break
    | some line =>
        let t := line.trim
        if t = "TOKENS" || line.startsWith "EMBEDDINGS" then
          break
        if line.startsWith "num_layers=" then
          numLayers := (line.drop 11 |>.trim).toNat!
        else if line.startsWith "num_heads=" then
          numHeads := (line.drop 10 |>.trim).toNat!
        else if line.startsWith "model_dim=" then
          modelDim := (line.drop 10 |>.trim).toNat!
        else if line.startsWith "head_dim=" then
          headDim := (line.drop 9 |>.trim).toNat!
        else if line.startsWith "hidden_dim=" then
          hiddenDim := (line.drop 11 |>.trim).toNat!
        else if line.startsWith "vocab_size=" then
          vocabSize := (line.drop 11 |>.trim).toNat!
        else if line.startsWith "seq_len=" then
          seqLen := (line.drop 8 |>.trim).toNat!
        line? ← readLine? h

  if modelDim = 0 || numLayers = 0 || numHeads = 0 then
    return .error s!"Invalid header: modelDim={modelDim}, numLayers={numLayers}, \
      numHeads={numHeads} (all must be > 0)"

  -- Skip blank lines.
  while true do
    match line? with
    | some line =>
        if line.trim.isEmpty then
          line? ← readLine? h
        else
          break
    | none => break

  -- Optional TOKENS section
  let mut inputTokens : Option (Array Nat) := none
  if line?.map (·.trim) = some "TOKENS" then
    let mut toks : Array Nat := #[]
    line? ← readLine? h
    while line?.map (·.trim) != some "EMBEDDINGS" do
      match line? with
      | none => return .error "Missing EMBEDDINGS section"
      | some line =>
          toks := appendNatsFromLine line toks
          line? ← readLine? h
    if toks.size != seqLen then
      return .error s!"TOKENS length mismatch: expected {seqLen}, got {toks.size}"
    inputTokens := some toks

  IO.println s!"[2/5] Loading input embeddings (seq_len={seqLen}, model_dim={modelDim})..."

  -- Ensure we are at EMBEDDINGS.
  line? ← skipUntil h line? (fun s => s.trim = "EMBEDDINGS")
  if line? = none then
    return .error "Missing EMBEDDINGS section"
  -- Consume EMBEDDINGS marker.
  line? ← readLine? h

  let mut embFloats : Array Float := Array.mkEmpty (seqLen * modelDim)
  while true do
    match line? with
    | none => break
    | some line =>
        if line.startsWith "LAYER" then
          break
        embFloats := appendFloatsFromLine line embFloats
        line? ← readLine? h

  let inputEmbeddings := buildMatrix seqLen modelDim embFloats
  if inputEmbeddings.numRows != seqLen || inputEmbeddings.numCols != modelDim then
    return .error s!"Input embeddings dimension mismatch: \
      expected {seqLen}×{modelDim}, got {inputEmbeddings.numRows}×{inputEmbeddings.numCols}"

  IO.println s!"[3/5] Loading {numLayers} layers with {numHeads} heads each..."

  let mut layers : Array (Array ConcreteAttentionLayer) := #[]
  let mut attnProjBias : Array ConcreteMatrix := #[]
  let mut mlps : Array ConcreteMLPLayer := #[]
  let mut ln1 : Array ConcreteLayerNormParams := #[]
  let mut ln2 : Array ConcreteLayerNormParams := #[]

  for l in [:numLayers] do
    IO.println s!"  Loading layer {l}/{numLayers}..."
    -- Ensure we are at a LAYER marker.
    line? ← skipUntil h line? (fun s => s.startsWith "LAYER")
    if line? = none then
      return .error s!"Missing LAYER {l} section"
    line? ← readLine? h

    let mut layerHeads : Array ConcreteAttentionLayer := #[]
    for _h in [:numHeads] do
      line? ← skipUntil h line? (fun s => s.startsWith "HEAD")
      if line? = none then
        return .error s!"Layer {l}: missing HEAD section"
      line? ← readLine? h

      line? ← skipUntil h line? (fun s => s.trim = "W_Q")
      if line? = none then
        return .error s!"Layer {l}: missing W_Q"
      line? ← readLine? h
      let mut wqLines : Array String := #[]
      while line?.isSome &&
            !(line?.get!.startsWith "b_Q") &&
            !(line?.get!.startsWith "W_K") do
        wqLines := wqLines.push (line?.get!)
        line? ← readLine? h
      let tWq := spawnParseFloats wqLines (modelDim * headDim)
      let mut bqFloats : Array Float := Array.mkEmpty headDim
      if line?.map (·.trim) = some "b_Q" then
        line? ← readLine? h
        let mut bqLines : Array String := #[]
        while line?.isSome && !(line?.get!.startsWith "W_K") do
          bqLines := bqLines.push (line?.get!)
          line? ← readLine? h
        bqFloats := parseFloatsFromLines bqLines headDim

      if line? = none then
        return .error s!"Layer {l}: missing W_K"
      line? ← readLine? h
      let mut wkLines : Array String := #[]
      while line?.isSome &&
            !(line?.get!.startsWith "b_K") &&
            !(line?.get!.startsWith "W_V") do
        wkLines := wkLines.push (line?.get!)
        line? ← readLine? h
      let tWk := spawnParseFloats wkLines (modelDim * headDim)
      let mut bkFloats : Array Float := Array.mkEmpty headDim
      if line?.map (·.trim) = some "b_K" then
        line? ← readLine? h
        let mut bkLines : Array String := #[]
        while line?.isSome && !(line?.get!.startsWith "W_V") do
          bkLines := bkLines.push (line?.get!)
          line? ← readLine? h
        bkFloats := parseFloatsFromLines bkLines headDim

      if line? = none then
        return .error s!"Layer {l}: missing W_V"
      line? ← readLine? h
      let mut wvLines : Array String := #[]
      while line?.isSome &&
            !(line?.get!.startsWith "b_V") &&
            !(line?.get!.startsWith "W_O") do
        wvLines := wvLines.push (line?.get!)
        line? ← readLine? h
      let tWv := spawnParseFloats wvLines (modelDim * headDim)
      let mut bvFloats : Array Float := Array.mkEmpty headDim
      if line?.map (·.trim) = some "b_V" then
        line? ← readLine? h
        let mut bvLines : Array String := #[]
        while line?.isSome && !(line?.get!.startsWith "W_O") do
          bvLines := bvLines.push (line?.get!)
          line? ← readLine? h
        bvFloats := parseFloatsFromLines bvLines headDim

      if line? = none then
        return .error s!"Layer {l}: missing W_O"
      line? ← readLine? h
      let mut woLines : Array String := #[]
      while line?.isSome do
        let line := line?.get!
        if line.startsWith "HEAD" || line.startsWith "ATTN_BIAS" || line.startsWith "MLP" then
          break
        woLines := woLines.push line
        line? ← readLine? h

      let tWo := spawnParseFloats woLines (headDim * modelDim)

      let wqFloats := tWq.get
      let wkFloats := tWk.get
      let wvFloats := tWv.get
      let woFloats := tWo.get
      let head := mkAttentionLayer modelDim headDim wqFloats wkFloats wvFloats woFloats bqFloats bkFloats bvFloats
      layerHeads := layerHeads.push head

    if layerHeads.size != numHeads then
      return .error s!"Layer {l}: expected {numHeads} heads, got {layerHeads.size}"
    layers := layers.push layerHeads

    -- Optional attention projection bias (c_proj.bias), applied once after combining heads.
    let mut layerAttnBias : ConcreteMatrix := ConcreteMatrix.zeros 1 modelDim
    if line?.map (·.trim) = some "ATTN_BIAS" then
      line? ← readLine? h
      let mut biasLines : Array String := #[]
      while line?.isSome && (line?.get!.trim != "MLP") do
        biasLines := biasLines.push (line?.get!)
        line? ← readLine? h
      let biasFloats := parseFloatsFromLines biasLines modelDim
      layerAttnBias := buildMatrix 1 modelDim biasFloats
    attnProjBias := attnProjBias.push layerAttnBias

    -- MLP section.
    line? ← skipUntil h line? (fun s => s.trim = "MLP")
    if line? = none then
      return .error s!"Layer {l}: missing MLP section"
    line? ← readLine? h

    line? ← skipUntil h line? (fun s => s.trim = "W_in")
    if line? = none then
      return .error s!"Layer {l}: missing W_in"
    line? ← readLine? h
    let mut winLines : Array String := #[]
    while line?.isSome do
      let line := line?.get!
      if line.startsWith "W_out" || line.startsWith "b_in" then
        break
      winLines := winLines.push line
      line? ← readLine? h

    if line? = none then
      return .error s!"Layer {l}: incomplete MLP section after W_in"

    let mut binFloats : Array Float := #[]
    let mut tWout? : Option (Task (Array Float)) := none
    let tWin := spawnParseFloats winLines (modelDim * hiddenDim)

    if line?.map (·.trim) = some "b_in" then
      -- Ordering (2): b_in then W_out.
      line? ← readLine? h
      let mut binLines : Array String := #[]
      while line?.isSome && !(line?.get!.startsWith "W_out") do
        binLines := binLines.push (line?.get!)
        line? ← readLine? h
      binFloats := parseFloatsFromLines binLines hiddenDim
      if line? = none then
        return .error s!"Layer {l}: missing W_out after b_in"
      line? ← readLine? h
      let mut woutLines : Array String := #[]
      while line?.isSome && !(line?.get!.startsWith "b_out") do
        woutLines := woutLines.push (line?.get!)
        line? ← readLine? h
      tWout? := some (spawnParseFloats woutLines (hiddenDim * modelDim))
    else
      if line?.map (·.trim) != some "W_out" then
        return .error s!"Layer {l}: expected W_out or b_in after W_in"
      line? ← readLine? h
      let mut woutLines : Array String := #[]
      while line?.isSome && !(line?.get!.startsWith "b_in") do
        woutLines := woutLines.push (line?.get!)
        line? ← readLine? h
      tWout? := some (spawnParseFloats woutLines (hiddenDim * modelDim))
      if line? = none then
        return .error s!"Layer {l}: missing b_in after W_out"
      line? ← readLine? h
      let mut binLines : Array String := #[]
      while line?.isSome && !(line?.get!.startsWith "b_out") do
        binLines := binLines.push (line?.get!)
        line? ← readLine? h
      binFloats := parseFloatsFromLines binLines hiddenDim

    -- b_out
    line? ← skipUntil h line? (fun s => s.startsWith "b_out")
    if line? = none then
      return .error s!"Layer {l}: missing b_out"
    line? ← readLine? h
    let mut boutLines : Array String := #[]
    while line?.isSome do
      let line := line?.get!
      let stop :=
        if isV2 then
          line.startsWith "LN1_GAMMA"
        else
          line.startsWith "LAYER" || line.startsWith "UNEMBEDDING"
      if stop then
        break
      boutLines := boutLines.push line
      line? ← readLine? h

    let winFloats := tWin.get
    let woutFloats :=
      match tWout? with
      | some t => t.get
      | none => #[]
    let boutFloats := parseFloatsFromLines boutLines modelDim

    let mlp := mkMLPLayer modelDim hiddenDim winFloats woutFloats binFloats boutFloats
    mlps := mlps.push mlp

    if isV2 then
      -- LN1_GAMMA
      line? ← skipUntil h line? (fun s => s.trim = "LN1_GAMMA")
      if line? = none then
        return .error s!"Layer {l}: missing LN1_GAMMA section"
      line? ← readLine? h
      let mut ln1GammaFloats : Array Float := #[]
      while line?.isSome && !(line?.get!.startsWith "LN1_BETA") do
        ln1GammaFloats := appendFloatsFromLine (line?.get!) ln1GammaFloats
        line? ← readLine? h

      -- LN1_BETA
      if line? = none then
        return .error s!"Layer {l}: missing LN1_BETA section"
      line? ← readLine? h
      let mut ln1BetaFloats : Array Float := #[]
      while line?.isSome && !(line?.get!.startsWith "LN2_GAMMA") do
        ln1BetaFloats := appendFloatsFromLine (line?.get!) ln1BetaFloats
        line? ← readLine? h

      -- LN2_GAMMA
      if line? = none then
        return .error s!"Layer {l}: missing LN2_GAMMA section"
      line? ← readLine? h
      let mut ln2GammaFloats : Array Float := #[]
      while line?.isSome && !(line?.get!.startsWith "LN2_BETA") do
        ln2GammaFloats := appendFloatsFromLine (line?.get!) ln2GammaFloats
        line? ← readLine? h

      -- LN2_BETA
      if line? = none then
        return .error s!"Layer {l}: missing LN2_BETA section"
      line? ← readLine? h
      let mut ln2BetaFloats : Array Float := #[]
      while line?.isSome &&
          !(line?.get!.startsWith "LAYER") &&
          !(line?.get!.startsWith "LN_F_GAMMA") do
        ln2BetaFloats := appendFloatsFromLine (line?.get!) ln2BetaFloats
        line? ← readLine? h

      let ln1Params : ConcreteLayerNormParams := {
        gamma := buildMatrix 1 modelDim ln1GammaFloats
        beta := buildMatrix 1 modelDim ln1BetaFloats
      }
      let ln2Params : ConcreteLayerNormParams := {
        gamma := buildMatrix 1 modelDim ln2GammaFloats
        beta := buildMatrix 1 modelDim ln2BetaFloats
      }
      ln1 := ln1.push ln1Params
      ln2 := ln2.push ln2Params
    else
      ln1 := ln1.push (ConcreteLayerNormParams.identity modelDim)
      ln2 := ln2.push (ConcreteLayerNormParams.identity modelDim)

  IO.println "[4/5] Loading unembedding matrix..."

  let mut lnf : ConcreteLayerNormParams := ConcreteLayerNormParams.identity modelDim
  if isV2 then
    line? ← skipUntil h line? (fun s => s.trim = "LN_F_GAMMA")
    if line? = none then
      return .error "Missing LN_F_GAMMA section"
    line? ← readLine? h
    let mut lnfGammaFloats : Array Float := #[]
    while line?.isSome && !(line?.get!.startsWith "LN_F_BETA") do
      lnfGammaFloats := appendFloatsFromLine (line?.get!) lnfGammaFloats
      line? ← readLine? h

    if line? = none then
      return .error "Missing LN_F_BETA section"
    line? ← readLine? h
    let mut lnfBetaFloats : Array Float := #[]
    while line?.isSome && !(line?.get!.startsWith "UNEMBEDDING") do
      lnfBetaFloats := appendFloatsFromLine (line?.get!) lnfBetaFloats
      line? ← readLine? h

    lnf := {
      gamma := buildMatrix 1 modelDim lnfGammaFloats
      beta := buildMatrix 1 modelDim lnfBetaFloats
    }

  line? ← skipUntil h line? (fun s => s.trim = "UNEMBEDDING")
  if line? = none then
    return .error "Missing UNEMBEDDING section"
  line? ← readLine? h
  let mut unembFloats : Array Float := Array.mkEmpty (modelDim * vocabSize)
  while true do
    match line? with
    | none => break
    | some line =>
        unembFloats := appendFloatsFromLine line unembFloats
        line? ← readLine? h

  let unembedding := buildMatrix modelDim vocabSize unembFloats

  let model : ConcreteModel := {
    numLayers := numLayers
    layers := layers
    attnProjBias := attnProjBias
    mlps := mlps
    ln1 := ln1
    ln2 := ln2
    lnf := lnf
    seqLen := seqLen
    inputTokens := inputTokens
    inputEmbeddings := inputEmbeddings
    unembedding := some unembedding
  }

  IO.println "[5/5] Model loaded successfully!\n"
  return .ok model

/-! ## File IO Operations -/

/-- Load a model from a file path. Supports .nfpt (text) format. -/
def loadModel (path : System.FilePath) : IO LoadResult := do
  if path.extension = some "nfpt" then
    IO.FS.withFile path .read fun h =>
      loadFromTextHandle h
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
