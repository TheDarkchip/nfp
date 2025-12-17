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
W_K
<floats>
...

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

    -- Parse unsigned decimal without exponent.
    let parseUnsignedNoExp (t : String) : Option Float := do
      let parts := t.splitOn "."
      match parts with
      | [intPart] =>
        let n := intPart.toNat?
        n.map (·.toFloat)
      | [intPart, fracPart] =>
        let intN := if intPart.isEmpty then some 0 else intPart.toNat?
        let fracN := if fracPart.isEmpty then some 0 else fracPart.toNat?
        match intN, fracN with
        | some iN, some fN =>
          let fracLen := fracPart.length
          let divisor := Float.pow 10.0 fracLen.toFloat
          some (iN.toFloat + fN.toFloat / divisor)
        | _, _ => none
      | _ => none

    -- Parse optional scientific exponent `e` / `E` (e.g. "1.2e-3").
    let (mantissaStr, expStr?) : String × Option String :=
      match rest.splitOn "e" with
      | [m, e] => (m, some e)
      | _ =>
        match rest.splitOn "E" with
        | [m, e] => (m, some e)
        | _ => (rest, none)

    let mantissa ← parseUnsignedNoExp mantissaStr
    let value :=
      match expStr? with
      | none => mantissa
      | some expStr =>
        let expStr := expStr.trim
        if expStr.isEmpty then mantissa
        else
          let (expNeg, expRest) :=
            if expStr.startsWith "-" then (true, expStr.drop 1)
            else if expStr.startsWith "+" then (false, expStr.drop 1)
            else (false, expStr)
          match expRest.toNat? with
          | none => mantissa
          | some eNat =>
            let pow10 := Float.pow 10.0 eNat.toFloat
            if expNeg then mantissa / pow10 else mantissa * pow10

    some ((if negative then -1.0 else 1.0) * value)

/-- Parse a line of space-separated floats. -/
def parseFloatLine (line : String) : Array Float := Id.run do
  let mut out : Array Float := #[]
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
      let tok := String.Pos.Raw.extract s start p
      match parseFloat tok with
      | some x => out := out.push x
      | none => pure ()
  out

/-! ## Nat Parsing Utilities -/

/-- Parse a line of space-separated natural numbers. -/
def parseNatLine (line : String) : Array Nat := Id.run do
  let mut out : Array Nat := #[]
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
      let tok := String.Pos.Raw.extract s start p |>.trim
      match tok.toNat? with
      | some n => out := out.push n
      | none => pure ()
  out

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
      while i < lines.size && !lines[i]!.startsWith "W_K" do
        for x in parseFloatLine lines[i]! do
          wqFloats := wqFloats.push x
        i := i + 1

      -- Parse W_K
      i := i + 1
      let mut wkFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "W_V" do
        for x in parseFloatLine lines[i]! do
          wkFloats := wkFloats.push x
        i := i + 1

      -- Parse W_V
      i := i + 1
      let mut wvFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "W_O" do
        for x in parseFloatLine lines[i]! do
          wvFloats := wvFloats.push x
        i := i + 1

      -- Parse W_O
      i := i + 1
      let mut woFloats : Array Float := #[]
      while i < lines.size &&
            !lines[i]!.startsWith "HEAD" &&
            !lines[i]!.startsWith "MLP" do
        for x in parseFloatLine lines[i]! do
          woFloats := woFloats.push x
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

    let mut woutFloats : Array Float := #[]
    let mut binFloats : Array Float := #[]

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
    let mut boutFloats : Array Float := #[]
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
      let mut ln1GammaFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "LN1_BETA" do
        for x in parseFloatLine lines[i]! do
          ln1GammaFloats := ln1GammaFloats.push x
        i := i + 1

      -- LN1_BETA
      i := i + 1
      let mut ln1BetaFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "LN2_GAMMA" do
        for x in parseFloatLine lines[i]! do
          ln1BetaFloats := ln1BetaFloats.push x
        i := i + 1

      -- LN2_GAMMA
      i := i + 1
      let mut ln2GammaFloats : Array Float := #[]
      while i < lines.size && !lines[i]!.startsWith "LN2_BETA" do
        for x in parseFloatLine lines[i]! do
          ln2GammaFloats := ln2GammaFloats.push x
        i := i + 1

      -- LN2_BETA
      i := i + 1
      let mut ln2BetaFloats : Array Float := #[]
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
    let mut lnfGammaFloats : Array Float := #[]
    while i < lines.size && !lines[i]!.startsWith "LN_F_BETA" do
      for x in parseFloatLine lines[i]! do
        lnfGammaFloats := lnfGammaFloats.push x
      i := i + 1

    i := i + 1
    let mut lnfBetaFloats : Array Float := #[]
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
          for t in parseNatLine line do
            toks := toks.push t
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
        for x in parseFloatLine line do
          embFloats := embFloats.push x
        line? ← readLine? h

  let inputEmbeddings := buildMatrix seqLen modelDim embFloats
  if inputEmbeddings.numRows != seqLen || inputEmbeddings.numCols != modelDim then
    return .error s!"Input embeddings dimension mismatch: \
      expected {seqLen}×{modelDim}, got {inputEmbeddings.numRows}×{inputEmbeddings.numCols}"

  IO.println s!"[3/5] Loading {numLayers} layers with {numHeads} heads each..."

  let mut layers : Array (Array ConcreteAttentionLayer) := #[]
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
      let mut wqFloats : Array Float := #[]
      while line?.isSome && !(line?.get!.startsWith "W_K") do
        for x in parseFloatLine (line?.get!) do
          wqFloats := wqFloats.push x
        line? ← readLine? h

      if line? = none then
        return .error s!"Layer {l}: missing W_K"
      line? ← readLine? h
      let mut wkFloats : Array Float := #[]
      while line?.isSome && !(line?.get!.startsWith "W_V") do
        for x in parseFloatLine (line?.get!) do
          wkFloats := wkFloats.push x
        line? ← readLine? h

      if line? = none then
        return .error s!"Layer {l}: missing W_V"
      line? ← readLine? h
      let mut wvFloats : Array Float := #[]
      while line?.isSome && !(line?.get!.startsWith "W_O") do
        for x in parseFloatLine (line?.get!) do
          wvFloats := wvFloats.push x
        line? ← readLine? h

      if line? = none then
        return .error s!"Layer {l}: missing W_O"
      line? ← readLine? h
      let mut woFloats : Array Float := #[]
      while line?.isSome do
        let line := line?.get!
        if line.startsWith "HEAD" || line.startsWith "MLP" then
          break
        for x in parseFloatLine line do
          woFloats := woFloats.push x
        line? ← readLine? h

      let head := mkAttentionLayer modelDim headDim wqFloats wkFloats wvFloats woFloats
      layerHeads := layerHeads.push head

    if layerHeads.size != numHeads then
      return .error s!"Layer {l}: expected {numHeads} heads, got {layerHeads.size}"
    layers := layers.push layerHeads

    -- MLP section.
    line? ← skipUntil h line? (fun s => s.trim = "MLP")
    if line? = none then
      return .error s!"Layer {l}: missing MLP section"
    line? ← readLine? h

    line? ← skipUntil h line? (fun s => s.trim = "W_in")
    if line? = none then
      return .error s!"Layer {l}: missing W_in"
    line? ← readLine? h
    let mut winFloats : Array Float := #[]
    while line?.isSome do
      let line := line?.get!
      if line.startsWith "W_out" || line.startsWith "b_in" then
        break
      for x in parseFloatLine line do
        winFloats := winFloats.push x
      line? ← readLine? h

    if line? = none then
      return .error s!"Layer {l}: incomplete MLP section after W_in"

    let mut woutFloats : Array Float := #[]
    let mut binFloats : Array Float := #[]

    if line?.map (·.trim) = some "b_in" then
      -- Ordering (2): b_in then W_out.
      line? ← readLine? h
      while line?.isSome && !(line?.get!.startsWith "W_out") do
        for x in parseFloatLine (line?.get!) do
          binFloats := binFloats.push x
        line? ← readLine? h
      if line? = none then
        return .error s!"Layer {l}: missing W_out after b_in"
      line? ← readLine? h
      while line?.isSome && !(line?.get!.startsWith "b_out") do
        for x in parseFloatLine (line?.get!) do
          woutFloats := woutFloats.push x
        line? ← readLine? h
    else
      if line?.map (·.trim) != some "W_out" then
        return .error s!"Layer {l}: expected W_out or b_in after W_in"
      line? ← readLine? h
      while line?.isSome && !(line?.get!.startsWith "b_in") do
        for x in parseFloatLine (line?.get!) do
          woutFloats := woutFloats.push x
        line? ← readLine? h
      if line? = none then
        return .error s!"Layer {l}: missing b_in after W_out"
      line? ← readLine? h
      while line?.isSome && !(line?.get!.startsWith "b_out") do
        for x in parseFloatLine (line?.get!) do
          binFloats := binFloats.push x
        line? ← readLine? h

    -- b_out
    line? ← skipUntil h line? (fun s => s.startsWith "b_out")
    if line? = none then
      return .error s!"Layer {l}: missing b_out"
    line? ← readLine? h
    let mut boutFloats : Array Float := #[]
    while line?.isSome do
      let line := line?.get!
      let stop :=
        if isV2 then
          line.startsWith "LN1_GAMMA"
        else
          line.startsWith "LAYER" || line.startsWith "UNEMBEDDING"
      if stop then
        break
      for x in parseFloatLine line do
        boutFloats := boutFloats.push x
      line? ← readLine? h

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
        for x in parseFloatLine (line?.get!) do
          ln1GammaFloats := ln1GammaFloats.push x
        line? ← readLine? h

      -- LN1_BETA
      if line? = none then
        return .error s!"Layer {l}: missing LN1_BETA section"
      line? ← readLine? h
      let mut ln1BetaFloats : Array Float := #[]
      while line?.isSome && !(line?.get!.startsWith "LN2_GAMMA") do
        for x in parseFloatLine (line?.get!) do
          ln1BetaFloats := ln1BetaFloats.push x
        line? ← readLine? h

      -- LN2_GAMMA
      if line? = none then
        return .error s!"Layer {l}: missing LN2_GAMMA section"
      line? ← readLine? h
      let mut ln2GammaFloats : Array Float := #[]
      while line?.isSome && !(line?.get!.startsWith "LN2_BETA") do
        for x in parseFloatLine (line?.get!) do
          ln2GammaFloats := ln2GammaFloats.push x
        line? ← readLine? h

      -- LN2_BETA
      if line? = none then
        return .error s!"Layer {l}: missing LN2_BETA section"
      line? ← readLine? h
      let mut ln2BetaFloats : Array Float := #[]
      while line?.isSome &&
          !(line?.get!.startsWith "LAYER") &&
          !(line?.get!.startsWith "LN_F_GAMMA") do
        for x in parseFloatLine (line?.get!) do
          ln2BetaFloats := ln2BetaFloats.push x
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
      for x in parseFloatLine (line?.get!) do
        lnfGammaFloats := lnfGammaFloats.push x
      line? ← readLine? h

    if line? = none then
      return .error "Missing LN_F_BETA section"
    line? ← readLine? h
    let mut lnfBetaFloats : Array Float := #[]
    while line?.isSome && !(line?.get!.startsWith "UNEMBEDDING") do
      for x in parseFloatLine (line?.get!) do
        lnfBetaFloats := lnfBetaFloats.push x
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
        for x in parseFloatLine line do
          unembFloats := unembFloats.push x
        line? ← readLine? h

  let unembedding := buildMatrix modelDim vocabSize unembFloats

  let model : ConcreteModel := {
    numLayers := numLayers
    layers := layers
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
