-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Discovery

/-!
# Pure helpers for model IO

Pure parsing, construction, and tokenization utilities shared by the CLI-facing IO layer.
-/

namespace Nfp

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
    if p >= stop then
      return none
    let mut acc : Nat := 0
    let mut saw : Bool := false
    while p < stop do
      let c := p.get s
      if ('0' <= c) && (c <= '9') then
        acc := acc * 10 + (c.toNat - '0'.toNat)
        saw := true
        p := p.next s
      else
        return none
    if saw then some acc else none

  let mut p := start
  if p >= stop then
    return none

  let mut negative := false
  let c0 := p.get s
  if c0 = '-' then
    negative := true
    p := p.next s
  else if c0 = '+' then
    p := p.next s

  if p >= stop then
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
        if expStart >= stop then
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
    parseFloatRange s 0 s.rawEndPos

private def appendFloatsFromLine (line : String) (acc : Array Float) : Array Float := Id.run do
  let mut out := acc
  let s := line
  let mut p : String.Pos.Raw := 0
  let endPos := s.rawEndPos
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
  if p >= stop then
    return none
  let mut acc : Nat := 0
  let mut saw : Bool := false
  while p < stop do
    let c := p.get s
    if ('0' <= c) && (c <= '9') then
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
  let endPos := s.rawEndPos
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

/-! ## Matrix Construction for IO -/

/- Build a ConcreteMatrix from float data, padding or truncating as needed.
   This is safe because we ensure the data has exactly the right size. -/
def buildMatrix (rows cols : Nat) (data : Array Float) : ConcreteMatrix :=
  let expectedSize := rows * cols
  -- Use Array.ofFn to get the exact size we need while padding/truncating via getD.
  let finalData := Array.ofFn fun (i : Fin expectedSize) =>
    data.getD i.val 0.0
  {
    numRows := rows
    numCols := cols
    data := finalData
    size_eq := Array.size_ofFn
  }

/-! ## Load Result Helpers -/

/-- Result of loading a model. -/
inductive LoadResult
  | ok (model : ConcreteModel)
  | error (msg : String)

namespace LoadResult

def isOk : LoadResult -> Bool
  | ok _ => true
  | error _ => false

def getModel : LoadResult -> Option ConcreteModel
  | ok m => some m
  | error _ => none

def getError : LoadResult -> Option String
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

/- Build a ConcreteAttentionLayer from weight matrices.
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
    W_Q_dims := And.intro rfl rfl
    b_Q_dims := And.intro rfl rfl
    W_K_dims := And.intro rfl rfl
    b_K_dims := And.intro rfl rfl
    W_V_dims := And.intro rfl rfl
    b_V_dims := And.intro rfl rfl
    W_O_dims := And.intro rfl rfl
  }

/- Build a ConcreteMLPLayer from weight matrices.
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
    W_in_dims := And.intro rfl rfl
    W_out_dims := And.intro rfl rfl
    b_in_dims := And.intro rfl rfl
    b_out_dims := And.intro rfl rfl
  }

/-! ## Tokenization Utilities -/

/-- Simple tokenizer with vocabulary mapping. -/
structure Tokenizer where
  /-- Token strings in order of ID. -/
  tokens : Array String
  /-- Map from token string to its first ID. -/
  tokMap : Std.HashMap String Nat
  /-- Unknown token ID. -/
  unkId : Nat
  /-- Padding token ID. -/
  padId : Nat
  /-- End of sequence token ID. -/
  eosId : Nat

namespace Tokenizer

/-- Create a tokenizer from vocabulary list. -/
def fromVocabList (tokens : Array String)
    (unkId padId eosId : Nat := 0) : Tokenizer :=
  let tokMap :=
    Id.run do
      let mut out : Std.HashMap String Nat := Std.HashMap.emptyWithCapacity tokens.size
      let mut i := tokens.size
      while i > 0 do
        i := i - 1
        out := out.insert tokens[i]! i
      return out
  { tokens := tokens, tokMap := tokMap, unkId := unkId, padId := padId, eosId := eosId }

/-- Find a token's ID in the vocabulary. -/
def findToken (t : Tokenizer) (word : String) : Nat :=
  t.tokMap.getD word t.unkId

/-- Tokenize a string using simple whitespace splitting. -/
def tokenize (t : Tokenizer) (text : String) : Array Nat := Id.run do
  let mut ids : Array Nat := #[]
  let mut p : String.Pos.Raw := 0
  let stop := text.rawEndPos
  while p < stop do
    while p < stop && p.get text = ' ' do
      p := p.next text
    let start := p
    while p < stop && p.get text ≠ ' ' do
      p := p.next text
    if start < p then
      let word := String.Pos.Raw.extract text start p
      ids := ids.push (t.findToken word)
  ids

/-- Decode token IDs back to text. -/
def decode (t : Tokenizer) (ids : Array Nat) : String :=
  let tokens := ids.foldr
    (fun id acc =>
      if id < t.tokens.size then
        t.tokens[id]! :: acc
      else
        acc)
    []
  " ".intercalate tokens

end Tokenizer

/-! ## Embedding Utilities -/

/-- Look up embeddings for token IDs from the embedding matrix. -/
def lookupEmbeddings (embeddings : ConcreteMatrix) (tokenIds : Array Nat)
    (seqLen : Nat) (padId : Nat := 0) : ConcreteMatrix := Id.run do
  let modelDim := embeddings.numCols
  let rowCount := embeddings.numRows
  let tokenIdsSize := tokenIds.size
  let mut data : Array Float := Array.mkEmpty (seqLen * modelDim)

  for pos in [:seqLen] do
    let tokenId := if pos < tokenIdsSize then tokenIds[pos]! else padId
    -- Copy embedding row for this token.
    if tokenId < rowCount then
      let rowBase := tokenId * modelDim
      for dim in [:modelDim] do
        data := data.push embeddings.data[rowBase + dim]!
    else
      for _ in [:modelDim] do
        data := data.push 0.0

  buildMatrix seqLen modelDim data

/-- Set the input embeddings in a model for a given prompt (token IDs). -/
def ConcreteModel.withInputTokens (model : ConcreteModel)
    (embeddings : ConcreteMatrix) (tokenIds : Array Nat)
    (padId : Nat := 0) : ConcreteModel :=
  let inputEmb := lookupEmbeddings embeddings tokenIds model.seqLen padId
  { model with inputEmbeddings := inputEmb, inputTokens := some tokenIds }

end Nfp
