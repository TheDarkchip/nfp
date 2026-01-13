-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.List.Range
public import Nfp.Core.Basic
public import Nfp.Model.Gpt2
public import Nfp.Model.InductionHead
public import Nfp.Model.InductionPrompt

/-!
Pure parsing utilities for `NFP_BINARY_V1` model files.

These helpers parse headers and extract selected weight slices as rational values.
-/

public section

namespace Nfp

namespace IO

namespace NfptPure

/-- Required header fields for NFP binary models. -/
structure NfptHeader where
  /-- Number of transformer layers. -/
  numLayers : Nat
  /-- Number of attention heads per layer. -/
  numHeads : Nat
  /-- Model dimension. -/
  modelDim : Nat
  /-- Head dimension. -/
  headDim : Nat
  /-- MLP hidden dimension. -/
  hiddenDim : Nat
  /-- Vocabulary size. -/
  vocabSize : Nat
  /-- Sequence length used in the binary. -/
  seqLen : Nat
  /-- LayerNorm epsilon parameter. -/
  layerNormEps : Rat

/-- Array with a fixed size proof. -/
structure SizedArray (n : Nat) (α : Type) where
  /-- Underlying array data. -/
  data : Array α
  /-- Size proof for the array. -/
  size_eq : data.size = n

/-- Index into a `SizedArray` using a `Fin`. -/
def SizedArray.get {n : Nat} {α : Type} (arr : SizedArray n α) (i : Fin n) : α :=
  arr.data[i.val]'(by simp [arr.size_eq])

private def parseNat (s : String) : Except String Nat :=
  match s.toNat? with
  | some n => Except.ok n
  | none => Except.error s!"expected Nat, got '{s}'"

private def splitKV (line : String) : Option (String × String) :=
  match line.splitOn "=" with
  | [k, v] => some (k.trim, v.trim)
  | _ => none

private def readHeaderField (name : String) (fields : List (String × String)) :
    Except String Nat := do
  match fields.find? (fun kv => kv.1 = name) with
  | some kv => parseNat kv.2
  | none => throw s!"missing header field '{name}'"

private def parseInt (s : String) : Except String Int :=
  match s.toInt? with
  | some n => Except.ok n
  | none => Except.error s!"expected Int, got '{s}'"

private def pow10 (k : Nat) : Nat :=
  Nat.pow 10 k

private def parseRatScientific (s : String) : Except String Rat := do
  let s := s.trim
  let (sign, rest) :=
    if s.startsWith "-" then
      (-1, s.drop 1)
    else if s.startsWith "+" then
      (1, s.drop 1)
    else
      (1, s)
  let parts := rest.toLower.splitOn "e"
  let (mant, expStr?) ←
    match parts with
    | [m] => pure (m, none)
    | [m, e] => pure (m, some e)
    | _ => throw s!"invalid scientific literal '{s}'"
  let (intPart, fracPart) ←
    match mant.splitOn "." with
    | [i] => pure (i, "")
    | [i, f] => pure (i, f)
    | _ => throw s!"invalid decimal literal '{s}'"
  let digits := intPart ++ fracPart
  if digits = "" then
    throw s!"invalid decimal literal '{s}'"
  let n ← parseNat digits
  let scale := fracPart.length
  let base : Rat :=
    (Rat.ofInt (sign * Int.ofNat n)) / Rat.ofInt (Int.ofNat (pow10 scale))
  let exp ←
    match expStr? with
    | none => pure (0 : Int)
    | some e => parseInt e
  if exp ≥ 0 then
    let k := Int.toNat exp
    pure (ratRoundDown (base * Rat.ofInt (Int.ofNat (pow10 k))))
  else
    let k := Int.toNat (-exp)
    pure (ratRoundDown (base / Rat.ofInt (Int.ofNat (pow10 k))))

private def readHeaderFieldRat (names : List String) (fields : List (String × String)) :
    Except String Rat := do
  let rec loop : List String → Option String
    | [] => none
    | name :: rest =>
        match fields.find? (fun kv => kv.1 = name) with
        | some kv => some kv.2
        | none => loop rest
  match loop names with
  | some raw => parseRatScientific raw
  | none => throw s!"missing header field '{String.intercalate "|" names}'"

private def sentinelBytes : ByteArray :=
  "BINARY_START\n".toUTF8

private def findSentinel (data : ByteArray) : Option Nat :=
  let n := data.size
  let m := sentinelBytes.size
  if m ≤ n then
    let maxStart := n - m
    let rec loop (i : Nat) (remaining : Nat) : Option Nat :=
      match remaining with
      | 0 => none
      | Nat.succ rem =>
          let ok :=
            (List.range m).all (fun j => data.get! (i + j) = sentinelBytes.get! j)
          if ok then
            some i
          else
            loop (i + 1) rem
    loop 0 (maxStart + 1)
  else
    none

/-- Parse the NFP binary header and return the binary start offset. -/
def parseHeader (data : ByteArray) : Except String (NfptHeader × Nat) := do
  let idx ←
    match findSentinel data with
    | some i => pure i
    | none => throw "missing BINARY_START sentinel"
  let headerBytes := data.extract 0 idx
  let headerStr ←
    match String.fromUTF8? headerBytes with
    | some s => pure s
    | none => throw "invalid UTF-8 in header"
  let lines := headerStr.splitOn "\n" |>.filter (· ≠ "")
  match lines with
  | [] => throw "empty header"
  | magic :: rest =>
      if magic != "NFP_BINARY_V1" then
        throw s!"unexpected magic '{magic}'"
      let fields := rest.filterMap splitKV
      let numLayers ← readHeaderField "num_layers" fields
      let numHeads ← readHeaderField "num_heads" fields
      let modelDim ← readHeaderField "model_dim" fields
      let headDim ← readHeaderField "head_dim" fields
      let hiddenDim ← readHeaderField "hidden_dim" fields
      let vocabSize ← readHeaderField "vocab_size" fields
      let seqLen ← readHeaderField "seq_len" fields
      let layerNormEps ← readHeaderFieldRat ["layer_norm_eps", "eps"] fields
      if numLayers = 0 then
        throw "num_layers must be positive"
      if numHeads = 0 then
        throw "num_heads must be positive"
      if modelDim = 0 then
        throw "model_dim must be positive"
      if headDim = 0 then
        throw "head_dim must be positive"
      if hiddenDim = 0 then
        throw "hidden_dim must be positive"
      if vocabSize = 0 then
        throw "vocab_size must be positive"
      if seqLen = 0 then
        throw "seq_len must be positive"
      let start := idx + sentinelBytes.size
      return ({ numLayers := numLayers
                numHeads := numHeads
                modelDim := modelDim
                headDim := headDim
                hiddenDim := hiddenDim
                vocabSize := vocabSize
                seqLen := seqLen
                layerNormEps := layerNormEps }, start)

private def pow2 (k : Nat) : Nat :=
  Nat.pow 2 k

private def getBits (n hi lo : Nat) : Nat :=
  (n / pow2 lo) % pow2 (hi - lo + 1)

private def ratOfFloatBits (bits : Nat) : Option Rat :=
  let signBit := getBits bits 63 63
  let expBits := getBits bits 62 52
  let mantBits := getBits bits 51 0
  let sign : Int := if signBit = 0 then 1 else -1
  if expBits = 2047 then
    none
  else if expBits = 0 then
    if mantBits = 0 then
      some 0
    else
      let num : Int := sign * Int.ofNat mantBits
      some (ratOfIntWithPrec num 1074)
  else
    let mant := mantBits + pow2 52
    let exp : Int := Int.ofNat expBits - 1023
    let shift : Int := exp - 52
    let prec : Int := -shift
    some (ratOfIntWithPrec (sign * Int.ofNat mant) prec)

private def readNatLE (data : ByteArray) (off : Nat) (count : Nat) : Option Nat :=
  if off + count ≤ data.size then
    let rec loop (i : Nat) (acc : Nat) : Nat :=
      if i < count then
        let byte := data.get! (off + i)
        loop (i + 1) (acc + byte.toNat * pow2 (8 * i))
      else
        acc
    some (loop 0 0)
  else
    none

private def readI32 (data : ByteArray) (off : Nat) : Option Int := do
  let bits ← readNatLE data off 4
  let two31 := pow2 31
  let two32 := pow2 32
  if bits < two31 then
    some (Int.ofNat bits)
  else
    some (Int.ofNat bits - Int.ofNat two32)

private def readF64Rat (data : ByteArray) (off : Nat) : Option Rat := do
  let bits ← readNatLE data off 8
  ratOfFloatBits bits

private def bytesI32 (n : Nat) : Nat :=
  n * 4

private def bytesF64 (n : Nat) : Nat :=
  n * 8

private def sqrtNat? (n : Nat) : Option Nat :=
  let k := Nat.sqrt n
  if k * k = n then
    some k
  else
    none

private def scaleOfHeadDim (dHead : Nat) : Except String Rat := do
  match sqrtNat? dHead with
  | some k =>
      if k = 0 then
        throw "head_dim must be positive"
      else
        pure (ratRoundDown (Rat.ofInt 1 / Rat.ofInt (Int.ofNat k)))
  | none =>
      throw "head_dim must be a perfect square to compute scale"

private def matrixIndex {rows cols : Nat} (i : Fin rows) (j : Fin cols) : Fin (rows * cols) :=
  let idx := i.val * cols + j.val
  have hstep : i.val * cols + j.val < (i.val + 1) * cols := by
    have h' : i.val * cols + j.val < i.val * cols + cols :=
      Nat.add_lt_add_left j.isLt _
    have hmul : (i.val + 1) * cols = i.val * cols + cols := by
      simpa [Nat.succ_eq_add_one] using (Nat.succ_mul i.val cols)
    exact hmul ▸ h'
  have hle : (i.val + 1) * cols ≤ rows * cols :=
    Nat.mul_le_mul_right cols (Nat.succ_le_iff.mpr i.isLt)
  ⟨idx, lt_of_lt_of_le hstep hle⟩

private def readF64ListAux (data : ByteArray) (off : Nat) :
    Nat → List Rat → Except String (List Rat)
  | 0, acc => Except.ok acc.reverse
  | Nat.succ n, acc =>
      match readF64Rat data off with
      | some v => readF64ListAux data (off + bytesF64 1) n (v :: acc)
      | none => Except.error s!"invalid f64 at offset {off}"

private theorem readF64ListAux_length (data : ByteArray) :
    ∀ (off n : Nat) (acc xs : List Rat),
      readF64ListAux data off n acc = Except.ok xs →
      xs.length = acc.length + n := by
  intro off n acc xs h
  induction n generalizing off acc xs with
  | zero =>
      have h' := h
      simp only [readF64ListAux] at h'
      cases h'
      simp
  | succ n ih =>
      cases hread : readF64Rat data off with
      | none =>
          have h' := h
          simp only [readF64ListAux, hread] at h'
          cases h'
      | some v =>
          have h' := h
          simp only [readF64ListAux, hread] at h'
          have hlen := ih (off := off + bytesF64 1) (acc := v :: acc) (xs := xs) h'
          simpa [List.length, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hlen

private def readF64List (data : ByteArray) (off : Nat) (count : Nat) :
    Except String {xs : List Rat // xs.length = count} :=
  match h : readF64ListAux data off count [] with
  | Except.error msg => Except.error msg
  | Except.ok xs =>
      have hlen :
          xs.length = count := by
        simpa using readF64ListAux_length (data := data) (off := off)
          (n := count) (acc := []) (xs := xs) h
      Except.ok ⟨xs, hlen⟩

private def readI32ListAux (data : ByteArray) (off : Nat) :
    Nat → List Int → Except String (List Int)
  | 0, acc => Except.ok acc.reverse
  | Nat.succ n, acc =>
      match readI32 data off with
      | some v => readI32ListAux data (off + bytesI32 1) n (v :: acc)
      | none => Except.error s!"invalid i32 at offset {off}"

private theorem readI32ListAux_length (data : ByteArray) :
    ∀ (off n : Nat) (acc xs : List Int),
      readI32ListAux data off n acc = Except.ok xs →
      xs.length = acc.length + n := by
  intro off n acc xs h
  induction n generalizing off acc xs with
  | zero =>
      have h' := h
      simp only [readI32ListAux] at h'
      cases h'
      simp
  | succ n ih =>
      cases hread : readI32 data off with
      | none =>
          have h' := h
          simp only [readI32ListAux, hread] at h'
          cases h'
      | some v =>
          have h' := h
          simp only [readI32ListAux, hread] at h'
          have hlen := ih (off := off + bytesI32 1) (acc := v :: acc) (xs := xs) h'
          simpa [List.length, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hlen

private def readI32List (data : ByteArray) (off : Nat) (count : Nat) :
    Except String {xs : List Int // xs.length = count} :=
  match h : readI32ListAux data off count [] with
  | Except.error msg => Except.error msg
  | Except.ok xs =>
      have hlen :
          xs.length = count := by
        simpa using readI32ListAux_length (data := data) (off := off)
          (n := count) (acc := []) (xs := xs) h
      Except.ok ⟨xs, hlen⟩

private def readF64Matrix (data : ByteArray) (off : Nat) (rows cols : Nat) :
    Except String (Fin rows → Fin cols → Rat) := do
  let count := rows * cols
  let ⟨vals, hlen⟩ ← readF64List data off count
  let hlen' : vals.length = rows * cols := by
    simpa using hlen
  let mat : Fin rows → Fin cols → Rat := fun i j =>
    let idx := matrixIndex i j
    let hidx : idx.val < vals.length := lt_of_lt_of_eq idx.isLt hlen'.symm
    vals.get ⟨idx.val, hidx⟩
  return mat

private def readF64Vec (data : ByteArray) (off : Nat) (count : Nat) :
    Except String (Fin count → Rat) := do
  let ⟨vals, hlen⟩ ← readF64List data off count
  let hlen' : vals.length = count := by
    simpa using hlen
  let vec : Fin count → Rat := fun i =>
    vals.get ⟨i.val, lt_of_lt_of_eq i.isLt hlen'.symm⟩
  return vec

private def f64CountPerHead (h : NfptHeader) : Nat :=
  4 * h.modelDim * h.headDim + 3 * h.headDim

private def f64CountPerLayer (h : NfptHeader) : Nat :=
  h.numHeads * f64CountPerHead h +
    (2 * h.modelDim * h.hiddenDim + h.hiddenDim) +
    (6 * h.modelDim)

private def f64CountBeforeUnembed (h : NfptHeader) : Nat :=
  h.seqLen * h.modelDim +
    h.numLayers * f64CountPerLayer h +
    (2 * h.modelDim)

private def f64CountBeforeHeads (h : NfptHeader) : Nat :=
  h.seqLen * h.modelDim

/-- Byte offset from the binary start to the unembedding matrix. -/
def unembedOffset (h : NfptHeader) : Nat :=
  bytesI32 h.seqLen + bytesF64 (f64CountBeforeUnembed h)

private def finalLayerNormOffset (h : NfptHeader) : Nat :=
  bytesI32 h.seqLen +
    bytesF64 (f64CountBeforeHeads h + h.numLayers * f64CountPerLayer h)

/-- Read input embeddings stored in the binary. -/
def readEmbeddings (data : ByteArray) (start : Nat) (h : NfptHeader) :
    Except String (Fin h.seqLen → Fin h.modelDim → Rat) := do
  let base := start + bytesI32 h.seqLen
  readF64Matrix data base h.seqLen h.modelDim

/-- Read input token ids stored in the binary. -/
def readTokens (data : ByteArray) (start : Nat) (h : NfptHeader) :
    Except String (Fin h.seqLen → Nat) := do
  let ⟨vals, hlen⟩ ← readI32List data start h.seqLen
  let ok := vals.all (fun z => decide (0 ≤ z))
  if !ok then
    throw "token ids must be nonnegative"
  let hlen' : vals.length = h.seqLen := by
    simpa using hlen
  let tokens : Fin h.seqLen → Nat := fun i =>
    Int.toNat (vals.get ⟨i.val, lt_of_lt_of_eq i.isLt hlen'.symm⟩)
  return tokens

private def headOffset (h : NfptHeader) (layer head : Nat) : Nat :=
  bytesI32 h.seqLen +
    bytesF64 (f64CountBeforeHeads h +
      layer * f64CountPerLayer h +
      head * f64CountPerHead h)

private def layerExtrasOffset (h : NfptHeader) (layer : Nat) : Nat :=
  bytesI32 h.seqLen +
    bytesF64 (f64CountBeforeHeads h +
      layer * f64CountPerLayer h +
      h.numHeads * f64CountPerHead h)

/-- Read attention head weights and biases for a specific layer/head. -/
def readHeadWeights (data : ByteArray) (start : Nat) (h : NfptHeader)
    (layer head : Nat) :
    Except String (Model.Gpt2HeadWeights h.modelDim h.headDim) := do
  if layer < h.numLayers then
    if head < h.numHeads then
      let base := start + headOffset h layer head
      let wq ← readF64Matrix data base h.modelDim h.headDim
      let offbq := base + bytesF64 (h.modelDim * h.headDim)
      let bq ← readF64Vec data offbq h.headDim
      let offwk := offbq + bytesF64 h.headDim
      let wk ← readF64Matrix data offwk h.modelDim h.headDim
      let offbk := offwk + bytesF64 (h.modelDim * h.headDim)
      let bk ← readF64Vec data offbk h.headDim
      let offwv := offbk + bytesF64 h.headDim
      let wv ← readF64Matrix data offwv h.modelDim h.headDim
      let offbv := offwv + bytesF64 (h.modelDim * h.headDim)
      let bv ← readF64Vec data offbv h.headDim
      let offwo := offbv + bytesF64 h.headDim
      let woRaw ← readF64Matrix data offwo h.headDim h.modelDim
      let wo : Fin h.modelDim → Fin h.headDim → Rat := fun i j => woRaw j i
      return { wq := wq, bq := bq, wk := wk, bk := bk, wv := wv, bv := bv, wo := wo }
    else
      throw s!"head index out of range: {head}"
  else
    throw s!"layer index out of range: {layer}"

private def readLayerAttnBiasLn1 (data : ByteArray) (start : Nat) (h : NfptHeader)
    (layer : Nat) :
    Except String ((Fin h.modelDim → Rat) × (Fin h.modelDim → Rat) ×
      (Fin h.modelDim → Rat)) := do
  if layer < h.numLayers then
    let base := start + layerExtrasOffset h layer
    let attnBias ← readF64Vec data base h.modelDim
    let offWIn := base + bytesF64 h.modelDim
    let offBIn := offWIn + bytesF64 (h.modelDim * h.hiddenDim)
    let offWOut := offBIn + bytesF64 h.hiddenDim
    let offBOut := offWOut + bytesF64 (h.hiddenDim * h.modelDim)
    let offLn1Gamma := offBOut + bytesF64 h.modelDim
    let ln1Gamma ← readF64Vec data offLn1Gamma h.modelDim
    let offLn1Beta := offLn1Gamma + bytesF64 h.modelDim
    let ln1Beta ← readF64Vec data offLn1Beta h.modelDim
    return (attnBias, ln1Gamma, ln1Beta)
  else
    throw s!"layer index out of range: {layer}"

/-- Read GPT-2 layer parameters (MLP + LayerNorm) from the model binary. -/
def readLayerSlice (data : ByteArray) (start : Nat) (h : NfptHeader)
    (layer : Nat) : Except String (Model.Gpt2LayerSlice h.modelDim h.hiddenDim) := do
  if layer < h.numLayers then
    let base := start + layerExtrasOffset h layer
    let attnBias ← readF64Vec data base h.modelDim
    let offWIn := base + bytesF64 h.modelDim
    let mlpWIn ← readF64Matrix data offWIn h.modelDim h.hiddenDim
    let offBIn := offWIn + bytesF64 (h.modelDim * h.hiddenDim)
    let mlpBIn ← readF64Vec data offBIn h.hiddenDim
    let offWOut := offBIn + bytesF64 h.hiddenDim
    let mlpWOut ← readF64Matrix data offWOut h.hiddenDim h.modelDim
    let offBOut := offWOut + bytesF64 (h.hiddenDim * h.modelDim)
    let mlpBOut ← readF64Vec data offBOut h.modelDim
    let offLn1Gamma := offBOut + bytesF64 h.modelDim
    let ln1Gamma ← readF64Vec data offLn1Gamma h.modelDim
    let offLn1Beta := offLn1Gamma + bytesF64 h.modelDim
    let ln1Beta ← readF64Vec data offLn1Beta h.modelDim
    let offLn2Gamma := offLn1Beta + bytesF64 h.modelDim
    let ln2Gamma ← readF64Vec data offLn2Gamma h.modelDim
    let offLn2Beta := offLn2Gamma + bytesF64 h.modelDim
    let ln2Beta ← readF64Vec data offLn2Beta h.modelDim
    return { attnBias := attnBias
             mlpWIn := mlpWIn
             mlpBIn := mlpBIn
             mlpWOut := mlpWOut
             mlpBOut := mlpBOut
             ln1Gamma := ln1Gamma
             ln1Beta := ln1Beta
             ln2Gamma := ln2Gamma
             ln2Beta := ln2Beta }
  else
    throw s!"layer index out of range: {layer}"

/-- Read all GPT-2 layer slices from the model binary. -/
def readLayerSlices (data : ByteArray) (start : Nat) (h : NfptHeader) :
    Except String (SizedArray h.numLayers (Model.Gpt2LayerSlice h.modelDim h.hiddenDim)) := do
  let slices ← (List.finRange h.numLayers).foldlM
    (fun (acc : Array (Model.Gpt2LayerSlice h.modelDim h.hiddenDim)) layer => do
      let slice ← readLayerSlice data start h layer.val
      pure (acc.push slice))
    (#[] : Array (Model.Gpt2LayerSlice h.modelDim h.hiddenDim))
  if hlen : slices.size = h.numLayers then
    return { data := slices, size_eq := hlen }
  else
    throw "internal error: layer slice count mismatch"

/-- Read all attention head weights from the model binary. -/
def readLayerHeads (data : ByteArray) (start : Nat) (h : NfptHeader) :
    Except String
      (SizedArray h.numLayers
        (SizedArray h.numHeads (Model.Gpt2HeadWeights h.modelDim h.headDim))) := do
  let layers ← (List.finRange h.numLayers).foldlM
    (fun (acc : Array
        (SizedArray h.numHeads (Model.Gpt2HeadWeights h.modelDim h.headDim))) layer => do
      let heads ← (List.finRange h.numHeads).foldlM
        (fun (accHead : Array (Model.Gpt2HeadWeights h.modelDim h.headDim)) head => do
          let weights ← readHeadWeights data start h layer.val head.val
          pure (accHead.push weights))
        (#[] : Array (Model.Gpt2HeadWeights h.modelDim h.headDim))
      if hlen : heads.size = h.numHeads then
        let headArray : SizedArray h.numHeads (Model.Gpt2HeadWeights h.modelDim h.headDim) :=
          { data := heads, size_eq := hlen }
        pure (acc.push headArray)
      else
        throw "internal error: head count mismatch")
    (#[] : Array
      (SizedArray h.numHeads (Model.Gpt2HeadWeights h.modelDim h.headDim)))
  if hlen : layers.size = h.numLayers then
    return { data := layers, size_eq := hlen }
  else
    throw "internal error: layer head count mismatch"

/-- Read the final LayerNorm parameters from the model binary. -/
def readFinalLayerNorm (data : ByteArray) (start : Nat) (h : NfptHeader) :
    Except String (Model.Gpt2FinalLayerNorm h.modelDim) := do
  let base := start + finalLayerNormOffset h
  let gamma ← readF64Vec data base h.modelDim
  let offBeta := base + bytesF64 h.modelDim
  let beta ← readF64Vec data offBeta h.modelDim
  return { gamma := gamma, beta := beta }

/-- Read a single unembedding column as exact rationals. -/
def readUnembedColumn (data : ByteArray) (start : Nat) (h : NfptHeader) (col : Nat) :
    Except String (Fin h.modelDim → Rat) := do
  if col < h.vocabSize then
    let base := start + unembedOffset h
    let rows := List.range h.modelDim
    let vals ← rows.mapM (fun row => do
      let off := base + bytesF64 (row * h.vocabSize + col)
      match readF64Rat data off with
      | some v => pure v
      | none => throw s!"invalid f64 at offset {off}")
    if hlen : vals.length = h.modelDim then
      let vec : Fin h.modelDim → Rat := fun i =>
        vals.get ⟨i.val, by simp [hlen]⟩
      return vec
    else
      throw "internal error: unembed column length mismatch"
  else
    throw s!"column out of range: {col}"

/-- Read induction-head inputs directly from the model binary. -/
def buildInductionHeadInputs (h : NfptHeader) (scale : Rat)
    (tokens : Fin h.seqLen → Nat)
    (embed : Fin h.seqLen → Fin h.modelDim → Rat)
    (weights : Model.Gpt2HeadWeights h.modelDim h.headDim)
    (attnBias ln1Gamma ln1Beta : Fin h.modelDim → Rat)
    (dirTarget dirNegative : Nat)
    (colTarget colNegative : Fin h.modelDim → Rat)
    (period? : Option Nat) :
    Model.InductionHeadInputs h.seqLen h.modelDim h.headDim :=
  let direction : Fin h.modelDim → Rat := fun i => colTarget i - colNegative i
  let directionSpec : Circuit.DirectionSpec :=
    { target := dirTarget, negative := dirNegative }
  let active :=
    match period? with
    | some period => Model.activeOfPeriod (seq := h.seqLen) period
    | none => Model.activeOfTokens (seq := h.seqLen) tokens
  let prev :=
    match period? with
    | some period => Model.prevOfPeriod (seq := h.seqLen) period
    | none => Model.prevOfTokens (seq := h.seqLen) tokens
  { scale := scale
    active := active
    prev := prev
    embed := embed
    lnEps := h.layerNormEps
    ln1Gamma := ln1Gamma
    ln1Beta := ln1Beta
    wq := weights.wq
    bq := weights.bq
    wk := weights.wk
    bk := weights.bk
    wv := weights.wv
    bv := weights.bv
    wo := weights.wo
    attnBias := attnBias
    maskCausal := true
    maskValue := (-10000 : Rat)
    directionSpec := directionSpec
    direction := direction }

/-- Definitional characterization of `buildInductionHeadInputs`. -/
private theorem buildInductionHeadInputs_def (h : NfptHeader) (scale : Rat)
    (tokens : Fin h.seqLen → Nat)
    (embed : Fin h.seqLen → Fin h.modelDim → Rat)
    (weights : Model.Gpt2HeadWeights h.modelDim h.headDim)
    (attnBias ln1Gamma ln1Beta : Fin h.modelDim → Rat)
    (dirTarget dirNegative : Nat)
    (colTarget colNegative : Fin h.modelDim → Rat)
    (period? : Option Nat) :
    buildInductionHeadInputs h scale tokens embed weights attnBias ln1Gamma ln1Beta
        dirTarget dirNegative colTarget colNegative period? =
      { scale := scale
        active :=
          match period? with
          | some period => Model.activeOfPeriod (seq := h.seqLen) period
          | none => Model.activeOfTokens (seq := h.seqLen) tokens
        prev :=
          match period? with
          | some period => Model.prevOfPeriod (seq := h.seqLen) period
          | none => Model.prevOfTokens (seq := h.seqLen) tokens
        embed := embed
        lnEps := h.layerNormEps
        ln1Gamma := ln1Gamma
        ln1Beta := ln1Beta
        wq := weights.wq
        bq := weights.bq
        wk := weights.wk
        bk := weights.bk
        wv := weights.wv
        bv := weights.bv
        wo := weights.wo
        attnBias := attnBias
        maskCausal := true
        maskValue := (-10000 : Rat)
        directionSpec := { target := dirTarget, negative := dirNegative }
        direction := fun i => colTarget i - colNegative i } := rfl

/-- `buildInductionHeadInputs` uses the supplied direction ids and columns. -/
theorem buildInductionHeadInputs_direction_def (h : NfptHeader) (scale : Rat)
    (tokens : Fin h.seqLen → Nat)
    (embed : Fin h.seqLen → Fin h.modelDim → Rat)
    (weights : Model.Gpt2HeadWeights h.modelDim h.headDim)
    (attnBias ln1Gamma ln1Beta : Fin h.modelDim → Rat)
    (dirTarget dirNegative : Nat)
    (colTarget colNegative : Fin h.modelDim → Rat)
    (period? : Option Nat) :
    let inputs :=
      buildInductionHeadInputs h scale tokens embed weights attnBias ln1Gamma ln1Beta
        dirTarget dirNegative colTarget colNegative period?
    inputs.directionSpec = { target := dirTarget, negative := dirNegative } ∧
      inputs.direction = fun i => colTarget i - colNegative i := by
  simp [buildInductionHeadInputs]

/-- `buildInductionHeadInputs` derives `prev`/`active` from tokens or a fixed period. -/
theorem buildInductionHeadInputs_prev_active_def (h : NfptHeader) (scale : Rat)
    (tokens : Fin h.seqLen → Nat)
    (embed : Fin h.seqLen → Fin h.modelDim → Rat)
    (weights : Model.Gpt2HeadWeights h.modelDim h.headDim)
    (attnBias ln1Gamma ln1Beta : Fin h.modelDim → Rat)
    (dirTarget dirNegative : Nat)
    (colTarget colNegative : Fin h.modelDim → Rat)
    (period? : Option Nat) :
    let inputs :=
      buildInductionHeadInputs h scale tokens embed weights attnBias ln1Gamma ln1Beta
        dirTarget dirNegative colTarget colNegative period?
    inputs.active =
        (match period? with
        | some period => Model.activeOfPeriod (seq := h.seqLen) period
        | none => Model.activeOfTokens (seq := h.seqLen) tokens) ∧
    inputs.prev =
        (match period? with
        | some period => Model.prevOfPeriod (seq := h.seqLen) period
        | none => Model.prevOfTokens (seq := h.seqLen) tokens) := by
  constructor <;> rfl

/-- Active queries pick the maximal matching prior token when `period? = none`. -/
theorem buildInductionHeadInputs_prev_spec_of_active (h : NfptHeader) (scale : Rat)
    (tokens : Fin h.seqLen → Nat)
    (embed : Fin h.seqLen → Fin h.modelDim → Rat)
    (weights : Model.Gpt2HeadWeights h.modelDim h.headDim)
    (attnBias ln1Gamma ln1Beta : Fin h.modelDim → Rat)
    (dirTarget dirNegative : Nat)
    (colTarget colNegative : Fin h.modelDim → Rat) :
    ∀ {q},
        q ∈ (buildInductionHeadInputs h scale tokens embed weights attnBias ln1Gamma ln1Beta
              dirTarget dirNegative colTarget colNegative none).active →
          let p :=
            (buildInductionHeadInputs h scale tokens embed weights attnBias ln1Gamma ln1Beta
              dirTarget dirNegative colTarget colNegative none).prev q
          p < q ∧ tokens p = tokens q ∧
            ∀ k, k < q → tokens k = tokens q → k ≤ p := by
  intro q hq
  have hq' : q ∈ Model.activeOfTokens (seq := h.seqLen) tokens := by
    simpa [buildInductionHeadInputs] using hq
  have hspec := Model.prevOfTokens_spec_of_active (tokens := tokens) (q := q) hq'
  simpa [buildInductionHeadInputs] using hspec

/-- Read induction-head inputs directly from the model binary. -/
def readInductionHeadInputs (data : ByteArray) (start : Nat) (h : NfptHeader)
    (layer head dirTarget dirNegative : Nat) (period? : Option Nat) :
    Except String (Model.InductionHeadInputs h.seqLen h.modelDim h.headDim) := do
  let scale ← scaleOfHeadDim h.headDim
  let tokens ← readTokens data start h
  let embed ← readEmbeddings data start h
  let weights ← readHeadWeights data start h layer head
  let (attnBias, ln1Gamma, ln1Beta) ← readLayerAttnBiasLn1 data start h layer
  let colTarget ← readUnembedColumn data start h dirTarget
  let colNegative ← readUnembedColumn data start h dirNegative
  pure <|
    buildInductionHeadInputs h scale tokens embed weights attnBias ln1Gamma ln1Beta
      dirTarget dirNegative colTarget colNegative period?

end NfptPure

end IO

end Nfp
