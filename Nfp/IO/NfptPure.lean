-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.List.Range

/-!
Pure parsing utilities for `NFP_BINARY_V1` model files.

These helpers parse headers and extract selected weight slices as exact rationals.
-/

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
                seqLen := seqLen }, start)

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
      let denom : Int := Int.ofNat (pow2 1074)
      some (Rat.ofInt num / Rat.ofInt denom)
  else
    let mant := mantBits + pow2 52
    let exp := expBits - 1023
    let shift : Int := Int.ofNat exp - 52
    let base : Rat := Rat.ofInt (sign * Int.ofNat mant)
    if 0 ≤ shift then
      let k : Nat := Int.toNat shift
      some (base * Rat.ofInt (Int.ofNat (pow2 k)))
    else
      let k : Nat := Int.toNat (-shift)
      some (base / Rat.ofInt (Int.ofNat (pow2 k)))

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

private def readF64Rat (data : ByteArray) (off : Nat) : Option Rat := do
  let bits ← readNatLE data off 8
  ratOfFloatBits bits

private def bytesI32 (n : Nat) : Nat :=
  n * 4

private def bytesF64 (n : Nat) : Nat :=
  n * 8

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

/-- Byte offset from the binary start to the unembedding matrix. -/
def unembedOffset (h : NfptHeader) : Nat :=
  bytesI32 h.seqLen + bytesF64 (f64CountBeforeUnembed h)

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

end NfptPure

end IO

end Nfp
