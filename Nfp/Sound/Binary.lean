-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std

namespace Nfp.Sound

/-!
# Sound binary helpers (`NFP_BINARY_V1`)

This module provides minimal binary parsing utilities for the sound path.
It avoids Float arithmetic by decoding IEEE-754 bits and using integer math.
-/

structure BinaryHeader where
  numLayers : Nat
  numHeads : Nat
  modelDim : Nat
  headDim : Nat
  hiddenDim : Nat
  vocabSize : Nat
  seqLen : Nat
  deriving Repr

private def readLine? (h : IO.FS.Handle) : IO (Option String) := do
  let s ← h.getLine
  if s.isEmpty then
    return none
  return some s

private def parseHeaderLine (line : String) : Option (String × String) :=
  let line := line.trim
  if line.isEmpty then none
  else
    match line.splitOn "=" with
    | [k, v] => some (k.trim, v.trim)
    | _ => none

private def readHeaderNat (k v : String) : Option Nat :=
  if k = "num_layers" || k = "num_heads" || k = "model_dim" ||
      k = "head_dim" || k = "hidden_dim" || k = "vocab_size" || k = "seq_len" then
    v.toNat?
  else
    none

def readBinaryHeader (h : IO.FS.Handle) : IO (Except String BinaryHeader) := do
  let some magicLine ← readLine? h
    | return .error "empty file"
  let magic := magicLine.trim
  if magic != "NFP_BINARY_V1" then
    return .error "invalid magic: expected NFP_BINARY_V1"

  let mut numLayers : Option Nat := none
  let mut numHeads : Option Nat := none
  let mut modelDim : Option Nat := none
  let mut headDim : Option Nat := none
  let mut hiddenDim : Option Nat := none
  let mut vocabSize : Option Nat := none
  let mut seqLen : Option Nat := none

  let mut line? ← readLine? h
  while true do
    match line? with
    | none => return .error "unexpected EOF while reading header"
    | some line =>
        let t := line.trim
        if t = "BINARY_START" then
          break
        match parseHeaderLine t with
        | none => pure ()
        | some (k, v) =>
            let vNat? := readHeaderNat k v
            match k, vNat? with
            | "num_layers", some n => numLayers := some n
            | "num_heads", some n => numHeads := some n
            | "model_dim", some n => modelDim := some n
            | "head_dim", some n => headDim := some n
            | "hidden_dim", some n => hiddenDim := some n
            | "vocab_size", some n => vocabSize := some n
            | "seq_len", some n => seqLen := some n
            | _, _ => pure ()
        line? ← readLine? h

  let some L := numLayers | return .error "missing num_layers"
  let some H := numHeads | return .error "missing num_heads"
  let some d := modelDim | return .error "missing model_dim"
  let some dh := headDim | return .error "missing head_dim"
  let some dhid := hiddenDim | return .error "missing hidden_dim"
  let some v := vocabSize | return .error "missing vocab_size"
  let some n := seqLen | return .error "missing seq_len"
  if L = 0 || H = 0 || d = 0 || dh = 0 || dhid = 0 || v = 0 || n = 0 then
    return .error "invalid header: dimensions must be > 0"
  return .ok {
    numLayers := L
    numHeads := H
    modelDim := d
    headDim := dh
    hiddenDim := dhid
    vocabSize := v
    seqLen := n
  }

private def readExactly (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let mut out := ByteArray.empty
  while out.size < n do
    let chunk ← h.read (USize.ofNat (n - out.size))
    if chunk.isEmpty then
      throw (IO.userError "unexpected EOF")
    out := out ++ chunk
  return out

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

private def pow2Nat (k : Nat) : Nat := Nat.pow 2 k

private def ceilDivNat (a : Int) (d : Nat) : Int :=
  let di : Int := Int.ofNat d
  let q := a.ediv di
  let r := a.emod di
    if r = 0 then q else q + 1


private def floatAbsCeilScaled (scalePow10 : Nat) (bits : UInt64) : Except String Int :=
  let expBits : UInt64 := (bits >>> 52) &&& 0x7ff
  let mantBits : UInt64 := bits &&& 0x000f_ffff_ffff_ffff
  if expBits = 0x7ff then
    .error "invalid float: NaN/Inf not supported"
  else if expBits = 0 && mantBits = 0 then
    .ok 0
  else
    let scale : Nat := Nat.pow 10 scalePow10
    let mant : Nat :=
      if expBits = 0 then
        mantBits.toNat
      else
        (mantBits + ((1 : UInt64) <<< 52)).toNat
    let expVal : Int :=
      if expBits = 0 then
        -1074
      else
        (Int.ofNat expBits.toNat) - 1075
    let mInt : Int := Int.ofNat mant
    if expVal ≥ 0 then
      let pow2 := pow2Nat expVal.toNat
      let num := mInt * Int.ofNat scale
      .ok (num * Int.ofNat pow2)
    else
      let denPow := pow2Nat (-expVal).toNat
      let num := mInt * Int.ofNat scale
      .ok (ceilDivNat num denPow)

private def floatScaledCeilSigned (scalePow10 : Nat) (bits : UInt64) : Except String Int :=
  match floatAbsCeilScaled scalePow10 bits with
  | .error e => .error e
  | .ok absScaled =>
      let signNeg : Bool := (bits >>> 63) = (1 : UInt64)
      if signNeg then
        .ok (-absScaled)
      else
        .ok absScaled

def skipBytes (h : IO.FS.Handle) (n : Nat) : IO (Except String Unit) := do
  let mut remaining := n
  while remaining > 0 do
    let chunkSize := min remaining 65536
    let chunk ← h.read (USize.ofNat chunkSize)
    if chunk.isEmpty then
      return .error "unexpected EOF"
    remaining := remaining - chunk.size
  return .ok ()

def skipI32Array (h : IO.FS.Handle) (n : Nat) : IO (Except String Unit) :=
  skipBytes h (n * 4)

def skipF64Array (h : IO.FS.Handle) (n : Nat) : IO (Except String Unit) :=
  skipBytes h (n * 8)

def readVectorMaxAbsScaled (h : IO.FS.Handle) (n scalePow10 : Nat) :
    IO (Except String Int) := do
  if n = 0 then
    return .ok 0
  let bytesE : Except String ByteArray ←
    try
      pure (Except.ok (← readExactly h (n * 8)))
    catch
      | _ => pure (Except.error "unexpected EOF")
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      let mut maxAbs : Int := 0
      for i in [:n] do
        let bits := u64FromLE bytes (i * 8)
        match floatAbsCeilScaled scalePow10 bits with
        | .error e => return .error e
        | .ok absScaled =>
            if absScaled > maxAbs then
              maxAbs := absScaled
      return .ok maxAbs

def readMatrixNormInfScaled (h : IO.FS.Handle) (rows cols scalePow10 : Nat) :
    IO (Except String Int) := do
  if rows = 0 || cols = 0 then
    return .ok 0
  let count := rows * cols
  let bytesE : Except String ByteArray ←
    try
      pure (Except.ok (← readExactly h (count * 8)))
    catch
      | _ => pure (Except.error "unexpected EOF")
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      let mut maxRowSum : Int := 0
      let mut curRowSum : Int := 0
      for i in [:count] do
        let bits := u64FromLE bytes (i * 8)
        match floatAbsCeilScaled scalePow10 bits with
        | .error e => return .error e
        | .ok absScaled =>
            curRowSum := curRowSum + absScaled
            if (i + 1) % cols = 0 then
              if curRowSum > maxRowSum then
                maxRowSum := curRowSum
              curRowSum := 0
      return .ok maxRowSum

def readScaledFloatArray (h : IO.FS.Handle) (count scalePow10 : Nat) :
    IO (Except String (Array Int)) := do
  if count = 0 then
    return .ok #[]
  let bytesE : Except String ByteArray ←
    try
      pure (Except.ok (← readExactly h (count * 8)))
    catch
      | _ => pure (Except.error "unexpected EOF")
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      let mut out : Array Int := Array.mkEmpty count
      for i in [:count] do
        let bits := u64FromLE bytes (i * 8)
        match floatScaledCeilSigned scalePow10 bits with
        | .error e => return .error e
        | .ok v => out := out.push v
      return .ok out

def readMatrixNormOneInfScaled (h : IO.FS.Handle) (rows cols scalePow10 : Nat) :
    IO (Except String (Nat × Nat)) := do
  if rows = 0 || cols = 0 then
    return .ok (0, 0)
  let count := rows * cols
  let bytesE : Except String ByteArray ←
    try
      pure (Except.ok (← readExactly h (count * 8)))
    catch
      | _ => pure (Except.error "unexpected EOF")
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      let mut maxRowSum : Nat := 0
      let mut curRowSum : Nat := 0
      let mut colSums : Array Nat := Array.replicate cols 0
      for i in [:count] do
        let bits := u64FromLE bytes (i * 8)
        match floatAbsCeilScaled scalePow10 bits with
        | .error e => return .error e
        | .ok absScaled =>
            let absNat := Int.toNat absScaled
            curRowSum := curRowSum + absNat
            let colIdx := i % cols
            colSums := colSums.set! colIdx (colSums[colIdx]! + absNat)
            if (i + 1) % cols = 0 then
              if curRowSum > maxRowSum then
                maxRowSum := curRowSum
              curRowSum := 0
      let mut maxColSum : Nat := 0
      for c in colSums do
        if c > maxColSum then
          maxColSum := c
      return .ok (maxRowSum, maxColSum)

def opBoundScaledFromOneInf (rowSum colSum : Nat) : Nat :=
  max rowSum colSum

def readMatrixOpBoundScaled (h : IO.FS.Handle) (rows cols scalePow10 : Nat) :
    IO (Except String Nat) := do
  match ← readMatrixNormOneInfScaled h rows cols scalePow10 with
  | .error e => return .error e
  | .ok (rowSum, colSum) =>
      return .ok (opBoundScaledFromOneInf rowSum colSum)

def ratOfScaledNat (scalePow10 : Nat) (x : Nat) : Rat :=
  Rat.normalize (Int.ofNat x) (Nat.pow 10 scalePow10) (den_nz := by
    have h10pos : (0 : Nat) < 10 := by decide
    exact Nat.ne_of_gt (Nat.pow_pos (n := scalePow10) h10pos))

def ratOfScaledInt (scalePow10 : Nat) (x : Int) : Rat :=
  Rat.normalize x (Nat.pow 10 scalePow10) (den_nz := by
    have h10pos : (0 : Nat) < 10 := by decide
    exact Nat.ne_of_gt (Nat.pow_pos (n := scalePow10) h10pos))

end Nfp.Sound
