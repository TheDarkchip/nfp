-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.BinaryPure

namespace Nfp.Untrusted.SoundBinary

/-!
# Untrusted binary IO helpers (`NFP_BINARY_V1`)

This module provides IO wrappers for the sound binary path.
Pure parsing/decoding lives in `Nfp.Sound.BinaryPure`.
-/

private def readLine? (h : IO.FS.Handle) : IO (Option String) := do
  let s ← h.getLine
  if s.isEmpty then
    return none
  return some s

def readBinaryHeader (h : IO.FS.Handle) : IO (Except String Nfp.Sound.BinaryHeader) := do
  let some magicLine ← readLine? h
    | return .error "empty file"
  let mut lines : Array String := #[]
  let mut line? ← readLine? h
  while true do
    match line? with
    | none => return .error "unexpected EOF while reading header"
    | some line =>
        let t := line.trim
        if t = "BINARY_START" then
          break
        lines := lines.push line
        line? ← readLine? h
  return Nfp.Sound.parseBinaryHeaderLines magicLine lines

/-- Read exactly `n` bytes or throw on EOF. -/
def readExactly (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  if n = 0 then
    return ByteArray.empty
  let mut remaining := n
  let mut out : Array UInt8 := Array.mkEmpty n
  while remaining > 0 do
    let chunk ← h.read (USize.ofNat remaining)
    if chunk.isEmpty then
      throw (IO.userError "unexpected EOF")
    for b in chunk.data do
      out := out.push b
    remaining := remaining - chunk.size
  return ByteArray.mk out

@[inline] private def readExactlyExcept (h : IO.FS.Handle) (n : Nat) :
    IO (Except String ByteArray) := do
  try
    return .ok (← readExactly h n)
  catch
    | _ => return .error "unexpected EOF"

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
  let bytesE ← readExactlyExcept h (n * 8)
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      return Nfp.Sound.vectorMaxAbsScaledFromBytes bytes n scalePow10

def readMatrixNormInfScaled (h : IO.FS.Handle) (rows cols scalePow10 : Nat) :
    IO (Except String Int) := do
  if rows = 0 || cols = 0 then
    return .ok 0
  let count := rows * cols
  let bytesE ← readExactlyExcept h (count * 8)
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      return Nfp.Sound.matrixNormInfScaledFromBytes bytes rows cols scalePow10

def readScaledFloatArray (h : IO.FS.Handle) (count scalePow10 : Nat) :
    IO (Except String (Array Int)) := do
  if count = 0 then
    return .ok #[]
  let bytesE ← readExactlyExcept h (count * 8)
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      return Nfp.Sound.scaledFloatArrayFromBytes bytes count scalePow10

def readScaledFloat (h : IO.FS.Handle) (scalePow10 : Nat) : IO (Except String Int) := do
  let bytesE ← readExactlyExcept h 8
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      return Nfp.Sound.scaledFloatFromBytes bytes scalePow10

def readI32Array (h : IO.FS.Handle) (count : Nat) :
    IO (Except String (Array Int)) := do
  if count = 0 then
    return .ok #[]
  let bytesE ← readExactlyExcept h (count * 4)
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      return Nfp.Sound.i32ArrayFromBytes bytes count

def readMatrixNormOneInfScaled (h : IO.FS.Handle) (rows cols scalePow10 : Nat) :
    IO (Except String (Nat × Nat)) := do
  if rows = 0 || cols = 0 then
    return .ok (0, 0)
  let count := rows * cols
  let bytesE ← readExactlyExcept h (count * 8)
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      return Nfp.Sound.matrixNormOneInfScaledFromBytes bytes rows cols scalePow10

def readMatrixOpBoundScaled (h : IO.FS.Handle) (rows cols scalePow10 : Nat) :
    IO (Except String Nat) := do
  match ← readMatrixNormOneInfScaled h rows cols scalePow10 with
  | .error e => return .error e
  | .ok (rowSum, colSum) =>
      return .ok (Nfp.Sound.opBoundScaledFromOneInf rowSum colSum)

end Nfp.Untrusted.SoundBinary
