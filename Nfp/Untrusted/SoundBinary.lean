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

private def readExactly (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let mut out := ByteArray.empty
  while out.size < n do
    let chunk ← h.read (USize.ofNat (n - out.size))
    if chunk.isEmpty then
      throw (IO.userError "unexpected EOF")
    out := out ++ chunk
  return out

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
      return Nfp.Sound.vectorMaxAbsScaledFromBytes bytes n scalePow10

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
      return Nfp.Sound.matrixNormInfScaledFromBytes bytes rows cols scalePow10

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
      return Nfp.Sound.scaledFloatArrayFromBytes bytes count scalePow10

def readScaledFloat (h : IO.FS.Handle) (scalePow10 : Nat) : IO (Except String Int) := do
  let bytesE : Except String ByteArray ←
    try
      pure (Except.ok (← readExactly h 8))
    catch
      | _ => pure (Except.error "unexpected EOF")
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      return Nfp.Sound.scaledFloatFromBytes bytes scalePow10

def readI32Array (h : IO.FS.Handle) (count : Nat) :
    IO (Except String (Array Int)) := do
  if count = 0 then
    return .ok #[]
  let bytesE : Except String ByteArray ←
    try
      pure (Except.ok (← readExactly h (count * 4)))
    catch
      | _ => pure (Except.error "unexpected EOF")
  match bytesE with
  | .error e => return .error e
  | .ok bytes =>
      return Nfp.Sound.i32ArrayFromBytes bytes count

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
      return Nfp.Sound.matrixNormOneInfScaledFromBytes bytes rows cols scalePow10

def readMatrixOpBoundScaled (h : IO.FS.Handle) (rows cols scalePow10 : Nat) :
    IO (Except String Nat) := do
  match ← readMatrixNormOneInfScaled h rows cols scalePow10 with
  | .error e => return .error e
  | .ok (rowSum, colSum) =>
      return .ok (Nfp.Sound.opBoundScaledFromOneInf rowSum colSum)

end Nfp.Untrusted.SoundBinary
