-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Init.System.IO
import Nfp.Sound.Fixed

namespace Nfp.Sound

/-!
# SOUND fixed-point cache

Goal: avoid paying `Rat` normalization costs when repeatedly running SOUND local certification.

We build a cached binary file containing the tensors needed by the SOUND certify pass, encoded as
fixed-point base-10 integers with a global scale `S = 10^p`.

Correctness:
- During cache generation we parse decimals exactly (as base-10/scientific) and store a rounded
  scaled integer `ŵ ≈ w*S`.
- During certification we treat each stored scalar as an interval `[ŵ-1, ŵ+1] / S` which
  is guaranteed to contain the exact source value (so all resulting bounds remain rigorous).

We also store a stable `UInt64` file hash (FNV-1a) so we can detect mismatched caches.
-/

namespace SoundCache

def version : UInt32 := 1
def magic : ByteArray := "NFP_SND_CACHE_V1\n".toUTF8

structure Header where
  modelHash : UInt64
  modelSize : UInt64
  scalePow10 : UInt32
  numLayers : UInt32
  numHeads : UInt32
  modelDim : UInt32
  headDim : UInt32
  hiddenDim : UInt32
  deriving Repr

private def u32le (x : UInt32) : ByteArray :=
  let b0 := (x &&& 0xFF).toUInt8
  let b1 := ((x >>> 8) &&& 0xFF).toUInt8
  let b2 := ((x >>> 16) &&& 0xFF).toUInt8
  let b3 := ((x >>> 24) &&& 0xFF).toUInt8
  ByteArray.mk #[b0, b1, b2, b3]

private def u64le (x : UInt64) : ByteArray :=
  let b0 := (x &&& 0xFF).toUInt8
  let b1 := ((x >>> 8) &&& 0xFF).toUInt8
  let b2 := ((x >>> 16) &&& 0xFF).toUInt8
  let b3 := ((x >>> 24) &&& 0xFF).toUInt8
  let b4 := ((x >>> 32) &&& 0xFF).toUInt8
  let b5 := ((x >>> 40) &&& 0xFF).toUInt8
  let b6 := ((x >>> 48) &&& 0xFF).toUInt8
  let b7 := ((x >>> 56) &&& 0xFF).toUInt8
  ByteArray.mk #[b0, b1, b2, b3, b4, b5, b6, b7]

private def i32le (x : Int) : ByteArray :=
  -- encode as signed i32 little-endian (two's complement)
  let ux : UInt32 := UInt32.ofInt x
  u32le ux

private def readExactly (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let mut out := ByteArray.empty
  while out.size < n do
    let chunk ← h.read (USize.ofNat (n - out.size))
    if chunk.isEmpty then
      throw (IO.userError "unexpected EOF")
    out := out ++ chunk
  return out

private def readU32le (h : IO.FS.Handle) : IO UInt32 := do
  let b ← readExactly h 4
  let b0 := (b[0]!).toUInt32
  let b1 := (b[1]!).toUInt32
  let b2 := (b[2]!).toUInt32
  let b3 := (b[3]!).toUInt32
  return b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

private def readU64le (h : IO.FS.Handle) : IO UInt64 := do
  let b ← readExactly h 8
  let b0 := (b[0]!).toUInt64
  let b1 := (b[1]!).toUInt64
  let b2 := (b[2]!).toUInt64
  let b3 := (b[3]!).toUInt64
  let b4 := (b[4]!).toUInt64
  let b5 := (b[5]!).toUInt64
  let b6 := (b[6]!).toUInt64
  let b7 := (b[7]!).toUInt64
  return (
    b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24) |||
    (b4 <<< 32) ||| (b5 <<< 40) ||| (b6 <<< 48) ||| (b7 <<< 56)
  )

private def readI32le (h : IO.FS.Handle) : IO Int := do
  let u ← readU32le h
  let half : UInt32 := 0x80000000
  if u < half then
    return Int.ofNat u.toNat
  else
    let two32 : Int := Int.ofNat (Nat.pow 2 32)
    return (Int.ofNat u.toNat) - two32

def writeHeader (h : IO.FS.Handle) (hdr : Header) : IO Unit := do
  h.write magic
  h.write (u32le version)
  h.write (u64le hdr.modelHash)
  h.write (u64le hdr.modelSize)
  h.write (u32le hdr.scalePow10)
  h.write (u32le hdr.numLayers)
  h.write (u32le hdr.numHeads)
  h.write (u32le hdr.modelDim)
  h.write (u32le hdr.headDim)
  h.write (u32le hdr.hiddenDim)

def readHeader (h : IO.FS.Handle) : IO Header := do
  let m ← readExactly h magic.size
  if m ≠ magic then
    throw (IO.userError "invalid cache magic")
  let v ← readU32le h
  if v ≠ version then
    throw (IO.userError s!"unsupported cache version {v}")
  let modelHash ← readU64le h
  let modelSize ← readU64le h
  let scalePow10 ← readU32le h
  let numLayers ← readU32le h
  let numHeads ← readU32le h
  let modelDim ← readU32le h
  let headDim ← readU32le h
  let hiddenDim ← readU32le h
  return { modelHash, modelSize, scalePow10, numLayers, numHeads, modelDim, headDim, hiddenDim }

/-- FNV-1a 64-bit hash of a file's bytes (stable, deterministic). -/
def fnv1a64File (path : System.FilePath) : IO UInt64 := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let mut hash : UInt64 := 14695981039346656037
  let prime : UInt64 := 1099511628211
  let mut done := false
  while !done do
    let chunk ← h.read (USize.ofNat 1048576)
    if chunk.isEmpty then
      done := true
    else do
      for b in chunk.data do
        hash := (hash ^^^ (UInt64.ofNat b.toNat)) * prime
  return hash

def cacheDir : System.FilePath := "sound_cache"

def cachePath (modelPath : System.FilePath) (modelHash : UInt64) (scalePow10 : Nat) : System.FilePath :=
  let stem := modelPath.fileStem
  cacheDir / s!"{stem}_{modelHash.toNat}_p{scalePow10}.nfpc"

/-- Ensure the cache directory exists. -/
def ensureCacheDir : IO Unit := do
  IO.FS.createDirAll cacheDir

/-!
## Cache writer (text → binary)

We currently build the cache by reading the `.nfpt` text file into an array of lines.
This is memory-heavy but keeps implementation simple; once cached, repeated certifications
avoid the text parse cost entirely.
-/

private def parseHeaderLine (line : String) : Option (String × String) :=
  let line := line.trim
  if line.isEmpty then none
  else
    match line.splitOn "=" with
    | [k, v] => some (k.trim, v.trim)
    | _ => none

private def findLineIdxFrom (lines : Array String) (start : Nat) (p : String → Bool) : Option Nat :=
  Id.run do
    let mut i := start
    while i < lines.size do
      if p (lines[i]!.trim) then
        return some i
      i := i + 1
    return none

private def skipUntil (lines : Array String) (start : Nat) (p : String → Bool) : Nat :=
  match findLineIdxFrom lines start p with
  | some i => i
  | none => lines.size

private def skipBlankLines (lines : Array String) (start : Nat) : Nat :=
  Id.run do
    let mut i := start
    while i < lines.size && lines[i]!.trim.isEmpty do
      i := i + 1
    return i

private def countWsTokens (s : String) : Nat :=
  Id.run do
    let bytes := s.toUTF8
    let mut i : Nat := 0
    let mut inTok : Bool := false
    let mut cnt : Nat := 0
    while i < bytes.size do
      let b := bytes[i]!
      let isWs : Bool := b = 32 || b = 9
      if isWs then
        inTok := false
      else if !inTok then
        inTok := true
        cnt := cnt + 1
      i := i + 1
    return cnt

private def skipTokensFast (lines : Array String) (start : Nat) (numTokens : Nat) : Except String Nat :=
  Id.run do
    let mut iLine := start
    let mut remaining := numTokens
    while remaining > 0 do
      if iLine ≥ lines.size then
        return .error "unexpected end of file while skipping tokens"
      let line := lines[iLine]!.trim
      iLine := iLine + 1
      if line.isEmpty then
        pure ()
      else
        let c := countWsTokens line
        if c ≥ remaining then
          remaining := 0
        else
          remaining := remaining - c
    return .ok iLine

private def consumeFixedWrite
    (out : IO.FS.Handle)
    (scalePow10 : Nat)
    (lines : Array String)
    (start : Nat)
    (count : Nat) : IO (Except String Nat) := do
  let mut iLine := start
  let mut remaining := count
  let mut buf : ByteArray := ByteArray.empty
  while remaining > 0 do
    if iLine ≥ lines.size then
      return .error "unexpected end of file while reading fixed tokens"
    let line := lines[iLine]!.trim
    iLine := iLine + 1
    if line.isEmpty then
      pure ()
    else
      let bytes := line.toUTF8
      let mut j : Nat := 0
      while j < bytes.size && remaining > 0 do
        while j < bytes.size && (bytes[j]! = 32 || bytes[j]! = 9) do
          j := j + 1
        if j ≥ bytes.size then
          break
        let tokStart := j
        while j < bytes.size && (bytes[j]! ≠ 32 && bytes[j]! ≠ 9) do
          j := j + 1
        let tokStop := j
        match parseFixed10Rounded scalePow10 bytes tokStart tokStop with
        | .error e => return .error e
        | .ok x =>
            -- Store rounded i32 (two's complement). Runtime treats it as ±1 ulp interval.
            buf := buf ++ i32le x
            if buf.size ≥ 1048576 then
              out.write buf
              buf := ByteArray.empty
            remaining := remaining - 1
  if !buf.isEmpty then
    out.write buf
  return .ok iLine

private def readHeaderFromLines (lines : Array String) : Except String (Header × Nat) :=
  Id.run do
    let mut i : Nat := 0
    while i < lines.size && lines[i]!.trim.isEmpty do
      i := i + 1
    if i ≥ lines.size then
      return .error "empty model file"
    let headerTag := lines[i]!.trim
    if !headerTag.startsWith "NFP_TEXT" then
      return .error s!"unexpected header '{headerTag}'"
    i := i + 1

    let mut numLayers : Option UInt32 := none
    let mut numHeads : Option UInt32 := none
    let mut modelDim : Option UInt32 := none
    let mut headDim : Option UInt32 := none
    let mut hiddenDim : Option UInt32 := none

    while i < lines.size do
      let line := lines[i]!.trim
      if line.isEmpty then
        i := i + 1
        break
      match parseHeaderLine line with
      | none =>
          i := i + 1
      | some (k, v) =>
          match k with
          | "num_layers" => numLayers := (v.toNat?.map UInt32.ofNat)
          | "num_heads" => numHeads := (v.toNat?.map UInt32.ofNat)
          | "model_dim" => modelDim := (v.toNat?.map UInt32.ofNat)
          | "head_dim" => headDim := (v.toNat?.map UInt32.ofNat)
          | "hidden_dim" => hiddenDim := (v.toNat?.map UInt32.ofNat)
          | _ => pure ()
          i := i + 1

    let some L := numLayers | return .error "missing num_layers"
    let some H := numHeads | return .error "missing num_heads"
    let some d := modelDim | return .error "missing model_dim"
    let some dh := headDim | return .error "missing head_dim"
    let some dhid := hiddenDim | return .error "missing hidden_dim"
    let hdr : Header :=
      { modelHash := 0, modelSize := 0, scalePow10 := 0, numLayers := L, numHeads := H
        modelDim := d, headDim := dh, hiddenDim := dhid }
    return .ok (hdr, i)

private structure LNParamsFixed where
  gamma : Array Int
  beta : Array Int

private instance : Inhabited LNParamsFixed :=
  ⟨{ gamma := #[], beta := #[] }⟩

private def collectLayerNormParamsFixed
    (scalePow10 : Nat) (lines : Array String) (numLayers modelDim : Nat) :
    Except String (Array LNParamsFixed × Array LNParamsFixed) :=
  Id.run do
    let defP : LNParamsFixed :=
      { gamma := Array.replicate modelDim (0 : Int), beta := Array.replicate modelDim (0 : Int) }
    let mut ln1 : Array LNParamsFixed := Array.replicate numLayers defP
    let mut ln2 : Array LNParamsFixed := Array.replicate numLayers defP
    let mut curLayer : Nat := 0
    let mut i : Nat := 0
    while i < lines.size do
      let line := lines[i]!.trim
      if line.startsWith "LAYER" then
        let parts := line.splitOn " " |>.filter (· ≠ "")
        if parts.length >= 2 then
          curLayer := (parts[1]!).toNat? |>.getD curLayer
        i := i + 1
      else if line = "LN1_GAMMA" then
        match foldFixed10Tokens scalePow10 lines (i + 1) modelDim (Array.mkEmpty modelDim) (fun a x => a.push x) with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < ln1.size then
              let old := ln1[curLayer]!
              ln1 := ln1.set! curLayer { old with gamma := xs }
            i := next
      else if line = "LN1_BETA" then
        match foldFixed10Tokens scalePow10 lines (i + 1) modelDim (Array.mkEmpty modelDim) (fun a x => a.push x) with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < ln1.size then
              let old := ln1[curLayer]!
              ln1 := ln1.set! curLayer { old with beta := xs }
            i := next
      else if line = "LN2_GAMMA" then
        match foldFixed10Tokens scalePow10 lines (i + 1) modelDim (Array.mkEmpty modelDim) (fun a x => a.push x) with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < ln2.size then
              let old := ln2[curLayer]!
              ln2 := ln2.set! curLayer { old with gamma := xs }
            i := next
      else if line = "LN2_BETA" then
        match foldFixed10Tokens scalePow10 lines (i + 1) modelDim (Array.mkEmpty modelDim) (fun a x => a.push x) with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < ln2.size then
              let old := ln2[curLayer]!
              ln2 := ln2.set! curLayer { old with beta := xs }
            i := next
      else
        i := i + 1
    return .ok (ln1, ln2)

/-- Build (or overwrite) a SOUND fixed-point cache file. -/
def buildCacheFile
    (modelPath cachePath : System.FilePath)
    (scalePow10 : Nat := 9) : IO (Except String Unit) := do
  ensureCacheDir
  let contents ← IO.FS.readFile modelPath
  let lines : Array String := (contents.splitOn "\n").toArray

  let (hdr0, _afterHdr) ←
    match readHeaderFromLines lines with
    | .error e => return .error e
    | .ok x => pure x

  let L : Nat := hdr0.numLayers.toNat
  let H : Nat := hdr0.numHeads.toNat
  let d : Nat := hdr0.modelDim.toNat
  let dh : Nat := hdr0.headDim.toNat
  let dhid : Nat := hdr0.hiddenDim.toNat

  let modelHash ← fnv1a64File modelPath
  let mdata ← modelPath.metadata
  let modelSize : UInt64 := mdata.byteSize

  let (ln1, ln2) ←
    match collectLayerNormParamsFixed scalePow10 lines L d with
    | .error e => return .error e
    | .ok x => pure x

  -- Write to a temp file and rename atomically at the end to avoid leaving partial caches.
  let tmpPath := cachePath.withExtension "tmp"
  if (← tmpPath.pathExists) then
    IO.FS.removeFile tmpPath
  let out ← IO.FS.Handle.mk tmpPath IO.FS.Mode.write
  writeHeader out
    { hdr0 with
      modelHash := modelHash
      modelSize := modelSize
      scalePow10 := UInt32.ofNat scalePow10 }

  let mut pos : Nat := skipUntil lines 0 (fun s => s.startsWith "LAYER")
  for l in [:L] do
    -- write LN params first (so the reader can consume layers in a fixed order)
    let p1 := ln1.getD l { gamma := Array.replicate d (0 : Int), beta := Array.replicate d (0 : Int) }
    let p2 := ln2.getD l { gamma := Array.replicate d (0 : Int), beta := Array.replicate d (0 : Int) }
    for x in p1.gamma do out.write (i32le x)
    for x in p1.beta do out.write (i32le x)
    for x in p2.gamma do out.write (i32le x)
    for x in p2.beta do out.write (i32le x)

    -- scan to next layer marker
    pos := skipUntil lines pos (fun s => s.startsWith "LAYER")
    if pos ≥ lines.size then
      return .error s!"unexpected EOF while scanning layer {l}"
    pos := pos + 1

    -- per-head attention: skip Q/K, keep V/O and biases and attn bias.
    for _h in [:H] do
      pos := skipBlankLines lines pos
      if !(pos < lines.size && (lines[pos]!.trim.startsWith "HEAD")) then
        return .error "expected HEAD"
      pos := pos + 1

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "W_Q") then
        return .error "missing W_Q"
      match skipTokensFast lines (pos + 1) (d * dh) with
      | .error e => return .error e
      | .ok next => pos := next

      pos := skipBlankLines lines pos
      if pos < lines.size && lines[pos]!.trim = "b_Q" then
        match skipTokensFast lines (pos + 1) dh with
        | .error e => return .error e
        | .ok next => pos := next

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "W_K") then
        return .error "missing W_K"
      match skipTokensFast lines (pos + 1) (d * dh) with
      | .error e => return .error e
      | .ok next => pos := next

      pos := skipBlankLines lines pos
      if pos < lines.size && lines[pos]!.trim = "b_K" then
        match skipTokensFast lines (pos + 1) dh with
        | .error e => return .error e
        | .ok next => pos := next

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "W_V") then
        return .error "missing W_V"
      match (← consumeFixedWrite out scalePow10 lines (pos + 1) (d * dh)) with
      | .error e => return .error e
      | .ok next => pos := next

      pos := skipBlankLines lines pos
      if pos < lines.size && lines[pos]!.trim = "b_V" then
        match (← consumeFixedWrite out scalePow10 lines (pos + 1) dh) with
        | .error e => return .error e
        | .ok next => pos := next
      else
        -- default b_V = 0
        for _ in [:dh] do out.write (i32le 0)

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "W_O") then
        return .error "missing W_O"
      match (← consumeFixedWrite out scalePow10 lines (pos + 1) (dh * d)) with
      | .error e => return .error e
      | .ok next => pos := next

    pos := skipBlankLines lines pos
    if !(pos < lines.size && lines[pos]!.trim = "ATTN_BIAS") then
      return .error "missing ATTN_BIAS"
    match (← consumeFixedWrite out scalePow10 lines (pos + 1) d) with
    | .error e => return .error e
    | .ok next => pos := next

    -- MLP
    pos := skipBlankLines lines pos
    if !(pos < lines.size && lines[pos]!.trim = "MLP") then
      return .error "missing MLP"
    pos := pos + 1

    pos := skipBlankLines lines pos
    if !(pos < lines.size && lines[pos]!.trim = "W_in") then
      return .error "missing W_in"
    match (← consumeFixedWrite out scalePow10 lines (pos + 1) (d * dhid)) with
    | .error e => return .error e
    | .ok next => pos := next

    pos := skipBlankLines lines pos
    if !(pos < lines.size && lines[pos]!.trim = "b_in") then
      return .error "missing b_in"
    match (← consumeFixedWrite out scalePow10 lines (pos + 1) dhid) with
    | .error e => return .error e
    | .ok next => pos := next

    pos := skipBlankLines lines pos
    if !(pos < lines.size && lines[pos]!.trim = "W_out") then
      return .error "missing W_out"
    match (← consumeFixedWrite out scalePow10 lines (pos + 1) (dhid * d)) with
    | .error e => return .error e
    | .ok next => pos := next

    pos := skipBlankLines lines pos
    if !(pos < lines.size && lines[pos]!.trim = "b_out") then
      return .error "missing b_out"
    match (← consumeFixedWrite out scalePow10 lines (pos + 1) d) with
    | .error e => return .error e
    | .ok next => pos := next

    pos := skipUntil lines pos (fun s => s.startsWith "LAYER")

  out.flush
  if (← cachePath.pathExists) then
    IO.FS.removeFile cachePath
  IO.FS.rename tmpPath cachePath
  return .ok ()

/-!
## Cache reader (buffered)
-/

structure I32Reader where
  h : IO.FS.Handle
  buf : ByteArray
  pos : Nat

def I32Reader.init (h : IO.FS.Handle) : IO I32Reader :=
  pure { h := h, buf := ByteArray.empty, pos := 0 }

private def I32Reader.refill (r : I32Reader) : IO I32Reader := do
  let chunk ← r.h.read (USize.ofNat 1048576)
  if chunk.isEmpty then
    throw (IO.userError "unexpected EOF while reading cache")
  return { r with buf := chunk, pos := 0 }

def I32Reader.readI32 (r : I32Reader) : IO (Int × I32Reader) := do
  let r ←
    if r.pos + 4 ≤ r.buf.size then
      pure r
    else
      I32Reader.refill r
  let b0 := (r.buf[r.pos + 0]!).toUInt32
  let b1 := (r.buf[r.pos + 1]!).toUInt32
  let b2 := (r.buf[r.pos + 2]!).toUInt32
  let b3 := (r.buf[r.pos + 3]!).toUInt32
  let u : UInt32 := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let half : UInt32 := 0x80000000
  let x : Int :=
    if u < half then
      Int.ofNat u.toNat
    else
      (Int.ofNat u.toNat) - (Int.ofNat (Nat.pow 2 32))
  return (x, { r with pos := r.pos + 4 })

end SoundCache

end Nfp.Sound
