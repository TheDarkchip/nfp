-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Init.System.IO
import Nfp.Sound.CachePure
import Nfp.Untrusted.SoundBinary

namespace Nfp.Untrusted.SoundCacheIO

/-!
# Untrusted SOUND fixed-point cache

IO wrappers for the SOUND cache format. Pure parsing/encoding lives in `Nfp.Sound.CachePure`.
-/
private def appendI32LE (buf : Array UInt8) (x : Int) : Array UInt8 :=
  Id.run do
    let ux : UInt32 := UInt32.ofInt x
    let mut out := buf
    out := out.push (ux &&& 0xFF).toUInt8
    out := out.push ((ux >>> 8) &&& 0xFF).toUInt8
    out := out.push ((ux >>> 16) &&& 0xFF).toUInt8
    out := out.push ((ux >>> 24) &&& 0xFF).toUInt8
    return out

private def appendI32Array (buf : Array UInt8) (xs : Array Int) : Array UInt8 :=
  Id.run do
    let mut out := buf
    for x in xs do
      out := appendI32LE out x
    return out

private def appendBytes (buf : Array UInt8) (bytes : ByteArray) : Array UInt8 :=
  Id.run do
    let mut out := buf
    for b in bytes.data do
      out := out.push b
    return out

def isBinaryModelFile (path : System.FilePath) : IO (Except String Bool) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let line ← h.getLine
  if line.isEmpty then
    return .error "empty model file"
  let magic := line.trim
  return .ok (magic = "NFP_BINARY_V1")

def writeHeader (h : IO.FS.Handle) (hdr : Nfp.Sound.SoundCache.Header) : IO Unit := do
  h.write (Nfp.Sound.SoundCache.encodeHeader hdr)

def readHeader (h : IO.FS.Handle) : IO Nfp.Sound.SoundCache.Header := do
  let bytes ← Nfp.Untrusted.SoundBinary.readExactly h Nfp.Sound.SoundCache.headerBytes
  match Nfp.Sound.SoundCache.decodeHeader bytes with
  | .ok hdr => return hdr
  | .error e => throw (IO.userError e)

/-- FNV-1a 64-bit hash of a file's bytes (stable, deterministic). -/
def fnv1a64File (path : System.FilePath) : IO UInt64 := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let mut hash : UInt64 := Nfp.Sound.SoundCache.fnv1a64Init
  let mut done := false
  while !done do
    let chunk ← h.read (USize.ofNat 1048576)
    if chunk.isEmpty then
      done := true
    else
      hash := Nfp.Sound.SoundCache.fnv1a64Update hash chunk
  return hash

/-- Ensure the cache directory exists. -/
def ensureCacheDir : IO Unit := do
  IO.FS.createDirAll Nfp.Sound.SoundCache.cacheDir

def buildCacheBytesText
    (modelPath : System.FilePath)
    (scalePow10 : Nat)
    (modelHash modelSize : UInt64) : IO (Except String ByteArray) := do
  let contents ← IO.FS.readFile modelPath
  let lines : Array String := (contents.splitOn "\n").toArray
  return Nfp.Sound.SoundCache.buildCacheBytes lines scalePow10 modelHash modelSize

def buildCacheBytesBinary
    (modelPath : System.FilePath)
    (scalePow10 : Nat)
    (modelHash modelSize : UInt64) : IO (Except String ByteArray) := do
  let action : ExceptT String IO ByteArray := do
    let liftExcept {α : Type} (act : IO (Except String α)) : ExceptT String IO α :=
      ExceptT.mk act

    let h1 ← ExceptT.lift <| IO.FS.Handle.mk modelPath IO.FS.Mode.read
    let hdr1 ← liftExcept <| Nfp.Untrusted.SoundBinary.readBinaryHeader h1
    let d := hdr1.modelDim
    let dh := hdr1.headDim
    let dhid := hdr1.hiddenDim
    let L := hdr1.numLayers
    let H := hdr1.numHeads

    let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipI32Array h1 hdr1.seqLen
    let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 (hdr1.seqLen * d)

    let mut ln1Gamma : Array (Array Int) := Array.mkEmpty L
    let mut ln1Beta : Array (Array Int) := Array.mkEmpty L
    let mut ln2Gamma : Array (Array Int) := Array.mkEmpty L
    let mut ln2Beta : Array (Array Int) := Array.mkEmpty L

    for _l in [:L] do
      for _h in [:H] do
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 (d * dh)
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 dh
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 (d * dh)
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 dh
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 (d * dh)
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 dh
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 (dh * d)
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 d
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 (d * dhid)
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 dhid
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 (dhid * d)
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h1 d
      let ln1G ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h1 d scalePow10
      let ln1B ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h1 d scalePow10
      let ln2G ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h1 d scalePow10
      let ln2B ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h1 d scalePow10
      ln1Gamma := ln1Gamma.push ln1G
      ln1Beta := ln1Beta.push ln1B
      ln2Gamma := ln2Gamma.push ln2G
      ln2Beta := ln2Beta.push ln2B

    let hdrCache : Nfp.Sound.SoundCache.Header := {
      modelHash := modelHash
      modelSize := modelSize
      scalePow10 := UInt32.ofNat scalePow10
      numLayers := UInt32.ofNat L
      numHeads := UInt32.ofNat H
      modelDim := UInt32.ofNat d
      headDim := UInt32.ofNat dh
      hiddenDim := UInt32.ofNat dhid
    }

    let totalBytes : Nat :=
      Nfp.Sound.SoundCache.headerBytes +
        Nfp.Sound.SoundCache.expectedI32Count hdrCache * 4
    let mut out : Array UInt8 := Array.mkEmpty totalBytes
    out := appendBytes out (Nfp.Sound.SoundCache.encodeHeader hdrCache)

    let h2 ← ExceptT.lift <| IO.FS.Handle.mk modelPath IO.FS.Mode.read
    let hdr2 ← liftExcept <| Nfp.Untrusted.SoundBinary.readBinaryHeader h2
    if hdr2.numLayers ≠ L || hdr2.numHeads ≠ H || hdr2.modelDim ≠ d ||
        hdr2.headDim ≠ dh || hdr2.hiddenDim ≠ dhid then
      throw "binary header mismatch between passes"

    let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipI32Array h2 hdr2.seqLen
    let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 (hdr2.seqLen * d)

    for l in [:L] do
      out := appendI32Array out (ln1Gamma[l]!)
      out := appendI32Array out (ln1Beta[l]!)
      out := appendI32Array out (ln2Gamma[l]!)
      out := appendI32Array out (ln2Beta[l]!)
      for _h in [:H] do
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 (d * dh)
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 dh
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 (d * dh)
        let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 dh
        let wV ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h2 (d * dh) scalePow10
        let bV ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h2 dh scalePow10
        let wO ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h2 (dh * d) scalePow10
        out := appendI32Array out wV
        out := appendI32Array out bV
        out := appendI32Array out wO
      let attnBias ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h2 d scalePow10
      let wIn ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h2 (d * dhid) scalePow10
      let bIn ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h2 dhid scalePow10
      let wOut ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h2 (dhid * d) scalePow10
      let bOut ← liftExcept <| Nfp.Untrusted.SoundBinary.readScaledFloatArray h2 d scalePow10
      out := appendI32Array out attnBias
      out := appendI32Array out wIn
      out := appendI32Array out bIn
      out := appendI32Array out wOut
      out := appendI32Array out bOut
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 d
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 d
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 d
      let _ ← liftExcept <| Nfp.Untrusted.SoundBinary.skipF64Array h2 d

    if out.size ≠ totalBytes then
      throw s!"cache size mismatch: expected {totalBytes}, got {out.size}"
    return ByteArray.mk out
  action.run

/-- Build (or overwrite) a SOUND fixed-point cache file. -/
def buildCacheFile
    (modelPath cachePath : System.FilePath)
    (scalePow10 : Nat := 9) : IO (Except String Unit) := do
  ensureCacheDir
  let modelHash ← fnv1a64File modelPath
  let mdata ← modelPath.metadata
  let modelSize : UInt64 := mdata.byteSize
  match ← buildCacheBytesText modelPath scalePow10 modelHash modelSize with
  | .error e => return .error e
  | .ok bytes =>
      let tmpPath := cachePath.withExtension "tmp"
      if (← tmpPath.pathExists) then
        IO.FS.removeFile tmpPath
      let out ← IO.FS.Handle.mk tmpPath IO.FS.Mode.write
      out.write bytes
      out.flush
      if (← cachePath.pathExists) then
        IO.FS.removeFile cachePath
      IO.FS.rename tmpPath cachePath
      return .ok ()

/-! ## Consistency checks (for CI and debugging) -/

/-- Check that for each numeric token in the text file, its exact `Rat` value lies in the
`±1`-ulp interval induced by `parseFixed10Rounded scalePow10`. -/
def checkTextTokenEnvelope
    (modelPath : System.FilePath)
    (scalePow10 : Nat := 9)
    (maxTokens : Nat := 0) : IO (Except String Unit) := do
  let contents ← IO.FS.readFile modelPath
  let lines : Array String := (contents.splitOn "\n").toArray
  return Nfp.Sound.SoundCache.checkTextTokenEnvelopeLines lines scalePow10 maxTokens

/-- Check that the cache file size matches the expected tensor stream length. -/
def checkCacheFileSize (cachePath : System.FilePath) (hdr : Nfp.Sound.SoundCache.Header) :
    IO (Except String Unit) := do
  let mdata ← cachePath.metadata
  let expectedBytes := Nfp.Sound.SoundCache.expectedCacheBytes hdr
  if mdata.byteSize = expectedBytes then
    return .ok ()
  else
    return .error s!"cache size mismatch: expected {expectedBytes}, got {mdata.byteSize}"

/-! ## Cache reader (buffered) -/

def I32Reader.init (h : IO.FS.Handle) : IO Nfp.Sound.SoundCache.I32Reader :=
  pure { h := h, buf := ByteArray.empty, pos := 0 }

private def I32Reader.refill (r : Nfp.Sound.SoundCache.I32Reader) :
    IO Nfp.Sound.SoundCache.I32Reader := do
  let chunk ← r.h.read (USize.ofNat 1048576)
  if chunk.isEmpty then
    throw (IO.userError "unexpected EOF while reading cache")
  return { r with buf := chunk, pos := 0 }

def I32Reader.readI32 (r : Nfp.Sound.SoundCache.I32Reader) :
    IO (Int × Nfp.Sound.SoundCache.I32Reader) := do
  let r ←
    if r.pos + 4 ≤ r.buf.size then
      pure r
    else
      I32Reader.refill r
  let x := Nfp.Sound.SoundCache.i32FromBuffer r.buf r.pos
  return (x, { r with pos := r.pos + 4 })
end Nfp.Untrusted.SoundCacheIO
