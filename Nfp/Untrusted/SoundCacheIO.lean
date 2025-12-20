-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Init.System.IO
import Nfp.Sound.CachePure

namespace Nfp.Untrusted.SoundCacheIO

/-!
# Untrusted SOUND fixed-point cache

IO wrappers for the SOUND cache format. Pure parsing/encoding lives in `Nfp.Sound.CachePure`.
-/
private def readExactly (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let mut out := ByteArray.empty
  while out.size < n do
    let chunk ← h.read (USize.ofNat (n - out.size))
    if chunk.isEmpty then
      throw (IO.userError "unexpected EOF")
    out := out ++ chunk
  return out

def writeHeader (h : IO.FS.Handle) (hdr : Nfp.Sound.SoundCache.Header) : IO Unit := do
  h.write (Nfp.Sound.SoundCache.encodeHeader hdr)

def readHeader (h : IO.FS.Handle) : IO Nfp.Sound.SoundCache.Header := do
  let bytes ← readExactly h Nfp.Sound.SoundCache.headerBytes
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

/-- Build (or overwrite) a SOUND fixed-point cache file. -/
def buildCacheFile
    (modelPath cachePath : System.FilePath)
    (scalePow10 : Nat := 9) : IO (Except String Unit) := do
  ensureCacheDir
  let contents ← IO.FS.readFile modelPath
  let lines : Array String := (contents.splitOn "\n").toArray
  let modelHash ← fnv1a64File modelPath
  let mdata ← modelPath.metadata
  let modelSize : UInt64 := mdata.byteSize
  match Nfp.Sound.SoundCache.buildCacheBytes lines scalePow10 modelHash modelSize with
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
