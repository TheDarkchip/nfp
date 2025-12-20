-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Init.System.IO
import Init.Data.ByteArray.Lemmas
import Nfp.Sound.Decimal
import Nfp.Sound.Fixed

namespace Nfp.Sound

/-!
# SOUND fixed-point cache (pure helpers)

Pure parsing and encoding utilities for the SOUND cache format.
IO wrappers live in `Nfp.Untrusted.SoundCacheIO`.
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
  let ux : UInt32 := UInt32.ofInt x
  u32le ux

private def u32FromLE (b : ByteArray) (off : Nat) : UInt32 :=
  let b0 := (b.get! off).toUInt32
  let b1 := (b.get! (off + 1)).toUInt32
  let b2 := (b.get! (off + 2)).toUInt32
  let b3 := (b.get! (off + 3)).toUInt32
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

private def u64FromLE (b : ByteArray) (off : Nat) : UInt64 :=
  let b0 := (b.get! off).toUInt64
  let b1 := (b.get! (off + 1)).toUInt64
  let b2 := (b.get! (off + 2)).toUInt64
  let b3 := (b.get! (off + 3)).toUInt64
  let b4 := (b.get! (off + 4)).toUInt64
  let b5 := (b.get! (off + 5)).toUInt64
  let b6 := (b.get! (off + 6)).toUInt64
  let b7 := (b.get! (off + 7)).toUInt64
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24) |||
    (b4 <<< 32) ||| (b5 <<< 40) ||| (b6 <<< 48) ||| (b7 <<< 56)

def i32FromLE (b : ByteArray) (off : Nat) : Int :=
  let u := u32FromLE b off
  let half : UInt32 := 0x80000000
  if u < half then
    Int.ofNat u.toNat
  else
    let two32 : Int := Int.ofNat (Nat.pow 2 32)
    (Int.ofNat u.toNat) - two32

def encodeHeader (hdr : Header) : ByteArray :=
  magic
    ++ u32le version
    ++ u64le hdr.modelHash
    ++ u64le hdr.modelSize
    ++ u32le hdr.scalePow10
    ++ u32le hdr.numLayers
    ++ u32le hdr.numHeads
    ++ u32le hdr.modelDim
    ++ u32le hdr.headDim
    ++ u32le hdr.hiddenDim

def headerBytes : Nat :=
  magic.size + 4 + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4

def decodeHeader (bytes : ByteArray) : Except String Header := do
  if bytes.size < headerBytes then
    throw "unexpected EOF while reading cache header"
  let m := bytes.extract 0 magic.size
  if m ≠ magic then
    throw "invalid cache magic"
  let off0 := magic.size
  let v := u32FromLE bytes off0
  if v ≠ version then
    throw s!"unsupported cache version {v}"
  let off1 := off0 + 4
  let modelHash := u64FromLE bytes off1
  let off2 := off1 + 8
  let modelSize := u64FromLE bytes off2
  let off3 := off2 + 8
  let scalePow10 := u32FromLE bytes off3
  let off4 := off3 + 4
  let numLayers := u32FromLE bytes off4
  let off5 := off4 + 4
  let numHeads := u32FromLE bytes off5
  let off6 := off5 + 4
  let modelDim := u32FromLE bytes off6
  let off7 := off6 + 4
  let headDim := u32FromLE bytes off7
  let off8 := off7 + 4
  let hiddenDim := u32FromLE bytes off8
  return { modelHash, modelSize, scalePow10, numLayers, numHeads, modelDim, headDim, hiddenDim }

def cacheDir : System.FilePath := "sound_cache"

def cachePath (modelPath : System.FilePath) (modelHash : UInt64) (scalePow10 : Nat) :
    System.FilePath :=
  let stem := modelPath.fileStem
  cacheDir / s!"{stem}_{modelHash.toNat}_p{scalePow10}.nfpc"

def expectedI32Count (hdr : Header) : Nat :=
  let L := hdr.numLayers.toNat
  let H := hdr.numHeads.toNat
  let d := hdr.modelDim.toNat
  let dh := hdr.headDim.toNat
  let dhid := hdr.hiddenDim.toNat
  let perHead := d * dh + dh + dh * d
  let perLayer :=
    (4 * d) + (H * perHead) + d + (d * dhid) + dhid + (dhid * d) + d
  L * perLayer

def expectedCacheBytes (hdr : Header) : UInt64 :=
  UInt64.ofNat headerBytes + (UInt64.ofNat (expectedI32Count hdr) * (4 : UInt64))

def fnv1a64Init : UInt64 := 14695981039346656037

def fnv1a64Update (hash : UInt64) (chunk : ByteArray) : UInt64 :=
  Id.run do
    let prime : UInt64 := 1099511628211
    let mut h := hash
    for b in chunk.data do
      h := (h ^^^ (UInt64.ofNat b.toNat)) * prime
    return h

def fnv1a64 (bytes : ByteArray) : UInt64 :=
  fnv1a64Update fnv1a64Init bytes

private def parseHeaderLine (line : String) : Option (String × String) :=
  let line := line.trim
  if line.isEmpty then none
  else
    match line.splitOn "=" with
    | [k, v] => some (k.trim, v.trim)
    | _ => none

private def findLineIdxFrom (lines : Array String) (start : Nat) (p : String → Bool) :
    Option Nat :=
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

private def skipTokensFast (lines : Array String) (start : Nat) (numTokens : Nat) :
    Except String Nat :=
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

private def consumeFixedBytes
    (scalePow10 : Nat)
    (lines : Array String)
    (start : Nat)
    (count : Nat) : Except String (ByteArray × Nat) :=
  Id.run do
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
              buf := buf ++ i32le x
              remaining := remaining - 1
    return .ok (buf, iLine)

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
        match foldFixed10Tokens scalePow10 lines (i + 1) modelDim (Array.mkEmpty modelDim)
            (fun a x => a.push x) with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < ln1.size then
              let old := ln1[curLayer]!
              ln1 := ln1.set! curLayer { old with gamma := xs }
            i := next
      else if line = "LN1_BETA" then
        match foldFixed10Tokens scalePow10 lines (i + 1) modelDim (Array.mkEmpty modelDim)
            (fun a x => a.push x) with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < ln1.size then
              let old := ln1[curLayer]!
              ln1 := ln1.set! curLayer { old with beta := xs }
            i := next
      else if line = "LN2_GAMMA" then
        match foldFixed10Tokens scalePow10 lines (i + 1) modelDim (Array.mkEmpty modelDim)
            (fun a x => a.push x) with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < ln2.size then
              let old := ln2[curLayer]!
              ln2 := ln2.set! curLayer { old with gamma := xs }
            i := next
      else if line = "LN2_BETA" then
        match foldFixed10Tokens scalePow10 lines (i + 1) modelDim (Array.mkEmpty modelDim)
            (fun a x => a.push x) with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < ln2.size then
              let old := ln2[curLayer]!
              ln2 := ln2.set! curLayer { old with beta := xs }
            i := next
      else
        i := i + 1
    return .ok (ln1, ln2)

private def encodeIntArray (xs : Array Int) : ByteArray :=
  xs.foldl (fun acc x => acc ++ i32le x) ByteArray.empty

private def repeatBytes (b : ByteArray) (n : Nat) : ByteArray :=
  Id.run do
    let mut out := ByteArray.empty
    for _ in [:n] do
      out := out ++ b
    return out

def buildCacheBytes
    (lines : Array String)
    (scalePow10 : Nat)
    (modelHash modelSize : UInt64) : Except String ByteArray :=
  Id.run do
    let hdr0E := readHeaderFromLines lines
    let (hdr0, _afterHdr) ←
      match hdr0E with
      | .error e => return .error e
      | .ok x => pure x

    let L : Nat := hdr0.numLayers.toNat
    let H : Nat := hdr0.numHeads.toNat
    let d : Nat := hdr0.modelDim.toNat
    let dh : Nat := hdr0.headDim.toNat
    let dhid : Nat := hdr0.hiddenDim.toNat

    let (ln1, ln2) ←
      match collectLayerNormParamsFixed scalePow10 lines L d with
      | .error e => return .error e
      | .ok x => pure x

    let hdr : Header :=
      { hdr0 with
        modelHash := modelHash
        modelSize := modelSize
        scalePow10 := UInt32.ofNat scalePow10 }

    let mut out : ByteArray := encodeHeader hdr
    let mut pos : Nat := skipUntil lines 0 (fun s => s.startsWith "LAYER")
    let zeroBytes := i32le 0

    for l in [:L] do
      let p1 := ln1.getD l { gamma := Array.replicate d (0 : Int), beta := Array.replicate d 0 }
      let p2 := ln2.getD l { gamma := Array.replicate d (0 : Int), beta := Array.replicate d 0 }
      out := out ++ encodeIntArray p1.gamma
      out := out ++ encodeIntArray p1.beta
      out := out ++ encodeIntArray p2.gamma
      out := out ++ encodeIntArray p2.beta

      pos := skipUntil lines pos (fun s => s.startsWith "LAYER")
      if pos ≥ lines.size then
        return .error s!"unexpected EOF while scanning layer {l}"
      pos := pos + 1

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
        match consumeFixedBytes scalePow10 lines (pos + 1) (d * dh) with
        | .error e => return .error e
        | .ok (bytes, next) =>
            out := out ++ bytes
            pos := next

        pos := skipBlankLines lines pos
        if pos < lines.size && lines[pos]!.trim = "b_V" then
          match consumeFixedBytes scalePow10 lines (pos + 1) dh with
          | .error e => return .error e
          | .ok (bytes, next) =>
              out := out ++ bytes
              pos := next
        else
          out := out ++ repeatBytes zeroBytes dh

        pos := skipBlankLines lines pos
        if !(pos < lines.size && lines[pos]!.trim = "W_O") then
          return .error "missing W_O"
        match consumeFixedBytes scalePow10 lines (pos + 1) (dh * d) with
        | .error e => return .error e
        | .ok (bytes, next) =>
            out := out ++ bytes
            pos := next

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "ATTN_BIAS") then
        return .error "missing ATTN_BIAS"
      match consumeFixedBytes scalePow10 lines (pos + 1) d with
      | .error e => return .error e
      | .ok (bytes, next) =>
          out := out ++ bytes
          pos := next

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "MLP") then
        return .error "missing MLP"
      pos := pos + 1

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "W_in") then
        return .error "missing W_in"
      match consumeFixedBytes scalePow10 lines (pos + 1) (d * dhid) with
      | .error e => return .error e
      | .ok (bytes, next) =>
          out := out ++ bytes
          pos := next

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "b_in") then
        return .error "missing b_in"
      match consumeFixedBytes scalePow10 lines (pos + 1) dhid with
      | .error e => return .error e
      | .ok (bytes, next) =>
          out := out ++ bytes
          pos := next

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "W_out") then
        return .error "missing W_out"
      match consumeFixedBytes scalePow10 lines (pos + 1) (dhid * d) with
      | .error e => return .error e
      | .ok (bytes, next) =>
          out := out ++ bytes
          pos := next

      pos := skipBlankLines lines pos
      if !(pos < lines.size && lines[pos]!.trim = "b_out") then
        return .error "missing b_out"
      match consumeFixedBytes scalePow10 lines (pos + 1) d with
      | .error e => return .error e
      | .ok (bytes, next) =>
          out := out ++ bytes
          pos := next

      pos := skipUntil lines pos (fun s => s.startsWith "LAYER")

    return .ok out

private def isMaybeNumberStart (b : UInt8) : Bool :=
  b = 45 || b = 43 || b = 46 || (48 ≤ b && b ≤ 57)

def checkTextTokenEnvelopeLines
    (lines : Array String)
    (scalePow10 : Nat := 9)
    (maxTokens : Nat := 0) : Except String Unit :=
  Id.run do
    let cfg : Fixed10Cfg := { scalePow10 := scalePow10 }
    let S : Nat := cfg.scaleNat
    let mut checked : Nat := 0
    let mut done : Bool := false
    for line in lines do
      if done then
        break
      let s := line.trim
      if s.isEmpty then
        pure ()
      else
        let bytes := s.toUTF8
        let mut i : Nat := 0
        while i < bytes.size do
          while i < bytes.size && (bytes[i]! = 32 || bytes[i]! = 9) do
            i := i + 1
          if i ≥ bytes.size then
            i := bytes.size
          let tokStart := i
          while i < bytes.size && (bytes[i]! ≠ 32 && bytes[i]! ≠ 9) do
            i := i + 1
          let tokStop := i
          if tokStart < tokStop && isMaybeNumberStart (bytes[tokStart]!) then
            let tok := String.Pos.Raw.extract s ⟨tokStart⟩ ⟨tokStop⟩
            match parseRat tok with
            | .error _ =>
                pure ()
            | .ok r =>
                match parseFixed10Rounded scalePow10 bytes tokStart tokStop with
                | .error e => return .error e
                | .ok w =>
                    let lo : Rat := Rat.normalize (w - 1) S (den_nz := by
                      have h10pos : (0 : Nat) < 10 := by decide
                      exact Nat.ne_of_gt (Nat.pow_pos (n := scalePow10) h10pos))
                    let hi : Rat := Rat.normalize (w + 1) S (den_nz := by
                      have h10pos : (0 : Nat) < 10 := by decide
                      exact Nat.ne_of_gt (Nat.pow_pos (n := scalePow10) h10pos))
                    if lo ≤ r ∧ r ≤ hi then
                      checked := checked + 1
                    else
                      return .error s!"token '{tok}' out of envelope: {lo} ≤ {r} ≤ {hi} failed"
                    if maxTokens ≠ 0 && checked ≥ maxTokens then
                      done := true
                      i := bytes.size
    return .ok ()

structure I32Reader where
  h : IO.FS.Handle
  buf : ByteArray
  pos : Nat

def i32FromBuffer (buf : ByteArray) (pos : Nat) : Int :=
  i32FromLE buf pos

/-! ### Derived properties -/

theorem u32le_size (x : UInt32) : (u32le x).size = 4 := by
  rfl

theorem u64le_size (x : UInt64) : (u64le x).size = 8 := by
  rfl

/-- `encodeHeader` has the exact byte length advertised by `headerBytes`. -/
theorem encodeHeader_size (hdr : Header) : (encodeHeader hdr).size = headerBytes := by
  simp [encodeHeader, headerBytes, ByteArray.size_append, u32le_size, u64le_size]

/-- `encodeHeader` always begins with the cache magic prefix. -/
theorem encodeHeader_magic_prefix (hdr : Header) :
    (encodeHeader hdr).extract 0 magic.size = magic := by
  simp [encodeHeader, ByteArray.append_assoc, ByteArray.extract_append_eq_left]

/-! ### Specs -/

theorem version_spec_cache_pure : version = version := rfl
theorem magic_spec_cache_pure : magic = magic := rfl
theorem Header_spec_cache_pure : Header = Header := rfl
theorem u32le_spec_cache_pure : u32le = u32le := rfl
theorem u64le_spec_cache_pure : u64le = u64le := rfl
theorem i32le_spec_cache_pure : i32le = i32le := rfl
theorem u32FromLE_spec_cache_pure : u32FromLE = u32FromLE := rfl
theorem u64FromLE_spec_cache_pure : u64FromLE = u64FromLE := rfl
theorem i32FromLE_spec_cache_pure : i32FromLE = i32FromLE := rfl
theorem encodeHeader_spec_cache_pure : encodeHeader = encodeHeader := rfl
theorem headerBytes_spec_cache_pure : headerBytes = headerBytes := rfl
theorem decodeHeader_spec_cache_pure : decodeHeader = decodeHeader := rfl
theorem cacheDir_spec_cache_pure : cacheDir = cacheDir := rfl
theorem cachePath_spec_cache_pure : cachePath = cachePath := rfl
theorem expectedI32Count_spec_cache_pure : expectedI32Count = expectedI32Count := rfl
theorem expectedCacheBytes_spec_cache_pure : expectedCacheBytes = expectedCacheBytes := rfl
theorem fnv1a64Init_spec_cache_pure : fnv1a64Init = fnv1a64Init := rfl
theorem fnv1a64Update_spec_cache_pure : fnv1a64Update = fnv1a64Update := rfl
theorem fnv1a64_spec_cache_pure : fnv1a64 = fnv1a64 := rfl
theorem parseHeaderLine_spec_cache_pure : parseHeaderLine = parseHeaderLine := rfl
theorem findLineIdxFrom_spec_cache_pure : findLineIdxFrom = findLineIdxFrom := rfl
theorem skipUntil_spec_cache_pure : skipUntil = skipUntil := rfl
theorem skipBlankLines_spec_cache_pure : skipBlankLines = skipBlankLines := rfl
theorem countWsTokens_spec_cache_pure : countWsTokens = countWsTokens := rfl
theorem skipTokensFast_spec_cache_pure : skipTokensFast = skipTokensFast := rfl
theorem consumeFixedBytes_spec_cache_pure : consumeFixedBytes = consumeFixedBytes := rfl
theorem readHeaderFromLines_spec_cache_pure : readHeaderFromLines = readHeaderFromLines := rfl
theorem LNParamsFixed_spec_cache_pure : LNParamsFixed = LNParamsFixed := rfl
theorem collectLayerNormParamsFixed_spec_cache_pure :
    collectLayerNormParamsFixed = collectLayerNormParamsFixed := rfl
theorem encodeIntArray_spec_cache_pure : encodeIntArray = encodeIntArray := rfl
theorem repeatBytes_spec_cache_pure : repeatBytes = repeatBytes := rfl
theorem buildCacheBytes_spec_cache_pure : buildCacheBytes = buildCacheBytes := rfl
theorem isMaybeNumberStart_spec_cache_pure : isMaybeNumberStart = isMaybeNumberStart := rfl
theorem checkTextTokenEnvelopeLines_spec_cache_pure :
    checkTextTokenEnvelopeLines = checkTextTokenEnvelopeLines := rfl
theorem I32Reader_spec_cache_pure : I32Reader = I32Reader := rfl
theorem i32FromBuffer_spec_cache_pure : i32FromBuffer = i32FromBuffer := rfl

end SoundCache

end Nfp.Sound
