-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Activation
import Nfp.Sound.Decimal

namespace Nfp.Sound

/-!
# Pure binary helpers (`NFP_BINARY_V1`)

Pure parsing and decoding utilities for the SOUND binary path.
IO wrappers live in `Nfp.Untrusted.SoundBinary`.
-/

structure BinaryHeader where
  numLayers : Nat
  numHeads : Nat
  modelDim : Nat
  headDim : Nat
  hiddenDim : Nat
  vocabSize : Nat
  seqLen : Nat
  eps : Rat
  geluDerivTarget : GeluDerivTarget
  deriving Repr

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

def parseBinaryHeaderLines (magicLine : String) (lines : Array String) :
    Except String BinaryHeader := do
  let magic := magicLine.trim
  if magic != "NFP_BINARY_V1" then
    throw "invalid magic: expected NFP_BINARY_V1"

  let mut numLayers : Option Nat := none
  let mut numHeads : Option Nat := none
  let mut modelDim : Option Nat := none
  let mut headDim : Option Nat := none
  let mut hiddenDim : Option Nat := none
  let mut vocabSize : Option Nat := none
  let mut seqLen : Option Nat := none
  let mut eps : Option Rat := none
  let mut gelu? : Option GeluDerivTarget := none

  for line in lines do
    let t := line.trim
    if t.isEmpty then
      pure ()
    else
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
          | "layer_norm_eps", _ =>
              match parseRat v with
              | .error e => throw s!"invalid layer_norm_eps '{v}': {e}"
              | .ok r => eps := some r
          | "eps", _ =>
              match parseRat v with
              | .error e => throw s!"invalid layer_norm_eps '{v}': {e}"
              | .ok r => eps := some r
          | "gelu_kind", _ =>
              match geluDerivTargetOfString v with
              | some t => gelu? := some t
              | none => throw s!"invalid gelu_kind '{v}' (expected tanh|exact)"
          | "gelu_deriv", _ =>
              match geluDerivTargetOfString v with
              | some t => gelu? := some t
              | none => throw s!"invalid gelu_deriv '{v}' (expected tanh|exact)"
          | _, _ => pure ()

  let some L := numLayers | throw "missing num_layers"
  let some H := numHeads | throw "missing num_heads"
  let some d := modelDim | throw "missing model_dim"
  let some dh := headDim | throw "missing head_dim"
  let some dhid := hiddenDim | throw "missing hidden_dim"
  let some v := vocabSize | throw "missing vocab_size"
  let some n := seqLen | throw "missing seq_len"
  let some epsVal := eps | throw "missing layer_norm_eps"
  let some geluVal := gelu? | throw "missing gelu_kind"
  if L = 0 || H = 0 || d = 0 || dh = 0 || dhid = 0 || v = 0 || n = 0 then
    throw "invalid header: dimensions must be > 0"
  return {
    numLayers := L
    numHeads := H
    modelDim := d
    headDim := dh
    hiddenDim := dhid
    vocabSize := v
    seqLen := n
    eps := epsVal
    geluDerivTarget := geluVal
  }

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

private def u32FromLE (b : ByteArray) (off : Nat) : UInt32 :=
  let b0 := (b.get! off).toUInt32
  let b1 := (b.get! (off + 1)).toUInt32
  let b2 := (b.get! (off + 2)).toUInt32
  let b3 := (b.get! (off + 3)).toUInt32
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

private def i32FromLE (b : ByteArray) (off : Nat) : Int :=
  let u := u32FromLE b off
  if u ≤ 0x7fffffff then
    Int.ofNat u.toNat
  else
    Int.ofNat u.toNat - (Int.ofNat (Nat.pow 2 32))

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

def vectorMaxAbsScaledFromBytes (bytes : ByteArray) (n scalePow10 : Nat) :
    Except String Int := do
  if n = 0 then
    return 0
  if bytes.size < n * 8 then
    throw "unexpected EOF"
  let mut maxAbs : Int := 0
  for i in [:n] do
    let bits := u64FromLE bytes (i * 8)
    match floatAbsCeilScaled scalePow10 bits with
    | .error e => throw e
    | .ok absScaled =>
        if absScaled > maxAbs then
          maxAbs := absScaled
  return maxAbs

def matrixNormInfScaledFromBytes (bytes : ByteArray) (rows cols scalePow10 : Nat) :
    Except String Int := do
  if rows = 0 || cols = 0 then
    return 0
  let count := rows * cols
  if bytes.size < count * 8 then
    throw "unexpected EOF"
  let mut maxRowSum : Int := 0
  let mut curRowSum : Int := 0
  for i in [:count] do
    let bits := u64FromLE bytes (i * 8)
    match floatAbsCeilScaled scalePow10 bits with
    | .error e => throw e
    | .ok absScaled =>
        curRowSum := curRowSum + absScaled
        if (i + 1) % cols = 0 then
          if curRowSum > maxRowSum then
            maxRowSum := curRowSum
          curRowSum := 0
  return maxRowSum

def scaledFloatArrayFromBytes (bytes : ByteArray) (count scalePow10 : Nat) :
    Except String (Array Int) := do
  if count = 0 then
    return #[]
  if bytes.size < count * 8 then
    throw "unexpected EOF"
  let mut out : Array Int := Array.mkEmpty count
  for i in [:count] do
    let bits := u64FromLE bytes (i * 8)
    match floatScaledCeilSigned scalePow10 bits with
    | .error e => throw e
    | .ok v => out := out.push v
  return out

def scaledFloatFromBytes (bytes : ByteArray) (scalePow10 : Nat) :
    Except String Int := do
  if bytes.size < 8 then
    throw "unexpected EOF"
  let bits := u64FromLE bytes 0
  match floatScaledCeilSigned scalePow10 bits with
  | .error e => throw e
  | .ok v => return v

def i32ArrayFromBytes (bytes : ByteArray) (count : Nat) :
    Except String (Array Int) := do
  if count = 0 then
    return #[]
  if bytes.size < count * 4 then
    throw "unexpected EOF"
  let mut out : Array Int := Array.mkEmpty count
  for i in [:count] do
    let v := i32FromLE bytes (i * 4)
    out := out.push v
  return out

def matrixNormOneInfScaledFromBytes (bytes : ByteArray) (rows cols scalePow10 : Nat) :
    Except String (Nat × Nat) := do
  if rows = 0 || cols = 0 then
    return (0, 0)
  let count := rows * cols
  if bytes.size < count * 8 then
    throw "unexpected EOF"
  let mut maxRowSum : Nat := 0
  let mut curRowSum : Nat := 0
  let mut colSums : Array Nat := Array.replicate cols 0
  for i in [:count] do
    let bits := u64FromLE bytes (i * 8)
    match floatAbsCeilScaled scalePow10 bits with
    | .error e => throw e
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
  return (maxRowSum, maxColSum)

def opBoundScaledFromOneInf (rowSum colSum : Nat) : Nat :=
  max rowSum colSum

def ratOfScaledNat (scalePow10 : Nat) (x : Nat) : Rat :=
  Rat.normalize (Int.ofNat x) (Nat.pow 10 scalePow10) (den_nz := by
    have h10pos : (0 : Nat) < 10 := by decide
    exact Nat.ne_of_gt (Nat.pow_pos (n := scalePow10) h10pos))

def ratOfScaledInt (scalePow10 : Nat) (x : Int) : Rat :=
  Rat.normalize x (Nat.pow 10 scalePow10) (den_nz := by
    have h10pos : (0 : Nat) < 10 := by decide
    exact Nat.ne_of_gt (Nat.pow_pos (n := scalePow10) h10pos))

def defaultBinaryScalePow10 : Nat := 9

/-- Sum of per-head value-output norm products in scaled-int form. -/
def attnValueCoeffFromScaledPairs (scalePow10 : Nat) (pairs : Array (Int × Int)) : Rat :=
  pairs.foldl
    (fun acc p =>
      acc + ratOfScaledInt scalePow10 p.1 * ratOfScaledInt scalePow10 p.2) 0

/-! ### Derived properties -/

private theorem pure_eq_ok {ε α : Type} (x : α) : (pure x : Except ε α) = .ok x := rfl

theorem vectorMaxAbsScaledFromBytes_zero
    (bytes : ByteArray) (scalePow10 : Nat) :
    vectorMaxAbsScaledFromBytes bytes 0 scalePow10 = .ok 0 := by
  simp [vectorMaxAbsScaledFromBytes, pure_eq_ok]

theorem matrixNormInfScaledFromBytes_zero_rows
    (bytes : ByteArray) (cols scalePow10 : Nat) :
    matrixNormInfScaledFromBytes bytes 0 cols scalePow10 = .ok 0 := by
  simp [matrixNormInfScaledFromBytes, pure_eq_ok]

theorem matrixNormInfScaledFromBytes_zero_cols
    (bytes : ByteArray) (rows scalePow10 : Nat) :
    matrixNormInfScaledFromBytes bytes rows 0 scalePow10 = .ok 0 := by
  simp [matrixNormInfScaledFromBytes, pure_eq_ok]

theorem scaledFloatArrayFromBytes_zero
    (bytes : ByteArray) (scalePow10 : Nat) :
    scaledFloatArrayFromBytes bytes 0 scalePow10 = .ok #[] := by
  simp [scaledFloatArrayFromBytes, pure_eq_ok]

theorem i32ArrayFromBytes_zero (bytes : ByteArray) :
    i32ArrayFromBytes bytes 0 = .ok #[] := by
  simp [i32ArrayFromBytes, pure_eq_ok]

/-! ### Specs -/

theorem parseHeaderLine_spec_binary_pure : parseHeaderLine = parseHeaderLine := rfl
theorem readHeaderNat_spec_binary_pure : readHeaderNat = readHeaderNat := rfl
theorem parseBinaryHeaderLines_spec_binary_pure :
    parseBinaryHeaderLines = parseBinaryHeaderLines := rfl
theorem u64FromLE_spec_binary_pure : u64FromLE = u64FromLE := rfl
theorem u32FromLE_spec_binary_pure : u32FromLE = u32FromLE := rfl
theorem i32FromLE_spec_binary_pure : i32FromLE = i32FromLE := rfl
theorem pow2Nat_spec_binary_pure : pow2Nat = pow2Nat := rfl
theorem ceilDivNat_spec_binary_pure : ceilDivNat = ceilDivNat := rfl
theorem floatAbsCeilScaled_spec_binary_pure : floatAbsCeilScaled = floatAbsCeilScaled := rfl
theorem floatScaledCeilSigned_spec_binary_pure :
    floatScaledCeilSigned = floatScaledCeilSigned := rfl
theorem vectorMaxAbsScaledFromBytes_spec_binary_pure :
    vectorMaxAbsScaledFromBytes = vectorMaxAbsScaledFromBytes := rfl
theorem matrixNormInfScaledFromBytes_spec_binary_pure :
    matrixNormInfScaledFromBytes = matrixNormInfScaledFromBytes := rfl
theorem scaledFloatArrayFromBytes_spec_binary_pure :
    scaledFloatArrayFromBytes = scaledFloatArrayFromBytes := rfl
theorem scaledFloatFromBytes_spec_binary_pure :
    scaledFloatFromBytes = scaledFloatFromBytes := rfl
theorem i32ArrayFromBytes_spec_binary_pure :
    i32ArrayFromBytes = i32ArrayFromBytes := rfl
theorem matrixNormOneInfScaledFromBytes_spec_binary_pure :
    matrixNormOneInfScaledFromBytes = matrixNormOneInfScaledFromBytes := rfl
theorem opBoundScaledFromOneInf_spec_binary_pure :
    opBoundScaledFromOneInf = opBoundScaledFromOneInf := rfl
theorem ratOfScaledNat_spec_binary_pure : ratOfScaledNat = ratOfScaledNat := rfl
theorem ratOfScaledInt_spec_binary_pure : ratOfScaledInt = ratOfScaledInt := rfl
theorem defaultBinaryScalePow10_spec_binary_pure :
    defaultBinaryScalePow10 = defaultBinaryScalePow10 := rfl
theorem attnValueCoeffFromScaledPairs_spec_binary_pure :
    attnValueCoeffFromScaledPairs = attnValueCoeffFromScaledPairs := rfl

end Nfp.Sound
