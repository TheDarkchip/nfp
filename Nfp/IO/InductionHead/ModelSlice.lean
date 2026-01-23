-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.List.Range
public import Nfp.Bounds.Cache
public import Nfp.Sound.Induction.ScoreSlice

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/-- Minimal model slice for recomputing attention scores. -/
structure ModelSlice (seq : Nat) where
  /-- Hidden dimension. -/
  dModel : Nat
  /-- Head dimension. -/
  headDim : Nat
  /-- Layer index (metadata). -/
  layer : Nat
  /-- Head index (metadata). -/
  head : Nat
  /-- Scale factor for attention scores. -/
  scoreScale : Rat
  /-- Mask value for causal attention (k > q). -/
  scoreMask : Rat
  /-- Whether to apply a causal mask to attention scores. -/
  maskCausal : Bool
  /-- Optional decimal precision for fast integer score checks. -/
  decimals? : Option Nat
  /-- Residual inputs (post-layernorm). -/
  resid : Fin seq → Fin dModel → Rat
  /-- Q projection slice. -/
  wq : Fin dModel → Fin headDim → Rat
  /-- K projection slice. -/
  wk : Fin dModel → Fin headDim → Rat
  /-- Q bias slice. -/
  bq : Fin headDim → Rat
  /-- K bias slice. -/
  bk : Fin headDim → Rat

/-! Internal helpers for fast integer score checks. -/

private def stripFactorAux (p : Nat) (hp : 1 < p) :
    Nat → Nat → Nat × Nat
  | n, k =>
      if h0 : n = 0 then
        (k, 0)
      else if _ : n % p = 0 then
        stripFactorAux p hp (n / p) (k + 1)
      else
        (k, n)
termination_by n _ => n
decreasing_by
  have hn : 0 < n := Nat.pos_of_ne_zero h0
  exact Nat.div_lt_self hn hp

/-- Count the exponent of `p` and return the remaining cofactor. -/
private def stripFactor (p : Nat) (hp : 1 < p) (n : Nat) : Nat × Nat :=
  stripFactorAux p hp n 0

/-- Minimal decimal exponent for a denominator with only 2/5 factors. -/
private def decimalsNeeded? (x : Rat) : Option Nat :=
  let n := x.den
  let (a, n1) := stripFactor 2 (by decide) n
  let (b, n2) := stripFactor 5 (by decide) n1
  if n2 = 1 then
    some (Nat.max a b)
  else
    none

/-- Update the inferred decimal exponent with a new rational. -/
private def updateDec? (acc : Nat) (x : Rat) : Option Nat := do
  let d ← decimalsNeeded? x
  pure (Nat.max acc d)

/-- Fold over a vector of rationals to infer a decimal exponent. -/
private def foldDec1? (n : Nat) (acc : Nat) (f : Fin n → Rat) : Option Nat :=
  (List.finRange n).foldlM (m := Option) (fun acc i =>
      updateDec? acc (f i)) acc

/-- Fold over a matrix of rationals to infer a decimal exponent. -/
private def foldDec2? (m n : Nat) (acc : Nat) (f : Fin m → Fin n → Rat) :
    Option Nat :=
  (List.finRange m).foldlM (m := Option) (fun acc q =>
      (List.finRange n).foldlM (m := Option) (fun acc i =>
          updateDec? acc (f q i)) acc) acc

/-- Infer a decimal exponent from model and score data. -/
private def inferDecimals? {seq : Nat} (slice : ModelSlice seq)
    (scores : Fin seq → Fin seq → Rat) : Option Nat := do
  let acc ← foldDec2? seq slice.dModel 0 slice.resid
  let acc ← foldDec2? slice.dModel slice.headDim acc slice.wq
  let acc ← foldDec2? slice.dModel slice.headDim acc slice.wk
  let acc ← foldDec1? slice.headDim acc slice.bq
  let acc ← foldDec1? slice.headDim acc slice.bk
  foldDec2? seq seq acc scores

/-- Scale a rational to a numerator at a fixed denominator when compatible. -/
private def ratScaledNum? (den : Nat) (x : Rat) : Option Int :=
  if _ : x.den ∣ den then
    let k := den / x.den
    some (x.num * Int.ofNat k)
  else
    none

/-- Scale a row of rationals to integer numerators at a fixed denominator. -/
private def scaleRow? (n : Nat) (den : Nat) (f : Fin n → Rat) :
    Option (Array Int) :=
  (List.finRange n).foldlM (m := Option) (fun row i => do
      let v ← ratScaledNum? den (f i)
      pure (row.push v)) (#[] : Array Int)

/-- Scale a matrix of rationals to integer numerators at a fixed denominator. -/
private def scaleMatrix? (m n : Nat) (den : Nat) (f : Fin m → Fin n → Rat) :
    Option (Array (Array Int)) :=
  (List.finRange m).foldlM (m := Option) (fun rows q => do
      let row ← scaleRow? n den (f q)
      pure (rows.push row)) (#[] : Array (Array Int))

/-- Integer dot product over `Fin n`. -/
private def dotFinInt (n : Nat) (x y : Fin n → Int) : Int :=
  Linear.foldlFin n (fun acc i => acc + x i * y i) 0

/-- Slow score check using the exact rational formula. -/
def scoresMatchModelSliceSlow {seq : Nat} (slice : ModelSlice seq)
    (scores : Fin seq → Fin seq → Rat) : Bool :=
  let scoresRef :=
    Sound.Induction.scoresRatOfSlice (seq := seq)
      (dModel := slice.dModel) (dHead := slice.headDim)
      slice.scoreScale slice.scoreMask slice.maskCausal
      slice.resid slice.wq slice.bq slice.wk slice.bk
  (List.finRange seq).all (fun q =>
    (List.finRange seq).all (fun k =>
      decide (scores q k = scoresRef q k)))

/-- Fast integer score check, when decimal scaling is available. -/
def scoresMatchModelSliceFast? {seq : Nat} (slice : ModelSlice seq)
    (scores : Fin seq → Fin seq → Rat) : Option Bool := do
  let dec ←
    match slice.decimals? with
    | some dec => some dec
    | none => inferDecimals? (seq := seq) slice scores
  let baseDen := Nat.pow 10 dec
  let baseDenInt := Int.ofNat baseDen
  let baseDenSq := baseDen * baseDen
  let scaleNum := slice.scoreScale.num
  let scaleDen := slice.scoreScale.den
  let scoreDenNat := baseDenSq * baseDenSq * scaleDen
  let scoreMaskNum ← ratScaledNum? scoreDenNat slice.scoreMask
  let residArr ← scaleMatrix? seq slice.dModel baseDen slice.resid
  let wqArr ← scaleMatrix? slice.dModel slice.headDim baseDen slice.wq
  let wkArr ← scaleMatrix? slice.dModel slice.headDim baseDen slice.wk
  let bqArr ← scaleRow? slice.headDim baseDen slice.bq
  let bkArr ← scaleRow? slice.headDim baseDen slice.bk
  let scoresArr ← scaleMatrix? seq seq scoreDenNat scores
  let residInt : Fin seq → Fin slice.dModel → Int := fun q i =>
    match residArr[q.1]? with
    | some row => (row[i.1]?).getD 0
    | none => 0
  let wqInt : Fin slice.dModel → Fin slice.headDim → Int := fun i d =>
    match wqArr[i.1]? with
    | some row => (row[d.1]?).getD 0
    | none => 0
  let wkInt : Fin slice.dModel → Fin slice.headDim → Int := fun i d =>
    match wkArr[i.1]? with
    | some row => (row[d.1]?).getD 0
    | none => 0
  let bqInt : Fin slice.headDim → Int := fun d =>
    (bqArr[d.1]?).getD 0
  let bkInt : Fin slice.headDim → Int := fun d =>
    (bkArr[d.1]?).getD 0
  let qNumArr : Array (Array Int) :=
    Array.ofFn (fun q : Fin seq =>
      Array.ofFn (fun d : Fin slice.headDim =>
        let dot := dotFinInt slice.dModel
          (fun i => residInt q i) (fun i => wqInt i d)
        dot + bqInt d * baseDenInt))
  let kNumArr : Array (Array Int) :=
    Array.ofFn (fun k : Fin seq =>
      Array.ofFn (fun d : Fin slice.headDim =>
        let dot := dotFinInt slice.dModel
          (fun i => residInt k i) (fun i => wkInt i d)
        dot + bkInt d * baseDenInt))
  let qNum : Fin seq → Fin slice.headDim → Int := fun q d =>
    match qNumArr[q.1]? with
    | some row => (row[d.1]?).getD 0
    | none => 0
  let kNum : Fin seq → Fin slice.headDim → Int := fun k d =>
    match kNumArr[k.1]? with
    | some row => (row[d.1]?).getD 0
    | none => 0
  let scoresInt : Fin seq → Fin seq → Int := fun q k =>
    match scoresArr[q.1]? with
    | some row => (row[k.1]?).getD 0
    | none => 0
  let ok :=
    (List.finRange seq).all (fun q =>
      (List.finRange seq).all (fun k =>
        if slice.maskCausal && k > q then
          scoresInt q k = scoreMaskNum
        else
          let scoreNum :=
            scaleNum * dotFinInt slice.headDim
              (fun d => qNum q d) (fun d => kNum k d)
          scoresInt q k = scoreNum))
  return ok

/-- Check whether scores match the model slice computation. -/
def scoresMatchModelSlice {seq : Nat} (slice : ModelSlice seq)
    (scores : Fin seq → Fin seq → Rat) : Bool :=
  match scoresMatchModelSliceFast? (seq := seq) slice scores with
  | some ok => ok
  | none => scoresMatchModelSliceSlow (seq := seq) slice scores

end InductionHeadCert

end IO

end Nfp
