-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Insert
public import Nfp.IO.Parse.Basic
public import Nfp.IO.Util

/-!
Untrusted parsing and checking for stripe-attention certificates.

Sequence indices in the payload are 0-based (file-format convention) and are converted
to `Fin` indices internally. The certificate must declare `kind stripe`.
-/

public section

namespace Nfp

namespace IO

open Nfp.IO.Parse

namespace Stripe

open scoped BigOperators

/-- Certificate payload for stripe-attention bounds (Rat-valued). -/
structure StripeCert (seq : Nat) where
  /-- Stripe period (q ↦ q - period). -/
  period : Nat
  /-- Lower bound on stripe-mean attention. -/
  stripeMeanLB : Rat
  /-- Lower bound on stripe-top1 rate. -/
  stripeTop1LB : Rat
  /-- Attention weights. -/
  weights : Fin seq → Fin seq → Rat

/-- Index used for the stripe key at query `q`. -/
def stripeIndex {seq : Nat} (period : Nat) (q : Fin seq) : Fin seq :=
  ⟨q.1 - period, lt_of_le_of_lt (Nat.sub_le _ _) q.2⟩

private def activeSet (seq period : Nat) : Finset (Fin seq) :=
  (Finset.univ : Finset (Fin seq)).filter (fun q => period ≤ q.1)

/-- Stripe-mean attention for active queries, if any. -/
def stripeMean {seq : Nat} [NeZero seq] (c : StripeCert seq) : Option Rat := by
  let active := activeSet seq c.period
  if h : active.card = 0 then
    exact none
  else
    let sum :=
      active.sum (fun q => c.weights q (stripeIndex c.period q))
    exact some (sum / (active.card : Rat))

private def rowMax {seq : Nat} [NeZero seq] (c : StripeCert seq) (q : Fin seq) : Rat := by
  classical
  have hpos : 0 < seq := Nat.pos_of_ne_zero (NeZero.ne _)
  haveI : Nonempty (Fin seq) := ⟨⟨0, hpos⟩⟩
  let vals := (Finset.univ : Finset (Fin seq)).image (fun k => c.weights q k)
  have hnonempty_univ : (Finset.univ : Finset (Fin seq)).Nonempty :=
    Finset.univ_nonempty
  have hnonempty : vals.Nonempty := hnonempty_univ.image (fun k => c.weights q k)
  exact vals.max' hnonempty

/-- Stripe-top1 rate for active queries, if any. -/
def stripeTop1 {seq : Nat} [NeZero seq] (c : StripeCert seq) : Option Rat := by
  let active := activeSet seq c.period
  if h : active.card = 0 then
    exact none
  else
    let good :=
      active.filter (fun q => c.weights q (stripeIndex c.period q) ≥ rowMax c q)
    exact some (good.card / (active.card : Rat))

/-- Boolean checker for stripe-attention certificates. -/
def checkStripeCert {seq : Nat} [NeZero seq] (c : StripeCert seq) : Bool :=
  decide (c.period < seq) &&
    match stripeMean (seq := seq) c with
    | none => false
    | some mean => decide (c.stripeMeanLB ≤ mean) &&
      match stripeTop1 (seq := seq) c with
      | none => false
      | some top1 => decide (c.stripeTop1LB ≤ top1)

/-! Parsing -/

/-- State for parsing stripe-attention certificates. -/
structure ParseState (seq : Nat) where
  /-- Optional certificate kind tag. -/
  kind : Option String
  /-- Optional period. -/
  period : Option Nat
  /-- Optional stripe-mean lower bound. -/
  stripeMeanLB : Option Rat
  /-- Optional stripe-top1 lower bound. -/
  stripeTop1LB : Option Rat
  /-- Optional weight matrix entries. -/
  weights : Array (Array (Option Rat))

/-- Initialize a parse state. -/
def initState (seq : Nat) : ParseState seq :=
  let row : Array (Option Rat) := Array.replicate seq none
  { kind := none
    period := none
    stripeMeanLB := none
    stripeTop1LB := none
    weights := Array.replicate seq row }

private def toIndex0 {seq : Nat} (label : String) (idx : Nat) : Except String (Fin seq) := do
  if h : idx < seq then
    return ⟨idx, h⟩
  else
    throw s!"{label} index out of range: {idx}"

private def setMatrixEntry {seq : Nat} (mat : Array (Array (Option Rat)))
    (q k : Nat) (v : Rat) : Except String (Array (Array (Option Rat))) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  let kFin ← toIndex0 (seq := seq) "k" k
  let row := mat[qFin.1]!
  match row[kFin.1]! with
  | some _ =>
      throw s!"duplicate matrix entry at ({q}, {k})"
  | none =>
      let row' := row.set! kFin.1 (some v)
      return mat.set! qFin.1 row'

/-- Parse a tokenized line into the parse state. -/
def parseLine {seq : Nat} (st : ParseState seq) (tokens : List String) :
    Except String (ParseState seq) := do
  match tokens with
  | ["kind", k] =>
      if st.kind.isSome then
        throw "duplicate kind entry"
      else
        return { st with kind := some k }
  | ["period", n] =>
      if st.period.isSome then
        throw "duplicate period entry"
      else
        return { st with period := some (← parseNat n) }
  | ["stripe-mean-lb", val] =>
      if st.stripeMeanLB.isSome then
        throw "duplicate stripe-mean-lb entry"
      else
        return { st with stripeMeanLB := some (← parseRat val) }
  | ["stripe-top1-lb", val] =>
      if st.stripeTop1LB.isSome then
        throw "duplicate stripe-top1-lb entry"
      else
        return { st with stripeTop1LB := some (← parseRat val) }
  | ["weight", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.weights (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with weights := mat }
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

/-- Extract the `seq` header from tokenized lines. -/
def parseSeq (tokens : List (List String)) : Except String Nat := do
  let mut seq? : Option Nat := none
  for t in tokens do
    match t with
    | ["seq", n] =>
        if seq?.isSome then
          throw "duplicate seq entry"
        else
          seq? := some (← parseNat n)
    | _ => pure ()
  match seq? with
  | some v => pure v
  | none => throw "missing seq entry"

private def finalizeState {seq : Nat} (st : ParseState seq) :
    Except String (StripeCert seq) := do
  let kind ←
    match st.kind with
    | some v => pure v
    | none => throw "missing kind entry"
  if kind != "stripe" then
    throw s!"unexpected kind: {kind}"
  let period ←
    match st.period with
    | some v => pure v
    | none => throw "missing period entry"
  let stripeMeanLB ←
    match st.stripeMeanLB with
    | some v => pure v
    | none => throw "missing stripe-mean-lb entry"
  let stripeTop1LB ←
    match st.stripeTop1LB with
    | some v => pure v
    | none => throw "missing stripe-top1-lb entry"
  if !st.weights.all (fun row => row.all Option.isSome) then
    throw "missing weight entries"
  let weightsFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.weights[q.1]!
    (row[k.1]!).getD 0
  pure
    { period := period
      stripeMeanLB := stripeMeanLB
      stripeTop1LB := stripeTop1LB
      weights := weightsFun }

end Stripe

/-- Parse an explicit stripe-attention certificate from a text payload. -/
def parseStripeCert (input : String) :
    Except String (Sigma Stripe.StripeCert) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let seq ← Stripe.parseSeq tokens
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let st0 : Stripe.ParseState seq := Stripe.initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => Stripe.parseLine st t) st0
      let cert ← Stripe.finalizeState st
      return ⟨seq, cert⟩

/-- Load a stripe-attention certificate from disk. -/
def loadStripeCert (path : System.FilePath) :
    IO (Except String (Sigma Stripe.StripeCert)) := do
  let data ← IO.FS.readFile path
  return parseStripeCert data

/-- Check a stripe-attention certificate from disk. -/
def runStripeCertCheck (certPath : System.FilePath)
    (minStripeMeanStr? : Option String) (minStripeTop1Str? : Option String) : IO UInt32 := do
  let minMean?E := parseRatOpt "min-stripe-mean" minStripeMeanStr?
  let minTop1?E := parseRatOpt "min-stripe-top1" minStripeTop1Str?
  match minMean?E, minTop1?E with
  | Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minMean?, Except.ok minTop1? =>
      let parsed ← loadStripeCert certPath
      match parsed with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          match seq with
          | 0 =>
              IO.eprintln "error: seq must be positive"
              return 2
          | Nat.succ n =>
              let seq := Nat.succ n
              let _ : NeZero seq := ⟨by simp⟩
              if !Stripe.checkStripeCert cert then
                IO.eprintln "error: stripe certificate rejected"
                return 2
              let minMean := minMean?.getD cert.stripeMeanLB
              let minTop1 := minTop1?.getD cert.stripeTop1LB
              let meanOk :=
                match Stripe.stripeMean (seq := seq) cert with
                | some mean => decide (minMean ≤ mean)
                | none => false
              let top1Ok :=
                match Stripe.stripeTop1 (seq := seq) cert with
                | some top1 => decide (minTop1 ≤ top1)
                | none => false
              if !meanOk then
                IO.eprintln "error: stripe-mean below minimum"
                return 2
              if !top1Ok then
                IO.eprintln "error: stripe-top1 below minimum"
                return 2
              IO.println s!"ok: stripe certificate checked (seq={seq})"
              return 0

end IO

end Nfp
