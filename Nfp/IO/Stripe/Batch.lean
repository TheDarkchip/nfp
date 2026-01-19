-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Stripe.Cert
public import Nfp.IO.Parse.Basic
public import Nfp.IO.Util

/-!
Batch checking for stripe-attention certificates.
-/

public section

namespace Nfp

namespace IO

open Nfp.IO.Parse

namespace StripeBatch

/-- A single batch item (certificate path). -/
structure BatchItem where
  /-- Path to the stripe certificate file. -/
  certPath : System.FilePath

/-- Parsed batch-level options. -/
structure BatchOpts where
  /-- Optional per-item minimum stripe mean. -/
  minStripeMean? : Option Rat
  /-- Optional per-item minimum stripe top1. -/
  minStripeTop1? : Option Rat
  /-- Optional minimum average stripe mean across items. -/
  minAvgStripeMean? : Option Rat
  /-- Optional minimum average stripe top1 across items. -/
  minAvgStripeTop1? : Option Rat

/-- Parser state for batch files. -/
structure ParseState where
  /-- Items accumulated so far. -/
  items : Array BatchItem
  /-- Optional per-item minimum stripe mean. -/
  minStripeMean? : Option Rat
  /-- Optional per-item minimum stripe top1. -/
  minStripeTop1? : Option Rat
  /-- Optional minimum average stripe mean across items. -/
  minAvgStripeMean? : Option Rat
  /-- Optional minimum average stripe top1 across items. -/
  minAvgStripeTop1? : Option Rat

/-- Initialize a batch parse state. -/
def initState : ParseState :=
  { items := #[]
    minStripeMean? := none
    minStripeTop1? := none
    minAvgStripeMean? := none
    minAvgStripeTop1? := none }

private def ratToString (x : Rat) : String :=
  if x.den = 1 then
    s!"{x.num}"
  else
    s!"{x.num}/{x.den}"

private def setOptRat (label : String) (st : ParseState) (val : Rat) :
    Except String ParseState := do
  match label with
  | "min-stripe-mean" =>
      if st.minStripeMean?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minStripeMean? := some val }
  | "min-stripe-top1" =>
      if st.minStripeTop1?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minStripeTop1? := some val }
  | "min-avg-stripe-mean" =>
      if st.minAvgStripeMean?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minAvgStripeMean? := some val }
  | "min-avg-stripe-top1" =>
      if st.minAvgStripeTop1?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minAvgStripeTop1? := some val }
  | _ => throw s!"unknown rat option: {label}"

/-- Parse a tokenized line into the batch parse state. -/
def parseLine (st : ParseState) (tokens : List String) : Except String ParseState := do
  match tokens with
  | ["item", certPath] =>
      let item : BatchItem := { certPath := certPath }
      pure { st with items := st.items.push item }
  | ["min-stripe-mean", v] =>
      setOptRat "min-stripe-mean" st (← parseRat v)
  | ["min-stripe-top1", v] =>
      setOptRat "min-stripe-top1" st (← parseRat v)
  | ["min-avg-stripe-mean", v] =>
      setOptRat "min-avg-stripe-mean" st (← parseRat v)
  | ["min-avg-stripe-top1", v] =>
      setOptRat "min-avg-stripe-top1" st (← parseRat v)
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeOpts (st : ParseState) : BatchOpts :=
  { minStripeMean? := st.minStripeMean?
    minStripeTop1? := st.minStripeTop1?
    minAvgStripeMean? := st.minAvgStripeMean?
    minAvgStripeTop1? := st.minAvgStripeTop1? }

private def checkOne (item : BatchItem) (opts : BatchOpts) :
    IO (Except String (Rat × Rat)) := do
  let parsed ← loadStripeCert item.certPath
  match parsed with
  | Except.error msg => return Except.error msg
  | Except.ok ⟨seq, cert⟩ =>
      match seq with
      | 0 => return Except.error "seq must be positive"
      | Nat.succ n =>
          let seq := Nat.succ n
          let _ : NeZero seq := ⟨by simp⟩
          if !Stripe.checkStripeCert cert then
            return Except.error "stripe certificate rejected"
          let mean? := Stripe.stripeMean (seq := seq) cert
          let top1? := Stripe.stripeTop1 (seq := seq) cert
          match mean?, top1? with
          | some mean, some top1 =>
              if let some minMean := opts.minStripeMean? then
                if mean < minMean then
                  return Except.error
                    s!"stripe-mean {ratToString mean} below minimum {ratToString minMean}"
              if let some minTop1 := opts.minStripeTop1? then
                if top1 < minTop1 then
                  return Except.error
                    s!"stripe-top1 {ratToString top1} below minimum {ratToString minTop1}"
              return Except.ok (mean, top1)
          | _, _ =>
              return Except.error "empty active set for stripe stats"

end StripeBatch

/-- Check a batch of stripe-attention certificates from disk. -/
def runStripeBatchCheck (batchPath : System.FilePath) : IO UInt32 := do
  let data ← IO.FS.readFile batchPath
  let lines := data.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let st0 := StripeBatch.initState
  let stE := tokens.foldlM (fun st t => StripeBatch.parseLine st t) st0
  match stE with
  | Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok st =>
      if st.items.isEmpty then
        IO.eprintln "error: batch file has no items"
        return 2
      let opts := StripeBatch.finalizeOpts st
      let mut sumMean : Rat := 0
      let mut sumTop1 : Rat := 0
      let mut count : Nat := 0
      for item in st.items do
        let res ← StripeBatch.checkOne item opts
        match res with
        | Except.error msg =>
            IO.eprintln s!"error: {msg}"
            return 2
        | Except.ok (mean, top1) =>
            sumMean := sumMean + mean
            sumTop1 := sumTop1 + top1
            count := count + 1
      if let some minAvg := opts.minAvgStripeMean? then
        let avg : Rat := sumMean / (count : Rat)
        if avg < minAvg then
          IO.eprintln
            s!"error: avg stripe-mean {StripeBatch.ratToString avg} \
            below minimum {StripeBatch.ratToString minAvg}"
          return 2
      if let some minAvg := opts.minAvgStripeTop1? then
        let avg : Rat := sumTop1 / (count : Rat)
        if avg < minAvg then
          IO.eprintln
            s!"error: avg stripe-top1 {StripeBatch.ratToString avg} \
            below minimum {StripeBatch.ratToString minAvg}"
          return 2
      IO.println s!"ok: stripe batch checked ({st.items.size} items)"
      return 0

end IO

end Nfp
