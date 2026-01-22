-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.Circuit.Cert.ValueRange
public import Nfp.IO.InductionHead.Cert
public import Nfp.IO.Parse.Basic
public import Nfp.IO.Stripe
public import Nfp.IO.Util
public import Nfp.Model.InductionPrompt

/-!
Batch checking for induction-head certificates.

The batch file lists certificate paths (and optional token lists) and applies the same
verification checks to each. Sequence indices in per-cert/token payloads are 0-based.
Both `kind onehot-approx` and `kind induction-aligned` are supported; the latter
uses prefix-matching metrics instead of softmax-margin/onehot gates.
-/

public section

namespace Nfp

namespace IO

open Nfp.Circuit
open Nfp.IO.Parse

namespace InductionHeadBatch

/-- A single batch item (certificate path plus optional token list). -/
structure BatchItem where
  /-- Path to the certificate file. -/
  certPath : System.FilePath
  /-- Optional path to the token list file. -/
  tokensPath? : Option System.FilePath

/-- Parsed batch-level options. -/
structure BatchOpts where
  /-- Optional minimum active count. -/
  minActive? : Option Nat
  /-- Optional minimum pass count for batch items. -/
  minPass? : Option Nat
  /-- Optional per-item logit-diff lower bound. -/
  minLogitDiff? : Option Rat
  /-- Minimum margin threshold. -/
  minMargin : Rat
  /-- Maximum eps threshold. -/
  maxEps : Rat
  /-- Optional per-item minimum stripe mean. -/
  minStripeMean? : Option Rat
  /-- Optional average logit-diff lower bound across items. -/
  minAvgLogitDiff? : Option Rat

/-- Parser state for batch files. -/
structure ParseState where
  /-- Items accumulated so far. -/
  items : Array BatchItem
  /-- Optional minimum active count. -/
  minActive? : Option Nat
  /-- Optional minimum pass count. -/
  minPass? : Option Nat
  /-- Optional per-item logit-diff lower bound. -/
  minLogitDiff? : Option Rat
  /-- Optional minimum margin threshold. -/
  minMargin? : Option Rat
  /-- Optional maximum eps threshold. -/
  maxEps? : Option Rat
  /-- Optional per-item minimum stripe mean. -/
  minStripeMean? : Option Rat
  /-- Optional average logit-diff lower bound. -/
  minAvgLogitDiff? : Option Rat

/-- Initialize a batch parse state. -/
def initState : ParseState :=
  { items := #[]
    minActive? := none
    minPass? := none
    minLogitDiff? := none
    minMargin? := none
    maxEps? := none
    minStripeMean? := none
    minAvgLogitDiff? := none }

private def ratToString (x : Rat) : String :=
  if x.den = 1 then
    s!"{x.num}"
  else
    s!"{x.num}/{x.den}"

private def setOptNat (label : String) (st : ParseState) (val : Nat) :
    Except String ParseState := do
  match label with
  | "min-active" =>
      if st.minActive?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minActive? := some val }
  | "min-pass" =>
      if st.minPass?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minPass? := some val }
  | _ => throw s!"unknown nat option: {label}"

private def setOptRat (label : String) (st : ParseState) (val : Rat) :
    Except String ParseState := do
  match label with
  | "min-logit-diff" =>
      if st.minLogitDiff?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minLogitDiff? := some val }
  | "min-margin" =>
      if st.minMargin?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minMargin? := some val }
  | "max-eps" =>
      if st.maxEps?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with maxEps? := some val }
  | "min-stripe-mean" =>
      if st.minStripeMean?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minStripeMean? := some val }
  | "min-avg-logit-diff" =>
      if st.minAvgLogitDiff?.isSome then
        throw s!"duplicate {label} entry"
      else
        pure { st with minAvgLogitDiff? := some val }
  | _ => throw s!"unknown rat option: {label}"

/-- Parse a tokenized line into the batch parse state. -/
def parseLine (st : ParseState) (tokens : List String) : Except String ParseState := do
  match tokens with
  | ["item", certPath] =>
      let item : BatchItem := { certPath := certPath, tokensPath? := none }
      pure { st with items := st.items.push item }
  | ["item", certPath, tokensPath] =>
      let item : BatchItem := { certPath := certPath, tokensPath? := some tokensPath }
      pure { st with items := st.items.push item }
  | ["min-active", n] =>
      setOptNat "min-active" st (← parseNat n)
  | ["min-pass", n] =>
      setOptNat "min-pass" st (← parseNat n)
  | ["min-logit-diff", v] =>
      setOptRat "min-logit-diff" st (← parseRat v)
  | ["min-margin", v] =>
      setOptRat "min-margin" st (← parseRat v)
  | ["max-eps", v] =>
      setOptRat "max-eps" st (← parseRat v)
  | ["min-stripe-mean", v] =>
      setOptRat "min-stripe-mean" st (← parseRat v)
  | ["min-avg-logit-diff", v] =>
      setOptRat "min-avg-logit-diff" st (← parseRat v)
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeOpts (st : ParseState) : BatchOpts :=
  { minActive? := st.minActive?
    minPass? := st.minPass?
    minLogitDiff? := st.minLogitDiff?
    minMargin := st.minMargin?.getD (0 : Rat)
    maxEps := st.maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
    minStripeMean? := st.minStripeMean?
    minAvgLogitDiff? := st.minAvgLogitDiff? }

private def effectiveMinLogitDiff (minLogitDiff? : Option Rat)
    (direction? : Option Circuit.DirectionSpec) : Option Rat :=
  match minLogitDiff? with
  | some v => some v
  | none =>
      match direction? with
      | some _ => some (0 : Rat)
      | none => none

private def tokensPeriodic {seq : Nat} (period : Nat) (tokens : Fin seq → Nat) : Bool :=
  (List.finRange seq).all (fun q =>
    if period ≤ q.val then
      decide (tokens q = tokens (Model.prevOfPeriod (seq := seq) period q))
    else
      true)

private def checkOne (item : BatchItem) (opts : BatchOpts)
    (requireLogitDiff : Bool) : IO (Except String (Option Rat)) := do
  let parsed ← loadInductionHeadCert item.certPath
  match parsed with
  | Except.error msg => return Except.error msg
  | Except.ok ⟨seq, payload⟩ =>
      match seq with
      | 0 => return Except.error "seq must be positive"
      | Nat.succ n =>
          let seq := Nat.succ n
          let _ : NeZero seq := ⟨by simp⟩
          let cert := payload.cert
          let kind := payload.kind
          let period? := payload.period?
          let copyLogits? := payload.copyLogits?
          if kind != "onehot-approx" && kind != "induction-aligned" then
            return Except.error s!"unexpected kind {kind}"
          if kind = "onehot-approx" then
            if opts.minStripeMean?.isSome then
              return Except.error
                "stripe thresholds are not used for onehot-approx"
          if kind = "induction-aligned" then
            let period ←
              match period? with
              | some v => pure v
              | none =>
                  return Except.error "missing period entry for induction-aligned"
            if period = 0 then
              return Except.error "period must be positive for induction-aligned"
            if seq ≤ period then
              return Except.error "period must be less than seq for induction-aligned"
            let expectedActive := Model.activeOfPeriod (seq := seq) period
            if !decide (cert.active = expectedActive) then
              return Except.error "active set does not match induction-aligned period"
            let prevOk :=
              (List.finRange seq).all (fun q =>
                decide (cert.prev q = Model.prevOfPeriod (seq := seq) period q))
            if !prevOk then
              return Except.error "prev map does not match induction-aligned period"
            if opts.minLogitDiff?.isSome || opts.minAvgLogitDiff?.isSome ||
                opts.minMargin != 0 || opts.maxEps != ratRoundDown (Rat.divInt 1 2) then
              return Except.error
                "min-logit-diff/min-margin/max-eps are not used for induction-aligned"
            if copyLogits?.isNone then
              return Except.error "missing copy-logit entries for induction-aligned"
          let activeCount := cert.active.card
          let defaultMinActive := max 1 (seq / 8)
          let minActive := opts.minActive?.getD defaultMinActive
          if activeCount < minActive then
            return Except.error
              s!"active queries {activeCount} below minimum {minActive}"
          if kind = "onehot-approx" then
            if !Circuit.checkInductionHeadCert cert then
              return Except.error "induction-head certificate rejected"
            if cert.margin < opts.minMargin then
              return Except.error
                s!"margin {ratToString cert.margin} below minimum {ratToString opts.minMargin}"
            if opts.maxEps < cert.eps then
              return Except.error
                s!"eps {ratToString cert.eps} above maximum {ratToString opts.maxEps}"
          if let some tokensPath := item.tokensPath? then
            let tokensParsed ← loadInductionHeadTokens tokensPath
            match tokensParsed with
            | Except.error msg => return Except.error msg
            | Except.ok ⟨seqTokens, tokens⟩ =>
                if hseq : seqTokens = seq then
                  let tokens' : Fin seq → Nat := by
                    simpa [hseq] using tokens
                  if kind = "induction-aligned" then
                    let period ←
                      match period? with
                      | some v => pure v
                      | none =>
                          return Except.error "missing period entry for induction-aligned"
                    if !tokensPeriodic (seq := seq) period tokens' then
                      return Except.error "tokens are not periodic for induction-aligned period"
                  else
                    let activeTokens := Model.activeOfTokens (seq := seq) tokens'
                    if !decide (cert.active ⊆ activeTokens) then
                      return Except.error "active set not contained in token repeats"
                    let prevTokens := Model.prevOfTokens (seq := seq) tokens'
                    let prevOk :=
                      (List.finRange seq).all (fun q =>
                        if decide (q ∈ cert.active) then
                          decide (prevTokens q = cert.prev q)
                        else
                          true)
                    if !prevOk then
                      return Except.error "prev map does not match tokens on active queries"
                else
                  return Except.error
                    s!"tokens seq {seqTokens} does not match cert seq {seq}"
          if kind = "induction-aligned" then
            let period ←
              match period? with
              | some v => pure v
              | none =>
                  return Except.error "missing period entry for induction-aligned"
            let stripeCert : Stripe.StripeCert seq :=
              { period := period
                stripeMeanLB := 0
                stripeTop1LB := 0
                weights := cert.weights }
            let stripeMean? := Stripe.stripeMean (seq := seq) stripeCert
            match stripeMean? with
            | some mean =>
                let minMean := opts.minStripeMean?.getD (0 : Rat)
                if mean < minMean then
                  return Except.error
                    s!"stripe-mean {ratToString mean} below minimum {ratToString minMean}"
            | none =>
                return Except.error "empty active set for stripe stats"
            return Except.ok none
          let minLogitDiff? := effectiveMinLogitDiff opts.minLogitDiff? cert.values.direction
          let logitDiffLB? :=
            Circuit.logitDiffLowerBoundAt cert.active cert.prev cert.epsAt
              cert.values.lo cert.values.hi cert.values.vals
          match minLogitDiff? with
          | none =>
              if requireLogitDiff then
                return Except.error "missing direction metadata for avg logit-diff"
              else
                return Except.ok none
          | some minLogitDiff =>
              match logitDiffLB? with
              | none =>
                  return Except.error "empty active set for logit-diff bound"
              | some logitDiffLB =>
                  if logitDiffLB < minLogitDiff then
                    return Except.error
                      s!"logitDiffLB {ratToString logitDiffLB} \
                      below minimum {ratToString minLogitDiff}"
                  else
                    return Except.ok (some logitDiffLB)

end InductionHeadBatch

/-- Check a batch of induction-head certificates from disk. -/
def runInductionHeadBatchCheck (batchPath : System.FilePath) : IO UInt32 := do
  let data ← IO.FS.readFile batchPath
  let lines := data.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let st0 := InductionHeadBatch.initState
  let stE := tokens.foldlM (fun st t => InductionHeadBatch.parseLine st t) st0
  match stE with
  | Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok st =>
      if st.items.isEmpty then
        IO.eprintln "error: batch file has no items"
        return 2
      let opts := InductionHeadBatch.finalizeOpts st
      let requireLogitDiff := opts.minAvgLogitDiff?.isSome
      let mut sum : Rat := 0
      let mut count : Nat := 0
      let mut passCount : Nat := 0
      let mut failCount : Nat := 0
      let mut firstErr : Option String := none
      let total := st.items.size
      for item in st.items do
        let res ← InductionHeadBatch.checkOne item opts requireLogitDiff
        match res with
        | Except.error msg =>
            if firstErr.isNone then
              firstErr := some msg
            failCount := failCount + 1
            if opts.minPass?.isNone then
              IO.eprintln s!"error: {msg}"
              return 2
        | Except.ok logitDiffLB? =>
            passCount := passCount + 1
            if let some v := logitDiffLB? then
              sum := sum + v
              count := count + 1
      if let some minPass := opts.minPass? then
        if total < minPass then
          IO.eprintln s!"error: min-pass {minPass} exceeds total items {total}"
          return 2
        if passCount < minPass then
          let firstMsg := (firstErr.getD "unknown failure")
          IO.eprintln s!"error: only {passCount}/{total} items passed (min-pass {minPass}); \
            first failure: {firstMsg}"
          return 2
      if let some minAvg := opts.minAvgLogitDiff? then
        if count = 0 then
          IO.eprintln "error: no logit-diff bounds available for avg check"
          return 2
        let avg : Rat := sum / (count : Rat)
        if avg < minAvg then
          IO.eprintln
            s!"error: avg logitDiffLB {InductionHeadBatch.ratToString avg} \
            below minimum {InductionHeadBatch.ratToString minAvg}"
          return 2
      if opts.minPass?.isSome then
        IO.println s!"ok: batch checked ({passCount}/{total} items passed, {failCount} failed)"
      else
        IO.println s!"ok: batch checked ({total} items)"
      return 0

end IO

end Nfp
