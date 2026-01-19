-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Insert
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.IO.InductionHead.Tokens
public import Nfp.IO.Parse.Basic
public import Nfp.IO.Stripe
public import Nfp.IO.Util
public import Nfp.Model.InductionPrompt

/-!
Untrusted parsing and checking for explicit induction-head certificates.

All sequence indices in the certificate payload are 0-based (file-format convention) and
are converted to `Fin` indices internally. Supported certificate kinds:
- `kind onehot-approx` (proxy bounds only)
- `kind induction-aligned` (requires `period` and checks stripe metrics)
-/

public section

namespace Nfp

namespace IO

open Nfp.Circuit
open Nfp.IO.Parse

namespace InductionHeadCert

/-- State for parsing induction-head certificates. -/
structure ParseState (seq : Nat) where
  /-- Optional certificate kind tag. -/
  kind : Option String
  /-- Optional induction prompt period (for `kind induction-aligned`). -/
  period : Option Nat
  /-- Optional epsilon bound. -/
  eps : Option Rat
  /-- Optional margin bound. -/
  margin : Option Rat
  /-- Active query set. -/
  active : Finset (Fin seq)
  /-- Whether any active entries were parsed. -/
  activeSeen : Bool
  /-- Optional predecessor pointer per query. -/
  prev : Array (Option (Fin seq))
  /-- Optional score matrix entries. -/
  scores : Array (Array (Option Rat))
  /-- Optional weight matrix entries. -/
  weights : Array (Array (Option Rat))
  /-- Optional per-query epsilon bounds. -/
  epsAt : Array (Option Rat)
  /-- Optional per-key weight bounds. -/
  weightBoundAt : Array (Array (Option Rat))
  /-- Optional lower bound for values. -/
  lo : Option Rat
  /-- Optional upper bound for values. -/
  hi : Option Rat
  /-- Optional per-key lower bounds. -/
  valsLo : Array (Option Rat)
  /-- Optional per-key upper bounds. -/
  valsHi : Array (Option Rat)
  /-- Optional per-key exact values. -/
  vals : Array (Option Rat)
  /-- Optional direction target index. -/
  directionTarget : Option Nat
  /-- Optional direction negative index. -/
  directionNegative : Option Nat

/-- Initialize a parse state. -/
def initState (seq : Nat) : ParseState seq :=
  let row : Array (Option Rat) := Array.replicate seq none
  { kind := none
    period := none
    eps := none
    margin := none
    active := ∅
    activeSeen := false
    prev := Array.replicate seq none
    scores := Array.replicate seq row
    weights := Array.replicate seq row
    epsAt := Array.replicate seq none
    weightBoundAt := Array.replicate seq row
    lo := none
    hi := none
    valsLo := Array.replicate seq none
    valsHi := Array.replicate seq none
    vals := Array.replicate seq none
    directionTarget := none
    directionNegative := none }

private def toIndex0 {seq : Nat} (label : String) (idx : Nat) : Except String (Fin seq) := do
  if h : idx < seq then
    return ⟨idx, h⟩
  else
    throw s!"{label} index out of range: {idx}"

private def setActive {seq : Nat} (st : ParseState seq) (q : Nat) :
    Except String (ParseState seq) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  if qFin ∈ st.active then
    throw s!"duplicate active entry for q={q}"
  else
    return { st with active := insert qFin st.active, activeSeen := true }

private def setPrev {seq : Nat} (st : ParseState seq) (q k : Nat) :
    Except String (ParseState seq) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  let kFin ← toIndex0 (seq := seq) "k" k
  match st.prev[qFin.1]! with
  | some _ =>
      throw s!"duplicate prev entry for q={q}"
  | none =>
      let prev' := st.prev.set! qFin.1 (some kFin)
      return { st with prev := prev' }

private def setVecEntry {seq : Nat} (arr : Array (Option Rat)) (idx : Nat) (v : Rat) :
    Except String (Array (Option Rat)) := do
  let kFin ← toIndex0 (seq := seq) "k" idx
  match arr[kFin.1]! with
  | some _ =>
      throw s!"duplicate entry for k={idx}"
  | none =>
      return arr.set! kFin.1 (some v)

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
  | ["period", val] =>
      if st.period.isSome then
        throw "duplicate period entry"
      else
        return { st with period := some (← parseNat val) }
  | ["eps", val] =>
      if st.eps.isSome then
        throw "duplicate eps entry"
      else
        return { st with eps := some (← parseRat val) }
  | ["margin", val] =>
      if st.margin.isSome then
        throw "duplicate margin entry"
      else
        return { st with margin := some (← parseRat val) }
  | ["active", q] =>
      setActive st (← parseNat q)
  | ["prev", q, k] =>
      setPrev st (← parseNat q) (← parseNat k)
  | ["score", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.scores (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with scores := mat }
  | ["weight", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.weights (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with weights := mat }
  | ["eps-at", q, val] =>
      let arr ← setVecEntry (seq := seq) st.epsAt (← parseNat q) (← parseRat val)
      return { st with epsAt := arr }
  | ["weight-bound", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.weightBoundAt (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with weightBoundAt := mat }
  | ["lo", val] =>
      if st.lo.isSome then
        throw "duplicate lo entry"
      else
        return { st with lo := some (← parseRat val) }
  | ["hi", val] =>
      if st.hi.isSome then
        throw "duplicate hi entry"
      else
        return { st with hi := some (← parseRat val) }
  | ["val", k, val] =>
      let arr ← setVecEntry (seq := seq) st.vals (← parseNat k) (← parseRat val)
      return { st with vals := arr }
  | ["val-lo", k, val] =>
      let arr ← setVecEntry (seq := seq) st.valsLo (← parseNat k) (← parseRat val)
      return { st with valsLo := arr }
  | ["val-hi", k, val] =>
      let arr ← setVecEntry (seq := seq) st.valsHi (← parseNat k) (← parseRat val)
      return { st with valsHi := arr }
  | ["direction-target", tok] =>
      if st.directionTarget.isSome then
        throw "duplicate direction-target entry"
      else
        return { st with directionTarget := some (← parseNat tok) }
  | ["direction-negative", tok] =>
      if st.directionNegative.isSome then
        throw "duplicate direction-negative entry"
      else
        return { st with directionNegative := some (← parseNat tok) }
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

/-- Parsed induction-head certificate payload plus kind metadata. -/
structure InductionHeadCertPayload (seq : Nat) where
  /-- Certificate kind tag. -/
  kind : String
  /-- Optional prompt period (only for `kind induction-aligned`). -/
  period? : Option Nat
  /-- Verified certificate payload. -/
  cert : Circuit.InductionHeadCert seq

private def finalizeStateCore {seq : Nat} (hpos : 0 < seq) (st : ParseState seq) :
    Except String (Circuit.InductionHeadCert seq) := do
  let eps ←
    match st.eps with
    | some v => pure v
    | none => throw "missing eps entry"
  let margin ←
    match st.margin with
    | some v => pure v
    | none => throw "missing margin entry"
  let lo ←
    match st.lo with
    | some v => pure v
    | none => throw "missing lo entry"
  let hi ←
    match st.hi with
    | some v => pure v
    | none => throw "missing hi entry"
  if !st.prev.all Option.isSome then
    throw "missing prev entries"
  if !st.scores.all (fun row => row.all Option.isSome) then
    throw "missing score entries"
  if !st.weights.all (fun row => row.all Option.isSome) then
    throw "missing weight entries"
  if !st.epsAt.all Option.isSome then
    throw "missing eps-at entries"
  if !st.weightBoundAt.all (fun row => row.all Option.isSome) then
    throw "missing weight-bound entries"
  if !st.valsLo.all Option.isSome then
    throw "missing val-lo entries"
  if !st.valsHi.all Option.isSome then
    throw "missing val-hi entries"
  if !st.vals.all Option.isSome then
    throw "missing val entries"
  let defaultPrev : Fin seq := ⟨0, hpos⟩
  let prevFun : Fin seq → Fin seq := fun q =>
    (st.prev[q.1]!).getD defaultPrev
  let scoresFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.scores[q.1]!
    (row[k.1]!).getD 0
  let weightsFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.weights[q.1]!
    (row[k.1]!).getD 0
  let epsAtFun : Fin seq → Rat := fun q =>
    (st.epsAt[q.1]!).getD 0
  let weightBoundAtFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.weightBoundAt[q.1]!
    (row[k.1]!).getD 0
  let valsLoFun : Fin seq → Rat := fun k =>
    (st.valsLo[k.1]!).getD 0
  let valsHiFun : Fin seq → Rat := fun k =>
    (st.valsHi[k.1]!).getD 0
  let valsFun : Fin seq → Rat := fun k =>
    (st.vals[k.1]!).getD 0
  let direction ←
    match st.directionTarget, st.directionNegative with
    | none, none => pure none
    | some target, some negative =>
        pure (some { target := target, negative := negative })
    | _, _ =>
        throw "direction metadata requires both direction-target and direction-negative"
  let values : Circuit.ValueIntervalCert seq :=
    { lo := lo
      hi := hi
      valsLo := valsLoFun
      valsHi := valsHiFun
      vals := valsFun
      direction := direction }
  let active :=
    if st.activeSeen then
      st.active
    else
      (Finset.univ : Finset (Fin seq)).erase defaultPrev
  pure
    { eps := eps
      epsAt := epsAtFun
      weightBoundAt := weightBoundAtFun
      margin := margin
      active := active
      prev := prevFun
      scores := scoresFun
      weights := weightsFun
      values := values }

end InductionHeadCert

/-- Parse an explicit induction-head certificate from a text payload. -/
def parseInductionHeadCert (input : String) :
    Except String (Sigma InductionHeadCert.InductionHeadCertPayload) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let seq ← InductionHeadCert.parseSeq tokens
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let hpos : 0 < seq := Nat.succ_pos n
      let st0 : InductionHeadCert.ParseState seq := InductionHeadCert.initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => InductionHeadCert.parseLine st t) st0
      let kind ←
        match st.kind with
        | some v => pure v
        | none => throw "missing kind entry"
      if kind != "onehot-approx" && kind != "induction-aligned" then
        throw s!"unexpected kind: {kind}"
      let period? := st.period
      if kind = "onehot-approx" && period?.isSome then
        throw "unexpected period entry for kind onehot-approx"
      if kind = "induction-aligned" && period?.isNone then
        throw "missing period entry for kind induction-aligned"
      let cert ← InductionHeadCert.finalizeStateCore hpos st
      let payload : InductionHeadCert.InductionHeadCertPayload seq :=
        { kind := kind, period? := period?, cert := cert }
      return ⟨seq, payload⟩

/-- Load an induction-head certificate from disk. -/
def loadInductionHeadCert (path : System.FilePath) :
    IO (Except String (Sigma InductionHeadCert.InductionHeadCertPayload)) := do
  let data ← IO.FS.readFile path
  return parseInductionHeadCert data

/-- Parse a token list payload. -/
def parseInductionHeadTokens (input : String) :
    Except String (Sigma fun seq => Fin seq → Nat) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let seq ← InductionHeadTokens.parseSeq tokens
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let st0 : InductionHeadTokens.ParseState seq := InductionHeadTokens.initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => InductionHeadTokens.parseLine st t) st0
      let tokensFun ← InductionHeadTokens.finalizeState st
      return ⟨seq, tokensFun⟩

/-- Load a token list from disk. -/
def loadInductionHeadTokens (path : System.FilePath) :
    IO (Except String (Sigma fun seq => Fin seq → Nat)) := do
  let data ← IO.FS.readFile path
  return parseInductionHeadTokens data

private def ratToString (x : Rat) : String :=
  toString x

private def tokensPeriodic {seq : Nat} (period : Nat) (tokens : Fin seq → Nat) : Bool :=
  (List.finRange seq).all (fun q =>
    if period ≤ q.val then
      decide (tokens q = tokens (Model.prevOfPeriod (seq := seq) period q))
    else
      true)

/-- Check an explicit induction-head certificate from disk. -/
def runInductionHeadCertCheck (certPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String)
    (tokensPath? : Option String)
    (minStripeMeanStr? : Option String) (minStripeTop1Str? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  let minStripeMean?E := parseRatOpt "min-stripe-mean" minStripeMeanStr?
  let minStripeTop1?E := parseRatOpt "min-stripe-top1" minStripeTop1Str?
  match minLogitDiff?E, minMargin?E, maxEps?E, minStripeMean?E, minStripeTop1?E with
  | Except.error msg, _, _, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps?,
      Except.ok minStripeMean?, Except.ok minStripeTop1? =>
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let parsed ← loadInductionHeadCert certPath
      match parsed with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, payload⟩ =>
          match seq with
          | 0 =>
              IO.eprintln "error: seq must be positive"
              return 2
          | Nat.succ n =>
              let seq := Nat.succ n
              let _ : NeZero seq := ⟨by simp⟩
              let cert := payload.cert
              let kind := payload.kind
              let period? := payload.period?
              if kind != "onehot-approx" && kind != "induction-aligned" then
                IO.eprintln s!"error: unexpected kind {kind}"
                return 2
              if kind = "onehot-approx" then
                if minStripeMeanStr?.isSome || minStripeTop1Str?.isSome then
                  IO.eprintln
                    "error: stripe/induction thresholds are not used for onehot-approx"
                  return 2
              if kind = "induction-aligned" then
                let period ←
                  match period? with
                  | some v => pure v
                  | none =>
                      IO.eprintln "error: missing period entry for induction-aligned"
                      return 2
                if period = 0 then
                  IO.eprintln "error: period must be positive for induction-aligned"
                  return 2
                if seq ≤ period then
                  IO.eprintln "error: period must be less than seq for induction-aligned"
                  return 2
                let expectedActive := Model.activeOfPeriod (seq := seq) period
                if !decide (cert.active = expectedActive) then
                  IO.eprintln "error: active set does not match induction-aligned period"
                  return 2
                let prevOk :=
                  (List.finRange seq).all (fun q =>
                    decide (cert.prev q = Model.prevOfPeriod (seq := seq) period q))
                if !prevOk then
                  IO.eprintln "error: prev map does not match induction-aligned period"
                  return 2
                if minLogitDiffStr?.isSome || minMarginStr?.isSome || maxEpsStr?.isSome then
                  IO.eprintln
                    "error: min-logit-diff/min-margin/max-eps are not used for induction-aligned"
                  return 2
                if minStripeTop1Str?.isSome then
                  IO.eprintln "error: stripe-top1 is not used for induction-aligned"
                  return 2
              let activeCount := cert.active.card
              let defaultMinActive := max 1 (seq / 8)
              let minActive := minActive?.getD defaultMinActive
              if activeCount < minActive then
                IO.eprintln
                  s!"error: active queries {activeCount} below minimum {minActive}"
                return 2
              if kind = "onehot-approx" then
                let ok := Circuit.checkInductionHeadCert cert
                if !ok then
                  IO.eprintln "error: induction-head certificate rejected"
                  return 2
                if cert.margin < minMargin then
                  IO.eprintln
                    s!"error: margin {ratToString cert.margin} \
                    below minimum {ratToString minMargin}"
                  return 2
                if maxEps < cert.eps then
                  IO.eprintln
                    s!"error: eps {ratToString cert.eps} \
                    above maximum {ratToString maxEps}"
                  return 2
              if let some tokensPath := tokensPath? then
                let tokensParsed ← loadInductionHeadTokens tokensPath
                match tokensParsed with
                | Except.error msg =>
                    IO.eprintln s!"error: {msg}"
                    return 2
                | Except.ok ⟨seqTokens, tokens⟩ =>
                    if hseq : seqTokens = seq then
                      let tokens' : Fin seq → Nat := by
                        simpa [hseq] using tokens
                      if kind = "induction-aligned" then
                        let period ←
                          match period? with
                          | some v => pure v
                          | none =>
                              IO.eprintln "error: missing period entry for induction-aligned"
                              return 2
                        if !tokensPeriodic (seq := seq) period tokens' then
                          IO.eprintln "error: tokens are not periodic for induction-aligned period"
                          return 2
                      else
                        let activeTokens := Model.activeOfTokens (seq := seq) tokens'
                        if !decide (cert.active ⊆ activeTokens) then
                          IO.eprintln "error: active set not contained in token repeats"
                          return 2
                        let prevTokens := Model.prevOfTokens (seq := seq) tokens'
                        let prevOk :=
                          (List.finRange seq).all (fun q =>
                            if decide (q ∈ cert.active) then
                              decide (prevTokens q = cert.prev q)
                            else
                              true)
                        if !prevOk then
                          IO.eprintln "error: prev map does not match tokens on active queries"
                          return 2
                    else
                      IO.eprintln
                        s!"error: tokens seq {seqTokens} does not match cert seq {seq}"
                      return 2
              if kind = "induction-aligned" then
                let period ←
                  match period? with
                  | some v => pure v
                  | none =>
                      IO.eprintln "error: missing period entry for induction-aligned"
                      return 2
                let stripeCert : Stripe.StripeCert seq :=
                  { period := period
                    stripeMeanLB := 0
                    stripeTop1LB := 0
                    weights := cert.weights }
                let stripeMean? := Stripe.stripeMean (seq := seq) stripeCert
                let stripeTop1? := Stripe.stripeTop1 (seq := seq) stripeCert
                match stripeMean?, stripeTop1? with
                | some mean, some top1 =>
                    let defaultMean : Rat := Rat.divInt 1 1000
                    let minMean := minStripeMean?.getD defaultMean
                    if mean < minMean then
                      IO.eprintln
                        s!"error: stripe-mean {ratToString mean} below minimum \
                        {ratToString minMean}"
                      return 2
                    IO.println
                      s!"ok: induction-aligned certificate checked \
                      (seq={seq}, active={activeCount}, \
                      stripeMean={ratToString mean})"
                    return 0
                | _, _ =>
                    IO.eprintln "error: empty active set for stripe stats"
                    return 2
              let effectiveMinLogitDiff :=
                match minLogitDiff?, cert.values.direction with
                | some v, _ => some v
                | none, some _ => some (0 : Rat)
                | none, none => none
              match effectiveMinLogitDiff with
              | none =>
                  let kindLabel := "onehot-approx (proxy)"
                  IO.println
                    s!"ok: {kindLabel} certificate checked \
                    (seq={seq}, active={activeCount}, \
                    margin={ratToString cert.margin}, eps={ratToString cert.eps})"
                  return 0
              | some minLogitDiff =>
                  let logitDiffLB? :=
                    Circuit.logitDiffLowerBoundAt cert.active cert.prev cert.epsAt
                      cert.values.lo cert.values.hi cert.values.vals
                  match logitDiffLB? with
                  | none =>
                      IO.eprintln "error: empty active set for logit-diff bound"
                      return 2
                  | some logitDiffLB =>
                      if logitDiffLB < minLogitDiff then
                        IO.eprintln
                          s!"error: logitDiffLB {ratToString logitDiffLB} \
                          below minimum {ratToString minLogitDiff}"
                        return 2
                      else
                        let kindLabel := "onehot-approx (proxy)"
                        IO.println
                          s!"ok: {kindLabel} certificate checked \
                          (seq={seq}, active={activeCount}, \
                          margin={ratToString cert.margin}, eps={ratToString cert.eps}, \
                          logitDiffLB={ratToString logitDiffLB})"
                        return 0

end IO

end Nfp
