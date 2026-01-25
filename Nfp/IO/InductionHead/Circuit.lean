-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.InductionHead.Cert
public import Nfp.IO.InductionHead.Tokens
public import Nfp.Model.InductionPrompt

/-!
Circuit-level induction checks that compose a prev-token head with an induction head.

This is a lightweight wrapper over two explicit induction-head certificates:
the prev-token head is expected to use `prevOfPeriod` with period 1, while the
induction head uses the shifted-period map `prevOfPeriodShift` for the supplied
period. The checker ensures both head certificates pass and that the structural
prev/active requirements match the canonical circuit definition.
-/

public section

namespace Nfp

namespace IO

open Nfp.Circuit
open Nfp.Model

namespace InductionHeadCircuit

private def tokensPeriodic {seq : Nat} (period : Nat) (tokens : Fin seq → Nat) : Bool :=
  (List.finRange seq).all (fun q =>
    if period ≤ q.val then
      decide (tokens q = tokens (prevOfPeriod (seq := seq) period q))
    else
      true)

/--
Check a composed prev-token + induction-head circuit from two cert files.

Alignment notes:
- prev-token head: `activeOfPeriod 1` + `prevOfPeriod 1` (canonical previous-token map).
- induction head: `activeOfPeriodShift period` + `prevOfPeriodShift period`
  (the shifted-prev map used by `InductionCircuitSpecPeriodShift`).
- if tokens are provided, the additional periodicity check enforces the
  `InductionDiagnosticTokens` setting used to relate the period and token specs.
-/
def runInductionCircuitCertCheck
    (prevCertPath indCertPath : System.FilePath)
    (period : Nat)
    (tokensPath? : Option String) : IO UInt32 := do
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  if period = 0 then
    return (← fail "period must be positive")
  let prevParsed ← loadInductionHeadCert prevCertPath
  let indParsed ← loadInductionHeadCert indCertPath
  match prevParsed, indParsed with
  | Except.error msg, _ => fail msg
  | _, Except.error msg => fail msg
  | Except.ok ⟨0, _⟩, _ => fail "seq must be positive"
  | _, Except.ok ⟨0, _⟩ => fail "seq must be positive"
  | Except.ok ⟨Nat.succ n, prevPayload⟩, Except.ok ⟨Nat.succ m, indPayload⟩ =>
      if hnm : n = m then
        match hnm with
        | rfl =>
            let seq := Nat.succ n
            let _ : NeZero seq := ⟨by simp⟩
            if seq ≠ 2 * period then
              return (← fail "seq must equal 2 * period for circuit verification")
            let prevCert := prevPayload.cert
            let indCert := indPayload.cert
            if !(Circuit.checkInductionHeadCert prevCert) then
              return (← fail "prev-token head certificate rejected")
            if !(Circuit.checkInductionHeadCert indCert) then
              return (← fail "induction head certificate rejected")
            let expectedPrevActive := activeOfPeriod (seq := seq) 1
            if !decide (prevCert.active = expectedPrevActive) then
              return (← fail "prev-token active set does not match period=1")
            let prevMapOk := (List.finRange seq).all (fun q =>
              decide (prevCert.prev q = prevOfPeriod (seq := seq) 1 q))
            if !prevMapOk then
              return (← fail "prev-token prev map does not match period=1")
            let expectedIndActive := activeOfPeriodShift (seq := seq) period
            if !decide (indCert.active = expectedIndActive) then
              return (← fail "induction active set does not match shifted period")
            let indPrevOk := (List.finRange seq).all (fun q =>
              decide (indCert.prev q = prevOfPeriodShift (seq := seq) period q))
            if !indPrevOk then
              return (← fail "induction prev map does not match shifted period")
            let indPrevActive := indCert.active.image indCert.prev
            if !decide (indPrevActive ⊆ prevCert.active) then
              return (← fail "prev-token active set does not cover shifted prev indices")
            match tokensPath? with
            | none => pure ()
            | some tokensPath =>
                let tokensParsed ← loadInductionHeadTokens tokensPath
                match tokensParsed with
                | Except.error msg => return (← fail msg)
                | Except.ok ⟨seqTokens, tokens⟩ =>
                    if hseqTok : seqTokens = seq then
                      let tokens' : Fin seq → Nat := by
                        simpa [hseqTok] using tokens
                      if !tokensPeriodic (seq := seq) period tokens' then
                        return (← fail "tokens are not periodic for circuit period")
                    else
                      return (← fail s!"tokens seq {seqTokens} does not match cert seq {seq}")
            IO.println
              s!"ok: induction circuit checked (seq={seq}, period={period}, \
              prevCert={prevCertPath}, indCert={indCertPath})"
            return 0
      else
        let seqPrev := Nat.succ n
        let seqInd := Nat.succ m
        fail s!"seq mismatch: prev {seqPrev} vs ind {seqInd}"

end InductionHeadCircuit

end IO

end Nfp
