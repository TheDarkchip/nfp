-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.IO.Pure

/-!
IO wrappers for loading and checking induction certificates.
-/

namespace Nfp

namespace IO

open Nfp.Circuit

/-- Load a softmax-margin certificate from disk. -/
def loadSoftmaxMarginCert (path : System.FilePath) :
    IO (Except String (Sigma SoftmaxMarginCert)) := do
  let data ← IO.FS.readFile path
  return Pure.parseSoftmaxMarginCert data

/-- Load a value-range certificate from disk. -/
def loadValueRangeCert (path : System.FilePath) :
    IO (Except String (Sigma ValueRangeCert)) := do
  let data ← IO.FS.readFile path
  return Pure.parseValueRangeCert data

private def checkSoftmaxMargin (seq : Nat) (cert : SoftmaxMarginCert seq) :
    IO (Except String Unit) :=
  match seq with
  | 0 => return Except.error "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      let ok := Circuit.checkSoftmaxMarginCert cert
      if ok then
        return Except.ok ()
      else
        return Except.error "softmax-margin certificate rejected"

private def checkValueRange (seq : Nat) (cert : ValueRangeCert seq) :
    IO (Except String Unit) :=
  match seq with
  | 0 => return Except.error "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      let ok := Circuit.checkValueRangeCert cert
      if ok then
        return Except.ok ()
      else
        return Except.error "value-range certificate rejected"

/-- Check induction certificates and print a short status line. -/
def runInductionCertify (scoresPath : System.FilePath)
    (valuesPath? : Option System.FilePath) : IO UInt32 := do
  let parsedScores ← loadSoftmaxMarginCert scoresPath
  match parsedScores with
  | Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 1
  | Except.ok ⟨seq, cert⟩ =>
      let scoresOk ← checkSoftmaxMargin seq cert
      match scoresOk with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 2
      | Except.ok () =>
          match valuesPath? with
          | none =>
              IO.println s!"ok: softmax-margin certificate accepted (seq={seq})"
              return 0
          | some valuesPath =>
              let parsedValues ← loadValueRangeCert valuesPath
              match parsedValues with
              | Except.error msg =>
                  IO.eprintln s!"error: {msg}"
                  return 1
              | Except.ok ⟨seqVals, certVals⟩ =>
                  if seqVals ≠ seq then
                    IO.eprintln s!"error: seq mismatch (scores={seq}, values={seqVals})"
                    return 2
                  let valuesOk ← checkValueRange seqVals certVals
                  match valuesOk with
                  | Except.error msg =>
                      IO.eprintln s!"error: {msg}"
                      return 2
                  | Except.ok () =>
                      let tol := cert.eps * (certVals.hi - certVals.lo)
                      IO.println s!"ok: induction bound certified (seq={seq}, tol={tol})"
                      return 0

end IO

end Nfp
