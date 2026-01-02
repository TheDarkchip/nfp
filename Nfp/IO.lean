-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.IO.Pure

/-!
IO wrappers for loading and checking softmax-margin certificates.
-/

namespace Nfp

namespace IO

open Nfp.Circuit

/-- Load a softmax-margin certificate from disk. -/
def loadSoftmaxMarginCert (path : System.FilePath) :
    IO (Except String (Sigma SoftmaxMarginCert)) := do
  let data ← IO.FS.readFile path
  return Pure.parseSoftmaxMarginCert data

/-- Check a softmax-margin certificate file and print a short status line. -/
def runInductionCertify (path : System.FilePath) : IO UInt32 := do
  let parsed ← loadSoftmaxMarginCert path
  match parsed with
  | Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 1
  | Except.ok ⟨seq, cert⟩ =>
      match seq with
      | 0 =>
          IO.eprintln "error: seq must be positive"
          return 1
      | Nat.succ n =>
          let seq := Nat.succ n
          let _ : NeZero seq := ⟨by simpa using Nat.succ_ne_zero n⟩
          let ok := Circuit.checkSoftmaxMarginCert cert
          if ok then
            IO.println s!"ok: certificate accepted (seq={seq})"
            return 0
          else
            IO.eprintln s!"error: certificate rejected (seq={seq})"
            return 2

end IO

end Nfp
