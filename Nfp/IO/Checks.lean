-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange

/-!
IO checks for certificates.
-/

namespace Nfp

namespace IO

open Nfp.Circuit

/-- Check a softmax-margin certificate for a positive sequence length. -/
def checkSoftmaxMargin (seq : Nat) (cert : SoftmaxMarginCert seq) :
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

/-- Check a value-range certificate for a positive sequence length. -/
def checkValueRange (seq : Nat) (cert : ValueRangeCert seq) :
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

end IO

end Nfp
