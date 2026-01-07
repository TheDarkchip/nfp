-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std.Data.HashMap
import Nfp.Model.InductionHead

/-!
Call-count instrumentation for induction-head computations.

This is a placeholder-only benchmark: it records how often key functions would be
called in a score-bound pass without performing heavy arithmetic.
-/

namespace Nfp

namespace IO

open scoped BigOperators

private def bumpCount (ref : IO.Ref (Std.HashMap String Nat)) (key : String) (n : Nat) :
    IO Unit := do
  ref.modify (fun m =>
    let cur := (m.get? key).getD 0
    m.insert key (cur + n))

private def printCounts (ref : IO.Ref (Std.HashMap String Nat)) : IO Unit := do
  let m ← ref.get
  let entries := m.toList
  IO.println "counts:"
  for (k, v) in entries do
    IO.println s!"  {k}: {v}"

private def countScoreCalls {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (ref : IO.Ref (Std.HashMap String Nat)) : IO Unit := do
  let activeCount := inputs.active.card
  let otherCount := seq - 1
  let rowCount := activeCount * otherCount
  let elemCount := rowCount * dHead
  bumpCount ref "scoreBounds:scoreLo" rowCount
  bumpCount ref "scoreBounds:scoreHi" rowCount
  bumpCount ref "scoreBounds:qAbs" elemCount
  bumpCount ref "scoreBounds:qLo" elemCount
  bumpCount ref "scoreBounds:qHi" elemCount
  bumpCount ref "scoreBounds:kAbs" elemCount
  bumpCount ref "scoreBounds:kLo" elemCount
  bumpCount ref "scoreBounds:kHi" elemCount

private def countQKVCalls {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (ref : IO.Ref (Std.HashMap String Nat)) : IO Unit := do
  let activeCount := inputs.active.card
  let elemCount := activeCount * dHead
  bumpCount ref "qkvBounds:qLo" elemCount
  bumpCount ref "qkvBounds:qHi" elemCount
  bumpCount ref "qkvBounds:kLo" elemCount
  bumpCount ref "qkvBounds:kHi" elemCount
  bumpCount ref "qkvBounds:vLo" elemCount
  bumpCount ref "qkvBounds:vHi" elemCount
  bumpCount ref "qkvBounds:qAbs" elemCount
  bumpCount ref "qkvBounds:kAbs" elemCount

/-- Count calls used by score/QKV bounds on the active set. -/
def countInductionCalls {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : IO Unit := do
  let ref ← IO.mkRef (∅ : Std.HashMap String Nat)
  countQKVCalls inputs ref
  countScoreCalls inputs ref
  printCounts ref

end IO

end Nfp
