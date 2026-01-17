-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.InductionHead.Basic

/-!
IO helpers for induction-circuit checks (previous-token head + induction head).
-/

public section

namespace Nfp

namespace IO

/-- Check a two-head induction circuit directly from a model binary.

The induction head is certified with shifted `prev` (canonical circuit), while
the previous-token head uses the unshifted period-1 map.
-/
def runInductionCertifyCircuitModel (modelPath : System.FilePath)
    (prevLayer prevHead indLayer indHead dirTarget dirNegative : Nat) (period : Nat)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String)
    (timing? : Option Nat) (heartbeatMs? : Option Nat)
    (splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined? : Option Nat)
    (skipLogitDiff : Bool) :
    IO UInt32 := do
  let prevCode ←
    runInductionCertifyHeadModel
      modelPath
      prevLayer
      prevHead
      dirTarget
      dirNegative
      (some 1)
      false
      minActive?
      minLogitDiffStr?
      minMarginStr?
      maxEpsStr?
      timing?
      heartbeatMs?
      splitBudgetQ?
      splitBudgetK?
      splitBudgetDiffBase?
      splitBudgetDiffRefined?
      skipLogitDiff
  if prevCode ≠ 0 then
    return prevCode
  let indCode ←
    runInductionCertifyHeadModel
      modelPath
      indLayer
      indHead
      dirTarget
      dirNegative
      (some period)
      true
      minActive?
      minLogitDiffStr?
      minMarginStr?
      maxEpsStr?
      timing?
      heartbeatMs?
      splitBudgetQ?
      splitBudgetK?
      splitBudgetDiffBase?
      splitBudgetDiffRefined?
      skipLogitDiff
  if indCode ≠ 0 then
    return indCode
  IO.println
    "ok: circuit head certificates built (prev-token head + shifted-prev induction head)"
  return 0

end IO

end Nfp
