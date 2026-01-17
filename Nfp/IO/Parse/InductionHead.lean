-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Parse.InductionHead.Bytes

/-!
Parsing helpers for induction-head input payloads.
-/

public section

namespace Nfp

namespace IO

namespace Parse

/-- Parse a raw induction head input payload from text. -/
def parseInductionHeadInputs (input : String) :
    Except String (Sigma (fun seq =>
      Sigma (fun dModel => Sigma (fun dHead => Model.InductionHeadInputs seq dModel dHead)))) := do
  parseInductionHeadInputsBytes input.toUTF8

end Parse

end IO

end Nfp
