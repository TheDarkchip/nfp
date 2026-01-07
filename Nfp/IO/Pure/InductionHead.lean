-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.IO.Pure.InductionHead.Bytes

/-!
Parsing helpers for induction-head input payloads.
-/

namespace Nfp

namespace IO

namespace Pure

/-- Parse a raw induction head input payload from text. -/
def parseInductionHeadInputs (input : String) :
    Except String (Sigma (fun seq =>
      Sigma (fun dModel => Sigma (fun dHead => Model.InductionHeadInputs seq dModel dHead)))) := do
  parseInductionHeadInputsBytes input.toUTF8

end Pure

end IO

end Nfp
