-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.InductionHead
public import Nfp.IO.InductionHead.ModelDirectionSlice
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.ModelValueSlice
public import Nfp.IO.Pure.InductionHead.ValueCheck

/-!
Value-path checks for anchoring induction-head certificate values.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/--
Check that per-key values lie within bounds derived from model slices.

This is a thin wrapper around the trusted IO.Pure implementation.
-/
def valuesWithinModelBounds {seq : Nat}
    (lnSlice : ModelLnSlice seq) (valueSlice : ModelValueSlice)
    (dirSlice : ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (values : Circuit.ValueIntervalCert seq) : Bool :=
  Pure.InductionHeadCert.valuesWithinModelBounds
    lnSlice valueSlice dirSlice hLn hDir values

end InductionHeadCert

end IO

end Nfp
