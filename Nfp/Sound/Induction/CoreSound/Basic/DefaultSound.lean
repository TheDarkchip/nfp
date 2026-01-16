-- SPDX-License-Identifier: AGPL-3.0-or-later
module

import all Nfp.Sound.Induction.Core.Basic
public import Nfp.Sound.Induction.Core
public import Nfp.Sound.Induction.CoreSound.Basic.CertSound

public section

namespace Nfp
namespace Sound
open scoped BigOperators
open Nfp.Circuit
open Nfp.Sound.Bounds
variable {seq : Nat}
/-- Soundness for `buildInductionCertFromHeadCore?`. -/
theorem buildInductionCertFromHeadCore?_sound
      [NeZero seq] {dModel dHead : Nat}
      (inputs : Model.InductionHeadInputs seq dModel dHead) (c : InductionHeadCert seq)
      (hcore : buildInductionCertFromHeadCore? inputs = some c) :
      InductionHeadCertSound inputs c := by
  have hcore' :
      buildInductionCertFromHeadCoreWith? defaultInductionHeadSplitConfig inputs = some c := by
    simpa [buildInductionCertFromHeadCore?_def] using hcore
  exact
    buildInductionCertFromHeadCoreWith?_sound
      (cfg := defaultInductionHeadSplitConfig) inputs c hcore'
end Sound
end Nfp
