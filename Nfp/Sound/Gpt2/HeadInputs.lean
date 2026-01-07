-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Model.Gpt2
import Nfp.Model.InductionHead
import Nfp.Model.InductionPrompt

/-!
Sound builder for GPT-2 induction head inputs.

This converts exact GPT-2 head slices into `InductionHeadInputs` using a
periodic prompt description. The construction is purely definitional and is
captured by an explicit theorem, so the trusted core does not hide any logic.
-/

namespace Nfp

namespace Sound

namespace Gpt2

open Nfp.Model

/-- Build induction-head inputs from a GPT-2 head slice and prompt period. -/
def buildInductionHeadInputs {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) (period : Nat) :
    Model.InductionHeadInputs seq dModel dHead :=
  { scale := slice.scale
    active := activeOfPeriod (seq := seq) period
    prev := prevOfPeriod (seq := seq) period
    embed := slice.embed
    lnEps := slice.lnEps
    ln1Gamma := slice.ln1Gamma
    ln1Beta := slice.ln1Beta
    wq := slice.wq
    bq := slice.bq
    wk := slice.wk
    bk := slice.bk
    wv := slice.wv
    bv := slice.bv
    wo := slice.wo
    attnBias := slice.attnBias
    maskCausal := true
    maskValue := (-10000 : Dyadic)
    directionSpec := slice.direction.spec
    direction := slice.directionVec }

/-- Definitional characterization of `buildInductionHeadInputs`. -/
theorem buildInductionHeadInputs_def {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) (period : Nat) :
    buildInductionHeadInputs slice period =
      { scale := slice.scale
        active := activeOfPeriod (seq := seq) period
        prev := prevOfPeriod (seq := seq) period
        embed := slice.embed
        lnEps := slice.lnEps
        ln1Gamma := slice.ln1Gamma
        ln1Beta := slice.ln1Beta
        wq := slice.wq
        bq := slice.bq
        wk := slice.wk
        bk := slice.bk
        wv := slice.wv
        bv := slice.bv
        wo := slice.wo
        attnBias := slice.attnBias
        maskCausal := true
        maskValue := (-10000 : Dyadic)
        directionSpec := slice.direction.spec
        direction := slice.directionVec } := rfl

end Gpt2

end Sound

end Nfp
