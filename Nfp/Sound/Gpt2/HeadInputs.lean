-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Model.Gpt2
public import Nfp.Model.InductionHead
public import Nfp.Model.InductionPrompt

/-!
Sound builder for GPT-2 induction head inputs.

This converts exact GPT-2 head slices into `InductionHeadInputs` using a
periodic prompt description. The construction is purely definitional and is
captured by an explicit theorem, so the trusted core does not hide any logic.
-/

public section

namespace Nfp

namespace Sound

namespace Gpt2

open Nfp.Model

/--
Build induction-head inputs from a GPT-2 head slice and prompt period.

This uses the unshifted periodic prompt (`prev = q - period`), i.e. it matches
the current token rather than the canonical induction copy target.
-/
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
    maskValue := (-10000 : Rat)
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
        maskValue := (-10000 : Rat)
        directionSpec := slice.direction.spec
        direction := slice.directionVec } := by
  simp [buildInductionHeadInputs]

/--
Build induction-head inputs using the canonical shifted periodic prompt
(`prev = q - period + 1`, with `0 < period`). When `1 < period`, every active
query has `prev q < q`.
-/
def buildInductionHeadInputsShift {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) (period : Nat) :
    Model.InductionHeadInputs seq dModel dHead :=
  { scale := slice.scale
    active := activeOfPeriodShift (seq := seq) period
    prev := prevOfPeriodShift (seq := seq) period
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
    maskValue := (-10000 : Rat)
    directionSpec := slice.direction.spec
    direction := slice.directionVec }

/-- Definitional characterization of `buildInductionHeadInputsShift`. -/
theorem buildInductionHeadInputsShift_def {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) (period : Nat) :
    buildInductionHeadInputsShift slice period =
      { scale := slice.scale
        active := activeOfPeriodShift (seq := seq) period
        prev := prevOfPeriodShift (seq := seq) period
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
        maskValue := (-10000 : Rat)
        directionSpec := slice.direction.spec
        direction := slice.directionVec } := by
  simp [buildInductionHeadInputsShift]

/-- `buildInductionHeadInputsShift` satisfies the shifted-period prev/active spec. -/
theorem buildInductionHeadInputsShift_prev_spec {seq dModel dHead vocab : Nat}
    (slice : Gpt2HeadSlice seq dModel dHead vocab) (period : Nat) :
    InductionPrevSpecPeriodShift (seq := seq) period
      (buildInductionHeadInputsShift slice period) := by
  constructor <;> simp [buildInductionHeadInputsShift]

end Gpt2

end Sound

end Nfp
