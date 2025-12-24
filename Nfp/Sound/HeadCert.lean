-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Induction
import Nfp.Sound.Bounds

namespace Nfp.Sound

/-!
# Sound per-head contribution certificates

This module defines a minimal, checkable certificate for per-head weight-only
contribution bounds. These are intended as a lightweight starting point for
sound circuit certification.
-/

/-- Weight-only per-head operator-norm bounds and derived factors. -/
structure HeadContributionCert where
  layerIdx : Nat
  headIdx : Nat
  wqOpBound : Rat
  wkOpBound : Rat
  wvOpBound : Rat
  woOpBound : Rat
  qkFactorBound : Rat
  voFactorBound : Rat
  deriving Repr

namespace HeadContributionCert

/-- Internal consistency checks for derived factor bounds. -/
def Valid (c : HeadContributionCert) : Prop :=
  c.qkFactorBound = c.wqOpBound * c.wkOpBound ∧
    c.voFactorBound = c.wvOpBound * c.woOpBound

instance (c : HeadContributionCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadContributionCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadContributionCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end HeadContributionCert

/-- Local (input-dependent) per-head attention contribution bounds. -/
structure HeadLocalContributionCert where
  layerIdx : Nat
  headIdx : Nat
  /-- Precision in dyadic bits for local LayerNorm bounds. -/
  soundnessBits : Nat
  ln1MaxAbsGamma : Rat
  ln1VarianceLowerBound : Rat
  ln1Bound : Rat
  wqOpBound : Rat
  wkOpBound : Rat
  wvOpBound : Rat
  woOpBound : Rat
  qkFactorBound : Rat
  attnWeightContribution : Rat
  deriving Repr

namespace HeadLocalContributionCert

/-- Internal consistency checks for local per-head bounds. -/
def Valid (eps : Rat) (c : HeadLocalContributionCert) : Prop :=
  0 < eps ∧
    c.ln1Bound =
      (if c.ln1VarianceLowerBound > 0 then
        layerNormOpBoundLocal c.ln1MaxAbsGamma c.ln1VarianceLowerBound eps c.soundnessBits
      else
        layerNormOpBoundConservative c.ln1MaxAbsGamma eps c.soundnessBits) ∧
    c.qkFactorBound = c.wqOpBound * c.wkOpBound ∧
    c.attnWeightContribution =
      c.ln1Bound * softmaxJacobianNormInfWorst * c.wvOpBound * c.woOpBound

instance (eps : Rat) (c : HeadLocalContributionCert) : Decidable (Valid eps c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (eps : Rat) (c : HeadLocalContributionCert) : Bool :=
  decide (Valid eps c)

theorem check_iff (eps : Rat) (c : HeadLocalContributionCert) :
    c.check eps = true ↔ c.Valid eps := by
  simp [check]

end HeadLocalContributionCert

/-- Local per-head attention pattern certificate (target logit dominance). -/
structure HeadPatternCert where
  layerIdx : Nat
  headIdx : Nat
  seqLen : Nat
  targetOffset : Int
  targetCountLowerBound : Nat
  targetLogitLowerBound : Rat
  otherLogitUpperBound : Rat
  marginLowerBound : Rat
  /-- Effort level for the `expLB` portfolio used in margin-to-weight bounds. -/
  softmaxExpEffort : Nat
  targetWeightLowerBound : Rat
  deriving Repr

namespace HeadPatternCert

/-- Internal consistency checks for pattern bounds. -/
def Valid (c : HeadPatternCert) : Prop :=
  c.seqLen > 0 ∧
    c.targetCountLowerBound ≤ c.seqLen ∧
    c.marginLowerBound = c.targetLogitLowerBound - c.otherLogitUpperBound ∧
    c.targetWeightLowerBound =
      softmaxTargetWeightLowerBound c.seqLen c.targetCountLowerBound
        c.marginLowerBound c.softmaxExpEffort

instance (c : HeadPatternCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadPatternCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadPatternCert) : c.check = true ↔ c.Valid := by
  simp [check]

end HeadPatternCert

/-! ## Local value-direction bounds -/

/-- Safe lower bound for a convex mixture when only a lower bound on the match weight is known. -/
def mixLowerBound (w m n : Rat) : Rat :=
  min m (w * m + (1 - w) * n)

/-- Local per-head output lower bound for a single coordinate. -/
structure HeadValueLowerBoundCert where
  layerIdx : Nat
  headIdx : Nat
  coord : Nat
  matchWeightLowerBound : Rat
  matchCoordLowerBound : Rat
  nonmatchCoordLowerBound : Rat
  outputCoordLowerBound : Rat
  deriving Repr

namespace HeadValueLowerBoundCert

/-- Internal consistency checks for the coordinate lower bound. -/
def Valid (c : HeadValueLowerBoundCert) : Prop :=
  0 ≤ c.matchWeightLowerBound ∧
    c.matchWeightLowerBound ≤ 1 ∧
    c.outputCoordLowerBound =
      mixLowerBound c.matchWeightLowerBound c.matchCoordLowerBound c.nonmatchCoordLowerBound

instance (c : HeadValueLowerBoundCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadValueLowerBoundCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadValueLowerBoundCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end HeadValueLowerBoundCert

/-! ## Logit-direction bounds -/

/-- Local per-head logit-difference lower bound for a target direction. -/
structure HeadLogitDiffLowerBoundCert where
  layerIdx : Nat
  headIdx : Nat
  targetToken : Nat
  negativeToken : Nat
  matchWeightLowerBound : Rat
  matchLogitLowerBound : Rat
  nonmatchLogitLowerBound : Rat
  logitDiffLowerBound : Rat
  deriving Repr

namespace HeadLogitDiffLowerBoundCert

/-- Internal consistency checks for the logit-difference lower bound. -/
def Valid (c : HeadLogitDiffLowerBoundCert) : Prop :=
  0 ≤ c.matchWeightLowerBound ∧
    c.matchWeightLowerBound ≤ 1 ∧
    c.logitDiffLowerBound =
      mixLowerBound c.matchWeightLowerBound c.matchLogitLowerBound c.nonmatchLogitLowerBound

instance (c : HeadLogitDiffLowerBoundCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadLogitDiffLowerBoundCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadLogitDiffLowerBoundCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end HeadLogitDiffLowerBoundCert

/-! ## Best-match pattern bounds -/

/-- Local per-head attention pattern certificate (best-match, single query position). -/
structure HeadBestMatchPatternCert where
  layerIdx : Nat
  headIdx : Nat
  seqLen : Nat
  queryPos : Nat
  targetOffset : Int
  targetToken : Int
  bestMatchLogitLowerBound : Rat
  bestNonmatchLogitUpperBound : Rat
  marginLowerBound : Rat
  /-- Effort level for the `expLB` portfolio used in margin-to-probability bounds. -/
  softmaxExpEffort : Nat
  bestMatchWeightLowerBound : Rat
  /-- Softmax Jacobian row-sum bound derived from the max-probability lower bound. -/
  softmaxJacobianNormInfUpperBound : Rat
  deriving Repr

namespace HeadBestMatchPatternCert

/-- Internal consistency checks for best-match pattern bounds. -/
def Valid (c : HeadBestMatchPatternCert) : Prop :=
  c.seqLen > 0 ∧
    c.queryPos < c.seqLen ∧
    c.marginLowerBound = c.bestMatchLogitLowerBound - c.bestNonmatchLogitUpperBound ∧
    c.bestMatchWeightLowerBound =
      softmaxMaxProbLowerBound c.seqLen c.marginLowerBound c.softmaxExpEffort ∧
    c.softmaxJacobianNormInfUpperBound =
      softmaxJacobianNormInfBoundFromMargin c.seqLen c.marginLowerBound c.softmaxExpEffort

instance (c : HeadBestMatchPatternCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadBestMatchPatternCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadBestMatchPatternCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end HeadBestMatchPatternCert

/-! ## Best-match value/logit bounds -/

/-- Local per-head output lower bound for a single coordinate (single query position). -/
structure HeadValueLowerBoundPosCert where
  layerIdx : Nat
  headIdx : Nat
  queryPos : Nat
  coord : Nat
  matchWeightLowerBound : Rat
  matchCoordLowerBound : Rat
  nonmatchCoordLowerBound : Rat
  outputCoordLowerBound : Rat
  deriving Repr

namespace HeadValueLowerBoundPosCert

/-- Internal consistency checks for the coordinate lower bound. -/
def Valid (c : HeadValueLowerBoundPosCert) : Prop :=
  0 ≤ c.matchWeightLowerBound ∧
    c.matchWeightLowerBound ≤ 1 ∧
    c.outputCoordLowerBound =
      mixLowerBound c.matchWeightLowerBound c.matchCoordLowerBound c.nonmatchCoordLowerBound

instance (c : HeadValueLowerBoundPosCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadValueLowerBoundPosCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadValueLowerBoundPosCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end HeadValueLowerBoundPosCert

/-- Local per-head logit-difference lower bound (single query position). -/
structure HeadLogitDiffLowerBoundPosCert where
  layerIdx : Nat
  headIdx : Nat
  queryPos : Nat
  targetToken : Nat
  negativeToken : Nat
  matchWeightLowerBound : Rat
  matchLogitLowerBound : Rat
  nonmatchLogitLowerBound : Rat
  logitDiffLowerBound : Rat
  deriving Repr

namespace HeadLogitDiffLowerBoundPosCert

/-- Internal consistency checks for the logit-difference lower bound. -/
def Valid (c : HeadLogitDiffLowerBoundPosCert) : Prop :=
  0 ≤ c.matchWeightLowerBound ∧
    c.matchWeightLowerBound ≤ 1 ∧
    c.logitDiffLowerBound =
      mixLowerBound c.matchWeightLowerBound c.matchLogitLowerBound c.nonmatchLogitLowerBound

instance (c : HeadLogitDiffLowerBoundPosCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadLogitDiffLowerBoundPosCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadLogitDiffLowerBoundPosCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end HeadLogitDiffLowerBoundPosCert

namespace HeadPatternCert

/-- Convert a sound head pattern certificate into a token-match witness. -/
def toTokenMatchPattern (c : HeadPatternCert) : Nfp.TokenMatchPattern := {
  seqLen := c.seqLen
  targetOffset := c.targetOffset
  targetCountLowerBound := c.targetCountLowerBound
  softmaxExpEffort := c.softmaxExpEffort
  targetWeightLowerBound := c.targetWeightLowerBound
  marginLowerBound := c.marginLowerBound
}

theorem toTokenMatchPattern_valid (c : HeadPatternCert) (h : c.Valid) :
    (toTokenMatchPattern c).Valid := by
  rcases h with ⟨hseq, hcount, _hmargin, hweight⟩
  exact ⟨hseq, hcount, by simpa [toTokenMatchPattern] using hweight⟩

def toInductionPatternWitness
    (c : HeadPatternCert) (h : c.Valid) (hm : c.marginLowerBound > 0)
    (hcount : 0 < c.targetCountLowerBound) (hoff : c.targetOffset = -1) :
    Nfp.InductionPatternWitness :=
  Nfp.TokenMatchPattern.toInductionPatternWitness
    (toTokenMatchPattern c) (toTokenMatchPattern_valid c h) hm hcount hoff

end HeadPatternCert

/-! ## Induction head sound certificates -/

/-- Combined sound certificate for an induction-style head pair. -/
structure InductionHeadSoundCert where
  layer1Pattern : HeadPatternCert
  layer2Pattern : HeadPatternCert
  layer2Value : HeadValueLowerBoundCert
  layer2Logit? : Option HeadLogitDiffLowerBoundCert
  deltaLowerBound : Rat
  deriving Repr

namespace InductionHeadSoundCert

/-- Internal consistency checks for the combined certificate. -/
def Valid (c : InductionHeadSoundCert) : Prop :=
  HeadPatternCert.Valid c.layer1Pattern ∧
    HeadPatternCert.Valid c.layer2Pattern ∧
    HeadValueLowerBoundCert.Valid c.layer2Value ∧
    c.layer2Value.layerIdx = c.layer2Pattern.layerIdx ∧
    c.layer2Value.headIdx = c.layer2Pattern.headIdx ∧
    c.layer2Value.matchWeightLowerBound = c.layer2Pattern.targetWeightLowerBound ∧
    c.deltaLowerBound = c.layer2Value.outputCoordLowerBound ∧
    (match c.layer2Logit? with
      | none => True
      | some logit =>
          HeadLogitDiffLowerBoundCert.Valid logit ∧
            logit.layerIdx = c.layer2Pattern.layerIdx ∧
            logit.headIdx = c.layer2Pattern.headIdx ∧
            logit.matchWeightLowerBound = c.layer2Pattern.targetWeightLowerBound)

instance (c : InductionHeadSoundCert) : Decidable (Valid c) := by
  classical
  unfold Valid
  cases c.layer2Logit? <;> infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : InductionHeadSoundCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : InductionHeadSoundCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end InductionHeadSoundCert

/-! ## Best-match induction head certificates -/

/-- Combined sound certificate for an induction-style head pair (best-match pattern). -/
structure InductionHeadBestMatchSoundCert where
  layer1Pattern : HeadBestMatchPatternCert
  layer2Pattern : HeadBestMatchPatternCert
  layer2Value : HeadValueLowerBoundPosCert
  layer2Logit? : Option HeadLogitDiffLowerBoundPosCert
  deltaLowerBound : Rat
  deriving Repr

namespace InductionHeadBestMatchSoundCert

/-- Internal consistency checks for the combined certificate. -/
def Valid (c : InductionHeadBestMatchSoundCert) : Prop :=
  HeadBestMatchPatternCert.Valid c.layer1Pattern ∧
    HeadBestMatchPatternCert.Valid c.layer2Pattern ∧
    HeadValueLowerBoundPosCert.Valid c.layer2Value ∧
    c.layer2Value.layerIdx = c.layer2Pattern.layerIdx ∧
    c.layer2Value.headIdx = c.layer2Pattern.headIdx ∧
    c.layer2Value.queryPos = c.layer2Pattern.queryPos ∧
    c.layer2Value.matchWeightLowerBound = c.layer2Pattern.bestMatchWeightLowerBound ∧
    c.deltaLowerBound = c.layer2Value.outputCoordLowerBound ∧
    (match c.layer2Logit? with
      | none => True
      | some logit =>
          HeadLogitDiffLowerBoundPosCert.Valid logit ∧
            logit.layerIdx = c.layer2Pattern.layerIdx ∧
            logit.headIdx = c.layer2Pattern.headIdx ∧
            logit.queryPos = c.layer2Pattern.queryPos ∧
            logit.matchWeightLowerBound = c.layer2Pattern.bestMatchWeightLowerBound)

instance (c : InductionHeadBestMatchSoundCert) : Decidable (Valid c) := by
  classical
  unfold Valid
  cases c.layer2Logit? <;> infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : InductionHeadBestMatchSoundCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : InductionHeadBestMatchSoundCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end InductionHeadBestMatchSoundCert

/-! ### Specs -/

theorem HeadContributionCert.Valid_spec :
    HeadContributionCert.Valid = HeadContributionCert.Valid := rfl
theorem HeadContributionCert.check_spec :
    HeadContributionCert.check = HeadContributionCert.check := rfl
theorem HeadLocalContributionCert.Valid_spec :
    HeadLocalContributionCert.Valid = HeadLocalContributionCert.Valid := rfl
theorem HeadLocalContributionCert.check_spec :
    HeadLocalContributionCert.check = HeadLocalContributionCert.check := rfl
theorem mixLowerBound_spec :
    mixLowerBound = mixLowerBound := rfl
theorem HeadPatternCert.Valid_spec :
    HeadPatternCert.Valid = HeadPatternCert.Valid := rfl
theorem HeadPatternCert.check_spec :
    HeadPatternCert.check = HeadPatternCert.check := rfl
theorem HeadPatternCert.toTokenMatchPattern_spec :
    HeadPatternCert.toTokenMatchPattern = HeadPatternCert.toTokenMatchPattern := rfl
theorem HeadPatternCert.toInductionPatternWitness_spec :
    HeadPatternCert.toInductionPatternWitness = HeadPatternCert.toInductionPatternWitness := rfl
theorem HeadValueLowerBoundCert.Valid_spec :
    HeadValueLowerBoundCert.Valid = HeadValueLowerBoundCert.Valid := rfl
theorem HeadValueLowerBoundCert.check_spec :
    HeadValueLowerBoundCert.check = HeadValueLowerBoundCert.check := rfl
theorem HeadLogitDiffLowerBoundCert.Valid_spec :
    HeadLogitDiffLowerBoundCert.Valid = HeadLogitDiffLowerBoundCert.Valid := rfl
theorem HeadLogitDiffLowerBoundCert.check_spec :
    HeadLogitDiffLowerBoundCert.check = HeadLogitDiffLowerBoundCert.check := rfl
theorem HeadBestMatchPatternCert.Valid_spec :
    HeadBestMatchPatternCert.Valid = HeadBestMatchPatternCert.Valid := rfl
theorem HeadBestMatchPatternCert.check_spec :
    HeadBestMatchPatternCert.check = HeadBestMatchPatternCert.check := rfl
theorem HeadValueLowerBoundPosCert.Valid_spec :
    HeadValueLowerBoundPosCert.Valid = HeadValueLowerBoundPosCert.Valid := rfl
theorem HeadValueLowerBoundPosCert.check_spec :
    HeadValueLowerBoundPosCert.check = HeadValueLowerBoundPosCert.check := rfl
theorem HeadLogitDiffLowerBoundPosCert.Valid_spec :
    HeadLogitDiffLowerBoundPosCert.Valid = HeadLogitDiffLowerBoundPosCert.Valid := rfl
theorem HeadLogitDiffLowerBoundPosCert.check_spec :
    HeadLogitDiffLowerBoundPosCert.check = HeadLogitDiffLowerBoundPosCert.check := rfl
theorem InductionHeadSoundCert.Valid_spec :
    InductionHeadSoundCert.Valid = InductionHeadSoundCert.Valid := rfl
theorem InductionHeadSoundCert.check_spec :
    InductionHeadSoundCert.check = InductionHeadSoundCert.check := rfl
theorem InductionHeadBestMatchSoundCert.Valid_spec :
    InductionHeadBestMatchSoundCert.Valid = InductionHeadBestMatchSoundCert.Valid := rfl
theorem InductionHeadBestMatchSoundCert.check_spec :
    InductionHeadBestMatchSoundCert.check = InductionHeadBestMatchSoundCert.check := rfl

end Nfp.Sound
