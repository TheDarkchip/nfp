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
  c.ln1Bound =
      (if c.ln1VarianceLowerBound > 0 then
        layerNormOpBoundLocal c.ln1MaxAbsGamma c.ln1VarianceLowerBound eps
      else
        layerNormOpBoundConservative c.ln1MaxAbsGamma eps) ∧
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
  targetWeightLowerBound : Rat
  deriving Repr

namespace HeadPatternCert

/-- Internal consistency checks for pattern bounds. -/
def Valid (c : HeadPatternCert) : Prop :=
  c.seqLen > 0 ∧
    c.marginLowerBound = c.targetLogitLowerBound - c.otherLogitUpperBound ∧
    c.targetWeightLowerBound =
      (if c.marginLowerBound > 0 then
        (c.targetCountLowerBound : Rat) / (c.seqLen : Rat)
      else
        0)

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
  c.outputCoordLowerBound =
    c.matchWeightLowerBound * c.matchCoordLowerBound +
      (1 - c.matchWeightLowerBound) * c.nonmatchCoordLowerBound

instance (c : HeadValueLowerBoundCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadValueLowerBoundCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadValueLowerBoundCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end HeadValueLowerBoundCert

namespace HeadPatternCert

/-- Convert a sound head pattern certificate into a token-match witness. -/
def toTokenMatchPattern (c : HeadPatternCert) : Nfp.TokenMatchPattern := {
  seqLen := c.seqLen
  targetOffset := c.targetOffset
  targetCountLowerBound := c.targetCountLowerBound
  targetWeightLowerBound := c.targetWeightLowerBound
  marginLowerBound := c.marginLowerBound
}

theorem toTokenMatchPattern_valid (c : HeadPatternCert) (h : c.Valid) :
    (toTokenMatchPattern c).Valid := by
  rcases h with ⟨hseq, _hmargin, hweight⟩
  exact ⟨hseq, by simpa [toTokenMatchPattern] using hweight⟩

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
    c.deltaLowerBound = c.layer2Value.outputCoordLowerBound

instance (c : InductionHeadSoundCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : InductionHeadSoundCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : InductionHeadSoundCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end InductionHeadSoundCert

end Nfp.Sound
