-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Basic
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.Sound.Induction.ScoreBounds

/-!
Derived active-set selection for induction-aligned certificates.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

variable {seq : Nat}

/-- Active queries derived from score gaps and thresholds. -/
def derivedActive (cert : Circuit.InductionHeadCert seq)
    (expectedActive : Finset (Fin seq)) (minMargin maxEps : Rat) : Finset (Fin seq) :=
  let scoreGapLo := Sound.scoreGapLoOfBounds cert.prev cert.scores cert.scores
  let weightBoundAt := Sound.weightBoundAtOfScoreGap cert.prev scoreGapLo
  let epsAt := Sound.epsAtOfWeightBoundAt cert.prev weightBoundAt
  let marginAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (cert.prev q)
    if h : others.Nonempty then
      others.inf' h (fun k => scoreGapLo q k)
    else
      0
  expectedActive.filter (fun q =>
    decide (epsAt q ≤ maxEps) && decide (minMargin ≤ marginAt q))

/-- Coverage ratio for an active subset relative to the expected set. -/
def activeCoverage (active expectedActive : Finset (Fin seq)) : Option Rat :=
  if expectedActive.card = 0 then
    none
  else
    some (Rat.divInt (Int.ofNat active.card) (Int.ofNat expectedActive.card))

end InductionHeadCert

end IO

end Nfp
