-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.InductionHead
public import Nfp.IO.InductionHead.ModelSlice
public import Nfp.IO.Pure.InductionHead.ScoreCheck

/-!
Score-bound checks for anchoring induction-head certificate margins.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/--
Check that certificate margins/weight tolerances are justified by certificate scores.

This is a thin wrapper around the trusted IO.Pure implementation.
-/
def scoreBoundsWithinScores {seq : Nat}
    (cert : Circuit.InductionHeadCert seq) : Bool :=
  Pure.InductionHeadCert.scoreBoundsWithinScores cert

/-- Optional score-bound gate for model slices (returns an error message on failure). -/
def checkScoreBoundsWithinModel {seq : Nat}
    (modelSlice? : Option (ModelSlice seq))
    (cert : Circuit.InductionHeadCert seq) : Except String Unit :=
  match modelSlice? with
  | some modelSlice =>
      if Pure.InductionHeadCert.scoreBoundsWithinScoresWithMask
          (maskCausal := modelSlice.maskCausal) cert then
        pure ()
      else
        Except.error "score bounds not justified by certificate scores"
  | none => pure ()

end InductionHeadCert

end IO

end Nfp
