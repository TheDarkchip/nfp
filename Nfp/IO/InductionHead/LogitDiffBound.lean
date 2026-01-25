-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.InductionHead
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.Sound.Induction.ScoreBounds

/-!
Logit-diff lower bounds derived from induction-head certificates.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

variable {seq : Nat}

/-- Tight logit-diff lower bound over a chosen active set. -/
def logitDiffLowerBoundTightWithActive? (active : Finset (Fin seq))
    (cert : Circuit.InductionHeadCert seq) : Option Rat :=
  let base? :=
    Circuit.logitDiffLowerBoundAt active cert.prev cert.epsAt
      cert.values.lo cert.values.hi cert.values.vals
  let epsAtTight := Sound.epsAtOfWeightBoundAt cert.prev cert.weightBoundAt
  let valsLo := cert.values.valsLo
  let loAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (cert.prev q)
    if h : others.Nonempty then
      others.inf' h valsLo
    else
      cert.values.lo
  let tight? :=
    Circuit.logitDiffLowerBoundAtLoAt active cert.prev epsAtTight loAt valsLo
  match base?, tight? with
  | some a, some b => some (max a b)
  | some a, none => some a
  | none, some b => some b
  | none, none => none

/-- Tight logit-diff lower bound: max of global-interval and per-key lower bound variants. -/
def logitDiffLowerBoundTight? (cert : Circuit.InductionHeadCert seq) : Option Rat :=
  logitDiffLowerBoundTightWithActive? cert.active cert

end InductionHeadCert

end IO

end Nfp
