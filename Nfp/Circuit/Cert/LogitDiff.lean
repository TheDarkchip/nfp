-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Core.Basic
public import Mathlib.Data.Finset.Lattice.Fold
public import Nfp.Circuit.Layers.Induction

/-!
Lower bounds for logit-diff contributions from induction-style heads.
-/

public section

namespace Nfp

namespace Circuit

variable {seq : Nat}

/-- Compute a lower bound on the logit-diff contribution over active queries. -/
def logitDiffLowerBound (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (eps lo hi : Rat) (vals : Fin seq → Rat) : Option Rat := by
  classical
  if h : active.Nonempty then
    let gap := eps * (hi - lo)
    let f : Fin seq → Rat := fun q => vals (prev q) - gap
    exact some (active.inf' h f)
  else
    exact none

/-- Compute a lower bound on the logit-diff contribution with per-query eps. -/
def logitDiffLowerBoundAt (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (epsAt : Fin seq → Rat) (lo hi : Rat) (vals : Fin seq → Rat) : Option Rat := by
  classical
  if h : active.Nonempty then
    let gap : Fin seq → Rat := fun q => epsAt q * (hi - lo)
    let f : Fin seq → Rat := fun q => vals (prev q) - gap q
    exact some (active.inf' h f)
  else
    exact none

/-- Compute a lower bound on the logit-diff contribution using per-query eps and the global
    lower value bound. -/
def logitDiffLowerBoundAtLo (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (epsAt : Fin seq → Rat) (lo : Rat) (valsLo : Fin seq → Rat) : Option Rat := by
  classical
  if h : active.Nonempty then
    let f : Fin seq → Rat := fun q =>
      valsLo (prev q) - epsAt q * (valsLo (prev q) - lo)
    exact some (active.inf' h f)
  else
    exact none

/-- Compute a lower bound on the logit-diff contribution using per-query eps and per-query
    lower bounds for other values. -/
def logitDiffLowerBoundAtLoAt (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (epsAt : Fin seq → Rat) (loAt : Fin seq → Rat) (valsLo : Fin seq → Rat) : Option Rat := by
  classical
  if h : active.Nonempty then
    let f : Fin seq → Rat := fun q =>
      let delta := valsLo (prev q) - loAt q
      valsLo (prev q) - epsAt q * max (0 : Rat) delta
    exact some (active.inf' h f)
  else
    exact none

/-- Compute a lower bound on the logit-diff contribution using per-key weight bounds and
    per-key value lower bounds. -/
def logitDiffLowerBoundWeightedAt (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (weightBoundAt : Fin seq → Fin seq → Rat)
    (valsLo : Fin seq → Rat) : Option Rat := by
  classical
  if h : active.Nonempty then
    let gap : Fin seq → Rat := fun q =>
      (Finset.univ : Finset (Fin seq)).sum (fun k =>
        let diff := valsLo (prev q) - valsLo k
        weightBoundAt q k * max (0 : Rat) diff)
    let f : Fin seq → Rat := fun q => valsLo (prev q) - gap q
    exact some (active.inf' h f)
  else
    exact none

/-- Unfolding lemma for `logitDiffLowerBoundWeightedAt`. -/
theorem logitDiffLowerBoundWeightedAt_def (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (weightBoundAt : Fin seq → Fin seq → Rat)
    (valsLo : Fin seq → Rat) :
    logitDiffLowerBoundWeightedAt active prev weightBoundAt valsLo =
      by
        classical
        if h : active.Nonempty then
          let gap : Fin seq → Rat := fun q =>
            (Finset.univ : Finset (Fin seq)).sum (fun k =>
              let diff := valsLo (prev q) - valsLo k
              weightBoundAt q k * max (0 : Rat) diff)
          let f : Fin seq → Rat := fun q => valsLo (prev q) - gap q
          exact some (active.inf' h f)
        else
          exact none := by
  classical
  rfl

/-- The computed lower bound is below every active `prev` value minus the tolerance gap. -/
theorem logitDiffLowerBound_le (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (eps lo hi : Rat) (vals : Fin seq → Rat)
    (q : Fin seq) (hq : q ∈ active) :
    ∀ lb, logitDiffLowerBound active prev eps lo hi vals = some lb →
      lb ≤ vals (prev q) - eps * (hi - lo) := by
  classical
  intro lb hbound
  have hnonempty : active.Nonempty := ⟨q, hq⟩
  let gap := eps * (hi - lo)
  let f : Fin seq → Rat := fun q => vals (prev q) - gap
  have hbound' : active.inf' hnonempty f = lb := by
    simpa [logitDiffLowerBound, hnonempty, f, gap] using hbound
  have hmin : active.inf' hnonempty f ≤ f q :=
    Finset.inf'_le (s := active) (f := f) hq
  simpa [f, gap, hbound'] using hmin

/-- The per-query lower bound is below every active `prev` value minus the local gap. -/
theorem logitDiffLowerBoundAt_le (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (epsAt : Fin seq → Rat) (lo hi : Rat) (vals : Fin seq → Rat)
    (q : Fin seq) (hq : q ∈ active) :
    ∀ lb, logitDiffLowerBoundAt active prev epsAt lo hi vals = some lb →
      lb ≤ vals (prev q) - epsAt q * (hi - lo) := by
  classical
  intro lb hbound
  have hnonempty : active.Nonempty := ⟨q, hq⟩
  let gap : Fin seq → Rat := fun q => epsAt q * (hi - lo)
  let f : Fin seq → Rat := fun q => vals (prev q) - gap q
  have hbound' : active.inf' hnonempty f = lb := by
    simpa [logitDiffLowerBoundAt, hnonempty, f, gap] using hbound
  have hmin : active.inf' hnonempty f ≤ f q :=
    Finset.inf'_le (s := active) (f := f) hq
  simpa [f, gap, hbound'] using hmin

/-- The per-query lower bound is below every active `prev` value minus the `lo`-gap. -/
theorem logitDiffLowerBoundAtLo_le (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (epsAt : Fin seq → Rat) (lo : Rat) (valsLo : Fin seq → Rat)
    (q : Fin seq) (hq : q ∈ active) :
    ∀ lb, logitDiffLowerBoundAtLo active prev epsAt lo valsLo = some lb →
      lb ≤ valsLo (prev q) - epsAt q * (valsLo (prev q) - lo) := by
  classical
  intro lb hbound
  have hnonempty : active.Nonempty := ⟨q, hq⟩
  let f : Fin seq → Rat := fun q =>
    valsLo (prev q) - epsAt q * (valsLo (prev q) - lo)
  have hbound' : active.inf' hnonempty f = lb := by
    simpa [logitDiffLowerBoundAtLo, hnonempty, f] using hbound
  have hmin : active.inf' hnonempty f ≤ f q :=
    Finset.inf'_le (s := active) (f := f) hq
  simpa [f, hbound'] using hmin

/-- The per-query lower bound is below every active `prev` value minus the local `loAt` gap. -/
theorem logitDiffLowerBoundAtLoAt_le (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (epsAt : Fin seq → Rat) (loAt : Fin seq → Rat) (valsLo : Fin seq → Rat)
    (q : Fin seq) (hq : q ∈ active) :
    ∀ lb, logitDiffLowerBoundAtLoAt active prev epsAt loAt valsLo = some lb →
      lb ≤ valsLo (prev q) - epsAt q * max (0 : Rat) (valsLo (prev q) - loAt q) := by
  classical
  intro lb hbound
  have hnonempty : active.Nonempty := ⟨q, hq⟩
  let f : Fin seq → Rat := fun q =>
    let delta := valsLo (prev q) - loAt q
    valsLo (prev q) - epsAt q * max (0 : Rat) delta
  have hbound' : active.inf' hnonempty f = lb := by
    simpa [logitDiffLowerBoundAtLoAt, hnonempty, f] using hbound
  have hmin : active.inf' hnonempty f ≤ f q :=
    Finset.inf'_le (s := active) (f := f) hq
  simpa [f, hbound'] using hmin

/-- The weighted lower bound is below every active `prev` value minus the weighted gap. -/
theorem logitDiffLowerBoundWeightedAt_le (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (weightBoundAt : Fin seq → Fin seq → Rat)
    (valsLo : Fin seq → Rat)
    (q : Fin seq) (hq : q ∈ active) :
    ∀ lb, logitDiffLowerBoundWeightedAt active prev weightBoundAt valsLo = some lb →
      lb ≤
        valsLo (prev q) -
          (Finset.univ : Finset (Fin seq)).sum (fun k =>
            weightBoundAt q k * max (0 : Rat) (valsLo (prev q) - valsLo k)) := by
  classical
  intro lb hbound
  have hnonempty : active.Nonempty := ⟨q, hq⟩
  let gap : Fin seq → Rat := fun q =>
    (Finset.univ : Finset (Fin seq)).sum (fun k =>
      weightBoundAt q k * max (0 : Rat) (valsLo (prev q) - valsLo k))
  let f : Fin seq → Rat := fun q => valsLo (prev q) - gap q
  have hbound' : active.inf' hnonempty f = lb := by
    simpa [logitDiffLowerBoundWeightedAt, hnonempty, f, gap] using hbound
  have hmin : active.inf' hnonempty f ≤ f q :=
    Finset.inf'_le (s := active) (f := f) hq
  simpa [f, gap, hbound'] using hmin

end Circuit

end Nfp
