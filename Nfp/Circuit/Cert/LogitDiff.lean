-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Core.Basic
public import Mathlib.Data.Finset.Image
public import Nfp.Circuit.Layers.Induction

/-!
Lower bounds for logit-diff contributions from induction-style heads.
-/

@[expose] public section

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
    let img := active.image f
    have himg : img.Nonempty := h.image f
    exact some (Finset.min' img himg)
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
    let img := active.image f
    have himg : img.Nonempty := h.image f
    exact some (Finset.min' img himg)
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
    let img := active.image f
    have himg : img.Nonempty := h.image f
    exact some (Finset.min' img himg)
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
    let others : Fin seq → Finset (Fin seq) := fun q =>
      (Finset.univ : Finset (Fin seq)).erase (prev q)
    let gap : Fin seq → Rat := fun q =>
      (others q).sum (fun k =>
        let diff := valsLo (prev q) - valsLo k
        weightBoundAt q k * max (0 : Rat) diff)
    let f : Fin seq → Rat := fun q => valsLo (prev q) - gap q
    let img := active.image f
    have himg : img.Nonempty := h.image f
    exact some (Finset.min' img himg)
  else
    exact none

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
  have hbound' : (active.image f).min' (hnonempty.image f) = lb := by
    simpa [logitDiffLowerBound, hnonempty, f, gap] using hbound
  have hmem : f q ∈ (active.image f) := by
    refine Finset.mem_image.2 ?_
    exact ⟨q, hq, rfl⟩
  have hmin : (active.image f).min' (hnonempty.image f) ≤ f q :=
    Finset.min'_le _ _ hmem
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
  have hbound' : (active.image f).min' (hnonempty.image f) = lb := by
    simpa [logitDiffLowerBoundAt, hnonempty, f, gap] using hbound
  have hmem : f q ∈ (active.image f) := by
    refine Finset.mem_image.2 ?_
    exact ⟨q, hq, rfl⟩
  have hmin : (active.image f).min' (hnonempty.image f) ≤ f q :=
    Finset.min'_le _ _ hmem
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
  have hbound' : (active.image f).min' (hnonempty.image f) = lb := by
    simpa [logitDiffLowerBoundAtLo, hnonempty, f] using hbound
  have hmem : f q ∈ (active.image f) := by
    refine Finset.mem_image.2 ?_
    exact ⟨q, hq, rfl⟩
  have hmin : (active.image f).min' (hnonempty.image f) ≤ f q :=
    Finset.min'_le _ _ hmem
  have hmin' : lb ≤ f q := by
    simpa [hbound'] using hmin
  simpa [f] using hmin'

/-- The weighted lower bound is below every active `prev` value minus the weighted gap. -/
theorem logitDiffLowerBoundWeightedAt_le (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (weightBoundAt : Fin seq → Fin seq → Rat)
    (valsLo : Fin seq → Rat)
    (q : Fin seq) (hq : q ∈ active) :
    ∀ lb, logitDiffLowerBoundWeightedAt active prev weightBoundAt valsLo = some lb →
      lb ≤
        valsLo (prev q) -
          ((Finset.univ : Finset (Fin seq)).erase (prev q)).sum (fun k =>
            weightBoundAt q k * max (0 : Rat) (valsLo (prev q) - valsLo k)) := by
  classical
  intro lb hbound
  have hnonempty : active.Nonempty := ⟨q, hq⟩
  let others : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (prev q)
  let gap : Fin seq → Rat := fun q =>
    (others q).sum (fun k =>
      let diff := valsLo (prev q) - valsLo k
      weightBoundAt q k * max (0 : Rat) diff)
  let f : Fin seq → Rat := fun q => valsLo (prev q) - gap q
  have hbound' : (active.image f).min' (hnonempty.image f) = lb := by
    simpa [logitDiffLowerBoundWeightedAt, hnonempty, f, gap, others] using hbound
  have hmem : f q ∈ (active.image f) := by
    refine Finset.mem_image.2 ?_
    exact ⟨q, hq, rfl⟩
  have hmin : (active.image f).min' (hnonempty.image f) ≤ f q :=
    Finset.min'_le _ _ hmem
  have hmin' : lb ≤ f q := by
    simpa [hbound'] using hmin
  have hmin'' :
      lb ≤ valsLo (prev q) -
        (others q).sum (fun k =>
          let diff := valsLo (prev q) - valsLo k
          weightBoundAt q k * max (0 : Rat) diff) := by
    simpa [f, gap] using hmin'
  simpa [others] using hmin''

end Circuit

end Nfp
