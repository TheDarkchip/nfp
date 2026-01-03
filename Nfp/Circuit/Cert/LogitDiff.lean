-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Finset.Image
import Nfp.Circuit.Layers.Induction

/-!
Lower bounds for logit-diff contributions from induction-style heads.
-/

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
  have hbound' :
      (active.image (fun q => vals (prev q) - eps * (hi - lo))).min'
          (hnonempty.image (fun q => vals (prev q) - eps * (hi - lo))) = lb := by
    simpa [logitDiffLowerBound, hnonempty] using hbound
  let gap := eps * (hi - lo)
  let f : Fin seq → Rat := fun q => vals (prev q) - gap
  have hmem : f q ∈ (active.image f) := by
    refine Finset.mem_image.2 ?_
    exact ⟨q, hq, rfl⟩
  have hmin : (active.image f).min' (hnonempty.image f) ≤ f q :=
    Finset.min'_le _ _ hmem
  have hlb : lb = (active.image f).min' (hnonempty.image f) := by
    simpa [f, gap] using hbound'.symm
  simpa [f, gap, hlb] using hmin

end Circuit

end Nfp
