-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Logic.Equiv.Defs
import Mathlib.Data.Fintype.Card
import Nfp.Mixer
import Nfp.Uniqueness

/-
Bridge from row-stochastic mixers on finite DAGs to `LocalSystem`.  This is
useful for reusing the `tracer_unique` theorem on mixers that come equipped
with a topological ordering.
-/

namespace Nfp

open Finset

variable {Site : Type*} [Fintype Site] [DecidableEq Site]

namespace LocalSystem

/-- Interpret a mixer as a `LocalSystem` using an explicit numbering of sites. -/
noncomputable def ofMixerIdx {n : ℕ} (M : Mixer Site Site) (e : Site ≃ Fin n)
    (acyclic : ∀ s t, M.w s t ≠ 0 → e t < e s) :
    LocalSystem n := by
  classical
  let siteOf : Fin n → Site := e.symm
  refine
    {
      Pa := fun i =>
        (Finset.univ.filter
          (fun u : Fin n =>
            M.w (siteOf i) (siteOf u) ≠ 0))
      c := fun i u => M.w (siteOf i) (siteOf u)
      topo := by
        intro i u hu
        have hmem := Finset.mem_filter.mp hu
        have hweight : M.w (siteOf i) (siteOf u) ≠ 0 := hmem.2
        have htopo : e (siteOf u) < e (siteOf i) := acyclic _ _ hweight
        simpa [siteOf] using htopo
    }

/-- Interpret a mixer as a `LocalSystem`, given a topological index `topo` and
a compatibility witness `respect` showing that the canonical `Fintype` ordering
aligns with `topo`. The `acyclic` assumption enforces the DAG constraint
`topo t < topo s` whenever `M.w s t` is nonzero. -/
noncomputable def ofMixer (M : Mixer Site Site) (topo : Site → ℕ)
    (acyclic : ∀ s t, M.w s t ≠ 0 → topo t < topo s)
    (respect :
      ∀ {s t}, topo s < topo t →
        (Fintype.equivFin Site s).1 < (Fintype.equivFin Site t).1) :
    LocalSystem (Fintype.card Site) := by
  classical
  let e : Site ≃ Fin (Fintype.card Site) := Fintype.equivFin Site
  have hindex :
      ∀ s t, M.w s t ≠ 0 → e t < e s := by
    intro s t hwt
    have htopo := acyclic s t hwt
    have horder : (Fintype.equivFin Site t).1 < (Fintype.equivFin Site s).1 :=
      respect htopo
    simpa [e] using horder
  exact ofMixerIdx (Site := Site) (M := M) (e := e) hindex

end LocalSystem

end Nfp
