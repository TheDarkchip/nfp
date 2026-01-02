-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic

/-!
Directed acyclic graph foundations.
-/

namespace Nfp

universe u

/-- A finite directed acyclic graph with edge relation `rel`.
`rel u v` means there is an edge from `u` to `v`. -/
structure Dag (ι : Type u) [Fintype ι] where
  rel : ι → ι → Prop
  decRel : DecidableRel rel
  wf : WellFounded rel

attribute [instance] Dag.decRel

namespace Dag

variable {ι : Type u} [Fintype ι]

/-- Parents (incoming neighbors) of a node. -/
def parents (G : Dag ι) (i : ι) [DecidableEq ι] : Finset ι :=
  Finset.filter (fun j => G.rel j i) Finset.univ

/-- Children (outgoing neighbors) of a node. -/
def children (G : Dag ι) (i : ι) [DecidableEq ι] : Finset ι :=
  Finset.filter (fun j => G.rel i j) Finset.univ

@[simp] theorem mem_parents {G : Dag ι} [DecidableEq ι] {i j : ι} :
    j ∈ G.parents i ↔ G.rel j i := by
  simp [Dag.parents]

@[simp] theorem mem_children {G : Dag ι} [DecidableEq ι] {i j : ι} :
    j ∈ G.children i ↔ G.rel i j := by
  simp [Dag.children]

end Dag

end Nfp
