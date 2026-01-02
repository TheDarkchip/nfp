-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Combinatorics.Digraph.Basic
import Mathlib.Data.Finset.Basic

/-!
Directed acyclic graph foundations.
-/

namespace Nfp

universe u

/-- A finite directed acyclic graph, built on top of `Digraph`. -/
structure Dag (ι : Type u) [Fintype ι] where
  /-- The underlying directed graph. -/
  graph : Digraph ι
  /-- Decidable adjacency for `graph.Adj`. -/
  decAdj : DecidableRel graph.Adj
  /-- The adjacency relation is well-founded. -/
  wf : WellFounded graph.Adj

attribute [instance] Dag.decAdj

namespace Dag

variable {ι : Type u} [Fintype ι]

/-- The edge relation of a DAG. -/
def rel (G : Dag ι) : ι → ι → Prop := G.graph.Adj

/-- Parents (incoming neighbors) of a node. -/
def parents (G : Dag ι) (i : ι) : Finset ι := by
  let _ : DecidablePred (fun j => G.rel j i) := fun j => G.decAdj j i
  exact Finset.filter (fun j => G.rel j i) Finset.univ

/-- Children (outgoing neighbors) of a node. -/
def children (G : Dag ι) (i : ι) : Finset ι := by
  let _ : DecidablePred (fun j => G.rel i j) := fun j => G.decAdj i j
  exact Finset.filter (fun j => G.rel i j) Finset.univ

@[simp] theorem mem_parents {G : Dag ι} {i j : ι} :
    j ∈ G.parents i ↔ G.rel j i := by
  simp [Dag.parents]

@[simp] theorem mem_children {G : Dag ι} {i j : ι} :
    j ∈ G.children i ↔ G.rel i j := by
  simp [Dag.children]

end Dag

end Nfp
