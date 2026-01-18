-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public meta import Nfp.Tactic.Linter
public import Mathlib.Combinatorics.Digraph.Basic
public import Mathlib.Data.Fintype.Defs
public import Mathlib.Data.Finset.Basic

/-!
Directed acyclic graph foundations.
-/

public section

namespace Nfp

universe u u'

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

/-- Coerce a DAG to its underlying digraph. -/
instance : Coe (Dag ι) (Digraph ι) := ⟨Dag.graph⟩

/-- The edge relation of a DAG. -/
abbrev rel (G : Dag ι) : ι → ι → Prop := G.graph.Adj

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

section Relabel

variable {ι' : Type u'} [Fintype ι']

/-- Relabel a DAG along an equivalence of vertex types. -/
def relabel (G : Dag ι) (e : ι ≃ ι') : Dag ι' :=
  { graph := { Adj := fun a b => G.rel (e.symm a) (e.symm b) }
    decAdj := by
      intro a b
      exact G.decAdj (e.symm a) (e.symm b)
    wf := by
      simpa using (InvImage.wf (f := e.symm) (h := G.wf)) }

/-- Relabeling preserves adjacency via the equivalence. -/
theorem relabel_rel_iff (G : Dag ι) (e : ι ≃ ι') (a b : ι') :
    (G.relabel e).rel a b ↔ G.rel (e.symm a) (e.symm b) := by
  rfl

end Relabel

end Dag

end Nfp

end
