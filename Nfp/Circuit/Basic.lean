-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.System.Dag

/-!
Circuit foundations: a DAG with designated inputs/outputs and gate semantics.
-/

namespace Nfp

universe u v

/-- A finite circuit on a DAG with designated inputs/outputs and per-node gate semantics. -/
structure Circuit (ι : Type u) [Fintype ι] (α : Type v) where
  /-- The underlying DAG that orders dependencies. -/
  dag : Dag ι
  /-- Input nodes read from the external assignment. -/
  inputs : Finset ι
  /-- Output nodes observed after evaluation. -/
  outputs : Finset ι
  /-- Gate semantics at each node, given values of its parents. -/
  gate : ∀ i, (∀ j, dag.rel j i → α) → α

namespace Circuit

variable {ι : Type u} [Fintype ι] {α : Type v}

/-- External input assignment on the circuit's input nodes. -/
abbrev InputAssignment (C : Circuit ι α) : Type (max u v) :=
  { i // i ∈ C.inputs } → α

/-- Reinterpret input assignments along an equality of input sets. -/
def InputAssignment.cast {C₁ C₂ : Circuit ι α} (h : C₁.inputs = C₂.inputs) :
    InputAssignment C₁ → InputAssignment C₂ := by
  intro input i
  refine input ⟨i.1, ?_⟩
  simp [h]

end Circuit

end Nfp
