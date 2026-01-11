-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Basic

/-!
Well-formedness conditions for circuits.
-/

namespace Nfp

universe u v

namespace Circuit

variable {ι : Type u} [Fintype ι]
variable {α : Type v}

/-- A circuit is well-formed if every input node has no incoming edges. -/
def WellFormed (C : Circuit ι α) : Prop :=
  ∀ i ∈ C.inputs, ∀ j, ¬ C.dag.rel j i

/-- Inputs have no parents in a well-formed circuit. -/
theorem wellFormed_no_parent {C : Circuit ι α} (h : WellFormed C) {i j : ι} (hi : i ∈ C.inputs) :
    ¬ C.dag.rel j i :=
  by
    simpa using h i hi j

/-- Input nodes have empty parent sets in a well-formed circuit. -/
theorem wellFormed_parents_empty {C : Circuit ι α} (h : WellFormed C) {i : ι} (hi : i ∈ C.inputs) :
    C.dag.parents i = ∅ := by
  ext j
  simp [Dag.mem_parents, h i hi]

end Circuit

end Nfp
