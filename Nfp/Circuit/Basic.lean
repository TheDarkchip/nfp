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

end Nfp
