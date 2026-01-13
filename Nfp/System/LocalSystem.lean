-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Fintype.BigOperators
public import Nfp.Mixer.Basic
public import Nfp.System.Dag

/-!
Local mixing systems on finite DAGs.
-/

@[expose] public section

open scoped BigOperators

namespace Nfp

universe u

/-- A local mixing system on a DAG.
`weight i j` is the contribution from `j` into `i`. -/
structure LocalSystem (ι : Type u) [Fintype ι] where
  /-- The underlying DAG describing allowed dependencies. -/
  dag : Dag ι
  /-- Mixing weights for each target/source pair. -/
  weight : ι → ι → Mass
  /-- Weights vanish off the edge relation. -/
  support : ∀ i j, ¬ dag.rel j i → weight i j = 0

namespace LocalSystem

variable {ι : Type u} [Fintype ι]

/-- Row-stochasticity for a local system. -/
def IsRowStochastic (L : LocalSystem ι) : Prop :=
  ∀ i, (∑ j, L.weight i j) = 1

/-- View a local system as a global mixer. -/
def toMixer (L : LocalSystem ι) (h : IsRowStochastic L) : Mixer ι ι :=
  { weight := L.weight
    row_sum := h }

/-- Off-edge weights are zero. -/
theorem weight_eq_zero_of_not_parent (L : LocalSystem ι) {i j : ι} (h : ¬ L.dag.rel j i) :
    L.weight i j = 0 :=
  by
    simpa using L.support i j h

/-- One-step evaluation functional used by `eval`. -/
def evalStep (L : LocalSystem ι) (input : ι → Mass)
    (i : ι) (rec : ∀ j, L.dag.rel j i → Mass) : Mass :=
  input i +
    ∑ j, (if h : L.dag.rel j i then L.weight i j * rec j h else 0)

/-- Evaluate a local system with external input at each node. -/
def eval (L : LocalSystem ι) (input : ι → Mass) : ι → Mass :=
  L.dag.wf.fix (fun i rec => evalStep L input i rec)

/-- Unfolding equation for `eval`. -/
theorem eval_eq (L : LocalSystem ι) (input : ι → Mass) (i : ι) :
    eval L input i =
      input i +
        ∑ j, (if _ : L.dag.rel j i then L.weight i j * eval L input j else 0) := by
  classical
  set F : ∀ i, (∀ j, L.dag.rel j i → Mass) → Mass := fun i rec => evalStep L input i rec
  change L.dag.wf.fix F i =
    input i + ∑ j, (if _ : L.dag.rel j i then L.weight i j * L.dag.wf.fix F j else 0)
  rw [WellFounded.fix_eq]
  simp [F, evalStep]

end LocalSystem

end Nfp

end
