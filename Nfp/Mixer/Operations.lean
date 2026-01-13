-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Mixer.Basic
public import Nfp.Prob.Operations
public import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
Mixer operations (pushforward, composition, identity).
-/

@[expose] public section

open scoped BigOperators

namespace Nfp
namespace Mixer

universe u

variable {ι κ α : Type u} [Fintype ι] [Fintype κ] [Fintype α]

/-- Swap a double sum and factor the inner sum out by multiplication. -/
private lemma sum_mul_sum (p : ι → Mass) (w : ι → κ → Mass) :
    (∑ k, ∑ i, p i * w i k) = ∑ i, p i * ∑ k, w i k := by
  classical
  calc
    ∑ k, ∑ i, p i * w i k = ∑ i, ∑ k, p i * w i k := by
      exact
        (Finset.sum_comm (s := (Finset.univ : Finset κ)) (t := (Finset.univ : Finset ι))
          (f := fun k i => p i * w i k))
    _ = ∑ i, p i * ∑ k, w i k := by
      refine Finset.sum_congr rfl ?_
      intro i _
      simp [Finset.mul_sum]

/-- Push a probability vector forward along a mixer. -/
def push (M : Mixer ι κ) (p : ProbVec ι) : ProbVec κ :=
  { mass := fun k => ∑ i, p.mass i * M.weight i k
    sum_mass := by
      classical
      simp [sum_mul_sum] }

/-- Composition of two mixers. -/
def comp (M : Mixer ι κ) (N : Mixer κ α) : Mixer ι α :=
  { weight := fun i a => ∑ k, M.weight i k * N.weight k a
    row_sum := by
      classical
      intro i
      simp [sum_mul_sum] }

/-- Identity mixer. -/
def id (ι : Type u) [Fintype ι] [DecidableEq ι] : Mixer ι ι :=
  { weight := fun i j => (ProbVec.pure i).mass j
    row_sum := by
      intro i
      simp }

end Mixer
end Nfp

end
