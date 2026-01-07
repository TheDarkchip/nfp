-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Mixer.Basic
import Nfp.Prob.Operations
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
Mixer operations (pushforward, composition, identity).
-/

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
      simpa using
        (Finset.sum_comm :
          (∑ k : κ, ∑ i : ι, p i * w i k) = ∑ i : ι, ∑ k : κ, p i * w i k)
    _ = ∑ i, p i * ∑ k, w i k := by
      refine Finset.sum_congr rfl ?_
      intro i _
      simpa using
        (Finset.mul_sum (a := p i) (s := (Finset.univ : Finset κ))
          (f := fun k => w i k)).symm

/-- Push a probability vector forward along a mixer. -/
def push (M : Mixer ι κ) (p : ProbVec ι) : ProbVec κ :=
  { mass := fun k => ∑ i, p.mass i * M.weight i k
    sum_mass := by
      classical
      calc
        ∑ k, ∑ i, p.mass i * M.weight i k
            = ∑ i, p.mass i * ∑ k, M.weight i k := by
                simpa using sum_mul_sum (p := fun i => p.mass i) (w := fun i => M.weight i)
        _ = 1 := by simp }

/-- Composition of two mixers. -/
def comp (M : Mixer ι κ) (N : Mixer κ α) : Mixer ι α :=
  { weight := fun i a => ∑ k, M.weight i k * N.weight k a
    row_sum := by
      classical
      intro i
      calc
        ∑ a, ∑ k, M.weight i k * N.weight k a
            = ∑ k, M.weight i k * ∑ a, N.weight k a := by
                simpa using
                  sum_mul_sum (p := fun k => M.weight i k) (w := fun k => N.weight k)
        _ = 1 := by simp }

/-- Identity mixer. -/
def id (ι : Type u) [Fintype ι] [DecidableEq ι] : Mixer ι ι :=
  { weight := fun i j => (ProbVec.pure i).mass j
    row_sum := by
      intro i
      exact (ProbVec.pure i).sum_mass }

end Mixer
end Nfp
