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

/-- Push a probability vector forward along a mixer. -/
def push (M : Mixer ι κ) (p : ProbVec ι) : ProbVec κ :=
  { mass := fun k => ∑ i, p.mass i * M.weight i k
    sum_mass := by
      classical
      calc
        ∑ k, ∑ i, p.mass i * M.weight i k
            = ∑ i, ∑ k, p.mass i * M.weight i k := by
                simpa using
                  (Finset.sum_comm :
                    (∑ k : κ, ∑ i : ι, p.mass i * M.weight i k) =
                      ∑ i : ι, ∑ k : κ, p.mass i * M.weight i k)
        _ = ∑ i, p.mass i * ∑ k, M.weight i k := by
              refine Finset.sum_congr rfl ?_
              intro i _
              simpa using
                (Finset.mul_sum (a := p.mass i) (s := (Finset.univ : Finset κ))
                  (f := fun k => M.weight i k)).symm
        _ = ∑ i, p.mass i * 1 := by simp
        _ = 1 := by simp }

/-- Composition of two mixers. -/
def comp (M : Mixer ι κ) (N : Mixer κ α) : Mixer ι α :=
  { weight := fun i a => ∑ k, M.weight i k * N.weight k a
    row_sum := by
      classical
      intro i
      calc
        ∑ a, ∑ k, M.weight i k * N.weight k a
            = ∑ k, ∑ a, M.weight i k * N.weight k a := by
                simpa using
                  (Finset.sum_comm :
                    (∑ a : α, ∑ k : κ, M.weight i k * N.weight k a) =
                      ∑ k : κ, ∑ a : α, M.weight i k * N.weight k a)
        _ = ∑ k, M.weight i k * ∑ a, N.weight k a := by
              refine Finset.sum_congr rfl ?_
              intro k _
              simpa using
                (Finset.mul_sum (a := M.weight i k) (s := (Finset.univ : Finset α))
                  (f := fun a => N.weight k a)).symm
        _ = ∑ k, M.weight i k * 1 := by simp
        _ = 1 := by simp }

/-- Identity mixer. -/
def id (ι : Type u) [Fintype ι] [DecidableEq ι] : Mixer ι ι :=
  { weight := fun i j => (ProbVec.pure i).mass j
    row_sum := by
      intro i
      exact (ProbVec.pure i).sum_mass }

end Mixer
end Nfp
