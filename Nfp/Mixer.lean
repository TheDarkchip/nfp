-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Set.Basic
import Aesop
import Nfp.Prob

/-!
Forward mixers (Appendix A.1): abstract, finite, row-stochastic operators and
their basic closure properties. This formalizes the “forward mixer” notion from
the Appendix and shows that they preserve probability mass and are closed under
composition.
-/

namespace Nfp

open scoped BigOperators

variable {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]

/-- A row-stochastic nonnegative mixer from `S` to `T` (Appendix A.1). -/
structure Mixer (S T : Type*) [Fintype S] [Fintype T] where
  w : S → T → NNReal
  row_sum_one : ∀ i : S, (∑ j, w i j) = 1

namespace Mixer

variable (M : Mixer S T)

@[ext]
theorem ext {M N : Mixer S T} (h : ∀ i j, M.w i j = N.w i j) : M = N := by
  cases M; cases N; simp only [mk.injEq]; funext i j; exact h i j

/-- Apply a mixer to a probability row `p` to produce a probability row on `T`.
This corresponds to pushing tracer mass through a forward mixer (Appendix A.1). -/
noncomputable def push (p : ProbVec S) : ProbVec T :=
  {
    mass := fun j => ∑ i, p.mass i * M.w i j,
    norm_one := by
      classical
      -- Show ∑_j (∑_i p_i * w_{i,j}) = 1 by swapping sums and using row sums = 1
      have : (∑ j, (∑ i, p.mass i * M.w i j)) = 1 := by
        calc
          (∑ j, (∑ i, p.mass i * M.w i j))
              = (∑ i, (∑ j, p.mass i * M.w i j)) := by
                simpa [mul_comm, mul_left_comm, mul_assoc] using Finset.sum_comm
          _   = (∑ i, p.mass i * (∑ j, M.w i j)) := by
                simp [Finset.mul_sum]
          _   = (∑ i, p.mass i * 1) := by
                simp [M.row_sum_one]
          _   = (∑ i, p.mass i) := by
                simp
          _   = 1 := by
                simp [ProbVec.sum_mass (ι:=S) p]
      simp [this] }

/-- The `i`-th row of a mixer as a probability vector on the target type. -/
noncomputable def row (i : S) : ProbVec T :=
  {
    mass := fun j => M.w i j
    norm_one := by
      classical
      simpa using M.row_sum_one i
  }

@[simp] lemma row_mass (i : S) (j : T) : (M.row i).mass j = M.w i j := rfl

/-- Pushing a point-mass probability vector through a mixer selects the corresponding row. -/
theorem push_pure (i : S) : M.push (ProbVec.pure (ι := S) i) = M.row i := by
  ext j
  classical
  simp [Mixer.push, Mixer.row, ProbVec.pure, Finset.sum_ite_eq', Finset.mem_univ]

/-- `push` is affine: it commutes with convex mixtures of probability vectors. -/
theorem push_mix (c : NNReal) (hc : c ≤ 1) (p q : ProbVec S) :
    M.push (ProbVec.mix (ι := S) c hc p q) =
      ProbVec.mix (ι := T) c hc (M.push p) (M.push q) := by
  ext j
  classical
  simp [Mixer.push, ProbVec.mix, Finset.sum_add_distrib, add_mul, mul_assoc, Finset.mul_sum]

/-- Composition of mixers is again a mixer (closure under composition). -/
noncomputable def comp (N : Mixer T U) : Mixer S U :=
  {
    w := fun i k => ∑ j, M.w i j * N.w j k,
    row_sum_one := by
      classical
      intro i
      -- ∑_k ∑_j M i j * N j k = ∑_j M i j * (∑_k N j k) = ∑_j M i j * 1 = 1
      calc
        (∑ k, (∑ j, M.w i j * N.w j k)) = 1 := by
          -- Same calculation as above but stated in one calc for direct closure
          calc
            (∑ k, (∑ j, M.w i j * N.w j k))
                = (∑ j, (∑ k, M.w i j * N.w j k)) := by
                  simpa [mul_comm, mul_left_comm, mul_assoc] using Finset.sum_comm
            _   = (∑ j, M.w i j * (∑ k, N.w j k)) := by
                  simp [Finset.mul_sum]
            _   = (∑ j, M.w i j * 1) := by
                  simp [N.row_sum_one]
            _   = (∑ j, M.w i j) := by
                  simp
            _   = 1 := by
                  simpa using M.row_sum_one i
  }

end Mixer

end Nfp

/-!
Support-restricted mixers (Appendix A.1, support restriction): compact helpers
to state and propagate that a mixer has zero weights outside a given binary
relation `R : S → T → Prop`. These do not change the existing proofs; they
encode the “only route along executed edges” constraint as a property.
-/

namespace Nfp

open scoped BigOperators

variable {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]

namespace Mixer

/-- A mixer `M : Mixer S T` is `supported` by `R` if every weight outside `R` is zero. -/
def supported (M : Mixer S T) (R : S → T → Prop) : Prop :=
      ∀ i j, ¬ R i j → M.w i j = 0

@[simp] lemma supported_zero {M : Mixer S T} {R : S → T → Prop}
      (h : supported (S := S) (T := T) M R) {i : S} {j : T} (hij : ¬ R i j) : M.w i j = 0 :=
      h i j hij

attribute [aesop safe] supported_zero

/-- Relational composition of supports: `R ⋆ Q` allows an edge `i → k` iff there
exists `j` with `i → j` allowed by `R` and `j → k` allowed by `Q`. -/
def compSupport (R : S → T → Prop) (Q : T → U → Prop) : S → U → Prop :=
      fun i k => ∃ j, R i j ∧ Q j k

/-- If `M` is supported by `R` and `N` by `Q`, then `M.comp N` is supported by `R ⋆ Q`. -/
lemma supported_comp {M : Mixer S T} {N : Mixer T U}
      {R : S → T → Prop} {Q : T → U → Prop}
      (hM : supported (S := S) (T := T) M R) (hN : supported (S := T) (T := U) N Q) :
      supported (S := S) (T := U) (M.comp N) (compSupport R Q) := by
      classical
      intro i k hnot
      -- For every j, either ¬R i j or ¬Q j k; hence the product weight vanishes.
      have hforall : ∀ j, M.w i j * N.w j k = 0 := by
        intro j
        have hnot_and : ¬ (R i j ∧ Q j k) := by
          -- `¬ ∃ j, R i j ∧ Q j k` ⇒ `¬ (R i j ∧ Q j k)` for each `j`.
          intro hAnd
          exact hnot ⟨j, hAnd⟩
        have hdisj : ¬ R i j ∨ ¬ Q j k := (not_and_or.mp hnot_and)
        aesop (add safe [hM, hN])
      -- Sum of zero terms equals 0
      have : (∑ j, M.w i j * N.w j k) = 0 := by
            have hfun : (fun j => M.w i j * N.w j k) = (fun _ => (0 : NNReal)) := by
                  funext j; simpa using hforall j
            simp [hfun]
      -- This is exactly the weight on `(i,k)` inside `M.comp N`.
      simp [Mixer.comp, this]

/-- The support (positions with nonzero mass) of a probability vector. -/
def supp (p : ProbVec S) : Set S := fun i => p.mass i ≠ 0

/-- Image of a set of sources along a binary support relation. -/
def image (R : S → T → Prop) (A : Set S) : Set T := fun k => ∃ i, A i ∧ R i k

/-- Support propagation: if `M` is supported by `R`, then any nonzero pushed mass at `k`
originates from some source `i` with nonzero mass and an allowed edge `R i k`. -/
lemma supp_push_subset_image {M : Mixer S T} {R : S → T → Prop}
    (hM : supported (S := S) (T := T) M R) (p : ProbVec S) :
    supp (S := T) (M.push p) ⊆ image (S := S) (T := T) R (supp (S := S) p) := by
  classical
  intro k hk
  have hk_mass : (M.push p).mass k ≠ 0 := hk
  have hk' : (∑ i, p.mass i * M.w i k) ≠ 0 := by
    simpa [Mixer.push] using hk_mass
  obtain ⟨i, _hi, hne⟩ := Finset.exists_ne_zero_of_sum_ne_zero hk'
  have hpi : p.mass i ≠ 0 := by
    intro h0
    exact hne (by simp [h0])
  have hwi : M.w i k ≠ 0 := by
    intro h0
    exact hne (by simp [h0])
  have hR : R i k := by
    by_contra hR
    exact hwi (hM i k hR)
  exact ⟨i, ⟨hpi, hR⟩⟩

end Mixer

end Nfp
