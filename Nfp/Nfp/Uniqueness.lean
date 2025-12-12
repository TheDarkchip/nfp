import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Aesop

namespace Nfp

open scoped BigOperators
open Finset

variable {S : Type*}

/-!
Auxiliary linear-uniqueness lemma used to support the Appendix A.3 uniqueness
story (fixed masks ⇒ linear system). It shows that for a finite DAG ordered by a
topological index, any two tracer families satisfying the same homogeneous linear
recurrence coincide. This is a helper fact; Appendix A.5 in the paper is a
counterexample narrative (not a formal lemma), so we avoid referring to A.5 here.
Small parent-index checks in the inductive step are discharged by `aesop`; the
overall proof structure (nested induction on the index bound) remains explicit.
-/

/-- A local mixing system over `n` nodes, where each node `i` aggregates parents
with nonnegative coefficients, and every parent `u` of `i` has a strictly smaller index. -/
structure LocalSystem (n : ℕ) where
  Pa : Fin n → Finset (Fin n)
  c  : Fin n → Fin n → NNReal
  topo : ∀ {i u}, u ∈ Pa i → (u.1 < i.1)

namespace LocalSystem

variable {n : ℕ}

/-- A family of tracer masses at each node. It does not depend on `L`. -/
abbrev TracerFamily (n : ℕ) := Fin n → (S → NNReal)

/-- The recurrence equation for a tracer family against the local system. -/
def Satisfies (L : LocalSystem n) (T : TracerFamily (S := S) n) : Prop :=
  ∀ i s, T i s = (L.Pa i).sum (fun u => L.c i u * T u s)

/-- Homogeneous linear uniqueness on a topologically ordered finite DAG: if `T` and
`T'` both satisfy the same recurrence, then they are equal pointwise. -/
theorem tracer_unique (L : LocalSystem n) {T T' : TracerFamily (S := S) n}
  (hT : Satisfies (S := S) L T) (hT' : Satisfies (S := S) L T') : T = T' := by
  classical
  funext i s
  -- Prove by induction on an upper bound `k` of the index.
  have main : ∀ k, ∀ j : Fin n, j.1 ≤ k → T j s = T' j s := by
    intro k
    induction k with
    | zero =>
      intro j hj
      have hj0 : j.1 = 0 := Nat.le_zero.mp hj
      -- No parents at index 0
      have hPaEmpty : L.Pa j = (∅ : Finset (Fin n)) := by
        classical
        apply Finset.eq_empty_iff_forall_notMem.mpr
        intro u hu
        have hlt : u.1 < j.1 := L.topo (i := j) (u := u) hu
        have hlt0 : u.1 < 0 := by
          have := hlt
          rwa [hj0] at this
        exact (Nat.not_lt_zero _ hlt0).elim
      have hjT : T j s = (L.Pa j).sum (fun u => L.c j u * T u s) := (hT j s)
      have hjT' : T' j s = (L.Pa j).sum (fun u => L.c j u * T' u s) := (hT' j s)
      have hz1 : T j s = 0 := by simpa [hPaEmpty] using hjT
      have hz2 : T' j s = 0 := by simpa [hPaEmpty] using hjT'
      simpa [hz1] using hz2.symm
    | succ k ih =>
      intro j hj
      by_cases hle : j.1 ≤ k
      · exact ih j hle
      · have hklt : k < j.1 := Nat.not_le.mp hle
        have hjeq : j.1 = k.succ := le_antisymm hj (Nat.succ_le_of_lt hklt)
        have hjT : T j s = (L.Pa j).sum (fun u => L.c j u * T u s) := (hT j s)
        have hjT' : T' j s = (L.Pa j).sum (fun u => L.c j u * T' u s) := (hT' j s)
        have hparents : ∀ u ∈ L.Pa j, u.1 ≤ k := by
          intro u hu
          have hlt : u.1 < k.succ := by
            simpa [hjeq] using L.topo (i := j) (u := u) hu
          have hle : u.1 ≤ k := Nat.le_of_lt_succ hlt
          aesop
        have hsum : (L.Pa j).sum (fun u => L.c j u * T u s) =
                     (L.Pa j).sum (fun u => L.c j u * T' u s) := by
          classical
          apply Finset.sum_congr rfl
          intro u hu
          have := ih u (hparents u hu)
          simp [this]
        simp [hjT, hjT', hsum]
  exact main i.1 i le_rfl

end LocalSystem

end Nfp
