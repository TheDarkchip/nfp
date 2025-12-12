import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.FinCases

/-!
Partitions and reroute plans.  This file hosts the element-level lemmas
needed by the reroute weighting development (Stage 1).  It provides:

* a recursive definition of `unionParts` with rewrite-friendly lemmas
* helpers showing membership existence/uniqueness under disjointness
* an intrinsic notion of `ReroutePlan`
* incremental masks (`increments`) together with coverage and disjointness
* per-element sum lemmas that decompose along disjoint parts
-/

namespace Nfp

open scoped BigOperators
open Finset

section Partitions

variable {S : Type*} [DecidableEq S]

/-- Union a list of disjoint components. The recursion is `simp`-friendly so
Lean can peel the head element while reasoning about membership. -/
@[simp] def unionParts : List (Finset S) → Finset S
  | [] => ∅
  | A :: parts => A ∪ unionParts parts

@[simp] lemma unionParts_nil : unionParts ([] : List (Finset S)) = (∅ : Finset S) := rfl

@[simp] lemma unionParts_cons (A : Finset S) (parts : List (Finset S)) :
    unionParts (A :: parts) = A ∪ unionParts parts := rfl

@[simp] lemma unionParts_singleton (A : Finset S) :
    unionParts [A] = A := by simp

lemma unionParts_cons_mem {A : Finset S} {parts : List (Finset S)} {i : S} :
    i ∈ unionParts (A :: parts) ↔ i ∈ A ∨ i ∈ unionParts parts := by
  classical
  simp [unionParts, Finset.mem_union]

lemma mem_unionParts_iff_exists_mem (parts : List (Finset S)) (i : S) :
    i ∈ unionParts parts ↔ ∃ A ∈ parts, i ∈ A := by
  classical
  induction parts with
  | nil =>
      simp
  | cons A parts ih =>
      simp [unionParts, ih, List.mem_cons]

lemma mem_unionParts_of_mem {parts : List (Finset S)} {A : Finset S} (hA : A ∈ parts)
    {i : S} (hi : i ∈ A) : i ∈ unionParts parts := by
  classical
  exact (mem_unionParts_iff_exists_mem (parts:=parts) (i:=i)).2 ⟨A, hA, hi⟩

lemma disjoint_unionParts (A : Finset S) (parts : List (Finset S))
    (h : ∀ B ∈ parts, Disjoint A B) :
    Disjoint A (unionParts parts) := by
  classical
  induction parts with
  | nil =>
      simp
  | cons B parts ih =>
      have hAB : Disjoint A B := h B (by simp)
      have htail : ∀ C ∈ parts, Disjoint A C := by
        intro C hC
        exact h C (by simp [hC])
      have hArest := ih htail
      refine Finset.disjoint_left.mpr ?_
      intro x hxA hxUnion
      have : x ∈ B ∨ x ∈ unionParts parts := by
        simpa [unionParts] using hxUnion
      cases this with
      | inl hxB =>
          exact (Finset.disjoint_left.mp hAB) hxA hxB
      | inr hxRest =>
          exact (Finset.disjoint_left.mp hArest) hxA hxRest

example (A B : Finset S) (i : S) :
    i ∈ unionParts [A, B] ↔ i ∈ A ∨ i ∈ B := by
  classical
  simp [unionParts]

end Partitions

lemma pairwise_disjoint_cons_elim {S : Type*} {A : Finset S} {parts : List (Finset S)}
    (hpair : (A :: parts).Pairwise (fun B C => Disjoint B C)) :
    (∀ B ∈ parts, Disjoint A B) ∧
    parts.Pairwise (fun B C => Disjoint B C) :=
  List.pairwise_cons.mp hpair

lemma mem_unionParts_unique {S : Type*} (parts : List (Finset S))
    (hpair : parts.Pairwise (fun A B => Disjoint A B))
    {A B : Finset S} {i : S}
    (hA : A ∈ parts) (hB : B ∈ parts)
    (hiA : i ∈ A) (hiB : i ∈ B) :
    A = B := by
  classical
  revert A B
  induction parts with
  | nil =>
      intro A B hA hB _ _
      cases hA
  | cons C parts ih =>
      intro A B hA hB hiA hiB
      rcases List.pairwise_cons.mp hpair with ⟨hC, htail⟩
      rcases List.mem_cons.mp hA with hAC | hA'
      · rcases List.mem_cons.mp hB with hBC | hB'
        · cases hAC; cases hBC; exact rfl
        · have hiC : i ∈ C := by simpa [hAC] using hiA
          have hCB : Disjoint C B := hC B hB'
          exact ((Finset.disjoint_left.mp hCB) hiC hiB).elim
      · rcases List.mem_cons.mp hB with hBC | hB'
        · have hiC : i ∈ C := by simpa [hBC] using hiB
          have hCA : Disjoint C A := hC A hA'
          exact
            ((Finset.disjoint_left.mp hCA) hiC
                (by simpa [hA'] using hiA)).elim
        · exact ih htail hA' hB' hiA hiB

section PartitionsFinite

variable {S : Type*} [Fintype S] [DecidableEq S]

lemma mem_unionParts_of_univ (parts : List (Finset S))
    (hcover : unionParts parts = (Finset.univ : Finset S))
    (i : S) : i ∈ unionParts parts := by
  classical
  simp [hcover]

end PartitionsFinite

section SumLemmas

variable {S : Type*} [DecidableEq S]

lemma sum_unionParts_eq (m : S → NNReal) (parts : List (Finset S))
    (hpair : parts.Pairwise (fun A B => Disjoint A B)) :
    (unionParts parts).sum m =
      parts.foldr (fun A acc => A.sum m + acc) 0 := by
  classical
  induction parts with
  | nil =>
      simp [unionParts]
  | cons A parts ih =>
      rcases List.pairwise_cons.mp hpair with ⟨hA, htail⟩
      have hdisj : Disjoint A (unionParts parts) :=
        disjoint_unionParts (A:=A) (parts:=parts) (fun B hB => hA _ (by simpa using hB))
      have hsum := Finset.sum_union (s₁:=A) (s₂:=unionParts parts) (f:=m) hdisj
      have hfold :
          (A :: parts).foldr (fun B acc => B.sum m + acc) 0
              = A.sum m + parts.foldr (fun B acc => B.sum m + acc) 0 := by
        simp [List.foldr]
      calc
        (unionParts (A :: parts)).sum m
            = (A ∪ unionParts parts).sum m := rfl
        _ = A.sum m + (unionParts parts).sum m := hsum
        _ = A.sum m + parts.foldr (fun B acc => B.sum m + acc) 0 := by
              simp [ih htail]
        _ = (A :: parts).foldr (fun B acc => B.sum m + acc) 0 := by
              simp [hfold]

end SumLemmas

section ReroutePlanStruct

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- A reroute plan is a list of disjoint masks that cover the whole space. -/
structure ReroutePlan (S : Type*) [Fintype S] [DecidableEq S] where
  steps : List (Finset S)
  pairwise_disjoint : steps.Pairwise (fun A B => Disjoint A B)
  covers_univ : unionParts steps = (Finset.univ : Finset S)

namespace ReroutePlan

variable (S)

/-- Re-export the list of masks. Useful when inferring implicit arguments. -/
def masks (P : ReroutePlan (S := S)) : List (Finset S) := P.steps

variable {S}

lemma mem_cover (P : ReroutePlan (S := S)) (i : S) :
    i ∈ unionParts P.steps :=
  mem_unionParts_of_univ (parts:=P.steps) P.covers_univ i

lemma exists_block (P : ReroutePlan (S := S)) (i : S) :
    ∃ A ∈ P.steps, i ∈ A :=
  (mem_unionParts_iff_exists_mem (parts:=P.steps) (i:=i)).1 (P.mem_cover i)

lemma unique_block (P : ReroutePlan (S := S)) (i : S) :
    ∃! A, A ∈ P.steps ∧ i ∈ A := by
  classical
  obtain ⟨A, hA, hiA⟩ := P.exists_block i
  refine ⟨A, ⟨hA, hiA⟩, ?_⟩
  intro B hB
  apply mem_unionParts_unique (parts:=P.steps) P.pairwise_disjoint <;>
  tauto

end ReroutePlan

end ReroutePlanStruct

section Increments

namespace ReroutePlan

section Aux

variable {S : Type*} [DecidableEq S]

private def incrementsAux : List (Finset S) → Finset S → List (Finset S)
  | [], _ => []
  | A :: parts, seen => (A \ seen) :: incrementsAux parts (seen ∪ A)

@[simp] lemma incrementsAux_nil (seen : Finset S) :
    incrementsAux ([] : List (Finset S)) seen = [] := rfl

@[simp] lemma incrementsAux_cons (A : Finset S) (parts : List (Finset S))
    (seen : Finset S) :
    incrementsAux (A :: parts) seen =
      (A \ seen) :: incrementsAux parts (seen ∪ A) := rfl

lemma incrementsAux_length (parts : List (Finset S)) (seen : Finset S) :
    (incrementsAux parts seen).length = parts.length := by
  classical
  induction parts generalizing seen with
  | nil =>
      simp [incrementsAux]
  | cons A parts ih =>
      simp [incrementsAux, ih]

private lemma incrementsAux_mem_disjoint_seen :
    ∀ {parts : List (Finset S)} {seen B : Finset S},
      B ∈ incrementsAux parts seen → Disjoint B seen := by
  classical
  intro parts
  induction parts with
  | nil =>
      intro seen B hB
      cases hB
  | cons A parts ih =>
      intro seen B hB
      dsimp [incrementsAux] at hB
      rcases List.mem_cons.mp hB with hHead | hTail
      · subst hHead
        refine Finset.disjoint_left.mpr ?_
        intro x hxB hxSeen
        exact (Finset.mem_sdiff.mp hxB).2 hxSeen
      · have h := ih (seen := seen ∪ A) (B := B) hTail
        refine Finset.disjoint_left.mpr ?_
        intro x hxB hxSeen
        have hxUnion : x ∈ seen ∪ A := by
          have : x ∈ seen := hxSeen
          exact (Finset.mem_union.mpr (Or.inl this))
        exact (Finset.disjoint_left.mp h) hxB hxUnion

private lemma incrementsAux_pairwise (parts : List (Finset S)) (seen : Finset S) :
    (incrementsAux parts seen).Pairwise (fun A B => Disjoint A B) := by
  classical
  induction parts generalizing seen with
  | nil =>
      simp [incrementsAux]
  | cons A parts ih =>
      refine List.pairwise_cons.2 ?_
      constructor
      · intro B hB
        have hDisjoint :=
          incrementsAux_mem_disjoint_seen (parts:=parts) (seen:=seen ∪ A) (B:=B) hB
        refine Finset.disjoint_left.mpr ?_
        intro x hxHead hxB
        aesop (add simp [Finset.disjoint_left, incrementsAux])
      · simpa using ih (seen ∪ A)

private lemma sdiff_union_left (A B C : Finset S) :
    (A \ C) ∪ (B \ (C ∪ A)) = (A ∪ B) \ C := by
  classical
  ext i
  constructor
  · intro hx
    rcases Finset.mem_union.mp hx with hxA | hxB
    · rcases Finset.mem_sdiff.mp hxA with ⟨hiA, hiC⟩
      exact Finset.mem_sdiff.mpr ⟨Finset.mem_union.mpr (Or.inl hiA), hiC⟩
    · rcases Finset.mem_sdiff.mp hxB with ⟨hiB, hiCompl⟩
      have hiC : i ∉ C := by
        exact fun hCi => hiCompl (Finset.mem_union.mpr (Or.inl hCi))
      exact Finset.mem_sdiff.mpr ⟨Finset.mem_union.mpr (Or.inr hiB), hiC⟩
  · intro hx
    rcases Finset.mem_sdiff.mp hx with ⟨hiUnion, hiC⟩
    by_cases hiA : i ∈ A
    · exact Finset.mem_union.mpr (Or.inl (Finset.mem_sdiff.mpr ⟨hiA, hiC⟩))
    · have hiB : i ∈ B := by
        have : i ∈ A ∨ i ∈ B := (Finset.mem_union.mp hiUnion)
        exact this.resolve_left hiA
      have hiNot : i ∉ C ∪ A := by
        intro hmem
        rcases Finset.mem_union.mp hmem with hC | hA
        · exact hiC hC
        · exact hiA hA
      exact Finset.mem_union.mpr (Or.inr (Finset.mem_sdiff.mpr ⟨hiB, hiNot⟩))

private lemma unionParts_incrementsAux (parts : List (Finset S)) (seen : Finset S) :
    unionParts (incrementsAux parts seen) = unionParts parts \ seen := by
  classical
  induction parts generalizing seen with
  | nil =>
      simp [incrementsAux]
  | cons A parts ih =>
      simp [incrementsAux, unionParts, ih, sdiff_union_left]

@[simp] lemma unionParts_incrementsAux_empty (parts : List (Finset S)) :
    unionParts (incrementsAux parts (∅ : Finset S)) = unionParts parts := by
  classical
  have h :=
    unionParts_incrementsAux (parts:=parts) (seen:=(∅ : Finset S))
  have hzero : unionParts parts \ (∅ : Finset S) = unionParts parts := by simp
  exact h.trans hzero

end Aux

section WithPlan

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- Incremental “delta” masks: each block removes only the new elements from the
corresponding reroute step. -/
def increments (P : ReroutePlan (S := S)) : List (Finset S) :=
  incrementsAux P.steps ∅

/-- The incremental masks list has the same length as `steps`. -/
lemma increments_length (P : ReroutePlan (S := S)) :
    P.increments.length = P.steps.length := by
  classical
  simp [increments, incrementsAux_length]

lemma increments_pairwise (P : ReroutePlan (S := S)) :
    P.increments.Pairwise (fun A B => Disjoint A B) := by
  classical
  unfold increments
  exact
    incrementsAux_pairwise (parts:=P.steps) (seen:=(∅ : Finset S))

@[simp] lemma unionParts_increments (P : ReroutePlan (S := S)) :
    unionParts P.increments = unionParts P.steps := by
  classical
  have h :=
    unionParts_incrementsAux (parts:=P.steps) (seen:=(∅ : Finset S))
  have hzero : unionParts P.steps \ (∅ : Finset S) = unionParts P.steps := by simp
  unfold increments
  exact h.trans hzero

lemma sum_over_increments (P : ReroutePlan (S := S)) (m : S → NNReal) :
    (unionParts P.steps).sum m =
      P.increments.foldr (fun A acc => A.sum m + acc) 0 := by
  classical
  have hpair := increments_pairwise (P:=P)
  have hsum :=
    sum_unionParts_eq (m:=m) (parts:=P.increments) hpair
  simpa [unionParts_increments (P:=P)] using hsum

end WithPlan

end ReroutePlan

end Increments

section Examples

open ReroutePlan

@[simp] def fin2Plan : ReroutePlan (Fin 2) :=
by
  classical
  refine {
    steps := [{0}, {1}],
    pairwise_disjoint := ?_,
    covers_univ := ?_ }
  · refine List.pairwise_cons.2 ?_
    refine ⟨?_, ?_⟩
    · intro B hB
      have : B = ({1} : Finset (Fin 2)) := by simpa using hB
      subst this
      simp
    · simp
  · ext i
    fin_cases i <;> simp [unionParts]

example :
    (unionParts (fin2Plan.steps)).sum
        (fun i => if i = 0 then (2 : NNReal) else 5) =
      fin2Plan.increments.foldr
        (fun A acc =>
          A.sum (fun i => if i = 0 then (2 : NNReal) else 5) + acc)
        0 := by
  classical
  exact
    (ReroutePlan.sum_over_increments
      (P:=fin2Plan) (m:=fun i => if i = 0 then (2 : NNReal) else 5))

end Examples

end Nfp
