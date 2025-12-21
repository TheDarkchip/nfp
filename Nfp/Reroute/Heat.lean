-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Algebra.Order.Ring.Defs
import Mathlib.Algebra.Field.Defs
import Nfp.Prob
import Nfp.Reroute.Partition

namespace Nfp

namespace List

lemma mem_left_of_zip {α β : Type*} :
    ∀ {xs : List α} {ys : List β} {a : α} {b : β},
      (a, b) ∈ xs.zip ys → a ∈ xs
  | [], _, _, _, h => by cases h
  | _ :: _, [], _, _, h => by cases h
  | x :: xs, y :: ys, a, b, h => by
      have h' : (a, b) = (x, y) ∨ (a, b) ∈ xs.zip ys := by
        simpa [List.zip_cons_cons] using h
      rcases h' with h | h
      · rcases h with ⟨rfl, _⟩
        simp
      · exact List.mem_cons.mpr (Or.inr (mem_left_of_zip h))

lemma get_mem_zip {α β : Type*} :
    ∀ {xs : List α} {ys : List β} {k : Nat}
      (hk : k < xs.length) (hy : k < ys.length),
      (xs.get ⟨k, hk⟩, ys.get ⟨k, hy⟩) ∈ xs.zip ys
  | [], _, k, hk, _ => by
      cases hk
  | _ :: _, [], k, _, hy => by
      cases hy
  | x :: xs, y :: ys, k, hk, hy => by
      cases k with
      | zero =>
          simp [List.zip_cons_cons]
      | succ k =>
          have hk' : k < xs.length := Nat.lt_of_succ_lt_succ hk
          have hy' : k < ys.length := Nat.lt_of_succ_lt_succ hy
          have htail := get_mem_zip (xs:=xs) (ys:=ys) hk' hy'
          have hpair_eq :
              ((x :: xs).get ⟨Nat.succ k, by simpa [List.length_cons] using hk⟩,
                (y :: ys).get ⟨Nat.succ k, by simpa [List.length_cons] using hy⟩)
                = (xs.get ⟨k, hk'⟩, ys.get ⟨k, hy'⟩) := by
            simp
          have htail' :
              ((x :: xs).get ⟨Nat.succ k, by simpa [List.length_cons] using hk⟩,
                (y :: ys).get ⟨Nat.succ k, by simpa [List.length_cons] using hy⟩)
                ∈ xs.zip ys := by
            simpa [hpair_eq] using htail
          have hzip : (x :: xs).zip (y :: ys) = (x, y) :: xs.zip ys := by
            simp [List.zip_cons_cons]
          have hmem' :
              ((x :: xs).get ⟨Nat.succ k, by simpa [List.length_cons] using hk⟩,
                (y :: ys).get ⟨Nat.succ k, by simpa [List.length_cons] using hy⟩)
                ∈ (x, y) :: xs.zip ys := by
            exact List.mem_cons.mpr (Or.inr htail')
          simpa [hzip] using hmem'

end List

/-
Weighted reroute plans and the induced “heat” probability vector (Stage 2).
The structure couples a reroute plan with per-step weights and enforces the
alignment invariants required to distribute each weight over the incremental
masks.  `rerouteHeat` sums the per-block shares, normalizes by the total drop,
and produces a `ProbVec` on `S`.
-/

open scoped BigOperators
open Finset

noncomputable section

section Weighted

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- A reroute plan together with per-step nonnegative weights (logit drops). -/
structure WeightedReroutePlan (S : Type*) [Fintype S] [DecidableEq S] where
  plan : ReroutePlan S
  weights : List NNReal
  length_eq : weights.length = plan.steps.length
  zero_weight_of_empty :
    ∀ {A : Finset S} {w : NNReal},
      (A, w) ∈ plan.increments.zip weights →
      A.card = 0 → w = 0
  weights_sum_pos : 0 < weights.foldr (fun w acc => w + acc) 0

namespace WeightedReroutePlan

variable (P : WeightedReroutePlan (S := S))

/-- Helper: the total step weight (used for normalization). -/
def weightsSum : NNReal :=
  P.weights.foldr (fun w acc => w + acc) 0

@[simp] lemma weightsSum_pos : 0 < P.weightsSum := P.weights_sum_pos

lemma weightsSum_ne_zero : P.weightsSum ≠ 0 :=
  ne_of_gt P.weightsSum_pos

lemma length_eq_increments :
    P.weights.length = P.plan.increments.length := by
  simpa [P.plan.increments_length] using P.length_eq

private def share (A : Finset S) (w : NNReal) (i : S) : NNReal :=
  if A.card = 0 then 0 else if i ∈ A then w / (A.card : NNReal) else 0

lemma sum_share (A : Finset S) (w : NNReal) :
    (∑ i : S, share (S:=S) A w i) = if A.card = 0 then 0 else w := by
  classical
  by_cases hA : A.card = 0
  · simp [share, hA]
  · have hcard : (A.card : NNReal) ≠ 0 := by
      exact_mod_cast hA
    have hshare_zero : ∀ i ∉ A, share (S:=S) A w i = 0 := by
      intro i hi
      simp [share, hA, hi]
    have hsplit :
        (∑ i : S, share (S:=S) A w i)
            = ∑ i ∈ A, share (S:=S) A w i := by
      classical
      let U : Finset S := Finset.univ
      have hdisj : Disjoint A (U \ A) := by
        refine Finset.disjoint_left.mpr ?_
        intro x hxA hxDiff
        exact (Finset.mem_sdiff.mp hxDiff).2 hxA
      have hcover : A ∪ (U \ A) = U := by
        ext x
        by_cases hx : x ∈ A <;> simp [U, hx]
      calc
        (∑ i : S, share (S:=S) A w i)
            = ∑ i ∈ U, share (S:=S) A w i := by simp [U]
        _ = ∑ i ∈ A ∪ (U \ A), share (S:=S) A w i := by
              simp [U, hcover]
        _ = (∑ i ∈ A, share (S:=S) A w i) +
              ∑ i ∈ U \ A, share (S:=S) A w i := by
              simpa using
                (Finset.sum_union hdisj
                  (f := fun i => share (S:=S) A w i))
        _ = (∑ i ∈ A, share (S:=S) A w i) + 0 := by
              refine congrArg (fun t => (∑ i ∈ A, share (S:=S) A w i) + t) ?_
              classical
              refine Finset.sum_eq_zero ?_
              intro i hi
              have hiA : i ∉ A := (Finset.mem_sdiff.mp hi).2
              simpa [U] using hshare_zero i hiA
        _ = ∑ i ∈ A, share (S:=S) A w i := by simp
    have hsum' :
        (∑ i : S, share (S:=S) A w i)
            = (A.card : NNReal) * (w / (A.card : NNReal)) := by
      have hconst :
          (∑ i ∈ A, share (S:=S) A w i)
              = (A.card : NNReal) * (w / (A.card : NNReal)) := by
        classical
        simp [share, hA]
      exact hsplit.trans hconst
    have hratio :
        (A.card : NNReal) * (w / (A.card : NNReal)) = w :=
      mul_div_cancel₀ _ hcard
    have := hsum'.trans hratio
    simpa [hA] using this

lemma sum_share_self (A : Finset S) (w : NNReal) :
    (∑ i ∈ A, share (S:=S) A w i) = if A.card = 0 then 0 else w := by
  classical
  have hshare := sum_share (S:=S) A w
  by_cases hA : A.card = 0
  · have hA' : A = ∅ := Finset.card_eq_zero.mp hA
    simp [share, hA']
  · have hzero :
        (∑ i ∈ ((Finset.univ : Finset S) \ A), share (S:=S) A w i) = 0 := by
        refine Finset.sum_eq_zero ?_
        intro i hi
        have hiA : i ∉ A := (Finset.mem_sdiff.mp hi).2
        simp [share, hiA]
    have hdisj :
        Disjoint A ((Finset.univ : Finset S) \ A) := by
        refine Finset.disjoint_left.mpr ?_
        intro i hiA hiDiff
        exact (Finset.mem_sdiff.mp hiDiff).2 hiA
    have hcover :
        A ∪ ((Finset.univ : Finset S) \ A) = (Finset.univ : Finset S) := by
        ext i
        by_cases hi : i ∈ A <;> simp [hi]
    have hsplit :=
      Finset.sum_union hdisj (f := fun i => share (S:=S) A w i)
    have hx_univ :
        (∑ i : S, share (S:=S) A w i)
            = ∑ i ∈ (Finset.univ : Finset S), share (S:=S) A w i := by
      simp
    have hxA :
        (∑ i ∈ (Finset.univ : Finset S), share (S:=S) A w i)
            = (∑ i ∈ A, share (S:=S) A w i) := by
      simpa [hcover, hzero] using hsplit
    have hx := hx_univ.trans hxA
    have hx' := hx.symm
    simpa using hx'.trans hshare

omit [Fintype S] in
lemma sum_share_of_disjoint {A B : Finset S} (w : NNReal)
    (hdisj : Disjoint A B) :
    (∑ i ∈ A, share (S:=S) B w i) = 0 := by
  classical
  refine Finset.sum_eq_zero ?_
  intro i hi
  have hiB : i ∉ B := by
    intro hmem
    exact (Finset.disjoint_left.mp hdisj) hi hmem
  simp [share, hiB]

private def heatRawAux :
    ∀ (parts : List (Finset S)) (weights : List NNReal),
      weights.length = parts.length → S → NNReal
  | [], [], _, _ => 0
  | A :: parts, w :: weights, hlen, i =>
      let htail : weights.length = parts.length := by
        have hlen' :
            Nat.succ weights.length = Nat.succ parts.length := by
          simpa [List.length_cons] using hlen
        exact Nat.succ.inj hlen'
      share (S:=S) A w i + heatRawAux parts weights htail i
  | _, _, _, _ => 0

private lemma sum_heatRawAux (parts : List (Finset S)) (weights : List NNReal)
    (hlen : weights.length = parts.length)
    (hzero :
      ∀ {A : Finset S} {w : NNReal},
        (A, w) ∈ parts.zip weights → A.card = 0 → w = 0) :
    (∑ i : S, heatRawAux (S:=S) parts weights hlen i)
      = weights.foldr (fun w acc => w + acc) 0 := by
  classical
  revert weights hlen
  induction parts with
  | nil =>
      intro weights hlen hzero
      cases weights with
      | nil =>
          simp [heatRawAux]
      | cons w weights =>
          cases hlen
  | cons A parts ih =>
      intro weights hlen hzero
      cases weights with
      | nil =>
          cases hlen
      | cons w weights =>
          have hlen' :
              weights.length = parts.length := by
            have hlen'' :
                Nat.succ weights.length = Nat.succ parts.length := by
              simpa [List.length_cons] using hlen
            exact Nat.succ.inj hlen''
          have hzero_head :
              A.card = 0 → w = 0 := by
            intro hcard
            have hpair :
                (A, w) ∈ (A :: parts).zip (w :: weights) := by
              simp [List.zip_cons_cons]
            exact hzero hpair hcard
          have hzero_tail :
              ∀ {B : Finset S} {w' : NNReal},
                (B, w') ∈ parts.zip weights → B.card = 0 → w' = 0 := by
            intro B w' hpair hcard
            have hpair' :
                (B, w') ∈ (A :: parts).zip (w :: weights) := by
              have : (B, w') ∈ (A, w) :: parts.zip weights :=
                List.mem_cons.mpr (Or.inr hpair)
              simpa [List.zip_cons_cons] using this
            exact hzero hpair' hcard
          have hsum_tail :=
            ih weights hlen' hzero_tail
          have hsum_head :
              (∑ i : S, share (S:=S) A w i) = w := by
            have hshare := sum_share (S:=S) A w
            by_cases hcard : A.card = 0
            · have hw0 : w = 0 := hzero_head hcard
              have : (∑ i : S, share (S:=S) A w i) = 0 := by
                simp [share, hcard]
              simpa [hshare, hcard, hw0, this]
            · simp [hshare, hcard]
          have hsum_current :
              (∑ i : S, heatRawAux (S:=S) (A :: parts) (w :: weights) hlen i)
                = w + ∑ i : S, heatRawAux (S:=S) parts weights hlen' i := by
            simp [heatRawAux, Finset.sum_add_distrib, hsum_head]
          have hfold :
              (w :: weights).foldr (fun w acc => w + acc) 0
                  = w + weights.foldr (fun w acc => w + acc) 0 := by
            simp
          calc
            (∑ i : S, heatRawAux (S:=S) (A :: parts) (w :: weights) hlen i)
                = w + ∑ i : S, heatRawAux (S:=S) parts weights hlen' i :=
                  hsum_current
            _ = w + weights.foldr (fun w acc => w + acc) 0 := by
                  simp [hsum_tail]
            _ = (w :: weights).foldr (fun w acc => w + acc) 0 :=
                  by simp [hfold]

omit [Fintype S] in
private lemma sum_heatRawAux_disjoint
    (parts : List (Finset S)) (weights : List NNReal)
    (hlen : weights.length = parts.length)
    (hpair : parts.Pairwise (fun A B => Disjoint A B))
    {A : Finset S}
    (hdisj : ∀ B ∈ parts, Disjoint A B) :
    A.sum (fun i => heatRawAux (S:=S) parts weights hlen i) = 0 := by
  classical
  revert weights hlen hpair hdisj
  induction parts generalizing A with
  | nil =>
      intro weights hlen _ _
      cases weights with
      | nil => simp [heatRawAux]
      | cons _ _ => cases hlen
  | cons B parts ih =>
      intro weights hlen hpair hdisj
      cases weights with
      | nil => cases hlen
      | cons w weights =>
          have hlen' : weights.length = parts.length := by
            have hlen'' : Nat.succ weights.length = Nat.succ parts.length := by
              simpa [List.length_cons] using hlen
            exact Nat.succ.inj hlen''
          rcases List.pairwise_cons.mp hpair with ⟨hB, htail⟩
          have hdisj_tail : ∀ C ∈ parts, Disjoint A C := fun C hC => hdisj C (by simp [hC])
          have hdisj_head : Disjoint A B := hdisj B (by simp)
          have hshare_zero :
              A.sum (fun i => share (S:=S) B w i) = 0 :=
            sum_share_of_disjoint (S:=S) (A:=A) (B:=B) (w:=w) hdisj_head
          have htail_zero := ih weights hlen' htail hdisj_tail
          simp [heatRawAux, Finset.sum_add_distrib, hshare_zero, htail_zero]

private lemma sum_heatRawAux_mem_zip
    (parts : List (Finset S)) (weights : List NNReal)
    (hlen : weights.length = parts.length)
    (hpair : parts.Pairwise (fun A B => Disjoint A B))
    (hzero :
      ∀ {B : Finset S} {w : NNReal},
        (B, w) ∈ parts.zip weights → B.card = 0 → w = 0)
    {A : Finset S} {w : NNReal}
    (hmem : (A, w) ∈ parts.zip weights) :
    A.sum (fun i => heatRawAux (S:=S) parts weights hlen i) = w := by
  classical
  revert weights hlen hpair hzero hmem
  induction parts generalizing A w with
  | nil =>
      intro weights hlen _ _ hmem
      cases weights with
      | nil => cases hmem
      | cons _ _ => cases hlen
  | cons B parts ih =>
      intro weights hlen hpair hzero hmem
      cases weights with
      | nil => cases hlen
      | cons z weights =>
          have hlen' : weights.length = parts.length := by
            have hlen'' : Nat.succ weights.length = Nat.succ parts.length := by
              simpa [List.length_cons] using hlen
            exact Nat.succ.inj hlen''
          rcases List.pairwise_cons.mp hpair with ⟨hB, htail⟩
          have hzero_head : B.card = 0 → z = 0 := by
            intro hcard
            have : (B, z) ∈ (B :: parts).zip (z :: weights) := by
              simp [List.zip_cons_cons]
            exact hzero this hcard
          have hzero_tail :
              ∀ {C : Finset S} {w' : NNReal},
                (C, w') ∈ parts.zip weights → C.card = 0 → w' = 0 := by
            intro C w' hC hcard
            have : (C, w') ∈ (B :: parts).zip (z :: weights) := by
              have : (C, w') ∈ (B, z) :: parts.zip weights :=
                List.mem_cons.mpr (Or.inr hC)
              simpa [List.zip_cons_cons] using this
            exact hzero this hcard
          have hmem_cons : (A, w) ∈ (B, z) :: parts.zip weights := by
            simpa [List.zip_cons_cons] using hmem
          rcases List.mem_cons.mp hmem_cons with hhead | htail_mem
          · cases hhead
            have hshare := sum_share_self (S:=S) (A:=B) (w:=w)
            have htail_zero :
                B.sum (fun i => heatRawAux (S:=S) parts weights hlen' i) = 0 := by
              have hdisjB : ∀ C ∈ parts, Disjoint B C := fun C hC => hB C hC
              exact sum_heatRawAux_disjoint (S:=S)
                parts weights hlen' htail hdisjB
            set rest := fun i => heatRawAux (S:=S) parts weights hlen' i
            have hsum_split :
                B.sum (fun i => heatRawAux (S:=S) (B :: parts) (w :: weights) hlen i)
                    = B.sum (fun i => share (S:=S) B w i) + B.sum rest := by
              have := Finset.sum_add_distrib
                (s:=B) (f:=fun i => share (S:=S) B w i) (g:=rest)
              simpa [rest, heatRawAux]
                using this
            by_cases hcard : B.card = 0
            · have hw : w = 0 := hzero_head hcard
              have hshare_zero :
                  B.sum (fun i => share (S:=S) B w i) = 0 := by
                simpa [hcard] using hshare
              have hBempty : B = (∅ : Finset S) := Finset.card_eq_zero.mp hcard
              have hx :
                  B.sum (fun i => share (S:=S) B w i) + B.sum rest = 0 := by
                have hxshare := hshare_zero
                have hxrest := htail_zero
                rw [hxshare, hxrest]
                simp
              have hs :
                  B.sum (fun i => heatRawAux (S:=S) (B :: parts) (w :: weights) hlen i)
                      = 0 := by
                simpa [hsum_split] using hx
              simpa [hw] using hs
            · have hshare_eq :
                  B.sum (fun i => share (S:=S) B w i) = w := by
                simpa [hcard] using hshare
              have hx :
                  B.sum (fun i => share (S:=S) B w i) + B.sum rest = w := by
                have hxrest := htail_zero
                rw [hshare_eq, hxrest]
                simp
              have hs :
                  B.sum (fun i => heatRawAux (S:=S) (B :: parts) (w :: weights) hlen i)
                      = w := by
                simpa [hsum_split] using hx
              exact hs
          · have htail_result :=
              ih weights hlen' htail hzero_tail htail_mem
            have hA_mem : A ∈ parts := List.mem_left_of_zip htail_mem
            have hdisjBA : Disjoint B A := hB A hA_mem
            have hdisjAB : Disjoint A B := by
              refine Finset.disjoint_left.mpr ?_
              intro i hiA hiB
              exact (Finset.disjoint_left.mp hdisjBA) hiB hiA
            have hshare_zero :
                A.sum (fun i => share (S:=S) B z i) = 0 :=
              sum_share_of_disjoint (S:=S) (A:=A) (B:=B) (w:=z) hdisjAB
            set rest := fun i => heatRawAux (S:=S) parts weights hlen' i
            have hsum_split :
                A.sum (fun i => heatRawAux (S:=S) (B :: parts) (z :: weights) hlen i)
                    = A.sum (fun i => share (S:=S) B z i) + A.sum rest := by
              have := Finset.sum_add_distrib
                (s:=A) (f:=fun i => share (S:=S) B z i) (g:=rest)
              simpa [rest, heatRawAux]
                using this
            have hx :
                A.sum (fun i => share (S:=S) B z i) + A.sum rest = w := by
              rw [hshare_zero, htail_result]
              simp
            have hs :
                A.sum (fun i => heatRawAux (S:=S) (B :: parts) (z :: weights) hlen i)
                    = w := by
              simpa [hsum_split] using hx
            exact hs

def heatRaw (i : S) : NNReal :=
  heatRawAux (S:=S) P.plan.increments P.weights
    P.length_eq_increments i

lemma sum_heatRaw :
    (∑ i : S, P.heatRaw i) = P.weightsSum := by
  classical
  have hzero :
      ∀ {A : Finset S} {w : NNReal},
        (A, w) ∈ P.plan.increments.zip P.weights →
        A.card = 0 → w = 0 :=
    fun {A} {w} => P.zero_weight_of_empty
  exact
    sum_heatRawAux (S:=S) P.plan.increments P.weights
      P.length_eq_increments hzero

noncomputable def rerouteHeat : ProbVec S :=
  {
    mass := fun i => P.heatRaw i / P.weightsSum,
    norm_one := by
      classical
      have hdiv :
          (∑ i : S, P.heatRaw i / P.weightsSum)
              = (∑ i : S, P.heatRaw i) / P.weightsSum :=
        (Finset.sum_div (s:=Finset.univ) (f:=fun i => P.heatRaw i)
              (a:=P.weightsSum)).symm
      have hsum := P.sum_heatRaw
      have hne := P.weightsSum_ne_zero
      change (∑ i : S, P.heatRaw i / P.weightsSum) = 1
      simp [hdiv, hsum, hne]
  }

@[simp] lemma rerouteHeat_mass (i : S) :
    (P.rerouteHeat).mass i = P.heatRaw i / P.weightsSum := rfl

lemma heatRaw_sum_increment {A : Finset S} {w : NNReal}
    (hmem : (A, w) ∈ P.plan.increments.zip P.weights) :
    A.sum (fun i => P.heatRaw i) = w := by
  classical
  have hpair := ReroutePlan.increments_pairwise (P:=P.plan)
  have hzero :
      ∀ {B : Finset S} {w' : NNReal},
        (B, w') ∈ P.plan.increments.zip P.weights → B.card = 0 → w' = 0 :=
    fun {B} {w'} => P.zero_weight_of_empty
  simpa [heatRaw] using
    sum_heatRawAux_mem_zip (S:=S)
      P.plan.increments P.weights P.length_eq_increments
      hpair hzero hmem

lemma rerouteHeat_sum_increment {A : Finset S} {w : NNReal}
    (hmem : (A, w) ∈ P.plan.increments.zip P.weights) :
    A.sum (fun i => (P.rerouteHeat).mass i) = w / P.weightsSum := by
  classical
  have hsum := heatRaw_sum_increment (P:=P) hmem
  have hdiv :
      A.sum (fun i => P.heatRaw i / P.weightsSum)
          = (A.sum fun i => P.heatRaw i) / P.weightsSum := by
    simpa using
      (Finset.sum_div (s:=A) (f:=fun i => P.heatRaw i)
          (a:=P.weightsSum)).symm
  simp [rerouteHeat_mass, hdiv, hsum]

end WeightedReroutePlan

end Weighted

end

end Nfp
