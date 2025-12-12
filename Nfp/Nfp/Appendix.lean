import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Indicator
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Order.MinMax
import Mathlib.Data.Finset.Lattice.Basic
import Nfp.Prob
import Nfp.PCC
import Nfp.Uniqueness
import Nfp.Mixer
import Nfp.Influence
import Nfp.Reroute.Partition
import Nfp.Reroute.Heat

namespace Nfp

open scoped BigOperators
open Finset

namespace List

lemma take_succ_drop_append {α : Type*} :
    ∀ {xs : List α} {k : ℕ},
      xs.take (k + 1) = xs.take k ++ (xs.drop k).take 1
  | [], 0 => by simp
  | [], k+1 => by simp
  | _ :: _, 0 => by simp
  | x :: xs, k + 1 => by
      have ih :=
        take_succ_drop_append (xs:=xs) (k:=k)
      simp [ih]

end List

/-!
# Appendix A mapping (Lean formalization)

This file collects the statements used to mirror the Appendix A results:

- Appendix A.4 (PCC curve): `normMass`, `feasible`, `pccArg`, `pccMax`,
  `PCC`, monotonicity (`PCC_monotone`), and the dominance/upper-bound lemmas
  (`pccMax_dominates`, `PCC_upper_bounds_masks`), plus the existence/greedy view
  `greedy_topmass_optimal`.
- Appendix A.3 (Residual rule): `lambdaEC`, `lambdaEC_sum_one`,
  `residual_lambda_from_norm`, and the global characterization
  `lambdaEC_scale_invariant_global`.
- Appendix A.2 (Normalization on a subset): `normalizeOn` and its properties
  (`normalizeOn_outside`, `normalizeOn_inside`, `normalizeOn_sum_one`), and the
  uniqueness theorem `proportional_row_unique`.

All proofs are constructive in mathlib and avoid `sorry`.
-/

section PCC

variable {S : Type*} [Fintype S]

variable (m : Nfp.Contrib S)

lemma drop_eq_mass (A : Finset S) :
  (A.sum m) / (∑ i, m i) = (A.sum (fun i => m i / (∑ j, m j))) := by
  classical
  -- Division distributes over finite sums in `NNReal`.
  simp [Finset.sum_div]

/-- Appendix A.4 (upper bound): any mask with tracer budget ≤ τ removes at most τ
of normalized logit. -/
lemma pcc_upper_bound_for_any_mask (A : Finset S) (τ : NNReal)
  (hA : (A.sum (fun i => m i / (∑ j, m j))) ≤ τ) :
  (A.sum m) / (∑ j, m j) ≤ τ := by
  simpa [drop_eq_mass (m:=m) (A:=A)] using hA

/-- Appendix A.4 (monotonicity helper): normalized removed mass is monotone along
any nested mask chain. -/
lemma normalized_sum_monotone (A : ℕ → Finset S)
  (hchain : ∀ k, A k ⊆ A (k + 1)) :
  Monotone (fun k => (A k).sum (fun i => m i / (∑ j, m j))) := by
  classical
  -- Apply monotonicity of raw sums to the pointwise scaled weights.
  have := Nfp.sum_monotone_chain (A:=A) (w:=fun i => m i / (∑ j, m j)) hchain
  simpa using this

/-- Appendix A.4: normalized mass of a mask `A`. -/
noncomputable def normMass (m : S → NNReal) (A : Finset S) : NNReal :=
  (A.sum m) / (∑ j, m j)

@[simp] lemma normMass_empty (m : S → NNReal) : normMass m (∅ : Finset S) = 0 := by
  simp [normMass]

lemma normMass_union [DecidableEq S] (m : S → NNReal) (A B : Finset S)
    (hdisj : Disjoint A B) :
    normMass m (A ∪ B) = normMass m A + normMass m B := by
  classical
  have hsum := Finset.sum_union hdisj (f := m)
  simp [normMass, hsum, add_div]

lemma normMass_partitionUnion [DecidableEq S] (m : S → NNReal)
    (parts : List (Finset S))
    (hpair : parts.Pairwise (fun A B => Disjoint A B)) :
    normMass m (unionParts parts) =
      parts.foldr (fun A acc => normMass m A + acc) 0 := by
  classical
  induction parts with
  | nil =>
      simp [unionParts]
  | cons A parts ih =>
      rcases List.pairwise_cons.mp hpair with ⟨hA, htail⟩
      have hdisj : Disjoint A (unionParts parts) :=
        disjoint_unionParts A parts (by
          intro B hB
          exact hA _ (by simpa using hB))
      have hAdd :=
        normMass_union (m:=m) A (unionParts parts) hdisj
      simp [unionParts, hAdd, ih htail, List.foldr]

lemma normMass_partition_eq_one [DecidableEq S] (m : S → NNReal)
    (parts : List (Finset S))
    (hpair : parts.Pairwise (fun A B => Disjoint A B))
    (hcover : unionParts parts = (Finset.univ : Finset S))
    (htot : (∑ j, m j) ≠ 0) :
    parts.foldr (fun A acc => normMass m A + acc) 0 = 1 := by
  classical
  have hsum :
      parts.foldr (fun A acc => normMass m A + acc) 0 = normMass m (unionParts parts) := by
    simpa using (normMass_partitionUnion (m:=m) parts hpair).symm
  simpa [hcover, normMass, htot] using hsum

namespace ReroutePlan

section

variable {S : Type*} [Fintype S] [DecidableEq S]

lemma normMass_sum_one (m : S → NNReal) (P : ReroutePlan (S := S))
    (htot : (∑ j, m j) ≠ 0) :
    P.masks.foldr (fun A acc => normMass m A + acc) 0 = 1 := by
  simpa [ReroutePlan.masks] using
    normMass_partition_eq_one (S := S) m P.masks P.pairwise_disjoint P.covers_univ htot

end

end ReroutePlan

lemma normMass_rerouteHeat_increment {S : Type*} [Fintype S] [DecidableEq S]
    (P : WeightedReroutePlan (S := S)) {A : Finset S} {w : NNReal}
    (hmem : (A, w) ∈ P.plan.increments.zip P.weights) :
    normMass (fun i => (P.rerouteHeat).mass i) A = w / P.weightsSum := by
  classical
  have hsum := WeightedReroutePlan.rerouteHeat_sum_increment (P:=P) hmem
  have hdenom : (∑ i, (P.rerouteHeat).mass i) = 1 := ProbVec.sum_mass _
  have hsum' :
      (∑ i ∈ A, P.heatRaw i / P.weightsSum) = w / P.weightsSum := by
    simpa [WeightedReroutePlan.rerouteHeat_mass] using hsum
  have hdenom' :
      (∑ i, P.heatRaw i / P.weightsSum) = 1 := by
    simpa [WeightedReroutePlan.rerouteHeat_mass] using hdenom
  simp [normMass, hsum', hdenom']


/-- Appendix A.4: feasible masks under budget `τ` (by normalized mass). -/
noncomputable def feasible (m : S → NNReal) (τ : NNReal) : Finset (Finset S) :=
  (Finset.univ : Finset S).powerset.filter (fun A => normMass m A ≤ τ)

lemma feasible_nonempty (m : S → NNReal) (τ : NNReal) :
  (feasible (S:=S) m τ).Nonempty := by
  classical
  refine ⟨∅, ?_⟩
  simp [feasible, normMass]

-- Appendix A.4: argmax mask under budget τ w.r.t. normalized mass.
noncomputable def pccArg (m : S → NNReal) (τ : NNReal) : Finset S := by
  classical
  exact Classical.choose
    (Finset.exists_max_image (feasible (S:=S) m τ) (fun A => normMass m A)
      (feasible_nonempty (S:=S) m τ))

private lemma pccArg_mem (m : S → NNReal) (τ : NNReal) :
  pccArg (S:=S) m τ ∈ feasible (S:=S) m τ := by
  classical
  have h := Classical.choose_spec
    (Finset.exists_max_image (feasible (S:=S) m τ) (fun A => normMass m A)
      (feasible_nonempty (S:=S) m τ))
  exact h.left

private lemma pccArg_is_max (m : S → NNReal) (τ : NNReal) :
  ∀ B ∈ feasible (S:=S) m τ, normMass m B ≤ normMass m (pccArg (S:=S) m τ) := by
  classical
  have h := Classical.choose_spec
    (Finset.exists_max_image (feasible (S:=S) m τ) (fun A => normMass m A)
      (feasible_nonempty (S:=S) m τ))
  exact h.right

/-- Appendix A.4: maximal normalized drop (the PCC value at budget `τ`). -/
noncomputable def pccMax (m : S → NNReal) (τ : NNReal) : NNReal :=
  normMass m (pccArg (S:=S) m τ)

lemma pccMax_le_tau (m : S → NNReal) (τ : NNReal) :
  pccMax (S:=S) m τ ≤ τ := by
  classical
  have hmem := pccArg_mem (S:=S) m τ
  simpa [pccMax] using (Finset.mem_filter.mp hmem).2

lemma pccMax_monotone (m : S → NNReal) :
  Monotone (fun τ : NNReal => pccMax (S:=S) m τ) := by
  classical
  intro τ₁ τ₂ hle
  -- Any τ₁-feasible set is τ₂-feasible when τ₁ ≤ τ₂.
  have hsubset : feasible (S:=S) m τ₁ ⊆ feasible (S:=S) m τ₂ := by
    intro A hA
    rcases Finset.mem_filter.mp hA with ⟨hpow, hbudget⟩
    have hbudget' : normMass m A ≤ τ₂ := hbudget.trans hle
    aesop (add simp [feasible])
  -- The τ₂-argmax is ≥ any τ₁-feasible value, in particular the τ₁-argmax.
  have hAτ1 := pccArg_mem (S:=S) m τ₁
  have hAτ1_in : pccArg (S:=S) m τ₁ ∈ feasible (S:=S) m τ₂ := hsubset hAτ1
  have hmaxτ2 := pccArg_is_max (S:=S) m τ₂ (pccArg (S:=S) m τ₁) hAτ1_in
  simpa [pccMax] using hmaxτ2


/-- Appendix A.4 (dominance/upper bound): for any mask `B` that removes at most
`τ` of tracer mass (normalized), its normalized drop is upper-bounded by `pccMax m τ`.
Equivalently, `pccMax` is the supremum over all feasible masks at budget `τ`. -/
lemma pccMax_dominates (m : S → NNReal) (τ : NNReal) (B : Finset S)
  (hB : normMass m B ≤ τ) :
  normMass m B ≤ pccMax (S:=S) m τ := by
  classical
  -- Feasibility of B at budget τ
  have hB_mem : B ∈ feasible (S:=S) m τ := by
    have hpow : B ∈ (Finset.univ : Finset S).powerset := by
      simp
    exact (Finset.mem_filter.mpr ⟨hpow, hB⟩)
  -- Maximality of the arg at τ
  have hmax := pccArg_is_max (S:=S) m τ B hB_mem
  simpa [pccMax] using hmax

/-- Appendix A.4 (PCC curve): normalized logit drop achievable at tracer-mass
budget `t`. This matches the budgeted maximum `pccMax` over feasible masks. -/
noncomputable abbrev PCC (m : S → NNReal) (t : NNReal) : NNReal := pccMax (S:=S) m t

@[simp] lemma PCC_def (m : S → NNReal) (t : NNReal) : PCC (S:=S) m t = pccMax (S:=S) m t := rfl

/-- Appendix A.4 (monotonicity): the PCC curve is monotone in the budget `t`. -/
lemma PCC_monotone (m : S → NNReal) : Monotone (fun t : NNReal => PCC (S:=S) m t) := by
  simpa [PCC_def] using pccMax_monotone (S:=S) m

/-- Appendix A.4 (upper bound, mask phrasing): for any mask `B` that
preserves at least `1 - t` tracer mass (i.e., removes ≤ `t`), the normalized
logit drop of `B` is at most `PCC m t`. -/
lemma PCC_upper_bounds_masks (m : S → NNReal) (t : NNReal) (B : Finset S)
  (hBudget : normMass m B ≤ t) :
  normMass m B ≤ PCC (S:=S) m t := by
  simpa [PCC_def] using pccMax_dominates (S:=S) m t B hBudget

lemma rerouteHeat_pcc_drop {S : Type*} [Fintype S] [DecidableEq S]
    (P : WeightedReroutePlan (S := S))
    {A : Finset S} {w : NNReal}
    (hmem : (A, w) ∈ P.plan.increments.zip P.weights) :
    PCC (S:=S) (fun i => (P.rerouteHeat).mass i) (w / P.weightsSum) = w / P.weightsSum := by
  classical
  set m := fun i => (P.rerouteHeat).mass i
  have hnorm' : normMass m A = w / P.weightsSum := by
    simpa [m] using normMass_rerouteHeat_increment (P:=P) hmem
  have hfeas : normMass m A ≤ w / P.weightsSum := by
    simp [hnorm']
  have hdom := pccMax_dominates (S:=S) m (w / P.weightsSum) A hfeas
  have hupper : PCC (S:=S) m (w / P.weightsSum) ≤ w / P.weightsSum := by
    simpa [PCC_def] using pccMax_le_tau (S:=S) m (w / P.weightsSum)
  have hdom' : w / P.weightsSum ≤ pccMax m (w / P.weightsSum) := by
    simpa [hnorm'] using hdom
  have hlower : w / P.weightsSum ≤ PCC (S:=S) m (w / P.weightsSum) := by
    simpa [PCC_def] using hdom'
  exact le_antisymm hupper hlower

/-- Convenience corollary: the `k`-th reroute increment (mask/weight pair)
from a weighted plan achieves equality with the PCC budget that matches its
normalized weight.  This version avoids spelunking through `zip` membership
and is tailored for the exported segment lists used in the Python notebooks. -/
lemma rerouteHeat_pcc_drop_index {S : Type*} [Fintype S] [DecidableEq S]
    (P : WeightedReroutePlan (S := S)) {k : Nat}
    (hk : k < P.plan.increments.length) :
    PCC (S:=S) (fun i => (P.rerouteHeat).mass i)
        (P.weights.get ⟨k, by simpa [P.length_eq_increments] using hk⟩ / P.weightsSum)
        = P.weights.get ⟨k, by simpa [P.length_eq_increments] using hk⟩ / P.weightsSum := by
  classical
  have hlen : P.plan.increments.length = P.weights.length := (P.length_eq_increments).symm
  have hw : k < P.weights.length := hlen ▸ hk
  have hmem :
      (P.plan.increments.get ⟨k, hk⟩,
        P.weights.get ⟨k, hw⟩) ∈ P.plan.increments.zip P.weights := by
    simpa [hw, hlen] using
      List.get_mem_zip (xs:=P.plan.increments) (ys:=P.weights)
        (k:=k) hk hw
  have h := rerouteHeat_pcc_drop (P:=P) (hmem := hmem)
  exact h

namespace WeightedReroutePlan

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- Stage-4 prefix objective: restrict the AUC evaluation to the first `k`
intervals induced by `weights`. -/
noncomputable def rerouteHeatObjectivePrefix
    (P : WeightedReroutePlan (S := S)) (k : ℕ) : NNReal :=
  PCC.evalFromWeights
    (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
    P.weightsSum (P.weights.take k)

/-- Objective used in Stage 4: the PCC AUC achieved by the reroute plan’s
`rerouteHeat` distribution, evaluated via the generic weights-to-interval map. -/
noncomputable def rerouteHeatObjective (P : WeightedReroutePlan (S := S)) : NNReal :=
  PCC.evalFromWeights
    (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
    P.weightsSum P.weights

lemma rerouteHeatObjectivePrefix_as_auc (P : WeightedReroutePlan (S := S)) (k : ℕ) :
    PCC.AUC
        (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
        ((P.aucIntervals).take k)
        = P.rerouteHeatObjectivePrefix k := by
  classical
  simpa [rerouteHeatObjectivePrefix, aucIntervals] using
    PCC.AUC_intervalsFromWeights_take
      (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
      P.weightsSum P.weights k

lemma rerouteHeatObjective_as_auc (P : WeightedReroutePlan (S := S)) :
    PCC.AUC
        (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
        (P.aucIntervals)
        = P.rerouteHeatObjective := by
  simp [rerouteHeatObjective, auc_eval]

lemma rerouteHeatObjectivePrefix_le (P : WeightedReroutePlan (S := S)) (k : ℕ) :
    P.rerouteHeatObjectivePrefix k ≤ P.rerouteHeatObjective := by
  classical
  have hmono :=
    PCC.AUC_monotone_append
      (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
      (P.aucIntervals.take k) (P.aucIntervals.drop k)
  have hsplit := List.take_append_drop k P.aucIntervals
  have hle :
      PCC.AUC
          (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
          (P.aucIntervals.take k)
        ≤
          PCC.AUC
            (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
            (P.aucIntervals) := by
    simpa [hsplit] using hmono
  have hprefix := (rerouteHeatObjectivePrefix_as_auc (P:=P) (k:=k)).symm
  have htotal := (rerouteHeatObjective_as_auc (P:=P)).symm
  simpa [hprefix, htotal] using hle

lemma rerouteHeatObjective_split (P : WeightedReroutePlan (S := S)) (k : ℕ) :
    P.rerouteHeatObjective =
      P.rerouteHeatObjectivePrefix k +
        PCC.AUC
          (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
          (P.aucIntervals.drop k) := by
  classical
  have h :=
    PCC.AUC_append
      (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
      (P.aucIntervals.take k) (P.aucIntervals.drop k)
  have hsplit := List.take_append_drop k P.aucIntervals
  have hsum :
      PCC.AUC
          (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
          (P.aucIntervals)
        =
          PCC.AUC
            (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
            (P.aucIntervals.take k) +
          PCC.AUC
            (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
            (P.aucIntervals.drop k) := by
    simpa [hsplit] using h
  have hprefix := (rerouteHeatObjectivePrefix_as_auc (P:=P) (k:=k)).symm
  have htotal := (rerouteHeatObjective_as_auc (P:=P)).symm
  simpa [hprefix, htotal] using hsum

/-- Prefix addition lemma: the `(k+1)`-st prefix equals the `k`-prefix plus the
contribution coming from the `(k+1)`-st interval (if present). -/
lemma rerouteHeatObjectivePrefix_succ (P : WeightedReroutePlan (S := S)) (k : ℕ) :
    P.rerouteHeatObjectivePrefix (k + 1) =
      P.rerouteHeatObjectivePrefix k +
        PCC.AUC
          (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
          ((P.aucIntervals.drop k).take 1) := by
  classical
  have hsum :
      PCC.AUC
          (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
          (P.aucIntervals.take (k + 1))
        =
          PCC.AUC
              (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
              (P.aucIntervals.take k) +
            PCC.AUC
              (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
              ((P.aucIntervals.drop k).take 1) := by
    simpa [List.take_succ_drop_append (xs:=P.aucIntervals) (k:=k)] using
      PCC.AUC_append
        (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
        (P.aucIntervals.take k)
        ((P.aucIntervals.drop k).take 1)
  have hprefix := rerouteHeatObjectivePrefix_as_auc (P:=P) (k:=k)
  have hprefix_succ := rerouteHeatObjectivePrefix_as_auc (P:=P) (k:=k+1)
  have hsum' := hsum
  rw [hprefix_succ] at hsum'
  have hsum'' := hsum'
  rw [hprefix] at hsum''
  exact hsum''

lemma rerouteHeatObjectivePrefix_succ_le (P : WeightedReroutePlan (S := S)) (k : ℕ) :
    P.rerouteHeatObjectivePrefix k ≤ P.rerouteHeatObjectivePrefix (k + 1) := by
  classical
  have h := rerouteHeatObjectivePrefix_succ (P:=P) (k:=k)
  have hnonneg :
      0 ≤
        PCC.AUC
          (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
          ((P.aucIntervals.drop k).take 1) :=
    PCC.AUC_nonneg
      (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
      ((P.aucIntervals.drop k).take 1)
  have hle :
      P.rerouteHeatObjectivePrefix k ≤
        P.rerouteHeatObjectivePrefix k +
          PCC.AUC
            (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
            ((P.aucIntervals.drop k).take 1) :=
    le_add_of_nonneg_right hnonneg
  calc
    P.rerouteHeatObjectivePrefix k
        ≤ P.rerouteHeatObjectivePrefix k +
            PCC.AUC
              (f:=fun t => PCC (S:=S) (fun i => (P.rerouteHeat).mass i) t)
              ((P.aucIntervals.drop k).take 1) := hle
    _ = P.rerouteHeatObjectivePrefix (k + 1) := by
      simp [h]

lemma rerouteHeatObjectivePrefix_mono (P : WeightedReroutePlan (S := S))
    {k₁ k₂ : ℕ} (hle : k₁ ≤ k₂) :
    P.rerouteHeatObjectivePrefix k₁ ≤ P.rerouteHeatObjectivePrefix k₂ := by
  classical
  have hmono :
      ∀ d k, P.rerouteHeatObjectivePrefix k ≤ P.rerouteHeatObjectivePrefix (k + d) := by
    intro d
    induction d with
    | zero =>
        intro k
        simp
    | succ d ih =>
        intro k
        have hsucc :=
          rerouteHeatObjectivePrefix_succ_le
            (P:=P) (k:=k + d)
        have := ih k
        have := this.trans hsucc
        simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using this
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hle
  simpa using hmono d k₁

/-- Feasibility predicate for Stage 4: each incremental mask carries the exact
normalized mass dictated by its weight ratio. -/
def rerouteFeasible (m : S → NNReal) (P : WeightedReroutePlan (S := S)) : Prop :=
  ∀ {A : Finset S} {w : NNReal},
    (A, w) ∈ P.plan.increments.zip P.weights →
      normMass m A = w / P.weightsSum

lemma rerouteHeat_feasible (P : WeightedReroutePlan (S := S)) :
    rerouteFeasible (S:=S) (fun i => (P.rerouteHeat).mass i) P := by
  intro A w hmem
  simpa using normMass_rerouteHeat_increment (P:=P) (hmem:=hmem)

/-- Stage-4 goal statement (stub): a “delta-weighted” optimal plan maximizes
`rerouteHeatObjective` among feasible plans.  Proof to be supplied once the
majorization lemmas are in place. -/
def optimal_delta_weighting_statement (m : S → NNReal) : Prop :=
  ∃ P : WeightedReroutePlan (S := S),
    rerouteFeasible (S:=S) m P ∧
    ∀ Q : WeightedReroutePlan (S := S),
      rerouteFeasible (S:=S) m Q →
        Q.rerouteHeatObjective ≤ P.rerouteHeatObjective

end WeightedReroutePlan


/-- Appendix A.4 (greedy optimality for mass budgets): there exists a
subset `A` (a feasible mask under budget `τ`) that maximizes normalized
removed mass among all feasible masks. Concretely we can take `A = pccArg m τ`.
This ties the constructive “pick a best feasible mask” view to our
`pccArg/pccMax` definitions; ties are handled by the argmax choice. -/
lemma greedy_topmass_optimal (m : S → NNReal) (τ : NNReal) :
  ∃ A : Finset S,
    A ∈ feasible (S:=S) m τ ∧
    (∀ B : Finset S, B ∈ feasible (S:=S) m τ →
      normMass m B ≤ normMass m A) := by
  classical
  refine ⟨pccArg (S:=S) m τ, ?_, ?_⟩
  · exact pccArg_mem (S:=S) m τ
  · intro B hB
    exact pccArg_is_max (S:=S) m τ B hB


/-- Greedy top-k optimality (helper for A.4): among all subsets with cardinality ≤ k,
there exists
an optimal set A maximizing `A.sum m` such that no outside element has larger weight
than an inside element (swap-optimality). This captures the compact majorization-style
“top-k is optimal” property without constructing an explicit sort. -/
lemma greedy_topk_optimal (m : S → NNReal) [DecidableEq S] (k : ℕ) :
  ∃ A : Finset S,
    A.card ≤ k ∧
    (∀ B : Finset S, B.card ≤ k → B.sum m ≤ A.sum m) ∧
    (∀ {i j}, i ∈ A → j ∉ A → m i ≥ m j) := by
  classical
  -- Candidates: all subsets of `univ` with card ≤ k.
  let C : Finset (Finset S) := (Finset.univ : Finset S).powerset.filter (fun A => A.card ≤ k)
  have hC_nonempty : C.Nonempty := by
    refine ⟨∅, ?_⟩
    simp [C]
  -- Pick A ∈ C maximizing the sum.
  obtain ⟨A, hA_mem, hAmax⟩ := Finset.exists_max_image C (fun A : Finset S => A.sum m) hC_nonempty
  -- Basic facts about A
  have hA_card_le : A.card ≤ k := by
    rcases Finset.mem_filter.mp hA_mem with ⟨hA_ps, hk⟩
    exact hk
  -- Show optimality among all B with card ≤ k.
  have hA_opt : ∀ B : Finset S, B.card ≤ k → B.sum m ≤ A.sum m := by
    intro B hBk
    have hB_mem : B ∈ C := by
      -- Any B ⊆ univ is in the powerset, and `B.card ≤ k` ensures it passes the filter.
      simp [C, hBk]
    exact hAmax B hB_mem
  -- Swap optimality: no beneficial swap exists.
  have hswap : ∀ {i j}, i ∈ A → j ∉ A → m i ≥ m j := by
    intro i j hiA hjA
    by_contra hij
    have hij' : m i < m j := lt_of_not_ge hij
    -- Construct the swapped set B = insert j (A.erase i)
    let B := insert j (A.erase i)
    have hj_not : j ∉ A.erase i := by
      simp [Finset.mem_erase, hjA]
    have hcard_eq : B.card = A.card := by
      have hIns : insert i (A.erase i) = A := by simpa using Finset.insert_erase hiA
      have hi_not : i ∉ A.erase i := by simp
      have hA_card' : (A.erase i).card + 1 = A.card := by
        -- From card(insert i (erase i A)) = card(erase i A) + 1
        calc
          (A.erase i).card + 1 = (insert i (A.erase i)).card := by
            simp [Finset.card_insert_of_notMem hi_not]
          _ = A.card := by simp [hIns]
      have hB_card : B.card = (A.erase i).card + 1 := by
        simp [B, Finset.card_insert_of_notMem hj_not]
      simpa [hB_card] using hA_card'
    have hB_card_le : B.card ≤ k := by simpa [hcard_eq] using hA_card_le
    -- Compare sums via erase/insert decomposition
    have hsumA : A.sum m = m i + (A.erase i).sum m := by
      have hi_not : i ∉ A.erase i := by simp
      have hIns : insert i (A.erase i) = A := by simpa using Finset.insert_erase hiA
      have sumIns := Finset.sum_insert (s:=A.erase i) (a:=i) (f:=m) hi_not
      simpa [hIns, add_comm] using sumIns
    have hsumB : B.sum m = m j + (A.erase i).sum m := by
      have sumInsB := Finset.sum_insert (s:=A.erase i) (a:=j) (f:=m) hj_not
      simpa [B, add_comm] using sumInsB
    have : A.sum m < B.sum m := by
      have h := add_lt_add_right hij' ((A.erase i).sum m)
      simpa [hsumA, hsumB, add_comm, add_left_comm, add_assoc] using h
    have hB_le := hA_opt B hB_card_le
    exact (not_le_of_gt this) hB_le
  exact ⟨A, hA_card_le, hA_opt, hswap⟩

end PCC

/-! Residual additions: energy-consistent coefficients -/

section Residual

/-- Energy-consistent residual coefficient (Appendix A.3). -/
noncomputable def lambdaEC (x a : NNReal) : NNReal := x / (x + a)

lemma lambdaEC_sum_one (x a : NNReal) (hx : x + a ≠ 0) :
  lambdaEC x a + a / (x + a) = 1 := by
  unfold lambdaEC
  have : x / (x + a) + a / (x + a) = (x + a) / (x + a) := by
    simp [add_div]
  simp [this, div_self hx]

/-- If a residual coefficient `λ` satisfies `λ + a/(x+a) = 1` (with `x+a ≠ 0`),
then necessarily `λ = x/(x+a)` in `NNReal`. -/
lemma residual_lambda_from_norm
  (lcoeff : NNReal → NNReal → NNReal)
  (x a : NNReal) (hx : x + a ≠ 0)
  (hnorm : lcoeff x a + a / (x + a) = 1) :
  lcoeff x a = x / (x + a) := by
  -- Work in ℝ via coercions to use field operations.
  have hR : (lcoeff x a : ℝ) + (a : ℝ) / (x + a) = 1 := by
    simpa using congrArg (fun t : NNReal => (t : ℝ)) hnorm
  have hxneR : (x + a : ℝ) ≠ 0 := by exact_mod_cast hx
  have hval : (lcoeff x a : ℝ) = x / (x + a) := by
    -- From hR: l + a/(x+a) = 1 ⇒ l = 1 - a/(x+a) = x/(x+a)
    have : (lcoeff x a : ℝ) = 1 - (a : ℝ) / (x + a) := by
      exact (eq_sub_iff_add_eq).mpr hR
    have hunit : (1 : ℝ) = (x + a) / (x + a) := by simp [div_self hxneR]
    have hdiff : (x + a : ℝ) / (x + a) - (a : ℝ) / (x + a) = ((x + a : ℝ) - a) / (x + a) := by
      simpa [sub_eq_add_neg] using (sub_div ((x + a : ℝ)) (a : ℝ) (x + a)).symm
    have : (lcoeff x a : ℝ) = ((x + a : ℝ) - a) / (x + a) := by simp [this, hunit, hdiff]
    simpa [sub_eq_add_neg, add_comm]
  -- Conclude equality in NNReal by extensionality on coercions to ℝ.
  apply Subtype.ext
  simpa using hval

/-- Global characterization: any residual rule which is pointwise row-normalized
(`lcoeff x a + a/(x+a) = 1` whenever `x+a ≠ 0`) must be `x/(x+a)` (Appendix A.3).
The Appendix mentions scale-invariance for motivation, but the normalization
equation alone determines the coefficients, so we keep the statement minimal. -/
lemma lambdaEC_scale_invariant_global
  (lcoeff : NNReal → NNReal → NNReal)
  (hnorm : ∀ (x a : NNReal), x + a ≠ 0 → lcoeff x a + a / (x + a) = 1) :
  ∀ (x a : NNReal), x + a ≠ 0 → lcoeff x a = x / (x + a) := by
  intro x a hx
  exact residual_lambda_from_norm lcoeff x a hx (hnorm x a hx)

end Residual

/-! Normalization uniqueness for local mixers -/

section Normalize

variable {S : Type*} [DecidableEq S]

/-- Appendix A.2: normalize nonnegative weights on a finite subset `A` to a probability row. -/
noncomputable def normalizeOn (A : Finset S) (w : S → NNReal) : S → NNReal :=
  fun i => if i ∈ A then w i / (A.sum w) else 0

/-- Appendix A.2: outside the support `A`, `normalizeOn A w` is zero. -/
@[simp] lemma normalizeOn_outside (A : Finset S) (w : S → NNReal) {i : S} (hi : i ∉ A) :
  normalizeOn (S:=S) A w i = 0 := by
  classical
  simp [normalizeOn, hi]

/-- Appendix A.2: inside the support `A`, `normalizeOn A w` is proportional to `w`. -/
@[simp] lemma normalizeOn_inside (A : Finset S) (w : S → NNReal) {i : S} (hi : i ∈ A) :
  normalizeOn (S:=S) A w i = w i / (A.sum w) := by
  classical
  simp [normalizeOn, hi]

/-- Appendix A.2: `normalizeOn A w` is a probability row (sums to 1). -/
lemma normalizeOn_sum_one [Fintype S] (A : Finset S) (w : S → NNReal) (h : A.sum w ≠ 0) :
  (∑ i, normalizeOn (S:=S) A w i) = 1 := by
  classical
  -- Sum over `univ` of an indicator-style function equals the sum over `A`.
  have hsumA :=
    (Finset.sum_indicator_subset (s:=A) (t:=(Finset.univ : Finset S))
      (f:=fun i => w i / (A.sum w)) (Finset.subset_univ A))
  -- simplify the indicator
  have hsumA' : (∑ i, normalizeOn (S:=S) A w i) = A.sum (fun i => w i / (A.sum w)) := by
    convert hsumA using 1
    simp [normalizeOn, Set.indicator]
  have hsumA'' : A.sum (fun i => w i / (A.sum w)) = (A.sum w) / (A.sum w) := by
    simp [Finset.sum_div]
  have hsumA1 : A.sum (fun i => w i / (A.sum w)) = 1 := by
    simpa [div_self h] using hsumA''
  simpa [hsumA'] using hsumA1

-- (To be extended) A local uniqueness lemma will connect proportional rows supported on `A`
-- and the `normalizeOn` construction via the budget/sum constraint.

/-- Appendix A.2 (uniqueness): if `p` is supported on `A`,
and proportional to `w` on `A` with total mass 1, then `p` is exactly `normalizeOn A w`.
We also derive the proportionality constant as `1 / A.sum w` using the mass constraint. -/
lemma proportional_row_unique [Fintype S]
  (A : Finset S) (w p : S → NNReal) (k : NNReal)
  (hAw : A.sum w ≠ 0)
  (hout : ∀ i ∉ A, p i = 0)
  (hin : ∀ i ∈ A, p i = k * w i)
  (hsum : (∑ i, p i) = 1) :
  p = normalizeOn (S:=S) A w := by
  classical
  -- Represent `p` as an indicator-style function using `hin`/`hout`.
  have hrepr : (fun i => if i ∈ A then k * w i else 0) = p := by
    funext i
    by_cases hi : i ∈ A
    · simp [hi, hin i hi]
    · simp [hi, hout i hi]
  -- The total mass constraint identifies `k`.
  -- Sum of indicator over `univ` reduces to a sum over `A`.
  have hsumA0 :=
    (Finset.sum_indicator_subset (s:=A) (t:=(Finset.univ : Finset S))
      (f:=fun i => k * w i) (Finset.subset_univ A))
  have hsumA0' :
      (∑ i, ((↑A : Set S).indicator (fun i => k * w i) i : NNReal))
        = A.sum (fun i => k * w i) := by
    simpa using hsumA0
  -- Identify the indicator function with the `if-then-else` representation and then with `p`.
  have hind : (fun i => ((↑A : Set S).indicator (fun i => k * w i) i : NNReal))
      = (fun i => if i ∈ A then k * w i else 0) := by
    funext i; simp [Set.indicator]
  have hfun : (fun i => ((↑A : Set S).indicator (fun i => k * w i) i : NNReal)) = p := by
    simpa [hrepr] using hind
  have hsumA : (∑ i, p i) = A.sum (fun i => k * w i) := by
    have : (∑ i, p i) = (∑ i, ((↑A : Set S).indicator (fun i => k * w i) i : NNReal)) := by
      simp [hfun]
    exact this.trans hsumA0'
  have hsumA1 : A.sum (fun i => k * w i) = 1 := by simpa [hsumA] using hsum
  -- Move to ℝ to use field-style algebra and cancel the nonzero sum.
  have hsumA1R : (A.sum (fun i => ((k : ℝ) * (w i : ℝ)))) = 1 := by
    simpa using congrArg (fun t : NNReal => (t : ℝ)) hsumA1
  have hR : (k : ℝ) * (((A.sum w : NNReal) : ℝ)) = 1 := by
    -- factor out the constant from the sum and identify the coerced sum
    simpa [Finset.mul_sum] using hsumA1R
  have hAwR : (((A.sum w : NNReal) : ℝ)) ≠ 0 := by exact_mod_cast hAw
  have k_eq : (k : ℝ) = 1 / (((A.sum w : NNReal) : ℝ)) := by
    -- from k * S = 1
    exact (eq_div_iff_mul_eq hAwR).2 (by simpa [mul_comm] using hR)
  -- Conclude pointwise equality with normalizeOn.
  apply funext; intro i; by_cases hiA : i ∈ A
  · -- inside A: p i = k * w i = (1 / sum) * w i = w i / sum
    have : p i = k * w i := hin i hiA
    have : (p i : ℝ) = (k : ℝ) * (w i : ℝ) := by
      simpa using congrArg (fun t : NNReal => (t : ℝ)) this
    have : (p i : ℝ) = (1 / (((A.sum w : NNReal) : ℝ))) * (w i : ℝ) := by
      simpa [k_eq, mul_comm, mul_left_comm, mul_assoc]
        using this
    -- convert back to NNReal
    apply Subtype.ext
    -- use commutativity to write as division
    have : (p i : ℝ) = (w i : ℝ) / (((A.sum w : NNReal) : ℝ)) := by
      simpa [div_eq_mul_inv, mul_comm] using this
    -- now identify with normalizeOn
    have hnorm : (normalizeOn (S:=S) A w i : ℝ) = (w i : ℝ) / (((A.sum w : NNReal) : ℝ)) := by
      -- simplify normalizeOn since i ∈ A
      have := normalizeOn_inside (S:=S) A w (i:=i) hiA
      -- coerce and rewrite
      simp [this]
    simpa [hnorm]
  · -- outside A: both are 0
    have : p i = 0 := hout i hiA
    simp [normalizeOn_outside (S:=S) A w hiA, this]

end Normalize

end Nfp


/-!
## Consolidated Theorem A.1 (wrapper statements)

We collect three core ingredients used in Appendix A.1’s uniqueness story as
named wrappers. These are restatements (aliases) of lemmas proved above or in
`Nfp.Uniqueness` so downstream developments can refer to a single place.

- Residual coefficient uniqueness from row-normalization.
- Normalization uniqueness for proportional rows supported on `A`.
- Global uniqueness of tracer families for a linear local system on a finite DAG.

These wrappers introduce no new proof obligations; they simply expose the
relevant facts under Appendix A.1’s umbrella.
-/

namespace Nfp

/-! Residual coefficient uniqueness (Appendix A.3 inside A.1) -/

theorem A1_residual_unique
  (lcoeff : NNReal → NNReal → NNReal)
  (x a : NNReal) (hx : x + a ≠ 0)
  (hnorm : lcoeff x a + a / (x + a) = 1) :
  lcoeff x a = x / (x + a) :=
  residual_lambda_from_norm lcoeff x a hx hnorm

/-! Normalization uniqueness on a support (Appendix A.2 inside A.1) -/

theorem A1_normalize_unique
  {S : Type*} [Fintype S] [DecidableEq S]
  (A : Finset S) (w p : S → NNReal) (k : NNReal)
  (hAw : A.sum w ≠ 0)
  (hout : ∀ i ∉ A, p i = 0)
  (hin : ∀ i ∈ A, p i = k * w i)
  (hsum : (∑ i, p i) = 1) :
  p = normalizeOn (S:=S) A w :=
  proportional_row_unique (S:=S) A w p k hAw hout hin hsum

/-! Global linear uniqueness on a DAG (Appendix A.1 via `LocalSystem`) -/

theorem A1_global_tracer_unique
  {S : Type*} {n : ℕ}
  (L : LocalSystem n)
  {T T' : LocalSystem.TracerFamily (S := S) n}
  (hT : LocalSystem.Satisfies (S := S) L T)
  (hT' : LocalSystem.Satisfies (S := S) L T') :
  T = T' :=
  LocalSystem.tracer_unique (S:=S) L hT hT'

/-!
## Packaged Appendix A.1 theorem

We bundle the three core uniqueness components of Appendix A.1 into a single
exported statement `A1` returning a triple of universally quantified facts:

- residual coefficient uniqueness from row-normalization (A.3 used in A.1),
- normalization uniqueness on a given support (A.2 used in A.1),
- global tracer uniqueness for a linear local system on a finite DAG (A.1).
-/

/-- Appendix A.1 (packaged): a conjunction of the three core uniqueness results
used in the Appendix narrative. Each component is universally quantified over
its parameters and follows directly from the dedicated wrapper theorems above. -/
theorem A1 :
  (∀ (lcoeff : NNReal → NNReal → NNReal) (x a : NNReal),
    x + a ≠ 0 → lcoeff x a + a / (x + a) = 1 →
    lcoeff x a = x / (x + a)) ∧
  (∀ {S : Type*} [Fintype S] [DecidableEq S]
    (A : Finset S) (w p : S → NNReal) (k : NNReal),
    A.sum w ≠ 0 →
    (∀ i ∉ A, p i = 0) →
    (∀ i ∈ A, p i = k * w i) →
    (∑ i, p i) = 1 →
    p = normalizeOn (S:=S) A w) ∧
  (∀ {S : Type*} {n : ℕ}
    (L : LocalSystem n)
    {T T' : LocalSystem.TracerFamily (S := S) n},
    LocalSystem.Satisfies (S := S) L T →
    LocalSystem.Satisfies (S := S) L T' →
    T = T') := by
  refine ⟨?residual, ?normalize, ?global⟩
  · intro lcoeff x a hx hnorm; exact A1_residual_unique lcoeff x a hx hnorm
  · intro S _ _ A w p k hAw hout hin hsum;
    exact A1_normalize_unique (S:=S) A w p k hAw hout hin hsum
  · intro S n L T T' hT hT'; exact A1_global_tracer_unique (S:=S) L hT hT'

end Nfp

/-!
## Appendix A.2 (axioms-as-theorems wrappers)

We expose the A.2 "axioms" as formal properties derived from our basic
constructions. Rather than assuming them as axioms, we state them as theorems
that hold for our generic primitives:

- Graph faithfulness: composition preserves support restrictions.
- Residual additivity: the energy-consistent coefficients sum to 1 and are
  uniquely determined by row-normalization.
- Row-normalization: `normalizeOn` produces rows that sum to 1 over a finite
  support.

These are lightweight restatements of previously proven lemmas so the Appendix
can reference stable names.
-/

namespace Nfp

/-! Graph faithfulness under composition (Axiom 1) -/

/-
Appendix A.2 (Graph-faithfulness under composition).
If mixers `M : S → T` and `N : T → U` are supported by relations `R` and `Q`
respectively, then their composition is supported by the composed relation.
This is a wrapper around `Mixer.supported_comp` so Appendix A can refer to a
stable name. See docs/APPENDIX.md §A.2.
-/
theorem A2_graph_faithful_comp
  {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]
  (M : Mixer S T) (N : Mixer T U)
  (R : S → T → Prop) (Q : T → U → Prop)
  (hM : Mixer.supported (S := S) (T := T) M R)
  (hN : Mixer.supported (S := T) (T := U) N Q) :
  Mixer.supported (S := S) (T := U) (M.comp N) (Mixer.compSupport R Q) :=
  Mixer.supported_comp (S := S) (T := T) (U := U) (M := M) (N := N) hM hN

/-! Residual additivity and energy-consistency (Axiom 2) -/

/-
Appendix A.2 (Residual additivity / energy consistency).
For residual coefficient `λ := lambdaEC x a = x/(x+a)`, we have
`λ + a/(x+a) = 1`. See docs/APPENDIX.md §A.2.
-/
theorem A2_residual_energy_consistent (x a : NNReal) (hx : x + a ≠ 0) :
  lambdaEC x a + a / (x + a) = 1 :=
  lambdaEC_sum_one x a hx

/-
Appendix A.2 (Residual uniqueness from row-normalization).
Any residual rule that satisfies the row-normalization equation is uniquely
determined as `x/(x+a)`. Wrapper of `residual_lambda_from_norm`.
See docs/APPENDIX.md §A.2.
-/
theorem A2_residual_unique
  (lcoeff : NNReal → NNReal → NNReal)
  (x a : NNReal) (hx : x + a ≠ 0)
  (hnorm : lcoeff x a + a / (x + a) = 1) :
  lcoeff x a = x / (x + a) :=
  residual_lambda_from_norm lcoeff x a hx hnorm

/-! Row-normalization on a finite support (part of Axioms 1/3/6) -/

/-
Appendix A.2 (Row-normalization on a finite support).
The `normalizeOn` construction sums to 1 provided the total mass on `A` is
nonzero. Wrapper of `normalizeOn_sum_one`. See docs/APPENDIX.md §A.2.
-/
theorem A2_normalize_row_sum_one {S : Type*} [Fintype S] [DecidableEq S]
  (A : Finset S) (w : S → NNReal) (h : A.sum w ≠ 0) :
  (∑ i, normalizeOn (S := S) A w i) = 1 :=
  normalizeOn_sum_one (S := S) A w h

/-!
### Packaged Appendix A.2 statement

We bundle three generic A.2 facets into a single theorem `A2`:

1. Residual uniqueness from row-normalization at a residual node.
2. Row-normalization of proportional weights on a finite support via `normalizeOn`.
3. Preservation of support/faithfulness under composition of mixers.

This mirrors the intention that the "axioms" are consequences of our formal
definitions rather than assumptions.
-/

/-
Appendix A.2 (packaged statement).
Conjunction of: residual uniqueness, row-normalization via `normalizeOn`, and
graph-faithfulness of mixer composition. See docs/APPENDIX.md §A.2.
-/
theorem A2 :
  (∀ (lcoeff : NNReal → NNReal → NNReal) (x a : NNReal),
    x + a ≠ 0 → lcoeff x a + a / (x + a) = 1 →
    lcoeff x a = x / (x + a)) ∧
  (∀ {S : Type*} [Fintype S] [DecidableEq S]
    (A : Finset S) (w : S → NNReal),
    A.sum w ≠ 0 → (∑ i, normalizeOn (S := S) A w i) = 1) ∧
  (∀ {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]
    (M : Mixer S T) (N : Mixer T U)
    (R : S → T → Prop) (Q : T → U → Prop),
  Mixer.supported (S := S) (T := T) M R →
  Mixer.supported (S := T) (T := U) N Q →
  Mixer.supported (S := S) (T := U) (M.comp N) (Mixer.compSupport R Q)) := by
  refine ⟨?resid, ?row, ?faith⟩
  · intro lcoeff x a hx hnorm; exact A2_residual_unique lcoeff x a hx hnorm
  · intro S _ _ A w h; exact A2_normalize_row_sum_one (S := S) A w h
  · intro S T U _ _ _ M N R Q hM hN;
    exact A2_graph_faithful_comp (M:=M) (N:=N) R Q hM hN

/-- Appendix A.2 compatibility: an `InfluenceSpec` with nonzero rows yields a
graph-faithful mixer via `ofInfluenceSpec`, ready to feed into `A2` results. -/
lemma A2_for_ofInfluenceSpec {Site : Type*} [Fintype Site] [DecidableEq Site]
    (I : InfluenceSpec Site) (hZ : ∀ s, InfluenceSpec.rowTotal (Site := Site) I s ≠ 0) :
    Mixer.supported (S := Site) (T := Site) (Mixer.ofInfluenceSpec (Site := Site) I) I.adj :=
  Mixer.ofInfluenceSpec_supported (Site := Site) (I := I) hZ

end Nfp
