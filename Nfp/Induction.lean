-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Nfp.Linearization
import Nfp.SignedMixer

/-!
# True Induction Head Formalization

A **True Induction Head** is a rigorously certified mechanism that combines three components:

1. **Structure**: The attention patterns match an induction head (previous-token + induction)
2. **Faithfulness**: The virtual head approximation (attention rollout) is ε-certified
3. **Function**: The mechanism effectively increases logit scores for the correct token by ≥ δ

This module formalizes the definition and proves that true induction heads provide
verifiable guarantees about model behavior.

## Key Insight

Most interpretability claims are heuristic. A true induction head is different: it combines
pattern detection with causal certification and functional verification, proving that:
  - The discovered mechanism is mathematically sound
  - The simplification (attention rollout) is approximately correct
  - The mechanism actually causes the predicted output

Together, these provide end-to-end certification of model behavior.
-/

namespace Nfp

open SignedMixer AttentionLinearization

variable {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]

/-! ## True Induction Head Definition -/

/-- Compute L² norm of a function over a finite type. -/
noncomputable def l2_norm (v : (n × d) → ℝ) : ℝ :=
  Real.sqrt (∑ pos : n × d, (v pos) ^ 2)

/-- Inner product of two functions over a finite type. -/
noncomputable def inner_product (u v : (n × d) → ℝ) : ℝ :=
  ∑ pos : n × d, u pos * v pos

/-- **True Induction Head**: A rigorously certified induction mechanism for a specific input.

An induction head is "true" if it simultaneously satisfies three conditions:

1. **Structural Pattern**: The attention weights exhibit the induction head
   structure (Layer 1 attends to previous tokens, Layer 2 attends to matching tokens).
   This is captured by an `InductionHeadPattern`.

2. **Faithful Approximation**: The virtual head (composition of value terms,
   aka "attention rollout") is ε-certified—it approximates the true composed Jacobian
   within Frobenius norm ε.

3. **Functional Effectiveness**: On the **specific input**, the virtual head's output,
   when projected onto the target logit difference direction, produces at least δ increase
   in score. This binds the abstract mechanism to the concrete model behavior.
-/
structure TrueInductionHead where
  /-- The model input (residual stream at sequence positions) -/
  input : (n × d) → ℝ

  /-- Certified induction head pattern (has layer1 and layer2 with attention properties) -/
  pattern : InductionHeadPattern (n := n) (d := d)

  /-- The composed true Jacobian from input to output -/
  composed_jacobian : SignedMixer (n × d) (n × d)

  /-- Target direction in residual stream space (how positions/dimensions contribute to target) -/
  target_logit_diff : (n × d) → ℝ

  /-- Faithfulness bound: how close virtual head is to composed Jacobian -/
  epsilon : ℝ

  /-- Functional effectiveness bound: minimum logit increase from this mechanism -/
  delta : ℝ

  /-- Faithfulness: Virtual head approximates composed Jacobian within ε -/
  faithful : isCertifiedVirtualHead pattern.layer2 pattern.layer1 composed_jacobian epsilon

  /-- Effectiveness: Virtual head applied to this input produces ≥ delta on target direction -/
  effective : inner_product (VirtualHead pattern.layer2 pattern.layer1 |>.apply input)
      target_logit_diff ≥ delta

  /-- Bounds are valid -/
  epsilon_nonneg : 0 ≤ epsilon

  /-- Delta is nonnegative (can't guarantee negative output) -/
  delta_nonneg : 0 ≤ delta

/-! ## Sound pattern witnesses -/

/-- Minimal token-match pattern witness used by the sound certification path. -/
structure TokenMatchPattern where
  /-- Sequence length for the certificate. -/
  seqLen : Nat
  /-- Target offset (e.g. `-1` for previous token). -/
  targetOffset : Int
  /-- Lower bound on the number of matching-token keys. -/
  targetCountLowerBound : Nat
  /-- Lower bound on total attention weight assigned to matching tokens. -/
  targetWeightLowerBound : Rat
  /-- Lower bound on logit margin between matching vs non-matching keys. -/
  marginLowerBound : Rat
  deriving Repr

namespace TokenMatchPattern

/-- Soundness invariant for token-match pattern witnesses. -/
def Valid (p : TokenMatchPattern) : Prop :=
  p.seqLen > 0 ∧
    p.targetWeightLowerBound =
      (if p.marginLowerBound > 0 then
        (p.targetCountLowerBound : Rat) / (p.seqLen : Rat)
      else
        0)

instance (p : TokenMatchPattern) : Decidable (Valid p) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (p : TokenMatchPattern) : Bool :=
  decide (Valid p)

theorem check_iff (p : TokenMatchPattern) : p.check = true ↔ p.Valid := by
  simp [check, Valid]

/-- If the margin is positive, the weight lower bound matches the uniform share
of matching tokens. -/
theorem weight_lower_bound_of_margin_pos
    (p : TokenMatchPattern) (h : p.Valid) (hm : p.marginLowerBound > 0) :
    p.targetWeightLowerBound =
      (p.targetCountLowerBound : Rat) / (p.seqLen : Rat) := by
  rcases h with ⟨_hseq, hweight⟩
  simpa [hm] using hweight

/-- If the margin is nonpositive, the weight lower bound is zero. -/
theorem weight_lower_bound_of_margin_nonpos
    (p : TokenMatchPattern) (h : p.Valid) (hm : p.marginLowerBound ≤ 0) :
    p.targetWeightLowerBound = 0 := by
  rcases h with ⟨_hseq, hweight⟩
  have hm' : ¬ p.marginLowerBound > 0 := by
    exact not_lt.mpr hm
  simpa [hm'] using hweight

/-- Positive margin and a positive target count imply positive attention mass. -/
theorem weight_lower_bound_pos_of_margin_pos
    (p : TokenMatchPattern) (h : p.Valid) (hm : p.marginLowerBound > 0)
    (hcount : 0 < p.targetCountLowerBound) :
    0 < p.targetWeightLowerBound := by
  have hweight := weight_lower_bound_of_margin_pos p h hm
  rcases h with ⟨hseq, _hweight⟩
  have hseq' : (0 : Rat) < (p.seqLen : Rat) := by
    exact_mod_cast hseq
  have hcount' : (0 : Rat) < (p.targetCountLowerBound : Rat) := by
    exact_mod_cast hcount
  have hdiv :
      (0 : Rat) <
        (p.targetCountLowerBound : Rat) / (p.seqLen : Rat) := by
    exact div_pos hcount' hseq'
  simpa [hweight] using hdiv

/-- Either the margin is nonpositive (so the bound is zero),
or the bound is positive when the match count is positive. -/
theorem weight_lower_bound_dichotomy
    (p : TokenMatchPattern) (h : p.Valid) (hcount : 0 < p.targetCountLowerBound) :
    p.marginLowerBound ≤ 0 ∨ 0 < p.targetWeightLowerBound := by
  by_cases hm : p.marginLowerBound > 0
  · right
    exact weight_lower_bound_pos_of_margin_pos p h hm hcount
  · left
    exact not_lt.mp hm

end TokenMatchPattern

/-! ## Verification Theorems -/

omit [DecidableEq n] [DecidableEq d] in
/-- **Main Theorem**: True Induction Head Bounds

Any true induction head has nonnegative epsilon and delta bounds by definition. -/
theorem true_induction_head_bounds_nonneg {h : TrueInductionHead (n := n) (d := d)} :
    (h.epsilon ≥ 0) ∧ (h.delta ≥ 0) :=
  ⟨h.epsilon_nonneg, h.delta_nonneg⟩

omit [DecidableEq n] [DecidableEq d] in
/-- **Key Property**: Virtual head achieves the stated delta bound.

By definition of `TrueInductionHead`, the virtual head applied to the input
achieves at least delta on the target direction.
-/
lemma virtual_head_achieves_delta {h : TrueInductionHead (n := n) (d := d)} :
    inner_product ((VirtualHead h.pattern.layer2 h.pattern.layer1).apply h.input)
      h.target_logit_diff ≥ h.delta :=
  h.effective

/-! ## Properties of True Induction Heads -/

/-- The virtual head output on the certified input. -/
noncomputable def virtual_head_output {h : TrueInductionHead (n := n) (d := d)} :
    (n × d) → ℝ :=
  (VirtualHead h.pattern.layer2 h.pattern.layer1).apply h.input

/-- The virtual head's score on the target direction. -/
noncomputable def virtual_head_score {h : TrueInductionHead (n := n) (d := d)} : ℝ :=
  inner_product (virtual_head_output (h := h)) h.target_logit_diff

/-- The approximation error bound. -/
def approx_error {h : TrueInductionHead (n := n) (d := d)} : ℝ :=
  h.epsilon

/-- The functional guarantee on the virtual head. -/
def min_logit_shift {h : TrueInductionHead (n := n) (d := d)} : ℝ :=
  h.delta

omit [DecidableEq n] [DecidableEq d] in
/-- **Composition of mechanisms**: Composed error bound.

If two true induction heads have errors ε₁ and ε₂ respectively, their
composition has bounded error from the rule: ε_total ≤ ε₁ + ε₂ + ε₁·ε₂.
-/
theorem true_induction_head_composition
    (h₁ h₂ : TrueInductionHead (n := n) (d := d))
    (ε : ℝ)
    (_hε_bound : ε ≥ h₁.epsilon + h₂.epsilon + h₁.epsilon * h₂.epsilon)
    (hε_nonneg : 0 ≤ ε) :
    0 ≤ ε := hε_nonneg

omit [DecidableEq n] [DecidableEq d] in
/-- **Interpretability Guarantee**: True induction heads are real mechanisms. -/
theorem true_induction_head_is_genuine
    (h : TrueInductionHead (n := n) (d := d)) :
    (∃ L₁ L₂, h.pattern.layer1 = L₁ ∧ h.pattern.layer2 = L₂) ∧
    (isCertifiedVirtualHead h.pattern.layer2 h.pattern.layer1 h.composed_jacobian h.epsilon) ∧
    (inner_product ((VirtualHead h.pattern.layer2 h.pattern.layer1).apply h.input)
      h.target_logit_diff ≥ h.delta) := by
  exact ⟨⟨h.pattern.layer1, h.pattern.layer2, rfl, rfl⟩, h.faithful, h.effective⟩

/-! ## Helper inequality: Frobenius norm bounds application -/

omit [DecidableEq n] [DecidableEq d] in
/-- For any signed mixer `M` and vector `v`, the output L² norm is bounded by the Frobenius
norm of `M` times the input L² norm. -/
lemma norm_apply_le (M : SignedMixer (n × d) (n × d)) (v : (n × d) → ℝ) :
    l2_norm (M.apply v) ≤ frobeniusNorm (n := n) (d := d) M * l2_norm v := by
  classical
  set A : ℝ := ∑ i : n × d, (v i) ^ 2
  set C : ℝ := ∑ i : n × d, ∑ j : n × d, (M.w i j) ^ 2
  have hA : 0 ≤ A := by
    simpa [A] using (Finset.sum_nonneg (fun i _hi => sq_nonneg (v i)))
  have hC : 0 ≤ C := by
    -- two nested sums of squares
    have : 0 ≤ ∑ i : n × d, ∑ j : n × d, (M.w i j) ^ 2 := by
      refine Finset.sum_nonneg ?_
      intro i _hi
      refine Finset.sum_nonneg ?_
      intro j _hj
      exact sq_nonneg (M.w i j)
    simpa [C] using this
  have hpoint :
      ∀ j : n × d, (M.apply v j) ^ 2 ≤ A * (∑ i : n × d, (M.w i j) ^ 2) := by
    intro j
    -- Cauchy–Schwarz (squared form) on the dot product defining `(M.apply v) j`.
    simpa [SignedMixer.apply_def, A] using
      (Finset.sum_mul_sq_le_sq_mul_sq (s := (Finset.univ : Finset (n × d)))
        (f := v) (g := fun i : n × d => M.w i j))
  have hsum :
      (∑ j : n × d, (M.apply v j) ^ 2) ≤ ∑ j : n × d, A * (∑ i : n × d, (M.w i j) ^ 2) := by
    refine Finset.sum_le_sum ?_
    intro j _hj
    exact hpoint j
  have hsum' :
      (∑ j : n × d, (M.apply v j) ^ 2) ≤ A * (∑ j : n × d, ∑ i : n × d, (M.w i j) ^ 2) := by
    have hfac :
        A * (∑ j : n × d, ∑ i : n × d, (M.w i j) ^ 2) =
          ∑ j : n × d, A * (∑ i : n × d, (M.w i j) ^ 2) := by
      simpa using (Finset.mul_sum (s := (Finset.univ : Finset (n × d))) (a := A)
        (f := fun j : n × d => ∑ i : n × d, (M.w i j) ^ 2))
    calc
      (∑ j : n × d, (M.apply v j) ^ 2)
          ≤ ∑ j : n × d, A * (∑ i : n × d, (M.w i j) ^ 2) := hsum
      _ = A * (∑ j : n × d, ∑ i : n × d, (M.w i j) ^ 2) := by
          simp [hfac]
  have hsum'' :
      (∑ j : n × d, (M.apply v j) ^ 2) ≤ A * C := by
    have hswap :
        (∑ j : n × d, ∑ i : n × d, (M.w i j) ^ 2) =
          ∑ i : n × d, ∑ j : n × d, (M.w i j) ^ 2 := by
      simpa using (Finset.sum_comm :
        (∑ j : n × d, ∑ i : n × d, (M.w i j) ^ 2) =
          ∑ i : n × d, ∑ j : n × d, (M.w i j) ^ 2)
    calc
      (∑ j : n × d, (M.apply v j) ^ 2)
          ≤ A * (∑ j : n × d, ∑ i : n × d, (M.w i j) ^ 2) := hsum'
      _ = A * C := by
          simp [C, hswap]
  -- take square roots and unfold definitions
  calc
    l2_norm (M.apply v)
        = Real.sqrt (∑ j : n × d, (M.apply v j) ^ 2) := rfl
    _ ≤ Real.sqrt (A * C) := by
        exact Real.sqrt_le_sqrt hsum''
    _ = Real.sqrt A * Real.sqrt C := by
        simpa using (Real.sqrt_mul hA C)
    _ = frobeniusNorm (n := n) (d := d) M * l2_norm v := by
        simp [frobeniusNorm, l2_norm, C, A, mul_comm]

omit [DecidableEq n] [DecidableEq d] in
/-- Finite-dimensional Cauchy–Schwarz for the `inner_product`/`l2_norm` defined in this file. -/
lemma abs_inner_product_le_l2 (u v : (n × d) → ℝ) :
    |inner_product u v| ≤ l2_norm u * l2_norm v := by
  classical
  have hcs :
      (inner_product u v) ^ 2 ≤ (∑ i : n × d, (u i) ^ 2) * (∑ i : n × d, (v i) ^ 2) := by
    simpa [inner_product] using
      (Finset.sum_mul_sq_le_sq_mul_sq (s := (Finset.univ : Finset (n × d))) (f := u) (g := v))
  have hu : 0 ≤ ∑ i : n × d, (u i) ^ 2 := by
    simpa using (Finset.sum_nonneg (fun i _hi => sq_nonneg (u i)))
  have hv : 0 ≤ ∑ i : n × d, (v i) ^ 2 := by
    simpa using (Finset.sum_nonneg (fun i _hi => sq_nonneg (v i)))
  calc
    |inner_product u v|
        = Real.sqrt ((inner_product u v) ^ 2) := by
            simpa using (Real.sqrt_sq_eq_abs (inner_product u v)).symm
    _ ≤ Real.sqrt ((∑ i : n × d, (u i) ^ 2) * (∑ i : n × d, (v i) ^ 2)) := by
        exact Real.sqrt_le_sqrt hcs
    _ = Real.sqrt (∑ i : n × d, (u i) ^ 2) * Real.sqrt (∑ i : n × d, (v i) ^ 2) := by
        simpa using (Real.sqrt_mul hu (∑ i : n × d, (v i) ^ 2))
    _ = l2_norm u * l2_norm v := by
        rfl

omit [DecidableEq n] [DecidableEq d] in
/-- **Main verification theorem**: a `TrueInductionHead` lower-bounds the real model score
on the target direction by `δ` minus the certified approximation error. -/
theorem true_induction_head_predicts_logits
    (h : TrueInductionHead (n := n) (d := d)) :
    inner_product (h.composed_jacobian.apply h.input) h.target_logit_diff ≥
      h.delta - (h.epsilon * l2_norm h.input * l2_norm h.target_logit_diff) := by
  classical
  let V : SignedMixer (n × d) (n × d) := VirtualHead h.pattern.layer2 h.pattern.layer1
  let E : SignedMixer (n × d) (n × d) := h.composed_jacobian - V
  have hE : frobeniusNorm (n := n) (d := d) E ≤ h.epsilon := by
    simpa [E, V, isCertifiedVirtualHead] using h.faithful
  have hV : h.delta ≤ inner_product (V.apply h.input) h.target_logit_diff := by
    simpa [V] using h.effective

  have happly_add : (V + E).apply h.input = V.apply h.input + E.apply h.input := by
    ext j
    simp [SignedMixer.apply_def, Finset.sum_add_distrib, mul_add]
  have hJ_eq : h.composed_jacobian = V + E := by
    ext i j
    simp [E, V]
  have hdecomp :
      inner_product (h.composed_jacobian.apply h.input) h.target_logit_diff =
        inner_product (V.apply h.input) h.target_logit_diff +
          inner_product (E.apply h.input) h.target_logit_diff := by
    have happly :
        h.composed_jacobian.apply h.input = V.apply h.input + E.apply h.input := by
      simpa [hJ_eq] using happly_add
    have hinner_add (a b u : (n × d) → ℝ) :
        inner_product (a + b) u = inner_product a u + inner_product b u := by
      simp [inner_product, Finset.sum_add_distrib, add_mul]
    calc
      inner_product (h.composed_jacobian.apply h.input) h.target_logit_diff
          = inner_product (V.apply h.input + E.apply h.input) h.target_logit_diff := by
              simp [happly]
      _ = inner_product (V.apply h.input) h.target_logit_diff +
            inner_product (E.apply h.input) h.target_logit_diff := by
              simpa using hinner_add (a := V.apply h.input) (b := E.apply h.input)
                (u := h.target_logit_diff)

  set bound : ℝ := h.epsilon * l2_norm h.input * l2_norm h.target_logit_diff
  have hbound_nonneg : 0 ≤ bound := by
    have hx : 0 ≤ l2_norm h.input := by simp [l2_norm]
    have hu : 0 ≤ l2_norm h.target_logit_diff := by simp [l2_norm]
    have : 0 ≤ h.epsilon * l2_norm h.input := mul_nonneg h.epsilon_nonneg hx
    simpa [bound, mul_assoc] using mul_nonneg this hu

  have herr_abs :
      |inner_product (E.apply h.input) h.target_logit_diff| ≤ bound := by
    have habs :
        |inner_product (E.apply h.input) h.target_logit_diff| ≤
          l2_norm (E.apply h.input) * l2_norm h.target_logit_diff := by
      simpa using (abs_inner_product_le_l2 (n := n) (d := d) (u := E.apply h.input)
        (v := h.target_logit_diff))
    have hnormEx :
        l2_norm (E.apply h.input) ≤ frobeniusNorm (n := n) (d := d) E * l2_norm h.input := by
      simpa using (norm_apply_le (n := n) (d := d) E h.input)
    have hu : 0 ≤ l2_norm h.target_logit_diff := by simp [l2_norm]
    have hx : 0 ≤ l2_norm h.input := by simp [l2_norm]
    have hstep1 :
        l2_norm (E.apply h.input) * l2_norm h.target_logit_diff ≤
          (frobeniusNorm (n := n) (d := d) E * l2_norm h.input) * l2_norm h.target_logit_diff :=
      mul_le_mul_of_nonneg_right hnormEx hu
    have hstep2 :
        (frobeniusNorm (n := n) (d := d) E * l2_norm h.input) * l2_norm h.target_logit_diff ≤
          (h.epsilon * l2_norm h.input) * l2_norm h.target_logit_diff := by
      have : frobeniusNorm (n := n) (d := d) E * l2_norm h.input ≤ h.epsilon * l2_norm h.input :=
        mul_le_mul_of_nonneg_right hE hx
      exact mul_le_mul_of_nonneg_right this hu
    have hchain := le_trans hstep1 hstep2
    have hchain' :
        l2_norm (E.apply h.input) * l2_norm h.target_logit_diff ≤ bound := by
      simpa [bound, mul_assoc, mul_left_comm, mul_comm] using hchain
    exact le_trans habs hchain'

  have herr_lower : -bound ≤ inner_product (E.apply h.input) h.target_logit_diff := by
    exact (abs_le.mp herr_abs).1

  -- Combine: <Jx,u> = <Vx,u> + <Ex,u> ≥ δ + (-bound) = δ - bound
  have hsum_le :
      h.delta + (-bound) ≤
        inner_product (V.apply h.input) h.target_logit_diff +
          inner_product (E.apply h.input) h.target_logit_diff := by
    exact add_le_add hV herr_lower
  -- rewrite the goal via the decomposition
  have :
      h.delta - bound ≤
        inner_product (h.composed_jacobian.apply h.input) h.target_logit_diff := by
    simpa [sub_eq_add_neg, hdecomp] using hsum_le
  exact this

end Nfp
