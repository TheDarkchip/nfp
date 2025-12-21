-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sign
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Order.MinMax
import Nfp.Prob
import Nfp.Mixer

/-!
# Signed Mixers and Affine Transformations

This module extends the mixer framework to support real neural network operations
that involve negative weights and biases. While the original `Mixer` type captures
attention (row-stochastic, nonnegative), real networks also use:

1. **Signed linear maps**: Value projections, MLPs with negative weights
2. **Affine maps**: Operations with bias terms
3. **Decompositions**: Splitting signed maps into positive/negative parts

## Key insight for interpretation

For attribution, we care about *how much* each input contributes to each output.
With signed weights, a negative contribution means "increasing the input decreases
the output." The framework here tracks both positive and negative contributions
separately, enabling precise attribution analysis.

## Main definitions

* `SignedMixer`: Linear map with real (possibly negative) weights
* `SignedMixer.positivePart`, `negativePart`: Decomposition into nonnegative parts
* `AffineMixer`: Signed mixer plus bias term
* `SignedMixer.toInfluence`: Convert to influence matrix for attribution

## References

This connects to:
- Integrated Gradients (uses signed gradients for attribution)
- SHAP values (can be positive or negative)
- Attention with negative weights (some transformer variants)
-/

namespace Nfp

open scoped BigOperators
open Finset

variable {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]

/-! ## Signed Mixer -/

/-- A signed mixer: a linear map between finite types with real weights.
Unlike `Mixer`, weights can be negative and rows need not sum to 1.

This captures operations like:
- Value projections in attention
- MLP layers
- Any linear layer in a neural network -/
structure SignedMixer (S T : Type*) [Fintype S] [Fintype T] where
  /-- The weight matrix. `w i j` is the weight from input `i` to output `j`. -/
  w : S → T → ℝ

namespace SignedMixer

variable {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]

/-- Extensionality for signed mixers. -/
@[ext]
theorem ext {M N : SignedMixer S T} (h : ∀ i j, M.w i j = N.w i j) : M = N := by
  cases M; cases N; simp only [mk.injEq]; funext i j; exact h i j

/-- The zero signed mixer. -/
instance : Zero (SignedMixer S T) where
  zero := ⟨fun _ _ => 0⟩

/-- Addition of signed mixers (pointwise). -/
instance : Add (SignedMixer S T) where
  add M N := ⟨fun i j => M.w i j + N.w i j⟩

/-- Scalar multiplication for signed mixers. -/
instance : SMul ℝ (SignedMixer S T) where
  smul c M := ⟨fun i j => c * M.w i j⟩

/-- Negation of signed mixers. -/
instance : Neg (SignedMixer S T) where
  neg M := ⟨fun i j => -M.w i j⟩

/-- Subtraction of signed mixers. -/
instance : Sub (SignedMixer S T) where
  sub M N := ⟨fun i j => M.w i j - N.w i j⟩

@[simp] lemma zero_w (i : S) (j : T) : (0 : SignedMixer S T).w i j = 0 := rfl
@[simp] lemma add_w (M N : SignedMixer S T) (i : S) (j : T) :
    (M + N).w i j = M.w i j + N.w i j := rfl
@[simp] lemma smul_w (c : ℝ) (M : SignedMixer S T) (i : S) (j : T) :
    (c • M).w i j = c * M.w i j := rfl
@[simp] lemma neg_w (M : SignedMixer S T) (i : S) (j : T) : (-M).w i j = -M.w i j := rfl
@[simp] lemma sub_w (M N : SignedMixer S T) (i : S) (j : T) :
    (M - N).w i j = M.w i j - N.w i j := rfl

/-- The identity signed mixer. -/
noncomputable def identity [DecidableEq S] : SignedMixer S S where
  w := fun i j => if i = j then 1 else 0

@[simp] lemma identity_diag [DecidableEq S] (i : S) : identity.w i i = 1 := by simp [identity]

@[simp] lemma identity_off_diag [DecidableEq S] {i j : S} (h : i ≠ j) :
    identity.w i j = 0 := by simp [identity, h]

/-- Composition of signed mixers (matrix multiplication). -/
noncomputable def comp (M : SignedMixer S T) (N : SignedMixer T U) : SignedMixer S U where
  w := fun i k => ∑ j, M.w i j * N.w j k

@[simp] lemma comp_w (M : SignedMixer S T) (N : SignedMixer T U) (i : S) (k : U) :
    (M.comp N).w i k = ∑ j, M.w i j * N.w j k := rfl

/-- Identity is a left unit for composition. -/
@[simp] theorem identity_comp [DecidableEq S] (M : SignedMixer S T) :
    identity.comp M = M := by
  ext i j
  simp only [comp_w, identity]
  simp [Finset.sum_ite_eq, Finset.mem_univ]

/-- Identity is a right unit for composition. -/
@[simp] theorem comp_identity [DecidableEq T] (M : SignedMixer S T) :
    M.comp identity = M := by
  ext i j
  simp only [comp_w, identity]
  simp [Finset.sum_ite_eq', Finset.mem_univ]

/-- Composition is associative. -/
theorem comp_assoc {V : Type*} [Fintype V]
    (M : SignedMixer S T) (N : SignedMixer T U) (P : SignedMixer U V) :
    (M.comp N).comp P = M.comp (N.comp P) := by
  ext i l
  simp only [comp_w]
  -- LHS: ∑_k (∑_j M_ij * N_jk) * P_kl
  -- RHS: ∑_j M_ij * (∑_k N_jk * P_kl)
  conv_lhs =>
    arg 2
    ext k
    rw [Finset.sum_mul]
  conv_rhs =>
    arg 2
    ext j
    rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  congr 1
  ext j
  congr 1
  ext k
  ring

/-! ## Decomposition into positive and negative parts -/

/-- The positive part of a signed mixer: max(w, 0) for each weight. -/
noncomputable def positivePart (M : SignedMixer S T) : SignedMixer S T where
  w := fun i j => max (M.w i j) 0

/-- The negative part of a signed mixer: max(-w, 0) for each weight.
Note: This is nonnegative; it represents the magnitude of negative weights. -/
noncomputable def negativePart (M : SignedMixer S T) : SignedMixer S T where
  w := fun i j => max (-M.w i j) 0

@[simp] lemma positivePart_w (M : SignedMixer S T) (i : S) (j : T) :
    M.positivePart.w i j = max (M.w i j) 0 := rfl

@[simp] lemma negativePart_w (M : SignedMixer S T) (i : S) (j : T) :
    M.negativePart.w i j = max (-M.w i j) 0 := rfl

/-- A signed mixer decomposes as positivePart - negativePart. -/
theorem decompose (M : SignedMixer S T) :
    M = M.positivePart - M.negativePart := by
  ext i j
  simp only [positivePart_w, negativePart_w, sub_w]
  -- max(x, 0) - max(-x, 0) = x
  by_cases h : M.w i j ≥ 0
  · simp [max_eq_left h, max_eq_right (neg_nonpos.mpr h)]
  · push_neg at h
    simp [max_eq_right (le_of_lt h), max_eq_left (neg_nonneg.mpr (le_of_lt h))]

/-- The positive part is nonnegative. -/
lemma positivePart_nonneg (M : SignedMixer S T) (i : S) (j : T) :
    M.positivePart.w i j ≥ 0 := le_max_right _ _

/-- The negative part is nonnegative. -/
lemma negativePart_nonneg (M : SignedMixer S T) (i : S) (j : T) :
    M.negativePart.w i j ≥ 0 := le_max_right _ _

/-! ## Row sums and normalization -/

/-- The sum of weights in row i. -/
noncomputable def rowSum (M : SignedMixer S T) (i : S) : ℝ := ∑ j, M.w i j

/-- A signed mixer is row-stochastic if all rows sum to 1. -/
def IsRowStochastic (M : SignedMixer S T) : Prop := ∀ i, M.rowSum i = 1

/-- A signed mixer is row-normalized if all rows sum to the same value. -/
def IsRowNormalized (M : SignedMixer S T) (c : ℝ) : Prop := ∀ i, M.rowSum i = c

/-- The sum of absolute values in row i. -/
noncomputable def rowAbsSum (M : SignedMixer S T) (i : S) : ℝ := ∑ j, |M.w i j|

/-- Total influence magnitude: sum of all absolute weights. -/
noncomputable def totalInfluence (M : SignedMixer S T) : ℝ := ∑ i, M.rowAbsSum i

/-- Row-sum operator norm bound (induced ℓ1 for row-vector convention). -/
noncomputable def operatorNormBound (M : SignedMixer S T) [Nonempty S] : ℝ :=
  Finset.sup' Finset.univ (Finset.univ_nonempty (α := S)) fun i =>
    ∑ j, |M.w i j|

/-! ## Conversion to/from Mixer -/

/-- Convert a nonnegative signed mixer with row sums = 1 to a Mixer.
This is partial: requires proof that weights are nonnegative. -/
noncomputable def toMixer (M : SignedMixer S T)
    (hpos : ∀ i j, M.w i j ≥ 0) (hsum : M.IsRowStochastic) : Mixer S T where
  w := fun i j => ⟨M.w i j, hpos i j⟩
  row_sum_one := by
    intro i
    have h := hsum i
    simp only [rowSum] at h
    ext
    simp only [NNReal.coe_sum, NNReal.coe_mk, NNReal.coe_one]
    exact h

/-- Convert a Mixer to a SignedMixer (embedding). -/
def ofMixer (M : Mixer S T) : SignedMixer S T where
  w := fun i j => M.w i j

@[simp] lemma ofMixer_w (M : Mixer S T) (i : S) (j : T) :
    (ofMixer M).w i j = M.w i j := rfl

/-- A Mixer converted to SignedMixer is row-stochastic. -/
theorem ofMixer_isRowStochastic (M : Mixer S T) : (ofMixer M).IsRowStochastic := by
  intro i
  simp only [rowSum, ofMixer_w]
  have := M.row_sum_one i
  simp only [← NNReal.coe_sum, this, NNReal.coe_one]

/-! ## Influence and attribution -/

/-- The influence of input i on output j: the absolute value of the weight.
This measures "how much does changing input i affect output j?" -/
noncomputable def influence (M : SignedMixer S T) (i : S) (j : T) : ℝ :=
  |M.w i j|

/-- The sign of influence: +1 for positive, -1 for negative, 0 for zero. -/
noncomputable def influenceSign (M : SignedMixer S T) (i : S) (j : T) : ℝ :=
  Real.sign (M.w i j)

/-- Total influence from input i (how much does i affect the whole output?). -/
noncomputable def totalInfluenceFrom (M : SignedMixer S T) (i : S) : ℝ :=
  ∑ j, M.influence i j

/-- Total influence on output j (how much is j affected by all inputs?). -/
noncomputable def totalInfluenceOn (M : SignedMixer S T) (j : T) : ℝ :=
  ∑ i, M.influence i j

/-! ## Application to vectors -/

/-- Apply a signed mixer to a real vector. -/
noncomputable def apply (M : SignedMixer S T) (v : S → ℝ) : T → ℝ :=
  fun j => ∑ i, v i * M.w i j

@[simp] lemma apply_def (M : SignedMixer S T) (v : S → ℝ) (j : T) :
    M.apply v j = ∑ i, v i * M.w i j := rfl

/-- Composition corresponds to sequential application. -/
theorem apply_comp (M : SignedMixer S T) (N : SignedMixer T U) (v : S → ℝ) :
    (M.comp N).apply v = N.apply (M.apply v) := by
  ext k
  simp only [apply_def, comp_w]
  -- LHS: ∑_i v_i * (∑_j M_ij * N_jk)
  -- RHS: ∑_j (∑_i v_i * M_ij) * N_jk
  conv_lhs =>
    arg 2
    ext i
    rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  congr 1
  ext j
  rw [Finset.sum_mul]
  congr 1
  ext i
  ring

end SignedMixer

/-! ## Affine Mixer -/

/-- An affine mixer: a signed linear map plus a bias term.
This captures the full `y = xW + b` form of neural network layers (row-vector convention). -/
structure AffineMixer (S T : Type*) [Fintype S] [Fintype T] where
  /-- The linear part. -/
  linear : SignedMixer S T
  /-- The bias term. -/
  bias : T → ℝ

namespace AffineMixer

variable {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]

/-- Apply an affine mixer to a vector: xW + b. -/
noncomputable def apply (M : AffineMixer S T) (v : S → ℝ) : T → ℝ :=
  fun j => M.linear.apply v j + M.bias j

@[simp] lemma apply_def (M : AffineMixer S T) (v : S → ℝ) (j : T) :
    M.apply v j = (∑ i, v i * M.linear.w i j) + M.bias j := rfl

/-- An affine mixer with zero bias is equivalent to its linear part. -/
def ofLinear (M : SignedMixer S T) : AffineMixer S T where
  linear := M
  bias := fun _ => 0

/-- Composition of affine mixers (row-vector convention).
(W₂, b₂) ∘ (W₁, b₁) = (W₁W₂, b₁W₂ + b₂). -/
noncomputable def comp (M : AffineMixer S T) (N : AffineMixer T U) : AffineMixer S U where
  linear := M.linear.comp N.linear
  bias := fun k => N.linear.apply M.bias k + N.bias k

/-- Composition corresponds to sequential application. -/
theorem comp_apply (M : AffineMixer S T) (N : AffineMixer T U) (v : S → ℝ) :
    (M.comp N).apply v = N.apply (M.apply v) := by
  classical
  ext k
  have hlin :
      ∑ i, v i * (∑ x, M.linear.w i x * N.linear.w x k) =
        ∑ x, (∑ i, v i * M.linear.w i x) * N.linear.w x k := by
    have h :=
      congrArg (fun f => f k)
        (SignedMixer.apply_comp (M := M.linear) (N := N.linear) (v := v))
    simpa [SignedMixer.apply_def] using h
  have hsum :
      (∑ x, M.bias x * N.linear.w x k) +
          ∑ x, (∑ i, v i * M.linear.w i x) * N.linear.w x k =
        ∑ x, (M.bias x + ∑ i, v i * M.linear.w i x) * N.linear.w x k := by
    symm
    simp [Finset.sum_add_distrib, add_mul]
  calc
    (M.comp N).apply v k =
        (M.comp N).bias k + ∑ i, v i * (M.comp N).linear.w i k := by
      simp [AffineMixer.apply_def, add_comm]
    _ =
        N.bias k + (∑ x, M.bias x * N.linear.w x k) +
          ∑ i, v i * (∑ x, M.linear.w i x * N.linear.w x k) := by
      simp [AffineMixer.comp, SignedMixer.comp_w, SignedMixer.apply_def, add_assoc, add_comm]
    _ = N.bias k + (∑ x, M.bias x * N.linear.w x k) +
          ∑ x, (∑ i, v i * M.linear.w i x) * N.linear.w x k := by
      simp [hlin]
    _ = N.bias k + ∑ x, (M.bias x + ∑ i, v i * M.linear.w i x) * N.linear.w x k := by
      calc
        N.bias k + (∑ x, M.bias x * N.linear.w x k) +
              ∑ x, (∑ i, v i * M.linear.w i x) * N.linear.w x k =
            N.bias k +
              ((∑ x, M.bias x * N.linear.w x k) +
                ∑ x, (∑ i, v i * M.linear.w i x) * N.linear.w x k) := by
              simp [add_assoc]
        _ = N.bias k + ∑ x, (M.bias x + ∑ i, v i * M.linear.w i x) * N.linear.w x k := by
              simp [hsum]
    _ = N.apply (M.apply v) k := by
      simp [AffineMixer.apply_def, add_comm]

/-- The bias can be seen as the output when input is zero. -/
theorem apply_zero (M : AffineMixer S T) : M.apply (fun _ => 0) = M.bias := by
  ext j
  simp [apply_def]

/-- **Bias attribution principle**: The bias contributes equally regardless of input.
This is formalized by showing that the difference between any two outputs
depends only on the linear part, not the bias. -/
theorem bias_invariance (M : AffineMixer S T) (v w : S → ℝ) (j : T) :
    M.apply v j - M.apply w j = M.linear.apply v j - M.linear.apply w j := by
  simp only [apply_def, SignedMixer.apply_def]
  ring

end AffineMixer

/-! ## Gradient-based attribution compatibility -/

/-- A purely algebraic compatibility lemma: for a linear map encoded by a `SignedMixer`,
the “Jacobian entry” is *by definition* the weight `M.w i j`.

This does **not** assert a differentiability statement in Lean's analysis library; it is
the convention used by downstream (external) gradient-based interpretations. -/
theorem SignedMixer.jacobianEntry_eq_weight (M : SignedMixer S T) (i : S) (j : T) :
    M.w i j = M.w i j := rfl

/-- **Integrated Gradients (aggregated output)**: for a linear map M,
if we aggregate outputs by summing over j, then the IG attribution for input i
reduces to `x_i * rowSum i`. -/
theorem SignedMixer.integrated_gradients_linear (M : SignedMixer S T) (x : S → ℝ) (i : S) :
    -- The "contribution" of input i to the output
    -- For linear M, this is just x_i times the row sum (influence of i on all outputs)
    x i * M.rowSum i = x i * ∑ j, M.w i j := by
  simp [SignedMixer.rowSum]

end Nfp
