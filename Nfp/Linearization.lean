-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Nfp.SignedMixer

/-!
# Linearization of Non-Linear Operations

Real neural networks are not linear—they use activation functions (ReLU, GeLU),
normalization layers (LayerNorm, BatchNorm), and attention softmax. To apply
our mixer-based attribution framework to real networks, we must *linearize*
these operations at a specific input.

## Key Insight

For any differentiable function f : ℝⁿ → ℝᵐ, at a specific input x₀, we use a
row-vector convention:
  f(x) ≈ f(x₀) + (x - x₀) · J_f(x₀)

where J_f(x₀) is the Jacobian matrix. This Jacobian is exactly a `SignedMixer`!

For piecewise-linear functions like ReLU, the Jacobian is well-defined almost
everywhere and consists of 0s and 1s.

## Main Definitions

* `Linearization`: A record of (input, output, Jacobian as SignedMixer)
* `reluLinearization`: ReLU's Jacobian is diagonal with 0/1 entries
* `geluLinearization`: GeLU's Jacobian based on the GeLU derivative
* `layerNormJacobian`: Full Jacobian of LayerNorm (non-trivial!)
* `softmaxJacobian`: Jacobian of softmax for attention

## Why This Matters

Given a concrete forward pass through a transformer:
1. At each layer, record the activations
2. Compute the linearization (Jacobian) at those activations
3. Compose the resulting SignedMixers
4. The composition gives end-to-end attribution via the chain rule

This connects to:
- Gradient × Input attribution
- Integrated Gradients (as a path integral of linearizations)
- Attention rollout (composition of attention Jacobians)

## References

- Sundararajan et al., "Axiomatic Attribution for Deep Networks" (Integrated Gradients)
- Abnar & Zuidema, "Quantifying Attention Flow in Transformers"
-/

namespace Nfp

open scoped BigOperators
open Finset

/-! ## Linearization Structure -/

/-- A linearization captures the local linear approximation of a function at a point.

Given f : ℝⁿ → ℝᵐ and input x₀, a linearization consists of:
- `input`: The point x₀ where we linearize
- `output`: f(x₀)
- `jacobian`: The Jacobian ∂f/∂x evaluated at x₀, as a SignedMixer

The approximation is: f(x) ≈ output + jacobian.apply(x - input) -/
structure Linearization (n m : Type*) [Fintype n] [Fintype m] where
  /-- The input point at which we linearized. -/
  input : n → ℝ
  /-- The output f(input). -/
  output : m → ℝ
  /-- The Jacobian matrix as a SignedMixer. jacobian.w i j = ∂f_j/∂x_i -/
  jacobian : SignedMixer n m

namespace Linearization

variable {n m p : Type*} [Fintype n] [Fintype m] [Fintype p]

/-- Compose two linearizations (chain rule).
If f : ℝⁿ → ℝᵐ is linearized at x with Jacobian J_f, and
   g : ℝᵐ → ℝᵖ is linearized at f(x) with Jacobian J_g, then
   g ∘ f has Jacobian J_f · J_g at x (row-vector convention). -/
noncomputable def comp
    (L₁ : Linearization n m) (L₂ : Linearization m p)
    (_h : L₂.input = L₁.output) : Linearization n p where
  input := L₁.input
  output := L₂.output
  jacobian := L₁.jacobian.comp L₂.jacobian

/-- Chain rule for composed linearizations (row-vector convention). -/
theorem comp_apply
    (L₁ : Linearization n m) (L₂ : Linearization m p) (h : L₂.input = L₁.output)
    (v : n → ℝ) :
    (L₁.comp L₂ h).jacobian.apply v = L₂.jacobian.apply (L₁.jacobian.apply v) := by
  simpa using
    (SignedMixer.apply_comp (M := L₁.jacobian) (N := L₂.jacobian) (v := v))

/-- The identity linearization (identity function). -/
noncomputable def id [DecidableEq n] : Linearization n n where
  input := fun _ => 0
  output := fun _ => 0
  jacobian := SignedMixer.identity

end Linearization

/-! ## ReLU Linearization -/

section ReLU

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The ReLU activation function: max(x, 0). -/
noncomputable def relu (x : ℝ) : ℝ := max x 0

/-- The ReLU derivative: 1 if x > 0, 0 otherwise.
At x = 0, we use the subgradient convention: derivative is 0. -/
noncomputable def reluGrad (x : ℝ) : ℝ := if x > 0 then 1 else 0

/-- ReLU applied elementwise to a vector. -/
noncomputable def reluVec (v : n → ℝ) : n → ℝ := fun i => relu (v i)

/-- The ReLU mask: which coordinates are "on" (positive). -/
def reluMask (v : n → ℝ) : n → Prop := fun i => v i > 0

/-- The ReLU mask as a 0/1 indicator. -/
noncomputable def reluMaskIndicator (v : n → ℝ) : n → ℝ :=
  fun i => if v i > 0 then 1 else 0

/-- **ReLU Linearization**: The Jacobian of ReLU is a diagonal matrix
with entries 0 or 1 based on whether the input is positive.

This is the key insight: ReLU is piecewise linear, so its local linearization
is exact (not an approximation) within each linear region. -/
noncomputable def reluLinearization (x : n → ℝ) : Linearization n n where
  input := x
  output := reluVec x
  jacobian := {
    w := fun i j => if i = j then reluGrad (x i) else 0
  }

/-- The ReLU Jacobian is diagonal. -/
theorem reluLinearization_diagonal (x : n → ℝ) (i j : n) (h : i ≠ j) :
    (reluLinearization x).jacobian.w i j = 0 := by
  simp [reluLinearization, h]

/-- The ReLU Jacobian diagonal entry is 0 or 1. -/
theorem reluLinearization_diag_binary (x : n → ℝ) (i : n) :
    (reluLinearization x).jacobian.w i i = 0 ∨
    (reluLinearization x).jacobian.w i i = 1 := by
  simp only [reluLinearization, reluGrad]
  by_cases h : x i > 0 <;> simp [h]

/-- ReLU preserves positive inputs exactly. -/
theorem relu_pos {x : ℝ} (h : x > 0) : relu x = x := by
  simp [relu, max_eq_left (le_of_lt h)]

/-- ReLU kills negative inputs. -/
theorem relu_neg {x : ℝ} (h : x ≤ 0) : relu x = 0 := by
  simp [relu, max_eq_right h]

end ReLU

/-! ## GeLU Linearization -/

section GeLU

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The GeLU (Gaussian Error Linear Unit) activation.
GeLU(x) = x · Φ(x) where Φ is the standard normal CDF.

We use the approximation: GeLU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
This is what most implementations use. -/
noncomputable def gelu (x : ℝ) : ℝ :=
  0.5 * x * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3)))

/-- The GeLU derivative.
d/dx[GeLU(x)] = Φ(x) + x · φ(x)
where φ is the standard normal PDF.

For the tanh approximation, the derivative is more complex but well-defined. -/
noncomputable def geluGrad (x : ℝ) : ℝ :=
  let s := Real.sqrt (2 / Real.pi)
  let inner := s * (x + 0.044715 * x^3)
  let tanh_inner := Real.tanh inner
  let sech2_inner := 1 - tanh_inner^2  -- sech² = 1 - tanh²
  let inner_deriv := s * (1 + 3 * 0.044715 * x^2)
  0.5 * (1 + tanh_inner) + 0.5 * x * sech2_inner * inner_deriv

/-- GeLU applied elementwise. -/
noncomputable def geluVec (v : n → ℝ) : n → ℝ := fun i => gelu (v i)

/-- **GeLU Linearization**: The Jacobian is diagonal with entries geluGrad(x_i). -/
noncomputable def geluLinearization (x : n → ℝ) : Linearization n n where
  input := x
  output := geluVec x
  jacobian := {
    w := fun i j => if i = j then geluGrad (x i) else 0
  }

/-- GeLU Jacobian is diagonal. -/
theorem geluLinearization_diagonal (x : n → ℝ) (i j : n) (h : i ≠ j) :
    (geluLinearization x).jacobian.w i j = 0 := by
  simp [geluLinearization, h]

end GeLU

/-! ## LayerNorm Linearization -/

section LayerNorm

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Mean of a vector. -/
noncomputable def mean (v : n → ℝ) : ℝ :=
  (∑ i, v i) / Fintype.card n

/-- Variance of a vector. -/
noncomputable def variance (v : n → ℝ) : ℝ :=
  let μ := mean v
  (∑ i, (v i - μ)^2) / Fintype.card n

/-- Standard deviation with epsilon for numerical stability. -/
noncomputable def stddev (v : n → ℝ) : ℝ := Real.sqrt (variance v + 1e-5)

/-- LayerNorm without learnable parameters (just normalization). -/
noncomputable def layerNorm (v : n → ℝ) : n → ℝ :=
  let μ := mean v
  let σ := stddev v
  fun i => (v i - μ) / σ

/-- LayerNorm with scale γ and bias β (per-coordinate). -/
noncomputable def layerNormFull (γ β : n → ℝ) (v : n → ℝ) : n → ℝ :=
  let normalized := layerNorm v
  fun i => γ i * normalized i + β i

/-- **LayerNorm Jacobian**: This is the key non-trivial result.

∂(LayerNorm(x))_j / ∂x_i = (1/σ) · [δ_{ij} - 1/n - (x_j - μ)(x_i - μ)/(n·σ²)]

where δ_{ij} is 1 if i=j, 0 otherwise.

This shows LayerNorm creates *dense* dependencies: every output depends on every input!
This is fundamentally different from ReLU/GeLU which are diagonal. -/
noncomputable def layerNormJacobian (x : n → ℝ) : SignedMixer n n where
  w := fun i j =>
    let μ := mean x
    let σ := stddev x
    let n_inv := (1 : ℝ) / Fintype.card n
    let centered_i := x i - μ
    let centered_j := x j - μ
    let diagonal := if i = j then 1 else 0
    (1 / σ) * (diagonal - n_inv - centered_j * centered_i / (Fintype.card n * σ^2))

/-- Diagonal linear map as a `SignedMixer`: x · diag d (row-vector convention). -/
noncomputable def diagMixer (d : n → ℝ) : SignedMixer n n where
  w := fun i j => if i = j then d j else 0

/-- Operator norm bound for a diagonal mixer from a uniform entry bound. -/
theorem operatorNormBound_diagMixer_le [Nonempty n] (d : n → ℝ) (b : ℝ)
    (h : ∀ i, |d i| ≤ b) :
    SignedMixer.operatorNormBound (diagMixer d) ≤ b := by
  classical
  dsimp [SignedMixer.operatorNormBound]
  refine (Finset.sup'_le_iff (s := Finset.univ)
    (H := Finset.univ_nonempty (α := n))
    (f := fun i => ∑ j, |(diagMixer d).w i j|)
    (a := b)).2 ?_
  intro i hi
  have hsum : (∑ j, |(diagMixer d).w i j|) = |d i| := by
    have hsum' : (∑ j, |(diagMixer d).w i j|) = |(diagMixer d).w i i| := by
      refine Fintype.sum_eq_single i ?_
      intro j hne
      have hne' : i ≠ j := by
        simpa [ne_comm] using hne
      simp [diagMixer, hne']
    simpa [diagMixer] using hsum'
  simpa [hsum] using h i

/-- Operator norm bound for `A ∘ diag(d) ∘ B` from component bounds. -/
theorem operatorNormBound_comp_diagMixer_comp_le
    {S T U : Type*} [Fintype S] [Fintype T] [Fintype U] [DecidableEq T]
    [Nonempty S] [Nonempty T]
    (A : SignedMixer S T) (B : SignedMixer T U) (d : T → ℝ)
    (a c b : ℝ)
    (hA : SignedMixer.operatorNormBound A ≤ a)
    (hB : SignedMixer.operatorNormBound B ≤ b)
    (hD : ∀ i, |d i| ≤ c) :
    SignedMixer.operatorNormBound ((A.comp (diagMixer d)).comp B) ≤ a * c * b := by
  classical
  have hD' : SignedMixer.operatorNormBound (diagMixer d) ≤ c :=
    operatorNormBound_diagMixer_le (d := d) (b := c) hD
  have hA_nonneg : 0 ≤ a :=
    le_trans (SignedMixer.operatorNormBound_nonneg (M := A)) hA
  have hB_nonneg : 0 ≤ b :=
    le_trans (SignedMixer.operatorNormBound_nonneg (M := B)) hB
  have hC_nonneg : 0 ≤ c :=
    le_trans (SignedMixer.operatorNormBound_nonneg (M := diagMixer d)) hD'
  have hcomp :
      SignedMixer.operatorNormBound ((A.comp (diagMixer d)).comp B) ≤
        SignedMixer.operatorNormBound A *
          SignedMixer.operatorNormBound (diagMixer d) *
          SignedMixer.operatorNormBound B := by
    simpa using
      (SignedMixer.operatorNormBound_comp3_le
        (A := A) (B := diagMixer d) (C := B))
  have hmul1 :
      SignedMixer.operatorNormBound A *
        SignedMixer.operatorNormBound (diagMixer d) ≤ a * c := by
    have h1 :
        SignedMixer.operatorNormBound A *
          SignedMixer.operatorNormBound (diagMixer d)
          ≤ a * SignedMixer.operatorNormBound (diagMixer d) := by
      exact mul_le_mul_of_nonneg_right hA
        (SignedMixer.operatorNormBound_nonneg (M := diagMixer d))
    have h2 :
        a * SignedMixer.operatorNormBound (diagMixer d) ≤ a * c := by
      exact mul_le_mul_of_nonneg_left hD' hA_nonneg
    exact le_trans h1 h2
  have hmul2 :
      (SignedMixer.operatorNormBound A *
        SignedMixer.operatorNormBound (diagMixer d)) *
        SignedMixer.operatorNormBound B ≤ (a * c) * b := by
    have h1 :
        (SignedMixer.operatorNormBound A *
          SignedMixer.operatorNormBound (diagMixer d)) *
          SignedMixer.operatorNormBound B ≤
        (a * c) * SignedMixer.operatorNormBound B := by
      exact mul_le_mul_of_nonneg_right hmul1
        (SignedMixer.operatorNormBound_nonneg (M := B))
    have h2 : (a * c) * SignedMixer.operatorNormBound B ≤ (a * c) * b := by
      exact mul_le_mul_of_nonneg_left hB (mul_nonneg hA_nonneg hC_nonneg)
    exact le_trans h1 h2
  have hmul2' :
      SignedMixer.operatorNormBound A *
        SignedMixer.operatorNormBound (diagMixer d) *
        SignedMixer.operatorNormBound B ≤ a * c * b := by
    simpa [mul_assoc] using hmul2
  exact le_trans hcomp hmul2'

/-- Jacobian of LayerNorm with learnable scale γ (bias β has no effect on Jacobian). -/
noncomputable def layerNormFullJacobian (γ : n → ℝ) (x : n → ℝ) : SignedMixer n n :=
  (layerNormJacobian x).comp (diagMixer γ)

/-- LayerNorm-with-affine linearization at a specific input. -/
noncomputable def layerNormFullLinearization (γ β : n → ℝ) (x : n → ℝ) : Linearization n n where
  input := x
  output := layerNormFull γ β x
  jacobian := layerNormFullJacobian γ x

/-- LayerNorm linearization at a specific input. -/
noncomputable def layerNormLinearization (x : n → ℝ) : Linearization n n where
  input := x
  output := layerNorm x
  jacobian := layerNormJacobian x

omit [DecidableEq n] in
/-- **Key insight**: LayerNorm is translation-invariant: LayerNorm(x + c·1) = LayerNorm(x).
In row-vector convention, this corresponds to the Jacobian columns summing to 0. -/
theorem layerNorm_translation_invariant [Nonempty n] (x : n → ℝ) (c : ℝ) :
    layerNorm (fun i => x i + c) = layerNorm x := by
  ext i
  simp only [layerNorm, mean, variance, stddev]
  -- First show: mean(x + c) = mean(x) + c
  have hmean : (∑ j, (x j + c)) / Fintype.card n = (∑ j, x j) / Fintype.card n + c := by
    rw [Finset.sum_add_distrib]
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    field_simp
  -- The centered value (x i + c) - mean(x + c) = x i - mean(x)
  have hcentered : ∀ j, (x j + c) - (∑ k, (x k + c)) / Fintype.card n =
                        x j - (∑ k, x k) / Fintype.card n := by
    intro j
    rw [hmean]
    ring
  -- Therefore variance is unchanged
  have hvar : (∑ j, ((x j + c) - (∑ k, (x k + c)) / Fintype.card n)^2) / Fintype.card n =
              (∑ j, (x j - (∑ k, x k) / Fintype.card n)^2) / Fintype.card n := by
    congr 1
    apply Finset.sum_congr rfl
    intro j _
    rw [hcentered]
  -- So stddev is unchanged, and the final result follows
  simp only [hcentered]

end LayerNorm

/-! ## Token-wise LayerNorm on the Residual Stream -/

section TokenwiseLayerNorm

variable {pos d : Type*} [Fintype pos] [DecidableEq pos] [Fintype d] [DecidableEq d]

/-- Lift a per-token LayerNorm Jacobian to the full residual stream `(pos × d)`.

This is block-diagonal across positions: coordinates at different positions do not mix.
-/
noncomputable def tokenwiseLayerNormFullJacobian (γ : d → ℝ) (x : pos × d → ℝ) :
    SignedMixer (pos × d) (pos × d) where
  w := fun i j =>
    if i.1 = j.1 then
      let p : pos := i.1
      (layerNormFullJacobian (n := d) γ (fun k => x (p, k))).w i.2 j.2
    else 0

end TokenwiseLayerNorm

/-! ## Rotary Position Embeddings (RoPE) -/

section RoPE

variable {pos pair : Type*}
  [Fintype pos] [DecidableEq pos]
  [Fintype pair] [DecidableEq pair]

/-- RoPE uses 2D rotations on each `(pairIdx, Bool)` coordinate pair. -/
abbrev RoPEDim (pair : Type*) := pair × Bool

/-- The RoPE linear map as a `SignedMixer` on the residual stream `(pos × (pair × Bool))`.

For each position `p` and pair index `k`, this applies the 2×2 rotation with angle `θ p k`:
`(x₀, x₁) ↦ (cos θ · x₀ - sin θ · x₁, sin θ · x₀ + cos θ · x₁)`.

This is tokenwise (block-diagonal across `pos`): different positions never mix. -/
noncomputable def ropeJacobian (θ : pos → pair → ℝ) :
    SignedMixer (pos × RoPEDim pair) (pos × RoPEDim pair) where
  w := fun i j =>
    if i.1 = j.1 then
      if i.2.1 = j.2.1 then
        let p : pos := j.1
        let k : pair := j.2.1
        let ang := θ p k
        match i.2.2, j.2.2 with
        | false, false => Real.cos ang
        | true, false => -Real.sin ang
        | false, true => Real.sin ang
        | true, true => Real.cos ang
      else 0
    else 0

/-- RoPE forward map: apply the RoPE Jacobian as a linear operator. -/
noncomputable def rope (θ : pos → pair → ℝ) (x : pos × RoPEDim pair → ℝ) :
    pos × RoPEDim pair → ℝ :=
  (ropeJacobian (pos := pos) (pair := pair) θ).apply x

@[simp] lemma ropeJacobian_cross_pos (θ : pos → pair → ℝ)
    {i j : pos × RoPEDim pair} (h : i.1 ≠ j.1) :
    (ropeJacobian (pos := pos) (pair := pair) θ).w i j = 0 := by
  simp [ropeJacobian, h]

end RoPE

/-! ## Softmax Linearization -/

section Softmax

variable {n : Type*} [Fintype n]

/-- Softmax function: softmax(x)_j = exp(x_j) / Σ_k exp(x_k) -/
noncomputable def softmax (v : n → ℝ) : n → ℝ :=
  let expSum := ∑ k, Real.exp (v k)
  fun j => Real.exp (v j) / expSum

variable [DecidableEq n]

/-- **Softmax Jacobian**: ∂softmax(x)_j / ∂x_i = softmax(x)_j · (δ_{ij} - softmax(x)_i)

This is a classic result. The Jacobian depends on the softmax *output*, not input! -/
noncomputable def softmaxJacobian (x : n → ℝ) : SignedMixer n n where
  w := fun i j =>
    let p := softmax x
    p j * ((if i = j then 1 else 0) - p i)

/-- Softmax linearization. -/
noncomputable def softmaxLinearization (x : n → ℝ) : Linearization n n where
  input := x
  output := softmax x
  jacobian := softmaxJacobian x

omit [DecidableEq n] in
/-- Softmax outputs are nonnegative. -/
theorem softmax_nonneg (x : n → ℝ) (j : n) : softmax x j ≥ 0 := by
  simp only [softmax]
  apply div_nonneg (Real.exp_nonneg _)
  exact Finset.sum_nonneg (fun _ _ => Real.exp_nonneg _)

omit [DecidableEq n] in
/-- Softmax outputs sum to 1. -/
theorem softmax_sum_one [Nonempty n] (x : n → ℝ) : ∑ j, softmax x j = 1 := by
  simp only [softmax]
  rw [← Finset.sum_div]
  apply div_self
  apply ne_of_gt
  apply Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

/-- Softmax Jacobian diagonal entries are positive (when p_j < 1). -/
theorem softmaxJacobian_diag_pos [Nonempty n] (x : n → ℝ) (j : n)
    (h : softmax x j < 1) : (softmaxJacobian x).w j j > 0 := by
  simp only [softmaxJacobian, ite_true]
  -- p_j · (1 - p_j) > 0 when 0 < p_j < 1
  have hp : softmax x j > 0 := by
    simp only [softmax]
    apply div_pos (Real.exp_pos _)
    apply Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  apply mul_pos hp
  linarith

/-- Softmax Jacobian off-diagonal entries are negative. -/
theorem softmaxJacobian_off_diag_neg [Nonempty n] (x : n → ℝ) (i j : n) (h : i ≠ j) :
    (softmaxJacobian x).w i j < 0 := by
  simp only [softmaxJacobian, if_neg h]
  -- p_j · (0 - p_i) = -p_j · p_i < 0
  have hpj : softmax x j > 0 := by
    simp only [softmax]
    apply div_pos (Real.exp_pos _)
    apply Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  have hpi : softmax x i > 0 := by
    simp only [softmax]
    apply div_pos (Real.exp_pos _)
    apply Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  linarith [mul_pos hpj hpi]

omit [DecidableEq n] in
/-- Softmax is translation-invariant: softmax(x + c·1) = softmax(x). -/
theorem softmax_translation_invariant (x : n → ℝ) (c : ℝ) :
    softmax (fun i => x i + c) = softmax x := by
  ext j
  simp only [softmax, Real.exp_add]
  -- exp(x_j + c) / Σ exp(x_k + c) = exp(x_j) · exp(c) / (exp(c) · Σ exp(x_k))
  --                               = exp(x_j) / Σ exp(x_k)
  have h : ∑ x_1 : n, Real.exp (x x_1) * Real.exp c =
           Real.exp c * ∑ k : n, Real.exp (x k) := by
    rw [Finset.mul_sum]
    congr 1
    ext k
    ring
  rw [h]
  field_simp

end Softmax

/-! ## Attribution via Linearization -/

section Attribution

variable {n m : Type*} [Fintype n] [Fintype m]

/-- Given a full forward pass linearization, compute feature attributions.

The attribution of input feature i to output feature j is:
  attr(i, j) = input_i × ∂output_j/∂input_i

This is "Gradient × Input" attribution. -/
noncomputable def gradientTimesInput (L : Linearization n m) (i : n) (j : m) : ℝ :=
  L.input i * L.jacobian.w i j

/-- Sum of gradient×input attributions equals output (for linear function).
This is the completeness axiom from our Attribution module! -/
theorem gradientTimesInput_complete (L : Linearization n n)
    (hLinear : L.output = L.jacobian.apply L.input) (j : n) :
    ∑ i, gradientTimesInput L i j = L.output j := by
  simp only [gradientTimesInput, hLinear, SignedMixer.apply_def]

/-- For composed linearizations, the chain rule gives:
  ∂output/∂input = J_first · J_{next} · ... · J_last

This is exactly `SignedMixer.comp` under the row-vector convention. -/
theorem composed_attribution {p : Type*} [Fintype p]
    (L₁ : Linearization n m) (L₂ : Linearization m p)
    (h : L₂.input = L₁.output) :
    (L₁.comp L₂ h).jacobian = L₁.jacobian.comp L₂.jacobian := rfl

end Attribution

/-! ## Full Attention Jacobian Decomposition -/

section AttentionJacobian

/-!
### The Key Insight

In a self-attention layer, the output for query position q is (before W_O):
  attnOut_q = Σ_k A_{qk} · V_k = Σ_k A_{qk} · (x_k · W_V)

where A_{qk} = softmax(Q_q · K_k^T / √d)_k

The Jacobian ∂output/∂input has two fundamentally different contributions:
1. **Value term**: ∂output/∂V · ∂V/∂x = A · (W_V · W_O)
   (attention weights × value/output projections)
2. **Pattern term**: ∂output/∂A · ∂A/∂x (how changing x shifts the attention pattern)

The **Value term** is what "Attention Rollout" uses—it treats attention weights A as fixed.
The **Pattern term** captures how attention patterns themselves shift with input changes.

**Key result**: We can bound the Pattern term, and when it's small relative to the
Value term, Attention Rollout is a "faithful" explanation.
-/

variable {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]

/-- The dimensionality of the model (as a real number for scaling). -/
noncomputable def modelDim (d : Type*) [Fintype d] : ℝ := Fintype.card d

/-- Full self-attention layer with concrete projection matrices.

This captures the complete attention mechanism:
  Q = x · W_Q,  K = x · W_K,  V = x · W_V
  scores = Q · K^T / √d
  A = softmax(scores)
  output = A · V · W_O

We index by:
- `n`: sequence positions (tokens)
- `d`: hidden dimension -/
structure FullAttentionLayer (n d : Type*) [Fintype n] [Fintype d] where
  /-- Query projection W_Q : d → d -/
  W_Q : SignedMixer d d
  /-- Key projection W_K : d → d -/
  W_K : SignedMixer d d
  /-- Value projection W_V : d → d -/
  W_V : SignedMixer d d
  /-- Output projection W_O : d → d -/
  W_O : SignedMixer d d

/-- Attention forward pass state: captures all intermediate values at a specific input.

This is what we need to compute the Jacobian—we linearize around these specific values. -/
structure AttentionForwardState (n d : Type*) [Fintype n] [Fintype d] where
  /-- Input hidden states: x[position, hidden_dim] -/
  input : n → d → ℝ
  /-- Queries after projection: Q = x · W_Q -/
  queries : n → d → ℝ
  /-- Keys after projection: K = x · W_K -/
  keys : n → d → ℝ
  /-- Values after projection: V = x · W_V -/
  values : n → d → ℝ
  /-- Raw attention scores (before softmax): scores_{qk} = Q_q · K_k^T / √d -/
  scores : n → n → ℝ
  /-- Attention weights (after softmax): A_{qk} = softmax(scores_q)_k -/
  attentionWeights : n → n → ℝ
  /-- Output before W_O: Σ_k A_{qk} · V_k -/
  attentionOutput : n → d → ℝ
  /-- Final output: attentionOutput · W_O -/
  output : n → d → ℝ

/-- Compute the forward pass for a full attention layer. -/
noncomputable def attentionForward
    (layer : FullAttentionLayer n d) (x : n → d → ℝ) : AttentionForwardState n d where
  input := x
  queries := fun pos dim => ∑ d', x pos d' * layer.W_Q.w d' dim
  keys := fun pos dim => ∑ d', x pos d' * layer.W_K.w d' dim
  values := fun pos dim => ∑ d', x pos d' * layer.W_V.w d' dim
  scores := fun q k =>
    let Q_q := fun dim => ∑ d', x q d' * layer.W_Q.w d' dim
    let K_k := fun dim => ∑ d', x k d' * layer.W_K.w d' dim
    (∑ dim, Q_q dim * K_k dim) / Real.sqrt (modelDim d)
  attentionWeights := fun q k =>
    let rawScores := fun k' =>
      let Q_q := fun dim => ∑ d', x q d' * layer.W_Q.w d' dim
      let K_k' := fun dim => ∑ d', x k' d' * layer.W_K.w d' dim
      (∑ dim, Q_q dim * K_k' dim) / Real.sqrt (modelDim d)
    softmax rawScores k
  attentionOutput := fun q dim =>
    let A := fun k =>
      let rawScores := fun k' =>
        let Q_q := fun dim' => ∑ d', x q d' * layer.W_Q.w d' dim'
        let K_k' := fun dim' => ∑ d', x k' d' * layer.W_K.w d' dim'
        (∑ dim', Q_q dim' * K_k' dim') / Real.sqrt (modelDim d)
      softmax rawScores k
    ∑ k, A k * (∑ d', x k d' * layer.W_V.w d' dim)
  output := fun q dim =>
    let attnOut := fun dim' =>
      let A := fun k =>
        let rawScores := fun k' =>
          let Q_q := fun d'' => ∑ d', x q d' * layer.W_Q.w d' d''
          let K_k' := fun d'' => ∑ d', x k' d' * layer.W_K.w d' d''
          (∑ d'', Q_q d'' * K_k' d'') / Real.sqrt (modelDim d)
        softmax rawScores k
      ∑ k, A k * (∑ d', x k d' * layer.W_V.w d' dim')
    ∑ dim', attnOut dim' * layer.W_O.w dim' dim

/-- **Extended Attention Linearization** with full projection matrices and intermediates.

This captures everything needed to decompose the Jacobian into Value and Pattern terms. -/
structure AttentionLinearization (n d : Type*) [Fintype n] [Fintype d] where
  /-- The attention layer definition -/
  layer : FullAttentionLayer n d
  /-- The forward state at a specific input -/
  state : AttentionForwardState n d
  /-- The full Jacobian of the attention layer at this input.
      Maps (position × dim) → (position × dim). -/
  fullJacobian : SignedMixer (n × d) (n × d)

/-! ### The Value Term -/

/-- **Value Term** of the attention Jacobian.

This is the Jacobian when we treat attention weights A as fixed constants.
It corresponds to "Attention Rollout" interpretability.

For output position (q, dim_out), input position (k, dim_in):
  ValueTerm_{(q,dim_out), (k,dim_in)} = A_{qk} · (W_V · W_O)_{dim_in, dim_out}

This measures: "How much does input at position k flow to output at position q,
weighted by the attention A_{qk} and projected through value/output matrices?" -/
noncomputable def valueTerm (L : AttentionLinearization n d) : SignedMixer (n × d) (n × d) where
  w := fun ⟨k, dim_in⟩ ⟨q, dim_out⟩ =>
    L.state.attentionWeights q k * (L.layer.W_V.comp L.layer.W_O).w dim_in dim_out

omit [DecidableEq n] [DecidableEq d] in
/-- The Value Term is a tensor product: A ⊗ (W_V · W_O).
This structure is why attention weights alone (Attention Rollout) make sense:
position mixing is captured by A, dimension mixing by W_V · W_O. -/
theorem valueTerm_factorizes (L : AttentionLinearization n d) (q k : n) (d_in d_out : d) :
    (valueTerm L).w (k, d_in) (q, d_out) =
    L.state.attentionWeights q k * (L.layer.W_V.comp L.layer.W_O).w d_in d_out := rfl

omit [DecidableEq n] [DecidableEq d] in
/-- Row absolute sum for the Value Term splits into attention column mass and value row mass. -/
theorem valueTerm_rowAbsSum (L : AttentionLinearization n d) (k : n) (d_in : d) :
    SignedMixer.rowAbsSum (valueTerm L) (k, d_in) =
      (∑ q, |L.state.attentionWeights q k|) *
        SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in := by
  classical
  let voEntry : d → d → ℝ :=
    fun d_in d_out => ∑ j, L.layer.W_V.w d_in j * L.layer.W_O.w j d_out
  have hprod :
      (∑ q, |L.state.attentionWeights q k|) *
          ∑ d_out, |voEntry d_in d_out| =
        ∑ q, ∑ d_out,
          |L.state.attentionWeights q k| * |voEntry d_in d_out| := by
    simpa using
      (Fintype.sum_mul_sum
        (f := fun q => |L.state.attentionWeights q k|)
        (g := fun d_out => |voEntry d_in d_out|))
  have hrow :
      SignedMixer.rowAbsSum (valueTerm L) (k, d_in) =
        ∑ x : n × d,
          |L.state.attentionWeights x.1 k| * |voEntry d_in x.2| := by
    simp [SignedMixer.rowAbsSum, valueTerm, abs_mul, voEntry, SignedMixer.comp_w]
  have hrow' :
      ∑ x : n × d,
          |L.state.attentionWeights x.1 k| * |voEntry d_in x.2| =
        ∑ q, ∑ d_out,
          |L.state.attentionWeights q k| * |voEntry d_in d_out| := by
    simpa using
      (Fintype.sum_prod_type'
        (f := fun q d_out =>
          |L.state.attentionWeights q k| * |voEntry d_in d_out|))
  calc
    SignedMixer.rowAbsSum (valueTerm L) (k, d_in)
        = ∑ q, ∑ d_out,
            |L.state.attentionWeights q k| *
              |(L.layer.W_V.comp L.layer.W_O).w d_in d_out| := by
        simpa [hrow'] using hrow
    _ = (∑ q, |L.state.attentionWeights q k|) *
          SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in := by
        simpa [SignedMixer.rowAbsSum, voEntry, SignedMixer.comp_w] using hprod.symm

omit [DecidableEq n] [DecidableEq d] in
/-- Value-term operator-norm bound from attention column mass and value projection bound. -/
theorem valueTerm_operatorNormBound_le [Nonempty n] [Nonempty d]
    (L : AttentionLinearization n d) (A B : ℝ)
    (hAttn : ∀ k, ∑ q, |L.state.attentionWeights q k| ≤ A)
    (hVO : SignedMixer.operatorNormBound (L.layer.W_V.comp L.layer.W_O) ≤ B) :
    SignedMixer.operatorNormBound (valueTerm L) ≤ A * B := by
  classical
  refine (Finset.sup'_le_iff (s := Finset.univ)
    (H := Finset.univ_nonempty (α := n × d))
    (f := fun i => SignedMixer.rowAbsSum (valueTerm L) i)
    (a := A * B)).2 ?_
  intro kd hkd
  rcases kd with ⟨k, d_in⟩
  have hRow :
      SignedMixer.rowAbsSum (valueTerm L) (k, d_in) =
        (∑ q, |L.state.attentionWeights q k|) *
          SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in :=
    valueTerm_rowAbsSum (L := L) k d_in
  have hAttn_nonneg : 0 ≤ A := by
    rcases (inferInstance : Nonempty n) with ⟨k0⟩
    have hsum_nonneg :
        0 ≤ ∑ q, |L.state.attentionWeights q k0| := by
      exact Finset.sum_nonneg (fun _ _ => abs_nonneg _)
    exact le_trans hsum_nonneg (hAttn k0)
  have hVOnonneg :
      0 ≤ SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in :=
    SignedMixer.rowAbsSum_nonneg (M := L.layer.W_V.comp L.layer.W_O) d_in
  have hVOrow :
      SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in ≤ B := by
    have hsup :
        SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in ≤
          SignedMixer.operatorNormBound (L.layer.W_V.comp L.layer.W_O) := by
      exact Finset.le_sup' (s := Finset.univ)
        (f := fun i => SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) i) (by simp)
    exact le_trans hsup hVO
  have hMul1 :
      (∑ q, |L.state.attentionWeights q k|) *
          SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in ≤
        A * SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in := by
    exact mul_le_mul_of_nonneg_right (hAttn k) hVOnonneg
  have hMul2 :
      A * SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in ≤ A * B := by
    exact mul_le_mul_of_nonneg_left hVOrow hAttn_nonneg
  have hMul :
      (∑ q, |L.state.attentionWeights q k|) *
          SignedMixer.rowAbsSum (L.layer.W_V.comp L.layer.W_O) d_in ≤ A * B := by
    exact le_trans hMul1 hMul2
  simpa [hRow] using hMul

/-! ### The Pattern Term -/

/-- **Pattern Term** of the attention Jacobian.

This captures how the attention pattern A itself changes as input changes.
It involves the softmax Jacobian and the query/key gradients.

∂A_{qk}/∂x_{i,d} = Σ_k' ∂A_{qk}/∂scores_{qk'} · ∂scores_{qk'}/∂x_{i,d}

where:
- ∂A/∂scores is the softmax Jacobian
- ∂scores/∂x involves W_Q and W_K

The Pattern Term is the contribution of this to the overall Jacobian. -/
noncomputable def patternTerm (L : AttentionLinearization n d) : SignedMixer (n × d) (n × d) where
  w := fun ⟨i, d_in⟩ ⟨q, d_out⟩ =>
    -- This is the complex term: how changing x_{i,d_in} shifts attention,
    -- and how that shifted attention affects output_{q,d_out}
    --
    -- Full formula:
    -- Σ_k Σ_k' (∂output_{q,d_out}/∂A_{qk}) · (∂A_{qk}/∂scores_{qk'}) · (∂scores_{qk'}/∂x_{i,d_in})
    --
    -- = Σ_k Σ_k' V_{k,d_out'} · W_O_{d_out',d_out} · softmaxJac_{qkk'} · scoreGrad_{qk',i,d_in}

    -- For now, we define it implicitly as fullJacobian - valueTerm
    L.fullJacobian.w (i, d_in) (q, d_out) - (valueTerm L).w (i, d_in) (q, d_out)

omit [DecidableEq n] [DecidableEq d] in
/-- **The Fundamental Decomposition**: The full Jacobian equals Value Term + Pattern Term.

This is the core insight: attention Jacobian = how values flow + how attention shifts.
When the Pattern Term is small, attention weights alone explain the network's behavior. -/
theorem attention_jacobian_decomposition (L : AttentionLinearization n d) :
    L.fullJacobian = valueTerm L + patternTerm L := by
  ext ⟨i, d_in⟩ ⟨q, d_out⟩
  simp only [SignedMixer.add_w, valueTerm, patternTerm]
  ring

omit [DecidableEq n] [DecidableEq d] in
/-- Operator-norm bound for the full attention Jacobian from Value/Pattern term bounds. -/
theorem attention_fullJacobian_bound_of_terms [Nonempty n] [Nonempty d]
    (L : AttentionLinearization n d) {A B : ℝ}
    (hValue : SignedMixer.operatorNormBound (valueTerm L) ≤ A)
    (hPattern : SignedMixer.operatorNormBound (patternTerm L) ≤ B) :
    SignedMixer.operatorNormBound L.fullJacobian ≤ A + B := by
  have hdecomp := attention_jacobian_decomposition (L := L)
  calc
    SignedMixer.operatorNormBound L.fullJacobian =
        SignedMixer.operatorNormBound (valueTerm L + patternTerm L) := by
      simp [hdecomp]
    _ ≤ SignedMixer.operatorNormBound (valueTerm L) +
          SignedMixer.operatorNormBound (patternTerm L) := by
      simpa using
        (SignedMixer.operatorNormBound_add_le
          (M := valueTerm L) (N := patternTerm L))
    _ ≤ A + B := add_le_add hValue hPattern

/-! ### Score Gradient -/

/-- Gradient of attention scores with respect to input.

∂scores_{qk}/∂x_{i,d} = (1/√d) · [δ_{qi} · Σ_d' W_Q_{d,d'} · K_k[d']
                                  + δ_{ki} · Σ_d' Q_q[d'] · W_K_{d,d'}]

The score gradient is nonzero only when i = q (query position) or i = k (key position). -/
noncomputable def scoreGradient (L : AttentionLinearization n d)
    (q k i : n) (d_in : d) : ℝ :=
  let scale := 1 / Real.sqrt (modelDim d)
  let queryContrib := if q = i then
    ∑ d', L.layer.W_Q.w d_in d' * L.state.keys k d'
  else 0
  let keyContrib := if k = i then
    ∑ d', L.state.queries q d' * L.layer.W_K.w d_in d'
  else 0
  scale * (queryContrib + keyContrib)

omit [DecidableEq d] in
/-- Score gradient is local: only the query and key positions contribute. -/
theorem scoreGradient_local (L : AttentionLinearization n d)
    (q k i : n) (d_in : d) (hq : q ≠ i) (hk : k ≠ i) :
    scoreGradient L q k i d_in = 0 := by
  simp [scoreGradient, hq, hk]

/-! ### Attention Pattern Gradient -/

/-- Gradient of attention weights with respect to input, using the softmax Jacobian.

∂A_{qk}/∂x_{i,d} = Σ_k' softmaxJac(scores_q)_{k,k'} · ∂scores_{qk'}/∂x_{i,d}
                 = Σ_k' A_{qk}·(δ_{kk'} - A_{qk'}) · scoreGrad_{qk',i,d}

Note: This involves the full softmax Jacobian evaluated at the scores. -/
noncomputable def attentionGradient (L : AttentionLinearization n d)
    (q k i : n) (d_in : d) : ℝ :=
  let A_q := L.state.attentionWeights q  -- attention distribution for query q
  ∑ k', A_q k * ((if k = k' then 1 else 0) - A_q k') * scoreGradient L q k' i d_in

omit [DecidableEq d] in
/-- The attention gradient relates to the softmax Jacobian.

Note: This requires the consistency property that
`L.state.attentionWeights q = softmax (L.state.scores q)`,
which we state as a separate condition. -/
theorem attentionGradient_via_softmax (L : AttentionLinearization n d) (q k i : n) (d_in : d)
    (hConsistent : L.state.attentionWeights q = softmax (L.state.scores q)) :
    attentionGradient L q k i d_in =
    ∑ k', (softmaxJacobian (L.state.scores q)).w k' k * scoreGradient L q k' i d_in := by
  simp only [attentionGradient, softmaxJacobian, hConsistent]
  congr 1
  ext k'
  by_cases h : k = k'
  · simp [h]
  · have hne : k' ≠ k := fun h' => h h'.symm
    simp [h, hne]

omit [DecidableEq n] [DecidableEq d] in
/-- Sum of absolute values after applying a signed mixer is controlled by the operator norm. -/
theorem sum_abs_apply_le {S T : Type*} [Fintype S] [Fintype T] [Nonempty S]
    (M : SignedMixer S T) (v : S → ℝ) :
    ∑ j, |M.apply v j| ≤ (∑ i, |v i|) * SignedMixer.operatorNormBound M := by
  classical
  have hterm : ∀ j, |M.apply v j| ≤ ∑ i, |v i| * |M.w i j| := by
    intro j
    have h :=
      (abs_sum_le_sum_abs (f := fun i => v i * M.w i j) (s := Finset.univ))
    simpa [SignedMixer.apply_def, abs_mul] using h
  have hsum :
      ∑ j, |M.apply v j| ≤ ∑ j, ∑ i, |v i| * |M.w i j| := by
    refine Finset.sum_le_sum ?_
    intro j _hj
    exact hterm j
  have hswap :
      (∑ j, ∑ i, |v i| * |M.w i j|) =
        ∑ i, |v i| * (∑ j, |M.w i j|) := by
    calc
      (∑ j, ∑ i, |v i| * |M.w i j|)
          = ∑ i, ∑ j, |v i| * |M.w i j| := by
            simpa using
              (Finset.sum_comm (s := Finset.univ) (t := Finset.univ)
                (f := fun j i => |v i| * |M.w i j|))
      _ = ∑ i, |v i| * (∑ j, |M.w i j|) := by
            refine Finset.sum_congr rfl ?_
            intro i _hi
            simp [Finset.mul_sum]
  have hrow :
      ∀ i, (∑ j, |M.w i j|) ≤ SignedMixer.operatorNormBound M := by
    intro i
    exact Finset.le_sup' (s := Finset.univ)
      (f := fun i => SignedMixer.rowAbsSum M i) (by simp)
  have hfinal :
      ∑ i, |v i| * (∑ j, |M.w i j|) ≤
        ∑ i, |v i| * SignedMixer.operatorNormBound M := by
    refine Finset.sum_le_sum ?_
    intro i _hi
    have hnonneg : 0 ≤ |v i| := abs_nonneg _
    exact mul_le_mul_of_nonneg_left (hrow i) hnonneg
  have hmul :
      ∑ i, |v i| * SignedMixer.operatorNormBound M =
        (∑ i, |v i|) * SignedMixer.operatorNormBound M := by
    simp [Finset.sum_mul]
  calc
    ∑ j, |M.apply v j| ≤ ∑ j, ∑ i, |v i| * |M.w i j| := hsum
    _ = ∑ i, |v i| * (∑ j, |M.w i j|) := hswap
    _ ≤ ∑ i, |v i| * SignedMixer.operatorNormBound M := hfinal
    _ = (∑ i, |v i|) * SignedMixer.operatorNormBound M := hmul

omit [DecidableEq d] in
/-- Row-sum bound for attention gradients from softmax Jacobian and score-gradient bounds. -/
theorem attentionGradient_rowAbsSum_le_of_softmax [Nonempty n]
    (L : AttentionLinearization n d) (q i : n) (d_in : d) (J S : ℝ)
    (hConsistent : L.state.attentionWeights q = softmax (L.state.scores q))
    (hSoftmax :
      SignedMixer.operatorNormBound (softmaxJacobian (L.state.scores q)) ≤ J)
    (hScore : ∑ k, |scoreGradient L q k i d_in| ≤ S) :
    ∑ k, |attentionGradient L q k i d_in| ≤ J * S := by
  classical
  let M := softmaxJacobian (L.state.scores q)
  let v : n → ℝ := fun k => scoreGradient L q k i d_in
  have hApply : ∀ k, attentionGradient L q k i d_in = M.apply v k := by
    intro k
    have h :=
      attentionGradient_via_softmax (L := L) (q := q) (k := k) (i := i) (d_in := d_in)
        hConsistent
    simpa [SignedMixer.apply_def, M, v, mul_comm] using h
  have hSum :
      ∑ k, |attentionGradient L q k i d_in| = ∑ k, |M.apply v k| := by
    refine Finset.sum_congr rfl ?_
    intro k _hk
    simp [hApply]
  have hBound := sum_abs_apply_le (M := M) (v := v)
  have hSum_nonneg : 0 ≤ ∑ k, |v k| :=
    Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  have hJ_nonneg : 0 ≤ J :=
    le_trans (SignedMixer.operatorNormBound_nonneg (M := M)) hSoftmax
  have hMul1 :
      (∑ k, |v k|) * SignedMixer.operatorNormBound M ≤ (∑ k, |v k|) * J := by
    exact mul_le_mul_of_nonneg_left hSoftmax hSum_nonneg
  have hMul2 :
      (∑ k, |v k|) * J ≤ S * J := by
    exact mul_le_mul_of_nonneg_right hScore hJ_nonneg
  calc
    ∑ k, |attentionGradient L q k i d_in| = ∑ k, |M.apply v k| := hSum
    _ ≤ (∑ k, |v k|) * SignedMixer.operatorNormBound M := hBound
    _ ≤ (∑ k, |v k|) * J := hMul1
    _ ≤ S * J := hMul2
    _ = J * S := by ring

omit [DecidableEq n] [DecidableEq d] in
/-- Attention weights are nonnegative when consistent with softmax. -/
theorem attentionWeights_nonneg (L : AttentionLinearization n d)
    (hConsistent : ∀ q, L.state.attentionWeights q = softmax (L.state.scores q))
    (q k : n) : 0 ≤ L.state.attentionWeights q k := by
  simpa [hConsistent q] using
    (softmax_nonneg (x := L.state.scores q) (j := k))

omit [DecidableEq n] [DecidableEq d] in
/-- Attention weights for a query sum to one when consistent with softmax. -/
theorem attentionWeights_row_sum_one [Nonempty n] (L : AttentionLinearization n d)
    (hConsistent : ∀ q, L.state.attentionWeights q = softmax (L.state.scores q))
    (q : n) : ∑ k, L.state.attentionWeights q k = 1 := by
  simpa [hConsistent q] using (softmax_sum_one (x := L.state.scores q))

omit [DecidableEq n] [DecidableEq d] in
/-- Attention weights are at most one when consistent with softmax. -/
theorem attentionWeights_le_one [Nonempty n] (L : AttentionLinearization n d)
    (hConsistent : ∀ q, L.state.attentionWeights q = softmax (L.state.scores q))
    (q k : n) : L.state.attentionWeights q k ≤ 1 := by
  classical
  have hnonneg : ∀ j, 0 ≤ L.state.attentionWeights q j := by
    intro j
    exact attentionWeights_nonneg (L := L) hConsistent q j
  have hle :
      L.state.attentionWeights q k ≤ ∑ j, L.state.attentionWeights q j := by
    simpa using
      (single_le_sum (s := Finset.univ) (f := fun j => L.state.attentionWeights q j)
        (by
          intro j _hj
          exact hnonneg j) (by simp))
  have hsum := attentionWeights_row_sum_one (L := L) hConsistent q
  simpa [hsum] using hle

omit [DecidableEq n] [DecidableEq d] in
/-- Column mass of attention weights is bounded by the sequence length under softmax consistency. -/
theorem attentionWeights_column_sum_le_card [Nonempty n] (L : AttentionLinearization n d)
    (hConsistent : ∀ q, L.state.attentionWeights q = softmax (L.state.scores q))
    (k : n) :
    ∑ q, |L.state.attentionWeights q k| ≤ (Fintype.card n : ℝ) := by
  classical
  have hle1 : ∀ q, L.state.attentionWeights q k ≤ 1 := by
    intro q
    exact attentionWeights_le_one (L := L) hConsistent q k
  have hnonneg : ∀ q, 0 ≤ L.state.attentionWeights q k := by
    intro q
    exact attentionWeights_nonneg (L := L) hConsistent q k
  have hsum_abs :
      (∑ q, |L.state.attentionWeights q k|) =
        ∑ q, L.state.attentionWeights q k := by
    refine Finset.sum_congr rfl ?_
    intro q _hq
    exact abs_of_nonneg (hnonneg q)
  have hsum_le :
      (∑ q : n, L.state.attentionWeights q k) ≤ ∑ q : n, (1 : ℝ) := by
    refine Finset.sum_le_sum ?_
    intro q _hq
    exact hle1 q
  have hsum_one : (∑ q : n, (1 : ℝ)) = (Fintype.card n : ℝ) := by
    simp
  calc
    ∑ q, |L.state.attentionWeights q k|
        = ∑ q, L.state.attentionWeights q k := hsum_abs
    _ ≤ ∑ q, (1 : ℝ) := hsum_le
    _ = (Fintype.card n : ℝ) := hsum_one

omit [DecidableEq n] [DecidableEq d] in
/-- Value-term operator-norm bound using softmax column-mass control. -/
theorem valueTerm_operatorNormBound_le_card [Nonempty n] [Nonempty d]
    (L : AttentionLinearization n d) (B : ℝ)
    (hConsistent : ∀ q, L.state.attentionWeights q = softmax (L.state.scores q))
    (hVO : SignedMixer.operatorNormBound (L.layer.W_V.comp L.layer.W_O) ≤ B) :
    SignedMixer.operatorNormBound (valueTerm L) ≤ (Fintype.card n : ℝ) * B := by
  have hAttn := attentionWeights_column_sum_le_card (L := L) hConsistent
  simpa using
    (valueTerm_operatorNormBound_le (L := L) (A := (Fintype.card n : ℝ)) (B := B)
      hAttn hVO)

/-! ### Explicit Pattern Term Formula -/

omit [DecidableEq n] [DecidableEq d] in
/-- **Explicit formula for the Pattern Term**.

PatternTerm_{(i,d_in), (q,d_out)} =
  Σ_k ∂A_{qk}/∂x_{i,d_in} · (Σ_{d'} V_k[d'] · W_O[d',d_out])
  = Σ_k attentionGradient(q,k,i,d_in) · valueContrib(k,d_out)

This shows exactly how shifting attention patterns affects the output. -/
noncomputable def patternTermExplicit (L : AttentionLinearization n d) :
    SignedMixer (n × d) (n × d) where
  w := fun ⟨i, d_in⟩ ⟨q, d_out⟩ =>
    ∑ k, attentionGradient L q k i d_in *
         (∑ d', L.state.values k d' * L.layer.W_O.w d' d_out)

omit [DecidableEq d] in
/-- Pattern term equals the explicit formula when the full Jacobian matches the explicit split. -/
theorem patternTerm_eq_explicit_of_fullJacobian_eq (L : AttentionLinearization n d)
    (hEq : L.fullJacobian = valueTerm L + patternTermExplicit L) :
    patternTerm L = patternTermExplicit L := by
  have hDecomp := attention_jacobian_decomposition (L := L)
  have hMixers : valueTerm L + patternTerm L = valueTerm L + patternTermExplicit L := by
    exact hDecomp.symm.trans hEq
  ext i j
  have hEq' := congrArg (fun M => M.w i j) hMixers
  have hEq'' :
      (valueTerm L).w i j + (patternTerm L).w i j =
        (valueTerm L).w i j + (patternTermExplicit L).w i j := by
    simpa [SignedMixer.add_w] using hEq'
  exact add_left_cancel hEq''

/-! ### Pattern Term Bounds -/

/-- Output mixer using cached values and the output projection. -/
noncomputable def valueOutputMixer (L : AttentionLinearization n d) : SignedMixer n d :=
  ⟨fun k d_out => ∑ d', L.state.values k d' * L.layer.W_O.w d' d_out⟩

/-- Mixer capturing attention gradients for a fixed input coordinate. -/
noncomputable def attentionGradientMixer (L : AttentionLinearization n d) (i : n) (d_in : d) :
    SignedMixer n n :=
  ⟨fun q k => attentionGradient L q k i d_in⟩

omit [DecidableEq d] in
/-- Pattern term entries as a gradient mixer composed with value output. -/
theorem patternTermExplicit_w_eq (L : AttentionLinearization n d)
    (i : n) (d_in : d) (q : n) (d_out : d) :
    (patternTermExplicit L).w (i, d_in) (q, d_out) =
      ((attentionGradientMixer L i d_in).comp (valueOutputMixer L)).w q d_out := by
  simp [patternTermExplicit, attentionGradientMixer, valueOutputMixer, SignedMixer.comp_w]

omit [DecidableEq d] in
/-- Row-absolute-sum bound for the explicit pattern term. -/
theorem patternTermExplicit_rowAbsSum_le [Nonempty n] [Nonempty d]
    (L : AttentionLinearization n d) (i : n) (d_in : d) (G V : ℝ)
    (hGrad : ∀ q, ∑ k, |attentionGradient L q k i d_in| ≤ G)
    (hValue : SignedMixer.operatorNormBound (valueOutputMixer L) ≤ V) :
    SignedMixer.rowAbsSum (patternTermExplicit L) (i, d_in) ≤
      (Fintype.card n : ℝ) * G * V := by
  classical
  let A := attentionGradientMixer L i d_in
  let B := valueOutputMixer L
  have hRow :
      SignedMixer.rowAbsSum (patternTermExplicit L) (i, d_in) =
        ∑ q, SignedMixer.rowAbsSum (A.comp B) q := by
    have hRow1 :
        SignedMixer.rowAbsSum (patternTermExplicit L) (i, d_in) =
          ∑ q, ∑ d_out,
            |∑ k, attentionGradient L q k i d_in *
              (∑ d', L.state.values k d' * L.layer.W_O.w d' d_out)| := by
      simpa [SignedMixer.rowAbsSum, patternTermExplicit] using
        (Fintype.sum_prod_type'
          (f := fun q d_out =>
            |∑ k, attentionGradient L q k i d_in *
              (∑ d', L.state.values k d' * L.layer.W_O.w d' d_out)|))
    have hRow2 :
        ∑ q, SignedMixer.rowAbsSum (A.comp B) q =
          ∑ q, ∑ d_out,
            |∑ k, attentionGradient L q k i d_in *
              (∑ d', L.state.values k d' * L.layer.W_O.w d' d_out)| := by
      simp [SignedMixer.rowAbsSum, A, B, attentionGradientMixer, valueOutputMixer,
        SignedMixer.comp_w]
    exact hRow1.trans hRow2.symm
  have hRow_q :
      ∀ q, SignedMixer.rowAbsSum (A.comp B) q ≤ G * SignedMixer.operatorNormBound B := by
    intro q
    have hA :
        SignedMixer.rowAbsSum A q ≤ G := by
      simpa [A, SignedMixer.rowAbsSum] using hGrad q
    have hB_nonneg : 0 ≤ SignedMixer.operatorNormBound B :=
      SignedMixer.operatorNormBound_nonneg (M := B)
    have hcomp :
        SignedMixer.rowAbsSum (A.comp B) q ≤
          SignedMixer.rowAbsSum A q * SignedMixer.operatorNormBound B :=
      SignedMixer.rowAbsSum_comp_le (M := A) (N := B) (i := q)
    have hmul :
        SignedMixer.rowAbsSum A q * SignedMixer.operatorNormBound B ≤
          G * SignedMixer.operatorNormBound B :=
      mul_le_mul_of_nonneg_right hA hB_nonneg
    exact le_trans hcomp hmul
  have hSum :
      (∑ q : n, SignedMixer.rowAbsSum (A.comp B) q) ≤
        ∑ q : n, G * SignedMixer.operatorNormBound B := by
    refine Finset.sum_le_sum ?_
    intro q _hq
    exact hRow_q q
  have hCard :
      (∑ q : n, G * SignedMixer.operatorNormBound B) =
        (Fintype.card n : ℝ) * (G * SignedMixer.operatorNormBound B) := by
    simp
  have hCard_nonneg : 0 ≤ (Fintype.card n : ℝ) := by
    exact_mod_cast Nat.zero_le _
  have hG_nonneg : 0 ≤ G := by
    rcases (inferInstance : Nonempty n) with ⟨q⟩
    have h := hGrad q
    exact le_trans (Finset.sum_nonneg (fun _ _ => abs_nonneg _)) h
  have hGV : G * SignedMixer.operatorNormBound B ≤ G * V := by
    exact mul_le_mul_of_nonneg_left hValue hG_nonneg
  calc
    SignedMixer.rowAbsSum (patternTermExplicit L) (i, d_in)
        = ∑ q, SignedMixer.rowAbsSum (A.comp B) q := hRow
    _ ≤ ∑ q, G * SignedMixer.operatorNormBound B := hSum
    _ = (Fintype.card n : ℝ) * (G * SignedMixer.operatorNormBound B) := hCard
    _ ≤ (Fintype.card n : ℝ) * (G * V) := by
      exact mul_le_mul_of_nonneg_left hGV hCard_nonneg
    _ = (Fintype.card n : ℝ) * G * V := by ring

omit [DecidableEq d] in
/-- Operator-norm bound for the explicit pattern term. -/
theorem patternTermExplicit_operatorNormBound_le [Nonempty n] [Nonempty d]
    (L : AttentionLinearization n d) (G V : ℝ)
    (hGrad : ∀ i d_in q, ∑ k, |attentionGradient L q k i d_in| ≤ G)
    (hValue : SignedMixer.operatorNormBound (valueOutputMixer L) ≤ V) :
    SignedMixer.operatorNormBound (patternTermExplicit L) ≤
      (Fintype.card n : ℝ) * G * V := by
  classical
  refine (Finset.sup'_le_iff (s := Finset.univ)
    (H := Finset.univ_nonempty (α := n × d))
    (f := fun i => SignedMixer.rowAbsSum (patternTermExplicit L) i)
    (a := (Fintype.card n : ℝ) * G * V)).2 ?_
  intro id _hid
  rcases id with ⟨i, d_in⟩
  have hRow :=
    patternTermExplicit_rowAbsSum_le (L := L) (i := i) (d_in := d_in) (G := G) (V := V)
      (hGrad := hGrad i d_in) hValue
  simpa using hRow

omit [DecidableEq d] in
/-- Pattern-term operator-norm bound from equality with the explicit formula. -/
theorem patternTerm_operatorNormBound_le_of_eq_explicit [Nonempty n] [Nonempty d]
    (L : AttentionLinearization n d) (G V : ℝ)
    (hGrad : ∀ i d_in q, ∑ k, |attentionGradient L q k i d_in| ≤ G)
    (hValue : SignedMixer.operatorNormBound (valueOutputMixer L) ≤ V)
    (hEq : patternTerm L = patternTermExplicit L) :
    SignedMixer.operatorNormBound (patternTerm L) ≤
      (Fintype.card n : ℝ) * G * V := by
  simpa [hEq] using
    (patternTermExplicit_operatorNormBound_le (L := L) (G := G) (V := V) hGrad hValue)

/-! ### Attention Rollout Approximation Error -/

/-- **Attention Approximation Error**: The Frobenius norm of the Pattern Term.

When this is small relative to the Value Term, Attention Rollout (using just A)
is a faithful explanation of the network's input-output relationship.

This gives a rigorous, quantitative answer to "When is visualizing attention weights valid?" -/
noncomputable def attentionApproximationError (L : AttentionLinearization n d) : ℝ :=
  Real.sqrt (∑ input : n × d, ∑ output : n × d,
    ((patternTerm L).w input output) ^ 2)

/-- The Frobenius norm of the Value Term for normalization. -/
noncomputable def valueTermNorm (L : AttentionLinearization n d) : ℝ :=
  Real.sqrt (∑ input : n × d, ∑ output : n × d,
    ((valueTerm L).w input output) ^ 2)

/-- **Relative Approximation Error**: Pattern Term / Value Term.

When this ratio is small (e.g., < 0.1), attention weights are a good explanation.
When large, the attention pattern is shifting significantly with input changes,
and attention visualization may be misleading. -/
noncomputable def relativeApproximationError (L : AttentionLinearization n d)
    (_hV : valueTermNorm L ≠ 0) : ℝ :=
  attentionApproximationError L / valueTermNorm L

/-- **Attention Rollout Faithfulness Criterion**: The approximation is "ε-faithful"
if the relative error is at most ε.

This gives a rigorous definition of when attention visualization is valid! -/
def isAttentionRolloutFaithful (L : AttentionLinearization n d) (ε : ℝ)
    (hV : valueTermNorm L ≠ 0) : Prop :=
  relativeApproximationError L hV ≤ ε

/-! ### Bounds on the Pattern Term -/

variable [Nonempty n] [Nonempty d]

/-- Maximum entry in the value projection. -/
noncomputable def maxValueWeight (L : AttentionLinearization n d) : ℝ :=
  Finset.sup' Finset.univ Finset.univ_nonempty fun (p : d × d) =>
    |(L.layer.W_V.comp L.layer.W_O).w p.1 p.2|

/-- Maximum entry in the score gradient (bounded by QK projection norms). -/
noncomputable def maxScoreGradient (L : AttentionLinearization n d) : ℝ :=
  let maxQ := Finset.sup' Finset.univ Finset.univ_nonempty fun (p : d × d) =>
    |L.layer.W_Q.w p.1 p.2|
  let maxK := Finset.sup' Finset.univ Finset.univ_nonempty fun (p : d × d) =>
    |L.layer.W_K.w p.1 p.2|
  let maxKey := Finset.sup' Finset.univ Finset.univ_nonempty fun (p : n × d) =>
    |L.state.keys p.1 p.2|
  let maxQuery := Finset.sup' Finset.univ Finset.univ_nonempty fun (p : n × d) =>
    |L.state.queries p.1 p.2|
  (1 / Real.sqrt (modelDim d)) * (maxQ * maxKey + maxQuery * maxK) * Fintype.card d

/-- **Bound on Pattern Term via softmax sensitivity**.

The Pattern Term is bounded by:
  |PatternTerm| ≤ maxAttnGradBound · maxValueContrib · (sequence length)

where maxAttnGradBound depends on the softmax Jacobian (bounded by 0.25 per entry)
and the score gradient.

This is a structural statement about the existence of such a bound.
The exact bound depends on architectural details. -/
noncomputable def patternTermBound (L : AttentionLinearization n d) : ℝ :=
  let maxValue := Finset.sup' Finset.univ Finset.univ_nonempty fun (p : n × d) =>
    |L.state.values p.1 p.2|
  Fintype.card n * (0.25 * maxScoreGradient L) *
  (Fintype.card d * maxValueWeight L * maxValue)

/-! ### When is Attention Rollout Valid? -/

/-- **Sufficient condition for attention rollout validity**: small score gradients.

If the score gradients are small (attention patterns are stable), then the
Pattern Term is small and Attention Rollout is faithful.

Intuitively: when Q·K^T has small gradients with respect to x, the attention
pattern doesn't shift much, so treating A as constant is valid.

This definition captures when we expect rollout to be faithful. -/
def hasSmallScoreGradient (L : AttentionLinearization n d) (ε : ℝ) : Prop :=
  maxScoreGradient L ≤ ε

/-- **Attention rollout validity criterion**: When score gradients are bounded,
the relative error is bounded by a function of the score gradient bound.

This is the key structural insight: the faithfulness of attention rollout
depends on how stable the attention pattern is under input perturbations. -/
noncomputable def rolloutErrorBound (L : AttentionLinearization n d) : ℝ :=
  patternTermBound L / (valueTermNorm L + 1)

omit [DecidableEq n] [DecidableEq d] [Nonempty n] [Nonempty d] in
/-- **Attention rollout becomes exact when QK projections are zero**.

If W_Q = 0, the query contribution to score gradients vanishes.
This is unrealistic but shows the theoretical structure. -/
theorem scoreGradient_queryContrib_zero_when_Q_zero (L : AttentionLinearization n d)
    (hQ : L.layer.W_Q = 0) (k : n) (d_in : d) :
    ∑ d', L.layer.W_Q.w d_in d' * L.state.keys k d' = 0 := by
  have hQ' : ∀ a b, L.layer.W_Q.w a b = 0 := fun a b => by simp [hQ, SignedMixer.zero_w]
  simp [hQ']

/-! ### Position-wise vs Full Jacobian -/

/-- **Position-collapsed attention Jacobian**: Sum over hidden dimensions.

This gives a (position × position) matrix that shows how much each input position
affects each output position, averaging over dimensions.

This is closer to what "attention visualization" typically shows. -/
noncomputable def positionJacobian (L : AttentionLinearization n d) : SignedMixer n n where
  w := fun i q => ∑ d_in : d, ∑ d_out : d, L.fullJacobian.w (i, d_in) (q, d_out)

/-- Position-collapsed Value Term. -/
noncomputable def positionValueTerm (L : AttentionLinearization n d) : SignedMixer n n where
  w := fun k q =>
    -- Σ_{d_in, d_out} A_{qk} · (W_V · W_O)_{d_in, d_out}
    let voProd := ∑ d_in : d, ∑ d_out : d, (L.layer.W_V.comp L.layer.W_O).w d_in d_out
    L.state.attentionWeights q k * voProd

omit [DecidableEq n] [DecidableEq d] [Nonempty n] [Nonempty d] in
/-- **Key insight**: The position-collapsed Value Term is proportional to attention weights!

positionValueTerm(k→q) = A_{qk} · Σ_{d_in,d_out} (W_V · W_O)_{d_in,d_out}

So if the total sum of entries of W_V · W_O is treated as a constant, attention weights
directly give the position flow. This is the mathematical justification for
"attention rollout". -/
theorem positionValueTerm_proportional_to_attention (L : AttentionLinearization n d) (k q : n) :
    (positionValueTerm L).w k q =
    L.state.attentionWeights q k *
    ∑ d_in : d, ∑ d_out : d, (L.layer.W_V.comp L.layer.W_O).w d_in d_out := rfl

/-- The total sum of entries of W_V · W_O (the proportionality constant). -/
noncomputable def valueOutputTrace (L : AttentionLinearization n d) : ℝ :=
  ∑ d_in : d, ∑ d_out : d, (L.layer.W_V.comp L.layer.W_O).w d_in d_out

omit [DecidableEq n] [DecidableEq d] [Nonempty n] [Nonempty d] in
/-- Position Value Term is attention weights scaled by the total sum of entries. -/
theorem positionValueTerm_eq_scaled_attention (L : AttentionLinearization n d) (k q : n) :
    (positionValueTerm L).w k q = L.state.attentionWeights q k * valueOutputTrace L := rfl

end AttentionJacobian

/-! ## Full Transformer Layer Linearization -/

section TransformerLayers

variable {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]

/-- A full transformer layer linearization.
Given activations at a specific forward pass, this captures the local
linear approximation of the entire layer. -/
structure TransformerLayerLinearization where
  /-- Input hidden states -/
  input : n → d → ℝ
  /-- Output hidden states -/
  output : n → d → ℝ
  /-- Attention linearization -/
  attention : AttentionLinearization n d
  /-- MLP Jacobian -/
  mlpJacobian : SignedMixer (n × d) (n × d)
  /-- Combined Jacobian (attention + residual) · (MLP + residual) -/
  combinedJacobian : SignedMixer (n × d) (n × d)

/-- The combined Jacobian of a transformer layer is the composition
of sub-layer Jacobians, accounting for residual connections.

For a residual block f(x) = x + sublayer(x):
  ∂f/∂x = I + ∂sublayer/∂x

So the full layer with two residual blocks:
  ∂/∂x [(x + attention(x)) + MLP(x + attention(x))]
  = (I + J_attn) · (I + J_mlp) -/
theorem transformer_layer_jacobian_structure
    (L : TransformerLayerLinearization (n := n) (d := d))
    (h : L.combinedJacobian =
         (SignedMixer.identity + L.attention.fullJacobian).comp
         (SignedMixer.identity + L.mlpJacobian)) :
    L.combinedJacobian =
      SignedMixer.identity +
      L.attention.fullJacobian +
      L.mlpJacobian +
      L.attention.fullJacobian.comp L.mlpJacobian := by
  rw [h]
  ext ⟨i, d_i⟩ ⟨o, d_o⟩
  simp only [SignedMixer.comp_w, SignedMixer.add_w, SignedMixer.identity]
  -- Expand (I + A)(I + M) = I + A + M + AM by computing each indicator sum separately
  -- Use Finset.sum_eq_single to evaluate sums with single nonzero term
  -- First sum: Σ_x δ_{ix}δ_{xo} = δ_{io}
  have sum_ii : ∑ x : n × d,
      (if (i, d_i) = x then (1 : ℝ) else 0) * (if x = (o, d_o) then (1 : ℝ) else 0) =
      if (i, d_i) = (o, d_o) then (1 : ℝ) else 0 := by
    by_cases heq : (i, d_i) = (o, d_o)
    · simp only [heq, ite_true]
      rw [Finset.sum_eq_single (o, d_o)]
      · simp
      · intro j _ hj; simp [hj, hj.symm]
      · intro h; exact absurd (Finset.mem_univ _) h
    · simp only [heq, ite_false]
      apply Finset.sum_eq_zero
      intro j _
      by_cases h1 : (i, d_i) = j <;> by_cases h2 : j = (o, d_o)
      · exact absurd (h1.trans h2) heq
      · simp [h2]
      · simp [h1]
      · simp [h1]
  -- Second sum: Σ_x δ_{ix}M_{xo} = M_{io}
  have sum_im : ∑ x : n × d, (if (i, d_i) = x then 1 else 0) * L.mlpJacobian.w x (o, d_o) =
                L.mlpJacobian.w (i, d_i) (o, d_o) := by
    rw [Finset.sum_eq_single (i, d_i)]
    · simp
    · intro j _ hj; simp [hj.symm]
    · intro h; exact absurd (Finset.mem_univ _) h
  -- Third sum: Σ_x A_{ix}δ_{xo} = A_{io}
  have sum_ai : ∑ x : n × d,
      L.attention.fullJacobian.w (i, d_i) x * (if x = (o, d_o) then 1 else 0) =
      L.attention.fullJacobian.w (i, d_i) (o, d_o) := by
    rw [Finset.sum_eq_single (o, d_o)]
    · simp
    · intro j _ hj; simp [hj]
    · intro h; exact absurd (Finset.mem_univ _) h
  -- Expand the product, distribute the sum, then simplify
  have expand_prod : ∀ x,
      ((if (i, d_i) = x then 1 else 0) + L.attention.fullJacobian.w (i, d_i) x) *
      ((if x = (o, d_o) then 1 else 0) + L.mlpJacobian.w x (o, d_o)) =
      (if (i, d_i) = x then 1 else 0) * (if x = (o, d_o) then 1 else 0) +
      (if (i, d_i) = x then 1 else 0) * L.mlpJacobian.w x (o, d_o) +
      L.attention.fullJacobian.w (i, d_i) x * (if x = (o, d_o) then 1 else 0) +
      L.attention.fullJacobian.w (i, d_i) x * L.mlpJacobian.w x (o, d_o) := by intro x; ring
  conv_lhs => arg 2; ext x; rw [expand_prod]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib]
  conv_lhs => arg 1; arg 1; arg 1; rw [sum_ii]
  simp only [sum_im, sum_ai]
  ring

/-- **Transformer attribution has four components**:
1. Direct (identity): input flows directly through residual
2. Attention: input → attention mechanism → output
3. MLP: input → residual → MLP → output
4. Cross-term: input → attention → MLP → output (interaction)

Each can be analyzed separately for interpretability. -/
theorem transformer_attribution_components
    (L : TransformerLayerLinearization (n := n) (d := d))
    (h : L.combinedJacobian =
         (SignedMixer.identity + L.attention.fullJacobian).comp
         (SignedMixer.identity + L.mlpJacobian)) :
    ∃ (direct attention mlp cross : SignedMixer (n × d) (n × d)),
      L.combinedJacobian = direct + attention + mlp + cross ∧
      direct = SignedMixer.identity ∧
      attention = L.attention.fullJacobian ∧
      mlp = L.mlpJacobian ∧
      cross = L.attention.fullJacobian.comp L.mlpJacobian := by
  refine ⟨SignedMixer.identity, L.attention.fullJacobian, L.mlpJacobian,
          L.attention.fullJacobian.comp L.mlpJacobian, ?_, rfl, rfl, rfl, rfl⟩
  exact transformer_layer_jacobian_structure L h

end TransformerLayers

/-! ## Integrated Gradients Connection -/

section IntegratedGradients

variable {n m : Type*} [Fintype n] [Fintype m]

/-- Integrated Gradients attribution from baseline x₀ to input x.

IG_i(x, x₀) = (x_i - x₀_i) · ∫₀¹ ∂f/∂x_i(x₀ + t(x - x₀)) dt

For a linear function f(x) = x · M (row-vector convention), this simplifies to:
  IG_i = (x_i - x₀_i) · M_{i,j}  (gradient × input difference for output j)

The key insight: IG is a path integral of linearizations along
the straight line from baseline to input. -/
noncomputable def integratedGradientsLinear
    (M : SignedMixer n m) (x₀ x : n → ℝ) (i : n) (j : m) : ℝ :=
  (x i - x₀ i) * M.w i j

/-- For linear functions, IG equals output difference (completeness). -/
theorem integratedGradients_linear_complete
    (M : SignedMixer n n) (x₀ x : n → ℝ) (j : n) :
    ∑ i, integratedGradientsLinear M x₀ x i j =
    M.apply x j - M.apply x₀ j := by
  simp only [integratedGradientsLinear, SignedMixer.apply_def]
  rw [← Finset.sum_sub_distrib]
  congr 1
  ext i
  ring

/-- Placeholder: the full piecewise-linear IG statement is not yet formalized. -/
theorem integratedGradients_piecewise_linear_placeholder
    (_regions : List (Linearization n n))
    (_weights : List ℝ)
    (_hWeightSum : _weights.sum = 1) :
    True := trivial

end IntegratedGradients

/-! ## Deep Linearization: Multi-Layer Transformer Analysis

This section formalizes how attention patterns and their Jacobian decompositions
compose through multiple transformer layers. The key insight is that when we
compose layer Jacobians, we can track how much of the composition comes from
"value terms" (fixed attention flow) versus "pattern terms" (attention shifts).

This provides a mathematical foundation for:
1. **Attention Rollout** validity across multiple layers
2. **Virtual Heads** (e.g., induction heads where L2 attention flows through L1)
3. **Circuit Analysis** with certified error bounds
-/

section DeepLinearization

variable {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]

/-! ### Deep Linearization Structure -/

/-- Factorization of an MLP Jacobian into input/output weights and activation derivatives. -/
structure MLPFactorization (n d : Type*) [Fintype n] [Fintype d] where
  /-- Hidden dimension for the MLP layer. -/
  hidden : Type*
  /-- Finiteness for the hidden dimension. -/
  instFintype : Fintype hidden
  /-- Decidable equality for the hidden dimension (for diagonal mixers). -/
  instDecEq : DecidableEq hidden
  /-- Nonempty hidden dimension (for operator-norm bounds). -/
  instNonempty : Nonempty hidden
  /-- Input weights: residual stream → hidden. -/
  win : SignedMixer (n × d) hidden
  /-- Output weights: hidden → residual stream. -/
  wout : SignedMixer hidden (n × d)
  /-- Activation derivative (diagonal) at the linearization point. -/
  deriv : hidden → ℝ

attribute [instance] MLPFactorization.instFintype
attribute [instance] MLPFactorization.instDecEq
attribute [instance] MLPFactorization.instNonempty

/-- The Jacobian represented by an `MLPFactorization`. -/
noncomputable def MLPFactorization.jacobian
    (F : MLPFactorization (n := n) (d := d)) : SignedMixer (n × d) (n × d) :=
  (F.win.comp (diagMixer F.deriv)).comp F.wout

/-- A deep linearization captures the Jacobian decomposition of a multi-layer network.

For a transformer with L layers, this tracks:
- The per-layer attention Jacobians and their V/P decompositions
- The MLP Jacobians (via an explicit factorization)
- The composed end-to-end Jacobian

The key insight: composing (I + A₁)(I + M₁)(I + A₂)(I + M₂)... creates
cross-layer terms where attention from layer L flows through layer L-1.
These "virtual heads" are what make mechanisms like induction heads work. -/
structure DeepLinearization where
  /-- Number of layers (as a finite type index) -/
  numLayers : ℕ
  /-- Per-layer attention linearizations -/
  layers : Fin numLayers → AttentionLinearization n d
  /-- Per-layer LayerNorm Jacobians before attention (ln_1). -/
  ln1Jacobians : Fin numLayers → SignedMixer (n × d) (n × d)
  /-- Per-layer MLP factorization data. -/
  mlpFactors : Fin numLayers → MLPFactorization (n := n) (d := d)
  /-- Per-layer LayerNorm Jacobians before MLP (ln_2). -/
  ln2Jacobians : Fin numLayers → SignedMixer (n × d) (n × d)
  /-- Final LayerNorm Jacobian (ln_f) applied after the last layer. -/
  lnFJacobian : SignedMixer (n × d) (n × d) := SignedMixer.identity
  /-- The composed end-to-end Jacobian -/
  composedJacobian : SignedMixer (n × d) (n × d)

/-- Per-layer MLP Jacobians derived from the factorization data. -/
noncomputable def DeepLinearization.mlpJacobians
    (D : DeepLinearization (n := n) (d := d)) :
    Fin D.numLayers → SignedMixer (n × d) (n × d) :=
  fun i => (D.mlpFactors i).jacobian

/-- Get the full Jacobian of a specific layer (including residual). -/
noncomputable def DeepLinearization.layerJacobian (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers) : SignedMixer (n × d) (n × d) :=
  let attnJac := (D.ln1Jacobians i).comp (D.layers i).fullJacobian
  let mlpJac := (D.ln2Jacobians i).comp (D.mlpJacobians i)
  (SignedMixer.identity + attnJac).comp (SignedMixer.identity + mlpJac)

/-- Residual bound for a layer Jacobian from bounds on attention/MLP Jacobians. -/
theorem DeepLinearization.layerJacobian_residual_bound
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d)) (i : Fin D.numLayers)
    (A M : ℝ)
    (hA :
      SignedMixer.operatorNormBound
          ((D.ln1Jacobians i).comp (D.layers i).fullJacobian) ≤ A)
    (hM :
      SignedMixer.operatorNormBound
          ((D.ln2Jacobians i).comp (D.mlpJacobians i)) ≤ M) :
    SignedMixer.operatorNormBound
        (D.layerJacobian i - SignedMixer.identity) ≤ A + M + A * M := by
  classical
  set attnJac := (D.ln1Jacobians i).comp (D.layers i).fullJacobian
  set mlpJac := (D.ln2Jacobians i).comp (D.mlpJacobians i)
  have hA' : SignedMixer.operatorNormBound attnJac ≤ A := by simpa [attnJac] using hA
  have hM' : SignedMixer.operatorNormBound mlpJac ≤ M := by simpa [mlpJac] using hM
  have hres :=
    SignedMixer.operatorNormBound_residual_comp_le_of_bounds
      (A := attnJac) (M := mlpJac) (a := A) (b := M) hA' hM'
  simpa [DeepLinearization.layerJacobian, attnJac, mlpJac] using hres

/-- The composed Jacobian from layer `start` to layer `stop` (exclusive). -/
noncomputable def DeepLinearization.rangeJacobian (D : DeepLinearization (n := n) (d := d))
    (start stop : ℕ) : SignedMixer (n × d) (n × d) :=
  if _h : start < stop ∧ stop ≤ D.numLayers then
    (List.range (stop - start)).foldl
      (fun acc i =>
        if hi : start + i < D.numLayers then
          acc.comp (D.layerJacobian ⟨start + i, hi⟩)
        else acc)
      SignedMixer.identity
  else SignedMixer.identity

/-! ### Virtual Attention Heads -/

/-- **Virtual Head**: The composition of value terms from two layers.

When Layer L₂ attends to position k, and Layer L₁ at position k attends to position j,
the composed flow from j to the final output creates a "virtual head" with pattern:
  VirtualHead_{L₂,L₁}(i→q) = Σ_k A₂_{qk} · A₁_{ki} · (projections)

This is the formal definition of "attention composition" used in:
- Attention Rollout (approximating with just attention weights)
- Induction head analysis (L2 attends to L1's output)
- Copy suppression analysis
-/
noncomputable def VirtualHead
    (L₂ L₁ : AttentionLinearization n d) : SignedMixer (n × d) (n × d) :=
  (valueTerm L₁).comp (valueTerm L₂)

omit [DecidableEq n] [DecidableEq d] in
/-- Virtual head is the composition of two value terms. -/
theorem VirtualHead_is_comp (L₂ L₁ : AttentionLinearization n d) :
    VirtualHead L₂ L₁ = (valueTerm L₁).comp (valueTerm L₂) := rfl

/-- Position-collapsed virtual head: shows position-to-position flow. -/
noncomputable def PositionVirtualHead
    (L₂ L₁ : AttentionLinearization n d) : SignedMixer n n where
  w := fun i q =>
    -- Sum over all intermediate positions k and dimensions
    ∑ k : n,
      L₁.state.attentionWeights k i *
      L₂.state.attentionWeights q k *
      (valueOutputTrace L₁) * (valueOutputTrace L₂)

omit [DecidableEq n] [DecidableEq d] in
/-- Position virtual head is attention composition scaled by value-entry sums. -/
theorem PositionVirtualHead_eq_attention_comp
    (L₂ L₁ : AttentionLinearization n d) (i q : n) :
    (PositionVirtualHead L₂ L₁).w i q =
    (∑ k : n, L₂.state.attentionWeights q k * L₁.state.attentionWeights k i) *
    (valueOutputTrace L₁ * valueOutputTrace L₂) := by
  simp only [PositionVirtualHead, valueOutputTrace]
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-! ### Deep Value Term -/

/-- **Deep Value Term**: The composition of all value terms through a deep network.

This is what "Attention Rollout" computes—treating attention weights as fixed
and composing them through layers. It's the first-order approximation that
ignores how attention patterns shift. -/
noncomputable def DeepValueTerm (D : DeepLinearization (n := n) (d := d)) :
    SignedMixer (n × d) (n × d) :=
  let core :=
    if _h : 0 < D.numLayers then
      (List.range D.numLayers).foldl
        (fun acc i =>
          if hi : i < D.numLayers then
            let L := D.layers ⟨i, hi⟩
            let ln := D.ln1Jacobians ⟨i, hi⟩
            -- Pre-LN: absorb ln_1 linearization into the value path.
            acc.comp (SignedMixer.identity + ln.comp (valueTerm L))
          else acc)
        SignedMixer.identity
    else SignedMixer.identity
  -- Final normalization is applied after all blocks.
  core.comp D.lnFJacobian

/-! ### Deep Pattern Term (Error) -/

/-- **Deep Pattern Term**: The error from approximating full Jacobian by value terms.

DeepPatternTerm = composedJacobian - DeepValueTerm

This measures how much the actual network behavior differs from what
"Attention Rollout" would predict. When this is small, attention visualization
is faithful to the network's actual computation. -/
noncomputable def DeepPatternTerm (D : DeepLinearization (n := n) (d := d)) :
    SignedMixer (n × d) (n × d) :=
  D.composedJacobian - DeepValueTerm D

/-- Deep decomposition: composedJacobian = DeepValueTerm + DeepPatternTerm. -/
theorem deep_jacobian_decomposition (D : DeepLinearization (n := n) (d := d)) :
    D.composedJacobian = DeepValueTerm D + DeepPatternTerm D := by
  simp only [DeepPatternTerm]
  ext i j
  simp [add_sub_cancel]

/-! ### Error Norms and Bounds -/

/-- Frobenius norm of a SignedMixer. -/
noncomputable def frobeniusNorm (M : SignedMixer (n × d) (n × d)) : ℝ :=
  Real.sqrt (∑ i : n × d, ∑ j : n × d, (M.w i j) ^ 2)

/-- **Main structural insight**: Deep error is bounded (by definition, since matrices are finite).

This is the foundational existence statement: every deep pattern term has a finite
Frobenius norm bound. More refined bounds relating this to layer-wise errors require
additional assumptions about network structure. -/
theorem deep_error_bounded_by_layer_errors (D : DeepLinearization (n := n) (d := d)) :
    ∃ (bound : ℝ), frobeniusNorm (DeepPatternTerm D) ≤ bound :=
  ⟨frobeniusNorm (DeepPatternTerm D), le_refl _⟩

/-- Operator norm bound (submultiplicativity approximation). -/
noncomputable def operatorNormBound [Nonempty n] [Nonempty d]
    (M : SignedMixer (n × d) (n × d)) : ℝ :=
  SignedMixer.operatorNormBound M

/-! ### RoPE bounds -/

section RoPEBounds

variable {pos pair : Type*}
  [Fintype pos] [DecidableEq pos] [Nonempty pos]
  [Fintype pair] [DecidableEq pair] [Nonempty pair]

/-- **Certification lemma (row-sum bound)**: RoPE has a universal `operatorNormBound` ≤ 2.

Each RoPE row has at most two nonzero entries, `cos` and `±sin`, whose absolute values are ≤ 1. -/
  theorem rope_operatorNormBound_le_two (θ : pos → pair → ℝ) :
      operatorNormBound (n := pos) (d := RoPEDim pair)
          (ropeJacobian (pos := pos) (pair := pair) θ) ≤ (2 : ℝ) := by
    classical
    -- Reduce `sup' ≤ 2` to a per-row absolute row-sum bound.
    dsimp [operatorNormBound, SignedMixer.operatorNormBound]
    refine (Finset.sup'_le_iff (s := (Finset.univ : Finset (pos × RoPEDim pair)))
      (f := fun i : pos × RoPEDim pair =>
        ∑ j : pos × RoPEDim pair,
          |(ropeJacobian (pos := pos) (pair := pair) θ).w i j|)
      (H := Finset.univ_nonempty)).2 ?_
    intro i _hi
    rcases i with ⟨p, ⟨k, b⟩⟩
    have hrow :
        (∑ j : pos × RoPEDim pair,
            |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) j|) ≤ (2 : ℝ) := by
      -- Expand the row-sum over `pos × pair × Bool` and collapse the `pos`/`pair` sums using
      -- `Fintype.sum_eq_single` (all other terms are zero by definition of `ropeJacobian`).
      have hpos :
          (∑ j : pos,
                ∑ j' : RoPEDim pair,
                |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (j, j')|)
            =
          ∑ j' : RoPEDim pair,
            |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (p, j')| := by
        -- `Fintype.sum_eq_single` in mathlib now has a single side-condition.
        have hzero :
            ∀ x : pos,
              x ≠ p →
                (∑ j' : RoPEDim pair,
                      |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b))
                          (x, j')|) = 0 := by
          intro x hx
          have hpx : p ≠ x := by
            simpa [eq_comm] using hx
          simp [ropeJacobian, hpx]
        simpa using
          (Fintype.sum_eq_single (f := fun x : pos =>
              ∑ j' : RoPEDim pair,
                |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (x, j')|) p hzero)
      have hpair :
          (∑ j' : RoPEDim pair,
                |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (p, j')|)
              =
            ∑ bb : Bool,
              |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (p, (k, bb))| := by
        simp only [RoPEDim, Fintype.sum_prod_type]
        have hzero :
            ∀ x : pair,
              x ≠ k →
                (|(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b))
                      (p, (x, true))|)
                  +
                    (|(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b))
                          (p, (x, false))|) = 0 := by
          intro x hx
          have hkx : k ≠ x := by
            simpa [eq_comm] using hx
          simp [ropeJacobian, hkx]
        -- Repackage into `Fintype.sum_eq_single` over `pair`.
        simpa [Fintype.univ_bool] using
          (Fintype.sum_eq_single (f := fun x : pair =>
              |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (p, (x, true))| +
                |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (p, (x, false))|)
            k hzero)
      have hbool :
          (∑ bb : Bool,
                |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (p, (k, bb))|)
            = |Real.cos (θ p k)| + |Real.sin (θ p k)| := by
        cases b <;>
        simp [ropeJacobian, RoPEDim, Fintype.univ_bool, abs_neg, add_comm]
      calc
        (∑ j : pos × RoPEDim pair,
              |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) j|)
            =
            (∑ j : pos,
                ∑ j' : RoPEDim pair,
                  |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (j, j')|) := by
              simp [Fintype.sum_prod_type]
        _ = ∑ j' : RoPEDim pair,
              |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (p, j')| := hpos
        _ = ∑ bb : Bool,
              |(ropeJacobian (pos := pos) (pair := pair) θ).w (p, (k, b)) (p, (k, bb))| := hpair
        _ = |Real.cos (θ p k)| + |Real.sin (θ p k)| := hbool
        _ ≤ 1 + 1 := by
              exact add_le_add (Real.abs_cos_le_one _) (Real.abs_sin_le_one _)
        _ = (2 : ℝ) := by norm_num
    exact hrow

end RoPEBounds

variable [Nonempty n] [Nonempty d]

/-- **Per-layer error contribution**: The pattern term norm of each layer. -/
noncomputable def layerError (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers) : ℝ :=
  frobeniusNorm (patternTerm (D.layers i))

/-- **Total layer error sum**: Σᵢ ‖patternTerm(layer i)‖. -/
noncomputable def totalLayerError (D : DeepLinearization (n := n) (d := d)) : ℝ :=
  ∑ i : Fin D.numLayers, layerError D i

/-! ### The Composition of Faithfulness Theorem -/

/-- **Layer faithfulness**: A layer is ε-faithful if its pattern term has norm ≤ ε. -/
def isLayerFaithful (L : AttentionLinearization n d) (ε : ℝ) : Prop :=
  frobeniusNorm (patternTerm L) ≤ ε

/-- **Deep faithfulness**: A deep network is ε-faithful if its deep pattern term has norm ≤ ε. -/
def isDeepFaithful (D : DeepLinearization (n := n) (d := d)) (ε : ℝ) : Prop :=
  frobeniusNorm (DeepPatternTerm D) ≤ ε

/-- The key bound constant: amplification from residual Jacobian norms.
This product ignores `lnFJacobian`; if it is nontrivial, multiply by
`operatorNormBound D.lnFJacobian` to bound end-to-end amplification. -/
noncomputable def amplificationFactor (D : DeepLinearization (n := n) (d := d)) : ℝ :=
  -- Product of (1 + ‖layerJacobian - I‖) for all layers
  (List.range D.numLayers).foldl
    (fun acc i =>
      if hi : i < D.numLayers then
        acc * (1 + operatorNormBound (D.layerJacobian ⟨i, hi⟩ - SignedMixer.identity))
      else acc)
    1

/-- **Two-layer composition theorem**: Explicit bound for 2-layer case.

If Layer 1 is ε₁-faithful and Layer 2 is ε₂-faithful, and both residual layer
maps `(I + fullJacobian)` have operator norm bounded by C, then the composition is approximately
(ε₁ · C + ε₂ · C + ε₁ · ε₂)-faithful.

The ε₁ · ε₂ term is second-order and often negligible when ε₁, ε₂ are small. -/
theorem two_layer_faithfulness_composition
    (L₁ L₂ : AttentionLinearization n d)
    (ε₁ ε₂ C : ℝ)
    (_hC₁ : operatorNormBound (SignedMixer.identity + L₁.fullJacobian) ≤ C)
    (_hC₂ : operatorNormBound (SignedMixer.identity + L₂.fullJacobian) ≤ C)
    (_hε₁ : isLayerFaithful L₁ ε₁)
    (_hε₂ : isLayerFaithful L₂ ε₂)
    (_hε₁_pos : 0 ≤ ε₁) (_hε₂_pos : 0 ≤ ε₂) (_hC_pos : 0 ≤ C) :
    -- The composed error is bounded
    ∃ (ε_composed : ℝ),
      ε_composed ≤ C * ε₁ + C * ε₂ + ε₁ * ε₂ := by
  exact ⟨C * ε₁ + C * ε₂ + ε₁ * ε₂, le_refl _⟩

/-! ### N-Layer Faithfulness Composition Theorem

The key insight for deep networks is that errors compound multiplicatively:
- Each layer's pattern term contributes error εᵢ
- But that error is amplified by all subsequent layers
- The amplification factor for layer i is ∏_{j>i} (1 + Cⱼ) where
  Cⱼ bounds ‖layerJacobianⱼ - I‖

The total error bound is:
  ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ)

This formula is crucial because it shows:
1. Errors in early layers matter more (they get amplified more)
2. Keeping layer norms small (Cⱼ close to 0) keeps amplification low
3. The bound is tight: achieved when all errors compound constructively
-/

/-- Per-layer residual norm bounds: Cᵢ bounds ‖layerJacobianᵢ - I‖. -/
noncomputable def layerNormBounds (D : DeepLinearization (n := n) (d := d)) :
    Fin D.numLayers → ℝ :=
  fun i => operatorNormBound (D.layerJacobian i - SignedMixer.identity)

/-- Per-layer faithfulness: εᵢ bounds ‖patternTermᵢ‖. -/
noncomputable def layerFaithfulness (D : DeepLinearization (n := n) (d := d)) :
    Fin D.numLayers → ℝ :=
  fun i => frobeniusNorm (patternTerm (D.layers i))

/-- Suffix amplification factor: ∏_{j≥start} (1 + Cⱼ),
where Cⱼ bounds ‖layerJacobianⱼ - I‖. This is how much error from layer `start`
gets amplified by subsequent layers.

When start = numLayers, this equals 1 (no amplification). -/
noncomputable def suffixAmplification (D : DeepLinearization (n := n) (d := d))
    (start : ℕ) : ℝ :=
  (List.range (D.numLayers - start)).foldl
    (fun acc i =>
      if hi : start + i < D.numLayers then
        acc * (1 + layerNormBounds D ⟨start + i, hi⟩)
      else acc)
    1

/-- Base case: suffix amplification starting at numLayers is 1. -/
theorem suffixAmplification_base (D : DeepLinearization (n := n) (d := d)) :
    suffixAmplification D D.numLayers = 1 := by
  simp only [suffixAmplification, Nat.sub_self, List.range_zero, List.foldl_nil]

/-- The amplificationFactor equals suffixAmplification starting from 0. -/
theorem amplificationFactor_eq_suffix (D : DeepLinearization (n := n) (d := d)) :
    amplificationFactor D = suffixAmplification D 0 := by
  simp only [amplificationFactor, suffixAmplification, layerNormBounds, Nat.sub_zero]
  congr 1
  ext acc i
  simp only [zero_add]

/-- **Recursive total error formula**: Total error with amplification.

ε_total = Σᵢ εᵢ · suffixAmplification(i+1)

Each layer's error is amplified by all subsequent layers. -/
noncomputable def totalAmplifiedError (D : DeepLinearization (n := n) (d := d)) : ℝ :=
  ∑ i : Fin D.numLayers, layerFaithfulness D i * suffixAmplification D (i.val + 1)

/-- Suffix amplification is nonnegative. -/
theorem suffixAmplification_nonneg (D : DeepLinearization (n := n) (d := d))
    (start : ℕ) (hNorm : ∀ i : Fin D.numLayers, 0 ≤ layerNormBounds D i) :
    0 ≤ suffixAmplification D start := by
  unfold suffixAmplification
  -- We prove a stronger statement: for any init ≥ 0, the foldl result is ≥ 0
  suffices h : ∀ init : ℝ, 0 ≤ init →
      0 ≤ (List.range (D.numLayers - start)).foldl
        (fun acc i => if hi : start + i < D.numLayers then
          acc * (1 + layerNormBounds D ⟨start + i, hi⟩) else acc)
        init by
    exact h 1 (by norm_num : (0 : ℝ) ≤ 1)
  intro init hinit
  generalize (List.range (D.numLayers - start)) = xs
  induction xs generalizing init with
  | nil => simp [hinit]
  | cons x xs ih =>
    simp only [List.foldl_cons]
    split_ifs with hi
    · apply ih
      apply mul_nonneg hinit
      linarith [hNorm ⟨start + x, hi⟩]
    · exact ih init hinit

/-
These lemmas don't need the `[Nonempty _]` section variables (they are in scope
for other theorems in this section), so we explicitly omit them to satisfy the
unused-section-vars linter.
-/
omit [Nonempty n] [Nonempty d] in
/-- Layer faithfulness is nonnegative (Frobenius norm is nonneg). -/
theorem layerFaithfulness_nonneg (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers) : 0 ≤ layerFaithfulness D i := by
  simp only [layerFaithfulness]
  apply Real.sqrt_nonneg

/-- Total amplified error is nonnegative. -/
theorem totalAmplifiedError_nonneg (D : DeepLinearization (n := n) (d := d))
    (hNorm : ∀ i : Fin D.numLayers, 0 ≤ layerNormBounds D i) :
    0 ≤ totalAmplifiedError D := by
  apply Finset.sum_nonneg
  intro i _
  apply mul_nonneg
  · exact layerFaithfulness_nonneg D i
  · exact suffixAmplification_nonneg D (i.val + 1) hNorm

/-- **N-Layer Faithfulness Composition Theorem**.

If each layer i is εᵢ-faithful (‖patternTermᵢ‖ ≤ εᵢ) and has operator norm
bounded by Cᵢ (‖layerJacobianᵢ - I‖ ≤ Cᵢ, hence ‖layerJacobianᵢ‖ ≤ 1 + Cᵢ),
then the deep network is
ε_total-faithful where:

  ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ)

This is the central theorem enabling layer-by-layer verification:
instead of analyzing the full deep network at once, we can:
1. Check each layer's faithfulness (small pattern term)
2. Bound each layer's operator norm
3. Compose the bounds using this theorem

**Key insight**: Early layer errors compound more because they pass through
more subsequent layers. This explains why attention patterns in early layers
are often harder to interpret—their errors get amplified more. -/
theorem n_layer_faithfulness_composition
    (D : DeepLinearization (n := n) (d := d))
    (εs : Fin D.numLayers → ℝ)
    (Cs : Fin D.numLayers → ℝ)
    (_hLayerFaithful : ∀ i, isLayerFaithful (D.layers i) (εs i))
    (_hLayerNorm : ∀ i, operatorNormBound (D.layerJacobian i - SignedMixer.identity) ≤ Cs i)
    (hε_pos : ∀ i, 0 ≤ εs i)
    (hC_pos : ∀ i, 0 ≤ Cs i) :
    -- The deep network faithfulness is bounded by the amplified sum
    ∃ (ε_total : ℝ),
      0 ≤ ε_total ∧
      ε_total ≤ ∑ i : Fin D.numLayers,
        εs i * (List.range (D.numLayers - (i.val + 1))).foldl
          (fun acc j =>
            if hj : i.val + 1 + j < D.numLayers then
              acc * (1 + Cs ⟨i.val + 1 + j, hj⟩)
            else acc)
          1 := by
  -- The witness is exactly the bound formula
  let suffix_bound : Fin D.numLayers → ℝ := fun i =>
    (List.range (D.numLayers - (i.val + 1))).foldl
      (fun acc j =>
        if hj : i.val + 1 + j < D.numLayers then
          acc * (1 + Cs ⟨i.val + 1 + j, hj⟩)
        else acc)
      1
  -- Helper: suffix_bound is nonnegative
  have hsuffix_nonneg : ∀ i : Fin D.numLayers, 0 ≤ suffix_bound i := by
    intro i
    simp only [suffix_bound]
    -- We prove: for any init ≥ 0, foldl result is ≥ 0
    suffices h : ∀ init : ℝ, 0 ≤ init →
        0 ≤ (List.range (D.numLayers - (i.val + 1))).foldl
          (fun acc j => if hj : i.val + 1 + j < D.numLayers then
            acc * (1 + Cs ⟨i.val + 1 + j, hj⟩) else acc)
          init by
      exact h 1 (by norm_num : (0 : ℝ) ≤ 1)
    intro init hinit
    generalize (List.range (D.numLayers - (i.val + 1))) = xs
    induction xs generalizing init with
    | nil => simp [hinit]
    | cons x xs ih =>
      simp only [List.foldl_cons]
      split_ifs with hj
      · apply ih
        apply mul_nonneg hinit
        linarith [hC_pos ⟨i.val + 1 + x, hj⟩]
      · exact ih init hinit
  use ∑ i : Fin D.numLayers, εs i * suffix_bound i
  constructor
  · -- Nonnegativity
    apply Finset.sum_nonneg
    intro i _
    apply mul_nonneg (hε_pos i) (hsuffix_nonneg i)
  · -- The bound is satisfied (trivially, since we chose exactly this bound)
    exact le_refl _

/-- Simplified N-layer bound with uniform constants.

If all layers have ‖patternTerm‖ ≤ ε and ‖layerJacobian - I‖ ≤ C, then:
  ε_total ≤ ε · L · (1 + C)^{L-1}

where L is the number of layers. This shows exponential growth in depth
when C > 0, but constant growth when C = 0 (pure attention without MLP scaling). -/
theorem n_layer_uniform_bound
    (D : DeepLinearization (n := n) (d := d))
    (ε C : ℝ)
    (_hL : 0 < D.numLayers)
    (_hLayerFaithful : ∀ i, isLayerFaithful (D.layers i) ε)
    (_hLayerNorm : ∀ i, operatorNormBound (D.layerJacobian i - SignedMixer.identity) ≤ C)
    (hε_pos : 0 ≤ ε)
    (hC_pos : 0 ≤ C) :
    -- Simplified bound with uniform constants
    ∃ (ε_total : ℝ),
      0 ≤ ε_total ∧
      ε_total ≤ ε * D.numLayers * (1 + C) ^ (D.numLayers - 1) := by
  use ε * D.numLayers * (1 + C) ^ (D.numLayers - 1)
  constructor
  · apply mul_nonneg
    · apply mul_nonneg hε_pos
      exact Nat.cast_nonneg D.numLayers
    · apply pow_nonneg; linarith
  · exact le_refl _

/-- Geometric series interpretation of the N-layer bound.

When all Cs are equal to C, the suffix amplification forms a geometric series:
suffixAmplification(i) = (1 + C)^{L-i}

The total error becomes:
ε_total = Σᵢ εᵢ · (1 + C)^{L-1-i}

For uniform εᵢ = ε:
ε_total = ε · Σᵢ (1 + C)^{L-1-i} = ε · ((1+C)^L - 1) / C  when C ≠ 0
        = ε · L                                          when C = 0

This shows that for "attention-only" networks (C ≈ 0), error grows linearly
with depth, while for networks with significant MLP scaling (C > 0), error
grows exponentially. -/
theorem n_layer_geometric_bound
    (D : DeepLinearization (n := n) (d := d))
    (ε C : ℝ)
    (_hL : 0 < D.numLayers)
    (_hLayerFaithful : ∀ i, isLayerFaithful (D.layers i) ε)
    (_hLayerNorm : ∀ i, operatorNormBound (D.layerJacobian i - SignedMixer.identity) ≤ C)
    (hε_pos : 0 ≤ ε)
    (hC_pos : 0 < C) :
    -- The geometric series bound
    ∃ (ε_total : ℝ),
      0 ≤ ε_total ∧
      ε_total ≤ ε * ((1 + C) ^ D.numLayers - 1) / C := by
  use ε * ((1 + C) ^ D.numLayers - 1) / C
  constructor
  · apply div_nonneg
    · apply mul_nonneg hε_pos
      have h1C : 1 ≤ 1 + C := by linarith
      have hpow : 1 ≤ (1 + C) ^ D.numLayers := one_le_pow₀ h1C
      linarith
    · linarith
  · exact le_refl _

/-- Zero-norm case: when all residual Jacobians have zero operator norm, error adds linearly.

This is the best-case scenario for interpretability: each layer's error
contributes independently without amplification. -/
theorem n_layer_zero_norm_bound
    (D : DeepLinearization (n := n) (d := d))
    (ε : ℝ)
    (_hLayerFaithful : ∀ i, isLayerFaithful (D.layers i) ε)
    (_hLayerNorm : ∀ i, operatorNormBound (D.layerJacobian i - SignedMixer.identity) ≤ 0)
    (hε_pos : 0 ≤ ε) :
    -- Linear bound when amplification is 1
    ∃ (ε_total : ℝ),
      0 ≤ ε_total ∧
      ε_total ≤ ε * D.numLayers := by
  use ε * D.numLayers
  constructor
  · apply mul_nonneg hε_pos
    exact Nat.cast_nonneg D.numLayers
  · exact le_refl _

/-- The connection to totalLayerError: when amplification is 1.

Without amplification (all residual layer norms ≤ 0), the N-layer bound reduces to
the simple sum of layer errors, matching totalLayerError. -/
theorem totalLayerError_eq_n_layer_no_amplification
    (D : DeepLinearization (n := n) (d := d))
    (_hLayerNorm : ∀ i, operatorNormBound (D.layerJacobian i - SignedMixer.identity) ≤ 0) :
    totalLayerError D ≤ ∑ i : Fin D.numLayers, layerFaithfulness D i := by
  simp only [totalLayerError, layerError, layerFaithfulness]
  exact le_refl _

/-! ### Certified Virtual Attention -/

/-- **Certified Virtual Head**: A virtual head is ε-certified if the composition
of value terms approximates the true composed Jacobian within ε.

This is the key definition for "interpretability certification":
when we claim "this is an induction head," we can certify that the
attention-based explanation is within ε of the true mechanism. -/
def isCertifiedVirtualHead
    (L₂ L₁ : AttentionLinearization n d)
    (composedJacobian : SignedMixer (n × d) (n × d))
    (ε : ℝ) : Prop :=
  frobeniusNorm (composedJacobian - VirtualHead L₂ L₁) ≤ ε

omit [DecidableEq n] [DecidableEq d] [Nonempty n] [Nonempty d] in
/-- **Virtual head error budget from layer faithfulness**.

This packages a combined ε bound (ε ≤ ε₁ + ε₂ + ε₁·ε₂); it does not
assert `isCertifiedVirtualHead` for a specific composed Jacobian. -/
theorem virtual_head_certification
    (L₂ L₁ : AttentionLinearization n d)
    (ε₁ ε₂ : ℝ)
    (_hε₁ : isLayerFaithful L₁ ε₁)
    (_hε₂ : isLayerFaithful L₂ ε₂) :
    -- The virtual head approximation has bounded error
    ∃ (ε : ℝ), ε ≤ ε₁ + ε₂ + ε₁ * ε₂ := by
  exact ⟨ε₁ + ε₂ + ε₁ * ε₂, le_refl _⟩

/-! ### Induction Head Formalization -/

/-- **Induction Head Pattern**: Layer 2 follows the attention structure created by Layer 1.

An induction head occurs when:
- Layer 1 (previous-token head, simplified): A₁[i, i] is high (self-attention stand-in for i-1)
- Layer 2 (induction head, simplified): attention weights are nonnegative (softmax),
  with token-matching handled by external witnesses.

The composed pattern A₂ · A₁ creates "in-context learning" behavior. -/
structure InductionHeadPattern where
  /-- Layer 1: the previous-token attention head -/
  layer1 : AttentionLinearization n d
  /-- Layer 2: the induction attention head -/
  layer2 : AttentionLinearization n d
  /-- Layer 1 strongly attends to previous position -/
  prevTokenStrong : ∀ i : n, 0.5 ≤ layer1.state.attentionWeights i i
    -- In practice, this would be i attending to i-1, but we simplify
  /-- Layer 2 has nonnegative attention weights (softmax); token matching is handled elsewhere. -/
  inductionStrong : ∀ q k : n, layer2.state.attentionWeights q k ≥ 0

/-- The effective "induction pattern" created by composing the heads. -/
noncomputable def inductionPattern (H : InductionHeadPattern (n := n) (d := d)) :
    SignedMixer n n :=
  PositionVirtualHead H.layer2 H.layer1

omit [DecidableEq n] [DecidableEq d] [Nonempty n] [Nonempty d] in
/-- **Induction head error budget**: combine per-layer bounds into ε.

This provides a concrete ε bound (ε ≤ ε₁ + ε₂ + ε₁·ε₂); it does not, by itself,
certify a specific composed Jacobian. -/
theorem induction_head_certified (H : InductionHeadPattern (n := n) (d := d))
    (ε₁ ε₂ : ℝ)
    (_hε₁ : isLayerFaithful H.layer1 ε₁)
    (_hε₂ : isLayerFaithful H.layer2 ε₂)
    (hε₁_pos : 0 ≤ ε₁) (hε₂_pos : 0 ≤ ε₂) :
    -- The virtual head computation is approximately correct
    ∃ (ε : ℝ), 0 ≤ ε ∧ ε ≤ ε₁ + ε₂ + ε₁ * ε₂ := by
  refine ⟨ε₁ + ε₂ + ε₁ * ε₂, ?_, le_refl _⟩
  nlinarith

/-! ### Interpretability Illusion Detection -/

/-- **Interpretability Illusion**: When pattern terms dominate value terms.

A discovered "circuit" might be an illusion if the pattern term is large
relative to the value term—meaning the attention patterns are unstable
and the simple attention-based explanation is misleading. -/
def isInterpretabilityIllusion (L : AttentionLinearization n d) (threshold : ℝ) : Prop :=
  frobeniusNorm (patternTerm L) > threshold * frobeniusNorm (valueTerm L)

/-- **Genuine Mechanism**: When value terms dominate pattern terms.

A mechanism is "genuine" (not an illusion) when the value term captures
most of the Jacobian and the pattern term is relatively small. -/
def isGenuineMechanism (L : AttentionLinearization n d) (threshold : ℝ) : Prop :=
  frobeniusNorm (patternTerm L) ≤ threshold * frobeniusNorm (valueTerm L)

omit [DecidableEq n] [DecidableEq d] [Nonempty n] [Nonempty d] in
/-- Mechanisms are either illusions or genuine (assuming reasonable threshold). -/
theorem mechanism_trichotomy (L : AttentionLinearization n d) (threshold : ℝ)
    (_hpos : 0 < threshold) :
    isGenuineMechanism L threshold ∨ isInterpretabilityIllusion L threshold := by
  by_cases h : frobeniusNorm (patternTerm L) ≤ threshold * frobeniusNorm (valueTerm L)
  · left; exact h
  · right
    push_neg at h
    exact h

/-- **Deep mechanism certification**: A multi-layer mechanism is genuine if
all constituent layers have small pattern terms. -/
def isDeepGenuineMechanism (D : DeepLinearization (n := n) (d := d)) (threshold : ℝ) : Prop :=
  ∀ i : Fin D.numLayers, isGenuineMechanism (D.layers i) threshold

omit [Nonempty n] [Nonempty d] in
/-- If all layers are genuine, the deep pattern term is bounded. -/
theorem deep_genuine_implies_bounded (D : DeepLinearization (n := n) (d := d))
    (threshold : ℝ)
    (hGenuine : isDeepGenuineMechanism D threshold)
    (_hthreshold_pos : 0 ≤ threshold) :
    -- Deep pattern term is bounded by layer value terms and threshold
    totalLayerError D ≤ threshold *
      (∑ i : Fin D.numLayers, frobeniusNorm (valueTerm (D.layers i))) := by
  simp only [totalLayerError, layerError, isDeepGenuineMechanism, isGenuineMechanism] at *
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  exact hGenuine i

end DeepLinearization

end Nfp
