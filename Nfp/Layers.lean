-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Nfp.Prob
import Nfp.Mixer
import Nfp.Uniqueness

/-!
# Neural Network Layer Mixers

This module formalizes common neural network layer operations as mixers,
establishing the connection between abstract row-stochastic operators and
concrete NN architectures. This enables applying the tracer uniqueness and
attribution theorems to real neural network interpretation.

## Main definitions

* `Mixer.identity` – identity/skip connection
* `Mixer.attention` – attention mechanism as a mixer
* `Mixer.selfAttention` – self-attention variant
* `Mixer.residual` – residual connection combining identity with transform

## Key theorems

* `Mixer.identity_comp` – identity is a left/right unit for composition
* `Mixer.comp_assoc` – composition is associative
* `effectiveAttention_normalized` – attention rollout forms valid distributions
* `Mixer.push_comp` – pushing through composition equals sequential pushing

## Neural network interpretation

The key insight is that many NN operations can be viewed as row-stochastic
operators when considering how "importance" or "relevance" flows:

- Attention: importance flows according to attention weights
- Skip connections: importance passes through unchanged
- Residual: weighted combination of skip and transform

## References

* Abnar & Zuidema: "Quantifying Attention Flow in Transformers" (2020)
-/

namespace Nfp

open scoped BigOperators
open Finset

/-! ## Identity mixer -/

section Identity

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- The identity mixer: each source routes entirely to itself. -/
noncomputable def Mixer.identity : Mixer S S where
  w := fun i j => if i = j then 1 else 0
  row_sum_one := by
    classical
    intro i
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]

@[simp] lemma Mixer.identity_diag (i : S) : Mixer.identity.w i i = 1 := by
  simp [Mixer.identity]

@[simp] lemma Mixer.identity_off_diag {i j : S} (h : i ≠ j) :
    Mixer.identity.w i j = 0 := by
  simp [Mixer.identity, h]

/-- Identity is a left unit for mixer composition. -/
@[simp] theorem Mixer.identity_comp (M : Mixer S S) :
    Mixer.identity.comp M = M := by
  ext i k
  simp only [Mixer.comp, Mixer.identity]
  classical
  simp only [ite_mul, one_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]

/-- Identity is a right unit for mixer composition. -/
@[simp] theorem Mixer.comp_identity (M : Mixer S S) :
    M.comp Mixer.identity = M := by
  ext i k
  simp only [Mixer.comp, Mixer.identity]
  classical
  simp only [mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]

end Identity

/-! ## Mixer composition is associative -/

section Associativity

variable {S T U V : Type*}
         [Fintype S] [Fintype T] [Fintype U] [Fintype V]

/-- Mixer composition is associative. -/
theorem Mixer.comp_assoc (M : Mixer S T) (N : Mixer T U) (P : Mixer U V) :
    (M.comp N).comp P = M.comp (N.comp P) := by
  ext i l
  simp only [Mixer.comp, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  simp_rw [mul_assoc]

end Associativity

/-! ## Attention as a mixer -/

section Attention

variable {Query Key : Type*} [Fintype Query] [Fintype Key]

/-- Attention weights derived from query-key scores.
Given attention scores `α : Query → Key → NNReal` that are row-normalized
(each query's weights over keys sum to 1), this produces a mixer. -/
noncomputable def Mixer.attention
    (α : Query → Key → NNReal)
    (hα : ∀ q, (∑ k, α q k) = 1) : Mixer Query Key where
  w := α
  row_sum_one := hα

/-- Self-attention: queries and keys are the same set of positions. -/
noncomputable def Mixer.selfAttention {Pos : Type*} [Fintype Pos]
    (α : Pos → Pos → NNReal)
    (hα : ∀ p, (∑ p', α p p') = 1) : Mixer Pos Pos :=
  Mixer.attention α hα

/-- Attention is supported on positions with nonzero attention weight. -/
lemma Mixer.attention_supported {Query Key : Type*} [Fintype Query] [Fintype Key]
    (α : Query → Key → NNReal)
    (hα : ∀ q, (∑ k, α q k) = 1) :
    Mixer.supported (Mixer.attention α hα) (fun q k => α q k ≠ 0) := by
  intro q k hne
  by_cases hzero : α q k = 0
  · simp [Mixer.attention, hzero]
  · exact (hne hzero).elim

end Attention

/-! ## Skip/Residual connections -/

section Residual

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- Residual mixer: mixes identity with another mixer using coefficient `c ∈ [0,1]`.
This models skip connections: `output = c * identity + (1-c) * transform`. -/
noncomputable def Mixer.residual (M : Mixer S S) (c : NNReal) (hc : c ≤ 1) : Mixer S S where
  w := fun i j => c * (if i = j then 1 else 0) + (1 - c) * M.w i j
  row_sum_one := by
    classical
    intro i
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
    simp only [M.row_sum_one, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte, mul_one]
    rw [add_comm]
    exact tsub_add_cancel_of_le hc

/-- A pure skip connection is the identity mixer. -/
lemma Mixer.residual_one (M : Mixer S S) :
    Mixer.residual M 1 le_rfl = Mixer.identity := by
  ext i j
  simp [Mixer.residual, Mixer.identity]

/-- No skip connection passes through the transform entirely. -/
lemma Mixer.residual_zero (M : Mixer S S) :
    Mixer.residual M 0 (zero_le _) = M := by
  ext i j
  simp [Mixer.residual]

/-- Residual weight decomposition: the weight splits into identity and transform parts. -/
lemma Mixer.residual_w (M : Mixer S S) (c : NNReal) (hc : c ≤ 1) (i j : S) :
    (Mixer.residual M c hc).w i j =
      c * Mixer.identity.w i j + (1 - c) * M.w i j := by
  simp only [Mixer.residual, Mixer.identity]

end Residual

/-! ## Attention flow composition theorems -/

section AttentionFlow

variable {Pos : Type*} [Fintype Pos] [DecidableEq Pos]

/-- Attention flow: composition of attention matrices across layers.
This captures how attention "flows" across layers in list order (row-vector convention).

This is the formal version of "attention rollout" from Abnar & Zuidema (2020). -/
noncomputable def attentionFlow (layers : List (Mixer Pos Pos)) : Mixer Pos Pos :=
  layers.foldl Mixer.comp Mixer.identity

/-- Single layer attention flow is identity composed with the layer. -/
lemma attentionFlow_singleton (M : Mixer Pos Pos) :
    attentionFlow [M] = Mixer.identity.comp M := rfl

/-- Empty attention flow is identity. -/
@[simp] lemma attentionFlow_nil :
    attentionFlow (Pos := Pos) [] = Mixer.identity := rfl

/-- Attention flow of a single layer simplifies to just the layer. -/
@[simp] lemma attentionFlow_singleton' (M : Mixer Pos Pos) :
    attentionFlow [M] = M := by
  simp [attentionFlow_singleton]

/-- Two-layer attention flow is just composition. -/
lemma attentionFlow_two (M₁ M₂ : Mixer Pos Pos) :
    attentionFlow [M₁, M₂] = M₁.comp M₂ := by
  simp [attentionFlow]

/-- Three-layer attention flow is associative composition. -/
lemma attentionFlow_three (M₁ M₂ M₃ : Mixer Pos Pos) :
    attentionFlow [M₁, M₂, M₃] = (M₁.comp M₂).comp M₃ := by
  simp [attentionFlow]

end AttentionFlow

/-! ## Layer composition helpers -/

section LayerComp

variable {S T U V : Type*}
         [Fintype S] [Fintype T] [Fintype U] [Fintype V]

/-- Compose three layers (common pattern: embed → transform → project). -/
noncomputable def Mixer.comp3
    (M₁ : Mixer S T) (M₂ : Mixer T U) (M₃ : Mixer U V) : Mixer S V :=
  (M₁.comp M₂).comp M₃

/-- comp3 is equivalent to right-associated composition. -/
lemma Mixer.comp3_eq_comp_comp (M₁ : Mixer S T) (M₂ : Mixer T U) (M₃ : Mixer U V) :
    M₁.comp3 M₂ M₃ = M₁.comp (M₂.comp M₃) := by
  simp [Mixer.comp3, Mixer.comp_assoc]

end LayerComp

/-! ## Transformer blocks -/

section TransformerBlock

variable {Pos : Type*} [Fintype Pos] [DecidableEq Pos]

/-- A full transformer block conceptually: attention + feedforward with residuals.

In a Pre-LN transformer (e.g. GPT-2): `y = x + Attention(LayerNorm(x))` followed by
`output = y + FFN(LayerNorm(y))`. We model this as composition of residual mixers.

The coefficients `c_attn` and `c_ff` control how much of the skip connection
vs the transformed value flows through. -/
noncomputable def Mixer.transformerBlock
    (attn : Mixer Pos Pos)
    (ff : Mixer Pos Pos)
    (c_attn c_ff : NNReal)
    (h_attn : c_attn ≤ 1) (h_ff : c_ff ≤ 1) : Mixer Pos Pos :=
  (Mixer.residual attn c_attn h_attn).comp (Mixer.residual ff c_ff h_ff)

/-- A transformer block with no skip connections is just attention then FFN. -/
lemma Mixer.transformerBlock_no_skip (attn ff : Mixer Pos Pos) :
    Mixer.transformerBlock attn ff 0 0 (zero_le _) (zero_le _) = attn.comp ff := by
  simp [Mixer.transformerBlock, Mixer.residual_zero]

/-- A transformer block with full skip connections is identity. -/
lemma Mixer.transformerBlock_full_skip (attn ff : Mixer Pos Pos) :
    Mixer.transformerBlock attn ff 1 1 le_rfl le_rfl = Mixer.identity := by
  simp [Mixer.transformerBlock, Mixer.residual_one]

end TransformerBlock

/-! ## Stacking transformer layers -/

section TransformerStack

variable {Pos : Type*} [Fintype Pos] [DecidableEq Pos]

/-- A stack of transformer blocks. -/
noncomputable def transformerStack (blocks : List (Mixer Pos Pos)) : Mixer Pos Pos :=
  attentionFlow blocks

/-- The effective attention from position `i` to position `j` through a stack
of `n` transformer layers is given by the composed mixer weight. -/
noncomputable def effectiveAttention
    (blocks : List (Mixer Pos Pos)) (i j : Pos) : NNReal :=
  (transformerStack blocks).w i j

/-- Effective attention forms a probability distribution over target positions
for each source position. This is a key property for interpretation:
it tells us "how much" each source position contributes to each target. -/
theorem effectiveAttention_normalized (blocks : List (Mixer Pos Pos)) (i : Pos) :
    (∑ j, effectiveAttention blocks i j) = 1 := by
  simp only [effectiveAttention, transformerStack]
  exact (attentionFlow blocks).row_sum_one i

end TransformerStack

/-! ## Path-based decomposition

This section provides the key insight connecting mixer composition to
path-based attribution. The weight `(M.comp N).w i k` can be decomposed
as a sum over intermediate positions, corresponding to paths through
the computation graph.
-/

section PathDecomposition

variable {S T U V : Type*} [Fintype S] [Fintype T] [Fintype U] [Fintype V]

/-- The composition weight decomposes as a sum over paths through the intermediate layer.
This is the foundation for path-integrated attribution methods. -/
theorem Mixer.comp_path_decomposition (M : Mixer S T) (N : Mixer T U) (i : S) (k : U) :
    (M.comp N).w i k = ∑ j, M.w i j * N.w j k := rfl

/-- The contribution of path `i → j → k` to the total weight `i → k`. -/
noncomputable def pathContrib (M : Mixer S T) (N : Mixer T U) (i : S) (j : T) (k : U) : NNReal :=
  M.w i j * N.w j k

/-- Path contributions sum to the total weight. -/
theorem pathContrib_sum (M : Mixer S T) (N : Mixer T U) (i : S) (k : U) :
    (∑ j, pathContrib M N i j k) = (M.comp N).w i k := by
  simp only [pathContrib, Mixer.comp]

/-- For three-layer composition, paths go through two intermediate positions. -/
theorem Mixer.comp3_path_decomposition
    (M₁ : Mixer S T) (M₂ : Mixer T U) (M₃ : Mixer U V) (i : S) (l : V) :
    (M₁.comp3 M₂ M₃).w i l = ∑ j, ∑ k, M₁.w i j * M₂.w j k * M₃.w k l := by
  simp only [Mixer.comp3, Mixer.comp, Finset.sum_mul]
  rw [Finset.sum_comm]

end PathDecomposition

/-! ## Conservation theorems

Key insight: mixer operations preserve total probability mass.
This connects to the completeness axiom in attribution theory.
-/

section Conservation

variable {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]

/-- Pushing a probability vector through a mixer preserves total mass.
This is the probabilistic interpretation of "conservation". -/
theorem Mixer.push_preserves_total_mass (M : Mixer S T) (p : ProbVec S) :
    (∑ j, (M.push p).mass j) = ∑ i, p.mass i := by
  simp only [ProbVec.sum_mass]

/-- The pushed mass at position `j` is the weighted sum of source masses. -/
lemma Mixer.push_mass_eq (M : Mixer S T) (p : ProbVec S) (j : T) :
    (M.push p).mass j = ∑ i, p.mass i * M.w i j := rfl

/-- Conservation for composition: pushing through composed mixers
equals pushing through each sequentially. -/
theorem Mixer.push_comp (M : Mixer S T) (N : Mixer T U) (p : ProbVec S) :
    (M.comp N).push p = N.push (M.push p) := by
  ext k
  simp only [Mixer.push, Mixer.comp]
  simp only [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  simp_rw [mul_assoc]

end Conservation

/-! ## Residual stream decomposition

A key insight for transformer interpretation: residual connections create
multiple "paths" through the network. The effective contribution from source
to target can be decomposed into:
1. Direct path: information flows unchanged through the skip connection
2. Indirect path: information is transformed by the attention/FFN layer

This decomposition is crucial for understanding how much a layer actually
contributes vs how much just passes through unchanged.
-/

section ResidualDecomposition

variable {S : Type*} [Fintype S] [DecidableEq S]

/-- **Residual decomposition theorem**: The residual mixer weight at (i,j) equals
the sum of direct (skip) and indirect (transform) path contributions.

This is the formal version of: "How much information flows directly vs through the layer?" -/
theorem Mixer.residual_decomposition (M : Mixer S S) (c : NNReal) (hc : c ≤ 1) (i j : S) :
    (Mixer.residual M c hc).w i j =
      c * (if i = j then 1 else 0) + (1 - c) * M.w i j := by
  simp only [Mixer.residual]

/-- **Skip connection dominance**: When the skip coefficient c is large,
the residual is close to identity. Specifically, the diagonal entries
are at least c. -/
theorem Mixer.residual_skip_dominance (M : Mixer S S) (c : NNReal) (hc : c ≤ 1) (i : S) :
    (Mixer.residual M c hc).w i i ≥ c := by
  simp only [Mixer.residual, ↓reduceIte, mul_one, le_add_iff_nonneg_right, zero_le]

/-- Off-diagonal entries are bounded by the indirect path contribution. -/
theorem Mixer.residual_off_diag_bound (M : Mixer S S) (c : NNReal) (hc : c ≤ 1)
    {i j : S} (hij : i ≠ j) :
    (Mixer.residual M c hc).w i j = (1 - c) * M.w i j := by
  simp only [Mixer.residual, hij, ↓reduceIte, mul_zero, zero_add]

/-- **Interpretation insight**: Off-diagonal entries are scaled down by (1-c).
If c = 0.9, off-diagonal influence is reduced to 10% of original.
This quantifies how residual connections "protect" self-information. -/
theorem Mixer.residual_off_diag_scaling (M : Mixer S S) (c : NNReal) (hc : c ≤ 1)
    {i j : S} (hij : i ≠ j) :
    (Mixer.residual M c hc).w i j ≤ M.w i j := by
  rw [Mixer.residual_off_diag_bound M c hc hij]
  have h1 : (1 - c) ≤ 1 := tsub_le_self
  calc (1 - c) * M.w i j ≤ 1 * M.w i j := mul_le_mul_of_nonneg_right h1 (zero_le _)
    _ = M.w i j := one_mul _

end ResidualDecomposition

/-! ## Attention concentration and information flow

When attention is concentrated on few positions, this limits how information
can spread through the network. These theorems formalize bounds on information flow.
-/

section AttentionConcentration

variable {S : Type*} [Fintype S]

/-- The maximum attention weight from position i determines an upper bound
on how much relevance can flow to any single source position. -/
noncomputable def Mixer.maxWeight (M : Mixer S S) (i : S) : NNReal :=
  Finset.sup' Finset.univ ⟨i, Finset.mem_univ i⟩ (fun j => M.w i j)

/-- Any weight is at most the maxWeight. -/
lemma Mixer.weight_le_maxWeight (M : Mixer S S) (i j : S) :
    M.w i j ≤ M.maxWeight i := by
  simp only [Mixer.maxWeight]
  exact Finset.le_sup' (fun k => M.w i k) (Finset.mem_univ j)

/-- **Attention bottleneck**: The pushed mass at any position j is bounded by
the sum over i of (mass at i) × (max attention weight from i).
In other words, if attention is spread thin, mass can't concentrate. -/
theorem Mixer.push_concentration_bound (M : Mixer S S) (p : ProbVec S) (j : S) :
    (M.push p).mass j ≤ ∑ i, p.mass i * M.maxWeight i := by
  simp only [Mixer.push]
  apply Finset.sum_le_sum
  intro i _
  exact mul_le_mul_of_nonneg_left (M.weight_le_maxWeight i j) (zero_le _)

end AttentionConcentration

/-! ## Ablation and masking analysis

When we "ablate" or "mask" certain positions, we effectively zero out their
contribution. This section formalizes the effect of such interventions.
-/

section Ablation

variable {S : Type*} [Fintype S]

/-- A masked weight function: positions in the mask set have their outgoing weights zeroed.
This models "what if we remove these positions from consideration?"

Note: This is sub-stochastic (rows of blocked positions don't sum to 1). -/
noncomputable def Mixer.maskFn (M : Mixer S S) (blocked : Set S) [DecidablePred blocked] :
    S → S → NNReal :=
  fun i j => if blocked i then 0 else M.w i j

/-- Masking a position removes its contribution entirely. -/
lemma Mixer.maskFn_blocked (M : Mixer S S) (blocked : Set S) [DecidablePred blocked]
    {i : S} (hi : blocked i) (j : S) :
    M.maskFn blocked i j = 0 := by
  simp [Mixer.maskFn, hi]

/-- Unblocked positions keep their original weights. -/
lemma Mixer.maskFn_unblocked (M : Mixer S S) (blocked : Set S) [DecidablePred blocked]
    {i : S} (hi : ¬blocked i) (j : S) :
    M.maskFn blocked i j = M.w i j := by
  simp [Mixer.maskFn, hi]

/-- The contribution from blocked positions to position j. -/
noncomputable def blockedContribution (M : Mixer S S) (p : ProbVec S)
    (blocked : Set S) [DecidablePred blocked] (j : S) : NNReal :=
  ∑ i : S, if blocked i then p.mass i * M.w i j else 0

/-- The contribution from unblocked positions to position j. -/
noncomputable def unblockedContribution (M : Mixer S S) (p : ProbVec S)
    (blocked : Set S) [DecidablePred blocked] (j : S) : NNReal :=
  ∑ i : S, if blocked i then 0 else p.mass i * M.w i j

/-- **Ablation decomposition**: The pushed mass equals blocked plus unblocked contributions. -/
theorem Mixer.ablation_decomposition (M : Mixer S S) (p : ProbVec S)
    (blocked : Set S) [DecidablePred blocked] (j : S) :
    (M.push p).mass j =
      unblockedContribution M p blocked j + blockedContribution M p blocked j := by
  simp only [Mixer.push, unblockedContribution, blockedContribution]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  split_ifs <;> simp

end Ablation

/-! ## Composition depth and information spread

As we compose more layers, information can spread to more positions.
These theorems characterize how "reach" grows with depth.
-/

section CompositionDepth

variable {S : Type*} [Fintype S]

/-- A position j is "reachable" from i through mixer M if M.w i j > 0. -/
def Mixer.reachable (M : Mixer S S) (i j : S) : Prop := M.w i j ≠ 0

/-- **Reach expansion**: If j is reachable from i through M, and k is reachable
from j through N, then the composition has at least the product weight.
This shows how influence compounds through layers. -/
theorem Mixer.reach_comp (M N : Mixer S S) {i j k : S}
    (_ : M.reachable i j) (_ : N.reachable j k) :
    (M.comp N).w i k ≥ M.w i j * N.w j k := by
  simp only [Mixer.comp, ge_iff_le]
  exact Finset.single_le_sum (f := fun x => M.w i x * N.w x k)
    (fun x _ => zero_le _) (Finset.mem_univ j)

/-- **Path contribution bound**: The contribution through any single intermediate j
is at most the composed weight. -/
theorem Mixer.path_contrib_le_comp (M N : Mixer S S) (i j k : S) :
    M.w i j * N.w j k ≤ (M.comp N).w i k := by
  simp only [Mixer.comp]
  exact Finset.single_le_sum (f := fun x => M.w i x * N.w x k)
    (fun x _ => zero_le _) (Finset.mem_univ j)

/-- Composing mixers preserves reachability through any nonzero path. -/
theorem Mixer.comp_reachable_of_path (M N : Mixer S S) {i j k : S}
    (hij : M.w i j ≠ 0) (hjk : N.w j k ≠ 0) :
    (M.comp N).reachable i k := by
  simp only [Mixer.reachable, Mixer.comp, ne_eq]
  intro h
  have hterm : M.w i j * N.w j k = 0 := by
    have hle : M.w i j * N.w j k ≤ ∑ x, M.w i x * N.w x k :=
      Finset.single_le_sum (f := fun x => M.w i x * N.w x k)
        (fun x _ => zero_le _) (Finset.mem_univ j)
    rw [h] at hle
    exact le_antisymm hle (zero_le _)
  cases mul_eq_zero.mp hterm with
  | inl h => exact hij h
  | inr h => exact hjk h

end CompositionDepth

/-! ## Information-theoretic bounds

These theorems connect mixer properties to information-theoretic concepts,
providing bounds on how much "information" can flow through attention layers.
-/

section InformationBounds

variable {S : Type*} [Fintype S]

/-- The "effective support size" from position i: how many positions receive
nonzero attention. Smaller support = more concentrated attention. -/
noncomputable def Mixer.supportSize (M : Mixer S S) (i : S) : ℕ :=
  (Finset.univ.filter (fun j => M.w i j ≠ 0)).card

/-- Row-stochasticity means at least one entry is nonzero (assuming S nonempty). -/
lemma Mixer.exists_nonzero [Nonempty S] (M : Mixer S S) (i : S) : ∃ j, M.w i j ≠ 0 := by
  by_contra h
  push_neg at h
  have hsum : ∑ j, M.w i j = 0 := Finset.sum_eq_zero (fun j _ => h j)
  rw [M.row_sum_one i] at hsum
  exact one_ne_zero hsum

/-- **Support size bound**: In a nonempty type, every row has positive support. -/
theorem Mixer.supportSize_pos [Nonempty S] (M : Mixer S S) (i : S) :
    M.supportSize i ≥ 1 := by
  simp only [Mixer.supportSize]
  obtain ⟨j, hj⟩ := M.exists_nonzero i
  exact Finset.one_le_card.mpr ⟨j, by simp [hj]⟩

end InformationBounds

/-! ## Gradient-attribution correspondence

A key insight: for linear layers, mixer-based attribution corresponds exactly
to gradient-based attribution. This section establishes this correspondence.
-/

section GradientCorrespondence

variable {S T U : Type*} [Fintype S] [Fintype T] [Fintype U]

/-- **Gradient-attribution alignment**: For composed linear layers, the composed
attribution equals the product of individual attributions, summed over paths.
This is analogous to the chain rule for gradients. -/
theorem Mixer.chain_rule_analog (M : Mixer S T) (N : Mixer T U) (i : S) (k : U) :
    (M.comp N).w i k = ∑ j, M.w i j * N.w j k := rfl

/-- **Three-layer chain rule**: The attribution through three layers
decomposes into a double sum over intermediate positions. -/
theorem Mixer.chain_rule_three (M₁ : Mixer S T) (M₂ : Mixer T U) (M₃ : Mixer U S) (i l : S) :
    (M₁.comp (M₂.comp M₃)).w i l = ∑ j, ∑ k, M₁.w i j * M₂.w j k * M₃.w k l := by
  simp only [Mixer.comp, Finset.mul_sum, mul_assoc]

end GradientCorrespondence

/-! ## Multi-head attention

Real transformers use multiple attention heads that are combined. This section
formalizes multi-head attention and proves key properties about how heads combine.

In practice, each head has its own attention pattern, and the outputs are
concatenated and projected. For relevance/attribution purposes, this is equivalent
to a weighted combination of the individual head attention patterns.
-/

section MultiHead

variable {Pos : Type*} [Fintype Pos]
variable {numHeads : ℕ}

/-- Multi-head attention combines multiple attention heads with weights.
The weights typically come from the output projection and represent how much
each head contributes to the final output.

For interpretation: if we want to know "how much does position j contribute
to position i", we sum over heads weighted by their importance. -/
noncomputable def Mixer.multiHead
    (heads : Fin numHeads → Mixer Pos Pos)
    (headWeights : Fin numHeads → NNReal)
    (hsum : ∑ h, headWeights h = 1) : Mixer Pos Pos :=
  { w := fun i j => ∑ h, headWeights h * (heads h).w i j,
    row_sum_one := by
      intro i
      rw [Finset.sum_comm]
      simp_rw [← Finset.mul_sum, Mixer.row_sum_one, mul_one, hsum] }

/-- Each head's contribution to the multi-head attention is bounded by its weight. -/
theorem Mixer.multiHead_head_contrib_bound
    (heads : Fin numHeads → Mixer Pos Pos)
    (headWeights : Fin numHeads → NNReal)
    (hsum : ∑ h, headWeights h = 1)
    (h : Fin numHeads) (i j : Pos) :
    headWeights h * (heads h).w i j ≤ (Mixer.multiHead heads headWeights hsum).w i j := by
  simp only [Mixer.multiHead]
  exact Finset.single_le_sum (f := fun k => headWeights k * (heads k).w i j)
    (fun _ _ => zero_le _) (Finset.mem_univ h)

/-- **Head importance theorem**: A head with zero weight contributes nothing.
This formalizes the intuition that "unimportant" heads can be pruned. -/
theorem Mixer.multiHead_zero_weight
    (heads : Fin numHeads → Mixer Pos Pos)
    (headWeights : Fin numHeads → NNReal)
    (h : Fin numHeads) (hw : headWeights h = 0) (i j : Pos) :
    headWeights h * (heads h).w i j = 0 := by
  simp [hw]

/-- **Single head dominance**: If one head has weight 1 (others have weight 0),
multi-head attention reduces to that single head's attention. -/
theorem Mixer.multiHead_single_head
    (heads : Fin numHeads → Mixer Pos Pos)
    (headWeights : Fin numHeads → NNReal)
    (hsum : ∑ h, headWeights h = 1)
    (h₀ : Fin numHeads) (hdom : headWeights h₀ = 1)
    (hzero : ∀ h, h ≠ h₀ → headWeights h = 0) (i j : Pos) :
    (Mixer.multiHead heads headWeights hsum).w i j = (heads h₀).w i j := by
  simp only [Mixer.multiHead]
  have hsplit : ∑ h, headWeights h * (heads h).w i j =
      headWeights h₀ * (heads h₀).w i j +
        ∑ h ∈ Finset.univ.erase h₀, headWeights h * (heads h).w i j := by
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ h₀)]
  rw [hsplit, hdom, one_mul, add_eq_left]
  apply Finset.sum_eq_zero
  intro h hh
  simp only [Finset.mem_erase, Finset.mem_univ, ne_eq] at hh
  simp [hzero h hh.1]

/-- The multi-head attention weight is a convex combination of individual head weights. -/
theorem Mixer.multiHead_convex
    (heads : Fin numHeads → Mixer Pos Pos)
    (headWeights : Fin numHeads → NNReal)
    (hsum : ∑ h, headWeights h = 1)
    (i j : Pos) :
    (Mixer.multiHead heads headWeights hsum).w i j ≤ 1 := by
  simp only [Mixer.multiHead]
  calc ∑ h, headWeights h * (heads h).w i j
      ≤ ∑ h, headWeights h * 1 := by
        apply Finset.sum_le_sum
        intro h _
        apply mul_le_mul_of_nonneg_left _ (zero_le _)
        have hrow := (heads h).row_sum_one i
        have hle : (heads h).w i j ≤ ∑ k, (heads h).w i k :=
          Finset.single_le_sum (f := fun k => (heads h).w i k)
            (fun _ _ => zero_le _) (Finset.mem_univ j)
        rw [hrow] at hle
        exact hle
    _ = ∑ h, headWeights h := by simp
    _ = 1 := hsum

end MultiHead

/-! ## Causal masking

Autoregressive models (GPT-style) use causal masking: position i can only attend
to positions j ≤ i. This creates a triangular attention pattern with important
consequences for information flow and attribution.
-/

section CausalMask

variable {n : ℕ}

/-- A causal attention mask: position i can attend to j only if j ≤ i.
This models autoregressive/decoder-only transformers like GPT. -/
def isCausal (M : Mixer (Fin n) (Fin n)) : Prop :=
  ∀ i j : Fin n, j.val > i.val → M.w i j = 0

/-- **Causal reachability**: In a causal mixer, information can only flow
from later to earlier positions (or stay in place). -/
theorem causal_reachable_dir (M : Mixer (Fin n) (Fin n)) (hcaus : isCausal M)
    {i j : Fin n} (hreach : M.reachable i j) : j.val ≤ i.val := by
  by_contra h
  push_neg at h
  have := hcaus i j h
  exact hreach this

/-- Composition of causal mixers is causal. This means stacking causal attention
layers preserves the causal property. -/
theorem causal_comp (M N : Mixer (Fin n) (Fin n))
    (hM : isCausal M) (hN : isCausal N) : isCausal (M.comp N) := by
  intro i j hij
  simp only [Mixer.comp]
  apply Finset.sum_eq_zero
  intro k _
  by_cases hk : k.val > i.val
  · simp [hM i k hk]
  · push_neg at hk
    by_cases hkj : j.val > k.val
    · simp [hN k j hkj]
    · push_neg at hkj
      -- k ≤ i and j ≤ k, so j ≤ i, contradicting hij
      omega

/-- **Causal information bound**: In a causal model, the total attention from
position i to future positions is zero. All attention goes to past/current. -/
theorem causal_future_attention_zero (M : Mixer (Fin n) (Fin n)) (hcaus : isCausal M)
    (i : Fin n) : ∑ j ∈ Finset.univ.filter (fun j => j.val > i.val), M.w i j = 0 := by
  apply Finset.sum_eq_zero
  intro j hj
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
  exact hcaus i j hj

/-- In a causal mixer, all attention mass goes to positions ≤ i. -/
theorem causal_past_attention_one (M : Mixer (Fin n) (Fin n)) (hcaus : isCausal M)
    (i : Fin n) : ∑ j ∈ Finset.univ.filter (fun j => j.val ≤ i.val), M.w i j = 1 := by
  have htotal := M.row_sum_one i
  have hfuture := causal_future_attention_zero M hcaus i
  -- Show the two filters partition univ
  have hpart : Finset.univ.filter (fun j : Fin n => j.val ≤ i.val) ∪
      Finset.univ.filter (fun j => i.val < j.val) = Finset.univ := by
    ext x
    simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
    exact le_or_gt x.val i.val
  have hdisj : Disjoint (Finset.univ.filter (fun j : Fin n => j.val ≤ i.val))
      (Finset.univ.filter (fun j => i.val < j.val)) := by
    rw [Finset.disjoint_filter]
    intro x _ hle hlt
    omega
  have key : ∑ j, M.w i j = ∑ j ∈ Finset.univ.filter (fun j => j.val ≤ i.val), M.w i j +
      ∑ j ∈ Finset.univ.filter (fun j => i.val < j.val), M.w i j := by
    conv_lhs => rw [← hpart]
    rw [Finset.sum_union hdisj]
  simp only [hfuture, add_zero, htotal] at key
  exact key.symm

/-- **First token theorem**: In a causal model, the first position (index 0)
can only attend to itself, so its self-attention weight is 1. -/
theorem causal_first_token_self (M : Mixer (Fin (n + 1)) (Fin (n + 1)))
    (hcaus : isCausal M) : M.w 0 0 = 1 := by
  have h := causal_past_attention_one M hcaus 0
  simp only [Fin.val_zero] at h
  have hfilt : Finset.univ.filter (fun j : Fin (n + 1) => j.val ≤ 0) = {0} := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    constructor
    · intro hx
      exact Fin.ext (Nat.le_zero.mp hx)
    · intro hx
      simp [hx]
  rw [hfilt] at h
  simpa using h

end CausalMask

/-! ## Attention head analysis

Tools for analyzing individual attention heads and their roles.
-/

section HeadAnalysis

variable {Pos : Type*} [Fintype Pos]

/-- The "concentration" of attention from position i: sum of squared weights.
Higher value = more concentrated on few positions. Lower = more spread out.
This is a measure related to the inverse of entropy. -/
noncomputable def Mixer.attentionConcentration (M : Mixer Pos Pos) (i : Pos) : NNReal :=
  ∑ j, (M.w i j) ^ 2

/-- Concentration is at most 1 (achieved when all attention on one position). -/
theorem Mixer.attentionConcentration_upper_bound (M : Mixer Pos Pos) (i : Pos) :
    M.attentionConcentration i ≤ 1 := by
  simp only [Mixer.attentionConcentration]
  have hsum := M.row_sum_one i
  calc ∑ j, (M.w i j) ^ 2
      ≤ ∑ j, M.w i j := by
        apply Finset.sum_le_sum
        intro j _
        rw [sq]
        have hle : M.w i j ≤ ∑ k, M.w i k :=
          Finset.single_le_sum (f := fun k => M.w i k) (fun _ _ => zero_le _) (Finset.mem_univ j)
        rw [hsum] at hle
        calc M.w i j * M.w i j ≤ M.w i j * 1 := mul_le_mul_of_nonneg_left hle (zero_le _)
          _ = M.w i j := mul_one _
    _ = 1 := hsum

/-- **Sparsity indicator**: An attention head is "sparse" at position i if its
concentration is high (close to 1). This indicates it focuses on few positions. -/
def Mixer.isSparseAt (M : Mixer Pos Pos) (i : Pos) (threshold : NNReal) : Prop :=
  M.attentionConcentration i ≥ threshold

/-- **Uniform attention indicator**: An attention head is "diffuse" at position i
if its concentration is low. This indicates it spreads attention broadly. -/
def Mixer.isDiffuseAt (M : Mixer Pos Pos) (i : Pos) (threshold : NNReal) : Prop :=
  M.attentionConcentration i ≤ threshold

/-- If all attention is on one position, concentration is 1. -/
theorem Mixer.attentionConcentration_one_hot
    (M : Mixer Pos Pos) (i j₀ : Pos)
    (h : M.w i j₀ = 1) (hz : ∀ j, j ≠ j₀ → M.w i j = 0) :
    M.attentionConcentration i = 1 := by
  classical
  simp only [Mixer.attentionConcentration]
  have hsplit : ∑ j, (M.w i j) ^ 2 =
      (M.w i j₀) ^ 2 + ∑ j ∈ Finset.univ.erase j₀, (M.w i j) ^ 2 := by
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ j₀)]
  rw [hsplit, h, one_pow, add_eq_left]
  apply Finset.sum_eq_zero
  intro j hj
  simp only [Finset.mem_erase, Finset.mem_univ, ne_eq] at hj
  simp [hz j hj.1]

end HeadAnalysis

/-! ## Residual dominance analysis

For interpreting transformers, we often want to know: "How dominant is the
skip connection vs the attention?" This section provides tools for this.
-/

section ResidualDominance

variable {S : Type*} [Fintype S]

/-- A residual layer with coefficient c gives diagonal elements at least c.
This is a key property for interpretation: high c means the skip connection
dominates, preserving information from earlier layers. -/
theorem residual_diagonal_lower [DecidableEq S]
    (M : Mixer S S) (c : NNReal) (hc : c ≤ 1) (i : S) :
    (Mixer.residual M c hc).w i i ≥ c := by
  simp only [Mixer.residual]
  have hnonneg : 0 ≤ (1 - c) * M.w i i := by
    have h1 : 0 ≤ (1 - c) := by
      exact zero_le _
    exact mul_nonneg h1 (by simp)
  calc c * 1 + (1 - c) * M.w i i ≥ c * 1 := by
         exact le_add_of_nonneg_right hnonneg
       _ = c * 1 + 0 := by simp
       _ = c := by ring

/-- Off-diagonal elements of a residual are scaled down by (1-c).
This quantifies how much the attention contribution is suppressed. -/
theorem residual_offdiag_scale [DecidableEq S]
    (M : Mixer S S) (c : NNReal) (hc : c ≤ 1)
    (i j : S) (hij : i ≠ j) :
    (Mixer.residual M c hc).w i j = (1 - c) * M.w i j := by
  simp only [Mixer.residual]
  simp [hij]

/-- The sum of off-diagonal elements in a residual row is at most (1-c).
This bounds how much attention "leaks" to other positions. -/
theorem residual_offdiag_sum_bound [DecidableEq S]
    (M : Mixer S S) (c : NNReal) (hc : c ≤ 1) (i : S) :
    ∑ j ∈ Finset.univ.filter (· ≠ i), (Mixer.residual M c hc).w i j ≤ 1 - c := by
  calc ∑ j ∈ Finset.univ.filter (· ≠ i), (Mixer.residual M c hc).w i j
      = ∑ j ∈ Finset.univ.filter (· ≠ i), (1 - c) * M.w i j := by
        apply Finset.sum_congr rfl
        intro j hj
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
        exact residual_offdiag_scale M c hc i j (Ne.symm hj)
    _ = (1 - c) * ∑ j ∈ Finset.univ.filter (· ≠ i), M.w i j := by
        rw [Finset.mul_sum]
    _ ≤ (1 - c) * 1 := by
        apply mul_le_mul_of_nonneg_left _ (by simp)
        calc ∑ j ∈ Finset.univ.filter (· ≠ i), M.w i j
            ≤ ∑ j, M.w i j := Finset.sum_le_sum_of_subset (fun x hx => Finset.mem_univ x)
          _ = 1 := M.row_sum_one i
    _ = 1 - c := by ring

end ResidualDominance

/-! ## Deep composition bounds

A key question for transformer interpretation: after L layers, how spread out
is the attribution? This section provides quantitative bounds.
-/

section DeepComposition

variable {Pos : Type*} [Fintype Pos]

/-- Composition weight is at most 1 (row-stochastic property preserved). -/
theorem comp_weight_le_one (M N : Mixer Pos Pos) (i k : Pos) :
    (M.comp N).w i k ≤ 1 := by
  have h := (M.comp N).row_sum_one i
  calc (M.comp N).w i k
      ≤ ∑ k', (M.comp N).w i k' :=
        Finset.single_le_sum (fun _ _ => zero_le _) (Finset.mem_univ k)
    _ = 1 := h

/-- Each term in a composition sum is bounded by the corresponding M weight. -/
theorem comp_term_bound (M N : Mixer Pos Pos) (i j k : Pos) :
    M.w i j * N.w j k ≤ M.w i j := by
  calc M.w i j * N.w j k ≤ M.w i j * 1 := by
        apply mul_le_mul_of_nonneg_left _ (zero_le _)
        calc N.w j k ≤ ∑ k', N.w j k' :=
              Finset.single_le_sum (fun _ _ => zero_le _) (Finset.mem_univ k)
          _ = 1 := N.row_sum_one j
    _ = M.w i j := by ring

end DeepComposition

/-! ## Cross-attention for encoder-decoder models

Real seq2seq models use cross-attention where queries come from the decoder
and keys/values come from the encoder. This is a mixer from decoder positions
to encoder positions (tracking where decoder attends in encoder).
-/

section CrossAttention

variable {EncPos DecPos : Type*} [Fintype EncPos] [Fintype DecPos]

/-- Cross-attention mixer: tracks where each decoder position attends in encoder.
This is the fundamental building block for encoder-decoder attribution. -/
noncomputable def Mixer.crossAttention
    (w : DecPos → EncPos → NNReal)
    (hw : ∀ d, ∑ e, w d e = 1) : Mixer DecPos EncPos where
  w := w
  row_sum_one := hw

/-- Cross-attention preserves the row-stochastic property. -/
theorem Mixer.crossAttention_normalized
    (w : DecPos → EncPos → NNReal)
    (hw : ∀ d, ∑ e, w d e = 1) (d : DecPos) :
    ∑ e, (Mixer.crossAttention w hw).w d e = 1 :=
  hw d

end CrossAttention

/-! ## Layer-wise attribution analysis

For understanding transformer behavior, it's crucial to decompose attribution
layer by layer. This section provides tools for such analysis.
-/

section LayerAttribution

variable {Pos : Type*} [Fintype Pos] [DecidableEq Pos]

/-- The attribution from position i to position j through a sequence of layers. -/
noncomputable def layerWiseAttribution
    (layers : List (Mixer Pos Pos)) (i j : Pos) : NNReal :=
  (layers.foldl Mixer.comp Mixer.identity).w i j

/-- Empty layer list gives identity attribution. -/
@[simp]
theorem layerWiseAttribution_nil (i j : Pos) :
    layerWiseAttribution (Pos := Pos) [] i j = Mixer.identity.w i j := rfl

/-- Single layer attribution equals the layer's weight. -/
theorem layerWiseAttribution_singleton (M : Mixer Pos Pos) (i j : Pos) :
    layerWiseAttribution [M] i j = M.w i j := by
  simp only [layerWiseAttribution, List.foldl_cons, List.foldl_nil, Mixer.identity_comp]

/-- Attribution through layers is bounded by 1 (probability). -/
theorem layerWiseAttribution_le_one (layers : List (Mixer Pos Pos)) (i j : Pos) :
    layerWiseAttribution layers i j ≤ 1 := by
  simp only [layerWiseAttribution]
  have h := (layers.foldl Mixer.comp Mixer.identity).row_sum_one i
  calc (layers.foldl Mixer.comp Mixer.identity).w i j
      ≤ ∑ k, (layers.foldl Mixer.comp Mixer.identity).w i k :=
        Finset.single_le_sum (fun _ _ => zero_le _) (Finset.mem_univ j)
    _ = 1 := h

/-- **Total attribution conservation**: Sum over all targets equals 1.
This is the formal statement that "attribution mass is conserved". -/
theorem layerWiseAttribution_sum_one (layers : List (Mixer Pos Pos)) (i : Pos) :
    ∑ j, layerWiseAttribution layers i j = 1 := by
  simp only [layerWiseAttribution]
  exact (layers.foldl Mixer.comp Mixer.identity).row_sum_one i

end LayerAttribution

/-! ## Tracer uniqueness for transformer interpretation

The key insight connecting this formalization to neural network interpretation:
the tracer uniqueness theorem (from `Uniqueness.lean`) implies that attribution
methods based on the mixer framework are uniquely determined by boundary conditions.

This provides formal justification for attention-based interpretation methods:
if two attribution methods both propagate mass according to the same mixers
(attention patterns) and agree on boundary conditions (e.g., start with
probability 1 at the output token), then they must produce identical attributions.
-/

section TracerInterpretation

variable {S : Type*} [Fintype S] [DecidableEq S]

end TracerInterpretation

/-- If two tracer propagation methods satisfy the same mixing recurrence
on a transformer's computation graph, they must coincide. This is the
formal statement that "attention-based attribution is unique given
the attention patterns and boundary conditions." -/
theorem transformer_attribution_unique
    {S : Type*}
    (n : ℕ)
    (parents : Fin n → Finset (Fin n))
    (htopo : ∀ k u, u ∈ parents k → u.val < k.val)
    (c : Fin n → Fin n → NNReal)
    (L : LocalSystem n := ⟨parents, c, fun {i u} h => htopo i u h⟩)
    (T T' : LocalSystem.TracerFamily (S := S) n)
    (hT : L.Satisfies T)
    (hT' : L.Satisfies T') : T = T' :=
  LocalSystem.tracer_unique L hT hT'

end Nfp
