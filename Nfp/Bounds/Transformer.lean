-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.Attention

/-!
Interval bounds for transformer stacks.
-/

public section

namespace Nfp

namespace Bounds

open scoped BigOperators

variable {seq dModel dHead numHeads hidden : Nat}

/-- Parameters for a single transformer layer (attention + MLP). -/
structure TransformerLayerParams (dModel dHead numHeads hidden : Nat) where
  /-- LayerNorm scale for the attention input. -/
  ln1Gamma : Fin dModel → Rat
  /-- LayerNorm bias for the attention input. -/
  ln1Beta : Fin dModel → Rat
  /-- LayerNorm scale for the MLP input. -/
  ln2Gamma : Fin dModel → Rat
  /-- LayerNorm bias for the MLP input. -/
  ln2Beta : Fin dModel → Rat
  /-- Per-head attention parameters. -/
  heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead
  /-- Attention output bias (shared across heads). -/
  attnBias : Fin dModel → Rat
  /-- MLP input projection weights. -/
  mlpWIn : Fin dModel → Fin hidden → Rat
  /-- MLP input projection bias. -/
  mlpBIn : Fin hidden → Rat
  /-- MLP output projection weights. -/
  mlpWOut : Fin hidden → Fin dModel → Rat
  /-- MLP output projection bias. -/
  mlpBOut : Fin dModel → Rat

/-- Scores attached to a single transformer layer. -/
structure TransformerLayerData (seq dModel dHead numHeads hidden : Nat) where
  /-- Layer parameters. -/
  params : TransformerLayerParams dModel dHead numHeads hidden
  /-- Attention score matrix for the layer. -/
  scores : Fin numHeads → Fin seq → Fin seq → Real

/-- Real-valued transformer-layer output (attention + MLP residual). -/
noncomputable def transformerLayerReal (eps : Rat)
    (params : TransformerLayerParams dModel dHead numHeads hidden)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (x : Fin seq → Fin dModel → Real) (q : Fin seq) (i : Fin dModel) : Real :=
  x q i +
      attentionOutputReal eps params.ln1Gamma params.ln1Beta params.heads params.attnBias
        scores x q i +
    mlpReal params.mlpWIn params.mlpBIn params.mlpWOut params.mlpBOut
      (layerNormRealOfReal eps params.ln2Gamma params.ln2Beta
        (fun j =>
          x q j +
            attentionOutputReal eps params.ln1Gamma params.ln1Beta params.heads params.attnBias
              scores x q j)) i

/-- Real-valued transformer stack defined by iterating layers. -/
noncomputable def transformerStackReal (eps : Rat)
    (layers : List (TransformerLayerData seq dModel dHead numHeads hidden))
    (x : Fin seq → Fin dModel → Real) : Fin seq → Fin dModel → Real :=
  match layers with
  | [] => x
  | layer :: rest =>
      let x' : Fin seq → Fin dModel → Real :=
        fun q i => transformerLayerReal eps layer.params layer.scores x q i
      transformerStackReal eps rest x'

/-- Interval bounds obtained by propagating through a transformer stack. -/
def transformerStackBounds (eps : Rat)
    (layers : List (TransformerLayerParams dModel dHead numHeads hidden))
    (lo hi : Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  match layers with
  | [] => (lo, hi)
  | layer :: rest =>
      let bounds :=
        transformerLayerBounds eps layer.ln1Gamma layer.ln1Beta layer.ln2Gamma layer.ln2Beta
          layer.heads layer.attnBias layer.mlpWIn layer.mlpBIn layer.mlpWOut layer.mlpBOut lo hi
      transformerStackBounds eps rest bounds.1 bounds.2

/-- `transformerStackBounds` soundness for transformer stack outputs. -/
theorem transformerStackBounds_spec {seq dModel dHead numHeads hidden : Nat} [NeZero seq]
    (eps : Rat)
    (layers : List (TransformerLayerData seq dModel dHead numHeads hidden))
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := transformerStackBounds eps (layers.map (fun layer => layer.params)) lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤ transformerStackReal eps layers x q i ∧
        transformerStackReal eps layers x q i ≤ (bounds.2 i : Real) := by
  classical
  induction layers generalizing lo hi x with
  | nil =>
      intro bounds q i
      simp [transformerStackBounds, transformerStackReal, hlo q i, hhi q i]
  | cons layer rest ih =>
      intro bounds q i
      let layerBounds :=
        transformerLayerBounds eps layer.params.ln1Gamma layer.params.ln1Beta
          layer.params.ln2Gamma layer.params.ln2Beta layer.params.heads layer.params.attnBias
          layer.params.mlpWIn layer.params.mlpBIn layer.params.mlpWOut layer.params.mlpBOut lo hi
      have hlayer :=
        transformerLayerBounds_spec (seq := seq) (dModel := dModel) (dHead := dHead)
          (numHeads := numHeads) (hidden := hidden) eps layer.params.ln1Gamma
          layer.params.ln1Beta layer.params.ln2Gamma layer.params.ln2Beta layer.params.heads
          layer.params.attnBias layer.params.mlpWIn layer.params.mlpBIn layer.params.mlpWOut
          layer.params.mlpBOut layer.scores lo hi x hne heps hsqrt hlo hhi
      let x' : Fin seq → Fin dModel → Real :=
        fun q i => transformerLayerReal eps layer.params layer.scores x q i
      have hlo' : ∀ q i, (layerBounds.1 i : Real) ≤ x' q i := by
        intro q i
        have h := (hlayer q i).1
        simpa [x', transformerLayerReal, layerBounds] using h
      have hhi' : ∀ q i, x' q i ≤ (layerBounds.2 i : Real) := by
        intro q i
        have h := (hlayer q i).2
        simpa [x', transformerLayerReal, layerBounds] using h
      have hrest :=
        ih (lo := layerBounds.1) (hi := layerBounds.2) (x := x') hne heps hsqrt hlo' hhi'
      simpa [transformerStackBounds, transformerStackReal, layerBounds, x'] using hrest q i

end Bounds

end Nfp
