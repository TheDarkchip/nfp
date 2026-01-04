-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.List.Range
import Mathlib.Data.Real.Basic
import Nfp.Model.Gpt2
import Nfp.Sound.Bounds.Attention
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Linear.FinFold

/-!
Interval bounds for transformer stacks and final LayerNorm outputs.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

private lemma fin_univ_nonempty (seq : Nat) [NeZero seq] :
    (Finset.univ : Finset (Fin seq)).Nonempty := by
  classical
  refine ⟨⟨0, ?_⟩, by simp⟩
  exact Nat.pos_of_ne_zero (NeZero.ne (n := seq))

/-- Interval bounds across tokens for an embedding map. -/
def embeddingIntervalBounds {seq dModel : Nat} [NeZero seq]
    (x : Fin seq → Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let h : (Finset.univ : Finset (Fin seq)).Nonempty := fin_univ_nonempty (seq := seq)
  (fun i => (Finset.univ).inf' h (fun q => x q i),
   fun i => (Finset.univ).sup' h (fun q => x q i))

/-- `embeddingIntervalBounds` bounds embeddings coordinatewise. -/
theorem embeddingIntervalBounds_spec {seq dModel : Nat} [NeZero seq]
    (x : Fin seq → Fin dModel → Rat) :
    let bounds := embeddingIntervalBounds x
    ∀ q i,
      (bounds.1 i : Real) ≤ (x q i : Real) ∧
        (x q i : Real) ≤ (bounds.2 i : Real) := by
  classical
  intro bounds q i
  have hloRat : bounds.1 i ≤ x q i := by
    have h :=
      Finset.inf'_le (s := (Finset.univ : Finset (Fin seq)))
        (f := fun k => x k i) (b := q) (by simp)
    simpa [bounds, embeddingIntervalBounds, fin_univ_nonempty] using h
  have hhiRat : x q i ≤ bounds.2 i := by
    have h :=
      Finset.le_sup' (s := (Finset.univ : Finset (Fin seq)))
        (f := fun k => x k i) (b := q) (by simp)
    simpa [bounds, embeddingIntervalBounds, fin_univ_nonempty] using h
  constructor
  · exact_mod_cast hloRat
  · exact_mod_cast hhiRat

/-- Real-valued output of a transformer layer. -/
noncomputable def transformerLayerReal {seq dModel dHead numHeads hidden : Nat} [NeZero seq]
    (eps : Rat) (layer : Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (x : Fin seq → Fin dModel → Real) (q : Fin seq) (i : Fin dModel) : Real :=
  x q i +
    attentionOutputReal eps layer.ln1Gamma layer.ln1Beta heads layer.attnBias scores x q i +
    mlpReal layer.mlpWIn layer.mlpBIn layer.mlpWOut layer.mlpBOut
      (layerNormRealOfReal eps layer.ln2Gamma layer.ln2Beta
        (fun j =>
          x q j + attentionOutputReal eps layer.ln1Gamma layer.ln1Beta heads
            layer.attnBias scores x q j)) i

/-- `transformerLayerBounds` soundness for `transformerLayerReal`. -/
theorem transformerLayerBounds_spec_real {seq dModel dHead numHeads hidden : Nat} [NeZero seq]
    (eps : Rat) (layer : Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := transformerLayerBounds eps layer.ln1Gamma layer.ln1Beta layer.ln2Gamma
      layer.ln2Beta heads layer.attnBias layer.mlpWIn layer.mlpBIn layer.mlpWOut
      layer.mlpBOut lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤ transformerLayerReal eps layer heads scores x q i ∧
        transformerLayerReal eps layer heads scores x q i ≤ (bounds.2 i : Real) := by
  classical
  simpa [transformerLayerReal] using
    (transformerLayerBounds_spec (eps := eps)
      (ln1Gamma := layer.ln1Gamma) (ln1Beta := layer.ln1Beta)
      (ln2Gamma := layer.ln2Gamma) (ln2Beta := layer.ln2Beta)
      (heads := heads) (attnBias := layer.attnBias)
      (mlpWIn := layer.mlpWIn) (mlpBIn := layer.mlpBIn)
      (mlpWOut := layer.mlpWOut) (mlpBOut := layer.mlpBOut)
      (scores := scores) (lo := lo) (hi := hi) (x := x)
      hne heps hlo hhi)

/-- Real-valued transformer stack output (folded left over layers). -/
noncomputable def transformerStackReal
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq] (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (x : Fin seq → Fin dModel → Real) : Fin seq → Fin dModel → Real :=
  let step := fun x layerIdx =>
    transformerLayerReal eps (layers layerIdx) (heads layerIdx) (scores layerIdx) x
  Linear.foldlFin numLayers step x

/-- Interval bounds for a transformer stack (folded left over layers). -/
def transformerStackBounds {dModel dHead numHeads hidden numLayers : Nat}
    (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (lo hi : Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let step := fun bounds layerIdx =>
    transformerLayerBounds eps (layers layerIdx).ln1Gamma (layers layerIdx).ln1Beta
      (layers layerIdx).ln2Gamma (layers layerIdx).ln2Beta (heads layerIdx)
      (layers layerIdx).attnBias (layers layerIdx).mlpWIn (layers layerIdx).mlpBIn
      (layers layerIdx).mlpWOut (layers layerIdx).mlpBOut bounds.1 bounds.2
  Linear.foldlFin numLayers step (lo, hi)

private theorem transformerStackBounds_spec_list
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) :
    ∀ (ls : List (Fin numLayers)) (lo hi : Fin dModel → Rat)
      (x : Fin seq → Fin dModel → Real),
      (∀ q i, (lo i : Real) ≤ x q i) →
      (∀ q i, x q i ≤ (hi i : Real)) →
      let bounds := (ls.foldl
        (fun bounds layerIdx =>
          transformerLayerBounds eps (layers layerIdx).ln1Gamma (layers layerIdx).ln1Beta
            (layers layerIdx).ln2Gamma (layers layerIdx).ln2Beta (heads layerIdx)
            (layers layerIdx).attnBias (layers layerIdx).mlpWIn (layers layerIdx).mlpBIn
            (layers layerIdx).mlpWOut (layers layerIdx).mlpBOut bounds.1 bounds.2)
        (lo, hi))
      let x' := (ls.foldl
        (fun x layerIdx =>
          transformerLayerReal eps (layers layerIdx) (heads layerIdx) (scores layerIdx) x)
        x)
      ∀ q i,
        (bounds.1 i : Real) ≤ x' q i ∧
          x' q i ≤ (bounds.2 i : Real) := by
  intro ls lo hi x hlo hhi
  induction ls generalizing lo hi x hlo hhi with
  | nil =>
      simpa using fun q i => And.intro (hlo q i) (hhi q i)
  | cons l ls ih =>
      have hstep :=
        transformerLayerBounds_spec_real eps (layers l) (heads l) (scores l) lo hi x
          hne heps hlo hhi
      let bounds1 :=
        transformerLayerBounds eps (layers l).ln1Gamma (layers l).ln1Beta (layers l).ln2Gamma
          (layers l).ln2Beta (heads l) (layers l).attnBias (layers l).mlpWIn (layers l).mlpBIn
          (layers l).mlpWOut (layers l).mlpBOut lo hi
      let x1 := transformerLayerReal eps (layers l) (heads l) (scores l) x
      have hlo1 : ∀ q i, (bounds1.1 i : Real) ≤ x1 q i := fun q i => (hstep q i).1
      have hhi1 : ∀ q i, x1 q i ≤ (bounds1.2 i : Real) := fun q i => (hstep q i).2
      have ih' := ih bounds1.1 bounds1.2 x1 hlo1 hhi1
      simpa [bounds1, x1] using ih'

/-- `transformerStackBounds` soundness for real transformer-stack outputs. -/
theorem transformerStackBounds_spec {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := transformerStackBounds eps layers heads lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤ transformerStackReal eps layers heads scores x q i ∧
        transformerStackReal eps layers heads scores x q i ≤ (bounds.2 i : Real) := by
  classical
  simpa [transformerStackBounds, transformerStackReal, Linear.foldlFin_eq_foldl,
    Fin.foldl_eq_foldl_finRange] using
    transformerStackBounds_spec_list eps layers heads scores hne heps
      (List.finRange numLayers) lo hi x hlo hhi

/-- Real-valued transformer stack output after the final LayerNorm. -/
noncomputable def transformerStackFinalReal {seq dModel dHead numHeads hidden numLayers : Nat}
    [NeZero seq] (eps : Rat) (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (x : Fin seq → Fin dModel → Real) (q : Fin seq) (i : Fin dModel) : Real :=
  layerNormRealOfReal eps finalLn.gamma finalLn.beta
    (fun j => transformerStackReal eps layers heads scores x q j) i

/-- Interval bounds for transformer stack outputs after the final LayerNorm. -/
def transformerStackFinalBounds {dModel dHead numHeads hidden numLayers : Nat}
    (eps : Rat) (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (lo hi : Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let stack := transformerStackBounds eps layers heads lo hi
  layerNormIntervalBounds eps finalLn.gamma finalLn.beta stack.1 stack.2

/-- `transformerStackFinalBounds` soundness for real outputs. -/
theorem transformerStackFinalBounds_spec {seq dModel dHead numHeads hidden numLayers : Nat}
    [NeZero seq] (eps : Rat) (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := transformerStackFinalBounds eps finalLn layers heads lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤ transformerStackFinalReal eps finalLn layers heads scores x q i ∧
        transformerStackFinalReal eps finalLn layers heads scores x q i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let stack := transformerStackBounds eps layers heads lo hi
  have hstack :=
    transformerStackBounds_spec eps layers heads scores lo hi x hne heps hlo hhi q
  have hlo' : ∀ k, (stack.1 k : Real) ≤ transformerStackReal eps layers heads scores x q k :=
    fun k => (hstack k).1
  have hhi' : ∀ k, transformerStackReal eps layers heads scores x q k ≤ (stack.2 k : Real) :=
    fun k => (hstack k).2
  have hln :=
    layerNormIntervalBounds_spec_real eps finalLn.gamma finalLn.beta stack.1 stack.2
      (fun j => transformerStackReal eps layers heads scores x q j) hne heps hlo' hhi'
  simpa [bounds, transformerStackFinalBounds, stack, transformerStackFinalReal] using hln i

/-- Residual interval bounds for a GPT-2 stack from exact embeddings. -/
def gpt2ResidualIntervalBounds
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq] (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (embed : Fin seq → Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let base := embeddingIntervalBounds embed
  transformerStackFinalBounds eps finalLn layers heads base.1 base.2

/-- `gpt2ResidualIntervalBounds` soundness for real GPT-2 outputs. -/
theorem gpt2ResidualIntervalBounds_spec
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq] (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (embed : Fin seq → Fin dModel → Rat)
    (hne : dModel ≠ 0) (heps : 0 < eps) :
    let bounds := gpt2ResidualIntervalBounds eps layers heads finalLn embed
    ∀ q i,
      (bounds.1 i : Real) ≤
          transformerStackFinalReal eps finalLn layers heads scores
            (fun q i => (embed q i : Real)) q i ∧
        transformerStackFinalReal eps finalLn layers heads scores
            (fun q i => (embed q i : Real)) q i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let base := embeddingIntervalBounds embed
  have hbase := embeddingIntervalBounds_spec embed
  have hlo : ∀ q i, (base.1 i : Real) ≤ (embed q i : Real) := fun q i => (hbase q i).1
  have hhi : ∀ q i, (embed q i : Real) ≤ (base.2 i : Real) := fun q i => (hbase q i).2
  have hstack :=
    transformerStackFinalBounds_spec eps finalLn layers heads scores base.1 base.2
      (fun q i => (embed q i : Real)) hne heps hlo hhi q i
  simpa [bounds, gpt2ResidualIntervalBounds, base] using hstack

end Bounds

end Sound

end Nfp
