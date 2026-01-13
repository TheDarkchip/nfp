-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Data.List.Range
public import Mathlib.Data.Real.Basic
public import Nfp.Circuit.Cert.ResidualInterval
public import Nfp.Model.Gpt2
public import Nfp.Sound.Bounds.Attention
public import Nfp.Sound.Bounds.LayerNorm
public import Nfp.Sound.Bounds.Transformer.Embedding
public import Nfp.Sound.Linear.FinFold

/-!
Interval bounds for transformer stacks and final LayerNorm outputs.
-/

@[expose] public section

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

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
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
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
      hne heps hsqrt hlo hhi)

/-- Interval bounds for a transformer layer from per-position bounds. -/
def transformerLayerBoundsPos {seq dModel dHead numHeads hidden : Nat} [NeZero seq]
    (eps : Rat) (layer : Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (lo hi : Fin seq → Fin dModel → Rat) :
    (Fin seq → Fin dModel → Rat) × (Fin seq → Fin dModel → Rat) :=
  let positions := (Finset.univ : Finset (Fin seq))
  let hpos : positions.Nonempty := by
    simp [positions]
  let loCached := cacheBound2 lo
  let hiCached := cacheBound2 hi
  let base := intervalBoundsOn positions hpos loCached hiCached
  let baseLo := cacheBound base.1
  let baseHi := cacheBound base.2
  let attn := attentionOutputBounds eps layer.ln1Gamma layer.ln1Beta heads layer.attnBias
    baseLo baseHi
  let attnLo := cacheBound attn.1
  let attnHi := cacheBound attn.2
  let yLo : Fin seq → Fin dModel → Rat := fun q i => loCached q i + attnLo i
  let yHi : Fin seq → Fin dModel → Rat := fun q i => hiCached q i + attnHi i
  let yLoCached := cacheBound2 yLo
  let yHiCached := cacheBound2 yHi
  let out := cacheBoundPair2 (fun q =>
    layerNormAbsMlpResidualBounds eps layer.ln2Gamma layer.ln2Beta
      layer.mlpWIn layer.mlpBIn layer.mlpWOut layer.mlpBOut
      (yLoCached q) (yHiCached q))
  out

/-- `transformerLayerBoundsPos` soundness for `transformerLayerReal`. -/
theorem transformerLayerBoundsPos_spec {seq dModel dHead numHeads hidden : Nat} [NeZero seq]
    (eps : Rat) (layer : Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin seq → Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo q i : Real) ≤ x q i)
    (hhi : ∀ q i, x q i ≤ (hi q i : Real)) :
    let bounds := transformerLayerBoundsPos eps layer heads lo hi
    ∀ q i,
      (bounds.1 q i : Real) ≤ transformerLayerReal eps layer heads scores x q i ∧
        transformerLayerReal eps layer heads scores x q i ≤ (bounds.2 q i : Real) := by
  classical
  intro bounds q i
  let positions := (Finset.univ : Finset (Fin seq))
  have hpos : positions.Nonempty := by
    simp [positions]
  let loCached := cacheBound2 lo
  let hiCached := cacheBound2 hi
  have hloCached : ∀ q i, (loCached q i : Real) ≤ x q i := by
    intro q i
    simpa [loCached, cacheBound2_apply] using hlo q i
  have hhiCached : ∀ q i, x q i ≤ (hiCached q i : Real) := by
    intro q i
    simpa [hiCached, cacheBound2_apply] using hhi q i
  let base := intervalBoundsOn positions hpos loCached hiCached
  have hbase := intervalBoundsOn_spec positions hpos loCached hiCached x
    (fun q _ i => hloCached q i) (fun q _ i => hhiCached q i)
  have hloBase : ∀ q i, (base.1 i : Real) ≤ x q i := fun q i =>
    (hbase q (by simp [positions]) i).1
  have hhiBase : ∀ q i, x q i ≤ (base.2 i : Real) := fun q i =>
    (hbase q (by simp [positions]) i).2
  let baseLo := cacheBound base.1
  let baseHi := cacheBound base.2
  have hloBaseCached : ∀ q i, (baseLo i : Real) ≤ x q i := by
    intro q i
    simpa [baseLo, cacheBound_apply] using hloBase q i
  have hhiBaseCached : ∀ q i, x q i ≤ (baseHi i : Real) := by
    intro q i
    simpa [baseHi, cacheBound_apply] using hhiBase q i
  let attn := attentionOutputBounds eps layer.ln1Gamma layer.ln1Beta heads layer.attnBias
    baseLo baseHi
  have hattn := attentionOutputBounds_spec eps layer.ln1Gamma layer.ln1Beta heads
    layer.attnBias scores baseLo baseHi x hne heps hsqrt hloBaseCached hhiBaseCached q
  let attnLo := cacheBound attn.1
  let attnHi := cacheBound attn.2
  let y := fun j =>
    x q j + attentionOutputReal eps layer.ln1Gamma layer.ln1Beta heads
      layer.attnBias scores x q j
  have yLo : ∀ j, (loCached q j : Real) + (attn.1 j : Real) ≤ y j := by
    intro j
    have hlow :
        (loCached q j : Real) + (attn.1 j : Real) ≤
          x q j +
            attentionOutputReal eps layer.ln1Gamma layer.ln1Beta heads
              layer.attnBias scores x q j := by
      exact add_le_add (hloCached q j) (hattn j).1
    simpa [y] using hlow
  have yHi : ∀ j, y j ≤ (hiCached q j : Real) + (attn.2 j : Real) := by
    intro j
    have hhigh :
        x q j +
            attentionOutputReal eps layer.ln1Gamma layer.ln1Beta heads
              layer.attnBias scores x q j ≤
          (hiCached q j : Real) + (attn.2 j : Real) := by
      exact add_le_add (hhiCached q j) (hattn j).2
    simpa [y] using hhigh
  let yLoCached := cacheBound2 (fun q i => loCached q i + attnLo i)
  let yHiCached := cacheBound2 (fun q i => hiCached q i + attnHi i)
  have yLoCached_bound : ∀ j, (yLoCached q j : Real) ≤ y j := by
    intro j
    simpa [yLoCached, attnLo, cacheBound_apply, cacheBound2_apply] using (yLo j)
  have yHiCached_bound : ∀ j, y j ≤ (yHiCached q j : Real) := by
    intro j
    simpa [yHiCached, attnHi, cacheBound_apply, cacheBound2_apply] using (yHi j)
  have hmlp :=
    layerNormAbsMlpResidualBounds_spec eps layer.ln2Gamma layer.ln2Beta
      layer.mlpWIn layer.mlpBIn layer.mlpWOut layer.mlpBOut
      (yLoCached q) (yHiCached q) y hne heps hsqrt yLoCached_bound yHiCached_bound
  have hmlp_i := hmlp i
  simpa [bounds, transformerLayerBoundsPos, positions, base, loCached, hiCached, baseLo, baseHi,
    attn, attnLo, attnHi, y, yLoCached, yHiCached, cacheBound2_apply, cacheBoundPair2_apply_left,
    cacheBoundPair2_apply_right, transformerLayerReal, cacheBound_apply] using hmlp_i

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

/-- Interval bounds for a transformer stack from per-position bounds. -/
def transformerStackBoundsPos {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (lo hi : Fin seq → Fin dModel → Rat) :
    (Fin seq → Fin dModel → Rat) × (Fin seq → Fin dModel → Rat) :=
  let step := fun bounds layerIdx =>
    transformerLayerBoundsPos eps (layers layerIdx) (heads layerIdx) bounds.1 bounds.2
  Linear.foldlFin numLayers step (lo, hi)

private theorem transformerStackBoundsPos_spec_list
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
    ∀ (ls : List (Fin numLayers)) (lo hi : Fin seq → Fin dModel → Rat)
      (x : Fin seq → Fin dModel → Real),
      (∀ q i, (lo q i : Real) ≤ x q i) →
      (∀ q i, x q i ≤ (hi q i : Real)) →
      let bounds := (ls.foldl
        (fun bounds layerIdx =>
          transformerLayerBoundsPos eps (layers layerIdx) (heads layerIdx) bounds.1 bounds.2)
        (lo, hi))
      let x' := (ls.foldl
        (fun x layerIdx =>
          transformerLayerReal eps (layers layerIdx) (heads layerIdx) (scores layerIdx) x)
        x)
      ∀ q i,
        (bounds.1 q i : Real) ≤ x' q i ∧
          x' q i ≤ (bounds.2 q i : Real) := by
  intro ls lo hi x hlo hhi
  induction ls generalizing lo hi x hlo hhi with
  | nil =>
      simpa using fun q i => And.intro (hlo q i) (hhi q i)
  | cons l ls ih =>
      have hstep :=
        transformerLayerBoundsPos_spec eps (layers l) (heads l) (scores l) lo hi x
          hne heps hsqrt hlo hhi
      let bounds1 := transformerLayerBoundsPos eps (layers l) (heads l) lo hi
      let x1 := transformerLayerReal eps (layers l) (heads l) (scores l) x
      have hlo1 : ∀ q i, (bounds1.1 q i : Real) ≤ x1 q i := fun q i => (hstep q i).1
      have hhi1 : ∀ q i, x1 q i ≤ (bounds1.2 q i : Real) := fun q i => (hstep q i).2
      have ih' := ih bounds1.1 bounds1.2 x1 hlo1 hhi1
      simpa [bounds1, x1] using ih'

/-- `transformerStackBoundsPos` soundness for real transformer-stack outputs. -/
theorem transformerStackBoundsPos_spec {seq dModel dHead numHeads hidden numLayers : Nat}
    [NeZero seq] (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin seq → Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo q i : Real) ≤ x q i)
    (hhi : ∀ q i, x q i ≤ (hi q i : Real)) :
    let bounds := transformerStackBoundsPos eps layers heads lo hi
    ∀ q i,
      (bounds.1 q i : Real) ≤ transformerStackReal eps layers heads scores x q i ∧
        transformerStackReal eps layers heads scores x q i ≤ (bounds.2 q i : Real) := by
  classical
  simpa [transformerStackBoundsPos, transformerStackReal, Linear.foldlFin_eq_foldl,
    Fin.foldl_eq_foldl_finRange] using
    transformerStackBoundsPos_spec_list eps layers heads scores hne heps hsqrt
      (List.finRange numLayers) lo hi x hlo hhi

private theorem transformerStackBounds_spec_list
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
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
          hne heps hsqrt hlo hhi
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
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := transformerStackBounds eps layers heads lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤ transformerStackReal eps layers heads scores x q i ∧
        transformerStackReal eps layers heads scores x q i ≤ (bounds.2 i : Real) := by
  classical
  simpa [transformerStackBounds, transformerStackReal, Linear.foldlFin_eq_foldl,
    Fin.foldl_eq_foldl_finRange] using
    transformerStackBounds_spec_list eps layers heads scores hne heps hsqrt
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
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := transformerStackFinalBounds eps finalLn layers heads lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤ transformerStackFinalReal eps finalLn layers heads scores x q i ∧
        transformerStackFinalReal eps finalLn layers heads scores x q i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let stack := transformerStackBounds eps layers heads lo hi
  have hstack :=
    transformerStackBounds_spec eps layers heads scores lo hi x hne heps hsqrt hlo hhi q
  have hlo' : ∀ k, (stack.1 k : Real) ≤ transformerStackReal eps layers heads scores x q k :=
    fun k => (hstack k).1
  have hhi' : ∀ k, transformerStackReal eps layers heads scores x q k ≤ (stack.2 k : Real) :=
    fun k => (hstack k).2
  have hln :=
    layerNormIntervalBounds_spec_real eps finalLn.gamma finalLn.beta stack.1 stack.2
      (fun j => transformerStackReal eps layers heads scores x q j) hne heps hsqrt hlo' hhi'
  simpa [bounds, transformerStackFinalBounds, stack, transformerStackFinalReal] using hln i

/-- Interval bounds for transformer stack outputs after the final LayerNorm (per-position). -/
def transformerStackFinalBoundsPos
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq] (eps : Rat)
    (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (lo hi : Fin seq → Fin dModel → Rat) :
    (Fin seq → Fin dModel → Rat) × (Fin seq → Fin dModel → Rat) :=
  let stack := transformerStackBoundsPos eps layers heads lo hi
  let ln := fun q =>
    layerNormIntervalBounds eps finalLn.gamma finalLn.beta (stack.1 q) (stack.2 q)
  (fun q i => (ln q).1 i, fun q i => (ln q).2 i)

/-- `transformerStackFinalBoundsPos` soundness for real outputs. -/
theorem transformerStackFinalBoundsPos_spec
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq] (eps : Rat)
    (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin seq → Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo q i : Real) ≤ x q i)
    (hhi : ∀ q i, x q i ≤ (hi q i : Real)) :
    let bounds := transformerStackFinalBoundsPos eps finalLn layers heads lo hi
    ∀ q i,
      (bounds.1 q i : Real) ≤
          transformerStackFinalReal eps finalLn layers heads scores x q i ∧
        transformerStackFinalReal eps finalLn layers heads scores x q i ≤
          (bounds.2 q i : Real) := by
  classical
  intro bounds q i
  let stack := transformerStackBoundsPos eps layers heads lo hi
  have hstack :=
    transformerStackBoundsPos_spec eps layers heads scores lo hi x hne heps hsqrt hlo hhi q
  have hlo' : ∀ j, (stack.1 q j : Real) ≤ transformerStackReal eps layers heads scores x q j :=
    fun j => (hstack j).1
  have hhi' : ∀ j, transformerStackReal eps layers heads scores x q j ≤ (stack.2 q j : Real) :=
    fun j => (hstack j).2
  have hln :=
    layerNormIntervalBounds_spec_real eps finalLn.gamma finalLn.beta (stack.1 q) (stack.2 q)
      (fun j => transformerStackReal eps layers heads scores x q j) hne heps hsqrt hlo' hhi'
  simpa [bounds, transformerStackFinalBoundsPos, stack, transformerStackFinalReal] using hln i

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
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
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
      (fun q i => (embed q i : Real)) hne heps hsqrt hlo hhi q i
  simpa [bounds, gpt2ResidualIntervalBounds, base] using hstack

/-- Residual interval bounds over an active set from exact embeddings. -/
def gpt2ResidualIntervalBoundsActive
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (active : Finset (Fin seq)) (hactive : active.Nonempty) (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (embed : Fin seq → Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let baseLo : Fin seq → Fin dModel → Rat := embed
  let baseHi : Fin seq → Fin dModel → Rat := embed
  let final := transformerStackFinalBoundsPos eps finalLn layers heads baseLo baseHi
  intervalBoundsOn active hactive final.1 final.2

/-- `gpt2ResidualIntervalBoundsActive` soundness for real GPT-2 outputs. -/
theorem gpt2ResidualIntervalBoundsActive_spec
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (active : Finset (Fin seq)) (hactive : active.Nonempty) (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (embed : Fin seq → Fin dModel → Rat)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
    let bounds := gpt2ResidualIntervalBoundsActive active hactive eps layers heads finalLn embed
    ∀ q, q ∈ active → ∀ i,
      (bounds.1 i : Real) ≤
          transformerStackFinalReal eps finalLn layers heads scores
            (fun q i => (embed q i : Real)) q i ∧
        transformerStackFinalReal eps finalLn layers heads scores
            (fun q i => (embed q i : Real)) q i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds q hq i
  let baseLo : Fin seq → Fin dModel → Rat := embed
  let baseHi : Fin seq → Fin dModel → Rat := embed
  let final := transformerStackFinalBoundsPos eps finalLn layers heads baseLo baseHi
  have hfinal :=
    transformerStackFinalBoundsPos_spec eps finalLn layers heads scores baseLo baseHi
      (fun q i => (embed q i : Real)) hne heps hsqrt
      (fun q i => by simp [baseLo])
      (fun q i => by simp [baseHi])
  have hlo : ∀ q, q ∈ active → ∀ i,
      (final.1 q i : Real) ≤
          transformerStackFinalReal eps finalLn layers heads scores
            (fun q i => (embed q i : Real)) q i := by
    intro q hq i
    simpa [final] using (hfinal q i).1
  have hhi : ∀ q, q ∈ active → ∀ i,
      transformerStackFinalReal eps finalLn layers heads scores
            (fun q i => (embed q i : Real)) q i ≤ (final.2 q i : Real) := by
    intro q hq i
    simpa [final] using (hfinal q i).2
  have hbounds := intervalBoundsOn_spec active hactive final.1 final.2
    (fun q i => transformerStackFinalReal eps finalLn layers heads scores
      (fun q i => (embed q i : Real)) q i)
    hlo hhi
  simpa [bounds, gpt2ResidualIntervalBoundsActive, final, baseLo, baseHi] using
    hbounds q hq i

/-- Package GPT-2 residual bounds into a residual-interval certificate. -/
theorem gpt2ResidualIntervalBoundsActive_sound
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (active : Finset (Fin seq)) (hactive : active.Nonempty) (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (embed : Fin seq → Fin dModel → Rat)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
    let bounds := gpt2ResidualIntervalBoundsActive active hactive eps layers heads finalLn embed
    let cert : Circuit.ResidualIntervalCert dModel := { lo := bounds.1, hi := bounds.2 }
    Circuit.ResidualIntervalBounds cert ∧
      ∀ q, q ∈ active → ∀ i,
        (cert.lo i : Real) ≤
            transformerStackFinalReal eps finalLn layers heads scores
              (fun q i => (embed q i : Real)) q i ∧
          transformerStackFinalReal eps finalLn layers heads scores
              (fun q i => (embed q i : Real)) q i ≤ (cert.hi i : Real) := by
  classical
  intro bounds cert
  have hspec :
      ∀ q, q ∈ active → ∀ i,
        (bounds.1 i : Real) ≤
            transformerStackFinalReal eps finalLn layers heads scores
              (fun q i => (embed q i : Real)) q i ∧
          transformerStackFinalReal eps finalLn layers heads scores
              (fun q i => (embed q i : Real)) q i ≤ (bounds.2 i : Real) := by
    simpa [bounds] using
      (gpt2ResidualIntervalBoundsActive_spec (active := active) (hactive := hactive)
        (eps := eps) (layers := layers) (heads := heads) (finalLn := finalLn)
        (scores := scores) (embed := embed) (hne := hne) (heps := heps) (hsqrt := hsqrt))
  have hbounds : Circuit.ResidualIntervalBounds cert := by
    refine { lo_le_hi := ?_ }
    intro i
    rcases hactive with ⟨q0, hq0⟩
    have hq := hspec q0 hq0 i
    have hreal : (bounds.1 i : Real) ≤ (bounds.2 i : Real) := hq.1.trans hq.2
    exact (ratToReal_le_iff (x := bounds.1 i) (y := bounds.2 i)).1 hreal
  refine And.intro hbounds ?_
  intro q hq i
  have hq' := hspec q hq i
  simpa [cert] using hq'

end Bounds

end Sound

end Nfp
