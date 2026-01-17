-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Real.Basic
public import Nfp.Circuit.Layers.Softmax
public import Nfp.Core.Basic
public import Nfp.Model.Gpt2
public import Nfp.Bounds.Cache
public import Nfp.Bounds.LayerNorm
public import Nfp.Bounds.MatrixNorm
public import Nfp.Bounds.Mlp

/-!
Interval bounds for multi-head attention and transformer layers.
-/

public section

namespace Nfp


namespace Bounds

open scoped BigOperators

/-- Real-valued attention output for a query token and model coordinate. -/
noncomputable def attentionOutputReal {seq dModel dHead numHeads : Nat} [NeZero seq]
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (x : Fin seq → Fin dModel → Real)
    (q : Fin seq) (i : Fin dModel) : Real :=
  let lnOut : Fin seq → Fin dModel → Real := fun k j =>
    layerNormRealOfReal eps ln1Gamma ln1Beta (x k) j
  let headValue : Fin numHeads → Fin seq → Fin dHead → Real := fun h k d =>
    dotProduct (fun j => ((heads h).wv j d : Real)) (lnOut k) + (heads h).bv d
  let headWeights : Fin numHeads → Fin seq → Fin seq → Real := fun h q k =>
    Circuit.softmax (scores h q) k
  let headOutput : Fin numHeads → Fin seq → Fin dHead → Real := fun h q d =>
    dotProduct (headWeights h q) (fun k => headValue h k d)
  let headProj : Fin numHeads → Fin seq → Fin dModel → Real := fun h q j =>
    dotProduct (fun d => ((heads h).wo j d : Real)) (fun d => headOutput h q d)
  (∑ h, headProj h q i) + (attnBias i : Real)

/-- Unfolding lemma for `attentionOutputReal`. -/
private theorem attentionOutputReal_def {seq dModel dHead numHeads : Nat} [NeZero seq]
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (x : Fin seq → Fin dModel → Real)
    (q : Fin seq) (i : Fin dModel) :
    attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i =
      let lnOut : Fin seq → Fin dModel → Real := fun k j =>
        layerNormRealOfReal eps ln1Gamma ln1Beta (x k) j
      let headValue : Fin numHeads → Fin seq → Fin dHead → Real := fun h k d =>
        dotProduct (fun j => ((heads h).wv j d : Real)) (lnOut k) + (heads h).bv d
      let headWeights : Fin numHeads → Fin seq → Fin seq → Real := fun h q k =>
        Circuit.softmax (scores h q) k
      let headOutput : Fin numHeads → Fin seq → Fin dHead → Real := fun h q d =>
        dotProduct (headWeights h q) (fun k => headValue h k d)
      let headProj : Fin numHeads → Fin seq → Fin dModel → Real := fun h q j =>
        dotProduct (fun d => ((heads h).wo j d : Real)) (fun d => headOutput h q d)
      (∑ h, headProj h q i) + (attnBias i : Real) := rfl

/-- Interval bounds for multi-head attention outputs from interval inputs. -/
def attentionOutputBounds {dModel dHead numHeads : Nat}
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) :
    (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let absBound := intervalAbsBound lo hi
  let ln := layerNormAbsBounds eps ln1Gamma ln1Beta absBound
  let lnLo := ln.1
  let lnHi := ln.2
  let vLo : Fin numHeads → Fin dHead → Rat := fun h d =>
    dotIntervalLower (fun j => (heads h).wv j d) lnLo lnHi + (heads h).bv d
  let vHi : Fin numHeads → Fin dHead → Rat := fun h d =>
    dotIntervalUpper (fun j => (heads h).wv j d) lnLo lnHi + (heads h).bv d
  let headLo : Fin numHeads → Fin dModel → Rat := fun h i =>
    dotIntervalLower (fun d => (heads h).wo i d) (vLo h) (vHi h)
  let headHi : Fin numHeads → Fin dModel → Rat := fun h i =>
    dotIntervalUpper (fun d => (heads h).wo i d) (vLo h) (vHi h)
  let sumLo : Fin dModel → Rat := fun i => ∑ h, headLo h i
  let sumHi : Fin dModel → Rat := fun i => ∑ h, headHi h i
  (fun i => sumLo i + attnBias i, fun i => sumHi i + attnBias i)

private theorem sum_weighted_const {seq : Nat} (w : Fin seq → Real) (c : Real)
    (hsum : ∑ k, w k = 1) :
    ∑ k, w k * c = c := by
  calc
    ∑ k, w k * c = (∑ k, w k) * c := by
      simpa using
        (Finset.sum_mul (s := (Finset.univ : Finset (Fin seq))) (f := w) (a := c)).symm
    _ = c := by simp [hsum]

/-- Weighted dot-products preserve interval bounds. -/
theorem dotProduct_bounds_of_weights {seq : Nat} {lo hi : Real}
    {vals w : Fin seq → Real}
    (hlo : ∀ k, lo ≤ vals k) (hhi : ∀ k, vals k ≤ hi)
    (hnonneg : ∀ k, 0 ≤ w k) (hsum : ∑ k, w k = 1) :
    lo ≤ dotProduct w vals ∧ dotProduct w vals ≤ hi := by
  have hsum_lo : ∑ k, w k * lo ≤ ∑ k, w k * vals k := by
    refine Finset.sum_le_sum ?_
    intro k _
    exact mul_le_mul_of_nonneg_left (hlo k) (hnonneg k)
  have hsum_lo' : ∑ k, w k * lo = lo := sum_weighted_const w lo hsum
  have hlow : lo ≤ dotProduct w vals := by
    have hsum_le : lo ≤ ∑ k, w k * vals k := by
      simpa [hsum_lo'] using hsum_lo
    simpa [dotProduct] using hsum_le
  have hsum_hi : ∑ k, w k * vals k ≤ ∑ k, w k * hi := by
    refine Finset.sum_le_sum ?_
    intro k _
    exact mul_le_mul_of_nonneg_left (hhi k) (hnonneg k)
  have hsum_hi' : ∑ k, w k * hi = hi := sum_weighted_const w hi hsum
  have hhigh : dotProduct w vals ≤ hi := by
    have hsum_le : ∑ k, w k * vals k ≤ hi := by
      simpa [hsum_hi'] using hsum_hi
    simpa [dotProduct] using hsum_le
  exact ⟨hlow, hhigh⟩

/-- `attentionOutputBounds` soundness for real attention outputs. -/
theorem attentionOutputBounds_spec {seq dModel dHead numHeads : Nat} [NeZero seq]
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := attentionOutputBounds eps ln1Gamma ln1Beta heads attnBias lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤
          attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ∧
        attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ≤
          (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let absBound := intervalAbsBound lo hi
  let lnBounds := layerNormAbsBounds eps ln1Gamma ln1Beta absBound
  let lnLo := lnBounds.1
  let lnHi := lnBounds.2
  let lnOut : Fin seq → Fin dModel → Real := fun k j =>
    layerNormRealOfReal eps ln1Gamma ln1Beta (x k) j
  let vLo : Fin numHeads → Fin dHead → Rat := fun h d =>
    dotIntervalLower (fun j => (heads h).wv j d) lnLo lnHi + (heads h).bv d
  let vHi : Fin numHeads → Fin dHead → Rat := fun h d =>
    dotIntervalUpper (fun j => (heads h).wv j d) lnLo lnHi + (heads h).bv d
  let headLo : Fin numHeads → Fin dModel → Rat := fun h j =>
    dotIntervalLower (fun d => (heads h).wo j d) (vLo h) (vHi h)
  let headHi : Fin numHeads → Fin dModel → Rat := fun h j =>
    dotIntervalUpper (fun d => (heads h).wo j d) (vLo h) (vHi h)
  let sumLo : Fin dModel → Rat := fun j => ∑ h, headLo h j
  let sumHi : Fin dModel → Rat := fun j => ∑ h, headHi h j
  let headValue : Fin numHeads → Fin seq → Fin dHead → Real := fun h k d =>
    dotProduct (fun j => ((heads h).wv j d : Real)) (lnOut k) + (heads h).bv d
  let softmaxWeights : Fin numHeads → Circuit.SoftmaxWeights seq := fun h =>
    Circuit.softmaxWeights (scores h)
  let headWeights : Fin numHeads → Fin seq → Fin seq → Real := fun h q k =>
    Circuit.softmax (scores h q) k
  let headOutput : Fin numHeads → Fin seq → Fin dHead → Real := fun h q d =>
    dotProduct (headWeights h q) (fun k => headValue h k d)
  let headProj : Fin numHeads → Fin seq → Fin dModel → Real := fun h q j =>
    dotProduct (fun d => ((heads h).wo j d : Real)) (fun d => headOutput h q d)
  have habs : ∀ q i, |x q i| ≤ (absBound : Real) := by
    intro q i
    have hbound :
        |x q i| ≤ max |(lo i : Real)| |(hi i : Real)| :=
      abs_le_max_abs_abs_of_interval_real (hlo q i) (hhi q i)
    have hsup : max |lo i| |hi i| ≤ intervalAbsBound lo hi :=
      max_abs_le_intervalAbsBound lo hi i
    have hsup_real :
        max |(lo i : Real)| |(hi i : Real)| ≤ (absBound : Real) := by
      have hsup' : ratToReal (max |lo i| |hi i|) ≤ ratToReal absBound :=
        ratToReal_le_of_le hsup
      simpa [ratToReal_abs, ratToReal_max, ratToReal_def] using hsup'
    exact le_trans hbound hsup_real
  have hln_bounds : ∀ q i, (lnLo i : Real) ≤ lnOut q i ∧ lnOut q i ≤ (lnHi i : Real) := by
    intro q i
    have hln := layerNormAbsBounds_spec_real eps ln1Gamma ln1Beta absBound (x q) hne heps hsqrt
      (fun j => habs q j)
    simpa [lnBounds, lnLo, lnHi, lnOut] using hln i
  have hval_bounds :
      ∀ h k d,
        (vLo h d : Real) ≤ headValue h k d ∧
          headValue h k d ≤ (vHi h d : Real) := by
    intro h k d
    have hln := hln_bounds k
    have hlo' : ∀ j, (lnLo j : Real) ≤ lnOut k j := fun j => (hln j).1
    have hhi' : ∀ j, lnOut k j ≤ (lnHi j : Real) := fun j => (hln j).2
    have hlow' :=
      dotIntervalLower_le_dotProduct_real_add (v := fun j => (heads h).wv j d)
        (lo := lnLo) (hi := lnHi) (x := lnOut k) (b := ((heads h).bv d : Real)) hlo' hhi'
    have hhigh' :=
      dotProduct_le_dotIntervalUpper_real_add (v := fun j => (heads h).wv j d)
        (lo := lnLo) (hi := lnHi) (x := lnOut k) (b := ((heads h).bv d : Real)) hlo' hhi'
    constructor
    · simpa [headValue, vLo] using hlow'
    · simpa [headValue, vHi] using hhigh'
  have hhead_output_bounds :
      ∀ h q d,
        (vLo h d : Real) ≤ headOutput h q d ∧
          headOutput h q d ≤ (vHi h d : Real) := by
    intro h q d
    have hvals := hval_bounds h
    have hlo' : ∀ k,
        (vLo h d : Real) ≤ headValue h k d := fun k => (hvals k d).1
    have hhi' : ∀ k,
        headValue h k d ≤ (vHi h d : Real) := fun k => (hvals k d).2
    have hnonneg : ∀ k, 0 ≤ headWeights h q k := by
      intro k
      simpa [headWeights, softmaxWeights, Circuit.softmaxWeights_weights] using
        (softmaxWeights h).nonneg q k
    have hsum : ∑ k, headWeights h q k = 1 := by
      simpa [headWeights, softmaxWeights, Circuit.softmaxWeights_weights] using
        (softmaxWeights h).sum_one q
    have h := dotProduct_bounds_of_weights (lo := (vLo h d : Real)) (hi := (vHi h d : Real))
      (vals := fun k => headValue h k d) (w := headWeights h q)
      hlo' hhi' hnonneg hsum
    simpa [headOutput] using h
  have hproj_bounds :
      ∀ h q i,
        (headLo h i : Real) ≤ headProj h q i ∧ headProj h q i ≤ (headHi h i : Real) := by
    intro h q i
    have hout := hhead_output_bounds h q
    have hlo' : ∀ d,
        (vLo h d : Real) ≤ headOutput h q d := fun d => (hout d).1
    have hhi' : ∀ d,
        headOutput h q d ≤ (vHi h d : Real) := fun d => (hout d).2
    have hlow :=
      dotIntervalLower_le_dotProduct_real (v := fun d => (heads h).wo i d)
        (lo := vLo h) (hi := vHi h)
        (x := fun d => headOutput h q d) hlo' hhi'
    have hhigh :=
      dotProduct_le_dotIntervalUpper_real (v := fun d => (heads h).wo i d)
        (lo := vLo h) (hi := vHi h)
        (x := fun d => headOutput h q d) hlo' hhi'
    constructor
    · simpa [headProj, headLo] using hlow
    · simpa [headProj, headHi] using hhigh
  have hsum_bounds :
      (sumLo i : Real) ≤ ∑ h, headProj h q i ∧
        ∑ h, headProj h q i ≤ (sumHi i : Real) := by
    have hlow : (sumLo i : Real) ≤ ∑ h, headProj h q i := by
      have hsum :=
        Finset.sum_le_sum (s := (Finset.univ : Finset (Fin numHeads)))
          (fun h _ => (hproj_bounds h q i).1)
      simpa [sumLo, Linear.ratToReal_sum_univ] using hsum
    have hhigh : ∑ h, headProj h q i ≤ (sumHi i : Real) := by
      have hsum :=
        Finset.sum_le_sum (s := (Finset.univ : Finset (Fin numHeads)))
          (fun h _ => (hproj_bounds h q i).2)
      simpa [sumHi, Linear.ratToReal_sum_univ] using hsum
    exact ⟨hlow, hhigh⟩
  have hlow :
      (sumLo i : Real) + (attnBias i : Real) ≤
        (∑ h, headProj h q i) + (attnBias i : Real) := by
    simpa [add_comm] using
      add_le_add_left hsum_bounds.1 (attnBias i : Real)
  have hhigh :
      (∑ h, headProj h q i) + (attnBias i : Real) ≤
        (sumHi i : Real) + (attnBias i : Real) := by
    simpa [add_comm] using
      add_le_add_left hsum_bounds.2 (attnBias i : Real)
  have hreal :
      attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i =
        (∑ h, headProj h q i) + (attnBias i : Real) := by
    simp [attentionOutputReal, lnOut, headValue, headWeights, headOutput, headProj]
  have hlo :
      (bounds.1 i : Real) ≤
        attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i := by
    simpa [bounds, attentionOutputBounds, absBound, lnBounds, lnLo, lnHi, vLo, vHi, headLo, headHi,
      sumLo, sumHi, hreal] using hlow
  have hhi :
      attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ≤
        (bounds.2 i : Real) := by
    simpa [bounds, attentionOutputBounds, absBound, lnBounds, lnLo, lnHi, vLo, vHi, headLo, headHi,
      sumLo, sumHi, hreal] using hhigh
  exact And.intro hlo hhi

/-- Interval bounds for the attention residual path. -/
def attentionResidualBounds {dModel dHead numHeads : Nat}
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) :
    (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let attn := attentionOutputBounds eps ln1Gamma ln1Beta heads attnBias lo hi
  (fun i => lo i + attn.1 i, fun i => hi i + attn.2 i)

/-- `attentionResidualBounds` soundness for attention residual outputs. -/
theorem attentionResidualBounds_spec {seq dModel dHead numHeads : Nat} [NeZero seq]
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := attentionResidualBounds eps ln1Gamma ln1Beta heads attnBias lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤
        x q i + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ∧
        x q i + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ≤
          (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let attn := attentionOutputBounds eps ln1Gamma ln1Beta heads attnBias lo hi
  have hattn :=
    attentionOutputBounds_spec eps ln1Gamma ln1Beta heads attnBias scores lo hi x
      hne heps hsqrt hlo hhi q i
  have hlow := add_le_add (hlo q i) hattn.1
  have hhigh := add_le_add (hhi q i) hattn.2
  constructor
  · simpa [bounds, attentionResidualBounds, attn] using hlow
  · simpa [bounds, attentionResidualBounds, attn] using hhigh

/-- Interval bounds for a full transformer layer (attention + MLP). -/
def transformerLayerBounds {dModel dHead numHeads hidden : Nat}
    (eps : Rat)
    (ln1Gamma ln1Beta ln2Gamma ln2Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (mlpWIn : Fin dModel → Fin hidden → Rat) (mlpBIn : Fin hidden → Rat)
    (mlpWOut : Fin hidden → Fin dModel → Rat) (mlpBOut : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) :
    (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let loCached := cacheBound lo
  let hiCached := cacheBound hi
  let attn := attentionResidualBounds eps ln1Gamma ln1Beta heads attnBias loCached hiCached
  let attnLo := cacheBound attn.1
  let attnHi := cacheBound attn.2
  let out := layerNormAbsMlpResidualBounds eps ln2Gamma ln2Beta mlpWIn mlpBIn mlpWOut mlpBOut
    attnLo attnHi
  let outLo := cacheBound out.1
  let outHi := cacheBound out.2
  (outLo, outHi)

/-- `transformerLayerBounds` soundness for full transformer-layer outputs. -/
theorem transformerLayerBounds_spec {seq dModel dHead numHeads hidden : Nat} [NeZero seq]
    (eps : Rat)
    (ln1Gamma ln1Beta ln2Gamma ln2Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (mlpWIn : Fin dModel → Fin hidden → Rat) (mlpBIn : Fin hidden → Rat)
    (mlpWOut : Fin hidden → Fin dModel → Rat) (mlpBOut : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := transformerLayerBounds eps ln1Gamma ln1Beta ln2Gamma ln2Beta heads attnBias
      mlpWIn mlpBIn mlpWOut mlpBOut lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤
        x q i + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i +
          mlpReal mlpWIn mlpBIn mlpWOut mlpBOut
            (layerNormRealOfReal eps ln2Gamma ln2Beta
              (fun j =>
                x q j + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q j)) i ∧
        x q i + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i +
          mlpReal mlpWIn mlpBIn mlpWOut mlpBOut
            (layerNormRealOfReal eps ln2Gamma ln2Beta
              (fun j =>
                x q j + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q j)) i ≤
          (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let loCached := cacheBound lo
  let hiCached := cacheBound hi
  have hloCached : ∀ q i, (loCached i : Real) ≤ x q i := by
    intro q i
    simpa [loCached, cacheBound_apply] using hlo q i
  have hhiCached : ∀ q i, x q i ≤ (hiCached i : Real) := by
    intro q i
    simpa [hiCached, cacheBound_apply] using hhi q i
  let attn := attentionResidualBounds eps ln1Gamma ln1Beta heads attnBias loCached hiCached
  have hattn := attentionResidualBounds_spec eps ln1Gamma ln1Beta heads attnBias scores
    loCached hiCached x hne heps hsqrt hloCached hhiCached q
  let attnLo := cacheBound attn.1
  let attnHi := cacheBound attn.2
  let y := fun j => x q j + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q j
  have hattnLo : ∀ j, (attnLo j : Real) ≤ y j := by
    intro j
    simpa [attnLo, cacheBound_apply, y] using (hattn j).1
  have hattnHi : ∀ j, y j ≤ (attnHi j : Real) := by
    intro j
    simpa [attnHi, cacheBound_apply, y] using (hattn j).2
  have hmlp := layerNormAbsMlpResidualBounds_spec eps ln2Gamma ln2Beta mlpWIn mlpBIn mlpWOut
    mlpBOut attnLo attnHi y hne heps hsqrt hattnLo hattnHi
  have hmlp_i := hmlp i
  simpa [bounds, transformerLayerBounds, attn, loCached, hiCached, attnLo, attnHi, y,
    cacheBound_apply] using hmlp_i

end Bounds


end Nfp
