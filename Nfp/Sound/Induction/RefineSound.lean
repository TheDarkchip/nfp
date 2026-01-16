-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Induction.LogitDiff
public import Nfp.Sound.Induction.OneHot
public import Nfp.Sound.Induction.Refine

/-!
Soundness lemmas for refine-on-demand overlays.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Circuit
open Nfp.Sound.Bounds

variable {seq dModel dHead : Nat}

/-- Refined score-gap bounds are sound when cache score and KV bounds are sound. -/
theorem scoreGapLoRefinedAt_real_at_of_bounds
    [NeZero seq] (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat)
    (active : Finset (Fin seq))
    (hq_bounds :
      ∀ q d, (cache.qLo q d : Real) ≤ qRealOfInputs inputs q d ∧
        qRealOfInputs inputs q d ≤ (cache.qHi q d : Real))
    (hk_bounds :
      ∀ q d, (cache.kLo q d : Real) ≤ kRealOfInputs inputs q d ∧
        kRealOfInputs inputs q d ≤ (cache.kHi q d : Real))
    (hscore_prev :
      ∀ q, q ∈ active →
        (cache.scoreLoPrev q : Real) ≤ scoresRealOfInputs inputs q (inputs.prev q))
    (hscore_hi :
      ∀ q k, scoresRealOfInputs inputs q k ≤ (cache.scoreHi q k : Real)) :
    ∀ q, q ∈ active → ∀ k, k ≠ inputs.prev q →
      scoresRealOfInputs inputs q k +
        (scoreGapLoRefinedAt inputs cache budget q k : Real) ≤
          scoresRealOfInputs inputs q (inputs.prev q) := by
  classical
  let scoresReal := scoresRealOfInputs inputs
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  have scoresReal_eq_base_of_not_masked :
      ∀ q k, ¬ masked q k →
        scoresReal q k =
          (inputs.scale : Real) *
            dotProduct (fun d => qRealOfInputs inputs q d)
              (fun d => kRealOfInputs inputs k d) := by
    intro q k hnot
    by_cases hcausal : inputs.maskCausal
    · have hnot_lt : ¬ q < k := by
        intro hlt
        exact hnot ⟨hcausal, hlt⟩
      have hle : k ≤ q := le_of_not_gt hnot_lt
      simp [scoresReal, scoresRealOfInputs_def, hcausal, hle]
    · simp [scoresReal, scoresRealOfInputs_def, hcausal]
  have scoresReal_eq_masked :
      ∀ q k, masked q k → scoresReal q k = (inputs.maskValue : Real) := by
    intro q k hmask
    have hmask' : inputs.maskCausal = true ∧ q < k := by
      simpa [masked] using hmask
    have hle : ¬ k ≤ q := not_le_of_gt hmask'.2
    simp [scoresReal, scoresRealOfInputs_def, hmask'.1, hle]
  have hdot_diff_bounds :
      ∀ q k,
        (dotDiffLoRefinedAt inputs cache budget q k : Real) ≤
            dotProduct (fun d => qRealOfInputs inputs q d)
              (fun d => kRealOfInputs inputs (inputs.prev q) d -
                kRealOfInputs inputs k d) ∧
          dotProduct (fun d => qRealOfInputs inputs q d)
              (fun d => kRealOfInputs inputs (inputs.prev q) d -
                kRealOfInputs inputs k d) ≤
            (dotDiffHiRefinedAt inputs cache budget q k : Real) := by
    intro q k
    have hlo1 : ∀ d, (cache.qLo q d : Real) ≤ qRealOfInputs inputs q d := fun d =>
      (hq_bounds q d).1
    have hhi1 : ∀ d, qRealOfInputs inputs q d ≤ (cache.qHi q d : Real) := fun d =>
      (hq_bounds q d).2
    have hlo2 :
        ∀ d,
          (cache.kLo (inputs.prev q) d - cache.kHi k d : Rat) ≤
            (kRealOfInputs inputs (inputs.prev q) d - kRealOfInputs inputs k d) := by
      intro d
      have hprev_lo := (hk_bounds (inputs.prev q) d).1
      have hk_hi := (hk_bounds k d).2
      have h := sub_le_sub hprev_lo hk_hi
      simpa [ratToReal_sub] using h
    have hhi2 :
        ∀ d,
          (kRealOfInputs inputs (inputs.prev q) d - kRealOfInputs inputs k d) ≤
            (cache.kHi (inputs.prev q) d - cache.kLo k d : Rat) := by
      intro d
      have hprev_hi := (hk_bounds (inputs.prev q) d).2
      have hk_lo := (hk_bounds k d).1
      have h := sub_le_sub hprev_hi hk_lo
      simpa [ratToReal_sub] using h
    have hspec :=
      _root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth_spec_real
        (dims1 := cache.splitDimsQ q) (dims2 := cache.splitDimsDiffCore budget q k)
        (lo1 := fun d => cache.qLo q d) (hi1 := fun d => cache.qHi q d)
        (lo2 := fun d => cache.kLo (inputs.prev q) d - cache.kHi k d)
        (hi2 := fun d => cache.kHi (inputs.prev q) d - cache.kLo k d)
        (x := fun d => qRealOfInputs inputs q d)
        (y := fun d =>
          kRealOfInputs inputs (inputs.prev q) d - kRealOfInputs inputs k d)
        hlo1 hhi1 hlo2 hhi2
    have hlow' :
        (dotDiffLoRefinedAt inputs cache budget q k : Real) ≤
          dotProduct (fun d => qRealOfInputs inputs q d)
            (fun d => kRealOfInputs inputs (inputs.prev q) d -
              kRealOfInputs inputs k d) := by
      simpa [dotDiffLoRefinedAt_def] using hspec.1
    have hhigh' :
        dotProduct (fun d => qRealOfInputs inputs q d)
            (fun d => kRealOfInputs inputs (inputs.prev q) d -
              kRealOfInputs inputs k d) ≤ (dotDiffHiRefinedAt inputs cache budget q k : Real) := by
      simpa [dotDiffHiRefinedAt_def] using hspec.2
    exact ⟨hlow', hhigh'⟩
  intro q hq k hk
  by_cases hprevmask : masked q (inputs.prev q)
  · have hscore_hi' : scoresReal q k ≤ (cache.scoreHi q k : Real) :=
      hscore_hi q k
    have hscore_prev' : (cache.scoreLoPrev q : Real) ≤ scoresReal q (inputs.prev q) :=
      hscore_prev q hq
    have hsum_le' :
        (cache.scoreLoPrev q : Real) - (cache.scoreHi q k : Real) + scoresReal q k ≤
          (cache.scoreLoPrev q : Real) := by
      have hsub :
          (cache.scoreLoPrev q : Real) - (cache.scoreHi q k : Real) ≤
            (cache.scoreLoPrev q : Real) - scoresReal q k :=
        sub_le_sub_left hscore_hi' (cache.scoreLoPrev q : Real)
      calc
        (cache.scoreLoPrev q : Real) - (cache.scoreHi q k : Real) + scoresReal q k
            ≤ (cache.scoreLoPrev q : Real) - scoresReal q k + scoresReal q k := by
              simpa [add_comm, add_left_comm, add_assoc] using
                (add_le_add_left hsub (scoresReal q k))
        _ = (cache.scoreLoPrev q : Real) := by
          simp [sub_add_cancel]
    calc
      scoresReal q k + (scoreGapLoRefinedAt inputs cache budget q k : Real)
          = (cache.scoreLoPrev q : Real) - (cache.scoreHi q k : Real) + scoresReal q k := by
            simp [scoreGapLoRefinedAt_def, hprevmask, masked, add_comm]
      _ ≤ (cache.scoreLoPrev q : Real) := hsum_le'
      _ ≤ scoresReal q (inputs.prev q) := hscore_prev'
  · by_cases hmask : masked q k
    · have hscore_prev' : (cache.scoreLoPrev q : Real) ≤ scoresReal q (inputs.prev q) :=
        hscore_prev q hq
      have hscore_k : scoresReal q k = (inputs.maskValue : Real) :=
        scoresReal_eq_masked q k hmask
      have hmask' : inputs.maskCausal = true ∧ q < k := by
        simpa [masked] using hmask
      have hnot_lt_prev : ¬ q < inputs.prev q := by
        intro hlt
        exact hprevmask ⟨hmask'.1, hlt⟩
      calc
        scoresReal q k + (scoreGapLoRefinedAt inputs cache budget q k : Real)
            = (inputs.maskValue : Real) + (cache.scoreLoPrev q : Real) -
                (inputs.maskValue : Real) := by
              simp [scoreGapLoRefinedAt_def, hmask', hnot_lt_prev, hscore_k]
        _ = (cache.scoreLoPrev q : Real) := by
              simp [add_sub_cancel_left]
        _ ≤ scoresReal q (inputs.prev q) := hscore_prev'
    · have hdiff := hdot_diff_bounds q k
      have hgap_le :
          (scoreGapLoRefinedAt inputs cache budget q k : Real) ≤
            (inputs.scale : Real) *
              dotProduct (fun d => qRealOfInputs inputs q d)
                (fun d => kRealOfInputs inputs (inputs.prev q) d -
                  kRealOfInputs inputs k d) := by
        by_cases hscale : 0 ≤ inputs.scale
        · have hscale_real : 0 ≤ (inputs.scale : Real) := by
            simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg hscale
          have hle := mul_le_mul_of_nonneg_left hdiff.1 hscale_real
          simpa [scoreGapLoRefinedAt_def, hprevmask, hmask, hscale, masked] using hle
        · have hscale_nonpos : inputs.scale ≤ 0 :=
            le_of_lt (lt_of_not_ge hscale)
          have hscale_real : (inputs.scale : Real) ≤ 0 := by
            simpa [ratToReal_def] using
              (ratToReal_nonpos_iff (x := inputs.scale)).2 hscale_nonpos
          have hle := mul_le_mul_of_nonpos_left hdiff.2 hscale_real
          simpa [scoreGapLoRefinedAt_def, hprevmask, hmask, hscale, masked] using hle
      have hscore_prev :
          scoresReal q (inputs.prev q) =
            (inputs.scale : Real) *
              dotProduct (fun d => qRealOfInputs inputs q d)
                (fun d => kRealOfInputs inputs (inputs.prev q) d) := by
        simpa using
          (scoresReal_eq_base_of_not_masked q (inputs.prev q) hprevmask)
      have hscore_k :
          scoresReal q k =
            (inputs.scale : Real) *
              dotProduct (fun d => qRealOfInputs inputs q d)
                (fun d => kRealOfInputs inputs k d) := by
        simpa using (scoresReal_eq_base_of_not_masked q k hmask)
      have hdot_sub :
          dotProduct (fun d => qRealOfInputs inputs q d)
              (fun d => kRealOfInputs inputs (inputs.prev q) d -
                kRealOfInputs inputs k d) =
            dotProduct (fun d => qRealOfInputs inputs q d)
                (fun d => kRealOfInputs inputs (inputs.prev q) d) -
              dotProduct (fun d => qRealOfInputs inputs q d)
                (fun d => kRealOfInputs inputs k d) := by
        classical
        simpa using
          (Nfp.Sound.Linear.dotProduct_sub_right
            (x := fun d => qRealOfInputs inputs q d)
            (y := fun d => kRealOfInputs inputs (inputs.prev q) d)
            (z := fun d => kRealOfInputs inputs k d))
      have hscore_diff :
          scoresReal q (inputs.prev q) - scoresReal q k =
            (inputs.scale : Real) *
              dotProduct (fun d => qRealOfInputs inputs q d)
                (fun d => kRealOfInputs inputs (inputs.prev q) d -
                  kRealOfInputs inputs k d) := by
        calc
          scoresReal q (inputs.prev q) - scoresReal q k
              =
                (inputs.scale : Real) *
                  dotProduct (fun d => qRealOfInputs inputs q d)
                    (fun d => kRealOfInputs inputs (inputs.prev q) d) -
                (inputs.scale : Real) *
                  dotProduct (fun d => qRealOfInputs inputs q d)
                    (fun d => kRealOfInputs inputs k d) := by
                  simp [hscore_prev, hscore_k]
          _ =
              (inputs.scale : Real) *
                (dotProduct (fun d => qRealOfInputs inputs q d)
                    (fun d => kRealOfInputs inputs (inputs.prev q) d) -
                  dotProduct (fun d => qRealOfInputs inputs q d)
                    (fun d => kRealOfInputs inputs k d)) := by
                  simp [mul_sub]
          _ =
              (inputs.scale : Real) *
                dotProduct (fun d => qRealOfInputs inputs q d)
                  (fun d => kRealOfInputs inputs (inputs.prev q) d -
                    kRealOfInputs inputs k d) := by
                  simp [hdot_sub]
      have hgap_le' :
          (scoreGapLoRefinedAt inputs cache budget q k : Real) ≤
            scoresReal q (inputs.prev q) - scoresReal q k := by
        simpa [hscore_diff] using hgap_le
      have hgap_add := add_le_add_right hgap_le' (scoresReal q k)
      have hgap_add' :
          scoresReal q k + (scoreGapLoRefinedAt inputs cache budget q k : Real) ≤
            scoresReal q (inputs.prev q) := by
        have hcancel :
            scoresReal q k + (scoresReal q (inputs.prev q) - scoresReal q k) =
              scoresReal q (inputs.prev q) := by
          calc
            scoresReal q k + (scoresReal q (inputs.prev q) - scoresReal q k)
                =
                  scoresReal q k + scoresReal q (inputs.prev q) -
                    scoresReal q k := by
                      symm
                      exact add_sub_assoc (scoresReal q k)
                        (scoresReal q (inputs.prev q)) (scoresReal q k)
            _ = scoresReal q (inputs.prev q) := by
              simp [add_sub_cancel_left]
        calc
          scoresReal q k + (scoreGapLoRefinedAt inputs cache budget q k : Real)
              ≤ scoresReal q k + (scoresReal q (inputs.prev q) - scoresReal q k) := hgap_add
          _ = scoresReal q (inputs.prev q) := hcancel
      exact hgap_add'

/-- Refined per-key weight bounds are sound when refined score gaps are sound. -/
theorem weight_bound_at_refinedAt_of_scoreGapLo
    [NeZero seq] (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat)
    (active : Finset (Fin seq))
    (hscore_gap_real_at :
      ∀ q, q ∈ active → ∀ k, k ≠ inputs.prev q →
        scoresRealOfInputs inputs q k +
          (scoreGapLoRefinedAt inputs cache budget q k : Real) ≤
            scoresRealOfInputs inputs q (inputs.prev q)) :
    ∀ q, q ∈ active → ∀ k, k ≠ inputs.prev q →
      Circuit.softmax (scoresRealOfInputs inputs q) k ≤
        (weightBoundAtRefinedAt inputs cache budget q k : Real) := by
  classical
  intro q hq k hk
  refine
    Sound.weight_bound_at_of_scoreGapLo
      (active := active)
      (prev := inputs.prev)
      (scoresReal := scoresRealOfInputs inputs)
      (scoreGapLo := scoreGapLoRefinedAt inputs cache budget)
      (weightBoundAt := weightBoundAtRefinedAt inputs cache budget)
      (hweightBoundAt := ?_)
      (hscore_gap_real_at := hscore_gap_real_at)
      q hq k hk
  intro q' k' hk'
  simp [weightBoundAtRefinedAt_def, hk']

/-- Overlayed per-key bounds are sound when base and refined bounds are sound. -/
theorem weight_bounds_at_overlay_of_refined
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (spec : InductionHeadRefineSpec seq)
    (active : Finset (Fin seq))
    (hbase :
      ∀ q, q ∈ active → ∀ k, k ≠ inputs.prev q →
        Circuit.softmax (scoresRealOfInputs inputs q) k ≤
          (cache.weightBoundAt q k : Real))
    (hrefine :
      ∀ q, q ∈ active → ∀ k, k ≠ inputs.prev q → k ∈ spec.refineKeys q →
        Circuit.softmax (scoresRealOfInputs inputs q) k ≤
          (weightBoundAtRefinedAt inputs cache spec.splitBudgetDiffRefined q k : Real)) :
    ∀ q, q ∈ active → ∀ k, k ≠ inputs.prev q →
      Circuit.softmax (scoresRealOfInputs inputs q) k ≤
        (weightBoundAtOverlay inputs cache spec q k : Real) := by
  classical
  intro q hq k hk
  by_cases hmem : k ∈ spec.refineKeys q
  · have h := hrefine q hq k hk hmem
    simpa [weightBoundAtOverlay_def, hk, hmem] using h
  · have h := hbase q hq k hk
    simpa [weightBoundAtOverlay_def, hk, hmem] using h

/-- One-hot bounds derived from an overlayed per-key bound. -/
theorem oneHot_bounds_at_overlay
    [NeZero seq] (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq)
    (hcert : c = cache.cert)
    (spec : InductionHeadRefineSpec seq)
    (hweight_overlay :
      ∀ q, q ∈ c.active → ∀ k, k ≠ c.prev q →
        Circuit.softmax (scoresRealOfInputs inputs q) k ≤
          (weightBoundAtOverlay inputs cache spec q k : Real)) :
    ∀ q, q ∈ c.active →
      Layers.OneHotApproxBoundsOnActive (Val := Real)
        (epsAtOverlay cache (weightBoundAtOverlay inputs cache spec) q : Real)
        (fun q' => q' = q) c.prev
        (fun q' k => Circuit.softmax (scoresRealOfInputs inputs q') k) := by
  classical
  intro q hq
  refine
    Sound.oneHot_bounds_at_of_weight_bounds
      (active := c.active)
      (prev := c.prev)
      (scoresReal := scoresRealOfInputs inputs)
      (weightBoundAt := weightBoundAtOverlay inputs cache spec)
      (epsAt := epsAtOverlay cache (weightBoundAtOverlay inputs cache spec))
      (hepsAt := ?_)
      (hweight_bounds := ?_) q hq
  · intro q'
    cases hcert
    simp [epsAtOverlay_def]
  · intro q' hq' k hk
    exact hweight_overlay q' hq' k hk

/-- The refined unweighted logit-diff lower bound is sound on active queries. -/
theorem logitDiffLowerBoundRefinedFromCache_le
    [NeZero seq] (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (logitCache : LogitDiffCache seq)
    (spec : InductionHeadRefineSpec seq)
    (hcert : c = cache.cert)
    (hcache : logitCache = logitDiffCache c)
    (hsound : InductionHeadCertSound inputs c)
    (hweight_overlay :
      ∀ q, q ∈ c.active → ∀ k, k ≠ c.prev q →
        Circuit.softmax (scoresRealOfInputs inputs q) k ≤
          (weightBoundAtOverlay inputs cache spec q k : Real))
    {lb : Rat}
    (hbound : logitDiffLowerBoundRefinedFromCache inputs cache c logitCache spec = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  have honeHot :
      ∀ q, q ∈ c.active →
        Layers.OneHotApproxBoundsOnActive (Val := Real)
          (epsAtOverlay cache (weightBoundAtOverlay inputs cache spec) q : Real)
          (fun q' => q' = q) c.prev
          (fun q' k => Circuit.softmax (scoresRealOfInputs inputs q') k) := by
    intro q hq
    exact oneHot_bounds_at_overlay (inputs := inputs) (cache := cache) (c := c) (hcert := hcert)
      (spec := spec) (hweight_overlay := hweight_overlay) q hq
  have hbound' :
      logitDiffLowerBoundFromCacheWithEps c (logitDiffCache c)
        (epsAtOverlay cache (weightBoundAtOverlay inputs cache spec)) = some lb := by
    simpa [logitDiffLowerBoundRefinedFromCache_def, hcache] using hbound
  exact
    logitDiffLowerBoundFromCacheWithEps_le
      (inputs := inputs)
      (c := c)
      (epsAtCustom := epsAtOverlay cache (weightBoundAtOverlay inputs cache spec))
      (hsound := hsound)
      (honeHot := honeHot)
      (hbound := hbound')
      (hq := hq)

/-- Refine-on-demand logit-diff lower bound using a supplied refinement spec is sound. -/
theorem logitDiffLowerBoundRefineOnDemandWithSpec_le
    [NeZero seq] (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (logitCache : LogitDiffCache seq)
    (spec : InductionHeadRefineSpec seq)
    (hcert : c = cache.cert)
    (hcache : logitCache = logitDiffCache c)
    (hsound : InductionHeadCertSound inputs c)
    (hweight_overlay :
      ∀ q, q ∈ c.active → ∀ k, k ≠ c.prev q →
        Circuit.softmax (scoresRealOfInputs inputs q) k ≤
          (weightBoundAtOverlay inputs cache spec q k : Real))
    {lb : Rat}
    (hbound :
      logitDiffLowerBoundRefineOnDemandWithSpec inputs cache c logitCache spec = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  have honeHot :
      ∀ q, q ∈ c.active →
        Layers.OneHotApproxBoundsOnActive (Val := Real)
          ((logitDiffCache c).epsAt q : Real)
          (fun q' => q' = q) c.prev
          (fun q' k => Circuit.softmax (scoresRealOfInputs inputs q') k) := by
    intro q hq
    have h := hsound.oneHot_bounds_at q hq
    have heps : (logitDiffCache c).epsAt q = c.epsAt q := by
      simp [logitDiffCache_def, Bounds.cacheBoundTask_apply]
    simpa [heps] using h
  have hbase_le :
      ∀ {lb0 : Rat},
        logitDiffLowerBoundFromCache c logitCache = some lb0 →
          (lb0 : Real) ≤ headLogitDiff inputs q := by
    intro lb0 hbound0
    have hbound0' :
        logitDiffLowerBoundFromCache c (logitDiffCache c) = some lb0 := by
      simpa [hcache] using hbound0
    have hbound0'' :
        logitDiffLowerBoundFromCacheWithEps c (logitDiffCache c)
          (logitDiffCache c).epsAt = some lb0 := by
      simpa [logitDiffLowerBoundFromCache_eq_withEps] using hbound0'
    exact
      logitDiffLowerBoundFromCacheWithEps_le
        (inputs := inputs)
        (c := c)
        (epsAtCustom := (logitDiffCache c).epsAt)
        (hsound := hsound)
        (honeHot := honeHot)
        (hbound := hbound0'')
        (hq := hq)
  cases h0 : logitDiffLowerBoundFromCache c logitCache with
  | none =>
      simp [logitDiffLowerBoundRefineOnDemandWithSpec_def, h0] at hbound
  | some lb0 =>
      by_cases hnonpos : lb0 ≤ 0
      · cases h1 : logitDiffLowerBoundRefinedFromCache inputs cache c logitCache spec with
        | none =>
            have hlb : lb = lb0 := by
              simpa [logitDiffLowerBoundRefineOnDemandWithSpec_def, h0, hnonpos, h1] using
                hbound.symm
            have hbase := hbase_le (lb0 := lb0) h0
            simpa [hlb] using hbase
        | some lb1 =>
            have hlb : lb = max lb0 lb1 := by
              simpa [logitDiffLowerBoundRefineOnDemandWithSpec_def, h0, hnonpos, h1] using
                hbound.symm
            have hbase := hbase_le (lb0 := lb0) h0
            have hrefine :=
              logitDiffLowerBoundRefinedFromCache_le
                (inputs := inputs)
                (cache := cache)
                (c := c)
                (logitCache := logitCache)
                (spec := spec)
                (hcert := hcert)
                (hcache := hcache)
                (hsound := hsound)
                (hweight_overlay := hweight_overlay)
                (hbound := h1)
                (hq := hq)
            have hmax' :
                max (lb0 : Real) (lb1 : Real) ≤ headLogitDiff inputs q := by
              exact max_le_iff.mpr ⟨hbase, hrefine⟩
            have hmax : (max lb0 lb1 : Real) ≤ headLogitDiff inputs q := by
              simpa [ratToReal_max] using hmax'
            simpa [hlb] using hmax
      · have hlb : lb = lb0 := by
          simpa [logitDiffLowerBoundRefineOnDemandWithSpec_def, h0, hnonpos] using hbound.symm
        have hbase := hbase_le (lb0 := lb0) h0
        simpa [hlb] using hbase

/-- Refine-on-demand logit-diff lower bound using argmin refinement keys is sound. -/
theorem logitDiffLowerBoundRefineOnDemand_le
    [NeZero seq] (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (logitCache : LogitDiffCache seq)
    (hcert : c = cache.cert)
    (hcache : logitCache = logitDiffCache c)
    (hsound : InductionHeadCertSound inputs c)
    (hweight_overlay :
      let refineBudget := max 1 cache.splitBudgetDiffRefined
      ∀ q0 : Fin seq,
        let spec := refineSpecForQueryWithWeightOnes inputs cache q0 refineBudget
        ∀ q, q ∈ c.active → ∀ k, k ≠ c.prev q →
          Circuit.softmax (scoresRealOfInputs inputs q) k ≤
            (weightBoundAtOverlay inputs cache spec q k : Real))
    {lb : Rat}
    (hbound : logitDiffLowerBoundRefineOnDemand inputs cache c logitCache = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  have hbase_le :
      ∀ {lb0 : Rat},
        logitDiffLowerBoundFromCache c logitCache = some lb0 →
          (lb0 : Real) ≤ headLogitDiff inputs q := by
    intro lb0 hbound0
    have hbound0' :
        logitDiffLowerBoundFromCache c (logitDiffCache c) = some lb0 := by
      simpa [hcache] using hbound0
    have hbound0'' :
        logitDiffLowerBoundFromCacheWithEps c (logitDiffCache c)
          (logitDiffCache c).epsAt = some lb0 := by
      simpa [logitDiffLowerBoundFromCache_eq_withEps] using hbound0'
    exact
      logitDiffLowerBoundFromCacheWithEps_le
        (inputs := inputs)
        (c := c)
        (epsAtCustom := (logitDiffCache c).epsAt)
        (hsound := hsound)
        (honeHot := by
          intro q' hq'
          have h := hsound.oneHot_bounds_at q' hq'
          have heps : (logitDiffCache c).epsAt q' = c.epsAt q' := by
            simp [logitDiffCache_def, Bounds.cacheBoundTask_apply]
          simpa [heps] using h)
        (hbound := hbound0'')
        (hq := hq)
  cases h0 : logitDiffLowerBoundFromCache c logitCache with
  | none =>
      simp [logitDiffLowerBoundRefineOnDemand_def, h0] at hbound
  | some lb0 =>
      by_cases hnonpos : lb0 ≤ 0
      · cases hargmin : logitDiffLowerBoundArgminFromCache c logitCache with
        | none =>
            have hlb : lb = lb0 := by
              simpa [logitDiffLowerBoundRefineOnDemand_def, h0, hnonpos, hargmin] using
                hbound.symm
            have hbase := hbase_le (lb0 := lb0) h0
            simpa [hlb] using hbase
        | some q0 =>
            let refineBudget := max 1 cache.splitBudgetDiffRefined
            let spec := refineSpecForQueryWithWeightOnes inputs cache q0 refineBudget
            cases h1 :
                logitDiffLowerBoundRefinedFromCache inputs cache c logitCache spec with
            | none =>
                have hlb : lb = lb0 := by
                  simpa [logitDiffLowerBoundRefineOnDemand_def, h0, hnonpos, hargmin, h1,
                    spec, refineBudget] using hbound.symm
                have hbase := hbase_le (lb0 := lb0) h0
                simpa [hlb] using hbase
            | some lb1 =>
                have hlb : lb = max lb0 lb1 := by
                  simpa [logitDiffLowerBoundRefineOnDemand_def, h0, hnonpos, hargmin, h1,
                    spec, refineBudget] using hbound.symm
                have hbase := hbase_le (lb0 := lb0) h0
                have hweight_overlay' :
                    ∀ q, q ∈ c.active → ∀ k, k ≠ c.prev q →
                      Circuit.softmax (scoresRealOfInputs inputs q) k ≤
                        (weightBoundAtOverlay inputs cache spec q k : Real) := by
                  simpa [spec, refineBudget] using hweight_overlay q0
                have hrefine :=
                  logitDiffLowerBoundRefinedFromCache_le
                    (inputs := inputs)
                    (cache := cache)
                    (c := c)
                    (logitCache := logitCache)
                    (spec := spec)
                    (hcert := hcert)
                    (hcache := hcache)
                    (hsound := hsound)
                    (hweight_overlay := hweight_overlay')
                    (hbound := h1)
                    (hq := hq)
                have hmax' :
                    max (lb0 : Real) (lb1 : Real) ≤ headLogitDiff inputs q := by
                  exact max_le_iff.mpr ⟨hbase, hrefine⟩
                have hmax : (max lb0 lb1 : Real) ≤ headLogitDiff inputs q := by
                  simpa [ratToReal_max] using hmax'
                simpa [hlb] using hmax
      · have hlb : lb = lb0 := by
          simpa [logitDiffLowerBoundRefineOnDemand_def, h0, hnonpos] using hbound.symm
        have hbase := hbase_le (lb0 := lb0) h0
        simpa [hlb] using hbase

end Sound

end Nfp
