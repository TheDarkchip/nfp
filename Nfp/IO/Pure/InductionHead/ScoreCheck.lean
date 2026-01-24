-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.Interval
public import Nfp.Bounds.LayerNorm
public import Nfp.Circuit.Cert.Basic
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.ModelSlice
public import Nfp.Sound.Induction.ScoreBounds

/-!
Pure score-bound checks for anchoring induction-head certificate margins.
-/

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

open Nfp.Bounds
open Nfp.Circuit
open Nfp.Sound

/-!
Score-bound checks derived directly from certificate scores.
-/

/--
Check that certificate margins/weight tolerances are justified by score gaps in the certificate.

This enforces:
- `margin` is no larger than the score-gap bound;
- `weightBoundAt` and `epsAt` are at least the derived bounds.
-/
def scoreBoundsWithinScores {seq : Nat}
    (cert : Circuit.InductionHeadCert seq) : Bool :=
  let scoreGapLo := Sound.scoreGapLoOfBounds cert.prev cert.scores cert.scores
  let weightBoundAt := Sound.weightBoundAtOfScoreGap cert.prev scoreGapLo
  let epsAt := Sound.epsAtOfWeightBoundAt cert.prev weightBoundAt
  finsetAll cert.active (fun q =>
    decide (epsAt q ≤ cert.epsAt q) &&
      finsetAll ((Finset.univ : Finset (Fin seq)).erase (cert.prev q)) (fun k =>
        decide (cert.margin ≤ scoreGapLo q k) &&
          decide (weightBoundAt q k ≤ cert.weightBoundAt q k)))

/-- Soundness of `scoreBoundsWithinScores` for the derived inequalities. -/
theorem scoreBoundsWithinScores_sound {seq : Nat}
    (cert : Circuit.InductionHeadCert seq) :
    scoreBoundsWithinScores cert = true →
      let scoreGapLo := Sound.scoreGapLoOfBounds cert.prev cert.scores cert.scores
      let weightBoundAt := Sound.weightBoundAtOfScoreGap cert.prev scoreGapLo
      let epsAt := Sound.epsAtOfWeightBoundAt cert.prev weightBoundAt
      ∀ q, q ∈ cert.active →
        epsAt q ≤ cert.epsAt q ∧
          ∀ k, k ≠ cert.prev q →
            cert.margin ≤ scoreGapLo q k ∧
              weightBoundAt q k ≤ cert.weightBoundAt q k := by
  classical
  intro hcheck scoreGapLo weightBoundAt epsAt
  have hcheck' :
      finsetAll cert.active (fun q =>
        decide (epsAt q ≤ cert.epsAt q) &&
          finsetAll ((Finset.univ : Finset (Fin seq)).erase (cert.prev q)) (fun k =>
            decide (cert.margin ≤ scoreGapLo q k) &&
              decide (weightBoundAt q k ≤ cert.weightBoundAt q k))) = true := by
    simpa [scoreBoundsWithinScores, scoreGapLo, weightBoundAt, epsAt] using hcheck
  have hall :=
    (finsetAll_eq_true_iff (s := cert.active)).1 hcheck'
  intro q hq
  have hq' := hall q hq
  have hq'' : decide (epsAt q ≤ cert.epsAt q) = true ∧
      finsetAll ((Finset.univ : Finset (Fin seq)).erase (cert.prev q)) (fun k =>
        decide (cert.margin ≤ scoreGapLo q k) &&
          decide (weightBoundAt q k ≤ cert.weightBoundAt q k)) = true := by
    simpa [Bool.and_eq_true] using hq'
  have hq_eps : epsAt q ≤ cert.epsAt q := by
    simpa [decide_eq_true_iff] using hq''.1
  have hkeys :=
    (finsetAll_eq_true_iff
      (s := (Finset.univ : Finset (Fin seq)).erase (cert.prev q))).1 hq''.2
  refine ⟨hq_eps, ?_⟩
  intro k hk
  have hk' := hkeys k (by simp [hk])
  have hk'' : decide (cert.margin ≤ scoreGapLo q k) = true ∧
      decide (weightBoundAt q k ≤ cert.weightBoundAt q k) = true := by
    simpa [Bool.and_eq_true] using hk'
  refine ⟨?_, ?_⟩
  · simpa [decide_eq_true_iff] using hk''.1
  · simpa [decide_eq_true_iff] using hk''.2

/--
Check that certificate margins/weight tolerances are justified by LN-derived score bounds.

This enforces:
- nonnegative score scale;
- `margin` is no larger than the interval-derived score gap;
- `weightBoundAt` and `epsAt` are at least the interval-derived bounds.
-/
def scoreBoundsWithinModel {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel)
    (cert : Circuit.InductionHeadCert seq) : Bool :=
  if _ : 0 ≤ modelSlice.scoreScale then
    let embed : Fin seq → Fin modelSlice.dModel → Rat :=
      Eq.ndrec (motive := fun n => Fin seq → Fin n → Rat) lnSlice.embed hLn
    let lnGamma : Fin modelSlice.dModel → Rat :=
      Eq.ndrec (motive := fun n => Fin n → Rat) lnSlice.lnGamma hLn
    let lnBeta : Fin modelSlice.dModel → Rat :=
      Eq.ndrec (motive := fun n => Fin n → Rat) lnSlice.lnBeta hLn
    let lnBoundsArr : Array (Array Rat × Array Rat) :=
      Array.ofFn (fun q : Fin seq =>
        let bounds :=
          match lnSlice.lnScale? with
          | some scale =>
              Bounds.layerNormBoundsWithScale scale lnSlice.lnEps lnGamma lnBeta (embed q)
          | none =>
              Bounds.layerNormBounds lnSlice.lnEps lnGamma lnBeta (embed q)
        (Array.ofFn (fun i : Fin modelSlice.dModel => bounds.1 i - lnSlice.lnSlack),
          Array.ofFn (fun i : Fin modelSlice.dModel => bounds.2 i + lnSlice.lnSlack)))
    -- `!` indexing is safe: all arrays are built by `Array.ofFn` on matching `Fin` domains.
    let lnLo : Fin seq → Fin modelSlice.dModel → Rat :=
      fun q i => (lnBoundsArr[q.1]!).1[i.1]!
    let lnHi : Fin seq → Fin modelSlice.dModel → Rat :=
      fun q i => (lnBoundsArr[q.1]!).2[i.1]!
    let qLoArr : Array (Array Rat) :=
      Array.ofFn (fun q : Fin seq =>
        Array.ofFn (fun d : Fin modelSlice.headDim =>
          dotIntervalLower (fun j => modelSlice.wq j d) (lnLo q) (lnHi q) + modelSlice.bq d))
    let qHiArr : Array (Array Rat) :=
      Array.ofFn (fun q : Fin seq =>
        Array.ofFn (fun d : Fin modelSlice.headDim =>
          dotIntervalUpper (fun j => modelSlice.wq j d) (lnLo q) (lnHi q) + modelSlice.bq d))
    let kLoArr : Array (Array Rat) :=
      Array.ofFn (fun k : Fin seq =>
        Array.ofFn (fun d : Fin modelSlice.headDim =>
          dotIntervalLower (fun j => modelSlice.wk j d) (lnLo k) (lnHi k) + modelSlice.bk d))
    let kHiArr : Array (Array Rat) :=
      Array.ofFn (fun k : Fin seq =>
        Array.ofFn (fun d : Fin modelSlice.headDim =>
          dotIntervalUpper (fun j => modelSlice.wk j d) (lnLo k) (lnHi k) + modelSlice.bk d))
    let qLo : Fin seq → Fin modelSlice.headDim → Rat :=
      fun q d => (qLoArr[q.1]!)[d.1]!
    let qHi : Fin seq → Fin modelSlice.headDim → Rat :=
      fun q d => (qHiArr[q.1]!)[d.1]!
    let kLo : Fin seq → Fin modelSlice.headDim → Rat :=
      fun k d => (kLoArr[k.1]!)[d.1]!
    let kHi : Fin seq → Fin modelSlice.headDim → Rat :=
      fun k d => (kHiArr[k.1]!)[d.1]!
    let baseLoArr : Array (Array Rat) :=
      Array.ofFn (fun q : Fin seq =>
        Array.ofFn (fun k : Fin seq =>
          modelSlice.scoreScale * dotIntervalMulLower (qLo q) (qHi q) (kLo k) (kHi k)))
    let baseHiArr : Array (Array Rat) :=
      Array.ofFn (fun q : Fin seq =>
        Array.ofFn (fun k : Fin seq =>
          modelSlice.scoreScale * dotIntervalMulUpper (qLo q) (qHi q) (kLo k) (kHi k)))
    let scoreLoArr : Array (Array Rat) :=
      Array.ofFn (fun q : Fin seq =>
        Array.ofFn (fun k : Fin seq =>
          if modelSlice.maskCausal then
            if k ≤ q then (baseLoArr[q.1]!)[k.1]! else modelSlice.scoreMask
          else
            (baseLoArr[q.1]!)[k.1]!))
    let scoreHiArr : Array (Array Rat) :=
      Array.ofFn (fun q : Fin seq =>
        Array.ofFn (fun k : Fin seq =>
          if modelSlice.maskCausal then
            if k ≤ q then (baseHiArr[q.1]!)[k.1]! else modelSlice.scoreMask
          else
            (baseHiArr[q.1]!)[k.1]!))
    let scoreLo : Fin seq → Fin seq → Rat := fun q k => (scoreLoArr[q.1]!)[k.1]!
    let scoreHi : Fin seq → Fin seq → Rat := fun q k => (scoreHiArr[q.1]!)[k.1]!
    let scoreGapLo := Sound.scoreGapLoOfBounds cert.prev scoreLo scoreHi
    let weightBoundAt := Sound.weightBoundAtOfScoreGap cert.prev scoreGapLo
    let epsAt := Sound.epsAtOfWeightBoundAt cert.prev weightBoundAt
    finsetAll cert.active (fun q =>
      decide (epsAt q ≤ cert.epsAt q) &&
        finsetAll ((Finset.univ : Finset (Fin seq)).erase (cert.prev q)) (fun k =>
          decide (cert.margin ≤ scoreGapLo q k) &&
            decide (weightBoundAt q k ≤ cert.weightBoundAt q k)))
  else
    false

/-- Soundness of `scoreBoundsWithinModel` for the derived inequalities. -/
theorem scoreBoundsWithinModel_sound {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel)
    (cert : Circuit.InductionHeadCert seq) :
    scoreBoundsWithinModel lnSlice modelSlice hLn cert = true →
      let embed : Fin seq → Fin modelSlice.dModel → Rat :=
        Eq.ndrec (motive := fun n => Fin seq → Fin n → Rat) lnSlice.embed hLn
      let lnGamma : Fin modelSlice.dModel → Rat :=
        Eq.ndrec (motive := fun n => Fin n → Rat) lnSlice.lnGamma hLn
      let lnBeta : Fin modelSlice.dModel → Rat :=
        Eq.ndrec (motive := fun n => Fin n → Rat) lnSlice.lnBeta hLn
      let lnBoundsArr : Array (Array Rat × Array Rat) :=
        Array.ofFn (fun q : Fin seq =>
          let bounds :=
            match lnSlice.lnScale? with
            | some scale =>
                Bounds.layerNormBoundsWithScale scale lnSlice.lnEps lnGamma lnBeta (embed q)
            | none =>
                Bounds.layerNormBounds lnSlice.lnEps lnGamma lnBeta (embed q)
          (Array.ofFn (fun i : Fin modelSlice.dModel => bounds.1 i - lnSlice.lnSlack),
            Array.ofFn (fun i : Fin modelSlice.dModel => bounds.2 i + lnSlice.lnSlack)))
      let lnLo : Fin seq → Fin modelSlice.dModel → Rat :=
        fun q i => (lnBoundsArr[q.1]!).1[i.1]!
      let lnHi : Fin seq → Fin modelSlice.dModel → Rat :=
        fun q i => (lnBoundsArr[q.1]!).2[i.1]!
      let qLoArr : Array (Array Rat) :=
        Array.ofFn (fun q : Fin seq =>
          Array.ofFn (fun d : Fin modelSlice.headDim =>
            dotIntervalLower (fun j => modelSlice.wq j d) (lnLo q) (lnHi q) + modelSlice.bq d))
      let qHiArr : Array (Array Rat) :=
        Array.ofFn (fun q : Fin seq =>
          Array.ofFn (fun d : Fin modelSlice.headDim =>
            dotIntervalUpper (fun j => modelSlice.wq j d) (lnLo q) (lnHi q) + modelSlice.bq d))
      let kLoArr : Array (Array Rat) :=
        Array.ofFn (fun k : Fin seq =>
          Array.ofFn (fun d : Fin modelSlice.headDim =>
            dotIntervalLower (fun j => modelSlice.wk j d) (lnLo k) (lnHi k) + modelSlice.bk d))
      let kHiArr : Array (Array Rat) :=
        Array.ofFn (fun k : Fin seq =>
          Array.ofFn (fun d : Fin modelSlice.headDim =>
            dotIntervalUpper (fun j => modelSlice.wk j d) (lnLo k) (lnHi k) + modelSlice.bk d))
      let qLo : Fin seq → Fin modelSlice.headDim → Rat :=
        fun q d => (qLoArr[q.1]!)[d.1]!
      let qHi : Fin seq → Fin modelSlice.headDim → Rat :=
        fun q d => (qHiArr[q.1]!)[d.1]!
      let kLo : Fin seq → Fin modelSlice.headDim → Rat :=
        fun k d => (kLoArr[k.1]!)[d.1]!
      let kHi : Fin seq → Fin modelSlice.headDim → Rat :=
        fun k d => (kHiArr[k.1]!)[d.1]!
      let baseLoArr : Array (Array Rat) :=
        Array.ofFn (fun q : Fin seq =>
          Array.ofFn (fun k : Fin seq =>
            modelSlice.scoreScale * dotIntervalMulLower (qLo q) (qHi q) (kLo k) (kHi k)))
      let baseHiArr : Array (Array Rat) :=
        Array.ofFn (fun q : Fin seq =>
          Array.ofFn (fun k : Fin seq =>
            modelSlice.scoreScale * dotIntervalMulUpper (qLo q) (qHi q) (kLo k) (kHi k)))
      let scoreLoArr : Array (Array Rat) :=
        Array.ofFn (fun q : Fin seq =>
          Array.ofFn (fun k : Fin seq =>
            if modelSlice.maskCausal then
              if k ≤ q then (baseLoArr[q.1]!)[k.1]! else modelSlice.scoreMask
            else
              (baseLoArr[q.1]!)[k.1]!))
      let scoreHiArr : Array (Array Rat) :=
        Array.ofFn (fun q : Fin seq =>
          Array.ofFn (fun k : Fin seq =>
            if modelSlice.maskCausal then
              if k ≤ q then (baseHiArr[q.1]!)[k.1]! else modelSlice.scoreMask
            else
              (baseHiArr[q.1]!)[k.1]!))
      let scoreLo : Fin seq → Fin seq → Rat := fun q k => (scoreLoArr[q.1]!)[k.1]!
      let scoreHi : Fin seq → Fin seq → Rat := fun q k => (scoreHiArr[q.1]!)[k.1]!
      let scoreGapLo := Sound.scoreGapLoOfBounds cert.prev scoreLo scoreHi
      let weightBoundAt := Sound.weightBoundAtOfScoreGap cert.prev scoreGapLo
      let epsAt := Sound.epsAtOfWeightBoundAt cert.prev weightBoundAt
      0 ≤ modelSlice.scoreScale ∧
        ∀ q, q ∈ cert.active →
          epsAt q ≤ cert.epsAt q ∧
            ∀ k, k ≠ cert.prev q →
              cert.margin ≤ scoreGapLo q k ∧
                weightBoundAt q k ≤ cert.weightBoundAt q k := by
  classical
  intro hcheck embed lnGamma lnBeta lnBoundsArr lnLo lnHi qLoArr qHiArr kLoArr kHiArr
    qLo qHi kLo kHi baseLoArr baseHiArr scoreLoArr scoreHiArr scoreLo scoreHi scoreGapLo
    weightBoundAt epsAt
  by_cases hscale : 0 ≤ modelSlice.scoreScale
  · have hcheck' : scoreBoundsWithinModel lnSlice modelSlice hLn cert = true := hcheck
    have hcheck'' : finsetAll cert.active (fun q =>
        decide (epsAt q ≤ cert.epsAt q) &&
          finsetAll ((Finset.univ : Finset (Fin seq)).erase (cert.prev q)) (fun k =>
            decide (cert.margin ≤ scoreGapLo q k) &&
              decide (weightBoundAt q k ≤ cert.weightBoundAt q k))) = true := by
      simpa [scoreBoundsWithinModel, hscale, embed, lnGamma, lnBeta, lnBoundsArr, lnLo, lnHi,
        qLoArr, qHiArr, kLoArr, kHiArr, qLo, qHi, kLo, kHi, baseLoArr, baseHiArr, scoreLoArr,
        scoreHiArr, scoreLo, scoreHi, scoreGapLo, weightBoundAt, epsAt]
        using hcheck'
    have hall :=
      (finsetAll_eq_true_iff (s := cert.active)).1 hcheck''
    refine ⟨hscale, ?_⟩
    intro q hq
    have hq' := hall q hq
    have hq'' : decide (epsAt q ≤ cert.epsAt q) = true ∧
        finsetAll ((Finset.univ : Finset (Fin seq)).erase (cert.prev q)) (fun k =>
          decide (cert.margin ≤ scoreGapLo q k) &&
            decide (weightBoundAt q k ≤ cert.weightBoundAt q k)) = true := by
      simpa [Bool.and_eq_true] using hq'
    have hq_eps : epsAt q ≤ cert.epsAt q := by
      simpa [decide_eq_true_iff] using hq''.1
    have hkeys :=
      (finsetAll_eq_true_iff
        (s := (Finset.univ : Finset (Fin seq)).erase (cert.prev q))).1 hq''.2
    refine ⟨hq_eps, ?_⟩
    intro k hk
    have hk' := hkeys k (by simp [hk])
    have hk'' : decide (cert.margin ≤ scoreGapLo q k) = true ∧
        decide (weightBoundAt q k ≤ cert.weightBoundAt q k) = true := by
      simpa [Bool.and_eq_true] using hk'
    refine ⟨?_, ?_⟩
    · simpa [decide_eq_true_iff] using hk''.1
    · simpa [decide_eq_true_iff] using hk''.2
  · have : scoreBoundsWithinModel lnSlice modelSlice hLn cert = false := by
      simp [scoreBoundsWithinModel, hscale]
    simp [this] at hcheck

end InductionHeadCert

end Pure

end IO

end Nfp
