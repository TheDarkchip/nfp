-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.Interval
public import Nfp.Bounds.LayerNorm
public import Nfp.Circuit.Cert.Basic
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.IO.InductionHead.ModelDirectionSlice
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.ModelValueSlice
public import Nfp.Linear.FinFold

/-!
Pure value-path checks for anchoring induction-head certificate values.
-/

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

open Nfp.Bounds
open Nfp.Circuit

/--
Check that per-key values lie within bounds derived from model slices.

This uses LayerNorm bounds from the pre-LN embeddings, value projection bounds
from `wv`/`bv`, and the unembedding-based direction for the final dot-product.
-/
def valuesWithinModelBounds {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (values : Circuit.ValueIntervalCert seq) : Bool :=
  let embed : Fin seq → Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.embed
  let lnGamma : Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.lnGamma
  let lnBeta : Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.lnBeta
  let direction : Fin valueSlice.dModel → Rat := by
    simpa [hDir] using dirSlice.direction
  let bias : Rat :=
    Linear.dotFin valueSlice.dModel (fun j => valueSlice.attnBias j) direction
  let dirHeadArr : Array Rat :=
    Array.ofFn (fun d : Fin valueSlice.headDim =>
      Linear.dotFin valueSlice.dModel (fun j => valueSlice.wo j d) direction)
  let dirHead : Fin valueSlice.headDim → Rat := fun d => dirHeadArr[d.1]!
  let wvColsArr : Array (Array Rat) :=
    Array.ofFn (fun d : Fin valueSlice.headDim =>
      Array.ofFn (fun j : Fin valueSlice.dModel => valueSlice.wv j d))
  let lnBoundsArr : Array (Array Rat × Array Rat) :=
    Array.ofFn (fun q : Fin seq =>
      let bounds :=
        match lnSlice.lnScale? with
        | some scale =>
            Bounds.layerNormBoundsWithScale scale lnSlice.lnEps lnGamma lnBeta (embed q)
        | none =>
            Bounds.layerNormBounds lnSlice.lnEps lnGamma lnBeta (embed q)
      (Array.ofFn (fun i : Fin valueSlice.dModel => bounds.1 i - lnSlice.lnSlack),
        Array.ofFn (fun i : Fin valueSlice.dModel => bounds.2 i + lnSlice.lnSlack)))
  -- `!` indexing is safe: all arrays are built by `Array.ofFn` on matching `Fin` domains.
  let lnLo : Fin seq → Fin valueSlice.dModel → Rat :=
    fun k i => (lnBoundsArr[k.1]!).1[i.1]!
  let lnHi : Fin seq → Fin valueSlice.dModel → Rat :=
    fun k i => (lnBoundsArr[k.1]!).2[i.1]!
  let wvDenArr : Array Nat :=
    Array.ofFn (fun d : Fin valueSlice.headDim =>
      (wvColsArr[d.1]!).foldl (fun acc v => Nat.lcm acc v.den) 1)
  let lnDenArr : Array Nat :=
    Array.ofFn (fun k : Fin seq =>
      let bounds := lnBoundsArr[k.1]!
      let denLo := bounds.1.foldl (fun acc v => Nat.lcm acc v.den) 1
      let denHi := bounds.2.foldl (fun acc v => Nat.lcm acc v.den) 1
      Nat.lcm denLo denHi)
  let vBoundsArr : Array (Array (Rat × Rat)) :=
    Array.ofFn (fun k : Fin seq =>
      Array.ofFn (fun d : Fin valueSlice.headDim =>
        let wvCol :=
          wvColsArr[d.1]'(by
            simp [wvColsArr])
        let lnBounds :=
          lnBoundsArr[k.1]'(by
            simp [lnBoundsArr])
        let lnLoArr := lnBounds.1
        let lnHiArr := lnBounds.2
        let den := Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!)
        let bounds :=
          dotIntervalBoundsFastArrScaled valueSlice.dModel den wvCol lnLoArr lnHiArr
            (by simp [wvCol, wvColsArr])
            (by simp [lnLoArr, lnBounds, lnBoundsArr])
            (by simp [lnHiArr, lnBounds, lnBoundsArr])
        (bounds.1 + valueSlice.bv d, bounds.2 + valueSlice.bv d)))
  let vLo : Fin seq → Fin valueSlice.headDim → Rat :=
    fun k d =>
      let vBounds := vBoundsArr[k.1]'(by
        simp [vBoundsArr])
      (vBounds[d.1]'(by
        simp [vBounds, vBoundsArr])).1
  let vHi : Fin seq → Fin valueSlice.headDim → Rat :=
    fun k d =>
      let vBounds := vBoundsArr[k.1]'(by
        simp [vBoundsArr])
      (vBounds[d.1]'(by
        simp [vBounds, vBoundsArr])).2
  let valBoundsArr : Array (Rat × Rat) :=
    let dirHeadDen := dirHeadArr.foldl (fun acc v => Nat.lcm acc v.den) 1
    Array.ofFn (fun k : Fin seq =>
      let vBounds := vBoundsArr[k.1]'(by
        simp [vBoundsArr])
      let vLoArr := Array.ofFn (fun d : Fin valueSlice.headDim =>
        (vBounds[d.1]'(by simp [vBounds, vBoundsArr])).1)
      let vHiArr := Array.ofFn (fun d : Fin valueSlice.headDim =>
        (vBounds[d.1]'(by simp [vBounds, vBoundsArr])).2)
      let denLo := vLoArr.foldl (fun acc v => Nat.lcm acc v.den) 1
      let denHi := vHiArr.foldl (fun acc v => Nat.lcm acc v.den) 1
      let den := Nat.lcm dirHeadDen (Nat.lcm denLo denHi)
      let bounds :=
        dotIntervalBoundsFastArrScaled valueSlice.headDim den dirHeadArr vLoArr vHiArr
          (by simp [dirHeadArr]) (by simp [vLoArr]) (by simp [vHiArr])
      (bounds.1 + bias, bounds.2 + bias))
  let valLo : Fin seq → Rat := fun k =>
    (valBoundsArr[k.1]'(by
      simp [valBoundsArr])).1
  let valHi : Fin seq → Rat := fun k =>
    (valBoundsArr[k.1]'(by
      simp [valBoundsArr])).2
  finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
    decide (valLo k ≤ values.vals k) && decide (values.vals k ≤ valHi k))

/-- Soundness of `valuesWithinModelBounds` for per-key value inequalities. -/
theorem valuesWithinModelBounds_sound {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (values : Circuit.ValueIntervalCert seq) :
    valuesWithinModelBounds lnSlice valueSlice dirSlice hLn hDir values = true →
      let embed : Fin seq → Fin valueSlice.dModel → Rat := by
        simpa [hLn] using lnSlice.embed
      let lnGamma : Fin valueSlice.dModel → Rat := by
        simpa [hLn] using lnSlice.lnGamma
      let lnBeta : Fin valueSlice.dModel → Rat := by
        simpa [hLn] using lnSlice.lnBeta
      let direction : Fin valueSlice.dModel → Rat := by
        simpa [hDir] using dirSlice.direction
      let bias : Rat :=
        Linear.dotFin valueSlice.dModel (fun j => valueSlice.attnBias j) direction
      let dirHead : Fin valueSlice.headDim → Rat := fun d =>
        Linear.dotFin valueSlice.dModel (fun j => valueSlice.wo j d) direction
      let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
        fun q =>
          match lnSlice.lnScale? with
          | some scale =>
              Bounds.layerNormBoundsWithScale scale lnSlice.lnEps lnGamma lnBeta (embed q)
          | none =>
              Bounds.layerNormBounds lnSlice.lnEps lnGamma lnBeta (embed q)
      let lnLo : Fin seq → Fin valueSlice.dModel → Rat :=
        fun k i => (lnBounds k).1 i - lnSlice.lnSlack
      let lnHi : Fin seq → Fin valueSlice.dModel → Rat :=
        fun k i => (lnBounds k).2 i + lnSlice.lnSlack
      let vBounds : Fin seq → Fin valueSlice.headDim → Rat × Rat := fun k d =>
        let bounds := dotIntervalBoundsFast (fun j => valueSlice.wv j d) (lnLo k) (lnHi k)
        (bounds.1 + valueSlice.bv d, bounds.2 + valueSlice.bv d)
      let vLo : Fin seq → Fin valueSlice.headDim → Rat := fun k d => (vBounds k d).1
      let vHi : Fin seq → Fin valueSlice.headDim → Rat := fun k d => (vBounds k d).2
      let valBounds : Fin seq → Rat × Rat :=
        fun k =>
          let bounds := dotIntervalBoundsFast dirHead (vLo k) (vHi k)
          (bounds.1 + bias, bounds.2 + bias)
      let valLo : Fin seq → Rat := fun k => (valBounds k).1
      let valHi : Fin seq → Rat := fun k => (valBounds k).2
      ∀ k, valLo k ≤ values.vals k ∧ values.vals k ≤ valHi k := by
  classical
  intro hcheck embed lnGamma lnBeta direction bias dirHead lnBounds lnLo lnHi vBounds vLo vHi
    valBounds valLo valHi k
  -- Array-based definitions mirroring `valuesWithinModelBounds`.
  let dirHeadArr : Array Rat := Array.ofFn dirHead
  let wvColsArr : Array (Array Rat) :=
    Array.ofFn (fun d : Fin valueSlice.headDim =>
      Array.ofFn (fun j : Fin valueSlice.dModel => valueSlice.wv j d))
  let lnBoundsArr : Array (Array Rat × Array Rat) :=
    Array.ofFn (fun q : Fin seq =>
      let bounds := lnBounds q
      (Array.ofFn (fun i : Fin valueSlice.dModel => bounds.1 i - lnSlice.lnSlack),
        Array.ofFn (fun i : Fin valueSlice.dModel => bounds.2 i + lnSlice.lnSlack)))
  let wvDenArr : Array Nat :=
    Array.ofFn (fun d : Fin valueSlice.headDim =>
      (wvColsArr[d.1]!).foldl (fun acc v => Nat.lcm acc v.den) 1)
  let lnDenArr : Array Nat :=
    Array.ofFn (fun k : Fin seq =>
      let bounds := lnBoundsArr[k.1]!
      let denLo := bounds.1.foldl (fun acc v => Nat.lcm acc v.den) 1
      let denHi := bounds.2.foldl (fun acc v => Nat.lcm acc v.den) 1
      Nat.lcm denLo denHi)
  let vBoundsArr : Array (Array (Rat × Rat)) :=
    Array.ofFn (fun k : Fin seq =>
      Array.ofFn (fun d : Fin valueSlice.headDim =>
        let wvCol :=
          wvColsArr[d.1]'(by
            simp [wvColsArr])
        let lnBounds :=
          lnBoundsArr[k.1]'(by
            simp [lnBoundsArr])
        let lnLoArr := lnBounds.1
        let lnHiArr := lnBounds.2
        let den := Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!)
        let bounds :=
          dotIntervalBoundsFastArrScaled valueSlice.dModel den wvCol lnLoArr lnHiArr
            (by simp [wvCol, wvColsArr])
            (by simp [lnLoArr, lnBounds, lnBoundsArr])
            (by simp [lnHiArr, lnBounds, lnBoundsArr])
        (bounds.1 + valueSlice.bv d, bounds.2 + valueSlice.bv d)))
  let vLoArr : Fin seq → Fin valueSlice.headDim → Rat :=
    fun k d =>
      let vBounds := vBoundsArr[k.1]'(by
        simp [vBoundsArr])
      (vBounds[d.1]'(by
        simp [vBounds, vBoundsArr])).1
  let vHiArr : Fin seq → Fin valueSlice.headDim → Rat :=
    fun k d =>
      let vBounds := vBoundsArr[k.1]'(by
        simp [vBoundsArr])
      (vBounds[d.1]'(by
        simp [vBounds, vBoundsArr])).2
  let dirHeadDen : Nat := dirHeadArr.foldl (fun acc v => Nat.lcm acc v.den) 1
  let valBoundsArr : Array (Rat × Rat) :=
    Array.ofFn (fun k : Fin seq =>
      let vBounds := vBoundsArr[k.1]'(by
        simp [vBoundsArr])
      let vLoArr' := Array.ofFn (fun d : Fin valueSlice.headDim =>
        (vBounds[d.1]'(by simp [vBounds, vBoundsArr])).1)
      let vHiArr' := Array.ofFn (fun d : Fin valueSlice.headDim =>
        (vBounds[d.1]'(by simp [vBounds, vBoundsArr])).2)
      let denLo := vLoArr'.foldl (fun acc v => Nat.lcm acc v.den) 1
      let denHi := vHiArr'.foldl (fun acc v => Nat.lcm acc v.den) 1
      let den := Nat.lcm dirHeadDen (Nat.lcm denLo denHi)
      let bounds :=
        dotIntervalBoundsFastArrScaled valueSlice.headDim den dirHeadArr vLoArr' vHiArr'
          (by simp [dirHeadArr]) (by simp [vLoArr']) (by simp [vHiArr'])
      (bounds.1 + bias, bounds.2 + bias))
  let valLoArr : Fin seq → Rat := fun k =>
    (valBoundsArr[k.1]'(by
      simp [valBoundsArr])).1
  let valHiArr : Fin seq → Rat := fun k =>
    (valBoundsArr[k.1]'(by
      simp [valBoundsArr])).2
  have hall :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hcheck
  have hk := hall k (by simp)
  have hkArr : decide (valLoArr k ≤ values.vals k) = true ∧
      decide (values.vals k ≤ valHiArr k) = true := by
    simpa [Bool.and_eq_true, valLoArr, valHiArr, valBoundsArr, vBoundsArr, dirHeadArr,
      dirHeadDen, wvColsArr, lnBoundsArr, vLoArr, vHiArr, wvDenArr, lnDenArr, bias] using hk
  have hdiv_wv :
      ∀ (d : Fin valueSlice.headDim) (j : Fin valueSlice.dModel),
        (valueSlice.wv j d).den ∣ wvDenArr[d.1]! := by
    intro d j
    have hsize : (wvColsArr[d.1]!).size = valueSlice.dModel := by
      simp [wvColsArr]
    have hdiv :=
      dvd_foldl_lcm_den_get (arr := wvColsArr[d.1]!) (i := Fin.cast hsize.symm j)
    simpa [wvDenArr, wvColsArr, hsize] using hdiv
  have hdiv_ln_lo :
      ∀ (k : Fin seq) (i : Fin valueSlice.dModel),
        ((lnBoundsArr[k.1]!).1[i.1]!).den ∣ lnDenArr[k.1]! := by
    intro k i
    let bounds := lnBoundsArr[k.1]!
    have hsize : bounds.1.size = valueSlice.dModel := by
      simp [lnBoundsArr, bounds]
    have hdiv := dvd_foldl_lcm_den_get (arr := bounds.1) (i := Fin.cast hsize.symm i)
    have hdiv' :
        ((bounds.1[Fin.cast hsize.symm i]).den) ∣
          Nat.lcm (bounds.1.foldl (fun acc v => Nat.lcm acc v.den) 1)
            (bounds.2.foldl (fun acc v => Nat.lcm acc v.den) 1) := by
      exact hdiv.trans (Nat.dvd_lcm_left _ _)
    simpa [lnDenArr, bounds, hsize] using hdiv'
  have hdiv_ln_hi :
      ∀ (k : Fin seq) (i : Fin valueSlice.dModel),
        ((lnBoundsArr[k.1]!).2[i.1]!).den ∣ lnDenArr[k.1]! := by
    intro k i
    let bounds := lnBoundsArr[k.1]!
    have hsize : bounds.2.size = valueSlice.dModel := by
      simp [lnBoundsArr, bounds]
    have hdiv := dvd_foldl_lcm_den_get (arr := bounds.2) (i := Fin.cast hsize.symm i)
    have hdiv' :
        ((bounds.2[Fin.cast hsize.symm i]).den) ∣
          Nat.lcm (bounds.1.foldl (fun acc v => Nat.lcm acc v.den) 1)
            (bounds.2.foldl (fun acc v => Nat.lcm acc v.den) 1) := by
      exact hdiv.trans (Nat.dvd_lcm_right _ _)
    simpa [lnDenArr, bounds, hsize] using hdiv'
  have hpos_wv : ∀ (d : Fin valueSlice.headDim), 0 < wvDenArr[d.1]! := by
    intro d
    have hpos := foldl_lcm_den_pos_array (arr := wvColsArr[d.1]!)
    simpa [wvDenArr, wvColsArr] using hpos
  have hpos_ln : ∀ (k : Fin seq), 0 < lnDenArr[k.1]! := by
    intro k
    let bounds := lnBoundsArr[k.1]!
    have hpos_lo := foldl_lcm_den_pos_array (arr := bounds.1)
    have hpos_hi := foldl_lcm_den_pos_array (arr := bounds.2)
    have hpos : 0 < Nat.lcm
        (bounds.1.foldl (fun acc v => Nat.lcm acc v.den) 1)
        (bounds.2.foldl (fun acc v => Nat.lcm acc v.den) 1) := by
      exact Nat.lcm_pos hpos_lo hpos_hi
    simpa [lnDenArr, bounds] using hpos
  have hdiv_dir : ∀ (d : Fin valueSlice.headDim), (dirHeadArr[d.1]!).den ∣ dirHeadDen := by
    intro d
    have hsize : dirHeadArr.size = valueSlice.headDim := by
      simp [dirHeadArr]
    have hdiv := dvd_foldl_lcm_den_get (arr := dirHeadArr) (i := Fin.cast hsize.symm d)
    simpa [dirHeadDen, dirHeadArr, hsize] using hdiv
  have hpos_dir : 0 < dirHeadDen := by
    simpa [dirHeadDen] using foldl_lcm_den_pos_array (arr := dirHeadArr)
  have hvalBounds : ∀ k, valBoundsArr[k.1]'(by simp [valBoundsArr]) = valBounds k := by
    intro k
    -- inner scaled bounds agree with exact bounds for this `k`
    have hinner :
        ∀ d : Fin valueSlice.headDim,
          dotIntervalBoundsFastArrScaled valueSlice.dModel
              (Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!))
              (Array.ofFn (fun j : Fin valueSlice.dModel => valueSlice.wv j d))
              (Array.ofFn (fun i : Fin valueSlice.dModel => (lnBounds k).1 i - lnSlice.lnSlack))
              (Array.ofFn (fun i : Fin valueSlice.dModel => (lnBounds k).2 i + lnSlice.lnSlack))
              (by simp) (by simp) (by simp) =
            dotIntervalBoundsFast (fun j => valueSlice.wv j d) (lnLo k) (lnHi k) := by
      intro d
      let wvArr : Array Rat :=
        Array.ofFn (fun j : Fin valueSlice.dModel => valueSlice.wv j d)
      let lnLoArr : Array Rat :=
        Array.ofFn (fun i : Fin valueSlice.dModel => (lnBounds k).1 i - lnSlice.lnSlack)
      let lnHiArr : Array Rat :=
        Array.ofFn (fun i : Fin valueSlice.dModel => (lnBounds k).2 i + lnSlice.lnSlack)
      have hden :
          Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!) ≠ 0 := by
        exact Nat.ne_of_gt (Nat.lcm_pos (hpos_wv d) (hpos_ln k))
      have hdiv_v :
          ∀ j : Fin valueSlice.dModel,
            (wvArr[j.1]'(by simp [wvArr])).den ∣
              Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!) := by
        intro j
        have hdiv' :
            (valueSlice.wv j d).den ∣ Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!) :=
          (hdiv_wv d j).trans (Nat.dvd_lcm_left _ _)
        simpa [wvArr] using hdiv'
      have hdiv_lo :
          ∀ i : Fin valueSlice.dModel,
            (lnLoArr[i.1]'(by simp [lnLoArr])).den ∣
              Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!) := by
        intro i
        have hdiv' :
            ((lnBounds k).1 i - lnSlice.lnSlack).den ∣
              Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!) := by
          have hdiv'' :
              ((lnBounds k).1 i - lnSlice.lnSlack).den ∣ lnDenArr[k.1]! := by
            simpa [lnBoundsArr] using hdiv_ln_lo k i
          exact hdiv''.trans (Nat.dvd_lcm_right _ _)
        simpa [lnLoArr] using hdiv'
      have hdiv_hi :
          ∀ i : Fin valueSlice.dModel,
            (lnHiArr[i.1]'(by simp [lnHiArr])).den ∣
              Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!) := by
        intro i
        have hdiv' :
            ((lnBounds k).2 i + lnSlice.lnSlack).den ∣
              Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!) := by
          have hdiv'' :
              ((lnBounds k).2 i + lnSlice.lnSlack).den ∣ lnDenArr[k.1]! := by
            simpa [lnBoundsArr] using hdiv_ln_hi k i
          exact hdiv''.trans (Nat.dvd_lcm_right _ _)
        simpa [lnHiArr] using hdiv'
      have hscaled :
          dotIntervalBoundsFastArrScaled valueSlice.dModel
              (Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!))
              wvArr lnLoArr lnHiArr (by simp [wvArr]) (by simp [lnLoArr]) (by simp [lnHiArr]) =
            dotIntervalBoundsFastArr valueSlice.dModel
              wvArr lnLoArr lnHiArr (by simp [wvArr]) (by simp [lnLoArr]) (by simp [lnHiArr]) :=
        dotIntervalBoundsFastArrScaled_eq
          (n := valueSlice.dModel)
          (den := Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!))
          (v := wvArr) (lo := lnLoArr) (hi := lnHiArr)
          (by simp [wvArr]) (by simp [lnLoArr]) (by simp [lnHiArr])
          hden hdiv_v hdiv_lo hdiv_hi
      have harr :
          dotIntervalBoundsFastArr valueSlice.dModel
              wvArr lnLoArr lnHiArr (by simp [wvArr]) (by simp [lnLoArr]) (by simp [lnHiArr]) =
            dotIntervalBoundsFast (fun j => valueSlice.wv j d) (lnLo k) (lnHi k) := by
        simpa [lnLo, lnHi, wvArr, lnLoArr, lnHiArr] using
          (dotIntervalBoundsFastArr_ofFn (n := valueSlice.dModel)
            (v := fun j => valueSlice.wv j d)
            (lo := fun i => (lnBounds k).1 i - lnSlice.lnSlack)
            (hi := fun i => (lnBounds k).2 i + lnSlice.lnSlack))
      exact hscaled.trans harr
    -- rewrite inner bounds and prove the outer scaled equality
    let vLoFun : Fin valueSlice.headDim → Rat := fun d =>
      (dotIntervalBoundsFast (fun j => valueSlice.wv j d) (lnLo k) (lnHi k)).1 + valueSlice.bv d
    let vHiFun : Fin valueSlice.headDim → Rat := fun d =>
      (dotIntervalBoundsFast (fun j => valueSlice.wv j d) (lnLo k) (lnHi k)).2 + valueSlice.bv d
    let vLoArr' : Array Rat := Array.ofFn vLoFun
    let vHiArr' : Array Rat := Array.ofFn vHiFun
    let denLo := vLoArr'.foldl (fun acc v => Nat.lcm acc v.den) 1
    let denHi := vHiArr'.foldl (fun acc v => Nat.lcm acc v.den) 1
    let den := Nat.lcm dirHeadDen (Nat.lcm denLo denHi)
    have hden : den ≠ 0 := by
      have hpos_lo : 0 < denLo := by
        simpa [denLo] using foldl_lcm_den_pos_array (arr := vLoArr')
      have hpos_hi : 0 < denHi := by
        simpa [denHi] using foldl_lcm_den_pos_array (arr := vHiArr')
      exact Nat.ne_of_gt (Nat.lcm_pos hpos_dir (Nat.lcm_pos hpos_lo hpos_hi))
    have hdiv_v :
        ∀ d : Fin valueSlice.headDim,
          (dirHeadArr[d.1]'(by simp [dirHeadArr])).den ∣ den := by
      intro d
      have hdiv' : (dirHeadArr[d.1]!).den ∣ den :=
        (hdiv_dir d).trans (Nat.dvd_lcm_left _ _)
      simpa [dirHeadArr] using hdiv'
    have hdiv_lo :
        ∀ d : Fin valueSlice.headDim,
          (vLoArr'[d.1]'(by simp [vLoArr'])).den ∣ den := by
      intro d
      have hsize : vLoArr'.size = valueSlice.headDim := by
        simp [vLoArr']
      have hdiv := dvd_foldl_lcm_den_get (arr := vLoArr') (i := Fin.cast hsize.symm d)
      have hdiv' : (vLoArr'[Fin.cast hsize.symm d]).den ∣ denLo := by
        simpa [denLo] using hdiv
      have hdiv'' : denLo ∣ Nat.lcm denLo denHi := Nat.dvd_lcm_left _ _
      have hdiv''' : Nat.lcm denLo denHi ∣ den := Nat.dvd_lcm_right _ _
      have hfinal : (vLoArr'[Fin.cast hsize.symm d]).den ∣ den :=
        hdiv'.trans (hdiv''.trans hdiv''')
      simpa [vLoArr', hsize] using hfinal
    have hdiv_hi :
        ∀ d : Fin valueSlice.headDim,
          (vHiArr'[d.1]'(by simp [vHiArr'])).den ∣ den := by
      intro d
      have hsize : vHiArr'.size = valueSlice.headDim := by
        simp [vHiArr']
      have hdiv := dvd_foldl_lcm_den_get (arr := vHiArr') (i := Fin.cast hsize.symm d)
      have hdiv' : (vHiArr'[Fin.cast hsize.symm d]).den ∣ denHi := by
        simpa [denHi] using hdiv
      have hdiv'' : denHi ∣ Nat.lcm denLo denHi := Nat.dvd_lcm_right _ _
      have hdiv''' : Nat.lcm denLo denHi ∣ den := Nat.dvd_lcm_right _ _
      have hfinal : (vHiArr'[Fin.cast hsize.symm d]).den ∣ den :=
        hdiv'.trans (hdiv''.trans hdiv''')
      simpa [vHiArr', hsize] using hfinal
    have hscaled :
        dotIntervalBoundsFastArrScaled valueSlice.headDim den
            dirHeadArr vLoArr' vHiArr' (by simp [dirHeadArr]) (by simp [vLoArr'])
              (by simp [vHiArr']) =
          dotIntervalBoundsFastArr valueSlice.headDim dirHeadArr vLoArr' vHiArr'
            (by simp [dirHeadArr]) (by simp [vLoArr']) (by simp [vHiArr']) :=
      dotIntervalBoundsFastArrScaled_eq
        (n := valueSlice.headDim) (den := den)
        (v := dirHeadArr) (lo := vLoArr') (hi := vHiArr')
        (by simp [dirHeadArr]) (by simp [vLoArr']) (by simp [vHiArr'])
        hden hdiv_v hdiv_lo hdiv_hi
    have harr :
        dotIntervalBoundsFastArr valueSlice.headDim dirHeadArr vLoArr' vHiArr'
            (by simp [dirHeadArr]) (by simp [vLoArr']) (by simp [vHiArr']) =
          dotIntervalBoundsFast dirHead vLoFun vHiFun := by
      simpa [dirHeadArr, vLoArr', vHiArr'] using
        (dotIntervalBoundsFastArr_ofFn (n := valueSlice.headDim)
          (v := dirHead) (lo := vLoFun) (hi := vHiFun))
    simpa [valBoundsArr, vBoundsArr, dirHeadArr, wvColsArr, lnBoundsArr, vBounds,
      vLo, vHi, valBounds, hinner, den, denLo, denHi, vLoArr', vHiArr', vLoFun, vHiFun, bias] using
      (Prod.ext_iff.mp (hscaled.trans harr))
  have hvalLo : valLoArr = valLo := by
    funext k
    simp [valLoArr, valLo, hvalBounds]
  have hvalHi : valHiArr = valHi := by
    funext k
    simp [valHiArr, valHi, hvalBounds]
  have hk' : decide (valLo k ≤ values.vals k) = true ∧
      decide (values.vals k ≤ valHi k) = true := by
    simpa [hvalLo, hvalHi] using hkArr
  exact
    ⟨by simpa [decide_eq_true_iff] using hk'.1,
      by simpa [decide_eq_true_iff] using hk'.2⟩

end InductionHeadCert

end Pure

end IO

end Nfp
