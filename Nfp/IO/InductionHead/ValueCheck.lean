-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.InductionHead
public import Nfp.IO.InductionHead.ModelDirectionSlice
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.ModelValueSlice
public import Nfp.IO.Pure.InductionHead.ValueCheck
public import Nfp.IO.Util

/-!
Value-path checks for anchoring induction-head certificate values.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

open Nfp.Bounds
open Nfp.Circuit
open Nfp.Linear

/--
Check that per-key values lie within bounds derived from model slices.

This is a thin wrapper around the trusted IO.Pure implementation.
-/
def valuesWithinModelBounds {seq : Nat}
    (lnSlice : ModelLnSlice seq) (valueSlice : ModelValueSlice)
    (dirSlice : ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (values : Circuit.ValueIntervalCert seq) : Bool :=
  Pure.InductionHeadCert.valuesWithinModelBounds
    lnSlice valueSlice dirSlice hLn hDir values

/--
Profile the model-value bound reconstruction used for value checks.

Returns the trusted check result while optionally emitting timing breakdowns.
-/
def valuesWithinModelBoundsProfile {seq : Nat}
    (lnSlice : ModelLnSlice seq) (valueSlice : ModelValueSlice)
    (dirSlice : ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (values : Circuit.ValueIntervalCert seq)
    (timeValues : Bool) : IO Bool := do
  let ok :=
    Pure.InductionHeadCert.valuesWithinModelBounds
      lnSlice valueSlice dirSlice hLn hDir values
  if !timeValues then
    return ok
  let embed : Fin seq → Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.embed
  let lnGamma : Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.lnGamma
  let lnBeta : Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.lnBeta
  let direction : Fin valueSlice.dModel → Rat := by
    simpa [hDir] using dirSlice.direction
  let dirHeadVecWithProof ← Nfp.IO.timeIO timeValues "values-dirhead" (fun _ =>
    pure <|
      (⟨Array.ofFn (fun (d : Fin valueSlice.headDim) =>
          Linear.dotFin valueSlice.dModel (fun j => valueSlice.wo j d) direction),
        by simp⟩ : { arr : Array Rat // arr.size = valueSlice.headDim }))
  let dirHeadVec : Array Rat := dirHeadVecWithProof.1
  have hDirHeadVec : dirHeadVec.size = valueSlice.headDim := dirHeadVecWithProof.2
  let dirHead : Fin valueSlice.headDim → Rat := fun d => dirHeadVec[d]!
  let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
    fun q =>
      match lnSlice.lnScale? with
      | some scale =>
          Bounds.layerNormBoundsWithScale scale lnSlice.lnEps lnGamma lnBeta (embed q)
      | none =>
          Bounds.layerNormBounds lnSlice.lnEps lnGamma lnBeta (embed q)
  let wvDenArr : Array Nat :=
    Array.ofFn (fun d : Fin valueSlice.headDim =>
      (Array.ofFn (fun j : Fin valueSlice.dModel => valueSlice.wv j d)).foldl
        (fun acc v => Nat.lcm acc v.den) 1)
  let lnBoundsAll ← Nfp.IO.timeIO timeValues "values-ln" (fun _ =>
    pure <|
      Array.ofFn (fun (k : Fin seq) =>
        let bounds := lnBounds k
        let lo := Array.ofFn (fun (i : Fin valueSlice.dModel) =>
          bounds.1 i - lnSlice.lnSlack)
        let hi := Array.ofFn (fun (i : Fin valueSlice.dModel) =>
          bounds.2 i + lnSlice.lnSlack)
        (lo, hi)))
  let lnDenArr : Array Nat :=
    Array.ofFn (fun (k : Fin seq) =>
      let bounds := lnBoundsAll[k]!
      let denLo := bounds.1.foldl (fun acc v => Nat.lcm acc v.den) 1
      let denHi := bounds.2.foldl (fun acc v => Nat.lcm acc v.den) 1
      Nat.lcm denLo denHi)
  let vBoundsAll ← Nfp.IO.timeIO timeValues "values-v" (fun _ =>
    pure <|
      Array.ofFn (fun (k : Fin seq) =>
        let bounds := lnBoundsAll[k]!
        let lnLo : Fin valueSlice.dModel → Rat := fun i => bounds.1[i]!
        let lnHi : Fin valueSlice.dModel → Rat := fun i => bounds.2[i]!
        Array.ofFn (fun (d : Fin valueSlice.headDim) =>
          let den := Nat.lcm (wvDenArr[d.1]!) (lnDenArr[k.1]!)
          let vb := dotIntervalBoundsFastArrFast valueSlice.dModel (some den)
            (Array.ofFn (fun j : Fin valueSlice.dModel => valueSlice.wv j d))
            (Array.ofFn lnLo) (Array.ofFn lnHi)
            (by simp) (by simp) (by simp)
          (vb.1 + valueSlice.bv d, vb.2 + valueSlice.bv d))))
  let dirHeadDen := dirHeadVec.foldl (fun acc v => Nat.lcm acc v.den) 1
  let valBounds ← Nfp.IO.timeIO timeValues "values-val" (fun _ =>
    pure <|
      Array.ofFn (fun (k : Fin seq) =>
        let bounds := vBoundsAll[k]!
        let vLoArr := Array.ofFn (fun d : Fin valueSlice.headDim => bounds[d]!.1)
        let vHiArr := Array.ofFn (fun d : Fin valueSlice.headDim => bounds[d]!.2)
        let denLo := vLoArr.foldl (fun acc v => Nat.lcm acc v.den) 1
        let denHi := vHiArr.foldl (fun acc v => Nat.lcm acc v.den) 1
        let den := Nat.lcm dirHeadDen (Nat.lcm denLo denHi)
        dotIntervalBoundsFastArrFast valueSlice.headDim (some den) dirHeadVec vLoArr vHiArr
          hDirHeadVec (by simp [vLoArr]) (by simp [vHiArr])))
  let checkOk ← Nfp.IO.timeIO timeValues "values-check" (fun _ =>
    pure <|
      finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
        let bounds := valBounds[k]!
        decide (bounds.1 ≤ values.vals k) && decide (values.vals k ≤ bounds.2)))
  IO.eprintln s!"info: values-check-ok {checkOk}"
  return ok

end InductionHeadCert

end IO

end Nfp
