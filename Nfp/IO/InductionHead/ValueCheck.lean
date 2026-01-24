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
  let dirHeadVec ← Nfp.IO.timeIO timeValues "values-dirhead" (fun _ =>
    pure <|
      Array.ofFn (fun (d : Fin valueSlice.headDim) =>
        Linear.dotFin valueSlice.dModel (fun j => valueSlice.wo j d) direction))
  let dirHead : Fin valueSlice.headDim → Rat := fun d => dirHeadVec[d]!
  let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
    fun q =>
      match lnSlice.lnScale? with
      | some scale =>
          Bounds.layerNormBoundsWithScale scale lnSlice.lnEps lnGamma lnBeta (embed q)
      | none =>
          Bounds.layerNormBounds lnSlice.lnEps lnGamma lnBeta (embed q)
  let lnBoundsAll ← Nfp.IO.timeIO timeValues "values-ln" (fun _ =>
    pure <|
      Array.ofFn (fun (k : Fin seq) =>
        let bounds := lnBounds k
        let lo := Array.ofFn (fun (i : Fin valueSlice.dModel) =>
          bounds.1 i - lnSlice.lnSlack)
        let hi := Array.ofFn (fun (i : Fin valueSlice.dModel) =>
          bounds.2 i + lnSlice.lnSlack)
        (lo, hi)))
  let vBoundsAll ← Nfp.IO.timeIO timeValues "values-v" (fun _ =>
    pure <|
      Array.ofFn (fun (k : Fin seq) =>
        let bounds := lnBoundsAll[k]!
        let lnLo : Fin valueSlice.dModel → Rat := fun i => bounds.1[i]!
        let lnHi : Fin valueSlice.dModel → Rat := fun i => bounds.2[i]!
        Array.ofFn (fun (d : Fin valueSlice.headDim) =>
          let vb := dotIntervalBoundsFast (fun j => valueSlice.wv j d) lnLo lnHi
          (vb.1 + valueSlice.bv d, vb.2 + valueSlice.bv d))))
  let valBounds ← Nfp.IO.timeIO timeValues "values-val" (fun _ =>
    pure <|
      Array.ofFn (fun (k : Fin seq) =>
        let bounds := vBoundsAll[k]!
        let vLo : Fin valueSlice.headDim → Rat := fun d => bounds[d]!.1
        let vHi : Fin valueSlice.headDim → Rat := fun d => bounds[d]!.2
        dotIntervalBoundsFast dirHead vLo vHi))
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
