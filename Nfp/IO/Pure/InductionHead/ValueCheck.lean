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
  let dirHeadArr : Array Rat :=
    Array.ofFn (fun d : Fin valueSlice.headDim =>
      Linear.dotFin valueSlice.dModel (fun j => valueSlice.wo j d) direction)
  let dirHead : Fin valueSlice.headDim → Rat := fun d => dirHeadArr[d.1]!
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
  let vBoundsArr : Array (Array (Rat × Rat)) :=
    Array.ofFn (fun k : Fin seq =>
      Array.ofFn (fun d : Fin valueSlice.headDim =>
        let bounds :=
          dotIntervalBoundsFast (fun j => valueSlice.wv j d) (lnLo k) (lnHi k)
        (bounds.1 + valueSlice.bv d, bounds.2 + valueSlice.bv d)))
  let vLo : Fin seq → Fin valueSlice.headDim → Rat :=
    fun k d => (vBoundsArr[k.1]!)[d.1]!.1
  let vHi : Fin seq → Fin valueSlice.headDim → Rat :=
    fun k d => (vBoundsArr[k.1]!)[d.1]!.2
  let valBoundsArr : Array (Rat × Rat) :=
    Array.ofFn (fun k : Fin seq =>
      dotIntervalBoundsFast dirHead (vLo k) (vHi k))
  let valLo : Fin seq → Rat := fun k => (valBoundsArr[k.1]!).1
  let valHi : Fin seq → Rat := fun k => (valBoundsArr[k.1]!).2
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
        fun k => dotIntervalBoundsFast dirHead (vLo k) (vHi k)
      let valLo : Fin seq → Rat := fun k => (valBounds k).1
      let valHi : Fin seq → Rat := fun k => (valBounds k).2
      ∀ k, valLo k ≤ values.vals k ∧ values.vals k ≤ valHi k := by
  classical
  intro hcheck embed lnGamma lnBeta direction dirHead lnBounds lnLo lnHi vBounds vLo vHi
    valBounds valLo valHi k
  have hall :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hcheck
  have hk := hall k (by simp)
  have hk' : decide (valLo k ≤ values.vals k) = true ∧
      decide (values.vals k ≤ valHi k) = true := by
    simpa [Bool.and_eq_true] using hk
  exact
    ⟨by simpa [decide_eq_true_iff] using hk'.1,
      by simpa [decide_eq_true_iff] using hk'.2⟩

end InductionHeadCert

end Pure

end IO

end Nfp
