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
  let dirHead : Fin valueSlice.headDim → Rat := fun d =>
    Linear.dotFin valueSlice.dModel (fun j => valueSlice.wo j d) direction
  let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
    fun q =>
      match lnSlice.lnScale? with
      | some scale =>
          Bounds.layerNormBoundsWithScale scale lnSlice.lnEps lnGamma lnBeta (embed q)
      | none =>
          Bounds.layerNormBounds lnSlice.lnEps lnGamma lnBeta (embed q)
  finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
    let bounds := lnBounds k
    let lnLo : Fin valueSlice.dModel → Rat := fun i => bounds.1 i - lnSlice.lnSlack
    let lnHi : Fin valueSlice.dModel → Rat := fun i => bounds.2 i + lnSlice.lnSlack
    let vLo : Fin valueSlice.headDim → Rat := fun d =>
      dotIntervalLower (fun j => valueSlice.wv j d) lnLo lnHi + valueSlice.bv d
    let vHi : Fin valueSlice.headDim → Rat := fun d =>
      dotIntervalUpper (fun j => valueSlice.wv j d) lnLo lnHi + valueSlice.bv d
    let valLo := dotIntervalLower dirHead vLo vHi
    let valHi := dotIntervalUpper dirHead vLo vHi
    decide (valLo ≤ values.vals k) && decide (values.vals k ≤ valHi))

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
      let vLo : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
        dotIntervalLower (fun j => valueSlice.wv j d) (lnLo k) (lnHi k) + valueSlice.bv d
      let vHi : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
        dotIntervalUpper (fun j => valueSlice.wv j d) (lnLo k) (lnHi k) + valueSlice.bv d
      let valLo : Fin seq → Rat := fun k => dotIntervalLower dirHead (vLo k) (vHi k)
      let valHi : Fin seq → Rat := fun k => dotIntervalUpper dirHead (vLo k) (vHi k)
      ∀ k, valLo k ≤ values.vals k ∧ values.vals k ≤ valHi k := by
  classical
  intro hcheck embed lnGamma lnBeta direction dirHead lnBounds lnLo lnHi vLo vHi valLo valHi k
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
