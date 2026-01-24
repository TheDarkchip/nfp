-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.LayerNorm
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.IO.InductionHead.ModelDirectionSlice
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.ModelValueSlice
public import Nfp.Linear.FinFold

/-!
Value-path checks for anchoring induction-head certificate values.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/-- Lower bound for a dot product given per-coordinate intervals. -/
private def dotIntervalLower {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun i =>
    if 0 ≤ v i then v i * lo i else v i * hi i)

/-- Upper bound for a dot product given per-coordinate intervals. -/
private def dotIntervalUpper {n : Nat} (v lo hi : Fin n → Rat) : Rat :=
  Linear.sumFin n (fun i =>
    if 0 ≤ v i then v i * hi i else v i * lo i)

/--
Check that per-key values lie within bounds derived from model slices.

This uses LayerNorm bounds from the pre-LN embeddings, value projection bounds
from `wv`/`bv`, and the unembedding-based direction for the final dot-product.
-/
def valuesWithinModelBounds {seq : Nat}
    (lnSlice : ModelLnSlice seq) (valueSlice : ModelValueSlice)
    (dirSlice : ModelDirectionSlice)
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
  (List.finRange seq).all (fun k =>
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

end InductionHeadCert

end IO

end Nfp
