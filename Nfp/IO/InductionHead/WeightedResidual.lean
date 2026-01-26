-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Max
public import Nfp.IO.Pure.InductionHead.ModelInputs
public import Nfp.IO.InductionHead.ModelDirectionSlice
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.ScoreUtils
public import Nfp.IO.InductionHead.ModelValueSlice
public import Nfp.IO.Util

/-!
Reporting helpers for weighted residual direction bounds.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/--
Report direction·(embed+headOutput) bounds derived from model slices and weights.
Optionally reuse precomputed value bounds to avoid recomputation.
-/
def reportWeightedResidualDir {seq : Nat}
    (modelLnSlice? : Option (ModelLnSlice seq))
    (modelValueSlice? : Option ModelValueSlice)
    (modelDirectionSlice? : Option ModelDirectionSlice)
    (valBoundsArr? : Option (Array (Rat × Rat)))
    (weightsPresent : Bool)
    (weights : Fin seq → Fin seq → Rat)
    (active : Finset (Fin seq))
    (timeStages : Bool) : IO Unit := do
  let _ ← timeIO timeStages "weighted-residual" (fun _ => do
    match modelLnSlice?, modelValueSlice?, modelDirectionSlice? with
    | some modelLnSlice, some modelValueSlice, some modelDirectionSlice =>
        if !weightsPresent then
          IO.eprintln "info: weighted-residual-dir skipped (missing weight entries)"
        else if hLn : modelLnSlice.dModel = modelValueSlice.dModel then
          if hDir : modelDirectionSlice.dModel = modelValueSlice.dModel then
            let bounds :=
              match valBoundsArr? with
              | some valBoundsArr =>
                  let valBounds : Fin seq → Rat × Rat := fun k => valBoundsArr[k.1]!
                  Nfp.IO.Pure.InductionHeadCert.residualDirectionBoundsWeightedFastFromValBounds
                    modelLnSlice modelValueSlice modelDirectionSlice weights valBounds hLn hDir
              | none =>
                  Nfp.IO.Pure.InductionHeadCert.residualDirectionBoundsWeightedFastFromSlices
                    modelLnSlice modelValueSlice modelDirectionSlice weights hLn hDir
            if hActive : active.Nonempty then
              let lbSet := active.image (fun q => bounds.1 q)
              let ubSet := active.image (fun q => bounds.2 q)
              let lb := lbSet.min' (hActive.image (fun q => bounds.1 q))
              let ub := ubSet.max' (hActive.image (fun q => bounds.2 q))
              IO.eprintln
                s!"info: weighted-residual-dir lb={ratToString lb} ub={ratToString ub}"
            else
              IO.eprintln "info: weighted-residual-dir skipped (empty active set)"
          else
            IO.eprintln "info: weighted-residual-dir skipped (direction dModel mismatch)"
        else
          IO.eprintln "info: weighted-residual-dir skipped (ln dModel mismatch)"
    | _, _, _ =>
        IO.eprintln "info: weighted-residual-dir skipped (missing model slices)"
    return ())
  return ()

end InductionHeadCert

end IO

end Nfp

end
