-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.List.Range
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Vector.Defs
import Nfp.IO.NfptPure
import Nfp.IO.Timing
import Nfp.Model.Gpt2
import Nfp.Sound.Bounds.Transformer
import Nfp.Sound.Induction
import Nfp.Sound.Induction.HeadBounds

/-!
IO derivations that build certificates from model binaries.
-/

namespace Nfp

namespace IO

open Nfp.Circuit

def deriveResidualIntervalFromModel (data : ByteArray) (start : Nat)
    (header : NfptPure.NfptHeader) (active? : Option (Finset (Fin header.seqLen))) :
    IO (Except String (ResidualIntervalCert header.modelDim)) := do
  if hseq : header.seqLen = 0 then
    return Except.error "seq must be positive"
  else
    have _ : NeZero header.seqLen := ⟨hseq⟩
    if header.modelDim = 0 then
      return Except.error "model dim must be positive"
    else if 0 < header.layerNormEps then
      let embedE ← timePure "read embeddings" (fun () =>
        NfptPure.readEmbeddings data start header)
      match embedE with
      | Except.error msg => return Except.error msg
      | Except.ok embed =>
          let layerSlicesE ← timePure "read layer slices" (fun () =>
            NfptPure.readLayerSlices data start header)
          match layerSlicesE with
          | Except.error msg => return Except.error msg
          | Except.ok layerSlices =>
              let headLayersE ← timePure "read layer heads" (fun () =>
                NfptPure.readLayerHeads data start header)
              match headLayersE with
              | Except.error msg => return Except.error msg
              | Except.ok headLayers =>
                  let finalLnE ← timePure "read final layer norm" (fun () =>
                    NfptPure.readFinalLayerNorm data start header)
                  match finalLnE with
                  | Except.error msg => return Except.error msg
                  | Except.ok finalLn =>
                      let layers :
                          Fin header.numLayers →
                            Model.Gpt2LayerSlice header.modelDim header.hiddenDim :=
                        fun l => NfptPure.SizedArray.get layerSlices l
                      let heads :
                          Fin header.numLayers → Fin header.numHeads →
                            Model.Gpt2HeadWeights header.modelDim header.headDim := fun l h =>
                        NfptPure.SizedArray.get (NfptPure.SizedArray.get headLayers l) h
                      let strict? ← IO.getEnv "NFP_TIMING_STRICT"
                      match strict? with
                      | some _ =>
                          logTiming "timing strict enabled"
                      | none =>
                          logTiming "timing strict disabled"
                      match active? with
                      | some active =>
                          if hactive : active.Nonempty then
                            logTiming "before transformer stack bounds (active)"
                            let bounds ← timePhaseThunk "transformer stack bounds (active)"
                              (fun () => do
                              let bounds := Sound.Bounds.gpt2ResidualIntervalBoundsActive
                                active hactive header.layerNormEps layers heads finalLn embed
                              match strict? with
                              | some _ =>
                                  let forced :=
                                    (List.finRange header.modelDim).foldl
                                      (fun acc i => acc + bounds.1 i + bounds.2 i) (0 : Dyadic)
                                  logTiming s!"forced transformer stack sum {forced}"
                              | none => pure ()
                              return bounds)
                            logTiming "after transformer stack bounds (active)"
                            return Except.ok { lo := bounds.1, hi := bounds.2 }
                          else
                            logTiming "active set empty; falling back to global bounds"
                            let base ← timePure "embedding interval bounds" (fun () =>
                              Sound.Bounds.embeddingIntervalBounds embed)
                            logTiming "before transformer stack bounds"
                            let stack ← timePhaseThunk "transformer stack bounds" (fun () => do
                              let stack := Sound.Bounds.transformerStackBounds
                                (eps := header.layerNormEps) layers heads base.1 base.2
                              match strict? with
                              | some _ =>
                                  let forced :=
                                    (List.finRange header.modelDim).foldl
                                      (fun acc i => acc + stack.1 i + stack.2 i) (0 : Dyadic)
                                  logTiming s!"forced transformer stack sum {forced}"
                              | none => pure ()
                              return stack)
                            logTiming "after transformer stack bounds"
                            logTiming "enter final layer norm bounds"
                            let bounds ← timePure "final layer norm bounds" (fun () =>
                              Sound.Bounds.layerNormIntervalBounds (eps := header.layerNormEps)
                                finalLn.gamma finalLn.beta stack.1 stack.2)
                            logTiming "exit final layer norm bounds"
                            return Except.ok { lo := bounds.1, hi := bounds.2 }
                      | none =>
                          let base ← timePure "embedding interval bounds" (fun () =>
                            Sound.Bounds.embeddingIntervalBounds embed)
                          logTiming "before transformer stack bounds"
                          let stack ← timePhaseThunk "transformer stack bounds" (fun () => do
                            let stack := Sound.Bounds.transformerStackBounds
                              (eps := header.layerNormEps) layers heads base.1 base.2
                            match strict? with
                            | some _ =>
                                let forced :=
                                  (List.finRange header.modelDim).foldl
                                    (fun acc i => acc + stack.1 i + stack.2 i) (0 : Dyadic)
                                logTiming s!"forced transformer stack sum {forced}"
                            | none => pure ()
                            return stack)
                          logTiming "after transformer stack bounds"
                          logTiming "enter final layer norm bounds"
                          let bounds ← timePure "final layer norm bounds" (fun () =>
                            Sound.Bounds.layerNormIntervalBounds (eps := header.layerNormEps)
                              finalLn.gamma finalLn.beta stack.1 stack.2)
                          logTiming "exit final layer norm bounds"
                          return Except.ok { lo := bounds.1, hi := bounds.2 }
    else
      return Except.error
        s!"layer norm epsilon {header.layerNormEps} must be positive"

end IO

end Nfp
