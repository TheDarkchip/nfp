-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Bounds.Transformer
public import Nfp.Sound.Induction.HeadOutput
public import Nfp.Sound.Induction.LogitDiff

/-!
End-to-end induction bounds that combine head certificates with transformer-stack intervals.
-/

public section

namespace Nfp

namespace Sound

/-- Compose head logit-diff bounds with GPT-2 stack output intervals. -/
theorem logitDiffLowerBound_end_to_end_gpt2
    {seq dModel dHead numHeads hidden numLayers : Nat} [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lb : Rat)
    (hlb : ∀ q, q ∈ inputs.active → (lb : Real) ≤ headLogitDiff inputs q)
    (headCert : Circuit.ResidualIntervalCert dModel)
    (hhead : HeadOutputIntervalSound inputs inputs.active headCert)
    (eps : Rat)
    (layers : Fin numLayers → Model.Gpt2LayerSlice dModel hidden)
    (heads : Fin numLayers → Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (finalLn : Model.Gpt2FinalLayerNorm dModel)
    (scores : Fin numLayers → Fin numHeads → Fin seq → Fin seq → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < Bounds.sqrtLower eps)
    (hactive : inputs.active.Nonempty) :
  let bounds :=
      Bounds.gpt2ResidualIntervalBoundsActive inputs.active hactive eps layers heads finalLn
        inputs.embed
  let output : Fin seq → Fin dModel → Real :=
      fun q i => Bounds.transformerStackFinalReal eps finalLn layers heads scores
        (fun q i => (inputs.embed q i : Real)) q i
  ∀ q, q ∈ inputs.active →
      (lb : Real) -
          (Bounds.dotIntervalAbsBound inputs.direction
            (fun i => bounds.1 i - headCert.hi i)
            (fun i => bounds.2 i - headCert.lo i) : Real) ≤
        dotProduct (fun i => (inputs.direction i : Real)) (fun i => output q i) := by
  classical
  intro bounds output q hq
  have hbounds :
      ∀ q, q ∈ inputs.active → ∀ i,
        (bounds.1 i : Real) ≤ output q i ∧ output q i ≤ (bounds.2 i : Real) := by
    simpa [bounds, output] using
      (Bounds.gpt2ResidualIntervalBoundsActive_spec
        (active := inputs.active)
        (hactive := hactive)
        (eps := eps)
        (layers := layers)
        (heads := heads)
        (finalLn := finalLn)
        (scores := scores)
        (embed := inputs.embed)
        (hne := hne)
        (heps := heps)
        (hsqrt := hsqrt))
  have hhead_out := hhead.output_mem
  have h :=
    logitDiffLowerBound_with_output_intervals
      (inputs := inputs)
      (lb := lb)
      (hlb := hlb)
      (output := output)
      (outLo := bounds.1)
      (outHi := bounds.2)
      (hout := hbounds)
      (headLo := headCert.lo)
      (headHi := headCert.hi)
      (hhead := hhead_out)
      q
      hq
  simpa [bounds] using h

end Sound

end Nfp
