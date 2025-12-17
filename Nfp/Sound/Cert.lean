-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Bounds

namespace Nfp.Sound

/-!
# Sound certification report structures

The sound path computes conservative bounds using exact `Rat` arithmetic.
These values are meant to be *trusted* (up to Lean's kernel/runtime), but may be
much looser than the Float-based heuristic analysis.
-/

/-- Default LayerNorm epsilon used by exported GPT-style models. -/
def defaultEps : Rat := (1 : Rat) / 100000

/-- Conservative upper bound on GeLU derivative for certification.

For the exact GeLU `x ↦ x·Φ(x)` the derivative is bounded well below 2.
We use `2` as a simple, obviously-safe constant to avoid importing heavy analysis.
-/
def defaultActDerivBound : Rat := 2

/-- Per-layer conservative amplification constant `Cᵢ` and its components. -/
structure LayerAmplificationCert where
  layerIdx : Nat
  ln1MaxAbsGamma : Rat
  ln2MaxAbsGamma : Rat
  ln1Bound : Rat
  ln2Bound : Rat
  attnWeightContribution : Rat
  mlpWeightContribution : Rat
  C : Rat
  deriving Repr

/-- Model-level certification report. -/
structure ModelCert where
  modelPath : String
  eps : Rat
  actDerivBound : Rat
  softmaxJacobianNormInfWorst : Rat
  layers : Array LayerAmplificationCert
  totalAmplificationFactor : Rat
  deriving Repr

namespace ModelCert

/-- Pretty printer. -/
def toString (c : ModelCert) : String :=
  let header :=
    s!"SOUND mode: conservative bounds; may be much looser than heuristic analysis.\n" ++
    s!"eps={c.eps}, actDerivBound={c.actDerivBound}, \
softmaxJacobianNormInfWorst={c.softmaxJacobianNormInfWorst}\n" ++
    s!"totalAmplificationFactor={c.totalAmplificationFactor}\n"
  let body :=
    c.layers.foldl (fun acc l =>
      acc ++
      s!"Layer {l.layerIdx}: C={l.C} (attn={l.attnWeightContribution}, \
mlp={l.mlpWeightContribution}, ln1Bound={l.ln1Bound}, ln2Bound={l.ln2Bound})\n") ""
  header ++ body

instance : ToString ModelCert := ⟨toString⟩

end ModelCert

end Nfp.Sound
