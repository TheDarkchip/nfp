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
  /-- Optional local variance lower bound used for LN1 (if available). -/
  ln1VarianceLowerBound? : Option Rat
  /-- Optional local variance lower bound used for LN2 (if available). -/
  ln2VarianceLowerBound? : Option Rat
  ln1Bound : Rat
  ln2Bound : Rat
  attnWeightContribution : Rat
  mlpWeightContribution : Rat
  C : Rat
  deriving Repr

namespace LayerAmplificationCert

instance : Inhabited LayerAmplificationCert :=
  ⟨{
    layerIdx := 0
    ln1MaxAbsGamma := 0
    ln2MaxAbsGamma := 0
    ln1VarianceLowerBound? := none
    ln2VarianceLowerBound? := none
    ln1Bound := 0
    ln2Bound := 0
    attnWeightContribution := 0
    mlpWeightContribution := 0
    C := 0
  }⟩

/-- Internal consistency checks for per-layer bounds. -/
def Valid (eps : Rat) (l : LayerAmplificationCert) : Prop :=
  l.ln1Bound =
      (match l.ln1VarianceLowerBound? with
       | some v => layerNormOpBoundLocal l.ln1MaxAbsGamma v eps
       | none => layerNormOpBoundConservative l.ln1MaxAbsGamma eps) ∧
    l.ln2Bound =
      (match l.ln2VarianceLowerBound? with
       | some v => layerNormOpBoundLocal l.ln2MaxAbsGamma v eps
       | none => layerNormOpBoundConservative l.ln2MaxAbsGamma eps) ∧
    l.C = l.attnWeightContribution + l.mlpWeightContribution

instance (eps : Rat) (l : LayerAmplificationCert) : Decidable (Valid eps l) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (eps : Rat) (l : LayerAmplificationCert) : Bool :=
  decide (Valid eps l)

theorem check_iff (eps : Rat) (l : LayerAmplificationCert) :
    l.check eps = true ↔ l.Valid eps := by
  simp [check]

end LayerAmplificationCert

/-- Model-level certification report. -/
structure ModelCert where
  modelPath : String
  inputPath? : Option String
  inputDelta : Rat
  eps : Rat
  actDerivBound : Rat
  softmaxJacobianNormInfWorst : Rat
  layers : Array LayerAmplificationCert
  totalAmplificationFactor : Rat
  deriving Repr

namespace ModelCert

/-- Internal consistency checks for a reported sound certificate. -/
def Valid (c : ModelCert) : Prop :=
  c.softmaxJacobianNormInfWorst = Nfp.Sound.softmaxJacobianNormInfWorst ∧
    List.Forall₂
      (fun i l => l.layerIdx = i ∧ LayerAmplificationCert.Valid c.eps l)
      (List.range c.layers.size) c.layers.toList ∧
    c.totalAmplificationFactor =
      c.layers.foldl (fun acc l => acc * (1 + l.C)) 1

instance (c : ModelCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : ModelCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : ModelCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

/-- Pretty printer. -/
def toString (c : ModelCert) : String :=
  let header :=
    s!"SOUND mode: conservative bounds; may be much looser than heuristic analysis.\n" ++
    (match c.inputPath? with
     | some p => s!"input={p}, delta={c.inputDelta}\n"
     | none => "") ++
    s!"eps={c.eps}, actDerivBound={c.actDerivBound}, \
softmaxJacobianNormInfWorst={c.softmaxJacobianNormInfWorst}\n" ++
    s!"totalAmplificationFactor={c.totalAmplificationFactor}\n"
  let body :=
    c.layers.foldl (fun acc l =>
      acc ++
      s!"Layer {l.layerIdx}: C={l.C} (attn={l.attnWeightContribution}, \
mlp={l.mlpWeightContribution}, ln1Bound={l.ln1Bound}, ln2Bound={l.ln2Bound}" ++
        (match l.ln1VarianceLowerBound? with
         | some v => s!", ln1Var≥{v}"
         | none => "") ++
        (match l.ln2VarianceLowerBound? with
         | some v => s!", ln2Var≥{v}"
         | none => "") ++
        ")\n") ""
  header ++ body

instance : ToString ModelCert := ⟨toString⟩

end ModelCert

end Nfp.Sound
