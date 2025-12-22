-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Mathlib.Data.Rat.Cast.Order
import Nfp.SignedMixer
import Nfp.Sound.Bounds

namespace Nfp.Sound

/-!
# Sound certification report structures

The sound path computes conservative bounds using exact `Rat` arithmetic.
These values are meant to be *trusted* (up to Lean's kernel/runtime), but may be
much looser than the Float-based heuristic analysis.
-/


/-- Per-layer conservative residual amplification constant `Cᵢ`
(bounds ‖layerJacobian - I‖) and its components.
`C` uses the safe algebraic form `attn + mlp + attn*mlp`. -/
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
  /-- Lower bound on the softmax probability interval used for Jacobian bounds. -/
  softmaxProbLo : Rat
  /-- Upper bound on the softmax probability interval used for Jacobian bounds. -/
  softmaxProbHi : Rat
  attnCoeff : Rat
  mlpCoeff : Rat
  /-- Upper bound on the operator norm of the MLP input weights. -/
  mlpWinBound : Rat
  /-- Upper bound on the operator norm of the MLP output weights. -/
  mlpWoutBound : Rat
  /-- Upper bound on the max GeLU derivative over this layer's preactivations. -/
  mlpActDerivBound : Rat
  attnWeightContribution : Rat
  mlpWeightContribution : Rat
  /-- Combined residual amplification bound: `attn + mlp + attn*mlp`. -/
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
    softmaxProbLo := 0
    softmaxProbHi := 1
    attnCoeff := 0
    mlpCoeff := 0
    mlpWinBound := 0
    mlpWoutBound := 0
    mlpActDerivBound := 0
    attnWeightContribution := 0
    mlpWeightContribution := 0
    C := 0
  }⟩

/-- Internal consistency checks for per-layer bounds. -/
def Valid (eps : Rat) (sqrtPrecBits : Nat) (l : LayerAmplificationCert) : Prop :=
  l.ln1Bound =
      (match l.ln1VarianceLowerBound? with
       | some v => layerNormOpBoundLocal l.ln1MaxAbsGamma v eps sqrtPrecBits
       | none => layerNormOpBoundConservative l.ln1MaxAbsGamma eps sqrtPrecBits) ∧
    l.ln2Bound =
      (match l.ln2VarianceLowerBound? with
       | some v => layerNormOpBoundLocal l.ln2MaxAbsGamma v eps sqrtPrecBits
       | none => layerNormOpBoundConservative l.ln2MaxAbsGamma eps sqrtPrecBits) ∧
    l.attnWeightContribution =
      l.ln1Bound *
        softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi *
        l.attnCoeff ∧
    l.mlpCoeff = l.mlpWinBound * l.mlpWoutBound ∧
    l.mlpWeightContribution =
      l.ln2Bound * (l.mlpCoeff * l.mlpActDerivBound) ∧
    l.C =
      l.attnWeightContribution + l.mlpWeightContribution +
        l.attnWeightContribution * l.mlpWeightContribution

instance (eps : Rat) (sqrtPrecBits : Nat) (l : LayerAmplificationCert) :
    Decidable (Valid eps sqrtPrecBits l) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (eps : Rat) (sqrtPrecBits : Nat) (l : LayerAmplificationCert) : Bool :=
  decide (Valid eps sqrtPrecBits l)

theorem check_iff (eps : Rat) (sqrtPrecBits : Nat) (l : LayerAmplificationCert) :
    l.check eps sqrtPrecBits = true ↔ l.Valid eps sqrtPrecBits := by
  simp [check]

/-- Extract the `C` identity from `Valid`. -/
theorem c_eq_of_valid (eps : Rat) (sqrtPrecBits : Nat) (l : LayerAmplificationCert)
    (h : l.Valid eps sqrtPrecBits) :
    l.C = l.attnWeightContribution + l.mlpWeightContribution +
      l.attnWeightContribution * l.mlpWeightContribution := by
  rcases h with ⟨_hln1, _hln2, _hattn, _hmlpCoeff, _hmlp, hC⟩
  exact hC

/-- Extract the attention contribution identity from `Valid`. -/
theorem attnWeightContribution_eq_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits) :
    l.attnWeightContribution =
      l.ln1Bound * softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi * l.attnCoeff := by
  rcases h with ⟨_hln1, _hln2, hattn, _hmlpCoeff, _hmlp, _hC⟩
  exact hattn

/-- Extract the MLP coefficient identity from `Valid`. -/
theorem mlpCoeff_eq_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits) :
    l.mlpCoeff = l.mlpWinBound * l.mlpWoutBound := by
  rcases h with ⟨_hln1, _hln2, _hattn, hCoeff, _hmlp, _hC⟩
  exact hCoeff

/-- Extract the MLP contribution identity from `Valid`. -/
theorem mlpWeightContribution_eq_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits) :
    l.mlpWeightContribution = l.ln2Bound * (l.mlpCoeff * l.mlpActDerivBound) := by
  rcases h with ⟨_hln1, _hln2, _hattn, _hmlpCoeff, hmlp, _hC⟩
  exact hmlp

/-- Cast the `C` identity to `ℝ` using `Valid`. -/
theorem c_eq_cast_of_valid (eps : Rat) (sqrtPrecBits : Nat) (l : LayerAmplificationCert)
    (h : l.Valid eps sqrtPrecBits) :
    (l.C : ℝ) =
      (l.attnWeightContribution : ℝ) +
        (l.mlpWeightContribution : ℝ) +
          (l.attnWeightContribution : ℝ) * (l.mlpWeightContribution : ℝ) := by
  have hC := c_eq_of_valid eps sqrtPrecBits l h
  have hC' := congrArg (fun (x : Rat) => (x : ℝ)) hC
  simpa [Rat.cast_add, Rat.cast_mul, add_assoc] using hC'

/-- Cast the attention contribution identity to `ℝ` using `Valid`. -/
theorem attnWeightContribution_eq_cast_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits) :
    (l.attnWeightContribution : ℝ) =
      (l.ln1Bound : ℝ) *
        (softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi : ℝ) *
        (l.attnCoeff : ℝ) := by
  have hAttn := attnWeightContribution_eq_of_valid eps sqrtPrecBits l h
  have hAttn' := congrArg (fun (x : Rat) => (x : ℝ)) hAttn
  simpa [Rat.cast_mul, mul_assoc] using hAttn'

/-- Cast the MLP coefficient identity to `ℝ` using `Valid`. -/
theorem mlpCoeff_eq_cast_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits) :
    (l.mlpCoeff : ℝ) = (l.mlpWinBound : ℝ) * (l.mlpWoutBound : ℝ) := by
  have hCoeff := mlpCoeff_eq_of_valid eps sqrtPrecBits l h
  have hCoeff' := congrArg (fun (x : Rat) => (x : ℝ)) hCoeff
  simpa [Rat.cast_mul] using hCoeff'

/-- Cast the MLP contribution identity to `ℝ` using `Valid`. -/
theorem mlpWeightContribution_eq_cast_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits) :
    (l.mlpWeightContribution : ℝ) =
      (l.ln2Bound : ℝ) * ((l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ)) := by
  have hMlp := mlpWeightContribution_eq_of_valid eps sqrtPrecBits l h
  have hMlp' := congrArg (fun (x : Rat) => (x : ℝ)) hMlp
  simpa [Rat.cast_mul, mul_assoc] using hMlp'

/-- Residual composition bound from component bounds and a cast `C` identity. -/
theorem residual_bound_of_component_bounds
    {S : Type*} [Fintype S] [DecidableEq S] [Nonempty S]
    (l : LayerAmplificationCert)
    (A M : SignedMixer S S)
    (hC : (l.C : ℝ) =
      (l.attnWeightContribution : ℝ) +
        (l.mlpWeightContribution : ℝ) +
          (l.attnWeightContribution : ℝ) * (l.mlpWeightContribution : ℝ))
    (hA : SignedMixer.operatorNormBound A ≤ (l.attnWeightContribution : ℝ))
    (hM : SignedMixer.operatorNormBound M ≤ (l.mlpWeightContribution : ℝ)) :
    SignedMixer.operatorNormBound
        ((SignedMixer.identity + A).comp (SignedMixer.identity + M) - SignedMixer.identity)
        ≤ (l.C : ℝ) := by
  have hres :=
    SignedMixer.operatorNormBound_residual_comp_le_of_bounds
      (A := A) (M := M)
      (a := (l.attnWeightContribution : ℝ))
      (b := (l.mlpWeightContribution : ℝ)) hA hM
  simpa [hC] using hres

/-- Residual composition bound from `Valid` plus component bounds. -/
theorem residual_bound_of_component_bounds_valid
    {S : Type*} [Fintype S] [DecidableEq S] [Nonempty S]
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (A M : SignedMixer S S)
    (hValid : l.Valid eps sqrtPrecBits)
    (hA : SignedMixer.operatorNormBound A ≤ (l.attnWeightContribution : ℝ))
    (hM : SignedMixer.operatorNormBound M ≤ (l.mlpWeightContribution : ℝ)) :
    SignedMixer.operatorNormBound
        ((SignedMixer.identity + A).comp (SignedMixer.identity + M) - SignedMixer.identity)
        ≤ (l.C : ℝ) := by
  have hC := c_eq_cast_of_valid eps sqrtPrecBits l hValid
  exact residual_bound_of_component_bounds (l := l) (A := A) (M := M) hC hA hM


theorem Valid_spec : Valid = Valid := rfl
theorem check_spec : check = check := rfl

end LayerAmplificationCert

/-- Model-level certification report. -/
structure ModelCert where
  modelPath : String
  inputPath? : Option String
  inputDelta : Rat
  eps : Rat
  /-- Precision in dyadic bits for local LayerNorm bounds. -/
  soundnessBits : Nat
  /-- Which GeLU derivative target the model uses. -/
  geluDerivTarget : GeluDerivTarget
  actDerivBound : Rat
  softmaxJacobianNormInfWorst : Rat
  layers : Array LayerAmplificationCert
  totalAmplificationFactor : Rat
  deriving Repr

namespace ModelCert

/-- Internal consistency checks for a reported sound certificate. -/
def Valid (c : ModelCert) : Prop :=
  0 < c.eps ∧
    c.softmaxJacobianNormInfWorst = Nfp.Sound.softmaxJacobianNormInfWorst ∧
    c.actDerivBound = c.layers.foldl (fun acc l => max acc l.mlpActDerivBound) 0 ∧
    List.Forall₂
      (fun i l => l.layerIdx = i ∧ LayerAmplificationCert.Valid c.eps c.soundnessBits l)
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
    s!"eps={c.eps}, soundnessBits={c.soundnessBits}, " ++
    s!"geluDerivTarget={geluDerivTargetToString c.geluDerivTarget}, " ++
    s!"actDerivBound={c.actDerivBound}, " ++
    s!"softmaxJacobianNormInfWorst={c.softmaxJacobianNormInfWorst}\n" ++
    s!"totalAmplificationFactor={c.totalAmplificationFactor}\n"
  let body :=
    c.layers.foldl (fun acc l =>
      acc ++
      s!"Layer {l.layerIdx}: C={l.C} (attn={l.attnWeightContribution}, \
mlp={l.mlpWeightContribution}, cross={l.attnWeightContribution * l.mlpWeightContribution}, \
mlpWinBound={l.mlpWinBound}, mlpWoutBound={l.mlpWoutBound}, \
mlpActDerivBound={l.mlpActDerivBound}, \
ln1Bound={l.ln1Bound}, ln2Bound={l.ln2Bound}" ++
        (match l.ln1VarianceLowerBound? with
         | some v => s!", ln1Var≥{v}"
         | none => "") ++
        (match l.ln2VarianceLowerBound? with
         | some v => s!", ln2Var≥{v}"
         | none => "") ++
        ")\n") ""
  header ++ body

instance : ToString ModelCert := ⟨toString⟩

theorem Valid_spec : Valid = Valid := rfl
theorem check_spec : check = check := rfl
theorem toString_spec : toString = toString := rfl

end ModelCert

/-! ### Specs -/

end Nfp.Sound
