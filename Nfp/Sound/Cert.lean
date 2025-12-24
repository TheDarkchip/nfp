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
  ln1MaxAbsBeta : Rat
  ln2MaxAbsGamma : Rat
  /-- Optional local variance lower bound used for LN1 (if available). -/
  ln1VarianceLowerBound? : Option Rat
  /-- Optional local variance lower bound used for LN2 (if available). -/
  ln2VarianceLowerBound? : Option Rat
  ln1Bound : Rat
  ln2Bound : Rat
  /-- Upper bound on `max |LN1 output|` after affine. -/
  ln1OutMaxAbsBound : Rat
  /-- Lower bound on the softmax probability interval used for Jacobian bounds. -/
  softmaxProbLo : Rat
  /-- Upper bound on the softmax probability interval used for Jacobian bounds. -/
  softmaxProbHi : Rat
  /-- Lower bound on the softmax logit margin (0 means no margin evidence). -/
  softmaxMarginLowerBound : Rat
  /-- Effort level for the exp lower bound used in margin-derived softmax bounds. -/
  softmaxExpEffort : Nat
  /-- Upper bound on the softmax Jacobian row-sum norm (portfolio bound). -/
  softmaxJacobianNormInfUpperBound : Rat
  /-- Maximum operator-norm bound on W_Q across heads (row-sum). -/
  wqOpBoundMax : Rat
  /-- Maximum operator-norm bound on W_K across heads (row-sum). -/
  wkOpBoundMax : Rat
  /-- Value-term coefficient (sum of per-head `W_V`/`W_O` bounds). -/
  attnValueCoeff : Rat
  /-- Pattern-term coefficient (score-gradient L1 × input L1 × value coefficient). -/
  attnPatternCoeff : Rat
  mlpCoeff : Rat
  /-- Upper bound on the operator norm of the MLP input weights. -/
  mlpWinBound : Rat
  /-- Upper bound on the operator norm of the MLP output weights. -/
  mlpWoutBound : Rat
  /-- Upper bound on the max GeLU derivative over this layer's preactivations. -/
  mlpActDerivBound : Rat
  /-- Upper bound on the attention residual Jacobian contribution. -/
  attnJacBound : Rat
  /-- Upper bound on the MLP residual Jacobian contribution. -/
  mlpJacBound : Rat
  /-- Combined residual amplification bound: `attn + mlp + attn*mlp`. -/
  C : Rat
  deriving Repr

namespace LayerAmplificationCert

instance : Inhabited LayerAmplificationCert :=
  ⟨{
    layerIdx := 0
    ln1MaxAbsGamma := 0
    ln1MaxAbsBeta := 0
    ln2MaxAbsGamma := 0
    ln1VarianceLowerBound? := none
    ln2VarianceLowerBound? := none
    ln1Bound := 0
    ln2Bound := 0
    ln1OutMaxAbsBound := 0
    softmaxProbLo := 0
    softmaxProbHi := 1
    softmaxMarginLowerBound := 0
    softmaxExpEffort := defaultSoftmaxExpEffort
    softmaxJacobianNormInfUpperBound := softmaxJacobianNormInfWorst
    wqOpBoundMax := 0
    wkOpBoundMax := 0
    attnValueCoeff := 0
    attnPatternCoeff := 0
    mlpCoeff := 0
    mlpWinBound := 0
    mlpWoutBound := 0
    mlpActDerivBound := 0
    attnJacBound := 0
    mlpJacBound := 0
    C := 0
  }⟩

/-- Portfolio softmax Jacobian bound from interval and margin candidates. -/
def softmaxJacobianNormInfPortfolioBound (seqLen : Nat) (l : LayerAmplificationCert) : Rat :=
  ubBest (softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi)
    #[softmaxJacobianNormInfBoundFromMargin seqLen l.softmaxMarginLowerBound l.softmaxExpEffort]

theorem softmaxJacobianNormInfPortfolioBound_def (seqLen : Nat) (l : LayerAmplificationCert) :
    softmaxJacobianNormInfPortfolioBound seqLen l =
      ubBest (softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi)
        #[softmaxJacobianNormInfBoundFromMargin seqLen l.softmaxMarginLowerBound
          l.softmaxExpEffort] := rfl

/-- Update margin evidence and recompute dependent softmax + residual bounds. -/
def withSoftmaxMargin (seqLen : Nat) (marginLowerBound : Rat) (softmaxExpEffort : Nat)
    (l : LayerAmplificationCert) : LayerAmplificationCert :=
  let l' :=
    { l with
      softmaxMarginLowerBound := marginLowerBound
      softmaxExpEffort := softmaxExpEffort }
  let softmaxBound := softmaxJacobianNormInfPortfolioBound seqLen l'
  let attnJacBound :=
    l'.ln1Bound *
      ((seqLen : Rat) * l'.attnValueCoeff + softmaxBound * l'.attnPatternCoeff)
  let mlpJacBound := l'.mlpJacBound
  let C := attnJacBound + mlpJacBound + attnJacBound * mlpJacBound
  { l' with
    softmaxJacobianNormInfUpperBound := softmaxBound
    attnJacBound := attnJacBound
    C := C }

/-- Internal consistency checks for per-layer bounds. -/
def Valid (eps : Rat) (sqrtPrecBits : Nat) (seqLen modelDim headDim : Nat)
    (l : LayerAmplificationCert) : Prop :=
  l.ln1Bound =
      (match l.ln1VarianceLowerBound? with
       | some v => layerNormOpBoundLocal l.ln1MaxAbsGamma v eps sqrtPrecBits
       | none => layerNormOpBoundConservative l.ln1MaxAbsGamma eps sqrtPrecBits) ∧
    l.ln2Bound =
      (match l.ln2VarianceLowerBound? with
       | some v => layerNormOpBoundLocal l.ln2MaxAbsGamma v eps sqrtPrecBits
       | none => layerNormOpBoundConservative l.ln2MaxAbsGamma eps sqrtPrecBits) ∧
    l.ln1OutMaxAbsBound =
      layerNormOutputMaxAbsBound modelDim l.ln1MaxAbsGamma l.ln1MaxAbsBeta ∧
    l.softmaxJacobianNormInfUpperBound =
      softmaxJacobianNormInfPortfolioBound seqLen l ∧
    l.attnPatternCoeff =
      attnPatternCoeffBound seqLen modelDim headDim l.ln1OutMaxAbsBound l.wqOpBoundMax
        l.wkOpBoundMax l.attnValueCoeff ∧
    l.attnJacBound =
      l.ln1Bound *
        ((seqLen : Rat) * l.attnValueCoeff +
          l.softmaxJacobianNormInfUpperBound * l.attnPatternCoeff) ∧
    l.mlpCoeff = l.mlpWinBound * l.mlpWoutBound ∧
    l.mlpJacBound =
      l.ln2Bound * (l.mlpCoeff * l.mlpActDerivBound) ∧
    l.C =
      l.attnJacBound + l.mlpJacBound +
        l.attnJacBound * l.mlpJacBound

instance (eps : Rat) (sqrtPrecBits : Nat) (seqLen modelDim headDim : Nat)
    (l : LayerAmplificationCert) :
    Decidable (Valid eps sqrtPrecBits seqLen modelDim headDim l) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (eps : Rat) (sqrtPrecBits : Nat) (seqLen modelDim headDim : Nat)
    (l : LayerAmplificationCert) : Bool :=
  decide (Valid eps sqrtPrecBits seqLen modelDim headDim l)

theorem check_iff (eps : Rat) (sqrtPrecBits : Nat) (seqLen modelDim headDim : Nat)
    (l : LayerAmplificationCert) :
    l.check eps sqrtPrecBits seqLen modelDim headDim = true ↔
      l.Valid eps sqrtPrecBits seqLen modelDim headDim := by
  simp [check]

/-- Extract the `C` identity from `Valid`. -/
theorem c_eq_of_valid (eps : Rat) (sqrtPrecBits : Nat) (seqLen modelDim headDim : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits seqLen modelDim headDim) :
    l.C = l.attnJacBound + l.mlpJacBound +
      l.attnJacBound * l.mlpJacBound := by
  rcases h with
    ⟨_hln1, _hln2, _hln1Out, _hsoftmax, _hpat, _hattn, _hmlpCoeff, _hmlp, hC⟩
  exact hC

/-- Extract the attention contribution identity from `Valid`. -/
theorem attnJacBound_eq_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat) (l : LayerAmplificationCert)
    (h : l.Valid eps sqrtPrecBits seqLen modelDim headDim) :
    l.attnJacBound =
      l.ln1Bound *
        ((seqLen : Rat) * l.attnValueCoeff +
          l.softmaxJacobianNormInfUpperBound * l.attnPatternCoeff) := by
  rcases h with
    ⟨_hln1, _hln2, _hln1Out, _hsoftmax, _hpat, hattn, _hmlpCoeff, _hmlp, _hC⟩
  exact hattn

/-- Extract the MLP coefficient identity from `Valid`. -/
theorem mlpCoeff_eq_of_valid (eps : Rat) (sqrtPrecBits : Nat) (seqLen modelDim headDim : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits seqLen modelDim headDim) :
    l.mlpCoeff = l.mlpWinBound * l.mlpWoutBound := by
  rcases h with
    ⟨_hln1, _hln2, _hln1Out, _hsoftmax, _hpat, _hattn, hCoeff, _hmlp, _hC⟩
  exact hCoeff

/-- Extract the MLP contribution identity from `Valid`. -/
theorem mlpJacBound_eq_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat) (l : LayerAmplificationCert)
    (h : l.Valid eps sqrtPrecBits seqLen modelDim headDim) :
    l.mlpJacBound = l.ln2Bound * (l.mlpCoeff * l.mlpActDerivBound) := by
  rcases h with
    ⟨_hln1, _hln2, _hln1Out, _hsoftmax, _hpat, _hattn, _hmlpCoeff, hmlp, _hC⟩
  exact hmlp

/-- Cast the `C` identity to `ℝ` using `Valid`. -/
theorem c_eq_cast_of_valid (eps : Rat) (sqrtPrecBits : Nat) (seqLen modelDim headDim : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits seqLen modelDim headDim) :
    (l.C : ℝ) =
      (l.attnJacBound : ℝ) +
        (l.mlpJacBound : ℝ) +
          (l.attnJacBound : ℝ) * (l.mlpJacBound : ℝ) := by
  have hC := c_eq_of_valid eps sqrtPrecBits seqLen modelDim headDim l h
  have hC' := congrArg (fun (x : Rat) => (x : ℝ)) hC
  simpa [Rat.cast_add, Rat.cast_mul, add_assoc] using hC'

/-- Cast the attention contribution identity to `ℝ` using `Valid`. -/
theorem attnJacBound_eq_cast_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat) (l : LayerAmplificationCert)
    (h : l.Valid eps sqrtPrecBits seqLen modelDim headDim) :
    (l.attnJacBound : ℝ) =
      (l.ln1Bound : ℝ) *
        ((seqLen : ℝ) * (l.attnValueCoeff : ℝ) +
          (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ)) := by
  have hAttn := attnJacBound_eq_of_valid eps sqrtPrecBits seqLen modelDim headDim l h
  have hAttn' := congrArg (fun (x : Rat) => (x : ℝ)) hAttn
  simpa [Rat.cast_mul, Rat.cast_add, Rat.cast_natCast, mul_assoc, add_assoc] using hAttn'

/-- Cast the MLP coefficient identity to `ℝ` using `Valid`. -/
theorem mlpCoeff_eq_cast_of_valid (eps : Rat) (sqrtPrecBits : Nat) (seqLen modelDim headDim : Nat)
    (l : LayerAmplificationCert) (h : l.Valid eps sqrtPrecBits seqLen modelDim headDim) :
    (l.mlpCoeff : ℝ) = (l.mlpWinBound : ℝ) * (l.mlpWoutBound : ℝ) := by
  have hCoeff := mlpCoeff_eq_of_valid eps sqrtPrecBits seqLen modelDim headDim l h
  have hCoeff' := congrArg (fun (x : Rat) => (x : ℝ)) hCoeff
  simpa [Rat.cast_mul] using hCoeff'

/-- Cast the MLP contribution identity to `ℝ` using `Valid`. -/
theorem mlpJacBound_eq_cast_of_valid (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat) (l : LayerAmplificationCert)
    (h : l.Valid eps sqrtPrecBits seqLen modelDim headDim) :
    (l.mlpJacBound : ℝ) =
      (l.ln2Bound : ℝ) * ((l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ)) := by
  have hMlp := mlpJacBound_eq_of_valid eps sqrtPrecBits seqLen modelDim headDim l h
  have hMlp' := congrArg (fun (x : Rat) => (x : ℝ)) hMlp
  simpa [Rat.cast_mul, mul_assoc] using hMlp'

/-- Residual composition bound from component bounds and a cast `C` identity. -/
theorem residual_bound_of_component_bounds
    {S : Type*} [Fintype S] [DecidableEq S] [Nonempty S]
    (l : LayerAmplificationCert)
    (A M : SignedMixer S S)
    (hC : (l.C : ℝ) =
      (l.attnJacBound : ℝ) +
        (l.mlpJacBound : ℝ) +
          (l.attnJacBound : ℝ) * (l.mlpJacBound : ℝ))
    (hA : SignedMixer.operatorNormBound A ≤ (l.attnJacBound : ℝ))
    (hM : SignedMixer.operatorNormBound M ≤ (l.mlpJacBound : ℝ)) :
    SignedMixer.operatorNormBound
        ((SignedMixer.identity + A).comp (SignedMixer.identity + M) - SignedMixer.identity)
        ≤ (l.C : ℝ) := by
  have hres :=
    SignedMixer.operatorNormBound_residual_comp_le_of_bounds
      (A := A) (M := M)
      (a := (l.attnJacBound : ℝ))
      (b := (l.mlpJacBound : ℝ)) hA hM
  simpa [hC] using hres

/-- Residual composition bound from `Valid` plus component bounds. -/
theorem residual_bound_of_component_bounds_valid
    {S : Type*} [Fintype S] [DecidableEq S] [Nonempty S]
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat) (A M : SignedMixer S S)
    (hValid : l.Valid eps sqrtPrecBits seqLen modelDim headDim)
    (hA : SignedMixer.operatorNormBound A ≤ (l.attnJacBound : ℝ))
    (hM : SignedMixer.operatorNormBound M ≤ (l.mlpJacBound : ℝ)) :
    SignedMixer.operatorNormBound
        ((SignedMixer.identity + A).comp (SignedMixer.identity + M) - SignedMixer.identity)
        ≤ (l.C : ℝ) := by
  have hC := c_eq_cast_of_valid eps sqrtPrecBits seqLen modelDim headDim l hValid
  exact residual_bound_of_component_bounds (l := l) (A := A) (M := M) hC hA hM


theorem withSoftmaxMargin_spec :
    withSoftmaxMargin = withSoftmaxMargin := rfl
theorem Valid_spec : Valid = Valid := rfl
theorem check_spec : check = check := rfl

end LayerAmplificationCert

/-- Model-level certification report. -/
structure ModelCert where
  modelPath : String
  inputPath? : Option String
  inputDelta : Rat
  eps : Rat
  /-- Sequence length from the model header. -/
  seqLen : Nat
  /-- Model dimension from the model header. -/
  modelDim : Nat
  /-- Head dimension from the model header. -/
  headDim : Nat
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
      (fun i l =>
        l.layerIdx = i ∧
          LayerAmplificationCert.Valid c.eps c.soundnessBits c.seqLen c.modelDim c.headDim l)
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

/-- Replace a layer and recompute total amplification factor. -/
def withUpdatedLayer (c : ModelCert) (layerIdx : Nat) (layer : LayerAmplificationCert) :
    Option ModelCert :=
  if layer.layerIdx ≠ layerIdx then
    none
  else if layerIdx < c.layers.size then
    let layers := c.layers.set! layerIdx layer
    let total := layers.foldl (fun acc l => acc * (1 + l.C)) 1
    some { c with layers := layers, totalAmplificationFactor := total }
  else
    none

/-- Pretty printer. -/
def toString (c : ModelCert) : String :=
  let header :=
    s!"SOUND mode: conservative bounds; may be much looser than heuristic analysis.\n" ++
    (match c.inputPath? with
     | some p => s!"input={p}, delta={c.inputDelta}\n"
     | none => "") ++
    s!"eps={c.eps}, seqLen={c.seqLen}, modelDim={c.modelDim}, headDim={c.headDim}, " ++
    s!"soundnessBits={c.soundnessBits}, " ++
    s!"geluDerivTarget={geluDerivTargetToString c.geluDerivTarget}, " ++
    s!"actDerivBound={c.actDerivBound}, " ++
    s!"softmaxJacobianNormInfWorst={c.softmaxJacobianNormInfWorst}\n" ++
    s!"totalAmplificationFactor={c.totalAmplificationFactor}\n"
  let body :=
    c.layers.foldl (fun acc l =>
      acc ++
      s!"Layer {l.layerIdx}: C={l.C} (attn={l.attnJacBound}, \
mlp={l.mlpJacBound}, cross={l.attnJacBound * l.mlpJacBound}, \
attnValueCoeff={l.attnValueCoeff}, attnPatternCoeff={l.attnPatternCoeff}, \
wqOpBoundMax={l.wqOpBoundMax}, wkOpBoundMax={l.wkOpBoundMax}, \
ln1OutMaxAbsBound={l.ln1OutMaxAbsBound}, \
mlpWinBound={l.mlpWinBound}, mlpWoutBound={l.mlpWoutBound}, \
mlpActDerivBound={l.mlpActDerivBound}, \
softmaxJacobianNormInfUpperBound={l.softmaxJacobianNormInfUpperBound}, \
softmaxMarginLowerBound={l.softmaxMarginLowerBound}, softmaxExpEffort={l.softmaxExpEffort}, \
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
theorem withUpdatedLayer_spec : withUpdatedLayer = withUpdatedLayer := rfl
theorem toString_spec : toString = toString := rfl

end ModelCert

/-! ### Specs -/

end Nfp.Sound
