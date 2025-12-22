-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Rat.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Data.Finset.Lattice.Fold
import Nfp.Linearization
import Nfp.SignedMixer
import Nfp.Sound.Bounds
import Nfp.Sound.Cert

namespace Nfp.Sound

open scoped BigOperators

namespace RatMatrix

variable {S T : Type*} [Fintype S] [Fintype T] [DecidableEq S] [DecidableEq T]

/-- Cast a rational matrix to a real SignedMixer. -/
noncomputable def toSignedMixer (M : RatMatrix S T) : SignedMixer S T :=
  ⟨fun i j => (M.w i j : ℝ)⟩

omit [DecidableEq S] [DecidableEq T] in
theorem toSignedMixer_spec (M : RatMatrix S T) : toSignedMixer M = toSignedMixer M := rfl

lemma ratAbs_eq_abs (x : Rat) : ratAbs x = |x| := by
  by_cases h : x < 0
  · simp [ratAbs, h, abs_of_neg h]
  · have h' : 0 ≤ x := le_of_not_gt h
    simp [ratAbs, h, abs_of_nonneg h']

/-- Casting the rational bound matches the real row-sum operator norm bound. -/
theorem operatorNormBound_cast (M : RatMatrix S T) [Nonempty S] :
    (operatorNormBound M : ℝ) = SignedMixer.operatorNormBound (M.toSignedMixer) := by
  classical
  have hsup_cast :
      (operatorNormBound M : ℝ) =
        Finset.sup' Finset.univ (Finset.univ_nonempty (α := S))
          (fun i => ((rowAbsSum M i : Rat) : ℝ)) := by
    apply le_antisymm
    · rcases Finset.exists_mem_eq_sup'
          (s := Finset.univ) (H := Finset.univ_nonempty (α := S))
          (f := fun i => rowAbsSum M i) with ⟨i, hi, hsup⟩
      have hcast : (operatorNormBound M : ℝ) = (rowAbsSum M i : ℝ) := by
        simp [operatorNormBound, hsup]
      calc
        (operatorNormBound M : ℝ) = (rowAbsSum M i : ℝ) := hcast
        _ ≤ Finset.sup' Finset.univ (Finset.univ_nonempty (α := S))
            (fun j => ((rowAbsSum M j : Rat) : ℝ)) := by
              exact Finset.le_sup' (s := Finset.univ)
                (f := fun j => ((rowAbsSum M j : Rat) : ℝ)) hi
    · refine (Finset.sup'_le_iff (s := Finset.univ)
          (H := Finset.univ_nonempty (α := S))
          (f := fun i => ((rowAbsSum M i : Rat) : ℝ))
          (a := (operatorNormBound M : ℝ))).2 ?_
      intro i hi
      have hle : rowAbsSum M i ≤ operatorNormBound M := by
        exact Finset.le_sup' (s := Finset.univ) (f := fun j => rowAbsSum M j) hi
      exact (Rat.cast_le (K := ℝ)).2 hle
  calc
    (operatorNormBound M : ℝ)
        = Finset.sup' Finset.univ (Finset.univ_nonempty (α := S))
            (fun i => ((rowAbsSum M i : Rat) : ℝ)) := hsup_cast
    _ = SignedMixer.operatorNormBound (M.toSignedMixer) := by
          simp [SignedMixer.operatorNormBound, rowAbsSum, toSignedMixer,
            ratAbs_eq_abs, Rat.cast_sum, Rat.cast_abs]

/-- Casted row-major bound agrees with the `SignedMixer` operator norm bound. -/
theorem matrixNormInfOfRowMajor_cast (rows cols : Nat) (data : Array Rat)
    [Nonempty (Fin rows)] (h : rows ≠ 0) :
    (matrixNormInfOfRowMajor rows cols data : ℝ) =
      SignedMixer.operatorNormBound (RatMatrix.ofRowMajor rows cols data).toSignedMixer := by
  classical
  simpa [matrixNormInfOfRowMajor, h] using
    (operatorNormBound_cast (M := RatMatrix.ofRowMajor rows cols data))

end RatMatrix

/-! ## Certificate-to-Jacobian bridge -/

/-- Assumptions needed to link certificate fields to component operator-norm bounds. -/
structure LayerComponentNormAssumptions
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) : Prop where
  ln1Bound :
    SignedMixer.operatorNormBound (D.ln1Jacobians i) ≤ (l.ln1Bound : ℝ)
  ln1Bound_nonneg : 0 ≤ (l.ln1Bound : ℝ)
  attnFullBound :
    SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
      (softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi : ℝ) *
        (l.attnCoeff : ℝ)
  ln2Bound :
    SignedMixer.operatorNormBound (D.ln2Jacobians i) ≤ (l.ln2Bound : ℝ)
  ln2Bound_nonneg : 0 ≤ (l.ln2Bound : ℝ)
  mlpWinBound :
    SignedMixer.operatorNormBound (D.mlpFactors i).win ≤ (l.mlpWinBound : ℝ)
  mlpWinBound_nonneg : 0 ≤ (l.mlpWinBound : ℝ)
  mlpWoutBound :
    SignedMixer.operatorNormBound (D.mlpFactors i).wout ≤ (l.mlpWoutBound : ℝ)
  mlpWoutBound_nonneg : 0 ≤ (l.mlpWoutBound : ℝ)
  mlpDerivBound :
    ∀ j, |(D.mlpFactors i).deriv j| ≤ (l.mlpActDerivBound : ℝ)

/-- MLP coefficient bound from certificate validity and component bounds. -/
theorem mlp_coeff_bound_of_valid
    {n d : Type*} [Fintype n] [Fintype d] [Nonempty n] [Nonempty d]
    (F : MLPFactorization (n := n) (d := d))
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (hValid : l.Valid eps sqrtPrecBits)
    (hWin : SignedMixer.operatorNormBound F.win ≤ (l.mlpWinBound : ℝ))
    (hWin_nonneg : 0 ≤ (l.mlpWinBound : ℝ))
    (hWout : SignedMixer.operatorNormBound F.wout ≤ (l.mlpWoutBound : ℝ)) :
    SignedMixer.operatorNormBound F.win *
      SignedMixer.operatorNormBound F.wout ≤ (l.mlpCoeff : ℝ) := by
  have hWout_nonneg :
      0 ≤ SignedMixer.operatorNormBound F.wout :=
    SignedMixer.operatorNormBound_nonneg (M := F.wout)
  have hMul1 :
      SignedMixer.operatorNormBound F.win * SignedMixer.operatorNormBound F.wout ≤
        (l.mlpWinBound : ℝ) * SignedMixer.operatorNormBound F.wout := by
    exact mul_le_mul_of_nonneg_right hWin hWout_nonneg
  have hMul2 :
      (l.mlpWinBound : ℝ) * SignedMixer.operatorNormBound F.wout ≤
        (l.mlpWinBound : ℝ) * (l.mlpWoutBound : ℝ) := by
    exact mul_le_mul_of_nonneg_left hWout hWin_nonneg
  have hMul :
      SignedMixer.operatorNormBound F.win * SignedMixer.operatorNormBound F.wout ≤
        (l.mlpWinBound : ℝ) * (l.mlpWoutBound : ℝ) := by
    exact le_trans hMul1 hMul2
  have hCoeff := LayerAmplificationCert.mlpCoeff_eq_cast_of_valid
    (eps := eps) (sqrtPrecBits := sqrtPrecBits) (l := l) hValid
  simpa [hCoeff] using hMul

/-- MLP operator-norm bound from a factored Jacobian plus coefficient bounds. -/
theorem mlp_bound_of_factorization
    {n d : Type*} [Fintype n] [Fintype d] [Nonempty n] [Nonempty d]
    (F : MLPFactorization (n := n) (d := d))
    (l : LayerAmplificationCert)
    (hDeriv : ∀ j, |F.deriv j| ≤ (l.mlpActDerivBound : ℝ))
    (hCoeff :
      SignedMixer.operatorNormBound F.win *
        SignedMixer.operatorNormBound F.wout ≤ (l.mlpCoeff : ℝ)) :
    SignedMixer.operatorNormBound F.jacobian ≤
      (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ) := by
  classical
  let a := SignedMixer.operatorNormBound F.win
  let b := SignedMixer.operatorNormBound F.wout
  have hBound :
      SignedMixer.operatorNormBound F.jacobian ≤
        a * (l.mlpActDerivBound : ℝ) * b := by
    simpa using
      (operatorNormBound_comp_diagMixer_comp_le
        (A := F.win) (B := F.wout) (d := F.deriv)
        (a := a) (c := (l.mlpActDerivBound : ℝ)) (b := b)
        (hA := by simp [a]) (hB := by simp [b]) hDeriv)
  have hAct_nonneg : 0 ≤ (l.mlpActDerivBound : ℝ) := by
    rcases (inferInstance : Nonempty F.hidden) with ⟨j⟩
    have h := hDeriv j
    exact le_trans (abs_nonneg _) h
  have hMul :
      a * (l.mlpActDerivBound : ℝ) * b ≤
        (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ) := by
    have hMul' :
        (a * b) * (l.mlpActDerivBound : ℝ) ≤
          (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ) := by
      exact mul_le_mul_of_nonneg_right hCoeff hAct_nonneg
    have hRearrange :
        a * (l.mlpActDerivBound : ℝ) * b =
          (a * b) * (l.mlpActDerivBound : ℝ) := by
      calc
        a * (l.mlpActDerivBound : ℝ) * b =
            a * b * (l.mlpActDerivBound : ℝ) := by
              simpa [mul_assoc] using (mul_right_comm a (l.mlpActDerivBound : ℝ) b)
        _ = (a * b) * (l.mlpActDerivBound : ℝ) := by simp [mul_assoc]
    simpa [hRearrange] using hMul'
  exact le_trans hBound hMul

/-- Attention-component bound from certificate identities and component bounds. -/
theorem attn_component_bound_of_cert
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (hValid : l.Valid eps sqrtPrecBits)
    (hLn1 :
      SignedMixer.operatorNormBound (D.ln1Jacobians i) ≤ (l.ln1Bound : ℝ))
    (hLn1_nonneg : 0 ≤ (l.ln1Bound : ℝ))
    (hFull :
      SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi : ℝ) *
          (l.attnCoeff : ℝ)) :
    SignedMixer.operatorNormBound
        ((D.ln1Jacobians i).comp (D.layers i).fullJacobian)
        ≤ (l.attnWeightContribution : ℝ) := by
  classical
  let softmaxBound : ℝ :=
    (softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi : ℝ)
  have hcomp :
      SignedMixer.operatorNormBound
          ((D.ln1Jacobians i).comp (D.layers i).fullJacobian) ≤
        SignedMixer.operatorNormBound (D.ln1Jacobians i) *
          SignedMixer.operatorNormBound (D.layers i).fullJacobian := by
    simpa using
      (SignedMixer.operatorNormBound_comp_le
        (M := D.ln1Jacobians i) (N := (D.layers i).fullJacobian))
  have hFull_nonneg :
      0 ≤ SignedMixer.operatorNormBound (D.layers i).fullJacobian :=
    SignedMixer.operatorNormBound_nonneg (M := (D.layers i).fullJacobian)
  have hmul1 :
      SignedMixer.operatorNormBound (D.ln1Jacobians i) *
          SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (l.ln1Bound : ℝ) * SignedMixer.operatorNormBound (D.layers i).fullJacobian := by
    exact mul_le_mul_of_nonneg_right hLn1 hFull_nonneg
  have hmul2 :
      (l.ln1Bound : ℝ) * SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (l.ln1Bound : ℝ) * (softmaxBound * (l.attnCoeff : ℝ)) := by
    simpa [softmaxBound] using
      (mul_le_mul_of_nonneg_left hFull hLn1_nonneg)
  have hmul :
      SignedMixer.operatorNormBound (D.ln1Jacobians i) *
          SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (l.ln1Bound : ℝ) * (softmaxBound * (l.attnCoeff : ℝ)) := by
    exact le_trans hmul1 hmul2
  have hAttn :
      (l.ln1Bound : ℝ) * (softmaxBound * (l.attnCoeff : ℝ)) =
        (l.attnWeightContribution : ℝ) := by
    have hAttn :=
      LayerAmplificationCert.attnWeightContribution_eq_cast_of_valid
        eps sqrtPrecBits l hValid
    simpa [softmaxBound, mul_assoc] using hAttn.symm
  calc
    SignedMixer.operatorNormBound
        ((D.ln1Jacobians i).comp (D.layers i).fullJacobian)
        ≤ SignedMixer.operatorNormBound (D.ln1Jacobians i) *
          SignedMixer.operatorNormBound (D.layers i).fullJacobian := hcomp
    _ ≤ (l.ln1Bound : ℝ) * (softmaxBound * (l.attnCoeff : ℝ)) := hmul
    _ = (l.attnWeightContribution : ℝ) := hAttn

/-- MLP-component bound from certificate identities and component bounds. -/
theorem mlp_component_bound_of_cert
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (hValid : l.Valid eps sqrtPrecBits)
    (hLn2 :
      SignedMixer.operatorNormBound (D.ln2Jacobians i) ≤ (l.ln2Bound : ℝ))
    (hLn2_nonneg : 0 ≤ (l.ln2Bound : ℝ))
    (hMlp :
      SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ)) :
    SignedMixer.operatorNormBound
        ((D.ln2Jacobians i).comp (D.mlpJacobians i))
        ≤ (l.mlpWeightContribution : ℝ) := by
  classical
  let mlpBound : ℝ := (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ)
  have hcomp :
      SignedMixer.operatorNormBound
          ((D.ln2Jacobians i).comp (D.mlpJacobians i)) ≤
        SignedMixer.operatorNormBound (D.ln2Jacobians i) *
          SignedMixer.operatorNormBound (D.mlpJacobians i) := by
    simpa using
      (SignedMixer.operatorNormBound_comp_le
        (M := D.ln2Jacobians i) (N := D.mlpJacobians i))
  have hMlp_nonneg : 0 ≤ SignedMixer.operatorNormBound (D.mlpJacobians i) :=
    SignedMixer.operatorNormBound_nonneg (M := D.mlpJacobians i)
  have hmul1 :
      SignedMixer.operatorNormBound (D.ln2Jacobians i) *
          SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.ln2Bound : ℝ) * SignedMixer.operatorNormBound (D.mlpJacobians i) := by
    exact mul_le_mul_of_nonneg_right hLn2 hMlp_nonneg
  have hmul2 :
      (l.ln2Bound : ℝ) * SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.ln2Bound : ℝ) * mlpBound := by
    simpa [mlpBound] using
      (mul_le_mul_of_nonneg_left hMlp hLn2_nonneg)
  have hmul :
      SignedMixer.operatorNormBound (D.ln2Jacobians i) *
          SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.ln2Bound : ℝ) * mlpBound := by
    exact le_trans hmul1 hmul2
  have hMlpEq :
      (l.ln2Bound : ℝ) * mlpBound = (l.mlpWeightContribution : ℝ) := by
    have hMlpEq :=
      LayerAmplificationCert.mlpWeightContribution_eq_cast_of_valid
        eps sqrtPrecBits l hValid
    simpa [mlpBound, mul_assoc] using hMlpEq.symm
  calc
    SignedMixer.operatorNormBound
        ((D.ln2Jacobians i).comp (D.mlpJacobians i))
        ≤ SignedMixer.operatorNormBound (D.ln2Jacobians i) *
          SignedMixer.operatorNormBound (D.mlpJacobians i) := hcomp
    _ ≤ (l.ln2Bound : ℝ) * mlpBound := hmul
    _ = (l.mlpWeightContribution : ℝ) := hMlpEq

/-- Layer Jacobian residual bound from a layer amplification certificate. -/
theorem layerJacobian_residual_bound_of_cert
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (hValid : l.Valid eps sqrtPrecBits)
    (hA :
      SignedMixer.operatorNormBound
        ((D.ln1Jacobians i).comp (D.layers i).fullJacobian)
        ≤ (l.attnWeightContribution : ℝ))
    (hM :
      SignedMixer.operatorNormBound
        ((D.ln2Jacobians i).comp (D.mlpJacobians i))
        ≤ (l.mlpWeightContribution : ℝ)) :
    SignedMixer.operatorNormBound
        (D.layerJacobian i - SignedMixer.identity) ≤ (l.C : ℝ) := by
  have hC := LayerAmplificationCert.c_eq_cast_of_valid eps sqrtPrecBits l hValid
  have hres :=
    DeepLinearization.layerJacobian_residual_bound
      (D := D) (i := i)
      (A := (l.attnWeightContribution : ℝ))
      (M := (l.mlpWeightContribution : ℝ)) hA hM
  simpa [hC] using hres

/-- Layer Jacobian residual bound from a certificate, with component-norm assumptions. -/
theorem layerJacobian_residual_bound_of_cert_components
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (hValid : l.Valid eps sqrtPrecBits)
    (hLn1 :
      SignedMixer.operatorNormBound (D.ln1Jacobians i) ≤ (l.ln1Bound : ℝ))
    (hLn1_nonneg : 0 ≤ (l.ln1Bound : ℝ))
    (hFull :
      SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (softmaxJacobianNormInfBound l.softmaxProbLo l.softmaxProbHi : ℝ) *
          (l.attnCoeff : ℝ))
    (hLn2 :
      SignedMixer.operatorNormBound (D.ln2Jacobians i) ≤ (l.ln2Bound : ℝ))
    (hLn2_nonneg : 0 ≤ (l.ln2Bound : ℝ))
    (hMlp :
      SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ)) :
    SignedMixer.operatorNormBound
        (D.layerJacobian i - SignedMixer.identity) ≤ (l.C : ℝ) := by
  have hA :=
    attn_component_bound_of_cert
      (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
      hValid hLn1 hLn1_nonneg hFull
  have hM :=
    mlp_component_bound_of_cert
      (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
      hValid hLn2 hLn2_nonneg hMlp
  exact layerJacobian_residual_bound_of_cert
    (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
    hValid hA hM

/-- Layer Jacobian residual bound from a certificate plus component-norm assumptions. -/
theorem layerJacobian_residual_bound_of_cert_assuming
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (hValid : l.Valid eps sqrtPrecBits)
    (h : LayerComponentNormAssumptions (D := D) (i := i) (l := l)) :
    SignedMixer.operatorNormBound
        (D.layerJacobian i - SignedMixer.identity) ≤ (l.C : ℝ) := by
  have hCoeff :
      SignedMixer.operatorNormBound (D.mlpFactors i).win *
        SignedMixer.operatorNormBound (D.mlpFactors i).wout ≤ (l.mlpCoeff : ℝ) :=
    mlp_coeff_bound_of_valid
      (F := D.mlpFactors i) (l := l)
      (eps := eps) (sqrtPrecBits := sqrtPrecBits) hValid
      h.mlpWinBound h.mlpWinBound_nonneg h.mlpWoutBound
  have hMlp :
      SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ) := by
    simpa using
      (mlp_bound_of_factorization (F := D.mlpFactors i) (l := l)
        h.mlpDerivBound hCoeff)
  exact layerJacobian_residual_bound_of_cert_components
    (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
    hValid h.ln1Bound h.ln1Bound_nonneg h.attnFullBound
    h.ln2Bound h.ln2Bound_nonneg hMlp

end Nfp.Sound
