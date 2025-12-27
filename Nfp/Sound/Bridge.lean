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
  attnValueBound :
    SignedMixer.operatorNormBound (valueTerm (D.layers i)) ≤
      (Fintype.card n : ℝ) * (l.attnValueCoeff : ℝ)
  attnPatternBound :
    SignedMixer.operatorNormBound (patternTerm (D.layers i)) ≤
      (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ)
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

/-- Pattern-term bound from the explicit formula and row-sum gradient/value bounds. -/
theorem attn_pattern_bound_of_explicit
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (G V : ℝ)
    (hGrad : ∀ (pos : n) (d_in : d) (q : n),
      ∑ k, |attentionGradient (D.layers i) q k pos d_in| ≤ G)
    (hValue : SignedMixer.operatorNormBound (valueOutputMixer (D.layers i)) ≤ V)
    (hEq : patternTerm (D.layers i) = patternTermExplicit (D.layers i)) :
    SignedMixer.operatorNormBound (patternTerm (D.layers i)) ≤
      (Fintype.card n : ℝ) * G * V := by
  simpa using
    (patternTerm_operatorNormBound_le_of_eq_explicit
      (L := D.layers i) (G := G) (V := V) hGrad hValue hEq)

/-- Pattern-term bound from certificate validity and attention-state assumptions. -/
theorem attn_pattern_bound_of_cert
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLenNat modelDimNat headDimNat : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLenNat modelDimNat headDimNat)
    (hSeqLen : seqLenNat = Fintype.card n)
    (hModelDim : modelDimNat = Fintype.card d)
    (hScale : (1 / Real.sqrt (modelDim d)) ≤ (invSqrtUpperBound headDimNat : ℝ))
    (hInputBound :
      ∀ pos d_in, |(D.layers i).state.input pos d_in| ≤ (l.ln1OutMaxAbsBound : ℝ))
    (hKeys :
      ∀ pos d', (D.layers i).state.keys pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_K.w d_in d')
    (hQueries :
      ∀ pos d', (D.layers i).state.queries pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_Q.w d_in d')
    (hValues :
      ∀ pos d', (D.layers i).state.values pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_V.w d_in d')
    (hWQ :
      SignedMixer.operatorNormBound (D.layers i).layer.W_Q ≤ (l.wqOpBoundMax : ℝ))
    (hWK :
      SignedMixer.operatorNormBound (D.layers i).layer.W_K ≤ (l.wkOpBoundMax : ℝ))
    (hVO :
      SignedMixer.operatorNormBound
          ((D.layers i).layer.W_V.comp (D.layers i).layer.W_O) ≤
        (l.attnValueCoeff : ℝ))
    (hConsistent :
      ∀ q, (D.layers i).state.attentionWeights q =
        softmax ((D.layers i).state.scores q))
    (hSoftmax :
      ∀ q, SignedMixer.operatorNormBound
        (softmaxJacobian ((D.layers i).state.scores q)) ≤
          (l.softmaxJacobianNormInfUpperBound : ℝ))
    (hEq : patternTerm (D.layers i) = patternTermExplicit (D.layers i)) :
    SignedMixer.operatorNormBound (patternTerm (D.layers i)) ≤
      (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ) := by
  classical
  let L := D.layers i
  let B : ℝ := (modelDimNat : ℝ) * (l.ln1OutMaxAbsBound : ℝ)
  let wq : ℝ := (l.wqOpBoundMax : ℝ)
  let wk : ℝ := (l.wkOpBoundMax : ℝ)
  let J : ℝ := (l.softmaxJacobianNormInfUpperBound : ℝ)
  let S : ℝ :=
    (Fintype.card n : ℝ) *
      ((1 / Real.sqrt (modelDim d)) * (2 * B * wq * wk))
  let V : ℝ := B * (l.attnValueCoeff : ℝ)
  have hSeqLen' : (Fintype.card n : ℝ) = (seqLenNat : ℝ) := by
    exact_mod_cast hSeqLen.symm
  have hModelDim' : (Fintype.card d : ℝ) = (modelDimNat : ℝ) := by
    exact_mod_cast hModelDim.symm
  have hB_nonneg : 0 ≤ B := by
    have hln1_nonneg : 0 ≤ (l.ln1OutMaxAbsBound : ℝ) := by
      rcases (inferInstance : Nonempty n) with ⟨pos⟩
      rcases (inferInstance : Nonempty d) with ⟨d_in⟩
      have h := hInputBound pos d_in
      exact le_trans (abs_nonneg _) h
    have hdim_nonneg : 0 ≤ (modelDimNat : ℝ) := by
      exact_mod_cast (Nat.zero_le _)
    exact mul_nonneg hdim_nonneg hln1_nonneg
  have hWQ_nonneg : 0 ≤ wq := by
    exact le_trans
      (SignedMixer.operatorNormBound_nonneg (M := L.layer.W_Q)) hWQ
  have hWK_nonneg : 0 ≤ wk := by
    exact le_trans
      (SignedMixer.operatorNormBound_nonneg (M := L.layer.W_K)) hWK
  have hInput :
      ∀ pos, ∑ d', |L.state.input pos d'| ≤ B := by
    intro pos
    have hsum :
        ∑ d', |L.state.input pos d'| ≤
          ∑ d' : d, (l.ln1OutMaxAbsBound : ℝ) := by
      refine Finset.sum_le_sum ?_
      intro d' _hd
      exact hInputBound pos d'
    have hconst :
        (∑ d' : d, (l.ln1OutMaxAbsBound : ℝ)) = B := by
      simp [B, hModelDim']
    exact le_trans hsum (by simp [hconst])
  have hScore :
      ∀ i' d_in q, ∑ k, |scoreGradient L q k i' d_in| ≤ S := by
    intro i' d_in q
    have h :=
      scoreGradient_sum_le_of_input (L := L) (q := q) (i := i') (d_in := d_in)
        (B := B) (wq := wq) (wk := wk)
        hInput (hKeys := hKeys) (hQueries := hQueries) hWQ hWK
    simpa [S, wq, wk] using h
  have hValue :
      SignedMixer.operatorNormBound (valueOutputMixer L) ≤ V := by
    have h :=
      valueOutputMixer_operatorNormBound_le_of_input (L := L) (B := B)
        (V := (l.attnValueCoeff : ℝ)) hInput (hValues := hValues) hVO
    simpa [V] using h
  have hPattern :=
    patternTerm_operatorNormBound_le_of_softmax (L := L) (J := J) (S := S) (V := V)
      (hConsistent := hConsistent) (hSoftmax := hSoftmax)
      (hScore := hScore) (hValue := hValue) hEq
  have hJ_nonneg : 0 ≤ J := by
    rcases (inferInstance : Nonempty n) with ⟨q⟩
    exact le_trans
      (SignedMixer.operatorNormBound_nonneg
        (M := softmaxJacobian (L.state.scores q))) (hSoftmax q)
  have hS_nonneg : 0 ≤ 2 * B * wq * wk := by
    have h2 : 0 ≤ (2 : ℝ) := by norm_num
    exact mul_nonneg (mul_nonneg (mul_nonneg h2 hB_nonneg) hWQ_nonneg) hWK_nonneg
  let S_bound : ℝ :=
    (seqLenNat : ℝ) * ((invSqrtUpperBound headDimNat : ℝ) * (2 * B * wq * wk))
  have hS_le : S ≤ S_bound := by
    have hSeq_nonneg : 0 ≤ (seqLenNat : ℝ) := by
      exact_mod_cast (Nat.zero_le _)
    calc
      S = (seqLenNat : ℝ) *
          ((1 / Real.sqrt (modelDim d)) * (2 * B * wq * wk)) := by
            simp [S, hSeqLen']
      _ ≤ (seqLenNat : ℝ) *
          ((invSqrtUpperBound headDimNat : ℝ) * (2 * B * wq * wk)) := by
            exact mul_le_mul_of_nonneg_left
              (mul_le_mul_of_nonneg_right hScale hS_nonneg) hSeq_nonneg
      _ = S_bound := rfl
  have hValue_nonneg : 0 ≤ V := by
    have hVal_nonneg : 0 ≤ (l.attnValueCoeff : ℝ) := by
      exact le_trans
        (SignedMixer.operatorNormBound_nonneg
          (M := L.layer.W_V.comp L.layer.W_O)) hVO
    exact mul_nonneg hB_nonneg hVal_nonneg
  have hSV_le :
      (Fintype.card n : ℝ) * S * V ≤ (seqLenNat : ℝ) * S_bound * V := by
    have hSeq_nonneg : 0 ≤ (seqLenNat : ℝ) := by
      exact_mod_cast (Nat.zero_le _)
    have hS_le' : (seqLenNat : ℝ) * S ≤ (seqLenNat : ℝ) * S_bound := by
      exact mul_le_mul_of_nonneg_left hS_le hSeq_nonneg
    calc
      (Fintype.card n : ℝ) * S * V = (seqLenNat : ℝ) * S * V := by
        simp [hSeqLen']
      _ ≤ (seqLenNat : ℝ) * S_bound * V := by
        exact mul_le_mul_of_nonneg_right hS_le' hValue_nonneg
  have hPat :
      (l.attnPatternCoeff : ℝ) =
        (attnPatternCoeffBound seqLenNat modelDimNat headDimNat
          l.ln1OutMaxAbsBound l.wqOpBoundMax l.wkOpBoundMax l.attnValueCoeff : ℝ) := by
    rcases hValid with ⟨_hln1, _hln2, _hln1Out, _hProbLo, _hProbHi,
      _hsoftmax, hpat, _hattn, _hmlpCoeff, _hmlp, _hC⟩
    exact congrArg (fun x : Rat => (x : ℝ)) hpat
  have hCoeff_eq :
      (seqLenNat : ℝ) * S_bound * V =
        (attnPatternCoeffBound seqLenNat modelDimNat headDimNat
          l.ln1OutMaxAbsBound l.wqOpBoundMax l.wkOpBoundMax l.attnValueCoeff : ℝ) := by
    simp [S_bound, V, B, wq, wk, attnPatternCoeffBound_def, attnScoreGradBound_def,
      Rat.cast_mul, Rat.cast_natCast, Rat.cast_ofNat, mul_assoc, mul_left_comm,
      mul_comm, -mul_eq_mul_left_iff, -mul_eq_mul_right_iff]
  have hCoeff :
      (Fintype.card n : ℝ) * S * V ≤ (l.attnPatternCoeff : ℝ) := by
    calc
      (Fintype.card n : ℝ) * S * V ≤ (seqLenNat : ℝ) * S_bound * V := hSV_le
      _ = (attnPatternCoeffBound seqLenNat modelDimNat headDimNat
            l.ln1OutMaxAbsBound l.wqOpBoundMax l.wkOpBoundMax l.attnValueCoeff : ℝ) := hCoeff_eq
      _ = (l.attnPatternCoeff : ℝ) := by simp [hPat]
  have hCoeffMul' :
      J * ((Fintype.card n : ℝ) * S * V) ≤ J * (l.attnPatternCoeff : ℝ) :=
    mul_le_mul_of_nonneg_left hCoeff hJ_nonneg
  have hCoeffMul :
      (Fintype.card n : ℝ) * J * S * V ≤ J * (l.attnPatternCoeff : ℝ) := by
    have hRearrange :
        (Fintype.card n : ℝ) * J * S * V = J * ((Fintype.card n : ℝ) * S * V) := by
      ring
    simpa [hRearrange] using hCoeffMul'
  exact le_trans hPattern hCoeffMul

/-- Full attention Jacobian bound from certificate validity and attention-state assumptions. -/
theorem attn_fullJacobian_bound_of_cert
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLenNat modelDimNat headDimNat : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLenNat modelDimNat headDimNat)
    (hSeqLen : seqLenNat = Fintype.card n)
    (hModelDim : modelDimNat = Fintype.card d)
    (hScale : (1 / Real.sqrt (modelDim d)) ≤ (invSqrtUpperBound headDimNat : ℝ))
    (hInputBound :
      ∀ pos d_in, |(D.layers i).state.input pos d_in| ≤ (l.ln1OutMaxAbsBound : ℝ))
    (hKeys :
      ∀ pos d', (D.layers i).state.keys pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_K.w d_in d')
    (hQueries :
      ∀ pos d', (D.layers i).state.queries pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_Q.w d_in d')
    (hValues :
      ∀ pos d', (D.layers i).state.values pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_V.w d_in d')
    (hWQ :
      SignedMixer.operatorNormBound (D.layers i).layer.W_Q ≤ (l.wqOpBoundMax : ℝ))
    (hWK :
      SignedMixer.operatorNormBound (D.layers i).layer.W_K ≤ (l.wkOpBoundMax : ℝ))
    (hVO :
      SignedMixer.operatorNormBound
          ((D.layers i).layer.W_V.comp (D.layers i).layer.W_O) ≤
        (l.attnValueCoeff : ℝ))
    (hConsistent :
      ∀ q, (D.layers i).state.attentionWeights q =
        softmax ((D.layers i).state.scores q))
    (hSoftmax :
      ∀ q, SignedMixer.operatorNormBound
        (softmaxJacobian ((D.layers i).state.scores q)) ≤
          (l.softmaxJacobianNormInfUpperBound : ℝ))
    (hEq : patternTerm (D.layers i) = patternTermExplicit (D.layers i)) :
    SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
      (seqLenNat : ℝ) * (l.attnValueCoeff : ℝ) +
        (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ) := by
  classical
  let L := D.layers i
  have hPattern :=
    attn_pattern_bound_of_cert
      (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
      (seqLenNat := seqLenNat) (modelDimNat := modelDimNat) (headDimNat := headDimNat)
      hValid hSeqLen hModelDim hScale hInputBound hKeys hQueries hValues hWQ hWK hVO
      hConsistent hSoftmax hEq
  have hValueBase :
      SignedMixer.operatorNormBound (valueTerm L) ≤
        (Fintype.card n : ℝ) * (l.attnValueCoeff : ℝ) := by
    simpa using
      (valueTerm_operatorNormBound_le_card (L := L) (B := (l.attnValueCoeff : ℝ))
        (hConsistent := hConsistent) (hVO := hVO))
  have hSeqLen' : (Fintype.card n : ℝ) = (seqLenNat : ℝ) := by
    exact_mod_cast hSeqLen.symm
  have hValue :
      SignedMixer.operatorNormBound (valueTerm L) ≤
        (seqLenNat : ℝ) * (l.attnValueCoeff : ℝ) := by
    simpa [hSeqLen'] using hValueBase
  simpa using
    (attention_fullJacobian_bound_of_terms (L := L) (hValue := hValue) (hPattern := hPattern))

/-- MLP coefficient bound from certificate validity and component bounds. -/
theorem mlp_coeff_bound_of_valid
    {n d : Type*} [Fintype n] [Fintype d] [Nonempty n] [Nonempty d]
    (F : MLPFactorization (n := n) (d := d))
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLen modelDim headDim)
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
    (eps := eps) (sqrtPrecBits := sqrtPrecBits)
    (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim) (l := l) hValid
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
    (seqLen modelDim headDim : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLen modelDim headDim)
    (hLn1 :
      SignedMixer.operatorNormBound (D.ln1Jacobians i) ≤ (l.ln1Bound : ℝ))
    (hLn1_nonneg : 0 ≤ (l.ln1Bound : ℝ))
    (hFull :
      SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (seqLen : ℝ) * (l.attnValueCoeff : ℝ) +
          (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ)) :
    SignedMixer.operatorNormBound
        ((D.ln1Jacobians i).comp (D.layers i).fullJacobian)
        ≤ (l.attnJacBound : ℝ) := by
  classical
  let attnTotal : ℝ :=
    (seqLen : ℝ) * (l.attnValueCoeff : ℝ) +
      (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ)
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
        (l.ln1Bound : ℝ) * attnTotal := by
    simpa [attnTotal] using
      (mul_le_mul_of_nonneg_left hFull hLn1_nonneg)
  have hmul :
      SignedMixer.operatorNormBound (D.ln1Jacobians i) *
          SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (l.ln1Bound : ℝ) * attnTotal := by
    exact le_trans hmul1 hmul2
  have hAttn :
      (l.ln1Bound : ℝ) * attnTotal =
        (l.attnJacBound : ℝ) := by
    have hAttn :=
      LayerAmplificationCert.attnJacBound_eq_cast_of_valid
        (eps := eps) (sqrtPrecBits := sqrtPrecBits)
        (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim) (l := l) hValid
    simpa [attnTotal, mul_assoc, add_assoc] using hAttn.symm
  calc
    SignedMixer.operatorNormBound
        ((D.ln1Jacobians i).comp (D.layers i).fullJacobian)
        ≤ SignedMixer.operatorNormBound (D.ln1Jacobians i) *
          SignedMixer.operatorNormBound (D.layers i).fullJacobian := hcomp
    _ ≤ (l.ln1Bound : ℝ) * attnTotal := hmul
    _ = (l.attnJacBound : ℝ) := hAttn

/-- MLP-component bound from certificate identities and component bounds. -/
theorem mlp_component_bound_of_cert
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLen modelDim headDim)
    (hLn2 :
      SignedMixer.operatorNormBound (D.ln2Jacobians i) ≤ (l.ln2Bound : ℝ))
    (hLn2_nonneg : 0 ≤ (l.ln2Bound : ℝ))
    (hMlp :
      SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ)) :
    SignedMixer.operatorNormBound
        ((D.ln2Jacobians i).comp (D.mlpJacobians i))
        ≤ (l.mlpJacBound : ℝ) := by
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
      (l.ln2Bound : ℝ) * mlpBound = (l.mlpJacBound : ℝ) := by
    have hMlpEq :=
      LayerAmplificationCert.mlpJacBound_eq_cast_of_valid
        (eps := eps) (sqrtPrecBits := sqrtPrecBits)
        (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim) (l := l) hValid
    simpa [mlpBound, mul_assoc] using hMlpEq.symm
  calc
    SignedMixer.operatorNormBound
        ((D.ln2Jacobians i).comp (D.mlpJacobians i))
        ≤ SignedMixer.operatorNormBound (D.ln2Jacobians i) *
          SignedMixer.operatorNormBound (D.mlpJacobians i) := hcomp
    _ ≤ (l.ln2Bound : ℝ) * mlpBound := hmul
    _ = (l.mlpJacBound : ℝ) := hMlpEq

/-- Layer Jacobian residual bound from a layer amplification certificate. -/
theorem layerJacobian_residual_bound_of_cert
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLen modelDim headDim)
    (hA :
      SignedMixer.operatorNormBound
        ((D.ln1Jacobians i).comp (D.layers i).fullJacobian)
        ≤ (l.attnJacBound : ℝ))
    (hM :
      SignedMixer.operatorNormBound
        ((D.ln2Jacobians i).comp (D.mlpJacobians i))
        ≤ (l.mlpJacBound : ℝ)) :
    SignedMixer.operatorNormBound
        (D.layerJacobian i - SignedMixer.identity) ≤ (l.C : ℝ) := by
  have hC := LayerAmplificationCert.c_eq_cast_of_valid
    (eps := eps) (sqrtPrecBits := sqrtPrecBits)
    (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim) (l := l) hValid
  have hres :=
    DeepLinearization.layerJacobian_residual_bound
      (D := D) (i := i)
      (A := (l.attnJacBound : ℝ))
      (M := (l.mlpJacBound : ℝ)) hA hM
  simpa [hC] using hres

/-- Layer Jacobian residual bound from a certificate, with component-norm assumptions. -/
theorem layerJacobian_residual_bound_of_cert_components
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLen modelDim headDim)
    (hLn1 :
      SignedMixer.operatorNormBound (D.ln1Jacobians i) ≤ (l.ln1Bound : ℝ))
    (hLn1_nonneg : 0 ≤ (l.ln1Bound : ℝ))
    (hFull :
      SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (seqLen : ℝ) * (l.attnValueCoeff : ℝ) +
          (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ))
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
      (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim)
      hValid hLn1 hLn1_nonneg hFull
  have hM :=
    mlp_component_bound_of_cert
      (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
      (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim)
      hValid hLn2 hLn2_nonneg hMlp
  exact layerJacobian_residual_bound_of_cert
    (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
    (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim)
    hValid hA hM

/-- Layer Jacobian residual bound from certificate validity and attention-state assumptions. -/
theorem layerJacobian_residual_bound_of_cert_from_state
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLenNat modelDimNat headDimNat : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLenNat modelDimNat headDimNat)
    (hSeqLen : seqLenNat = Fintype.card n)
    (hModelDim : modelDimNat = Fintype.card d)
    (hScale : (1 / Real.sqrt (modelDim d)) ≤ (invSqrtUpperBound headDimNat : ℝ))
    (hInputBound :
      ∀ pos d_in, |(D.layers i).state.input pos d_in| ≤ (l.ln1OutMaxAbsBound : ℝ))
    (hKeys :
      ∀ pos d', (D.layers i).state.keys pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_K.w d_in d')
    (hQueries :
      ∀ pos d', (D.layers i).state.queries pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_Q.w d_in d')
    (hValues :
      ∀ pos d', (D.layers i).state.values pos d' =
        ∑ d_in, (D.layers i).state.input pos d_in *
          (D.layers i).layer.W_V.w d_in d')
    (hWQ :
      SignedMixer.operatorNormBound (D.layers i).layer.W_Q ≤ (l.wqOpBoundMax : ℝ))
    (hWK :
      SignedMixer.operatorNormBound (D.layers i).layer.W_K ≤ (l.wkOpBoundMax : ℝ))
    (hVO :
      SignedMixer.operatorNormBound
          ((D.layers i).layer.W_V.comp (D.layers i).layer.W_O) ≤
        (l.attnValueCoeff : ℝ))
    (hConsistent :
      ∀ q, (D.layers i).state.attentionWeights q =
        softmax ((D.layers i).state.scores q))
    (hSoftmax :
      ∀ q, SignedMixer.operatorNormBound
        (softmaxJacobian ((D.layers i).state.scores q)) ≤
          (l.softmaxJacobianNormInfUpperBound : ℝ))
    (hEq : patternTerm (D.layers i) = patternTermExplicit (D.layers i))
    (hLn1 :
      SignedMixer.operatorNormBound (D.ln1Jacobians i) ≤ (l.ln1Bound : ℝ))
    (hLn1_nonneg : 0 ≤ (l.ln1Bound : ℝ))
    (hLn2 :
      SignedMixer.operatorNormBound (D.ln2Jacobians i) ≤ (l.ln2Bound : ℝ))
    (hLn2_nonneg : 0 ≤ (l.ln2Bound : ℝ))
    (hMlp :
      SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ)) :
    SignedMixer.operatorNormBound
        (D.layerJacobian i - SignedMixer.identity) ≤ (l.C : ℝ) := by
  have hFull :=
    attn_fullJacobian_bound_of_cert
      (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
      (seqLenNat := seqLenNat) (modelDimNat := modelDimNat) (headDimNat := headDimNat)
      hValid hSeqLen hModelDim hScale hInputBound hKeys hQueries hValues hWQ hWK hVO
      hConsistent hSoftmax hEq
  have hA :=
    attn_component_bound_of_cert
      (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
      (seqLen := seqLenNat) (modelDim := modelDimNat) (headDim := headDimNat)
      hValid hLn1 hLn1_nonneg hFull
  have hM :=
    mlp_component_bound_of_cert
      (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
      (seqLen := seqLenNat) (modelDim := modelDimNat) (headDim := headDimNat)
      hValid hLn2 hLn2_nonneg hMlp
  exact layerJacobian_residual_bound_of_cert
    (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
    (seqLen := seqLenNat) (modelDim := modelDimNat) (headDim := headDimNat)
    hValid hA hM

/-- Layer Jacobian residual bound from a certificate plus component-norm assumptions. -/
theorem layerJacobian_residual_bound_of_cert_assuming
    {n d : Type*} [Fintype n] [Fintype d] [DecidableEq n] [DecidableEq d]
    [Nonempty n] [Nonempty d]
    (D : DeepLinearization (n := n) (d := d))
    (i : Fin D.numLayers)
    (l : LayerAmplificationCert) (eps : Rat) (sqrtPrecBits : Nat)
    (seqLen modelDim headDim : Nat)
    (hValid : l.Valid eps sqrtPrecBits seqLen modelDim headDim)
    (hSeqLen : seqLen = Fintype.card n)
    (h : LayerComponentNormAssumptions (D := D) (i := i) (l := l)) :
    SignedMixer.operatorNormBound
        (D.layerJacobian i - SignedMixer.identity) ≤ (l.C : ℝ) := by
  have hCoeff :
      SignedMixer.operatorNormBound (D.mlpFactors i).win *
        SignedMixer.operatorNormBound (D.mlpFactors i).wout ≤ (l.mlpCoeff : ℝ) :=
    mlp_coeff_bound_of_valid
      (F := D.mlpFactors i) (l := l)
      (eps := eps) (sqrtPrecBits := sqrtPrecBits)
      (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim) hValid
      h.mlpWinBound h.mlpWinBound_nonneg h.mlpWoutBound
  have hMlp :
      SignedMixer.operatorNormBound (D.mlpJacobians i) ≤
        (l.mlpCoeff : ℝ) * (l.mlpActDerivBound : ℝ) := by
    simpa using
      (mlp_bound_of_factorization (F := D.mlpFactors i) (l := l)
        h.mlpDerivBound hCoeff)
  have hFull :
      SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
        (seqLen : ℝ) * (l.attnValueCoeff : ℝ) +
          (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ) := by
    have hFullBase :
        SignedMixer.operatorNormBound (D.layers i).fullJacobian ≤
          (Fintype.card n : ℝ) * (l.attnValueCoeff : ℝ) +
            (l.softmaxJacobianNormInfUpperBound : ℝ) * (l.attnPatternCoeff : ℝ) := by
      simpa using
        (attention_fullJacobian_bound_of_terms (L := D.layers i)
          (hValue := h.attnValueBound) (hPattern := h.attnPatternBound))
    have hSeqLen' : (Fintype.card n : ℝ) = (seqLen : ℝ) := by
      exact_mod_cast hSeqLen.symm
    simpa [hSeqLen'] using hFullBase
  exact layerJacobian_residual_bound_of_cert_components
    (D := D) (i := i) (l := l) (eps := eps) (sqrtPrecBits := sqrtPrecBits)
    (seqLen := seqLen) (modelDim := modelDim) (headDim := headDim)
    hValid h.ln1Bound h.ln1Bound_nonneg hFull
    h.ln2Bound h.ln2Bound_nonneg hMlp

end Nfp.Sound
