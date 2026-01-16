-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Aesop
public import Nfp.Sound.Induction.CoreSound

/-!
Head-output interval certificates for induction heads.
-/

public section

namespace Nfp

namespace Sound

open scoped BigOperators

open Nfp.Circuit
open Nfp.Sound.Bounds

variable {seq : Nat}

/-- Build and certify induction certificates from exact head inputs. -/
def buildInductionCertFromHeadWith? [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option {c : InductionHeadCert seq // InductionHeadCertSound inputs c} := by
  classical
  cases hcore : buildInductionCertFromHeadCoreWith? cfg inputs with
  | none => exact none
  | some c =>
      exact some ⟨c, buildInductionCertFromHeadCoreWith?_sound (cfg := cfg) inputs c hcore⟩

/-- Build and certify induction certificates from exact head inputs, retaining the core cache. -/
def buildInductionCertFromHeadWithCache? [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option {cache : InductionHeadCoreCache seq dModel dHead //
      InductionHeadCertSound inputs cache.cert} := by
  classical
  by_cases hEps : 0 < inputs.lnEps
  · by_cases hSqrt : 0 < sqrtLower inputs.lnEps
    · by_cases hmodel : dModel = 0
      · exact none
      · by_cases hactive : inputs.active.Nonempty
        · let cache := buildInductionHeadCoreCacheWith cfg inputs
          have hmodel' : dModel ≠ 0 := by
            exact hmodel
          have hcore :
              buildInductionCertFromHeadCoreWith? cfg inputs = some cache.cert := by
            simpa [cache] using
              (buildInductionCertFromHeadCoreWith?_eq_some
                (cfg := cfg) (inputs := inputs) hEps hSqrt hmodel' hactive)
          exact some ⟨cache,
            buildInductionCertFromHeadCoreWith?_sound (cfg := cfg) inputs cache.cert hcore⟩
        · exact none
    · exact none
  · exact none

/-- Build and certify induction certificates from exact head inputs using the default split
budgets. -/
def buildInductionCertFromHead? [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option {c : InductionHeadCert seq // InductionHeadCertSound inputs c} :=
  buildInductionCertFromHeadWith? defaultInductionHeadSplitConfig inputs

section HeadOutputInterval

variable {seq dModel dHead : Nat}

noncomputable section

/-- Real-valued head output using explicit score inputs. -/
def headOutputWithScores (scores : Fin seq → Fin seq → Real)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scores q) k
  let vals : Fin seq → Real := fun k => headValueRealOfInputs inputs k i
  dotProduct (weights q) vals

/-- Unfolding lemma for `headOutputWithScores`. -/
theorem headOutputWithScores_def (scores : Fin seq → Fin seq → Real)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutputWithScores scores inputs q i =
      let weights : Fin seq → Fin seq → Real := fun q k =>
        Circuit.softmax (scores q) k
      let vals : Fin seq → Real := fun k => headValueRealOfInputs inputs k i
      dotProduct (weights q) vals := by
  simp [headOutputWithScores]

/-- Real-valued head output for a query and model dimension. -/
def headOutput (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  headOutputWithScores (scoresRealOfInputs inputs) inputs q i

/-- Unfolding lemma for `headOutput`. -/
theorem headOutput_def (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutput inputs q i =
      headOutputWithScores (scoresRealOfInputs inputs) inputs q i := by
  simp [headOutput]

/-- Soundness predicate for head-output interval bounds. -/
structure HeadOutputIntervalSound [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (active : Finset (Fin seq))
    (c : Circuit.ResidualIntervalCert dModel) : Prop where
  /-- Interval bounds are ordered coordinatewise. -/
  bounds : Circuit.ResidualIntervalBounds c
  /-- Active-query outputs lie inside the interval bounds. -/
  output_mem :
    ∀ q, q ∈ active → ∀ i,
      (c.lo i : Real) ≤ headOutput inputs q i ∧
        headOutput inputs q i ≤ (c.hi i : Real)

/-- Certified head-output interval data for a specific active set. -/
structure HeadOutputIntervalResult [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) where
  /-- Active queries covered by the interval bounds. -/
  active : Finset (Fin seq)
  /-- Residual-interval certificate for head outputs. -/
  cert : Circuit.ResidualIntervalCert dModel
  /-- Soundness proof for the interval bounds. -/
  sound : HeadOutputIntervalSound inputs active cert

/-- Build residual-interval bounds for head outputs on active queries. -/
def buildHeadOutputIntervalFromHead? [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (HeadOutputIntervalResult inputs) := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      by_cases hEps : 0 < inputs.lnEps
      · by_cases hSqrt : 0 < sqrtLower inputs.lnEps
        · by_cases hmodel : dModel = 0
          · exact none
          · cases hbuild : buildInductionCertFromHead? inputs with
            | none => exact none
            | some certWithProof =>
                rcases certWithProof with ⟨cert, hcert⟩
                let lnBounds : Fin (Nat.succ n) → (Fin dModel → Rat) × (Fin dModel → Rat) :=
                  fun q =>
                    Bounds.layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta
                      (inputs.embed q)
                let lnLo : Fin (Nat.succ n) → Fin dModel → Rat := fun q => (lnBounds q).1
                let lnHi : Fin (Nat.succ n) → Fin dModel → Rat := fun q => (lnBounds q).2
                let vLo : Fin (Nat.succ n) → Fin dHead → Rat := fun q d =>
                  dotIntervalLower (fun j => inputs.wv j d) (lnLo q) (lnHi q) + inputs.bv d
                let vHi : Fin (Nat.succ n) → Fin dHead → Rat := fun q d =>
                  dotIntervalUpper (fun j => inputs.wv j d) (lnLo q) (lnHi q) + inputs.bv d
                let headValueLo : Fin (Nat.succ n) → Fin dModel → Rat := fun k i =>
                  dotIntervalLower (fun d => inputs.wo i d) (vLo k) (vHi k)
                let headValueHi : Fin (Nat.succ n) → Fin dModel → Rat := fun k i =>
                  dotIntervalUpper (fun d => inputs.wo i d) (vLo k) (vHi k)
                have hln_bounds :
                    ∀ q i, (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
                      lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
                  intro q i
                  have hln :=
                    Bounds.layerNormBounds_spec (eps := inputs.lnEps)
                      (gamma := inputs.ln1Gamma) (beta := inputs.ln1Beta)
                      (x := inputs.embed q) hmodel hEps hSqrt
                  simpa [lnBounds, lnLo, lnHi, lnRealOfInputs_def] using hln i
                have hv_bounds :
                    ∀ q d, (vLo q d : Real) ≤ vRealOfInputs inputs q d ∧
                      vRealOfInputs inputs q d ≤ (vHi q d : Real) := by
                  intro q d
                  have hln := hln_bounds q
                  have hlo : ∀ j, (lnLo q j : Real) ≤ lnRealOfInputs inputs q j := fun j =>
                    (hln j).1
                  have hhi : ∀ j, lnRealOfInputs inputs q j ≤ (lnHi q j : Real) := fun j =>
                    (hln j).2
                  have hlow' :
                      dotIntervalLower (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
                          (inputs.bv d : Real) ≤
                        dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs q) +
                          (inputs.bv d : Real) :=
                    by
                      simpa using
                        dotIntervalLower_le_dotProduct_real_add
                          (v := fun j => inputs.wv j d)
                          (lo := lnLo q) (hi := lnHi q)
                          (x := lnRealOfInputs inputs q) (b := (inputs.bv d : Real)) hlo hhi
                  have hhigh' :
                      dotProduct (fun j => (inputs.wv j d : Real)) (lnRealOfInputs inputs q) +
                          (inputs.bv d : Real) ≤
                        dotIntervalUpper (fun j => inputs.wv j d) (lnLo q) (lnHi q) +
                          (inputs.bv d : Real) :=
                    by
                      simpa using
                        dotProduct_le_dotIntervalUpper_real_add
                          (v := fun j => inputs.wv j d)
                          (lo := lnLo q) (hi := lnHi q)
                          (x := lnRealOfInputs inputs q) (b := (inputs.bv d : Real)) hlo hhi
                  constructor
                  · simpa [vLo, vRealOfInputs_def, Bounds.cacheBound2_apply,
                      Bounds.dotIntervalLowerCachedRat_eq, ratToReal_add] using
                      hlow'
                  · simpa [vHi, vRealOfInputs_def, Bounds.cacheBound2_apply,
                      Bounds.dotIntervalUpperCachedRat_eq, ratToReal_add] using
                      hhigh'
                have hhead_bounds :
                    ∀ k i, (headValueLo k i : Real) ≤ headValueRealOfInputs inputs k i ∧
                      headValueRealOfInputs inputs k i ≤ (headValueHi k i : Real) := by
                  intro k i
                  have hv := hv_bounds k
                  have hlo : ∀ d, (vLo k d : Real) ≤ vRealOfInputs inputs k d := fun d => (hv d).1
                  have hhi : ∀ d, vRealOfInputs inputs k d ≤ (vHi k d : Real) := fun d => (hv d).2
                  have hlow :=
                    dotIntervalLower_le_dotProduct_real (v := fun d => inputs.wo i d)
                      (lo := vLo k) (hi := vHi k)
                      (x := fun d => vRealOfInputs inputs k d) hlo hhi
                  have hhigh :=
                    dotProduct_le_dotIntervalUpper_real (v := fun d => inputs.wo i d)
                      (lo := vLo k) (hi := vHi k)
                      (x := fun d => vRealOfInputs inputs k d) hlo hhi
                  constructor
                  · simpa [headValueLo, headValueRealOfInputs_def] using hlow
                  · simpa [headValueHi, headValueRealOfInputs_def] using hhigh
                let scoresReal : Fin (Nat.succ n) → Fin (Nat.succ n) → Real :=
                  scoresRealOfInputs inputs
                let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := fun q k =>
                  Circuit.softmax (scoresReal q) k
                let activeSet : Finset (Fin (Nat.succ n)) := cert.active
                let univ : Finset (Fin (Nat.succ n)) := Finset.univ
                have huniv : univ.Nonempty := by simp [univ]
                let loVal : Fin dModel → Rat := fun i =>
                  univ.inf' huniv (fun k => headValueLo k i)
                let hiVal : Fin dModel → Rat := fun i =>
                  univ.sup' huniv (fun k => headValueHi k i)
                have hvalsBoundsReal :
                    ∀ i, Layers.ValueRangeBounds (Val := Real)
                      (loVal i : Real) (hiVal i : Real)
                      (fun k => headValueRealOfInputs inputs k i) := by
                  intro i
                  have hloVal : ∀ k, loVal i ≤ headValueLo k i := by
                    intro k
                    dsimp [loVal]
                    refine (Finset.inf'_le_iff (s := univ) (H := huniv)
                      (f := fun k => headValueLo k i) (a := headValueLo k i)).2 ?_
                    exact ⟨k, by simp [univ], le_rfl⟩
                  have hhiVal : ∀ k, headValueHi k i ≤ hiVal i := by
                    intro k
                    dsimp [hiVal]
                    refine (Finset.le_sup'_iff (s := univ) (H := huniv)
                      (f := fun k => headValueHi k i) (a := headValueHi k i)).2 ?_
                    exact ⟨k, ⟨by simp [univ], le_rfl⟩⟩
                  refine { lo_le_hi := ?_, lo_le := ?_, le_hi := ?_ }
                  · rcases (Finset.univ_nonempty : univ.Nonempty) with ⟨k0, hk0⟩
                    have hloRat : loVal i ≤ headValueLo k0 i := hloVal k0
                    have hhiRat : headValueHi k0 i ≤ hiVal i := hhiVal k0
                    have hbounds := hhead_bounds k0 i
                    have hloReal : (loVal i : Real) ≤ (headValueLo k0 i : Real) := by
                      simpa [ratToReal_def] using ratToReal_le_of_le hloRat
                    have hhiReal : (headValueHi k0 i : Real) ≤ (hiVal i : Real) := by
                      simpa [ratToReal_def] using ratToReal_le_of_le hhiRat
                    have hreal : (loVal i : Real) ≤ (hiVal i : Real) := by
                      exact le_trans hloReal (le_trans hbounds.1 (le_trans hbounds.2 hhiReal))
                    exact hreal
                  · intro k
                    have hloRat : loVal i ≤ headValueLo k i := hloVal k
                    have hbounds := hhead_bounds k i
                    have hloReal : (loVal i : Real) ≤ (headValueLo k i : Real) := by
                      simpa [ratToReal_def] using ratToReal_le_of_le hloRat
                    exact hloReal.trans hbounds.1
                  · intro k
                    have hhiRat : headValueHi k i ≤ hiVal i := hhiVal k
                    have hbounds := hhead_bounds k i
                    have hhiReal : (headValueHi k i : Real) ≤ (hiVal i : Real) := by
                      simpa [ratToReal_def] using ratToReal_le_of_le hhiRat
                    exact hbounds.2.trans hhiReal
                have hsoftmax :
                    Layers.SoftmaxMarginBoundsOn (Val := Real)
                      (cert.eps : Real) (cert.margin : Real)
                      (fun q => q ∈ activeSet) cert.prev scoresReal weights := by
                  simpa [scoresReal, weights, activeSet] using hcert.softmax_bounds
                have hweights :
                    Layers.OneHotApproxBoundsOnActive (Val := Real) (cert.eps : Real)
                      (fun q => q ∈ activeSet) cert.prev weights :=
                  Layers.oneHotApproxBoundsOnActive_of_softmaxMargin
                    (Val := Real)
                    (ε := (cert.eps : Real))
                    (margin := (cert.margin : Real))
                    (active := fun q => q ∈ activeSet)
                    (prev := cert.prev)
                    (scores := scoresReal)
                    (weights := weights)
                    hsoftmax
                have happrox :
                    ∀ i, Layers.InductionSpecApproxOn (Val := Real) (n := n)
                      ((cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)))
                      (fun q => q ∈ activeSet) cert.prev
                      (fun q => dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i))
                      (fun k => headValueRealOfInputs inputs k i) := by
                  intro i
                  exact
                    Layers.inductionSpecApproxOn_of_oneHotApprox_valueRange
                      (Val := Real)
                      (n := n)
                      (ε := (cert.eps : Real))
                      (lo := (loVal i : Real))
                      (hi := (hiVal i : Real))
                      (active := fun q => q ∈ activeSet)
                      (prev := cert.prev)
                      (weights := weights)
                      (vals := fun k => headValueRealOfInputs inputs k i)
                      (hweights := hweights)
                      (hvals := hvalsBoundsReal i)
                let delta : Fin dModel → Rat := fun i => hiVal i - loVal i
                let boundLoRat : Fin (Nat.succ n) → Fin dModel → Rat := fun q i =>
                  headValueLo (cert.prev q) i - cert.eps * delta i
                let boundHiRat : Fin (Nat.succ n) → Fin dModel → Rat := fun q i =>
                  headValueHi (cert.prev q) i + cert.eps * delta i
                let loOut : Fin dModel → Rat := fun i =>
                  if h : activeSet.Nonempty then
                    activeSet.inf' h (fun q => boundLoRat q i)
                  else
                    0
                let hiOut : Fin dModel → Rat := fun i =>
                  if h : activeSet.Nonempty then
                    activeSet.sup' h (fun q => boundHiRat q i)
                  else
                    0
                have hout :
                    ∀ q, q ∈ activeSet → ∀ i,
                      (loOut i : Real) ≤ headOutput inputs q i ∧
                        headOutput inputs q i ≤ (hiOut i : Real) := by
                  intro q hq i
                  have hactive : activeSet.Nonempty := ⟨q, hq⟩
                  have hspec :
                      dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) ≤
                          headValueRealOfInputs inputs (cert.prev q) i +
                            (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) ∧
                        headValueRealOfInputs inputs (cert.prev q) i ≤
                          dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) +
                            (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) := by
                    have happrox' :
                        ∀ q, q ∈ activeSet →
                          dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) ≤
                              headValueRealOfInputs inputs (cert.prev q) i +
                                (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) ∧
                            headValueRealOfInputs inputs (cert.prev q) i ≤
                              dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) +
                                (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) := by
                      simpa [Layers.InductionSpecApproxOn_def] using (happrox i)
                    exact happrox' q hq
                  have hout_def :
                      headOutput inputs q i =
                        dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) := by
                    simp [headOutput, headOutputWithScores, scoresReal, weights]
                  have hprev_bounds := hhead_bounds (cert.prev q) i
                  have hupper :
                      headOutput inputs q i ≤ (boundHiRat q i : Real) := by
                    have hupper' :
                        dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) ≤
                          headValueRealOfInputs inputs (cert.prev q) i +
                            (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) := by
                      exact hspec.1
                    have hupper'' :
                        dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) ≤
                          (headValueHi (cert.prev q) i : Real) +
                            (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) := by
                      have hprev_bounds' :=
                        (add_le_add_iff_right
                            ((cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)))).2
                          hprev_bounds.2
                      exact le_trans hupper' hprev_bounds'
                    simpa
                    [hout_def, boundHiRat, delta, ratToReal_add, ratToReal_mul,
                      ratToReal_sub] using
                      hupper''
                  have hlower :
                      (boundLoRat q i : Real) ≤ headOutput inputs q i := by
                    have hlower' :
                        (headValueRealOfInputs inputs (cert.prev q) i : Real) -
                            (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) ≤
                          dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) := by
                      exact (sub_le_iff_le_add).2 hspec.2
                    have hlower'' :
                        (headValueLo (cert.prev q) i : Real) -
                            (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) ≤
                          dotProduct (weights q) (fun k => headValueRealOfInputs inputs k i) := by
                      refine le_trans (sub_le_sub_right hprev_bounds.1
                        ((cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)))) ?_
                      exact hlower'
                    simpa [hout_def, boundLoRat, delta, ratToReal_mul, ratToReal_sub] using
                      hlower''
                  have hlo :
                      (loOut i : Real) ≤ (boundLoRat q i : Real) := by
                    have hloRat : loOut i ≤ boundLoRat q i := by
                      simpa [loOut, hactive] using
                        (Finset.inf'_le
                          (s := activeSet)
                          (f := fun q => boundLoRat q i)
                          (b := q) hq)
                    simpa [ratToReal_def] using ratToReal_le_of_le hloRat
                  have hhi :
                      (boundHiRat q i : Real) ≤ (hiOut i : Real) := by
                    have hhiRat : boundHiRat q i ≤ hiOut i := by
                      simpa [hiOut, hactive] using
                        (Finset.le_sup'
                          (s := activeSet)
                          (f := fun q => boundHiRat q i)
                          (b := q) hq)
                    simpa [ratToReal_def] using ratToReal_le_of_le hhiRat
                  exact ⟨le_trans hlo hlower, le_trans hupper hhi⟩
                have hbounds : Circuit.ResidualIntervalBounds { lo := loOut, hi := hiOut } := by
                  refine { lo_le_hi := ?_ }
                  intro i
                  by_cases hactive : activeSet.Nonempty
                  · rcases hactive with ⟨q, hq⟩
                    have hout_i := hout q hq i
                    have hreal : ratToReal (loOut i) ≤ ratToReal (hiOut i) := by
                      simpa [ratToReal_def] using le_trans hout_i.1 hout_i.2
                    exact (ratToReal_le_iff (x := loOut i) (y := hiOut i)).1 hreal
                  · simp [loOut, hiOut, hactive]
                let certOut : Circuit.ResidualIntervalCert dModel := { lo := loOut, hi := hiOut }
                exact some
                  { active := activeSet
                    cert := certOut
                    sound :=
                      { bounds := hbounds
                        output_mem := by
                          intro q hq i
                          exact hout q hq i } }
        · exact none
      · exact none

end

end HeadOutputInterval

end Sound

end Nfp
