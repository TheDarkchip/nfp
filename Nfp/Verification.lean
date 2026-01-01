-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Discovery

/-!
# Causal Verification (Head Ablation)

This module implements **causal verification** of candidate circuits using an executable
forward pass with **head ablation** (a.k.a. zero-ablation).

The core API is `verifyCircuit`, which:
- selects **energy-matched control heads** (same layer, closest output norm),
- checks runtime **axioms** (baseline competence, control independence, energy equivalence),
- and (only if axioms hold) measures **logit-difference impact** under ablation.

This is intended to operationalize interchange-intervention style causality checks in a
fully executable setting.
-/

namespace Nfp

/-! ## Head references -/

/-- A reference to a specific attention head `(layer, head)`. -/
structure HeadRef where
  layerIdx : Nat
  headIdx : Nat
deriving BEq, DecidableEq

namespace HeadRef

instance : Inhabited HeadRef := ⟨{ layerIdx := 0, headIdx := 0 }⟩

def toString (h : HeadRef) : String :=
  s!"L{h.layerIdx}H{h.headIdx}"

instance : ToString HeadRef := ⟨toString⟩

def toComponentId (h : HeadRef) : ComponentId :=
  .head h.layerIdx h.headIdx

end HeadRef

/-! ## Induction target utilities -/

/-- Derive the **induction target token** from the model's `inputTokens`, if available.

Let `T` be the token sequence and `t_curr = T[last]`. If `t_curr` appeared previously at index `k`,
the induction target is `t_tgt = T[k+1]` (the successor of the previous match).

Returns `none` if there is no prior repetition of the last token or no token history. -/
def inductionTargetTokenFromHistory (model : ConcreteModel) : Option Nat := do
  let tokens ← model.inputTokens
  if tokens.size = 0 then none else
    let lastIdx := tokens.size - 1
    let tCurr := tokens[lastIdx]!
    let mut foundIdx : Option Nat := none
    for offset in [:lastIdx] do
      if foundIdx.isNone then
        let idx := lastIdx - 1 - offset
        if tokens[idx]! = tCurr then
          foundIdx := some idx
    let k ← foundIdx
    some (tokens[k + 1]!)

/-! ## Verification configuration and results -/

/-- Runtime parameters for causal verification via head ablation. -/
structure VerificationConfig where
  /-- Competence threshold ε for baseline logit-difference Δ. -/
  competenceEpsilon : Float := 1e-3
  /-- Relative tolerance δ for energy matching: |‖out_cand‖ - ‖out_ctrl‖| < δ · ‖out_cand‖. -/
  energyRelTol : Float := 0.05
  /-- Absolute fallback tolerance for tiny-norm heads. -/
  energyAbsTol : Float := 1e-6
  /-- Precision (bits) for dyadic sqrt bounds in SOUND-mode checks. -/
  soundnessBits : Nat := 20
  /-- Whether to run attention causally (autoregressive). -/
  causal : Bool := true

/-- The outcome of checking the runtime axioms required for causal interpretation. -/
structure AxiomStatus where
  baselineCompetence : Bool
  controlIndependence : Bool
  energyEquivalence : Bool
  /-- Human-readable failures (empty iff all axioms hold). -/
  failures : Array String := #[]

namespace AxiomStatus

def verified (s : AxiomStatus) : Bool :=
  s.failures.isEmpty && s.baselineCompetence && s.controlIndependence && s.energyEquivalence

end AxiomStatus

/-- A single verification row for a candidate circuit. -/
structure CircuitVerificationRow where
  /-- Candidate circuit heads (ablated together). -/
  candidateHeads : Array HeadRef
  /-- Selected control heads (one per candidate head, same layers). -/
  controlHeads : Option (Array HeadRef)
  /-- Baseline logit-difference Δ = logit(t_tgt) - logit(t_neg). -/
  baseDelta : Float
  /-- Ablated Δ for the candidate circuit, if axioms verified. -/
  ablatedDelta : Option Float
  /-- Candidate causal impact = Δ_base - Δ_ablated, if axioms verified. -/
  impact : Option Float
  /-- Relative score = impact / Δ_base (only defined if baseline competence holds). -/
  relScore : Option Float
  /-- Control impact = Δ_base - Δ_ctrlAblated, if axioms verified. -/
  controlImpact : Option Float
  /-- Axiom status (checked *before* computing impact scores). -/
  axioms : AxiomStatus

namespace CircuitVerificationRow

def candidateLabel (r : CircuitVerificationRow) : String :=
  if r.candidateHeads.size = 2 then
    let h1 := r.candidateHeads[0]!
    let h2 := r.candidateHeads[1]!
    s!"{h1} -> {h2}"
  else
    String.intercalate "," (r.candidateHeads.toList.map (fun h => toString h))

end CircuitVerificationRow

/-! ## Low-level helpers: logits, controls, and head ablation -/

private def logitAt (residual : ConcreteMatrix) (pos : Nat)
    (W_U : ConcreteMatrix) (token : Nat) : Option Float :=
  if residual.numCols = W_U.numRows ∧ pos < residual.numRows ∧ token < W_U.numCols then
    some <| Id.run do
      let d := residual.numCols
      let vocab := W_U.numCols
      let rowBase := pos * d
      let mut acc : Float := 0.0
      for k in [:d] do
        -- SAFETY: `pos < residual.numRows` and `k < d = residual.numCols`.
        let x := residual.data[rowBase + k]!
        -- SAFETY: `k < W_U.numRows` and `token < vocab = W_U.numCols`.
        let w := W_U.data[k * vocab + token]!
        acc := acc + x * w
      return acc
  else none

private def deltaAt (residual : ConcreteMatrix) (pos : Nat)
    (W_U : ConcreteMatrix) (targetToken negativeToken : Nat) : Float :=
  let targetLogit := (logitAt residual pos W_U targetToken).getD 0.0
  let negLogit := (logitAt residual pos W_U negativeToken).getD 0.0
  targetLogit - negLogit

private def topNonTargetToken (residual : ConcreteMatrix) (pos : Nat)
    (W_U : ConcreteMatrix) (targetToken : Nat) : Option (Nat × Float) := Id.run do
  if residual.numCols = W_U.numRows ∧ pos < residual.numRows ∧
        targetToken < W_U.numCols ∧ W_U.numCols ≥ 2 then
    let d := residual.numCols
    let vocab := W_U.numCols
    let rowBase := pos * d
    let mut bestTok : Nat := 0
    let mut bestLogit : Float := (-Float.inf)
    let mut found : Bool := false
    for tok in [:vocab] do
      if tok ≠ targetToken then
        found := true
        let mut acc : Float := 0.0
        for k in [:d] do
          -- SAFETY: `pos < residual.numRows` and `k < d = residual.numCols`.
          let x := residual.data[rowBase + k]!
          -- SAFETY: `k < W_U.numRows` and `tok < vocab = W_U.numCols`.
          let w := W_U.data[k * vocab + tok]!
          acc := acc + x * w
        if acc > bestLogit then
          bestTok := tok
          bestLogit := acc
    if found then
      return some (bestTok, bestLogit)
    else return none
  else
    return none

private def fullCircuit (model : ConcreteModel) : ConcreteCircuit := Id.run do
  let headsPerLayer :=
    Array.ofFn (fun l : Fin model.numLayers => (model.layers.getD l.1 #[]).size)
  let neuronsPerLayer :=
    Array.ofFn (fun l : Fin model.numLayers => model.numNeuronsAtLayer l.1)
  ConcreteCircuit.full model.numLayers headsPerLayer neuronsPerLayer

private def runForwardAblatingHeads (model : ConcreteModel) (heads : Array HeadRef)
    (causal : Bool := true) : ForwardPassResult :=
  let base := fullCircuit model
  let circuit := heads.foldl (fun c h => c.removeComponent h.toComponentId) base
  model.runAblatedForward circuit causal

private def headOutputNorm? (fwd : ForwardPassResult) (h : HeadRef) : Option Float :=
  if hl : h.layerIdx < fwd.attnOutputs.size then
    let layerOut := fwd.attnOutputs[h.layerIdx]
    if hh : h.headIdx < layerOut.size then
      some ((layerOut[h.headIdx]'hh).frobeniusNorm)
    else none
  else none

private structure ControlSelection where
  candidate : HeadRef
  control : HeadRef
  candNorm : Float
  ctrlNorm : Float
  absDiff : Float

private def selectEnergyMatchedControl (fwd : ForwardPassResult)
    (cand : HeadRef) (exclude : Array HeadRef) : Option ControlSelection := Id.run do
  match headOutputNorm? fwd cand with
  | none => none
  | some candNorm =>
    if hl : cand.layerIdx < fwd.attnOutputs.size then
      let layerOut := fwd.attnOutputs[cand.layerIdx]
      let best : Option (HeadRef × Float × Float) := Id.run do
        let mut best : Option (HeadRef × Float × Float) := none
        for hIdx in [:layerOut.size] do
          let h : HeadRef := { layerIdx := cand.layerIdx, headIdx := hIdx }
          if !exclude.contains h then
            if hh : hIdx < layerOut.size then
              let norm := (layerOut[hIdx]'hh).frobeniusNorm
              let diff := Float.abs (candNorm - norm)
              match best with
              | none =>
                  best := some (h, norm, diff)
              | some (_, _, bestDiff) =>
                  if diff < bestDiff then
                    best := some (h, norm, diff)
        return best
      match best with
      | none => none
      | some (ctrl, ctrlNorm, absDiff) =>
          some { candidate := cand, control := ctrl, candNorm, ctrlNorm, absDiff }
    else
      none

private def energyEquivalent (cfg : VerificationConfig) (candNorm : Float)
    (absDiff : Float) : Bool :=
  let thresh := max cfg.energyAbsTol (cfg.energyRelTol * candNorm)
  absDiff < thresh

/-! ## Verification context and core API -/

/-- Shared baseline information for circuit verification on a fixed prompt. -/
structure VerificationContext where
  model : ConcreteModel
  W_U : ConcreteMatrix
  /-- Logits are evaluated at this position (typically the last token). -/
  pos : Nat
  targetToken : Nat
  negativeToken : Nat
  baseTargetLogit : Float
  baseNegLogit : Float
  baseDelta : Float
  baselineForward : ForwardPassResult
  cfg : VerificationConfig

namespace VerificationContext

/-- Build a verification context for a fixed target token.

The negative token `t_neg` is chosen as the **top non-target** logit at the evaluation position.
-/
def build (model : ConcreteModel) (targetToken : Nat)
    (cfg : VerificationConfig := {}) : Except String VerificationContext :=
  match model.unembedding with
  | none => .error "Model is missing UNEMBEDDING; cannot compute logits."
  | some W_U =>
      if model.seqLen = 0 then
        .error "Model has seqLen = 0; cannot evaluate logits."
      else
        let pos := model.seqLen - 1
        let baselineForward := model.runForward cfg.causal
        let residual := baselineForward.finalOutput
        match logitAt residual pos W_U targetToken with
        | none =>
            .error s!"Cannot compute logit(target={targetToken}) at pos={pos} \
              (dimension mismatch or token OOB)."
        | some baseTargetLogit =>
            match topNonTargetToken residual pos W_U targetToken with
            | none =>
                .error "Cannot select top non-target token (need vocab ≥ 2 and target in-bounds)."
            | some (negativeToken, baseNegLogit) =>
                let baseDelta := baseTargetLogit - baseNegLogit
                .ok {
                  model := model
                  W_U := W_U
                  pos := pos
                  targetToken := targetToken
                  negativeToken := negativeToken
                  baseTargetLogit := baseTargetLogit
                  baseNegLogit := baseNegLogit
                  baseDelta := baseDelta
                  baselineForward := baselineForward
                  cfg := cfg
                }

end VerificationContext

/-- Verify a candidate circuit by **head ablation** with an energy-matched control.

This function enforces the runtime axioms:
1. Baseline competence: `Δ_base > ε`
2. Control independence: control heads disjoint from candidate heads
3. Energy equivalence: per-head output norms match within tolerance

Only if these axioms are verified do we run the ablations and compute impact scores.
-/
def verifyCircuit (ctx : VerificationContext) (candidateHeads : Array HeadRef) :
    CircuitVerificationRow := Id.run do
  let baseDelta := ctx.baseDelta
  let competence := baseDelta > ctx.cfg.competenceEpsilon
  let mut failures : Array String := #[]
  if !competence then
    failures := failures.push s!"Axiom1(baseline competence) failed: Δ_base={baseDelta} ≤ \
      ε={ctx.cfg.competenceEpsilon}"
    let axioms : AxiomStatus := {
      baselineCompetence := false
      controlIndependence := true
      energyEquivalence := true
      failures := failures
    }
    return {
      candidateHeads := candidateHeads
      controlHeads := none
      baseDelta := baseDelta
      ablatedDelta := none
      impact := none
      relScore := none
      controlImpact := none
      axioms := axioms
    }

  -- Choose one control head per candidate head (same layer, closest output norm).
  let mut selections : Array ControlSelection := #[]
  for cand in candidateHeads do
    match selectEnergyMatchedControl ctx.baselineForward cand candidateHeads with
    | some sel => selections := selections.push sel
    | none =>
        failures := failures.push s!"No control head found in layer {cand.layerIdx} for {cand}"

  let controlsComplete := selections.size = candidateHeads.size
  let controlHeads : Array HeadRef := selections.map (·.control)
  let independence := !(controlHeads.any candidateHeads.contains)
  if !independence then
    failures := failures.push "Axiom2(control independence) failed: control overlaps candidate."

  let energyOk :=
    controlsComplete && selections.all (fun s => energyEquivalent ctx.cfg s.candNorm s.absDiff)
  if !energyOk then
    failures := failures.push "Axiom3(energy equivalence) failed: no sufficiently matched control."

  let axioms : AxiomStatus := {
    baselineCompetence := competence
    controlIndependence := independence
    energyEquivalence := energyOk
    failures := failures
  }

  if !axioms.verified then
    return {
      candidateHeads := candidateHeads
      controlHeads := if controlsComplete then some controlHeads else none
      baseDelta := baseDelta
      ablatedDelta := none
      impact := none
      relScore := none
      controlImpact := none
      axioms := axioms
    }

  -- Candidate ablation
  let fwdAblated := runForwardAblatingHeads ctx.model candidateHeads ctx.cfg.causal
  let residualAblated := fwdAblated.finalOutput
  let ablatedDelta :=
    deltaAt residualAblated ctx.pos ctx.W_U ctx.targetToken ctx.negativeToken
  let impact := baseDelta - ablatedDelta
  let relScore := if baseDelta > 0.0 then impact / baseDelta else 0.0

  -- Control ablation (energy-matched, layer-matched)
  let fwdCtrl := runForwardAblatingHeads ctx.model controlHeads ctx.cfg.causal
  let residualCtrl := fwdCtrl.finalOutput
  let ctrlDelta :=
    deltaAt residualCtrl ctx.pos ctx.W_U ctx.targetToken ctx.negativeToken
  let controlImpact := baseDelta - ctrlDelta

  return {
    candidateHeads := candidateHeads
    controlHeads := some controlHeads
    baseDelta := baseDelta
    ablatedDelta := some ablatedDelta
    impact := some impact
    relScore := some relScore
    controlImpact := some controlImpact
    axioms := axioms
  }

end Nfp
