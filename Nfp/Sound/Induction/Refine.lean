-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Induction.Core

/-!
Refine-on-demand helpers for induction-head bounds.

These definitions reuse cached core bounds to compute tightened score gaps and
weight bounds for selected query/key pairs without rebuilding the full cache.
-/

public section

namespace Nfp

namespace Sound

variable {seq dModel dHead : Nat}

/-- Specification for refining per-key bounds. -/
structure InductionHeadRefineSpec (seq : Nat) where
  /-- Keys to refine for each query. -/
  refineKeys : Fin seq → Finset (Fin seq)
  /-- Split budget for refined diff bounds. -/
  splitBudgetDiffRefined : Nat

/-- Heuristic boost for refinement budgets. -/
def refineBudgetBoost (budget : Nat) : Nat :=
  max (budget + 1) (2 * budget)

/-- Unfolding lemma for `refineBudgetBoost`. -/
theorem refineBudgetBoost_def (budget : Nat) :
    refineBudgetBoost budget = max (budget + 1) (2 * budget) := by
  rfl

/-- Worst key under the base score-gap lower bound (excluding `prev`). -/
def worstKeyBase
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) : Option (Fin seq) :=
  let ks := (List.finRange seq).filter (fun k => decide (k ≠ inputs.prev q))
  match ks with
  | [] => none
  | k :: ks =>
      let step (best : Rat × Fin seq) (k : Fin seq) :=
        let s := cache.scoreGapLoBase q k
        if s ≤ best.1 then (s, k) else best
      some (ks.foldl step (cache.scoreGapLoBase q k, k)).2

/-- Unfolding lemma for `worstKeyBase`. -/
theorem worstKeyBase_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) :
    worstKeyBase inputs cache q =
      let ks := (List.finRange seq).filter (fun k => decide (k ≠ inputs.prev q))
      match ks with
      | [] => none
      | k :: ks =>
          let step (best : Rat × Fin seq) (k : Fin seq) :=
            let s := cache.scoreGapLoBase q k
            if s ≤ best.1 then (s, k) else best
          some (ks.foldl step (cache.scoreGapLoBase q k, k)).2 := by
  rfl

/-- Keys whose base weight bounds are already `1`. -/
def weightOneKeysAt
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) : Finset (Fin seq) :=
  let others : Finset (Fin seq) :=
    (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
  others.filter (fun k => decide (cache.weightBoundAt q k = (1 : Rat)))

/-- Unfolding lemma for `weightOneKeysAt`. -/
theorem weightOneKeysAt_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) :
    weightOneKeysAt inputs cache q =
      let others : Finset (Fin seq) :=
        (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
      others.filter (fun k => decide (cache.weightBoundAt q k = (1 : Rat))) := by
  rfl

/-- Refinement keys for a query, seeded by negative base gaps and the worst key. -/
def refineKeysAt
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) : Finset (Fin seq) :=
  let neg :=
    (cache.otherKeys q).filter (fun k => decide (cache.scoreGapLoBase q k < 0))
  match worstKeyBase inputs cache q with
  | none => neg
  | some k => insert k neg

/-- Unfolding lemma for `refineKeysAt`. -/
theorem refineKeysAt_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) :
    refineKeysAt inputs cache q =
      let neg :=
        (cache.otherKeys q).filter (fun k => decide (cache.scoreGapLoBase q k < 0))
      match worstKeyBase inputs cache q with
      | none => neg
      | some k => insert k neg := by
  rfl

/-- Refinement keys that also include weight-one keys. -/
def refineKeysAtWithWeightOnes
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) : Finset (Fin seq) :=
  refineKeysAt inputs cache q ∪ weightOneKeysAt inputs cache q

/-- Unfolding lemma for `refineKeysAtWithWeightOnes`. -/
theorem refineKeysAtWithWeightOnes_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) :
    refineKeysAtWithWeightOnes inputs cache q =
      refineKeysAt inputs cache q ∪ weightOneKeysAt inputs cache q := by
  rfl

/-- Refinement spec focused on a single query. -/
def refineSpecForQuery
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) (budget : Nat) : InductionHeadRefineSpec seq :=
  let keys := refineKeysAt inputs cache q
  { refineKeys := fun q' => if _ : q' = q then keys else ∅
    splitBudgetDiffRefined := budget }

/-- Unfolding lemma for `refineSpecForQuery`. -/
theorem refineSpecForQuery_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) (budget : Nat) :
    refineSpecForQuery inputs cache q budget =
      let keys := refineKeysAt inputs cache q
      { refineKeys := fun q' => if _ : q' = q then keys else ∅
        splitBudgetDiffRefined := budget } := by
  rfl

/-- Refinement spec for a single query, including weight-one keys. -/
def refineSpecForQueryWithWeightOnes
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) (budget : Nat) : InductionHeadRefineSpec seq :=
  let keys := refineKeysAtWithWeightOnes inputs cache q
  { refineKeys := fun q' => if _ : q' = q then keys else ∅
    splitBudgetDiffRefined := budget }

/-- Unfolding lemma for `refineSpecForQueryWithWeightOnes`. -/
theorem refineSpecForQueryWithWeightOnes_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (q : Fin seq) (budget : Nat) :
    refineSpecForQueryWithWeightOnes inputs cache q budget =
      let keys := refineKeysAtWithWeightOnes inputs cache q
      { refineKeys := fun q' => if _ : q' = q then keys else ∅
        splitBudgetDiffRefined := budget } := by
  rfl

/-- Refined diff dot-product lower bound at a single `(q,k)` pair. -/
def dotDiffLoRefinedAt
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat) (q k : Fin seq) : Rat :=
  let dimsQ := cache.splitDimsQ q
  let dimsDiff := cache.splitDimsDiffCore budget q k
  let prev := inputs.prev q
  (_root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsDiff
    (fun d => cache.qLo q d) (fun d => cache.qHi q d)
    (fun d => cache.kLo prev d - cache.kHi k d)
    (fun d => cache.kHi prev d - cache.kLo k d)).1

/-- Unfolding lemma for `dotDiffLoRefinedAt`. -/
theorem dotDiffLoRefinedAt_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat) (q k : Fin seq) :
    dotDiffLoRefinedAt inputs cache budget q k =
      let dimsQ := cache.splitDimsQ q
      let dimsDiff := cache.splitDimsDiffCore budget q k
      let prev := inputs.prev q
      (_root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsDiff
        (fun d => cache.qLo q d) (fun d => cache.qHi q d)
        (fun d => cache.kLo prev d - cache.kHi k d)
        (fun d => cache.kHi prev d - cache.kLo k d)).1 := by
  rfl

/-- Refined diff dot-product upper bound at a single `(q,k)` pair. -/
def dotDiffHiRefinedAt
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat) (q k : Fin seq) : Rat :=
  let dimsQ := cache.splitDimsQ q
  let dimsDiff := cache.splitDimsDiffCore budget q k
  let prev := inputs.prev q
  (_root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsDiff
    (fun d => cache.qLo q d) (fun d => cache.qHi q d)
    (fun d => cache.kLo prev d - cache.kHi k d)
    (fun d => cache.kHi prev d - cache.kLo k d)).2

/-- Unfolding lemma for `dotDiffHiRefinedAt`. -/
theorem dotDiffHiRefinedAt_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat) (q k : Fin seq) :
    dotDiffHiRefinedAt inputs cache budget q k =
      let dimsQ := cache.splitDimsQ q
      let dimsDiff := cache.splitDimsDiffCore budget q k
      let prev := inputs.prev q
      (_root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsDiff
        (fun d => cache.qLo q d) (fun d => cache.qHi q d)
        (fun d => cache.kLo prev d - cache.kHi k d)
        (fun d => cache.kHi prev d - cache.kLo k d)).2 := by
  rfl

/-- Refined score-gap lower bound at `(q,k)` using a custom diff budget. -/
def scoreGapLoRefinedAt
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat) (q k : Fin seq) : Rat :=
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  if masked q (inputs.prev q) then
    cache.scoreLoPrev q - cache.scoreHi q k
  else if masked q k then
    cache.scoreLoPrev q - inputs.maskValue
  else if _ : 0 ≤ inputs.scale then
    inputs.scale * dotDiffLoRefinedAt inputs cache budget q k
  else
    inputs.scale * dotDiffHiRefinedAt inputs cache budget q k

/-- Unfolding lemma for `scoreGapLoRefinedAt`. -/
theorem scoreGapLoRefinedAt_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat) (q k : Fin seq) :
    scoreGapLoRefinedAt inputs cache budget q k =
      let masked : Fin seq → Fin seq → Prop := fun q k =>
        inputs.maskCausal = true ∧ q < k
      if masked q (inputs.prev q) then
        cache.scoreLoPrev q - cache.scoreHi q k
      else if masked q k then
        cache.scoreLoPrev q - inputs.maskValue
      else if _ : 0 ≤ inputs.scale then
        inputs.scale * dotDiffLoRefinedAt inputs cache budget q k
      else
        inputs.scale * dotDiffHiRefinedAt inputs cache budget q k := by
  rfl

/-- Refined per-key weight bound at `(q,k)` derived from refined score gaps. -/
def weightBoundAtRefinedAt
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat) (q k : Fin seq) : Rat :=
  if _ : k = inputs.prev q then
    (0 : Rat)
  else
    let gap := scoreGapLoRefinedAt inputs cache budget q k
    if gap < 0 then
      (1 : Rat)
    else
      ratDivUp 1 (1 + gap)

/-- Unfolding lemma for `weightBoundAtRefinedAt`. -/
theorem weightBoundAtRefinedAt_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (budget : Nat) (q k : Fin seq) :
    weightBoundAtRefinedAt inputs cache budget q k =
      if _ : k = inputs.prev q then
        (0 : Rat)
      else
        let gap := scoreGapLoRefinedAt inputs cache budget q k
        if gap < 0 then
          (1 : Rat)
        else
          ratDivUp 1 (1 + gap) := by
  rfl

/-- Overlay that refines only selected `(q,k)` weight bounds. -/
def weightBoundAtOverlay
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (spec : InductionHeadRefineSpec seq) :
    Fin seq → Fin seq → Rat := fun q k =>
  if _ : k = inputs.prev q then
    (0 : Rat)
  else if _ : k ∈ spec.refineKeys q then
    weightBoundAtRefinedAt inputs cache spec.splitBudgetDiffRefined q k
  else
    cache.weightBoundAt q k

/-- Unfolding lemma for `weightBoundAtOverlay`. -/
theorem weightBoundAtOverlay_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (cache : InductionHeadCoreCache seq dModel dHead)
    (spec : InductionHeadRefineSpec seq)
    (q k : Fin seq) :
    weightBoundAtOverlay inputs cache spec q k =
      if _ : k = inputs.prev q then
        (0 : Rat)
      else if _ : k ∈ spec.refineKeys q then
        weightBoundAtRefinedAt inputs cache spec.splitBudgetDiffRefined q k
      else
        cache.weightBoundAt q k := by
  rfl

/-- Overlayed eps bound derived from overlayed per-key bounds. -/
def epsAtOverlay
    (cache : InductionHeadCoreCache seq dModel dHead)
    (weightBoundAt : Fin seq → Fin seq → Rat) :
    Fin seq → Rat := fun q =>
  let other : Finset (Fin seq) :=
    (Finset.univ : Finset (Fin seq)).erase (cache.cert.prev q)
  let total := other.sum (fun k => weightBoundAt q k)
  min (1 : Rat) total

/-- Unfolding lemma for `epsAtOverlay`. -/
theorem epsAtOverlay_def
    (cache : InductionHeadCoreCache seq dModel dHead)
    (weightBoundAt : Fin seq → Fin seq → Rat)
    (q : Fin seq) :
    epsAtOverlay cache weightBoundAt q =
      let other : Finset (Fin seq) :=
        (Finset.univ : Finset (Fin seq)).erase (cache.cert.prev q)
      let total := other.sum (fun k => weightBoundAt q k)
      min (1 : Rat) total := by
  rfl

end Sound

end Nfp
