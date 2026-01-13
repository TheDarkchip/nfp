-- SPDX-License-Identifier: AGPL-3.0-or-later
module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.Field.Basic
public import Nfp.Core.Basic
public import Mathlib.Data.Finset.Lattice.Fold
public import Nfp.Circuit.Cert.ResidualInterval
public import Nfp.Circuit.Cert.SoftmaxMargin
public import Nfp.Circuit.Cert.ValueRange
public import Nfp.Sound.Bounds.Attention
public import Nfp.Sound.Bounds.Cache
public import Nfp.Sound.Bounds.LayerNorm
public import Nfp.Sound.Bounds.LayerNorm.InvStd
public import Nfp.Sound.Bounds.MatrixNorm
public import Nfp.Sound.Induction.CoreDefs
public import Nfp.Sound.Induction.OneHot
public import Nfp.Sound.Linear.FinFold
/-! Sound builders for induction certificates; recompute bounds inside Lean from exact inputs and
derive softmax tolerances from score margins rather than trusting external weight dumps. -/

@[expose] public section

namespace Nfp
namespace Sound
open scoped BigOperators
open Nfp.Circuit
open Nfp.Sound.Bounds
variable {seq : Nat}
/-- Build and certify a softmax-margin certificate from exact scores/weights. -/
def buildSoftmaxMarginCert? [NeZero seq]
    (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (scores : Fin seq → Fin seq → Rat)
    (weights : Fin seq → Fin seq → Rat) :
    Option {c : SoftmaxMarginCert seq // checkSoftmaxMarginCert c = true} := by
  classical
  let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (prev q)
  let epsAt : Fin seq → Rat := fun q =>
    let other := otherKeys q
    let maxOther :=
      if h : other.Nonempty then
        other.sup' h (fun k => weights q k)
      else
        (0 : Rat)
    let deficit := (1 : Rat) - weights q (prev q)
    max maxOther deficit
  let marginAt : Fin seq → Rat := fun q =>
    let other := otherKeys q
    if h : other.Nonempty then
      other.inf' h (fun k => scores q (prev q) - scores q k)
    else
      (0 : Rat)
  let eps :=
    if h : active.Nonempty then
      active.sup' h epsAt
    else
      (0 : Rat)
  let margin :=
    if h : active.Nonempty then
      active.inf' h marginAt
    else
      (0 : Rat)
  let cert : SoftmaxMarginCert seq :=
    { eps := eps
      margin := margin
      active := active
      prev := prev
      scores := scores
      weights := weights }
  if h : checkSoftmaxMarginCert cert = true then
    exact some ⟨cert, h⟩
  else
    exact none
/-- Build and certify a value-range certificate from exact values. -/
def buildValueRangeCert? [NeZero seq]
    (vals : Fin seq → Rat)
    (direction : Option DirectionSpec) :
    Option {c : ValueRangeCert seq // checkValueRangeCert c = true} := by
  classical
  let _ : Nonempty (Fin seq) := by
    refine ⟨⟨0, ?_⟩⟩
    exact Nat.pos_of_ne_zero (NeZero.ne seq)
  let univ : Finset (Fin seq) := Finset.univ
  let hnonempty : univ.Nonempty := Finset.univ_nonempty
  let lo := univ.inf' hnonempty vals
  let hi := univ.sup' hnonempty vals
  let cert : ValueRangeCert seq :=
    { lo := lo
      hi := hi
      vals := vals
      direction := direction }
  if h : checkValueRangeCert cert = true then
    exact some ⟨cert, h⟩
  else
    exact none
/-- Cached bounds and derived quantities for induction-head core certificates. -/
structure InductionHeadCoreCache (seq dModel dHead : Nat) where
  /-- Cached LayerNorm bound pair. -/
  lnBounds : (Fin seq → Fin dModel → Rat) × (Fin seq → Fin dModel → Rat)
  /-- LayerNorm lower bounds. -/
  lnLo : Fin seq → Fin dModel → Rat
  /-- LayerNorm upper bounds. -/
  lnHi : Fin seq → Fin dModel → Rat
  /-- Tasks for LayerNorm absolute maxima. -/
  lnAbsMaxTask : Fin seq → Rat
  /-- Cached LayerNorm absolute maxima. -/
  lnAbsMaxArr : Array Rat
  /-- LayerNorm absolute-max lookup. -/
  lnAbsMax : Fin seq → Rat
  /-- Tasks for inverse-std bounds. -/
  invStdBoundsTasks : Array (Task (Rat × Rat))
  /-- Cached inverse-std bounds. -/
  invStdBoundsArr : Array (Rat × Rat)
  /-- Inverse-std lower bounds. -/
  invStdLo : Fin seq → Rat
  /-- Inverse-std upper bounds. -/
  invStdHi : Fin seq → Rat
  /-- Cached query base terms. -/
  qBaseArr : Array Rat
  /-- Query base lookup. -/
  qBase : Fin dHead → Rat
  /-- Cached key base terms. -/
  kBaseArr : Array Rat
  /-- Key base lookup. -/
  kBase : Fin dHead → Rat
  /-- Tasks for query coefficient rows. -/
  qCoeffRowTasks : Array (Task { row : Array Rat // row.size = dHead })
  /-- Cached query coefficient rows. -/
  qCoeffArr : Array { row : Array Rat // row.size = dHead }
  /-- Query coefficient lookup. -/
  qCoeff : Fin seq → Fin dHead → Rat
  /-- Tasks for key coefficient rows. -/
  kCoeffRowTasks : Array (Task { row : Array Rat // row.size = dHead })
  /-- Cached key coefficient rows. -/
  kCoeffArr : Array { row : Array Rat // row.size = dHead }
  /-- Key coefficient lookup. -/
  kCoeff : Fin seq → Fin dHead → Rat
  /-- Query lower bounds. -/
  qLo : Fin seq → Fin dHead → Rat
  /-- Query upper bounds. -/
  qHi : Fin seq → Fin dHead → Rat
  /-- Key lower bounds. -/
  kLo : Fin seq → Fin dHead → Rat
  /-- Key upper bounds. -/
  kHi : Fin seq → Fin dHead → Rat
  /-- Query absolute bounds. -/
  qAbs : Fin seq → Fin dHead → Rat
  /-- Key absolute bounds. -/
  kAbs : Fin seq → Fin dHead → Rat
  /-- Cached max query abs bounds. -/
  qAbsMaxArr : Array Rat
  /-- Max query abs bound lookup. -/
  qAbsMax : Fin dHead → Rat
  /-- Cached max key abs bounds. -/
  kAbsMaxArr : Array Rat
  /-- Max key abs bound lookup. -/
  kAbsMax : Fin dHead → Rat
  /-- Causal mask predicate. -/
  masked : Fin seq → Fin seq → Prop
  /-- Split budget for query dims. -/
  splitBudgetQ : Nat
  /-- Split budget for key dims. -/
  splitBudgetK : Nat
  /-- Split budget for base diff dims. -/
  splitBudgetDiffBase : Nat
  /-- Split budget for refined diff dims. -/
  splitBudgetDiffRefined : Nat
  /-- Split dims for query bounds. -/
  splitDimsQ : Fin seq → List (Fin dHead)
  /-- Split dims for key bounds. -/
  splitDimsK : Fin seq → Fin seq → List (Fin dHead)
  /-- Split dims for diff bounds with budget. -/
  splitDimsDiffCore : Nat → Fin seq → Fin seq → List (Fin dHead)
  /-- Split dims for base diff bounds. -/
  splitDimsDiffBase : Fin seq → Fin seq → List (Fin dHead)
  /-- Split dims for refined diff bounds. -/
  splitDimsDiffRefined : Fin seq → Fin seq → List (Fin dHead)
  /-- Tasks for dot-product interval rows. -/
  dotRowTasks : Array (Task { row : Array (Rat × Rat) // row.size = seq })
  /-- Tasks for base diff dot rows. -/
  dotDiffRowTasksBase : Array (Task { row : Array (Rat × Rat) // row.size = seq })
  /-- Dot-product lower bounds. -/
  dotLo : Fin seq → Fin seq → Rat
  /-- Dot-product upper bounds. -/
  dotHi : Fin seq → Fin seq → Rat
  /-- Base diff dot-product lower bounds. -/
  dotDiffLoBase : Fin seq → Fin seq → Rat
  /-- Base diff dot-product upper bounds. -/
  dotDiffHiBase : Fin seq → Fin seq → Rat
  /-- Dot-product absolute bounds. -/
  dotAbs : Fin seq → Fin seq → Rat
  /-- Base score absolute bounds. -/
  scoreBaseAbs : Fin seq → Fin seq → Rat
  /-- Score lower bounds. -/
  scoreLo : Fin seq → Fin seq → Rat
  /-- Score upper bounds. -/
  scoreHi : Fin seq → Fin seq → Rat
  /-- Score lower bounds at prev key. -/
  scoreLoPrev : Fin seq → Rat
  /-- Base score-gap lower bounds. -/
  scoreGapLoBase : Fin seq → Fin seq → Rat
  /-- Other-key set for each query. -/
  otherKeys : Fin seq → Finset (Fin seq)
  /-- Worst key candidate per query. -/
  worstKey : Fin seq → Option (Fin seq)
  /-- Refined diff dot-product lower bounds. -/
  dotDiffLo : Fin seq → Fin seq → Rat
  /-- Refined diff dot-product upper bounds. -/
  dotDiffHi : Fin seq → Fin seq → Rat
  /-- Score-gap lower bounds. -/
  scoreGapLo : Fin seq → Fin seq → Rat
  /-- Margin per query. -/
  marginAt : Fin seq → Rat
  /-- Epsilon per query. -/
  epsAt : Fin seq → Rat
  /-- Per-key weight bounds derived from score gaps. -/
  weightBoundAt : Fin seq → Fin seq → Rat
  /-- Global margin. -/
  margin : Rat
  /-- Global epsilon. -/
  eps : Rat
  /-- Cached direction head vector. -/
  dirHeadVec : Vector Rat dHead
  /-- Direction head lookup. -/
  dirHead : Fin dHead → Rat
  /-- Value-direction weight dot products. -/
  wvDir : Fin dModel → Rat
  /-- Direction bias term. -/
  bDir : Rat
  /-- Value lower bounds. -/
  valsLo : Fin seq → Rat
  /-- Value upper bounds. -/
  valsHi : Fin seq → Rat
  /-- Universe of query indices. -/
  univ : Finset (Fin seq)
  /-- Global value lower bound. -/
  lo : Rat
  /-- Global value upper bound. -/
  hi : Rat
  /-- Value-interval certificate. -/
  valCert : ValueInterval seq
  /-- Induction-head certificate. -/
  cert : InductionHeadCert seq

/-- Build cached core quantities for induction-head certificates. -/
def buildInductionHeadCoreCacheWith [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    InductionHeadCoreCache seq dModel dHead := by
  classical
  let lnBounds := Bounds.cacheBoundPair2 (fun q =>
    Bounds.layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q))
  let lnLo : Fin seq → Fin dModel → Rat := lnBounds.1
  let lnHi : Fin seq → Fin dModel → Rat := lnBounds.2
  let lnAbsMaxTask : Fin seq → Rat :=
    Bounds.cacheBoundTask (fun q =>
      Bounds.intervalAbsBound (lnLo q) (lnHi q))
  let lnAbsMaxArr : Array Rat :=
    Array.ofFn (fun q : Fin seq => lnAbsMaxTask q)
  let lnAbsMax : Fin seq → Rat := fun q =>
    lnAbsMaxArr[q.1]'(by
      have hsize : lnAbsMaxArr.size = seq := by
        simp [lnAbsMaxArr]
      simp [hsize])
  let invStdBoundsTasks : Array (Task (Rat × Rat)) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ => invStdBounds inputs.lnEps (inputs.embed q)))
  let invStdBoundsArr : Array (Rat × Rat) :=
    Array.ofFn (fun q : Fin seq =>
      (invStdBoundsTasks[q.1]'(by
        have hsize : invStdBoundsTasks.size = seq := by
          simp [invStdBoundsTasks]
        simp [hsize])).get)
  let invStdLo : Fin seq → Rat := fun q =>
    (invStdBoundsArr[q.1]'(by
      have hsize : invStdBoundsArr.size = seq := by
        simp [invStdBoundsArr]
      simp [hsize])).1
  let invStdHi : Fin seq → Rat := fun q =>
    (invStdBoundsArr[q.1]'(by
      have hsize : invStdBoundsArr.size = seq := by
        simp [invStdBoundsArr]
      simp [hsize])).2
  let qBaseArr : Array Rat :=
    Array.ofFn (fun d : Fin dHead =>
      Linear.dotFin dModel (fun j => inputs.wq j d) (fun j => inputs.ln1Beta j) +
        inputs.bq d)
  let qBase : Fin dHead → Rat := fun d =>
    qBaseArr[d.1]'(by
      have hsize : qBaseArr.size = dHead := by
        simp [qBaseArr]
      simp [hsize])
  let kBaseArr : Array Rat :=
    Array.ofFn (fun d : Fin dHead =>
      Linear.dotFin dModel (fun j => inputs.wk j d) (fun j => inputs.ln1Beta j) +
        inputs.bk d)
  let kBase : Fin dHead → Rat := fun d =>
    kBaseArr[d.1]'(by
      have hsize : kBaseArr.size = dHead := by
        simp [kBaseArr]
      simp [hsize])
  let qCoeffRowTasks : Array (Task { row : Array Rat // row.size = dHead }) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        let μ := mean (inputs.embed q)
        let coeff : Fin dModel → Rat := fun j =>
          inputs.ln1Gamma j * (inputs.embed q j - μ)
        ⟨Array.ofFn (fun d : Fin dHead =>
            Linear.dotFin dModel (fun j => inputs.wq j d) coeff),
          by simp⟩))
  let qCoeffArr : Array { row : Array Rat // row.size = dHead } :=
    Array.ofFn (fun q : Fin seq =>
      (qCoeffRowTasks[q.1]'(by
        have hsize : qCoeffRowTasks.size = seq := by
          simp [qCoeffRowTasks]
        simp [hsize])).get)
  let qCoeff : Fin seq → Fin dHead → Rat := fun q d =>
    let row := qCoeffArr[q.1]'(by
      have hsize : qCoeffArr.size = seq := by
        simp [qCoeffArr]
      simp [hsize])
    row.1[d.1]'(by
      simp [row.2])
  let kCoeffRowTasks : Array (Task { row : Array Rat // row.size = dHead }) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        let μ := mean (inputs.embed q)
        let coeff : Fin dModel → Rat := fun j =>
          inputs.ln1Gamma j * (inputs.embed q j - μ)
        ⟨Array.ofFn (fun d : Fin dHead =>
            Linear.dotFin dModel (fun j => inputs.wk j d) coeff),
          by simp⟩))
  let kCoeffArr : Array { row : Array Rat // row.size = dHead } :=
    Array.ofFn (fun q : Fin seq =>
      (kCoeffRowTasks[q.1]'(by
        have hsize : kCoeffRowTasks.size = seq := by
          simp [kCoeffRowTasks]
        simp [hsize])).get)
  let kCoeff : Fin seq → Fin dHead → Rat := fun q d =>
    let row := kCoeffArr[q.1]'(by
      have hsize : kCoeffArr.size = seq := by
        simp [kCoeffArr]
      simp [hsize])
    row.1[d.1]'(by
      simp [row.2])
  let qLo : Fin seq → Fin dHead → Rat := fun q d =>
    let bounds := scaleInterval (qCoeff q d) (invStdLo q) (invStdHi q)
    qBase d + bounds.1
  let qHi : Fin seq → Fin dHead → Rat := fun q d =>
    let bounds := scaleInterval (qCoeff q d) (invStdLo q) (invStdHi q)
    qBase d + bounds.2
  let kLo : Fin seq → Fin dHead → Rat := fun q d =>
    let bounds := scaleInterval (kCoeff q d) (invStdLo q) (invStdHi q)
    kBase d + bounds.1
  let kHi : Fin seq → Fin dHead → Rat := fun q d =>
    let bounds := scaleInterval (kCoeff q d) (invStdLo q) (invStdHi q)
    kBase d + bounds.2
  let qAbs : Fin seq → Fin dHead → Rat := fun q d => max |qLo q d| |qHi q d|
  let kAbs : Fin seq → Fin dHead → Rat := fun q d => max |kLo q d| |kHi q d|
  let qAbsMaxArr : Array Rat :=
    Array.ofFn (fun d : Fin dHead =>
      let univ : Finset (Fin seq) := Finset.univ
      have hnonempty : univ.Nonempty := Finset.univ_nonempty
      univ.sup' hnonempty (fun q => qAbs q d))
  let qAbsMax : Fin dHead → Rat := fun d =>
    qAbsMaxArr[d.1]'(by
      have hsize : qAbsMaxArr.size = dHead := by
        simp [qAbsMaxArr]
      simp [hsize])
  let kAbsMaxArr : Array Rat :=
    Array.ofFn (fun d : Fin dHead =>
      let univ : Finset (Fin seq) := Finset.univ
      have hnonempty : univ.Nonempty := Finset.univ_nonempty
      univ.sup' hnonempty (fun k => kAbs k d))
  let kAbsMax : Fin dHead → Rat := fun d =>
    kAbsMaxArr[d.1]'(by
      have hsize : kAbsMaxArr.size = dHead := by
        simp [kAbsMaxArr]
      simp [hsize])
  let masked : Fin seq → Fin seq → Prop := fun q k =>
    inputs.maskCausal = true ∧ q < k
  let splitBudgetQ : Nat := cfg.splitBudgetQ
  let splitBudgetK : Nat := cfg.splitBudgetK
  let splitBudgetDiffBase : Nat := cfg.splitBudgetDiffBase
  let splitBudgetDiffRefined : Nat := cfg.splitBudgetDiffRefined
  let finRangeHead : List (Fin dHead) := List.finRange dHead
  let finRangeSeq : List (Fin seq) := List.finRange seq
  let splitDimsQ : Fin seq → List (Fin dHead) := fun q =>
    if splitBudgetQ = 0 then
      []
    else
      let ambig :=
        finRangeHead.filter (fun d => decide (qLo q d < 0 ∧ 0 < qHi q d))
      let score : Fin dHead → Rat := fun d => (qHi q d - qLo q d) * kAbsMax d
      let step
          (best : Option (Rat × Fin dHead) × Option (Rat × Fin dHead))
          (d : Fin dHead) :
          Option (Rat × Fin dHead) × Option (Rat × Fin dHead) :=
        let s := score d
        match best with
        | (none, none) => (some (s, d), none)
        | (some b1, none) =>
            if b1.1 < s then (some (s, d), some b1) else (some b1, some (s, d))
        | (some b1, some b2) =>
            if b1.1 < s then (some (s, d), some b1)
            else if b2.1 < s then (some b1, some (s, d)) else (some b1, some b2)
        | (none, some b2) =>
            if b2.1 < s then (some (s, d), some b2) else (some b2, some (s, d))
      let top2 : List (Fin dHead) → List (Fin dHead) := fun ambig =>
        match ambig.foldl step (none, none) with
        | (some b1, some b2) => [b1.2, b2.2]
        | (some b1, none) => [b1.2]
        | (none, _) => []
      let dims1 := top2 ambig
      let dims2 := top2 (ambig.filter (fun d => decide (d ∉ dims1)))
      (dims1 ++ dims2).take splitBudgetQ
  let splitDimsK : Fin seq → Fin seq → List (Fin dHead) := fun q k =>
    if splitBudgetK = 0 then
      []
    else
      let ambig :=
        finRangeHead.filter (fun d => decide (kLo k d < 0 ∧ 0 < kHi k d))
      let score : Fin dHead → Rat := fun d => (kHi k d - kLo k d) * qAbs q d
      let step
          (best : Option (Rat × Fin dHead) × Option (Rat × Fin dHead))
          (d : Fin dHead) :
          Option (Rat × Fin dHead) × Option (Rat × Fin dHead) :=
        let s := score d
        match best with
        | (none, none) => (some (s, d), none)
        | (some b1, none) =>
            if b1.1 < s then (some (s, d), some b1) else (some b1, some (s, d))
        | (some b1, some b2) =>
            if b1.1 < s then (some (s, d), some b1)
            else if b2.1 < s then (some b1, some (s, d)) else (some b1, some b2)
        | (none, some b2) =>
            if b2.1 < s then (some (s, d), some b2) else (some b2, some (s, d))
      let top2 : List (Fin dHead) → List (Fin dHead) := fun ambig =>
        match ambig.foldl step (none, none) with
        | (some b1, some b2) => [b1.2, b2.2]
        | (some b1, none) => [b1.2]
        | (none, _) => []
      let dims1 := top2 ambig
      let dims2 := top2 (ambig.filter (fun d => decide (d ∉ dims1)))
      (dims1 ++ dims2).take splitBudgetK
  let splitDimsDiffCore : Nat → Fin seq → Fin seq → List (Fin dHead) := fun budget q k =>
    if budget = 0 then
      []
    else
      let prev := inputs.prev q
      let diffLo : Fin dHead → Rat := fun d => kLo prev d - kHi k d
      let diffHi : Fin dHead → Rat := fun d => kHi prev d - kLo k d
      let ambig :=
        finRangeHead.filter (fun d => decide (diffLo d < 0 ∧ 0 < diffHi d))
      let score : Fin dHead → Rat := fun d => (diffHi d - diffLo d) * qAbs q d
      let step
          (best : Option (Rat × Fin dHead) × Option (Rat × Fin dHead))
          (d : Fin dHead) :
          Option (Rat × Fin dHead) × Option (Rat × Fin dHead) :=
        let s := score d
        match best with
        | (none, none) => (some (s, d), none)
        | (some b1, none) =>
            if b1.1 < s then (some (s, d), some b1) else (some b1, some (s, d))
        | (some b1, some b2) =>
            if b1.1 < s then (some (s, d), some b1)
            else if b2.1 < s then (some b1, some (s, d)) else (some b1, some b2)
        | (none, some b2) =>
            if b2.1 < s then (some (s, d), some b2) else (some b2, some (s, d))
      let top2 : List (Fin dHead) → List (Fin dHead) := fun ambig =>
        match ambig.foldl step (none, none) with
        | (some b1, some b2) => [b1.2, b2.2]
        | (some b1, none) => [b1.2]
        | (none, _) => []
      let dims1 : List (Fin dHead) := top2 ambig
      let dims2 : List (Fin dHead) :=
        top2 (ambig.filter (fun d => decide (d ∉ dims1)))
      let memDims2 : Fin dHead → Bool := fun d =>
        dims2.any (fun d' => decide (d' = d))
      let dims3 : List (Fin dHead) :=
        top2
          ((ambig.filter (fun d => decide (d ∉ dims1))).filter
            (fun d => !memDims2 d))
      (dims1 ++ dims2 ++ dims3).take budget
  let splitDimsDiffBase : Fin seq → Fin seq → List (Fin dHead) :=
    splitDimsDiffCore splitBudgetDiffBase
  let splitDimsDiffRefined : Fin seq → Fin seq → List (Fin dHead) :=
    splitDimsDiffCore splitBudgetDiffRefined
  let dotRowTasks : Array (Task { row : Array (Rat × Rat) // row.size = seq }) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        let dimsQ := splitDimsQ q
        ⟨Array.ofFn (fun k : Fin seq =>
            if masked q k then
              (0, 0)
            else
              let dimsK := splitDimsK q k
              _root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsK
                (fun d => qLo q d) (fun d => qHi q d)
                (fun d => kLo k d) (fun d => kHi k d)),
          by simp⟩))
  let dotDiffRowTasksBase : Array (Task { row : Array (Rat × Rat) // row.size = seq }) :=
    Array.ofFn (fun q : Fin seq =>
      Task.spawn (fun _ =>
        if hq : q ∈ inputs.active then
          let dimsQ := splitDimsQ q
          ⟨Array.ofFn (fun k : Fin seq =>
              if masked q k then
                (0, 0)
              else
                let dimsDiff := splitDimsDiffBase q k
                let prev := inputs.prev q
                _root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsDiff
                  (fun d => qLo q d) (fun d => qHi q d)
                  (fun d => kLo prev d - kHi k d)
                  (fun d => kHi prev d - kLo k d)),
            by simp⟩
        else
          ⟨Array.ofFn (fun _ : Fin seq => (0, 0)), by simp⟩))
  let dotLo : Fin seq → Fin seq → Rat := fun q k =>
    let row := (dotRowTasks[q.1]'(by
      simp [dotRowTasks, q.isLt])).get
    let entry := row.1[k.1]'(by
      simp [row.2, k.isLt])
    entry.1
  let dotHi : Fin seq → Fin seq → Rat := fun q k =>
    let row := (dotRowTasks[q.1]'(by
      simp [dotRowTasks, q.isLt])).get
    let entry := row.1[k.1]'(by
      simp [row.2, k.isLt])
    entry.2
  let dotDiffLoBase : Fin seq → Fin seq → Rat := fun q k =>
    let row := (dotDiffRowTasksBase[q.1]'(by
      simp [dotDiffRowTasksBase, q.isLt])).get
    let entry := row.1[k.1]'(by
      simp [row.2, k.isLt])
    entry.1
  let dotDiffHiBase : Fin seq → Fin seq → Rat := fun q k =>
    let row := (dotDiffRowTasksBase[q.1]'(by
      simp [dotDiffRowTasksBase, q.isLt])).get
    let entry := row.1[k.1]'(by
      simp [row.2, k.isLt])
    entry.2
  let dotAbs : Fin seq → Fin seq → Rat := fun q k => max |dotLo q k| |dotHi q k|
  let scoreBaseAbs : Fin seq → Fin seq → Rat := fun q k =>
    |inputs.scale| * dotAbs q k
  let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      if hscale : 0 ≤ inputs.scale then
        inputs.scale * dotLo q k
      else
        inputs.scale * dotHi q k
  let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
    if masked q k then
      inputs.maskValue
    else
      if hscale : 0 ≤ inputs.scale then
        inputs.scale * dotHi q k
      else
        inputs.scale * dotLo q k
  let scoreLoPrev : Fin seq → Rat := fun q =>
    scoreLo q (inputs.prev q)
  let scoreGapLoBaseRaw : Fin seq → Fin seq → Rat := fun q k =>
    if masked q (inputs.prev q) then
      scoreLoPrev q - scoreHi q k
    else if masked q k then
      scoreLoPrev q - inputs.maskValue
    else if hscale : 0 ≤ inputs.scale then
      inputs.scale * dotDiffLoBase q k
    else
      inputs.scale * dotDiffHiBase q k
  let scoreGapLoBase : Fin seq → Fin seq → Rat :=
    Bounds.cacheBound2 scoreGapLoBaseRaw
  let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
  let worstKeyRaw : Fin seq → Option (Fin seq) := fun q =>
    if hq : q ∈ inputs.active then
      let ks := finRangeSeq.filter (fun k => decide (k ≠ inputs.prev q))
      match ks with
      | [] => none
      | k :: ks =>
          let step (best : Rat × Fin seq) (k : Fin seq) :=
            let s := scoreGapLoBase q k
            if s ≤ best.1 then (s, k) else best
          some (ks.foldl step (scoreGapLoBase q k, k)).2
    else
      none
  let worstKeyArr : Array (Thunk (Option (Fin seq))) :=
    Array.ofFn (fun q => Thunk.mk (fun _ => worstKeyRaw q))
  let worstKey : Fin seq → Option (Fin seq) := fun q =>
    let t := worstKeyArr[q.1]'(by
      simp [worstKeyArr, q.isLt])
    Thunk.get t
  let dotDiffLo : Fin seq → Fin seq → Rat := fun q k =>
    match worstKey q with
    | some k' =>
        if hk : k = k' then
          let dimsQ := splitDimsQ q
          let dimsDiff := splitDimsDiffRefined q k
          let prev := inputs.prev q
          (_root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsDiff
            (fun d => qLo q d) (fun d => qHi q d)
            (fun d => kLo prev d - kHi k d)
            (fun d => kHi prev d - kLo k d)).1
        else
          dotDiffLoBase q k
    | none => dotDiffLoBase q k
  let dotDiffHi : Fin seq → Fin seq → Rat := fun q k =>
    match worstKey q with
    | some k' =>
        if hk : k = k' then
          let dimsQ := splitDimsQ q
          let dimsDiff := splitDimsDiffRefined q k
          let prev := inputs.prev q
          (_root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsDiff
            (fun d => qLo q d) (fun d => qHi q d)
            (fun d => kLo prev d - kHi k d)
            (fun d => kHi prev d - kLo k d)).2
        else
          dotDiffHiBase q k
    | none => dotDiffHiBase q k
  let scoreGapLoRaw : Fin seq → Fin seq → Rat := fun q k =>
    if masked q (inputs.prev q) then
      scoreLoPrev q - scoreHi q k
    else if masked q k then
      scoreLoPrev q - inputs.maskValue
    else if hscale : 0 ≤ inputs.scale then
      inputs.scale * dotDiffLo q k
    else
      inputs.scale * dotDiffHi q k
  let scoreGapLo : Fin seq → Fin seq → Rat :=
    Bounds.cacheBound2 scoreGapLoRaw
  let marginAt : Fin seq → Rat := fun q =>
    if hq : q ∈ inputs.active then
      let other := otherKeys q
      if h : other.Nonempty then
        other.inf' h (fun k => scoreGapLo q k)
      else
        (0 : Rat)
    else
      (0 : Rat)
  let weightBoundAtBase : Fin seq → Fin seq → Rat := fun q k =>
    if hk : k = inputs.prev q then
      (0 : Rat)
    else
      let gap := scoreGapLo q k
      if gap < 0 then
        (1 : Rat)
      else
        ratDivUp 1 (1 + gap)
  let weightBoundAt : Fin seq → Fin seq → Rat :=
    Bounds.cacheBound2 weightBoundAtBase
  let epsAtBase : Fin seq → Rat := fun q =>
    let other := otherKeys q
    let total := other.sum (fun k => weightBoundAt q k)
    min (1 : Rat) total
  let epsAt : Fin seq → Rat :=
    Bounds.cacheBoundThunk epsAtBase
  let margin : Rat :=
    if h : inputs.active.Nonempty then
      inputs.active.inf' h marginAt
    else
      (0 : Rat)
  let eps : Rat :=
    if h : inputs.active.Nonempty then
      inputs.active.sup' h epsAt
    else
      (0 : Rat)
  let dirHeadVec := dirHeadVecOfInputs inputs
  let dirHead : Fin dHead → Rat := fun d => dirHeadVec.get d
  let wvDir : Fin dModel → Rat :=
    Bounds.cacheBoundTask (fun j =>
      Linear.dotFin dHead dirHead (fun d => inputs.wv j d))
  let bDir : Rat :=
    Linear.dotFin dHead dirHead (fun d => inputs.bv d)
  let valsLo : Fin seq → Rat := fun q =>
    bDir + Bounds.dotIntervalLower (fun j => wvDir j) (lnLo q) (lnHi q)
  let valsHi : Fin seq → Rat := fun q =>
    bDir + Bounds.dotIntervalUpper (fun j => wvDir j) (lnLo q) (lnHi q)
  let univ : Finset (Fin seq) := Finset.univ
  have hnonempty : univ.Nonempty := by simp [univ]
  let lo := univ.inf' hnonempty valsLo
  let hi := univ.sup' hnonempty valsHi
  let valCert : ValueInterval seq :=
    { lo := lo
      hi := hi
      valsLo := valsLo
      valsHi := valsHi
      direction := some inputs.directionSpec }
  let cert : InductionHeadCert seq :=
    { eps := eps
      epsAt := epsAt
      weightBoundAt := weightBoundAt
      margin := margin
      active := inputs.active
      prev := inputs.prev
      values := valCert }
  exact
    { lnBounds := lnBounds
      lnLo := lnLo
      lnHi := lnHi
      lnAbsMaxTask := lnAbsMaxTask
      lnAbsMaxArr := lnAbsMaxArr
      lnAbsMax := lnAbsMax
      invStdBoundsTasks := invStdBoundsTasks
      invStdBoundsArr := invStdBoundsArr
      invStdLo := invStdLo
      invStdHi := invStdHi
      qBaseArr := qBaseArr
      qBase := qBase
      kBaseArr := kBaseArr
      kBase := kBase
      qCoeffRowTasks := qCoeffRowTasks
      qCoeffArr := qCoeffArr
      qCoeff := qCoeff
      kCoeffRowTasks := kCoeffRowTasks
      kCoeffArr := kCoeffArr
      kCoeff := kCoeff
      qLo := qLo
      qHi := qHi
      kLo := kLo
      kHi := kHi
      qAbs := qAbs
      kAbs := kAbs
      qAbsMaxArr := qAbsMaxArr
      qAbsMax := qAbsMax
      kAbsMaxArr := kAbsMaxArr
      kAbsMax := kAbsMax
      masked := masked
      splitBudgetQ := splitBudgetQ
      splitBudgetK := splitBudgetK
      splitBudgetDiffBase := splitBudgetDiffBase
      splitBudgetDiffRefined := splitBudgetDiffRefined
      splitDimsQ := splitDimsQ
      splitDimsK := splitDimsK
      splitDimsDiffCore := splitDimsDiffCore
      splitDimsDiffBase := splitDimsDiffBase
      splitDimsDiffRefined := splitDimsDiffRefined
      dotRowTasks := dotRowTasks
      dotDiffRowTasksBase := dotDiffRowTasksBase
      dotLo := dotLo
      dotHi := dotHi
      dotDiffLoBase := dotDiffLoBase
      dotDiffHiBase := dotDiffHiBase
      dotAbs := dotAbs
      scoreBaseAbs := scoreBaseAbs
      scoreLo := scoreLo
      scoreHi := scoreHi
      scoreLoPrev := scoreLoPrev
      scoreGapLoBase := scoreGapLoBase
      otherKeys := otherKeys
      worstKey := worstKey
      dotDiffLo := dotDiffLo
      dotDiffHi := dotDiffHi
      scoreGapLo := scoreGapLo
      marginAt := marginAt
      epsAt := epsAt
      weightBoundAt := weightBoundAt
      margin := margin
      eps := eps
      dirHeadVec := dirHeadVec
      dirHead := dirHead
      wvDir := wvDir
      bDir := bDir
      valsLo := valsLo
      valsHi := valsHi
      univ := univ
      lo := lo
      hi := hi
      valCert := valCert
      cert := cert }

/-- Build cached core quantities for induction-head certificates using the default split budgets. -/
def buildInductionHeadCoreCache [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    InductionHeadCoreCache seq dModel dHead :=
  buildInductionHeadCoreCacheWith defaultInductionHeadSplitConfig inputs

/-- The cached certificate is built from cache fields. -/
theorem buildInductionHeadCoreCache_cert_eq [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
  (buildInductionHeadCoreCache inputs).cert =
      { eps := (buildInductionHeadCoreCache inputs).eps
        epsAt := (buildInductionHeadCoreCache inputs).epsAt
        weightBoundAt := (buildInductionHeadCoreCache inputs).weightBoundAt
        margin := (buildInductionHeadCoreCache inputs).margin
        active := inputs.active
        prev := inputs.prev
        values := (buildInductionHeadCoreCache inputs).valCert } := by
  rfl
/-- Build induction certificates from exact head inputs (core computation). -/
def buildInductionCertFromHeadCoreWith? [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (InductionHeadCert seq) := by
  classical
  by_cases hEps : 0 < inputs.lnEps
  · by_cases hSqrt : 0 < sqrtLower inputs.lnEps
    · by_cases hmodel : dModel = 0
      · exact none
      · by_cases hactive : inputs.active.Nonempty
        · exact some (buildInductionHeadCoreCacheWith cfg inputs).cert
        · exact none
    · exact none
  · exact none

/-- Build induction certificates from exact head inputs using the default split budgets. -/
def buildInductionCertFromHeadCore? [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (InductionHeadCert seq) :=
  buildInductionCertFromHeadCoreWith? defaultInductionHeadSplitConfig inputs

/-- `buildInductionCertFromHeadCoreWith?` succeeds under the guard conditions. -/
theorem buildInductionCertFromHeadCoreWith?_eq_some [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel ≠ 0) (hactive : inputs.active.Nonempty) :
    buildInductionCertFromHeadCoreWith? cfg inputs =
      some (buildInductionHeadCoreCacheWith cfg inputs).cert := by
  classical
  simp [buildInductionCertFromHeadCoreWith?, hEps, hSqrt, hmodel, hactive]

/-- `buildInductionCertFromHeadCoreWith?` fails when `dModel = 0`. -/
theorem buildInductionCertFromHeadCoreWith?_eq_none_of_model_eq_zero
    [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel = 0) :
    buildInductionCertFromHeadCoreWith? cfg inputs = none := by
  classical
  simp [buildInductionCertFromHeadCoreWith?, hEps, hSqrt, hmodel]

/-- `buildInductionCertFromHeadCoreWith?` fails when `active` is empty. -/
theorem buildInductionCertFromHeadCoreWith?_eq_none_of_not_active
    [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel ≠ 0) (hactive : ¬inputs.active.Nonempty) :
    buildInductionCertFromHeadCoreWith? cfg inputs = none := by
  classical
  simp [buildInductionCertFromHeadCoreWith?, hEps, hSqrt, hmodel, hactive]

/-- `buildInductionCertFromHeadCoreWith?` fails when the sqrt lower bound is nonpositive. -/
theorem buildInductionCertFromHeadCoreWith?_eq_none_of_not_sqrt
    [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : ¬0 < sqrtLower inputs.lnEps) :
    buildInductionCertFromHeadCoreWith? cfg inputs = none := by
  classical
  simp [buildInductionCertFromHeadCoreWith?, hEps, hSqrt]

/-- `buildInductionCertFromHeadCoreWith?` fails when `lnEps` is nonpositive. -/
theorem buildInductionCertFromHeadCoreWith?_eq_none_of_not_eps
    [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : ¬0 < inputs.lnEps) :
    buildInductionCertFromHeadCoreWith? cfg inputs = none := by
  classical
  simp [buildInductionCertFromHeadCoreWith?, hEps]

/-- `buildInductionCertFromHeadCore?` succeeds under the guard conditions. -/
theorem buildInductionCertFromHeadCore?_eq_some [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel ≠ 0) (hactive : inputs.active.Nonempty) :
    buildInductionCertFromHeadCore? inputs =
      some (buildInductionHeadCoreCache inputs).cert := by
  classical
  simpa [buildInductionCertFromHeadCore?, buildInductionHeadCoreCache] using
    (buildInductionCertFromHeadCoreWith?_eq_some
      (cfg := defaultInductionHeadSplitConfig) (inputs := inputs)
      hEps hSqrt hmodel hactive)

/-- `buildInductionCertFromHeadCore?` fails when `dModel = 0`. -/
theorem buildInductionCertFromHeadCore?_eq_none_of_model_eq_zero [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel = 0) :
    buildInductionCertFromHeadCore? inputs = none := by
  classical
  simpa [buildInductionCertFromHeadCore?] using
    (buildInductionCertFromHeadCoreWith?_eq_none_of_model_eq_zero
      (cfg := defaultInductionHeadSplitConfig) (inputs := inputs) hEps hSqrt hmodel)

/-- `buildInductionCertFromHeadCore?` fails when `active` is empty. -/
theorem buildInductionCertFromHeadCore?_eq_none_of_not_active [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel ≠ 0) (hactive : ¬inputs.active.Nonempty) :
    buildInductionCertFromHeadCore? inputs = none := by
  classical
  simpa [buildInductionCertFromHeadCore?] using
    (buildInductionCertFromHeadCoreWith?_eq_none_of_not_active
      (cfg := defaultInductionHeadSplitConfig) (inputs := inputs) hEps hSqrt hmodel hactive)

/-- `buildInductionCertFromHeadCore?` fails when the sqrt lower bound is nonpositive. -/
theorem buildInductionCertFromHeadCore?_eq_none_of_not_sqrt [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : ¬0 < sqrtLower inputs.lnEps) :
    buildInductionCertFromHeadCore? inputs = none := by
  classical
  simpa [buildInductionCertFromHeadCore?] using
    (buildInductionCertFromHeadCoreWith?_eq_none_of_not_sqrt
      (cfg := defaultInductionHeadSplitConfig) (inputs := inputs) hEps hSqrt)

/-- `buildInductionCertFromHeadCore?` fails when `lnEps` is nonpositive. -/
theorem buildInductionCertFromHeadCore?_eq_none_of_not_eps [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : ¬0 < inputs.lnEps) :
    buildInductionCertFromHeadCore? inputs = none := by
  classical
  simpa [buildInductionCertFromHeadCore?] using
    (buildInductionCertFromHeadCoreWith?_eq_none_of_not_eps
      (cfg := defaultInductionHeadSplitConfig) (inputs := inputs) hEps)

end Sound
end Nfp
