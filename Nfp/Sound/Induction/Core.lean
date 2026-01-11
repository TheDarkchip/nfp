-- SPDX-License-Identifier: AGPL-3.0-or-later
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Nfp.Core.Basic
import Mathlib.Data.Finset.Lattice.Fold
import Nfp.Circuit.Cert.ResidualInterval
import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange
import Nfp.Sound.Bounds.Attention
import Nfp.Sound.Bounds.Cache
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Bounds.LayerNorm.InvStd
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Induction.CoreDefs
import Nfp.Sound.Induction.OneHot
import Nfp.Sound.Linear.FinFold
/-! Sound builders for induction certificates; recompute bounds inside Lean from exact inputs and
derive softmax tolerances from score margins rather than trusting external weight dumps. -/
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
/-- Build induction certificates from exact head inputs (core computation). -/
def buildInductionCertFromHeadCore? [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (InductionHeadCert seq) := by
  classical
  by_cases hEps : 0 < inputs.lnEps
  · by_cases hSqrt : 0 < sqrtLower inputs.lnEps
    · by_cases hmodel : dModel = 0
      · exact none
      · by_cases hactive : inputs.active.Nonempty
        · let lnBounds := Bounds.cacheBoundPair2 (fun q =>
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
          let splitBudgetQ : Nat := 2
          let splitBudgetK : Nat := 2
          let splitBudgetDiffBase : Nat := 0
          let splitBudgetDiffRefined : Nat := 12
          let splitDimsQ : Fin seq → List (Fin dHead) := fun q =>
            let ambig :=
              (List.finRange dHead).filter (fun d => decide (qLo q d < 0 ∧ 0 < qHi q d))
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
            let ambig :=
              (List.finRange dHead).filter (fun d => decide (kLo k d < 0 ∧ 0 < kHi k d))
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
            let prev := inputs.prev q
            let diffLo : Fin dHead → Rat := fun d => kLo prev d - kHi k d
            let diffHi : Fin dHead → Rat := fun d => kHi prev d - kLo k d
            let ambig :=
              (List.finRange dHead).filter (fun d => decide (diffLo d < 0 ∧ 0 < diffHi d))
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
          let scoreGapLoBase : Fin seq → Fin seq → Rat := fun q k =>
            if masked q (inputs.prev q) then
              scoreLoPrev q - scoreHi q k
            else if masked q k then
              scoreLoPrev q - inputs.maskValue
            else if hscale : 0 ≤ inputs.scale then
              inputs.scale * dotDiffLoBase q k
            else
              inputs.scale * dotDiffHiBase q k
          let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
            (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
          let worstKey : Fin seq → Option (Fin seq) := fun q =>
            if hq : q ∈ inputs.active then
              let ks := (List.finRange seq).filter (fun k => decide (k ≠ inputs.prev q))
              match ks with
              | [] => none
              | k :: ks =>
                  let step (best : Rat × Fin seq) (k : Fin seq) :=
                    let s := scoreGapLoBase q k
                    if s ≤ best.1 then (s, k) else best
                  some (ks.foldl step (scoreGapLoBase q k, k)).2
            else
              none
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
          let scoreGapLo : Fin seq → Fin seq → Rat := fun q k =>
            if masked q (inputs.prev q) then
              scoreLoPrev q - scoreHi q k
            else if masked q k then
              scoreLoPrev q - inputs.maskValue
            else if hscale : 0 ≤ inputs.scale then
              inputs.scale * dotDiffLo q k
            else
              inputs.scale * dotDiffHi q k
          let marginAt : Fin seq → Rat := fun q =>
            if hq : q ∈ inputs.active then
              let other := otherKeys q
              if h : other.Nonempty then
                other.inf' h (fun k => scoreGapLo q k)
              else
                (0 : Rat)
            else
              (0 : Rat)
          let epsAt : Fin seq → Rat := fun q =>
            if marginAt q < 0 then
              (1 : Rat)
            else
              ratDivUp (seq - 1) (1 + marginAt q)
          let margin : Rat :=
            if h : inputs.active.Nonempty then
              inputs.active.inf' h marginAt
            else
              (0 : Rat)
          let eps : Rat :=
            if margin < 0 then
              (1 : Rat)
            else
              ratDivUp (seq - 1) (1 + margin)
          let dirHeadVec := dirHeadVecOfInputs inputs
          let dirHead : Fin dHead → Rat := fun d => dirHeadVec.get d
          let wvDir : Fin dModel → Rat :=
            Bounds.cacheBoundTask (fun j =>
              Linear.dotFin dHead dirHead (fun d => inputs.wv j d))
          let bDir : Rat :=
            Linear.dotFin dHead dirHead (fun d => inputs.bv d)
          let valsAbs : Fin seq → Rat := fun q =>
            Linear.sumFin dModel (fun j => |wvDir j|) * lnAbsMax q
          let valsLo : Fin seq → Rat := fun q => bDir - valsAbs q
          let valsHi : Fin seq → Rat := fun q => bDir + valsAbs q
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
              margin := margin
              active := inputs.active
              prev := inputs.prev
              values := valCert }
          exact some cert
        · exact none
    · exact none
  · exact none

end Sound
end Nfp
