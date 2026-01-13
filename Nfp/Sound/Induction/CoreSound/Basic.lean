-- SPDX-License-Identifier: AGPL-3.0-or-later
module

public import Nfp.Sound.Induction.Core
public import Nfp.Sound.Induction.CoreSound.Values

public section

namespace Nfp
namespace Sound
open scoped BigOperators
open Nfp.Circuit
open Nfp.Sound.Bounds
variable {seq : Nat}
set_option maxHeartbeats 5000000 in
-- The soundness proof expands many cached bounds; extra heartbeats avoid spurious timeouts.
set_option synthInstance.maxHeartbeats 200000 in
-- Instance search also touches the expanded caches; allow more room to avoid timeouts.
/-- Soundness for `buildInductionCertFromHeadCoreWith?`. -/
theorem buildInductionCertFromHeadCoreWith?_sound [NeZero seq] {dModel dHead : Nat}
      (cfg : InductionHeadSplitConfig)
      (inputs : Model.InductionHeadInputs seq dModel dHead) (c : InductionHeadCert seq)
      (hcore : buildInductionCertFromHeadCoreWith? cfg inputs = some c) :
      InductionHeadCertSound inputs c := by
    classical
    by_cases hEps : 0 < inputs.lnEps
    · by_cases hSqrt : 0 < sqrtLower inputs.lnEps
      · by_cases hmodel : dModel = 0
        · have : False := by
            have hnone :=
              buildInductionCertFromHeadCoreWith?_eq_none_of_model_eq_zero
                (cfg := cfg) (inputs := inputs) hEps hSqrt hmodel
            have hcore' :
                (none : Option (InductionHeadCert seq)) = some c := by
              exact hnone.symm.trans hcore
            cases hcore'
          exact this.elim
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
                simp [lnAbsMaxArr])
            let invStdBoundsTasks : Array (Task (Rat × Rat)) :=
              Array.ofFn (fun q : Fin seq =>
                Task.spawn (fun _ => invStdBounds inputs.lnEps (inputs.embed q)))
            let invStdBoundsArr : Array (Rat × Rat) :=
              Array.ofFn (fun q : Fin seq =>
                (invStdBoundsTasks[q.1]'(by
                  simp [invStdBoundsTasks])).get)
            let invStdLo : Fin seq → Rat := fun q =>
              (invStdBoundsArr[q.1]'(by
                simp [invStdBoundsArr])).1
            let invStdHi : Fin seq → Rat := fun q =>
              (invStdBoundsArr[q.1]'(by
                simp [invStdBoundsArr])).2
            let lnCoeff : Fin seq → Fin dModel → Rat := fun q j =>
              inputs.ln1Gamma j * (inputs.embed q j - mean (inputs.embed q))
            let invStd : Fin seq → Real := fun q =>
              (Real.sqrt ((varianceRat (inputs.embed q) : Real) + (inputs.lnEps : Real)))⁻¹
            have hmeanRat :
                ∀ q, (mean (inputs.embed q) : Real) = meanRat (inputs.embed q) := by
              intro q
              have hmu_rat : mean (inputs.embed q) = meanRat (inputs.embed q) := by
                simp [mean_def, hmodel, ratRoundDown]
              simpa [ratToReal] using congrArg ratToReal hmu_rat
            have hln_affine :
                ∀ q j,
                  lnRealOfInputs inputs q j =
                    (inputs.ln1Beta j : Real) + (lnCoeff q j : Real) * invStd q := by
              intro q j
              have hmu := hmeanRat q
              simp [lnRealOfInputs_def, Bounds.layerNormReal_def, hmodel, lnCoeff, hmu, invStd,
                add_comm, mul_assoc, -mul_eq_mul_left_iff, -mul_eq_mul_right_iff]
            have hln_fun :
                ∀ q,
                  lnRealOfInputs inputs q =
                    fun j => (inputs.ln1Beta j : Real) + (lnCoeff q j : Real) * invStd q := by
              intro q
              funext j
              exact hln_affine q j
            have hinv_bounds :
                ∀ q, (invStdLo q : Real) ≤ invStd q ∧ invStd q ≤ (invStdHi q : Real) := by
              intro q
              simpa [invStd, invStdLo, invStdHi, invStdBoundsArr, invStdBoundsTasks,
                Bounds.invStdBounds_def, Task.spawn, Array.getElem_ofFn] using
                (Bounds.invStdBounds_spec (eps := inputs.lnEps) (x := inputs.embed q)
                  hmodel hEps hSqrt)
            let qBaseArr : Array Rat :=
              Array.ofFn (fun d : Fin dHead =>
                Linear.dotFin dModel (fun j => inputs.wq j d) (fun j => inputs.ln1Beta j) +
                  inputs.bq d)
            let qBase : Fin dHead → Rat := fun d =>
              qBaseArr[d.1]'(by
                simp [qBaseArr])
            let kBaseArr : Array Rat :=
              Array.ofFn (fun d : Fin dHead =>
                Linear.dotFin dModel (fun j => inputs.wk j d) (fun j => inputs.ln1Beta j) +
                  inputs.bk d)
            let kBase : Fin dHead → Rat := fun d =>
              kBaseArr[d.1]'(by
                simp [kBaseArr])
            let coeffRowTasks :
                (Fin dModel → Fin dHead → Rat) →
                  Array (Task { row : Array Rat // row.size = dHead }) :=
              fun w =>
                Array.ofFn (fun q : Fin seq =>
                  Task.spawn (fun _ =>
                    let μ := mean (inputs.embed q)
                    let coeff : Fin dModel → Rat := fun j =>
                      inputs.ln1Gamma j * (inputs.embed q j - μ)
                    ⟨Array.ofFn (fun d : Fin dHead =>
                        Linear.dotFin dModel (fun j => w j d) coeff),
                      by simp⟩))
            let qCoeffRowTasks : Array (Task { row : Array Rat // row.size = dHead }) :=
              coeffRowTasks inputs.wq
            let qCoeffArr : Array { row : Array Rat // row.size = dHead } :=
              Array.ofFn (fun q : Fin seq =>
                (qCoeffRowTasks[q.1]'(by
                  simp [qCoeffRowTasks, coeffRowTasks])).get)
            let qCoeff : Fin seq → Fin dHead → Rat := fun q d =>
              let row := qCoeffArr[q.1]'(by
                simp [qCoeffArr])
              row.1[d.1]'(by
                simp [row.2])
            let kCoeffRowTasks : Array (Task { row : Array Rat // row.size = dHead }) :=
              coeffRowTasks inputs.wk
            let kCoeffArr : Array { row : Array Rat // row.size = dHead } :=
              Array.ofFn (fun q : Fin seq =>
                (kCoeffRowTasks[q.1]'(by
                  simp [kCoeffRowTasks, coeffRowTasks])).get)
            let kCoeff : Fin seq → Fin dHead → Rat := fun q d =>
              let row := kCoeffArr[q.1]'(by
                simp [kCoeffArr])
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
                simp [qAbsMaxArr])
            let kAbsMaxArr : Array Rat :=
              Array.ofFn (fun d : Fin dHead =>
                let univ : Finset (Fin seq) := Finset.univ
                have hnonempty : univ.Nonempty := Finset.univ_nonempty
                univ.sup' hnonempty (fun k => kAbs k d))
            let kAbsMax : Fin dHead → Rat := fun d =>
              kAbsMaxArr[d.1]'(by
                simp [kAbsMaxArr])
            let masked : Fin seq → Fin seq → Prop := fun q k =>
              inputs.maskCausal = true ∧ q < k
            let splitBudgetQ : Nat := cfg.splitBudgetQ
            let splitBudgetK : Nat := cfg.splitBudgetK
            let splitBudgetDiffBase : Nat := cfg.splitBudgetDiffBase
            let splitBudgetDiffRefined : Nat := cfg.splitBudgetDiffRefined
            let top2ByScore :
                (Fin dHead → Rat) → List (Fin dHead) → List (Fin dHead) := fun score ambig =>
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
              match ambig.foldl step (none, none) with
              | (some b1, some b2) => [b1.2, b2.2]
              | (some b1, none) => [b1.2]
              | (none, _) => []
            let finRangeHead : List (Fin dHead) := List.finRange dHead
            let finRangeSeq : List (Fin seq) := List.finRange seq
            let splitDimsQ : Fin seq → List (Fin dHead) := fun q =>
              if splitBudgetQ = 0 then
                []
              else
                let ambig :=
                  finRangeHead.filter (fun d => decide (qLo q d < 0 ∧ 0 < qHi q d))
                let score : Fin dHead → Rat := fun d => (qHi q d - qLo q d) * kAbsMax d
                let dims1 := top2ByScore score ambig
                let dims2 := top2ByScore score (ambig.filter (fun d => decide (d ∉ dims1)))
                (dims1 ++ dims2).take splitBudgetQ
            let splitDimsK : Fin seq → Fin seq → List (Fin dHead) := fun q k =>
              if splitBudgetK = 0 then
                []
              else
                let ambig :=
                  finRangeHead.filter (fun d => decide (kLo k d < 0 ∧ 0 < kHi k d))
                let score : Fin dHead → Rat := fun d => (kHi k d - kLo k d) * qAbs q d
                let dims1 := top2ByScore score ambig
                let dims2 := top2ByScore score (ambig.filter (fun d => decide (d ∉ dims1)))
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
            have hseq : (1 : Nat) ≤ seq :=
              Nat.succ_le_iff.mpr (Nat.pos_of_ne_zero (NeZero.ne seq))
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
            have hcore' : buildInductionCertFromHeadCoreWith? cfg inputs = some cert := by
              have hcore'' :
                  buildInductionCertFromHeadCoreWith? cfg inputs =
                    some (buildInductionHeadCoreCacheWith cfg inputs).cert :=
                buildInductionCertFromHeadCoreWith?_eq_some
                  (cfg := cfg) (inputs := inputs) hEps hSqrt hmodel hactive
              simpa using hcore''
            have hc : c = cert := by
              have hcert : cert = c := by
                exact Option.some.inj (hcore'.symm.trans hcore)
              simpa using hcert.symm
            subst hc
            have hln_bounds :
                ∀ q i, (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
                  lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
              intro q i
              have hln :=
                Bounds.layerNormBounds_spec (eps := inputs.lnEps)
                  (gamma := inputs.ln1Gamma) (beta := inputs.ln1Beta)
                  (x := inputs.embed q) hmodel hEps hSqrt
              simpa [lnBounds, lnLo, lnHi, lnRealOfInputs_def,
                Bounds.cacheBoundPair2_apply_left, Bounds.cacheBoundPair2_apply_right] using
                hln i
            have dotFin_cast {n : Nat} (f g : Fin n → Rat) :
                (Linear.dotFin n f g : Real) =
                  dotProduct (fun j => (f j : Real)) (fun j => (g j : Real)) := by
              simp [Linear.dotFin_def, Linear.sumFin_eq_sum_univ, dotProduct]
            have proj_bounds
                (w : Fin dModel → Fin dHead → Rat)
                (b base : Fin dHead → Rat)
                (coeff : Fin seq → Fin dHead → Rat)
                (hbase : ∀ d,
                  (base d : Real) =
                    dotProduct (fun j => (w j d : Real))
                        (fun j => (inputs.ln1Beta j : Real)) +
                      (b d : Real))
                (hcoeff : ∀ q d,
                  (coeff q d : Real) =
                    dotProduct (fun j => (w j d : Real))
                      (fun j => (lnCoeff q j : Real))) :
                ∀ q d,
                  (base d + (scaleInterval (coeff q d) (invStdLo q) (invStdHi q)).1 : Rat) ≤
                    dotProduct (fun j => (w j d : Real)) (lnRealOfInputs inputs q) +
                      (b d : Real) ∧
                  dotProduct (fun j => (w j d : Real)) (lnRealOfInputs inputs q) +
                      (b d : Real) ≤
                    (base d + (scaleInterval (coeff q d) (invStdLo q) (invStdHi q)).2 : Rat) := by
              intro q d
              have hinv : (invStdLo q : Real) ≤ invStd q ∧ invStd q ≤ (invStdHi q : Real) :=
                hinv_bounds q
              have hln_fun_q :
                  lnRealOfInputs inputs q =
                    fun j => (inputs.ln1Beta j : Real) + (lnCoeff q j : Real) * invStd q := by
                exact hln_fun q
              have hdot_add :
                  dotProduct (fun j => (w j d : Real))
                      (fun j =>
                        (inputs.ln1Beta j : Real) + (lnCoeff q j : Real) * invStd q) =
                    dotProduct (fun j => (w j d : Real))
                        (fun j => (inputs.ln1Beta j : Real)) +
                      dotProduct (fun j => (w j d : Real))
                        (fun j => (lnCoeff q j : Real) * invStd q) := by
                simpa using
                  (Nfp.Sound.Linear.dotProduct_add_right
                    (x := fun j => (w j d : Real))
                    (y := fun j => (inputs.ln1Beta j : Real))
                    (z := fun j => (lnCoeff q j : Real) * invStd q))
              have hdot_coeff :
                  dotProduct (fun j => (w j d : Real))
                      (fun j => (lnCoeff q j : Real) * invStd q) =
                    dotProduct (fun j => (w j d : Real))
                        (fun j => (lnCoeff q j : Real)) *
                      invStd q := by
                simpa using
                  (Nfp.Sound.Linear.dotProduct_mul_right
                    (x := fun j => (w j d : Real))
                    (y := fun j => (lnCoeff q j : Real))
                    (a := invStd q))
              have hreal :
                  dotProduct (fun j => (w j d : Real)) (lnRealOfInputs inputs q) +
                      (b d : Real) =
                    (base d : Real) + (coeff q d : Real) * invStd q := by
                calc
                  dotProduct (fun j => (w j d : Real)) (lnRealOfInputs inputs q) +
                      (b d : Real) =
                      dotProduct (fun j => (w j d : Real))
                          (fun j =>
                            (inputs.ln1Beta j : Real) + (lnCoeff q j : Real) * invStd q) +
                        (b d : Real) := by
                        simp [hln_fun_q]
                  _ =
                      dotProduct (fun j => (w j d : Real))
                          (fun j => (inputs.ln1Beta j : Real)) +
                        dotProduct (fun j => (w j d : Real))
                            (fun j => (lnCoeff q j : Real)) *
                          invStd q +
                        (b d : Real) := by
                        simp [hdot_add, hdot_coeff, add_assoc]
                  _ =
                      (dotProduct (fun j => (w j d : Real))
                          (fun j => (inputs.ln1Beta j : Real)) +
                        (b d : Real)) +
                        dotProduct (fun j => (w j d : Real))
                            (fun j => (lnCoeff q j : Real)) *
                          invStd q := by ac_rfl
                  _ = (base d : Real) + (coeff q d : Real) * invStd q := by
                        simp [hbase, hcoeff]
              have hscale :
                  let bounds := scaleInterval (coeff q d) (invStdLo q) (invStdHi q)
                  (bounds.1 : Real) ≤ (coeff q d : Real) * invStd q ∧
                    (coeff q d : Real) * invStd q ≤ (bounds.2 : Real) := by
                exact scaleInterval_bounds_real (x := coeff q d) (lo := invStdLo q)
                  (hi := invStdHi q) (y := invStd q) hinv.1 hinv.2
              have hlow :
                  (base d + (scaleInterval (coeff q d) (invStdLo q) (invStdHi q)).1 : Rat) ≤
                    dotProduct (fun j => (w j d : Real)) (lnRealOfInputs inputs q) +
                      (b d : Real) := by
                simpa [hreal] using add_le_add_left hscale.1 (base d : Real)
              have hhigh :
                  dotProduct (fun j => (w j d : Real)) (lnRealOfInputs inputs q) +
                      (b d : Real) ≤
                    (base d + (scaleInterval (coeff q d) (invStdLo q) (invStdHi q)).2 : Rat) := by
                simpa [hreal] using add_le_add_left hscale.2 (base d : Real)
              exact ⟨hlow, hhigh⟩
            have hq_bounds :
                ∀ q d, (qLo q d : Real) ≤ qRealOfInputs inputs q d ∧
                  qRealOfInputs inputs q d ≤ (qHi q d : Real) := by
              intro q d
              have hbase :
                  ∀ d,
                    (qBase d : Real) =
                      dotProduct (fun j => (inputs.wq j d : Real))
                          (fun j => (inputs.ln1Beta j : Real)) +
                        (inputs.bq d : Real) := by
                intro d
                simp [qBase, qBaseArr, dotFin_cast]
              have hcoeff :
                  ∀ q' d,
                    (qCoeff q' d : Real) =
                      dotProduct (fun j => (inputs.wq j d : Real))
                        (fun j => (lnCoeff q' j : Real)) := by
                intro q' d
                simpa [qCoeff, qCoeffArr, qCoeffRowTasks, coeffRowTasks, lnCoeff, Task.spawn] using
                  (dotFin_cast (f := fun j => inputs.wq j d)
                    (g := fun j =>
                      inputs.ln1Gamma j * (inputs.embed q' j - mean (inputs.embed q'))))
              have h := proj_bounds (w := inputs.wq) (b := inputs.bq) (base := qBase)
                (coeff := qCoeff) hbase hcoeff q d
              simpa [qLo, qHi, qRealOfInputs_def] using h
            have hk_bounds :
                ∀ q d, (kLo q d : Real) ≤ kRealOfInputs inputs q d ∧
                  kRealOfInputs inputs q d ≤ (kHi q d : Real) := by
              intro q d
              have hbase :
                  ∀ d,
                    (kBase d : Real) =
                      dotProduct (fun j => (inputs.wk j d : Real))
                          (fun j => (inputs.ln1Beta j : Real)) +
                        (inputs.bk d : Real) := by
                intro d
                simp [kBase, kBaseArr, dotFin_cast]
              have hcoeff :
                  ∀ q' d,
                    (kCoeff q' d : Real) =
                      dotProduct (fun j => (inputs.wk j d : Real))
                        (fun j => (lnCoeff q' j : Real)) := by
                intro q' d
                simpa [kCoeff, kCoeffArr, kCoeffRowTasks, coeffRowTasks, lnCoeff, Task.spawn] using
                  (dotFin_cast (f := fun j => inputs.wk j d)
                    (g := fun j =>
                      inputs.ln1Gamma j * (inputs.embed q' j - mean (inputs.embed q'))))
              have h := proj_bounds (w := inputs.wk) (b := inputs.bk) (base := kBase)
                (coeff := kCoeff) hbase hcoeff q d
              simpa [kLo, kHi, kRealOfInputs_def] using h
            let scoresReal := scoresRealOfInputs inputs
            have scoresReal_eq_base_of_not_masked :
                ∀ q k, ¬ masked q k →
                  scoresReal q k =
                    (inputs.scale : Real) *
                      dotProduct (fun d => qRealOfInputs inputs q d)
                        (fun d => kRealOfInputs inputs k d) := by
              intro q k hnot
              by_cases hcausal : inputs.maskCausal
              · have hnot_lt : ¬ q < k := by
                  intro hlt
                  exact hnot ⟨hcausal, hlt⟩
                have hle : k ≤ q := le_of_not_gt hnot_lt
                simp [scoresReal, scoresRealOfInputs_def, hcausal, hle]
              · simp [scoresReal, scoresRealOfInputs_def, hcausal]
            have scoresReal_eq_masked :
                ∀ q k, masked q k → scoresReal q k = (inputs.maskValue : Real) := by
              intro q k hmask
              have hmask' : inputs.maskCausal = true ∧ q < k := by
                simpa [masked] using hmask
              have hle : ¬ k ≤ q := not_le_of_gt hmask'.2
              simp [scoresReal, scoresRealOfInputs_def, hmask'.1, hle]
            have hscore_bounds :
                ∀ q k, (scoreLo q k : Real) ≤ scoresReal q k ∧
                  scoresReal q k ≤ (scoreHi q k : Real) := by
              intro q k
              let base :=
                (inputs.scale : Real) *
                  dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d)
              have hdot_bounds (hnot : ¬ masked q k) :
                  (dotLo q k : Real) ≤
                    dotProduct (fun d => qRealOfInputs inputs q d)
                      (fun d => kRealOfInputs inputs k d) ∧
                  dotProduct (fun d => qRealOfInputs inputs q d)
                      (fun d => kRealOfInputs inputs k d) ≤ (dotHi q k : Real) := by
                have hq := hq_bounds q
                have hk := hk_bounds k
                have hlo1 : ∀ d, (qLo q d : Real) ≤ qRealOfInputs inputs q d := fun d =>
                  (hq d).1
                have hhi1 : ∀ d, qRealOfInputs inputs q d ≤ (qHi q d : Real) := fun d =>
                  (hq d).2
                have hlo2 : ∀ d, (kLo k d : Real) ≤ kRealOfInputs inputs k d := fun d =>
                  (hk d).1
                have hhi2 : ∀ d, kRealOfInputs inputs k d ≤ (kHi k d : Real) := fun d =>
                  (hk d).2
                have hspec :=
                  _root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth_spec_real
                    (dims1 := splitDimsQ q) (dims2 := splitDimsK q k)
                    (lo1 := fun d => qLo q d) (hi1 := fun d => qHi q d)
                    (lo2 := fun d => kLo k d) (hi2 := fun d => kHi k d)
                    (x := fun d => qRealOfInputs inputs q d)
                    (y := fun d => kRealOfInputs inputs k d)
                    hlo1 hhi1 hlo2 hhi2
                have hlow' :
                    (dotLo q k : Real) ≤
                      dotProduct (fun d => qRealOfInputs inputs q d)
                        (fun d => kRealOfInputs inputs k d) := by
                  simpa [dotLo, dotRowTasks, Task.spawn, Array.getElem_ofFn, hnot]
                    using hspec.1
                have hhigh' :
                    dotProduct (fun d => qRealOfInputs inputs q d)
                        (fun d => kRealOfInputs inputs k d) ≤ (dotHi q k : Real) := by
                  simpa [dotHi, dotRowTasks, Task.spawn, Array.getElem_ofFn, hnot]
                    using hspec.2
                exact ⟨hlow', hhigh'⟩
              have hscore_base_bounds (hnot : ¬ masked q k) :
                  (scoreLo q k : Real) ≤ base ∧ base ≤ (scoreHi q k : Real) := by
                by_cases hscale : 0 ≤ inputs.scale
                · have hscale_real : 0 ≤ (inputs.scale : Real) :=
                    ratToReal_nonneg_of_nonneg hscale
                  have hdot := hdot_bounds hnot
                  have hlow := mul_le_mul_of_nonneg_left hdot.1 hscale_real
                  have hhigh := mul_le_mul_of_nonneg_left hdot.2 hscale_real
                  constructor
                  · simpa [scoreLo, masked, hnot, hscale, base] using hlow
                  · simpa [scoreHi, masked, hnot, hscale, base] using hhigh
                · have hscale_nonpos : inputs.scale ≤ 0 :=
                    le_of_lt (lt_of_not_ge hscale)
                  have hscale_real : (inputs.scale : Real) ≤ 0 :=
                    (ratToReal_nonpos_iff (x := inputs.scale)).2 hscale_nonpos
                  have hdot := hdot_bounds hnot
                  have hlow := mul_le_mul_of_nonpos_left hdot.2 hscale_real
                  have hhigh := mul_le_mul_of_nonpos_left hdot.1 hscale_real
                  constructor
                  · simpa [scoreLo, masked, hnot, hscale, base] using hlow
                  · simpa [scoreHi, masked, hnot, hscale, base] using hhigh
              by_cases hcausal : inputs.maskCausal
              · by_cases hle : k ≤ q
                · have hnot : ¬ q < k := not_lt_of_ge hle
                  have hnot_masked : ¬ masked q k := fun hmk => hnot hmk.2
                  have hscore_eq : scoresReal q k = base :=
                    scoresReal_eq_base_of_not_masked q k hnot_masked
                  have hbase := hscore_base_bounds hnot_masked
                  constructor
                  · simpa [hscore_eq] using hbase.1
                  · simpa [hscore_eq] using hbase.2
                · have hlt : q < k := lt_of_not_ge hle
                  have hmask : masked q k := ⟨hcausal, hlt⟩
                  have hscore : scoresReal q k = (inputs.maskValue : Real) :=
                    scoresReal_eq_masked q k hmask
                  constructor
                  · simp [hscore, scoreLo, hmask]
                  · simp [hscore, scoreHi, hmask]
              · have hnot_masked : ¬ masked q k := by
                  simp [masked, hcausal]
                have hscore_eq : scoresReal q k = base :=
                  scoresReal_eq_base_of_not_masked q k hnot_masked
                have hbase := hscore_base_bounds hnot_masked
                constructor
                · simpa [hscore_eq] using hbase.1
                · simpa [hscore_eq] using hbase.2
            have hdot_diff_bounds :
                ∀ q, q ∈ inputs.active → ∀ k, ¬ masked q k →
                  (dotDiffLo q k : Real) ≤
                      dotProduct (fun d => qRealOfInputs inputs q d)
                        (fun d => kRealOfInputs inputs (inputs.prev q) d -
                          kRealOfInputs inputs k d) ∧
                    dotProduct (fun d => qRealOfInputs inputs q d)
                        (fun d => kRealOfInputs inputs (inputs.prev q) d -
                          kRealOfInputs inputs k d) ≤ (dotDiffHi q k : Real) := by
              intro q hq k hmask
              have hq_bounds' := hq_bounds q
              have hkprev := hk_bounds (inputs.prev q)
              have hk := hk_bounds k
              have hlo1 : ∀ d, (qLo q d : Real) ≤ qRealOfInputs inputs q d := fun d =>
                (hq_bounds' d).1
              have hhi1 : ∀ d, qRealOfInputs inputs q d ≤ (qHi q d : Real) := fun d =>
                (hq_bounds' d).2
              have hlo2 :
                  ∀ d,
                    (kLo (inputs.prev q) d - kHi k d : Rat) ≤
                      (kRealOfInputs inputs (inputs.prev q) d - kRealOfInputs inputs k d) := by
                intro d
                have hprev_lo := (hkprev d).1
                have hk_hi := (hk d).2
                have h := sub_le_sub hprev_lo hk_hi
                simpa [ratToReal_sub] using h
              have hhi2 :
                  ∀ d,
                    (kRealOfInputs inputs (inputs.prev q) d - kRealOfInputs inputs k d) ≤
                      (kHi (inputs.prev q) d - kLo k d : Rat) := by
                intro d
                have hprev_hi := (hkprev d).2
                have hk_lo := (hk d).1
                have h := sub_le_sub hprev_hi hk_lo
                simpa [ratToReal_sub] using h
              have hspec (dimsDiff : List (Fin dHead)) :=
                _root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth_spec_real
                  (dims1 := splitDimsQ q) (dims2 := dimsDiff)
                  (lo1 := fun d => qLo q d) (hi1 := fun d => qHi q d)
                  (lo2 := fun d => kLo (inputs.prev q) d - kHi k d)
                  (hi2 := fun d => kHi (inputs.prev q) d - kLo k d)
                  (x := fun d => qRealOfInputs inputs q d)
                  (y := fun d =>
                    kRealOfInputs inputs (inputs.prev q) d - kRealOfInputs inputs k d)
                  hlo1 hhi1 hlo2 hhi2
              have hspecBase := hspec (splitDimsDiffBase q k)
              have hspecRef := hspec (splitDimsDiffRefined q k)
              have hspecBase_bounds :
                  (dotDiffLoBase q k : Real) ≤
                      dotProduct (fun d => qRealOfInputs inputs q d)
                        (fun d => kRealOfInputs inputs (inputs.prev q) d -
                          kRealOfInputs inputs k d) ∧
                    dotProduct (fun d => qRealOfInputs inputs q d)
                        (fun d => kRealOfInputs inputs (inputs.prev q) d -
                          kRealOfInputs inputs k d) ≤ (dotDiffHiBase q k : Real) := by
                refine ⟨?_, ?_⟩
                · simpa [dotDiffLoBase, dotDiffRowTasksBase, hq, hmask, Task.spawn,
                    Array.getElem_ofFn] using hspecBase.1
                · simpa [dotDiffHiBase, dotDiffRowTasksBase, hq, hmask, Task.spawn,
                    Array.getElem_ofFn] using hspecBase.2
              cases hkey : worstKey q with
              | none =>
                  simpa [dotDiffLo, dotDiffHi, hkey] using hspecBase_bounds
              | some k' =>
                  by_cases hk : k = k'
                  · have hlow' :
                      (dotDiffLo q k : Real) ≤
                          dotProduct (fun d => qRealOfInputs inputs q d)
                            (fun d => kRealOfInputs inputs (inputs.prev q) d -
                              kRealOfInputs inputs k d) := by
                      simpa [dotDiffLo, hkey, hk] using hspecRef.1
                    have hhigh' :
                        dotProduct (fun d => qRealOfInputs inputs q d)
                              (fun d => kRealOfInputs inputs (inputs.prev q) d -
                                kRealOfInputs inputs k d) ≤ (dotDiffHi q k : Real) := by
                      simpa [dotDiffHi, hkey, hk] using hspecRef.2
                    exact ⟨hlow', hhigh'⟩
                  · have hlow' :
                      (dotDiffLo q k : Real) ≤
                          dotProduct (fun d => qRealOfInputs inputs q d)
                            (fun d => kRealOfInputs inputs (inputs.prev q) d -
                              kRealOfInputs inputs k d) := by
                      simpa [dotDiffLo, hkey, hk] using hspecBase_bounds.1
                    have hhigh' :
                        dotProduct (fun d => qRealOfInputs inputs q d)
                              (fun d => kRealOfInputs inputs (inputs.prev q) d -
                                kRealOfInputs inputs k d) ≤ (dotDiffHi q k : Real) := by
                      simpa [dotDiffHi, hkey, hk] using hspecBase_bounds.2
                    exact ⟨hlow', hhigh'⟩
            have hmarginAt_le :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  marginAt q ≤ scoreGapLo q k := by
              intro q hq k hk
              have hmem : k ∈ otherKeys q := by simp [otherKeys, hk]
              have hnonempty : (otherKeys q).Nonempty := ⟨k, hmem⟩
              have hle :
                  (otherKeys q).inf' hnonempty (fun k => scoreGapLo q k) ≤ scoreGapLo q k := by
                exact (Finset.inf'_le_iff (s := otherKeys q) (H := hnonempty)
                  (f := fun k => scoreGapLo q k) (a := scoreGapLo q k)).2
                  ⟨k, hmem, le_rfl⟩
              simpa [marginAt, hq, hnonempty] using hle
            have hscore_gap_real_at :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  scoresReal q k + (scoreGapLo q k : Real) ≤ scoresReal q (inputs.prev q) := by
              intro q hq k hk
              by_cases hprevmask : masked q (inputs.prev q)
              · have hscore_hi : scoresReal q k ≤ (scoreHi q k : Real) :=
                  (hscore_bounds q k).2
                have hscore_prev : (scoreLoPrev q : Real) ≤ scoresReal q (inputs.prev q) := by
                  have hprev_bounds := hscore_bounds q (inputs.prev q)
                  simpa [scoreLoPrev] using hprev_bounds.1
                have hsum_le' :
                    (scoreLoPrev q : Real) - (scoreHi q k : Real) + scoresReal q k ≤
                      (scoreLoPrev q : Real) := by
                  have hsub :
                      (scoreLoPrev q : Real) - (scoreHi q k : Real) ≤
                        (scoreLoPrev q : Real) - scoresReal q k :=
                    sub_le_sub_left hscore_hi (scoreLoPrev q : Real)
                  calc
                    (scoreLoPrev q : Real) - (scoreHi q k : Real) + scoresReal q k
                        ≤ (scoreLoPrev q : Real) - scoresReal q k + scoresReal q k := by
                          simpa [add_comm, add_left_comm, add_assoc] using
                            (add_le_add_left hsub (scoresReal q k))
                    _ = (scoreLoPrev q : Real) := by
                      simp [sub_add_cancel]
                calc
                  scoresReal q k + (scoreGapLo q k : Real)
                      = (scoreLoPrev q : Real) - (scoreHi q k : Real) + scoresReal q k := by
                        simp [scoreGapLo, scoreGapLoRaw, Bounds.cacheBound2_apply,
                          hprevmask, add_comm]
                  _ ≤ (scoreLoPrev q : Real) := hsum_le'
                  _ ≤ scoresReal q (inputs.prev q) := hscore_prev
              · by_cases hmask : masked q k
                · have hscore_prev : (scoreLoPrev q : Real) ≤ scoresReal q (inputs.prev q) := by
                    have hprev_bounds := hscore_bounds q (inputs.prev q)
                    simpa [scoreLoPrev] using hprev_bounds.1
                  have hscore_k : scoresReal q k = (inputs.maskValue : Real) :=
                    scoresReal_eq_masked q k hmask
                  calc
                    scoresReal q k + (scoreGapLo q k : Real)
                        = (inputs.maskValue : Real) + (scoreLoPrev q : Real) -
                            (inputs.maskValue : Real) := by
                          simp [scoreGapLo, scoreGapLoRaw, Bounds.cacheBound2_apply,
                            hprevmask, hmask, hscore_k]
                    _ = (scoreLoPrev q : Real) := by
                          simp [add_sub_cancel_left]
                    _ ≤ scoresReal q (inputs.prev q) := hscore_prev
                · have hdiff := hdot_diff_bounds q hq k hmask
                  have hgap_le :
                      (scoreGapLo q k : Real) ≤
                        (inputs.scale : Real) *
                          dotProduct (fun d => qRealOfInputs inputs q d)
                            (fun d => kRealOfInputs inputs (inputs.prev q) d -
                              kRealOfInputs inputs k d) := by
                    by_cases hscale : 0 ≤ inputs.scale
                    · have hscale_real : 0 ≤ (inputs.scale : Real) :=
                        ratToReal_nonneg_of_nonneg hscale
                      have hle := mul_le_mul_of_nonneg_left hdiff.1 hscale_real
                      simpa [scoreGapLo, scoreGapLoRaw, Bounds.cacheBound2_apply,
                        hprevmask, hmask, hscale] using hle
                    · have hscale_nonpos : inputs.scale ≤ 0 :=
                        le_of_lt (lt_of_not_ge hscale)
                      have hscale_real : (inputs.scale : Real) ≤ 0 :=
                        (ratToReal_nonpos_iff (x := inputs.scale)).2 hscale_nonpos
                      have hle := mul_le_mul_of_nonpos_left hdiff.2 hscale_real
                      simpa [scoreGapLo, scoreGapLoRaw, Bounds.cacheBound2_apply,
                        hprevmask, hmask, hscale] using hle
                  have hscore_prev :
                      scoresReal q (inputs.prev q) =
                        (inputs.scale : Real) *
                          dotProduct (fun d => qRealOfInputs inputs q d)
                            (fun d => kRealOfInputs inputs (inputs.prev q) d) := by
                    simpa using
                      (scoresReal_eq_base_of_not_masked q (inputs.prev q) hprevmask)
                  have hscore_k :
                      scoresReal q k =
                        (inputs.scale : Real) *
                          dotProduct (fun d => qRealOfInputs inputs q d)
                            (fun d => kRealOfInputs inputs k d) := by
                    simpa using (scoresReal_eq_base_of_not_masked q k hmask)
                  have hdot_sub :
                      dotProduct (fun d => qRealOfInputs inputs q d)
                          (fun d => kRealOfInputs inputs (inputs.prev q) d -
                            kRealOfInputs inputs k d) =
                        dotProduct (fun d => qRealOfInputs inputs q d)
                            (fun d => kRealOfInputs inputs (inputs.prev q) d) -
                          dotProduct (fun d => qRealOfInputs inputs q d)
                            (fun d => kRealOfInputs inputs k d) := by
                    classical
                    simpa using
                      (Nfp.Sound.Linear.dotProduct_sub_right
                        (x := fun d => qRealOfInputs inputs q d)
                        (y := fun d => kRealOfInputs inputs (inputs.prev q) d)
                        (z := fun d => kRealOfInputs inputs k d))
                  have hscore_diff :
                      scoresReal q (inputs.prev q) - scoresReal q k =
                        (inputs.scale : Real) *
                          dotProduct (fun d => qRealOfInputs inputs q d)
                            (fun d => kRealOfInputs inputs (inputs.prev q) d -
                              kRealOfInputs inputs k d) := by
                    calc
                      scoresReal q (inputs.prev q) - scoresReal q k
                          =
                            (inputs.scale : Real) *
                              dotProduct (fun d => qRealOfInputs inputs q d)
                                (fun d => kRealOfInputs inputs (inputs.prev q) d) -
                              (inputs.scale : Real) *
                                dotProduct (fun d => qRealOfInputs inputs q d)
                                  (fun d => kRealOfInputs inputs k d) := by
                              simp [hscore_prev, hscore_k]
                      _ =
                          (inputs.scale : Real) *
                            (dotProduct (fun d => qRealOfInputs inputs q d)
                                (fun d => kRealOfInputs inputs (inputs.prev q) d) -
                              dotProduct (fun d => qRealOfInputs inputs q d)
                                (fun d => kRealOfInputs inputs k d)) := by
                              simp [mul_sub]
                      _ =
                          (inputs.scale : Real) *
                            dotProduct (fun d => qRealOfInputs inputs q d)
                              (fun d => kRealOfInputs inputs (inputs.prev q) d -
                                kRealOfInputs inputs k d) := by
                              simp [hdot_sub]
                  have hgap_le' :
                      (scoreGapLo q k : Real) ≤
                        scoresReal q (inputs.prev q) - scoresReal q k := by
                    simpa [hscore_diff] using hgap_le
                  have hgap_add :=
                    add_le_add_right hgap_le' (scoresReal q k)
                  have hgap_add' :
                      scoresReal q k + (scoreGapLo q k : Real) ≤
                        scoresReal q (inputs.prev q) := by
                    have hcancel :
                        scoresReal q k + (scoresReal q (inputs.prev q) - scoresReal q k) =
                          scoresReal q (inputs.prev q) := by
                      calc
                        scoresReal q k + (scoresReal q (inputs.prev q) - scoresReal q k)
                            =
                              scoresReal q k + scoresReal q (inputs.prev q) -
                                scoresReal q k := by
                                  symm
                                  exact add_sub_assoc (scoresReal q k)
                                    (scoresReal q (inputs.prev q)) (scoresReal q k)
                        _ = scoresReal q (inputs.prev q) := by
                          simp [add_sub_cancel_left]
                    calc
                      scoresReal q k + (scoreGapLo q k : Real)
                          ≤ scoresReal q k + (scoresReal q (inputs.prev q) - scoresReal q k) :=
                        hgap_add
                      _ = scoresReal q (inputs.prev q) := hcancel
                  exact hgap_add'
            let weights : Fin seq → Fin seq → Real := fun q k =>
              Circuit.softmax (scoresReal q) k
            let others : Fin seq → Finset (Fin seq) := fun q =>
              (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
            have hscore_margin_real_at :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  scoresReal q k + (marginAt q : Real) ≤ scoresReal q (inputs.prev q) := by
              intro q hq k hk
              have hmargin_le : marginAt q ≤ scoreGapLo q k :=
                hmarginAt_le q hq k hk
              have hmargin_le_real : (marginAt q : Real) ≤ (scoreGapLo q k : Real) :=
                ratToReal_le_of_le hmargin_le
              have hscore_gap := hscore_gap_real_at q hq k hk
              have hstep := add_le_add_right hmargin_le_real (scoresReal q k)
              have hstep' :
                  scoresReal q k + (marginAt q : Real) ≤
                    scoresReal q k + (scoreGapLo q k : Real) := by
                exact hstep
              exact hstep'.trans hscore_gap
            have hscore_margin_real :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  scoresReal q k + (margin : Real) ≤ scoresReal q (inputs.prev q) := by
              intro q hq k hk
              have hmargin_le : margin ≤ marginAt q := by
                have hmem : q ∈ inputs.active := hq
                have hnonempty : inputs.active.Nonempty := hactive
                have hle :=
                  (Finset.inf'_le_iff (s := inputs.active) (H := hnonempty)
                    (f := marginAt) (a := marginAt q)).2 ⟨q, hmem, le_rfl⟩
                simpa [margin, hnonempty] using hle
              have hmargin_le_real : (margin : Real) ≤ (marginAt q : Real) :=
                ratToReal_le_of_le hmargin_le
              have hscore := hscore_margin_real_at q hq k hk
              have hscore' :
                  (marginAt q : Real) + scoresReal q k ≤ scoresReal q (inputs.prev q) := by
                simpa [add_comm] using hscore
              have hstep := add_le_add_right hmargin_le_real (scoresReal q k)
              have hstep' :
                  scoresReal q k + (margin : Real) ≤ (marginAt q : Real) + scoresReal q k := by
                calc
                  scoresReal q k + (margin : Real) ≤ scoresReal q k + (marginAt q : Real) := hstep
                  _ = (marginAt q : Real) + scoresReal q k := by
                    simp [add_comm]
              exact hstep'.trans hscore'
            have hweightBoundAt :
                ∀ q k, k ≠ inputs.prev q →
                  weightBoundAt q k =
                    if scoreGapLo q k < 0 then
                      (1 : Rat)
                    else
                      ratDivUp 1 (1 + scoreGapLo q k) := by
              intro q k hk
              simpa [weightBoundAt, weightBoundAtBase, hk] using
                (Bounds.cacheBound2_apply (f := weightBoundAtBase) q k)
            have hepsAt :
                ∀ q, epsAt q =
                  min (1 : Rat)
                    ((otherKeys q).sum (fun k =>
                      if scoreGapLo q k < 0 then
                        (1 : Rat)
                      else
                        ratDivUp 1 (1 + scoreGapLo q k))) := by
              intro q
              have hsum :
                  (otherKeys q).sum (fun k => weightBoundAt q k) =
                    (otherKeys q).sum (fun k =>
                      if scoreGapLo q k < 0 then
                        (1 : Rat)
                      else
                        ratDivUp 1 (1 + scoreGapLo q k)) := by
                refine Finset.sum_congr rfl ?_
                intro k hk
                have hk' : k ≠ inputs.prev q := (Finset.mem_erase.mp hk).1
                simp [hweightBoundAt q k hk']
              simpa [epsAt, epsAtBase, hsum] using
                (Bounds.cacheBoundThunk_apply (f := epsAtBase) q)
            have oneHot_bounds_at :
                ∀ q, q ∈ inputs.active →
                  Layers.OneHotApproxBoundsOnActive (Val := Real) (epsAt q : Real)
                    (fun q' => q' = q) inputs.prev weights := by
              intro q hq
              exact
                Sound.oneHot_bounds_at_of_scoreGapLo
                  (active := inputs.active)
                  (prev := inputs.prev)
                  (scoresReal := scoresReal)
                  (scoreGapLo := scoreGapLo)
                  (epsAt := epsAt)
                  (hepsAt := hepsAt)
                  (hscore_gap_real_at := hscore_gap_real_at)
                  q hq
            have weight_bounds_at :
                ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
                  weights q k ≤ (weightBoundAt q k : Real) := by
              intro q hq k hk
              exact
                Sound.weight_bound_at_of_scoreGapLo
                  (active := inputs.active)
                  (prev := inputs.prev)
                  (scoresReal := scoresReal)
                  (scoreGapLo := scoreGapLo)
                  (weightBoundAt := weightBoundAt)
                  (hweightBoundAt := hweightBoundAt)
                  (hscore_gap_real_at := hscore_gap_real_at)
                  q hq k hk
            have hepsAt_le_eps :
                ∀ q, q ∈ inputs.active → epsAt q ≤ eps := by
              intro q hq
              have hle :
                  epsAt q ≤ inputs.active.sup' hactive epsAt := by
                exact
                  (Finset.le_sup'_iff (s := inputs.active) (H := hactive)
                    (f := epsAt) (a := epsAt q)).2 ⟨q, hq, le_rfl⟩
              simpa [eps, hactive] using hle
            have hepsAt_le_eps_real :
                ∀ q, q ∈ inputs.active → (epsAt q : Real) ≤ (eps : Real) := by
              intro q hq
              exact ratToReal_le_of_le (hepsAt_le_eps q hq)
            have hsoftmax_bounds :
                Layers.SoftmaxMarginBoundsOn (Val := Real) (eps : Real) (margin : Real)
                  (fun q => q ∈ inputs.active) inputs.prev scoresReal weights := by
              classical
              refine
                { score_margin := ?_
                  nonneg := ?_
                  sum_one := ?_
                  prev_large := ?_
                  other_le := ?_ }
              · intro q hq k hk
                exact hscore_margin_real q hq k hk
              · intro q _ k
                simpa [weights] using
                  (Circuit.softmax_nonneg (scores := scoresReal q) k)
              · intro q _
                simpa [weights] using
                  (Circuit.softmax_sum_one (scores := scoresReal q))
              · intro q hq
                have honehot := oneHot_bounds_at q hq
                have hprev := honehot.prev_large q rfl
                have hle :
                    weights q (inputs.prev q) + (epsAt q : Real) ≤
                      weights q (inputs.prev q) + (eps : Real) := by
                  simpa [add_comm] using
                    (add_le_add_right (hepsAt_le_eps_real q hq) (weights q (inputs.prev q)))
                exact hprev.trans hle
              · intro q hq k hk
                have honehot := oneHot_bounds_at q hq
                have hother := honehot.other_le q rfl k hk
                exact hother.trans (hepsAt_le_eps_real q hq)
            have hdirHead :
                dirHead = fun d => (dirHeadVecOfInputs inputs).get d := by
              simp [dirHead, dirHeadVec]
            have hwvDir :
                ∀ j, wvDir j = Linear.dotFin dHead dirHead (fun d => inputs.wv j d) := by
              intro j
              simp [wvDir, Bounds.cacheBoundTask_apply]
            have hbDir :
                bDir = Linear.dotFin dHead dirHead (fun d => inputs.bv d) := by
              rfl
            have hdir_wv :
                ∀ j, (wvDir j : Real) = ∑ d, (dirHead d : Real) * (inputs.wv j d : Real) :=
              wvDir_real_eq_sum inputs dirHead wvDir hwvDir
            have hdir_bv :
                (bDir : Real) = ∑ d, (dirHead d : Real) * (inputs.bv d : Real) :=
              bDir_real_eq_sum inputs dirHead bDir hbDir
            have hvals_eq :
                ∀ k,
                  valsRealOfInputs inputs k =
                    dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) +
                      (bDir : Real) :=
              valsReal_eq_of_dir inputs dirHead hdirHead wvDir bDir hdir_wv hdir_bv
            have hvals_bounds_at :
                ∀ k,
                  (valCert.valsLo k : Real) ≤ valsRealOfInputs inputs k ∧
                    valsRealOfInputs inputs k ≤ (valCert.valsHi k : Real) := by
              intro k
              have hln := hln_bounds k
              have hlo : ∀ j, (lnLo k j : Real) ≤ lnRealOfInputs inputs k j := fun j =>
                (hln j).1
              have hhi : ∀ j, lnRealOfInputs inputs k j ≤ (lnHi k j : Real) := fun j =>
                (hln j).2
              have hlow' :
                  (Bounds.dotIntervalLower (fun j => wvDir j) (lnLo k) (lnHi k) : Real) +
                      (bDir : Real) ≤
                    dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) +
                      (bDir : Real) := by
                simpa using
                  (Bounds.dotIntervalLower_le_dotProduct_real_add
                    (v := fun j => wvDir j)
                    (lo := lnLo k) (hi := lnHi k)
                    (x := lnRealOfInputs inputs k) (b := (bDir : Real)) hlo hhi)
              have hhigh' :
                  dotProduct (fun j => (wvDir j : Real)) (lnRealOfInputs inputs k) +
                      (bDir : Real) ≤
                    (Bounds.dotIntervalUpper (fun j => wvDir j) (lnLo k) (lnHi k) : Real) +
                      (bDir : Real) := by
                simpa using
                  (Bounds.dotProduct_le_dotIntervalUpper_real_add
                    (v := fun j => wvDir j)
                    (lo := lnLo k) (hi := lnHi k)
                    (x := lnRealOfInputs inputs k) (b := (bDir : Real)) hlo hhi)
              have hlow :
                  (valCert.valsLo k : Real) ≤ valsRealOfInputs inputs k := by
                simpa [valCert, valsLo, hvals_eq k, ratToReal_add, add_comm, add_left_comm,
                  add_assoc] using hlow'
              have hhigh :
                  valsRealOfInputs inputs k ≤ (valCert.valsHi k : Real) := by
                simpa [valCert, valsHi, hvals_eq k, ratToReal_add, add_comm, add_left_comm,
                  add_assoc] using hhigh'
              exact ⟨hlow, hhigh⟩
            have hvals_bounds :
                ValueIntervalBounds (vals := valsRealOfInputs inputs) valCert := by
              refine
                { lo_le_hi := ?_
                  lo_le_valsLo := ?_
                  vals_bounds := ?_
                  valsHi_le_hi := ?_ }
              · rcases (Finset.univ_nonempty : univ.Nonempty) with ⟨k0, hk0⟩
                have hmem0 : k0 ∈ univ := hk0
                have hlo : (valCert.lo : Real) ≤ (valCert.valsLo k0 : Real) := by
                  have hloRat : valCert.lo ≤ valCert.valsLo k0 := by
                    change lo ≤ valsLo k0
                    dsimp [lo]
                    refine (Finset.inf'_le_iff (s := univ) (H := hnonempty)
                      (f := valsLo) (a := valsLo k0)).2 ⟨k0, hmem0, le_rfl⟩
                  exact ratToReal_le_of_le hloRat
                have hvals :
                    (valCert.valsLo k0 : Real) ≤ valsRealOfInputs inputs k0 ∧
                      valsRealOfInputs inputs k0 ≤ (valCert.valsHi k0 : Real) := by
                  exact hvals_bounds_at k0
                have hhi : (valCert.valsHi k0 : Real) ≤ (valCert.hi : Real) := by
                  have hhiRat : valCert.valsHi k0 ≤ valCert.hi := by
                    change valsHi k0 ≤ hi
                    dsimp [hi]
                    refine (Finset.le_sup'_iff (s := univ) (H := hnonempty)
                      (f := valsHi) (a := valsHi k0)).2 ⟨k0, ⟨hmem0, le_rfl⟩⟩
                  exact ratToReal_le_of_le hhiRat
                have hreal :
                    (valCert.lo : Real) ≤ (valCert.hi : Real) :=
                  le_trans hlo (le_trans hvals.1 (le_trans hvals.2 hhi))
                exact (ratToReal_le_iff (x := valCert.lo) (y := valCert.hi)).1 hreal
              · intro k
                have hloRat : valCert.lo ≤ valCert.valsLo k := by
                  change lo ≤ valsLo k
                  dsimp [lo]
                  refine (Finset.inf'_le_iff (s := univ) (H := hnonempty)
                    (f := valsLo) (a := valsLo k)).2 ⟨k, by simp [univ], le_rfl⟩
                exact ratToReal_le_of_le hloRat
              · intro k
                exact hvals_bounds_at k
              · intro k
                have hhiRat : valCert.valsHi k ≤ valCert.hi := by
                  change valsHi k ≤ hi
                  dsimp [hi]
                  refine (Finset.le_sup'_iff (s := univ) (H := hnonempty)
                    (f := valsHi) (a := valsHi k)).2 ⟨k, by simp [univ], le_rfl⟩
                exact ratToReal_le_of_le hhiRat
            exact
              { softmax_bounds := hsoftmax_bounds
                oneHot_bounds_at := oneHot_bounds_at
                weight_bounds_at := weight_bounds_at
                value_bounds := hvals_bounds }
          · have : False := by
              have hnone :=
                buildInductionCertFromHeadCoreWith?_eq_none_of_not_active
                  (cfg := cfg) (inputs := inputs) hEps hSqrt hmodel hactive
              have hcore' :
                  (none : Option (InductionHeadCert seq)) = some c := by
                exact hnone.symm.trans hcore
              cases hcore'
            exact this.elim
      · have : False := by
          have hnone :=
            buildInductionCertFromHeadCoreWith?_eq_none_of_not_sqrt
              (cfg := cfg) (inputs := inputs) hEps hSqrt
          have hcore' :
              (none : Option (InductionHeadCert seq)) = some c := by
            exact hnone.symm.trans hcore
          cases hcore'
        exact this.elim
    · have : False := by
        have hnone :=
          buildInductionCertFromHeadCoreWith?_eq_none_of_not_eps
            (cfg := cfg) (inputs := inputs) hEps
        have hcore' :
            (none : Option (InductionHeadCert seq)) = some c := by
          exact hnone.symm.trans hcore
        cases hcore'
      exact this.elim

/-- Soundness for `buildInductionCertFromHeadCore?`. -/
theorem buildInductionCertFromHeadCore?_sound
      [NeZero seq] {dModel dHead : Nat}
      (inputs : Model.InductionHeadInputs seq dModel dHead) (c : InductionHeadCert seq)
      (hcore : buildInductionCertFromHeadCore? inputs = some c) :
      InductionHeadCertSound inputs c := by
  have hcore' :
      buildInductionCertFromHeadCoreWith? defaultInductionHeadSplitConfig inputs = some c := by
    simpa [buildInductionCertFromHeadCore?_def] using hcore
  exact
    buildInductionCertFromHeadCoreWith?_sound
      (cfg := defaultInductionHeadSplitConfig) inputs c hcore'
end Sound
end Nfp
