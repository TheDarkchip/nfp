-- SPDX-License-Identifier: AGPL-3.0-or-later
module

import all Nfp.Sound.Induction.Core.Basic
public import Nfp.Sound.Induction.Core
public import Nfp.Sound.Induction.CoreSound.Values

public section

namespace Nfp
namespace Sound
open scoped BigOperators
open Nfp.Circuit
open Nfp.Sound.Bounds
variable {seq : Nat}
/-- Bounds for cached projections and scores from `buildInductionHeadCoreCacheWith`. -/
theorem buildInductionHeadCoreCacheWith_bounds
    [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel ≠ 0) :
    (∀ q d,
      ((buildInductionHeadCoreCacheWith cfg inputs).qLo q d : Real) ≤
        qRealOfInputs inputs q d ∧
      qRealOfInputs inputs q d ≤
        ((buildInductionHeadCoreCacheWith cfg inputs).qHi q d : Real)) ∧
    (∀ q d,
      ((buildInductionHeadCoreCacheWith cfg inputs).kLo q d : Real) ≤
        kRealOfInputs inputs q d ∧
      kRealOfInputs inputs q d ≤
        ((buildInductionHeadCoreCacheWith cfg inputs).kHi q d : Real)) ∧
    (∀ q k,
      ((buildInductionHeadCoreCacheWith cfg inputs).scoreLo q k : Real) ≤
        scoresRealOfInputs inputs q k ∧
      scoresRealOfInputs inputs q k ≤
        ((buildInductionHeadCoreCacheWith cfg inputs).scoreHi q k : Real)) := by
  classical
  set cache := buildInductionHeadCoreCacheWith cfg inputs with hcache
  have dotFin_cast {n : Nat} (f g : Fin n → Rat) :
      (Linear.dotFin n f g : Real) =
        dotProduct (fun j => (f j : Real)) (fun j => (g j : Real)) := by
    simp [Linear.dotFin_def, Linear.sumFin_eq_sum_univ, dotProduct]
  let lnCoeff : Fin seq → Fin dModel → Rat := fun q j =>
    inputs.ln1Gamma j * (inputs.embed q j - mean (inputs.embed q))
  let invStd : Fin seq → Real := fun q =>
    (Real.sqrt ((varianceRat (inputs.embed q) : Real) + (inputs.lnEps : Real)))⁻¹
  have hmeanRat :
      ∀ q, (mean (inputs.embed q) : Real) = meanRat (inputs.embed q) := by
    intro q
    have hmu_rat : mean (inputs.embed q) = meanRat (inputs.embed q) := by
      simp [mean_def, hmodel, ratRoundDown_def]
    simpa [ratToReal_def] using congrArg ratToReal hmu_rat
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
  have hinv_bounds :
      ∀ q, (invStdLo q : Real) ≤ invStd q ∧ invStd q ≤ (invStdHi q : Real) := by
    intro q
    simpa [invStd, invStdLo, invStdHi, invStdBoundsArr, invStdBoundsTasks,
      Bounds.invStdBounds_def, Task.spawn, Array.getElem_ofFn] using
      (Bounds.invStdBounds_spec (eps := inputs.lnEps) (x := inputs.embed q)
        hmodel hEps hSqrt)
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
  let finRangeHead : List (Fin dHead) := List.finRange dHead
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
  have dotLo_eq :
      ∀ q k,
        dotLo q k =
          if masked q k then
            (0 : Rat)
          else
            let dimsQ := splitDimsQ q
            let dimsK := splitDimsK q k
            (_root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsK
              (fun d => qLo q d) (fun d => qHi q d)
              (fun d => kLo k d) (fun d => kHi k d)).1 := by
    intro q k
    classical
    by_cases hmk : masked q k
    · simp [dotLo, dotRowTasks, Task.spawn, Array.getElem_ofFn, hmk]
    · simp [dotLo, dotRowTasks, Task.spawn, Array.getElem_ofFn, hmk]
  have dotHi_eq :
      ∀ q k,
        dotHi q k =
          if masked q k then
            (0 : Rat)
          else
            let dimsQ := splitDimsQ q
            let dimsK := splitDimsK q k
            (_root_.Nfp.Sound.Bounds.dotIntervalLowerUpper2SignSplitBoth dimsQ dimsK
              (fun d => qLo q d) (fun d => qHi q d)
              (fun d => kLo k d) (fun d => kHi k d)).2 := by
    intro q k
    classical
    by_cases hmk : masked q k
    · simp [dotHi, dotRowTasks, Task.spawn, Array.getElem_ofFn, hmk]
    · simp [dotHi, dotRowTasks, Task.spawn, Array.getElem_ofFn, hmk]
  have hq_bounds_local :
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
  have hk_bounds_local :
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
  have hscore_bounds_local :
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
      have hq := hq_bounds_local q
      have hk := hk_bounds_local k
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
        simpa [dotLo_eq, hnot] using hspec.1
      have hhigh' :
          dotProduct (fun d => qRealOfInputs inputs q d)
              (fun d => kRealOfInputs inputs k d) ≤ (dotHi q k : Real) := by
        simpa [dotHi_eq, hnot] using hspec.2
      exact ⟨hlow', hhigh'⟩
    have hscore_base_bounds (hnot : ¬ masked q k) :
        (scoreLo q k : Real) ≤ base ∧ base ≤ (scoreHi q k : Real) := by
      by_cases hscale : 0 ≤ inputs.scale
      · have hscale_real : 0 ≤ (inputs.scale : Real) := by
          simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg hscale
        have hdot := hdot_bounds hnot
        have hlow := mul_le_mul_of_nonneg_left hdot.1 hscale_real
        have hhigh := mul_le_mul_of_nonneg_left hdot.2 hscale_real
        constructor
        · simpa [scoreLo, masked, hnot, hscale, base] using hlow
        · simpa [scoreHi, masked, hnot, hscale, base] using hhigh
      · have hscale_nonpos : inputs.scale ≤ 0 :=
          le_of_lt (lt_of_not_ge hscale)
        have hscale_real : (inputs.scale : Real) ≤ 0 := by
          simpa [ratToReal_def] using
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
        · simp [hscore, scoreLo, hmask, masked]
        · simp [hscore, scoreHi, hmask, masked]
    · have hnot_masked : ¬ masked q k := by
        simp [masked, hcausal]
      have hscore_eq : scoresReal q k = base :=
        scoresReal_eq_base_of_not_masked q k hnot_masked
      have hbase := hscore_base_bounds hnot_masked
      constructor
      · simpa [hscore_eq] using hbase.1
      · simpa [hscore_eq] using hbase.2
  have hlocal :
      (∀ q d, (qLo q d : Real) ≤ qRealOfInputs inputs q d ∧
        qRealOfInputs inputs q d ≤ (qHi q d : Real)) ∧
      (∀ q d, (kLo q d : Real) ≤ kRealOfInputs inputs q d ∧
        kRealOfInputs inputs q d ≤ (kHi q d : Real)) ∧
      (∀ q k, (scoreLo q k : Real) ≤ scoresReal q k ∧
        scoresReal q k ≤ (scoreHi q k : Real)) := by
    exact ⟨hq_bounds_local, hk_bounds_local, hscore_bounds_local⟩
  simpa (config := { zeta := false }) [hcache, buildInductionHeadCoreCacheWith] using
    hlocal

/-- Query bounds for `buildInductionHeadCoreCacheWith`. -/
theorem buildInductionHeadCoreCacheWith_q_bounds
    [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel ≠ 0) :
    ∀ q d,
      ((buildInductionHeadCoreCacheWith cfg inputs).qLo q d : Real) ≤
        qRealOfInputs inputs q d ∧
      qRealOfInputs inputs q d ≤
        ((buildInductionHeadCoreCacheWith cfg inputs).qHi q d : Real) := by
  exact (buildInductionHeadCoreCacheWith_bounds cfg inputs hEps hSqrt hmodel).1

/-- Key bounds for `buildInductionHeadCoreCacheWith`. -/
theorem buildInductionHeadCoreCacheWith_k_bounds
    [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel ≠ 0) :
    ∀ q d,
      ((buildInductionHeadCoreCacheWith cfg inputs).kLo q d : Real) ≤
        kRealOfInputs inputs q d ∧
      kRealOfInputs inputs q d ≤
        ((buildInductionHeadCoreCacheWith cfg inputs).kHi q d : Real) := by
  exact (buildInductionHeadCoreCacheWith_bounds cfg inputs hEps hSqrt hmodel).2.1

/-- Score bounds for `buildInductionHeadCoreCacheWith`. -/
theorem buildInductionHeadCoreCacheWith_score_bounds
    [NeZero seq] {dModel dHead : Nat}
    (cfg : InductionHeadSplitConfig)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps)
    (hmodel : dModel ≠ 0) :
    ∀ q k,
      ((buildInductionHeadCoreCacheWith cfg inputs).scoreLo q k : Real) ≤
        scoresRealOfInputs inputs q k ∧
      scoresRealOfInputs inputs q k ≤
        ((buildInductionHeadCoreCacheWith cfg inputs).scoreHi q k : Real) := by
  exact (buildInductionHeadCoreCacheWith_bounds cfg inputs hEps hSqrt hmodel).2.2


end Sound
end Nfp
