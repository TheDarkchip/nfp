-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Aesop
public import Mathlib.Data.List.MinMax
public import Mathlib.Data.Vector.Basic
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.Sound.Bounds.MatrixNorm.Interval
public import Nfp.Sound.Induction.HeadOutput
public import Nfp.Sound.Induction.Refine

/-!
Logit-diff bounds derived from induction certificates.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Circuit

variable {seq : Nat}

section Direction

variable {seq dModel dHead : Nat}

/-- Direction projection of a single-key head output. -/
theorem direction_dot_headValue_eq_valsReal
    (inputs : Model.InductionHeadInputs seq dModel dHead) (k : Fin seq) :
    dotProduct (fun i => (inputs.direction i : Real))
        (fun i => headValueRealOfInputs inputs k i) =
      valsRealOfInputs inputs k := by
  classical
  let dir : Fin dModel → Real := fun i => (inputs.direction i : Real)
  let v : Fin dHead → Real := fun d => vRealOfInputs inputs k d
  have hswap :
      ∑ i, dir i * ∑ d, (inputs.wo i d : Real) * v d =
        ∑ d, (∑ i, dir i * (inputs.wo i d : Real)) * v d := by
    calc
      ∑ i, dir i * ∑ d, (inputs.wo i d : Real) * v d
          = ∑ i, ∑ d, dir i * ((inputs.wo i d : Real) * v d) := by
            simp [Finset.mul_sum]
      _ = ∑ d, ∑ i, dir i * ((inputs.wo i d : Real) * v d) := by
            simpa using
              (Finset.sum_comm (s := (Finset.univ : Finset (Fin dModel)))
                (t := (Finset.univ : Finset (Fin dHead)))
                (f := fun i d => dir i * ((inputs.wo i d : Real) * v d)))
      _ = ∑ d, (∑ i, dir i * (inputs.wo i d : Real)) * v d := by
            refine Finset.sum_congr rfl ?_
            intro d _
            simp [mul_assoc, Finset.sum_mul]
  have hdirHead :
      ∀ d, ((dirHeadVecOfInputs inputs).get d : Real) =
        ∑ i, (inputs.wo i d : Real) * (inputs.direction i : Real) := by
    intro d
    calc
      ((dirHeadVecOfInputs inputs).get d : Real) =
          ratToReal (Linear.dotFin dModel (fun i => inputs.wo i d)
            (fun i => inputs.direction i)) := by
          simp [dirHeadVecOfInputs_get, ratToReal_def]
      _ =
          ratToReal (dotProduct (fun i => inputs.wo i d) (fun i => inputs.direction i)) := by
          simp [Linear.dotFin_eq_dotProduct]
      _ =
          ∑ i, ratToReal (inputs.wo i d * inputs.direction i) := by
          simp [dotProduct, Linear.ratToReal_sum_univ]
      _ =
          ∑ i, (inputs.wo i d : Real) * (inputs.direction i : Real) := by
          simp [ratToReal_def]
  calc
    dotProduct dir (fun i => headValueRealOfInputs inputs k i)
        = ∑ i, dir i *
            ∑ d, (inputs.wo i d : Real) * v d := by
          simp [dir, v, headValueRealOfInputs_def, dotProduct]
    _ = ∑ d, (∑ i, dir i * (inputs.wo i d : Real)) * v d := by
          simp [hswap]
    _ = ∑ d, ((dirHeadVecOfInputs inputs).get d : Real) * v d := by
          refine Finset.sum_congr rfl ?_
          intro d _
          have hdir :
              ∑ i, dir i * (inputs.wo i d : Real) =
                ((dirHeadVecOfInputs inputs).get d : Real) := by
            calc
              ∑ i, dir i * (inputs.wo i d : Real)
                  = ∑ i, (inputs.wo i d : Real) * (inputs.direction i : Real) := by
                    simp [dir, mul_comm]
              _ = ((dirHeadVecOfInputs inputs).get d : Real) := by
                    simpa using (hdirHead d).symm
          simp [hdir]
    _ = valsRealOfInputs inputs k := by
          simp [valsRealOfInputs_def, v, dotProduct]

end Direction

section LogitDiffLowerBound

variable {seq dModel dHead : Nat}

/-- Real-valued logit-diff contribution for a query. -/
noncomputable def headLogitDiff (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) : Real :=
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scoresRealOfInputs inputs q) k
  dotProduct (weights q) (valsRealOfInputs inputs)

/-- Lower bound computed from the per-key lower bounds in an induction certificate. -/
def logitDiffLowerBoundFromCert (c : InductionHeadCert seq) : Option Rat :=
  let epsAt := Bounds.cacheBoundTask c.epsAt
  let valsLo := Bounds.cacheBoundTask c.values.valsLo
  let loAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (c.prev q)
    if h : others.Nonempty then
      others.inf' h valsLo
    else
      c.values.lo
  Circuit.logitDiffLowerBoundAtLoAt c.active c.prev epsAt loAt valsLo

/-- Lower bound computed from per-key weight bounds in an induction certificate. -/
def logitDiffLowerBoundFromCertWeighted (c : InductionHeadCert seq) : Option Rat :=
  let valsLo := Bounds.cacheBoundTask c.values.valsLo
  Circuit.logitDiffLowerBoundWeightedAt c.active c.prev c.weightBoundAt valsLo

/-- Cached eps and value lower bounds for logit-diff computations. -/
structure LogitDiffCache (seq : Nat) where
  /-- Per-query eps bounds. -/
  epsAt : Fin seq → Rat
  /-- Per-key value lower bounds. -/
  valsLo : Fin seq → Rat

/-- Build a shared cache for logit-diff computations from a certificate. -/
def logitDiffCache (c : InductionHeadCert seq) : LogitDiffCache seq :=
  { epsAt := Bounds.cacheBoundTask c.epsAt
    valsLo := Bounds.cacheBoundTask c.values.valsLo }

/-- Unfolding lemma for `logitDiffCache`. -/
theorem logitDiffCache_def (c : InductionHeadCert seq) :
    logitDiffCache c =
      { epsAt := Bounds.cacheBoundTask c.epsAt
        valsLo := Bounds.cacheBoundTask c.values.valsLo } := by
  rfl

/-- Unweighted logit-diff lower bound from a shared cache. -/
def logitDiffLowerBoundFromCache (c : InductionHeadCert seq) (cache : LogitDiffCache seq) :
    Option Rat :=
  let epsArr : Array Rat := Array.ofFn cache.epsAt
  let valsLoArr : Array Rat := Array.ofFn cache.valsLo
  let epsAt : Fin seq → Rat := fun q =>
    epsArr[q.1]'(by
      simp [epsArr, q.isLt])
  let valsLo : Fin seq → Rat := fun q =>
    valsLoArr[q.1]'(by
      simp [valsLoArr, q.isLt])
  let loAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (c.prev q)
    if h : others.Nonempty then
      others.inf' h valsLo
    else
      c.values.lo
  Circuit.logitDiffLowerBoundAtLoAt c.active c.prev epsAt loAt valsLo

/-- Query attaining the cached unweighted logit-diff lower bound, if any. -/
def logitDiffLowerBoundArgminFromCache (c : InductionHeadCert seq) (cache : LogitDiffCache seq) :
    Option (Fin seq) :=
  let epsArr : Array Rat := Array.ofFn cache.epsAt
  let valsLoArr : Array Rat := Array.ofFn cache.valsLo
  let epsAt : Fin seq → Rat := fun q =>
    epsArr[q.1]'(by
      simp [epsArr, q.isLt])
  let valsLo : Fin seq → Rat := fun q =>
    valsLoArr[q.1]'(by
      simp [valsLoArr, q.isLt])
  let loAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (c.prev q)
    if h : others.Nonempty then
      others.inf' h valsLo
    else
      c.values.lo
  let f : Fin seq → Rat := fun q =>
    let delta := valsLo (c.prev q) - loAt q
    valsLo (c.prev q) - epsAt q * max (0 : Rat) delta
  let qs := (List.finRange seq).filter (fun q => decide (q ∈ c.active))
  List.argmin f qs

/-- Unfolding lemma for `logitDiffLowerBoundArgminFromCache`. -/
theorem logitDiffLowerBoundArgminFromCache_def
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq) :
    logitDiffLowerBoundArgminFromCache c cache =
      let epsArr : Array Rat := Array.ofFn cache.epsAt
      let valsLoArr : Array Rat := Array.ofFn cache.valsLo
      let epsAt : Fin seq → Rat := fun q =>
        epsArr[q.1]'(by
          simp [epsArr, q.isLt])
      let valsLo : Fin seq → Rat := fun q =>
        valsLoArr[q.1]'(by
          simp [valsLoArr, q.isLt])
      let loAt : Fin seq → Rat := fun q =>
        let others : Finset (Fin seq) :=
          (Finset.univ : Finset (Fin seq)).erase (c.prev q)
        if h : others.Nonempty then
          others.inf' h valsLo
        else
          c.values.lo
      let f : Fin seq → Rat := fun q =>
        let delta := valsLo (c.prev q) - loAt q
        valsLo (c.prev q) - epsAt q * max (0 : Rat) delta
      let qs := (List.finRange seq).filter (fun q => decide (q ∈ c.active))
      List.argmin f qs := by
  rfl

/-- Unweighted logit-diff lower bound from a shared cache and custom `epsAt`. -/
def logitDiffLowerBoundFromCacheWithEps (c : InductionHeadCert seq) (cache : LogitDiffCache seq)
    (epsAtCustom : Fin seq → Rat) : Option Rat :=
  let epsArr : Array Rat := Array.ofFn epsAtCustom
  let valsLoArr : Array Rat := Array.ofFn cache.valsLo
  let epsAt : Fin seq → Rat := fun q =>
    epsArr[q.1]'(by
      simp [epsArr, q.isLt])
  let valsLo : Fin seq → Rat := fun q =>
    valsLoArr[q.1]'(by
      simp [valsLoArr, q.isLt])
  let loAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (c.prev q)
    if h : others.Nonempty then
      others.inf' h valsLo
    else
      c.values.lo
  Circuit.logitDiffLowerBoundAtLoAt c.active c.prev epsAt loAt valsLo

/-- Unfold `logitDiffLowerBoundFromCache` as the custom-eps variant. -/
theorem logitDiffLowerBoundFromCache_eq_withEps
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq) :
    logitDiffLowerBoundFromCache c cache =
      logitDiffLowerBoundFromCacheWithEps c cache cache.epsAt := by
  rfl

/-- Unfolding lemma for `logitDiffLowerBoundFromCacheWithEps`. -/
theorem logitDiffLowerBoundFromCacheWithEps_def
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq)
    (epsAtCustom : Fin seq → Rat) :
    logitDiffLowerBoundFromCacheWithEps c cache epsAtCustom =
      let epsArr : Array Rat := Array.ofFn epsAtCustom
      let valsLoArr : Array Rat := Array.ofFn cache.valsLo
      let epsAt : Fin seq → Rat := fun q =>
        epsArr[q.1]'(by
          simp [epsArr, q.isLt])
      let valsLo : Fin seq → Rat := fun q =>
        valsLoArr[q.1]'(by
          simp [valsLoArr, q.isLt])
      let loAt : Fin seq → Rat := fun q =>
        let others : Finset (Fin seq) :=
          (Finset.univ : Finset (Fin seq)).erase (c.prev q)
        if h : others.Nonempty then
          others.inf' h valsLo
        else
          c.values.lo
      Circuit.logitDiffLowerBoundAtLoAt c.active c.prev epsAt loAt valsLo := by
  rfl

/-- Refined unweighted logit-diff lower bound using an overlayed `epsAt`. -/
def logitDiffLowerBoundRefinedFromCache
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (core : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq)
    (spec : InductionHeadRefineSpec seq) : Option Rat :=
  let weightBoundAt := weightBoundAtOverlay inputs core spec
  let epsAt := epsAtOverlay core weightBoundAt
  logitDiffLowerBoundFromCacheWithEps c cache epsAt

/-- Unfolding lemma for `logitDiffLowerBoundRefinedFromCache`. -/
theorem logitDiffLowerBoundRefinedFromCache_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (core : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq)
    (spec : InductionHeadRefineSpec seq) :
    logitDiffLowerBoundRefinedFromCache inputs core c cache spec =
      let weightBoundAt := weightBoundAtOverlay inputs core spec
      let epsAt := epsAtOverlay core weightBoundAt
      logitDiffLowerBoundFromCacheWithEps c cache epsAt := by
  rfl

/-- Refine-on-demand unweighted logit-diff bound using a supplied refinement spec. -/
def logitDiffLowerBoundRefineOnDemandWithSpec
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (core : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq)
    (spec : InductionHeadRefineSpec seq) : Option Rat :=
  match logitDiffLowerBoundFromCache c cache with
  | none => none
  | some lb0 =>
      if lb0 ≤ 0 then
        match logitDiffLowerBoundRefinedFromCache inputs core c cache spec with
        | some lb1 => some (max lb0 lb1)
        | none => some lb0
      else
        some lb0

/-- Unfolding lemma for `logitDiffLowerBoundRefineOnDemandWithSpec`. -/
theorem logitDiffLowerBoundRefineOnDemandWithSpec_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (core : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq)
    (spec : InductionHeadRefineSpec seq) :
    logitDiffLowerBoundRefineOnDemandWithSpec inputs core c cache spec =
      match logitDiffLowerBoundFromCache c cache with
      | none => none
      | some lb0 =>
          if lb0 ≤ 0 then
            match logitDiffLowerBoundRefinedFromCache inputs core c cache spec with
            | some lb1 => some (max lb0 lb1)
            | none => some lb0
          else
            some lb0 := by
  rfl

/-- Refine-on-demand unweighted logit-diff bound, refining only the argmin query. -/
def logitDiffLowerBoundRefineOnDemand
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (core : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq) : Option Rat :=
  match logitDiffLowerBoundFromCache c cache with
  | none => none
  | some lb0 =>
      if lb0 ≤ 0 then
        match logitDiffLowerBoundArgminFromCache c cache with
        | none => some lb0
        | some q0 =>
            let refineBudget := max 1 core.splitBudgetDiffRefined
            let spec := refineSpecForQueryWithWeightOnes inputs core q0 refineBudget
            match logitDiffLowerBoundRefinedFromCache inputs core c cache spec with
            | none => some lb0
            | some lb1 =>
                let lb01 := max lb0 lb1
                if lb01 ≤ 0 then
                  let refineBudget' := refineBudgetBoost refineBudget
                  let spec' := refineSpecForQueryWithWeightOnes inputs core q0 refineBudget'
                  match logitDiffLowerBoundRefinedFromCache inputs core c cache spec' with
                  | some lb2 => some (max lb01 lb2)
                  | none => some lb01
                else
                  some lb01
      else
        some lb0

/-- Unfolding lemma for `logitDiffLowerBoundRefineOnDemand`. -/
theorem logitDiffLowerBoundRefineOnDemand_def
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (core : InductionHeadCoreCache seq dModel dHead)
    (c : InductionHeadCert seq) (cache : LogitDiffCache seq) :
    logitDiffLowerBoundRefineOnDemand inputs core c cache =
      match logitDiffLowerBoundFromCache c cache with
      | none => none
      | some lb0 =>
          if lb0 ≤ 0 then
            match logitDiffLowerBoundArgminFromCache c cache with
            | none => some lb0
            | some q0 =>
                let refineBudget := max 1 core.splitBudgetDiffRefined
                let spec := refineSpecForQueryWithWeightOnes inputs core q0 refineBudget
                match logitDiffLowerBoundRefinedFromCache inputs core c cache spec with
                | none => some lb0
                | some lb1 =>
                    let lb01 := max lb0 lb1
                    if lb01 ≤ 0 then
                      let refineBudget' := refineBudgetBoost refineBudget
                      let spec' := refineSpecForQueryWithWeightOnes inputs core q0 refineBudget'
                      match logitDiffLowerBoundRefinedFromCache inputs core c cache spec' with
                      | some lb2 => some (max lb01 lb2)
                      | none => some lb01
                    else
                      some lb01
          else
            some lb0 := by
  rfl

/-- Weighted logit-diff lower bound from a shared cache. -/
def logitDiffLowerBoundWeightedFromCache (c : InductionHeadCert seq) (cache : LogitDiffCache seq) :
    Option Rat :=
  let valsLoArr : Array Rat := Array.ofFn cache.valsLo
  let valsLo : Fin seq → Rat := fun k =>
    valsLoArr[k.1]'(by
      simp [valsLoArr, k.isLt])
  let weightRows : Array (Array Rat) :=
    Array.ofFn (fun q : Fin seq => Array.ofFn (fun k : Fin seq => c.weightBoundAt q k))
  let weightBoundAt : Fin seq → Fin seq → Rat := fun q k =>
    let row := weightRows[q.1]'(by
      simp [weightRows, q.isLt])
    row[k.1]'(by
      have hrow : row.size = seq := by
        simp [row, weightRows]
      simp [hrow, k.isLt])
  let gapBase : Fin seq → Rat := fun q =>
    let valsLoPrev := valsLo (c.prev q)
    Linear.sumFin seq (fun k =>
      let diff := valsLoPrev - valsLo k
      weightBoundAt q k * max (0 : Rat) diff)
  let gap : Fin seq → Rat := Bounds.cacheBoundTask gapBase
  if h : c.active.Nonempty then
    let f : Fin seq → Rat := fun q => valsLo (c.prev q) - gap q
    some (c.active.inf' h f)
  else
    none

/-- Debug payload for the unweighted logit-diff lower bound. -/
structure LogitDiffAtLoDebug (seq : Nat) where
  /-- Query attaining the bound, if found. -/
  q : Fin seq
  /-- Previous index for the query. -/
  prev : Fin seq
  /-- Per-query eps bound. -/
  eps : Rat
  /-- Lower bound for the previous value. -/
  valsPrevLo : Rat
  /-- Global lower value bound. -/
  lo : Rat
  /-- Per-query lower bound for other values. -/
  loAt : Rat
  /-- `valsPrevLo - loAt`. -/
  valsPrevLoMinusLoAt : Rat
  /-- `eps * max 0 (valsPrevLo - loAt)`. -/
  gap : Rat
  /-- Lower bound reported by `logitDiffLowerBoundFromCache`. -/
  lbAtQ : Rat

/-- Attempt to recover a query that attains the unweighted logit-diff bound. -/
def logitDiffLowerBoundAtLoDebug (c : InductionHeadCert seq) (cache : LogitDiffCache seq) :
    Option {d : LogitDiffAtLoDebug seq //
      logitDiffLowerBoundFromCache c cache = some d.lbAtQ} :=
  let epsArr : Array Rat := Array.ofFn cache.epsAt
  let valsLoArr : Array Rat := Array.ofFn cache.valsLo
  let epsAt : Fin seq → Rat := fun q =>
    epsArr[q.1]'(by
      simp [epsArr, q.isLt])
  let valsLo : Fin seq → Rat := fun q =>
    valsLoArr[q.1]'(by
      simp [valsLoArr, q.isLt])
  let loAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (c.prev q)
    if h : others.Nonempty then
      others.inf' h valsLo
    else
      c.values.lo
  let f : Fin seq → Rat := fun q =>
    let valsPrevLo := valsLo (c.prev q)
    let delta := valsPrevLo - loAt q
    valsPrevLo - epsAt q * max (0 : Rat) delta
  let best? : Option (Fin seq × Rat) :=
    Linear.foldlFin seq
      (fun acc q =>
        if hq : q ∈ c.active then
          let val := f q
          match acc with
          | none => some (q, val)
          | some (qBest, best) =>
              if val ≤ best then
                some (q, val)
              else
                some (qBest, best)
        else
          acc)
      none
  match logitDiffLowerBoundFromCache c cache with
  | none => none
  | some lb =>
      match best? with
      | none => none
      | some (q, _) =>
          let prev := c.prev q
          let valsPrevLo := valsLo prev
          let loAtQ := loAt q
          let delta := valsPrevLo - loAtQ
          let gap := epsAt q * max (0 : Rat) delta
          let d : LogitDiffAtLoDebug seq :=
            { q := q
              prev := prev
              eps := epsAt q
              valsPrevLo := valsPrevLo
              lo := c.values.lo
              loAt := loAtQ
              valsPrevLoMinusLoAt := delta
              gap := gap
              lbAtQ := lb }
          have h' : some lb = some d.lbAtQ := by
            simp [d]
          some ⟨d, h'⟩

/-- `logitDiffLowerBoundFromCache` matches the cached default computation. -/
theorem logitDiffLowerBoundFromCache_eq (c : InductionHeadCert seq) :
    logitDiffLowerBoundFromCache c (logitDiffCache c) = logitDiffLowerBoundFromCert c := by
  classical
  unfold logitDiffLowerBoundFromCache logitDiffLowerBoundFromCert logitDiffCache
  have heps : Bounds.cacheBoundTask c.epsAt = c.epsAt := by
    funext k
    simp [Bounds.cacheBoundTask_apply]
  have hvals : Bounds.cacheBoundTask c.values.valsLo = c.values.valsLo := by
    funext k
    simp [Bounds.cacheBoundTask_apply]
  simp [heps, hvals, Bounds.cacheBoundTask_apply]

/-- `logitDiffLowerBoundWeightedFromCache` matches the cached default computation. -/
theorem logitDiffLowerBoundWeightedFromCache_eq (c : InductionHeadCert seq) :
    logitDiffLowerBoundWeightedFromCache c (logitDiffCache c) =
      logitDiffLowerBoundFromCertWeighted c := by
  classical
  unfold logitDiffLowerBoundWeightedFromCache logitDiffLowerBoundFromCertWeighted logitDiffCache
  have hvals : Bounds.cacheBoundTask c.values.valsLo = c.values.valsLo := by
    funext k
    simp [Bounds.cacheBoundTask_apply]
  simp [hvals, Bounds.cacheBoundTask_apply, logitDiffLowerBoundWeightedAt_def,
    Linear.sumFin_eq_sum_univ]

/-- Best available logit-diff lower bound from an induction certificate. -/
def logitDiffLowerBoundFromCertBest (c : InductionHeadCert seq) : Option Rat :=
  match logitDiffLowerBoundFromCert c, logitDiffLowerBoundFromCertWeighted c with
  | some lb0, some lb1 => some (max lb0 lb1)
  | some lb0, none => some lb0
  | none, some lb1 => some lb1
  | none, none => none

section WithNeZero

variable [NeZero seq]

theorem logitDiffLowerBoundFromCert_le
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (c : InductionHeadCert seq) (hsound : InductionHeadCertSound inputs c)
    {lb : Rat} (hbound : logitDiffLowerBoundFromCert c = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := fun q k =>
        Circuit.softmax (scoresRealOfInputs inputs q) k
      let vals : Fin (Nat.succ n) → Real := valsRealOfInputs inputs
      let epsAt := Bounds.cacheBoundTask c.epsAt
      let valsLo := Bounds.cacheBoundTask c.values.valsLo
      let loAt : Fin (Nat.succ n) → Rat := fun q =>
        let others : Finset (Fin (Nat.succ n)) :=
          (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
        if h : others.Nonempty then
          others.inf' h valsLo
        else
          c.values.lo
      let others : Finset (Fin (Nat.succ n)) :=
        (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
      let sumOthers : Real := ∑ k ∈ others, weights q k
      let valsLoPrev : Real := (c.values.valsLo (c.prev q) : Real)
      let loAtRat : Rat := loAt q
      let loAtReal : Real := (loAtRat : Real)
      have hboundRat :
          lb ≤ valsLo (c.prev q) -
            epsAt q * max (0 : Rat) (valsLo (c.prev q) - loAt q) := by
        refine
          Circuit.logitDiffLowerBoundAtLoAt_le
            (active := c.active)
            (prev := c.prev)
            (epsAt := epsAt)
            (loAt := loAt)
            (valsLo := valsLo)
            q hq lb ?_
        simpa [logitDiffLowerBoundFromCert, loAt] using hbound
      have hboundRat' :
          lb ≤ c.values.valsLo (c.prev q) -
            c.epsAt q * max (0 : Rat) (c.values.valsLo (c.prev q) - loAt q) := by
        simpa [epsAt, valsLo, Bounds.cacheBoundTask_apply] using hboundRat
      have hboundReal :
          (lb : Real) ≤
            valsLoPrev - (c.epsAt q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
        simpa [loAtRat, loAtReal, ratToReal_sub, ratToReal_mul, ratToReal_max,
          ratToReal_def] using
          ratToReal_le_of_le hboundRat'
      have hweights_nonneg : ∀ k, 0 ≤ weights q k :=
        hsound.softmax_bounds.nonneg q hq
      have hweights := hsound.oneHot_bounds_at q hq
      have hsum_decomp :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := by
        simp [others]
      have hsum :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = 1 := by
        calc
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := hsum_decomp
          _ = 1 := hweights.sum_one q rfl
      have hsum_others_le : sumOthers ≤ (c.epsAt q : Real) := by
        have hprev : 1 ≤ weights q (c.prev q) + (c.epsAt q : Real) :=
          hweights.prev_large q rfl
        have hprev' :
            weights q (c.prev q) + sumOthers ≤
              weights q (c.prev q) + (c.epsAt q : Real) := by
          simpa [hsum, sumOthers] using hprev
        exact (add_le_add_iff_left (weights q (c.prev q))).1 hprev'
      have hloAt_le_valsLo : ∀ k ∈ others, loAtRat ≤ valsLo k := by
        intro k hk
        have hnonempty : others.Nonempty := ⟨k, hk⟩
        have hmin : others.inf' hnonempty valsLo ≤ valsLo k :=
          Finset.inf'_le (s := others) (f := valsLo) hk
        have hnonempty' : (Finset.univ.erase (c.prev q)).Nonempty := by
          simpa [others] using hnonempty
        have hloAt : loAtRat = others.inf' hnonempty valsLo := by
          dsimp [loAtRat, loAt]
          simp [hnonempty', others]
        calc
          loAtRat = others.inf' hnonempty valsLo := hloAt
          _ ≤ valsLo k := hmin
      have hvals_lo : ∀ k ∈ others, loAtReal ≤ vals k := by
        intro k hk
        have hloRat := hloAt_le_valsLo k hk
        have hloReal : loAtReal ≤ (valsLo k : Real) := by
          simpa [loAtReal, ratToReal_def] using (ratToReal_le_of_le hloRat)
        have hvals : (valsLo k : Real) ≤ vals k := by
          simpa [valsLo, Bounds.cacheBoundTask_apply] using
            (hsound.value_bounds.vals_bounds k).1
        exact le_trans hloReal hvals
      have hvalsLo_prev : valsLoPrev ≤ vals (c.prev q) := by
        exact (hsound.value_bounds.vals_bounds (c.prev q)).1
      have hsum_vals_ge :
          sumOthers * loAtReal ≤ ∑ k ∈ others, weights q k * vals k := by
        have hsum_lo :
            sumOthers * loAtReal = ∑ k ∈ others, weights q k * loAtReal := by
          have hsum_lo' :
              (∑ k ∈ others, weights q k) * loAtReal =
                ∑ k ∈ others, weights q k * loAtReal := by
            simpa using
              (Finset.sum_mul (s := others) (f := fun k => weights q k) (a := loAtReal))
          simpa [sumOthers] using hsum_lo'
        have hle :
            ∀ k ∈ others, weights q k * loAtReal ≤ weights q k * vals k := by
          intro k _hk
          have hval := hvals_lo k _hk
          have hnonneg := hweights_nonneg k
          exact mul_le_mul_of_nonneg_left hval hnonneg
        have hsum' :
            ∑ k ∈ others, weights q k * loAtReal ≤
              ∑ k ∈ others, weights q k * vals k := by
          exact Finset.sum_le_sum hle
        simpa [hsum_lo] using hsum'
      have hsum_prod :
          weights q (c.prev q) * vals (c.prev q) +
              ∑ k ∈ others, weights q k * vals k =
            ∑ k, weights q k * vals k := by
        simp [others]
      have hout_eq :
          dotProduct (weights q) vals =
            weights q (c.prev q) * vals (c.prev q) +
              ∑ k ∈ others, weights q k * vals k := by
        simpa [dotProduct] using hsum_prod.symm
      have hdot_ge :
          weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal ≤
            dotProduct (weights q) vals := by
        have hle :
            weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal ≤
              weights q (c.prev q) * vals (c.prev q) +
                ∑ k ∈ others, weights q k * vals k := by
          simpa [add_comm, add_left_comm, add_assoc] using
            (add_le_add_left hsum_vals_ge (weights q (c.prev q) * vals (c.prev q)))
        simpa [sumOthers, hout_eq, add_comm, add_left_comm, add_assoc] using hle
      have hprev_lo :
          weights q (c.prev q) * valsLoPrev ≤
            weights q (c.prev q) * vals (c.prev q) := by
        exact mul_le_mul_of_nonneg_left hvalsLo_prev (hweights_nonneg (c.prev q))
      have hdot_ge' :
          weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal ≤
            dotProduct (weights q) vals := by
        have hle :
            weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal ≤
              weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal := by
          simpa [add_comm, add_left_comm, add_assoc] using
            (add_le_add_right hprev_lo (sumOthers * loAtReal))
        exact hle.trans hdot_ge
      have hsplit :
          weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal =
            valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := by
        have hsplit' :
            weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal =
              (weights q (c.prev q) + sumOthers) * valsLoPrev -
                sumOthers * (valsLoPrev - loAtReal) := by
          ring
        calc
          weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal =
              (weights q (c.prev q) + sumOthers) * valsLoPrev -
                sumOthers * (valsLoPrev - loAtReal) := hsplit'
          _ = valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := by
              simp [hsum, sumOthers]
      have hdiff_le : valsLoPrev - loAtReal ≤ max (0 : Real) (valsLoPrev - loAtReal) := by
        exact le_max_right _ _
      have hsum_nonneg : 0 ≤ sumOthers := by
        have hnonneg : ∀ k ∈ others, 0 ≤ weights q k := by
          intro k _hk
          exact hweights_nonneg k
        have hsum_nonneg' : 0 ≤ ∑ k ∈ others, weights q k := by
          exact Finset.sum_nonneg hnonneg
        simpa [sumOthers] using hsum_nonneg'
      have hsum_mul_le_left :
          sumOthers * (valsLoPrev - loAtReal) ≤
            sumOthers * max (0 : Real) (valsLoPrev - loAtReal) := by
        exact mul_le_mul_of_nonneg_left hdiff_le hsum_nonneg
      have hmax_nonneg : 0 ≤ max (0 : Real) (valsLoPrev - loAtReal) := by
        exact le_max_left _ _
      have hsum_mul_le :
          sumOthers * (valsLoPrev - loAtReal) ≤
            (c.epsAt q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
        have hsum_mul_le_right :
            sumOthers * max (0 : Real) (valsLoPrev - loAtReal) ≤
              (c.epsAt q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
          exact mul_le_mul_of_nonneg_right hsum_others_le hmax_nonneg
        exact le_trans hsum_mul_le_left hsum_mul_le_right
      have hsub_le :
          valsLoPrev - (c.epsAt q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
            valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := by
        exact sub_le_sub_left hsum_mul_le valsLoPrev
      have hdot_lower :
          valsLoPrev - (c.epsAt q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
            dotProduct (weights q) vals := by
        calc
          valsLoPrev - (c.epsAt q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
              valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := hsub_le
          _ = weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal := by
              simp [hsplit]
          _ ≤ dotProduct (weights q) vals := hdot_ge'
      have hle : (lb : Real) ≤ dotProduct (weights q) vals :=
        le_trans hboundReal hdot_lower
      simpa [headLogitDiff, weights, vals] using hle

/-- The unweighted logit-diff lower bound is sound for any valid per-query `epsAt`. -/
theorem logitDiffLowerBoundFromCacheWithEps_le
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (c : InductionHeadCert seq) (epsAtCustom : Fin seq → Rat)
    (hsound : InductionHeadCertSound inputs c)
    (honeHot :
      ∀ q, q ∈ c.active →
        Layers.OneHotApproxBoundsOnActive (Val := Real) (epsAtCustom q : Real)
          (fun q' => q' = q) c.prev
          (fun q' k => Circuit.softmax (scoresRealOfInputs inputs q') k))
    {lb : Rat}
    (hbound :
      logitDiffLowerBoundFromCacheWithEps c (logitDiffCache c) epsAtCustom = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := fun q k =>
        Circuit.softmax (scoresRealOfInputs inputs q) k
      let vals : Fin (Nat.succ n) → Real := valsRealOfInputs inputs
      let epsArr : Array Rat := Array.ofFn epsAtCustom
      let valsLoArr : Array Rat := Array.ofFn (logitDiffCache c).valsLo
      let epsAt : Fin (Nat.succ n) → Rat := fun q =>
        epsArr[q.1]'(by
          simp [epsArr, q.isLt])
      let valsLo : Fin (Nat.succ n) → Rat := fun q =>
        valsLoArr[q.1]'(by
          simp [valsLoArr, q.isLt])
      let loAt : Fin (Nat.succ n) → Rat := fun q =>
        let others : Finset (Fin (Nat.succ n)) :=
          (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
        if h : others.Nonempty then
          others.inf' h valsLo
        else
          c.values.lo
      let others : Finset (Fin (Nat.succ n)) :=
        (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
      let sumOthers : Real := ∑ k ∈ others, weights q k
      let valsLoPrev : Real := (c.values.valsLo (c.prev q) : Real)
      let loAtRat : Rat := loAt q
      let loAtReal : Real := (loAtRat : Real)
      have hboundRat :
          lb ≤ valsLo (c.prev q) -
            epsAt q * max (0 : Rat) (valsLo (c.prev q) - loAt q) := by
        refine
          Circuit.logitDiffLowerBoundAtLoAt_le
            (active := c.active)
            (prev := c.prev)
            (epsAt := epsAt)
            (loAt := loAt)
            (valsLo := valsLo)
            q hq lb ?_
        simpa [logitDiffLowerBoundFromCacheWithEps, loAt, epsAt, valsLo, valsLoArr, epsArr,
          logitDiffCache] using hbound
      have hepsAt : epsAt q = epsAtCustom q := by
        simp [epsAt, epsArr]
      have hvalsLo : ∀ k, valsLo k = c.values.valsLo k := by
        intro k
        simp [valsLo, valsLoArr, logitDiffCache, Bounds.cacheBoundTask_apply]
      have hboundRat' :
          lb ≤ c.values.valsLo (c.prev q) -
            epsAtCustom q * max (0 : Rat) (c.values.valsLo (c.prev q) - loAt q) := by
        simpa [hepsAt, hvalsLo] using hboundRat
      have hboundReal :
          (lb : Real) ≤
            valsLoPrev - (epsAtCustom q : Real) *
              max (0 : Real) (valsLoPrev - loAtReal) := by
        simpa [loAtRat, loAtReal, ratToReal_sub, ratToReal_mul, ratToReal_max, ratToReal_def]
          using ratToReal_le_of_le hboundRat'
      have hweights_nonneg : ∀ k, 0 ≤ weights q k := by
        have hweights := honeHot q hq
        simpa [weights] using hweights.nonneg q rfl
      have hweights := honeHot q hq
      have hsum_decomp :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := by
        simp [others]
      have hsum :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = 1 := by
        have hsum_one : (∑ k, weights q k) = 1 := by
          simpa [weights] using hweights.sum_one q rfl
        calc
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := hsum_decomp
          _ = 1 := hsum_one
      have hsum_others_le : sumOthers ≤ (epsAtCustom q : Real) := by
        have hprev : 1 ≤ weights q (c.prev q) + (epsAtCustom q : Real) :=
          hweights.prev_large q rfl
        have hprev' :
            weights q (c.prev q) + sumOthers ≤
              weights q (c.prev q) + (epsAtCustom q : Real) := by
          simpa [hsum, sumOthers] using hprev
        exact (add_le_add_iff_left (weights q (c.prev q))).1 hprev'
      have hloAt_le_valsLo : ∀ k ∈ others, loAtRat ≤ c.values.valsLo k := by
        intro k hk
        have hnonempty : others.Nonempty := ⟨k, hk⟩
        have hmin : others.inf' hnonempty valsLo ≤ valsLo k :=
          Finset.inf'_le (s := others) (f := valsLo) hk
        have hnonempty' : (Finset.univ.erase (c.prev q)).Nonempty := by
          simpa [others] using hnonempty
        have hloAt : loAtRat = others.inf' hnonempty valsLo := by
          dsimp [loAtRat, loAt]
          simp [hnonempty', others]
        have hvalsLo' : valsLo k = c.values.valsLo k := hvalsLo k
        calc
          loAtRat = others.inf' hnonempty valsLo := hloAt
          _ ≤ valsLo k := hmin
          _ = c.values.valsLo k := hvalsLo'
      have hvals_lo : ∀ k ∈ others, loAtReal ≤ vals k := by
        intro k hk
        have hloRat := hloAt_le_valsLo k hk
        have hloReal : loAtReal ≤ (c.values.valsLo k : Real) := by
          simpa [loAtReal, ratToReal_def] using (ratToReal_le_of_le hloRat)
        have hvals : (c.values.valsLo k : Real) ≤ vals k := by
          simpa using (hsound.value_bounds.vals_bounds k).1
        exact le_trans hloReal hvals
      have hvalsLo_prev : valsLoPrev ≤ vals (c.prev q) := by
        exact (hsound.value_bounds.vals_bounds (c.prev q)).1
      have hsum_vals_ge :
          sumOthers * loAtReal ≤ ∑ k ∈ others, weights q k * vals k := by
        have hsum_lo :
            sumOthers * loAtReal = ∑ k ∈ others, weights q k * loAtReal := by
          have hsum_lo' :
              (∑ k ∈ others, weights q k) * loAtReal =
                ∑ k ∈ others, weights q k * loAtReal := by
            simpa using
              (Finset.sum_mul (s := others) (f := fun k => weights q k) (a := loAtReal))
          simpa [sumOthers] using hsum_lo'
        have hle :
            ∀ k ∈ others, weights q k * loAtReal ≤ weights q k * vals k := by
          intro k _hk
          have hval := hvals_lo k _hk
          have hnonneg := hweights_nonneg k
          exact mul_le_mul_of_nonneg_left hval hnonneg
        have hsum' :
            ∑ k ∈ others, weights q k * loAtReal ≤
              ∑ k ∈ others, weights q k * vals k := by
          exact Finset.sum_le_sum hle
        simpa [hsum_lo] using hsum'
      have hsum_prod :
          weights q (c.prev q) * vals (c.prev q) +
              ∑ k ∈ others, weights q k * vals k =
            ∑ k, weights q k * vals k := by
        simp [others]
      have hout_eq :
          dotProduct (weights q) vals =
            weights q (c.prev q) * vals (c.prev q) +
              ∑ k ∈ others, weights q k * vals k := by
        simpa [dotProduct] using hsum_prod.symm
      have hdot_ge :
          weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal ≤
            dotProduct (weights q) vals := by
        have hle :
            weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal ≤
              weights q (c.prev q) * vals (c.prev q) +
                ∑ k ∈ others, weights q k * vals k := by
          simpa [add_comm, add_left_comm, add_assoc] using
            (add_le_add_left hsum_vals_ge (weights q (c.prev q) * vals (c.prev q)))
        simpa [sumOthers, hout_eq, add_comm, add_left_comm, add_assoc] using hle
      have hprev_lo :
          weights q (c.prev q) * valsLoPrev ≤
            weights q (c.prev q) * vals (c.prev q) := by
        exact mul_le_mul_of_nonneg_left hvalsLo_prev (hweights_nonneg (c.prev q))
      have hdot_ge' :
          weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal ≤
            dotProduct (weights q) vals := by
        have hle :
            weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal ≤
              weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal := by
          simpa [add_comm, add_left_comm, add_assoc] using
            (add_le_add_right hprev_lo (sumOthers * loAtReal))
        exact hle.trans hdot_ge
      have hsplit :
          weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal =
            valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := by
        have hsplit' :
            weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal =
              (weights q (c.prev q) + sumOthers) * valsLoPrev -
                sumOthers * (valsLoPrev - loAtReal) := by
          ring
        calc
          weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal =
              (weights q (c.prev q) + sumOthers) * valsLoPrev -
                sumOthers * (valsLoPrev - loAtReal) := hsplit'
          _ = valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := by
              simp [hsum, sumOthers]
      have hdiff_le : valsLoPrev - loAtReal ≤ max (0 : Real) (valsLoPrev - loAtReal) := by
        exact le_max_right _ _
      have hsum_nonneg : 0 ≤ sumOthers := by
        have hnonneg : ∀ k ∈ others, 0 ≤ weights q k := by
          intro k _hk
          exact hweights_nonneg k
        have hsum_nonneg' : 0 ≤ ∑ k ∈ others, weights q k := by
          exact Finset.sum_nonneg hnonneg
        simpa [sumOthers] using hsum_nonneg'
      have hsum_mul_le_left :
          sumOthers * (valsLoPrev - loAtReal) ≤
            sumOthers * max (0 : Real) (valsLoPrev - loAtReal) := by
        exact mul_le_mul_of_nonneg_left hdiff_le hsum_nonneg
      have hmax_nonneg : 0 ≤ max (0 : Real) (valsLoPrev - loAtReal) := by
        exact le_max_left _ _
      have hsum_mul_le :
          sumOthers * (valsLoPrev - loAtReal) ≤
            (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
        have hsum_mul_le_right :
            sumOthers * max (0 : Real) (valsLoPrev - loAtReal) ≤
              (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
          exact mul_le_mul_of_nonneg_right hsum_others_le hmax_nonneg
        exact le_trans hsum_mul_le_left hsum_mul_le_right
      have hsub_le :
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
            valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := by
        exact sub_le_sub_left hsum_mul_le valsLoPrev
      have hdot_lower :
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
            dotProduct (weights q) vals := by
        calc
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
              valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := hsub_le
          _ = weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal := by
              simp [hsplit]
          _ ≤ dotProduct (weights q) vals := hdot_ge'
      have hle : (lb : Real) ≤ dotProduct (weights q) vals :=
        le_trans hboundReal hdot_lower
      simpa [headLogitDiff, weights, vals] using hle

/-- The weighted per-key logit-diff lower bound is sound on active queries. -/
theorem logitDiffLowerBoundFromCertWeighted_le
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (c : InductionHeadCert seq) (hsound : InductionHeadCertSound inputs c)
    {lb : Rat} (hbound : logitDiffLowerBoundFromCertWeighted c = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := fun q k =>
        Circuit.softmax (scoresRealOfInputs inputs q) k
      let vals : Fin (Nat.succ n) → Real := valsRealOfInputs inputs
      let valsLoCached := Bounds.cacheBoundTask c.values.valsLo
      let others : Finset (Fin (Nat.succ n)) :=
        (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
      let valsLoPrevRat : Rat := valsLoCached (c.prev q)
      let valsLoPrev : Real := (valsLoPrevRat : Real)
      have hboundRat :
          lb ≤ valsLoPrevRat -
            (Finset.univ : Finset (Fin (Nat.succ n))).sum (fun k =>
              c.weightBoundAt q k * max (0 : Rat) (valsLoPrevRat - valsLoCached k)) := by
        refine
          Circuit.logitDiffLowerBoundWeightedAt_le
            (active := c.active)
            (prev := c.prev)
            (weightBoundAt := c.weightBoundAt)
            (valsLo := valsLoCached)
            q hq lb ?_
        simpa [logitDiffLowerBoundFromCertWeighted] using hbound
      have hboundRat' :
          lb ≤ valsLoPrevRat -
            (Finset.univ : Finset (Fin (Nat.succ n))).sum (fun k =>
              c.weightBoundAt q k * max (0 : Rat) (valsLoPrevRat - c.values.valsLo k)) := by
        simpa [valsLoCached, valsLoPrevRat, Bounds.cacheBoundTask_apply] using hboundRat
      have hboundReal :
          (lb : Real) ≤
            valsLoPrev -
              (Finset.univ : Finset (Fin (Nat.succ n))).sum (fun k =>
                (c.weightBoundAt q k : Real) *
                  max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) := by
        simpa [valsLoPrevRat, valsLoPrev, ratToReal_sub, ratToReal_mul, ratToReal_max,
          ratToReal_def, Rat.cast_sum] using ratToReal_le_of_le hboundRat'
      let gapTerm : Fin (Nat.succ n) → Real := fun k =>
        (c.weightBoundAt q k : Real) *
          max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))
      have hgap_prev : gapTerm (c.prev q) = 0 := by
        have hdiff : valsLoPrev - (c.values.valsLo (c.prev q) : Real) = 0 := by
          simp [valsLoPrev, valsLoPrevRat, valsLoCached, Bounds.cacheBoundTask_apply]
        simp [gapTerm, hdiff]
      have hsum_gap :
          (Finset.univ : Finset (Fin (Nat.succ n))).sum gapTerm =
            ∑ k ∈ others, gapTerm k := by
        classical
        have hsum :=
          (Finset.sum_erase (s := (Finset.univ : Finset (Fin (Nat.succ n))))
            (f := gapTerm) (a := c.prev q) hgap_prev)
        simpa [others] using hsum.symm
      have hboundReal' :
          (lb : Real) ≤
            valsLoPrev - ∑ k ∈ others, gapTerm k := by
        simpa [gapTerm, hsum_gap] using hboundReal
      have hweights_nonneg : ∀ k, 0 ≤ weights q k :=
        hsound.softmax_bounds.nonneg q hq
      have hweights := hsound.oneHot_bounds_at q hq
      have hsum_decomp :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := by
        simp [others]
      have hsum :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = 1 := by
        calc
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := hsum_decomp
          _ = 1 := hweights.sum_one q rfl
      have hvalsLo_prev : valsLoPrev ≤ vals (c.prev q) := by
        have hvals := (hsound.value_bounds.vals_bounds (c.prev q)).1
        simpa [valsLoPrev, valsLoPrevRat, valsLoCached, Bounds.cacheBoundTask_apply] using hvals
      have hvals_lower :
          ∀ k,
            valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) ≤ vals k := by
        intro k
        by_cases hle : valsLoPrev ≤ (c.values.valsLo k : Real)
        · have hdiff : valsLoPrev - (c.values.valsLo k : Real) ≤ 0 := sub_nonpos.mpr hle
          have hmax :
              max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) = 0 := by
            simp [hdiff]
          have hvals_lo : (c.values.valsLo k : Real) ≤ vals k :=
            (hsound.value_bounds.vals_bounds k).1
          have hvals_prev : valsLoPrev ≤ vals k := le_trans hle hvals_lo
          simpa [hmax] using hvals_prev
        · have hlt : (c.values.valsLo k : Real) < valsLoPrev := lt_of_not_ge hle
          have hdiff_nonneg : 0 ≤ valsLoPrev - (c.values.valsLo k : Real) :=
            le_of_lt (sub_pos.mpr hlt)
          have hmax :
              max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) =
                valsLoPrev - (c.values.valsLo k : Real) := by
            simp [hdiff_nonneg]
          have hvals_lo : (c.values.valsLo k : Real) ≤ vals k :=
            (hsound.value_bounds.vals_bounds k).1
          simpa [hmax] using hvals_lo
      have hsum_vals_ge :
          ∑ k ∈ others, weights q k *
              (valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) ≤
            ∑ k ∈ others, weights q k * vals k := by
        refine Finset.sum_le_sum ?_
        intro k hk
        exact
          mul_le_mul_of_nonneg_left
            (hvals_lower k)
            (hweights_nonneg k)
      have hsum_prod :
          weights q (c.prev q) * vals (c.prev q) +
              ∑ k ∈ others, weights q k * vals k =
            ∑ k, weights q k * vals k := by
        simp [others]
      have hout_eq :
          dotProduct (weights q) vals =
            weights q (c.prev q) * vals (c.prev q) +
              ∑ k ∈ others, weights q k * vals k := by
        simpa [dotProduct] using hsum_prod.symm
      have hprev_lo :
          weights q (c.prev q) * valsLoPrev ≤
            weights q (c.prev q) * vals (c.prev q) := by
        exact mul_le_mul_of_nonneg_left hvalsLo_prev (hweights_nonneg (c.prev q))
      have hdot_ge' :
          weights q (c.prev q) * valsLoPrev +
              ∑ k ∈ others, weights q k *
                (valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) ≤
            dotProduct (weights q) vals := by
        have hle :
            weights q (c.prev q) * valsLoPrev +
                ∑ k ∈ others, weights q k *
                  (valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) ≤
              weights q (c.prev q) * vals (c.prev q) +
                ∑ k ∈ others, weights q k * vals k := by
          exact add_le_add hprev_lo hsum_vals_ge
        simpa [hout_eq, add_comm, add_left_comm, add_assoc] using hle
      have hsum_weights :
          (∑ k ∈ others, weights q k * valsLoPrev) =
            (∑ k ∈ others, weights q k) * valsLoPrev := by
        have hsum_mul :
            (∑ k ∈ others, weights q k) * valsLoPrev =
              ∑ k ∈ others, weights q k * valsLoPrev := by
          simpa using
            (Finset.sum_mul (s := others) (f := fun k => weights q k) (a := valsLoPrev))
        exact hsum_mul.symm
      have hsum_split :
          ∑ k ∈ others, weights q k *
              (valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) =
            (∑ k ∈ others, weights q k) * valsLoPrev -
              ∑ k ∈ others, weights q k *
                max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := by
        calc
          ∑ k ∈ others, weights q k *
              (valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) =
              ∑ k ∈ others,
                (weights q k * valsLoPrev -
                  weights q k * max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) := by
                simp [mul_sub]
          _ =
              (∑ k ∈ others, weights q k * valsLoPrev) -
                ∑ k ∈ others, weights q k *
                  max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := by
                simp [Finset.sum_sub_distrib]
          _ =
              (∑ k ∈ others, weights q k) * valsLoPrev -
                ∑ k ∈ others, weights q k *
                  max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := by
                simp [hsum_weights]
      have hsplit :
          weights q (c.prev q) * valsLoPrev +
              ∑ k ∈ others, weights q k *
                (valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) =
            valsLoPrev -
              ∑ k ∈ others, weights q k *
                max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := by
        calc
          weights q (c.prev q) * valsLoPrev +
              ∑ k ∈ others, weights q k *
                (valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) =
              weights q (c.prev q) * valsLoPrev +
                ((∑ k ∈ others, weights q k) * valsLoPrev -
                  ∑ k ∈ others, weights q k *
                    max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) := by
                simp [hsum_split]
          _ =
              (weights q (c.prev q) + ∑ k ∈ others, weights q k) * valsLoPrev -
                ∑ k ∈ others, weights q k *
                  max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := by
                ring
          _ =
              valsLoPrev -
                ∑ k ∈ others, weights q k *
                  max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := by
                simp [hsum]
      have hweight_bound :
          ∀ k ∈ others, weights q k ≤ (c.weightBoundAt q k : Real) := by
        intro k hk
        have hk' : k ≠ c.prev q := (Finset.mem_erase.mp hk).1
        exact hsound.weight_bounds_at q hq k hk'
      have hsum_gap_le :
          ∑ k ∈ others, weights q k *
              max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) ≤
            ∑ k ∈ others, (c.weightBoundAt q k : Real) *
              max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := by
        refine Finset.sum_le_sum ?_
        intro k hk
        have hnonneg :
            0 ≤ max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) :=
          le_max_left _ _
        exact mul_le_mul_of_nonneg_right (hweight_bound k hk) hnonneg
      have hsub_le :
          valsLoPrev -
              ∑ k ∈ others, (c.weightBoundAt q k : Real) *
                max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) ≤
            valsLoPrev -
              ∑ k ∈ others, weights q k *
                max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := by
        exact sub_le_sub_left hsum_gap_le valsLoPrev
      have hdot_lower :
          valsLoPrev -
              ∑ k ∈ others, (c.weightBoundAt q k : Real) *
                max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) ≤
            dotProduct (weights q) vals := by
        calc
          valsLoPrev -
              ∑ k ∈ others, (c.weightBoundAt q k : Real) *
                max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) ≤
              valsLoPrev -
                ∑ k ∈ others, weights q k *
                  max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real)) := hsub_le
          _ =
              weights q (c.prev q) * valsLoPrev +
                ∑ k ∈ others, weights q k *
                  (valsLoPrev - max (0 : Real) (valsLoPrev - (c.values.valsLo k : Real))) := by
                simpa using hsplit.symm
          _ ≤ dotProduct (weights q) vals := hdot_ge'
      have hle : (lb : Real) ≤ dotProduct (weights q) vals :=
        le_trans hboundReal' hdot_lower
      simpa [headLogitDiff, weights, vals] using hle

/-- The best available logit-diff lower bound is sound on active queries. -/
theorem logitDiffLowerBoundFromCertBest_le
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (c : InductionHeadCert seq) (hsound : InductionHeadCertSound inputs c)
    {lb : Rat} (hbound : logitDiffLowerBoundFromCertBest c = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  cases h0 : logitDiffLowerBoundFromCert c with
  | none =>
      cases h1 : logitDiffLowerBoundFromCertWeighted c with
      | none =>
          simp [logitDiffLowerBoundFromCertBest, h0, h1] at hbound
      | some lb1 =>
          have hbound' : lb1 = lb := by
            simpa [logitDiffLowerBoundFromCertBest, h0, h1] using hbound
          cases hbound'
          exact logitDiffLowerBoundFromCertWeighted_le inputs c hsound h1 hq
  | some lb0 =>
      cases h1 : logitDiffLowerBoundFromCertWeighted c with
      | none =>
          have hbound' : lb0 = lb := by
            simpa [logitDiffLowerBoundFromCertBest, h0, h1] using hbound
          cases hbound'
          exact logitDiffLowerBoundFromCert_le inputs c hsound h0 hq
      | some lb1 =>
          have hbound' : max lb0 lb1 = lb := by
            simpa [logitDiffLowerBoundFromCertBest, h0, h1] using hbound
          cases hbound'
          have h0le : (lb0 : Real) ≤ headLogitDiff inputs q :=
            logitDiffLowerBoundFromCert_le inputs c hsound h0 hq
          have h1le : (lb1 : Real) ≤ headLogitDiff inputs q :=
            logitDiffLowerBoundFromCertWeighted_le inputs c hsound h1 hq
          have hmax : max (lb0 : Real) (lb1 : Real) ≤ headLogitDiff inputs q :=
            max_le_iff.mpr ⟨h0le, h1le⟩
          simpa [ratToReal_max, ratToReal_def] using hmax

/-- Certified logit-diff lower bound derived from exact head inputs. -/
structure InductionLogitLowerBoundResult
    (inputs : Model.InductionHeadInputs seq dModel dHead) where
  /-- Induction certificate built from the head inputs. -/
  cert : InductionHeadCert seq
  /-- Soundness proof for the induction certificate. -/
  sound : InductionHeadCertSound inputs cert
  /-- Reported lower bound on logit diff. -/
  lb : Rat
  /-- `lb` is computed from `logitDiffLowerBoundFromCert`. -/
  lb_def : logitDiffLowerBoundFromCert cert = some lb
  /-- The lower bound is sound on active queries. -/
  lb_sound : ∀ q, q ∈ cert.active → (lb : Real) ≤ headLogitDiff inputs q

/-- Nonvacuous logit-diff bound (strictly positive). -/
structure InductionLogitLowerBoundNonvacuous
    (inputs : Model.InductionHeadInputs seq dModel dHead) where
  /-- Base logit-diff bound data. -/
  base : InductionLogitLowerBoundResult inputs
  /-- The reported bound is strictly positive. -/
  lb_pos : 0 < base.lb

/-- Build a logit-diff lower bound from exact head inputs. -/
def buildInductionLogitLowerBoundFromHead?
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (InductionLogitLowerBoundResult inputs) := by
  classical
  cases hcert : buildInductionCertFromHead? inputs with
  | none => exact none
  | some certWithProof =>
      rcases certWithProof with ⟨cert, hsound⟩
      cases hlb : logitDiffLowerBoundFromCert cert with
      | none => exact none
      | some lb =>
          refine some ?_
          refine
            { cert := cert
              sound := hsound
              lb := lb
              lb_def := hlb
              lb_sound := ?_ }
          intro q hq
          exact
            logitDiffLowerBoundFromCert_le
              (inputs := inputs)
              (c := cert)
              (hsound := hsound)
              (lb := lb)
              (hbound := hlb)
              (q := q)
              hq

/-- Build a strictly positive logit-diff lower bound from exact head inputs. -/
def buildInductionLogitLowerBoundNonvacuous?
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (InductionLogitLowerBoundNonvacuous inputs) := by
  classical
  cases hbase : buildInductionLogitLowerBoundFromHead? inputs with
  | none => exact none
  | some base =>
      by_cases hpos : 0 < base.lb
      · exact some ⟨base, hpos⟩
      · exact none

end WithNeZero

/-! End-to-end lower bounds from head certificates plus residual intervals. -/

/-- The head logit-diff equals the direction dot product of the head output. -/
theorem headLogitDiff_eq_direction_dot_headOutput
    (inputs : Model.InductionHeadInputs seq dModel dHead) (q : Fin seq) :
    headLogitDiff inputs q =
      dotProduct (fun i => (inputs.direction i : Real))
        (fun i => headOutput inputs q i) := by
  classical
  let dir : Fin dModel → Real := fun i => (inputs.direction i : Real)
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scoresRealOfInputs inputs q) k
  have hswap :
      dotProduct dir (fun i => headOutput inputs q i) =
        ∑ k, weights q k * dotProduct dir (fun i => headValueRealOfInputs inputs k i) := by
    calc
      dotProduct dir (fun i => headOutput inputs q i)
          = ∑ i, dir i * ∑ k, weights q k * headValueRealOfInputs inputs k i := by
            simp [dir, headOutput_def, headOutputWithScores_def, weights, dotProduct]
      _ = ∑ i, ∑ k, dir i * (weights q k * headValueRealOfInputs inputs k i) := by
            simp [Finset.mul_sum]
      _ = ∑ k, ∑ i, dir i * (weights q k * headValueRealOfInputs inputs k i) := by
            simpa using
              (Finset.sum_comm (s := (Finset.univ : Finset (Fin dModel)))
                (t := (Finset.univ : Finset (Fin seq)))
                (f := fun i k => dir i * (weights q k * headValueRealOfInputs inputs k i)))
      _ = ∑ k, weights q k * ∑ i, dir i * headValueRealOfInputs inputs k i := by
            refine Finset.sum_congr rfl ?_
            intro k _
            calc
              ∑ i, dir i * (weights q k * headValueRealOfInputs inputs k i) =
                  ∑ i, weights q k * (dir i * headValueRealOfInputs inputs k i) := by
                    refine Finset.sum_congr rfl ?_
                    intro i _
                    simp [mul_assoc, mul_left_comm, mul_comm]
              _ = weights q k * ∑ i, dir i * headValueRealOfInputs inputs k i := by
                    simp [Finset.mul_sum]
  have hsum :
      dotProduct dir (fun i => headOutput inputs q i) =
        ∑ k, weights q k * valsRealOfInputs inputs k := by
    calc
      dotProduct dir (fun i => headOutput inputs q i) =
          ∑ k, weights q k * dotProduct dir (fun i => headValueRealOfInputs inputs k i) := hswap
      _ = ∑ k, weights q k * valsRealOfInputs inputs k := by
          refine Finset.sum_congr rfl ?_
          intro k _
          have hdir := direction_dot_headValue_eq_valsReal (inputs := inputs) (k := k)
          exact congrArg (fun x => weights q k * x) (by simpa [dir] using hdir)
  calc
    headLogitDiff inputs q =
        dotProduct (weights q) (valsRealOfInputs inputs) := by
          simp [headLogitDiff, weights]
    _ = ∑ k, weights q k * valsRealOfInputs inputs k := by
          simp [dotProduct]
    _ = dotProduct (fun i => (inputs.direction i : Real))
          (fun i => headOutput inputs q i) := by
          simpa [dir] using hsum.symm

/-- Combine a head logit-diff bound with residual interval bounds. -/
theorem logitDiffLowerBound_with_residual
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lb : Rat)
    (hlb : ∀ q, q ∈ inputs.active → (lb : Real) ≤ headLogitDiff inputs q)
    (residual : Fin seq → Fin dModel → Real)
    (lo hi : Fin dModel → Rat)
    (hres : ∀ q, q ∈ inputs.active → ∀ i,
      (lo i : Real) ≤ residual q i ∧ residual q i ≤ (hi i : Real)) :
    ∀ q, q ∈ inputs.active →
      (lb : Real) - (Bounds.dotIntervalAbsBound inputs.direction lo hi : Real) ≤
        dotProduct (fun i => (inputs.direction i : Real))
          (fun i => headOutput inputs q i + residual q i) := by
  intro q hq
  have hhead := hlb q hq
  have hres' :
      |dotProduct (fun i => (inputs.direction i : Real)) (residual q)| ≤
        (Bounds.dotIntervalAbsBound inputs.direction lo hi : Real) := by
    have hlo : ∀ i, (lo i : Real) ≤ residual q i := fun i => (hres q hq i).1
    have hhi : ∀ i, residual q i ≤ (hi i : Real) := fun i => (hres q hq i).2
    simpa using
      (Bounds.abs_dotProduct_le_dotIntervalAbsBound_real
        (v := inputs.direction) (lo := lo) (hi := hi) (x := residual q) hlo hhi)
  have hres_lower :
      -(Bounds.dotIntervalAbsBound inputs.direction lo hi : Real) ≤
        dotProduct (fun i => (inputs.direction i : Real)) (residual q) := by
    exact (abs_le.mp hres').1
  have hsum :
      (lb : Real) - (Bounds.dotIntervalAbsBound inputs.direction lo hi : Real) ≤
        headLogitDiff inputs q +
          dotProduct (fun i => (inputs.direction i : Real)) (residual q) := by
    have hsum' : (lb : Real) + -(Bounds.dotIntervalAbsBound inputs.direction lo hi : Real) ≤
        headLogitDiff inputs q +
          dotProduct (fun i => (inputs.direction i : Real)) (residual q) :=
      add_le_add hhead hres_lower
    simpa [sub_eq_add_neg] using hsum'
  calc
    (lb : Real) - (Bounds.dotIntervalAbsBound inputs.direction lo hi : Real) ≤
        headLogitDiff inputs q +
          dotProduct (fun i => (inputs.direction i : Real)) (residual q) := hsum
    _ =
        dotProduct (fun i => (inputs.direction i : Real))
          (fun i => headOutput inputs q i + residual q i) := by
          have hdot :
              dotProduct (fun i => (inputs.direction i : Real))
                  (fun i => headOutput inputs q i + residual q i) =
                dotProduct (fun i => (inputs.direction i : Real))
                    (fun i => headOutput inputs q i) +
                  dotProduct (fun i => (inputs.direction i : Real)) (residual q) := by
            simpa using
              (Linear.dotProduct_add_right
                (x := fun i => (inputs.direction i : Real))
                (y := fun i => headOutput inputs q i)
                (z := residual q))
          simp [headLogitDiff_eq_direction_dot_headOutput, hdot]

/-- Combine a head logit-diff bound with intervals on head output and a downstream output. -/
theorem logitDiffLowerBound_with_output_intervals
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lb : Rat)
    (hlb : ∀ q, q ∈ inputs.active → (lb : Real) ≤ headLogitDiff inputs q)
    (output : Fin seq → Fin dModel → Real)
    (outLo outHi : Fin dModel → Rat)
    (hout : ∀ q, q ∈ inputs.active → ∀ i,
      (outLo i : Real) ≤ output q i ∧ output q i ≤ (outHi i : Real))
    (headLo headHi : Fin dModel → Rat)
    (hhead : ∀ q, q ∈ inputs.active → ∀ i,
      (headLo i : Real) ≤ headOutput inputs q i ∧
        headOutput inputs q i ≤ (headHi i : Real)) :
    ∀ q, q ∈ inputs.active →
      (lb : Real) -
          (Bounds.dotIntervalAbsBound inputs.direction
            (fun i => outLo i - headHi i) (fun i => outHi i - headLo i) : Real) ≤
        dotProduct (fun i => (inputs.direction i : Real)) (fun i => output q i) := by
  intro q hq
  let residual : Fin seq → Fin dModel → Real :=
    fun q i => output q i - headOutput inputs q i
  let lo : Fin dModel → Rat := fun i => outLo i - headHi i
  let hi : Fin dModel → Rat := fun i => outHi i - headLo i
  have hres : ∀ q, q ∈ inputs.active → ∀ i,
      (lo i : Real) ≤ residual q i ∧ residual q i ≤ (hi i : Real) := by
    intro q hq i
    have hout_q := hout q hq i
    have hhead_q := hhead q hq i
    have hlow :
        (outLo i : Real) - (headHi i : Real) ≤
          output q i - headOutput inputs q i := by
      exact sub_le_sub hout_q.1 hhead_q.2
    have hhigh :
        output q i - headOutput inputs q i ≤
          (outHi i : Real) - (headLo i : Real) := by
      exact sub_le_sub hout_q.2 hhead_q.1
    constructor
    · simpa [lo, residual, ratToReal_sub] using hlow
    · simpa [hi, residual, ratToReal_sub] using hhigh
  have hbound :=
    logitDiffLowerBound_with_residual
      (inputs := inputs)
      (lb := lb)
      (hlb := hlb)
      (residual := residual)
      (lo := lo)
      (hi := hi)
      hres
      q
      hq
  have hdot :
      dotProduct (fun i => (inputs.direction i : Real))
          (fun i => headOutput inputs q i + residual q i) =
        dotProduct (fun i => (inputs.direction i : Real))
          (fun i => output q i) := by
    refine Finset.sum_congr rfl ?_
    intro i _
    have hsum :
        headOutput inputs q i + residual q i = output q i := by
      simp [residual, sub_eq_add_neg, add_left_comm]
    simp [hsum]
  simpa [lo, hi, hdot] using hbound

end LogitDiffLowerBound

end Sound

end Nfp
