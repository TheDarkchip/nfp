-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.Interval
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.IO.InductionHead.ScoreUtils
public import Nfp.Sound.Induction.ScoreBounds

/-!
Logit-diff lower bounds derived from induction-head certificates.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

open Nfp.Bounds

variable {seq : Nat}

/-- Components of a logit-diff lower bound computation. -/
structure LogitDiffLowerBoundComponents (seq : Nat) where
  /-- Base bound from global value interval. -/
  base? : Option Rat
  /-- Tight bound from per-query eps and per-key lower bounds. -/
  tight? : Option Rat
  /-- Tight bound from per-key weight bounds. -/
  weighted? : Option Rat
  /-- Best available bound (max of present components). -/
  best? : Option Rat

/-- Compute logit-diff lower-bound components over an active set. -/
def logitDiffLowerBoundComponentsWithActive (active : Finset (Fin seq))
    (cert : Circuit.InductionHeadCert seq) : LogitDiffLowerBoundComponents seq :=
  let base? :=
    Circuit.logitDiffLowerBoundAt active cert.prev cert.epsAt
      cert.values.lo cert.values.hi cert.values.vals
  let epsAtTight := Sound.epsAtOfWeightBoundAt cert.prev cert.weightBoundAt
  let valsLo := cert.values.valsLo
  let weighted? :=
    Circuit.logitDiffLowerBoundWeightedAt active cert.prev cert.weightBoundAt valsLo
  let loAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (cert.prev q)
    if h : others.Nonempty then
      others.inf' h valsLo
    else
      cert.values.lo
  let tight? :=
    Circuit.logitDiffLowerBoundAtLoAt active cert.prev epsAtTight loAt valsLo
  let best? :=
    match base?, tight?, weighted? with
    | some a, some b, some c => some (max a (max b c))
    | some a, some b, none => some (max a b)
    | some a, none, some c => some (max a c)
    | none, some b, some c => some (max b c)
    | some a, none, none => some a
    | none, some b, none => some b
    | none, none, some c => some c
    | none, none, none => none
  { base? := base?, tight? := tight?, weighted? := weighted?, best? := best? }

/-- Tight logit-diff lower bound over a chosen active set. -/
def logitDiffLowerBoundTightWithActive? (active : Finset (Fin seq))
    (cert : Circuit.InductionHeadCert seq) : Option Rat :=
  (logitDiffLowerBoundComponentsWithActive active cert).best?

/-- Compute logit-diff lower-bound components from external per-key value bounds. -/
def logitDiffLowerBoundComponentsWithActiveValBounds (active : Finset (Fin seq))
    (cert : Circuit.InductionHeadCert seq)
    (valBounds : Fin seq → Rat × Rat) : LogitDiffLowerBoundComponents seq :=
  let valsLo : Fin seq → Rat := fun k => (valBounds k).1
  let epsAtTight := Sound.epsAtOfWeightBoundAt cert.prev cert.weightBoundAt
  let loAt : Fin seq → Rat := fun q =>
    let others : Finset (Fin seq) :=
      (Finset.univ : Finset (Fin seq)).erase (cert.prev q)
    if h : others.Nonempty then
      others.inf' h valsLo
    else
      valsLo (cert.prev q)
  let tight? :=
    Circuit.logitDiffLowerBoundAtLoAt active cert.prev epsAtTight loAt valsLo
  let weighted? :=
    Circuit.logitDiffLowerBoundWeightedAt active cert.prev cert.weightBoundAt valsLo
  let best? :=
    match tight?, weighted? with
    | some a, some b => some (max a b)
    | some a, none => some a
    | none, some b => some b
    | none, none => none
  { base? := none, tight? := tight?, weighted? := weighted?, best? := best? }

/--
Lower bound from exact weights and per-key value intervals.

Uses `dotIntervalLower` and returns `none` if weights are missing or the
active set is empty.
-/
def logitDiffLowerBoundExactWithActiveValBounds? (active : Finset (Fin seq))
    (weightsPresent : Bool)
    (weights : Fin seq → Fin seq → Rat)
    (valBounds : Fin seq → Rat × Rat) : Option Rat := by
  classical
  if !weightsPresent then
    exact none
  else
    if h : active.Nonempty then
      let valLo : Fin seq → Rat := fun k => (valBounds k).1
      let valHi : Fin seq → Rat := fun k => (valBounds k).2
      let f : Fin seq → Rat := fun q => dotIntervalLower (weights q) valLo valHi
      exact some (active.inf' h f)
    else
      exact none

/-- Report logit-diff lower-bound components, with optional model-derived bounds. -/
def reportLogitDiffComponents (active : Finset (Fin seq))
    (cert : Circuit.InductionHeadCert seq)
    (weightsPresent : Bool)
    (valBoundsArr? : Option (Array (Rat × Rat))) : IO Unit := do
  let showOpt : Option Rat → String := fun o =>
    match o with
    | some v => ratToString v
    | none => "none"
  let baseComponents := logitDiffLowerBoundComponentsWithActive active cert
  let exactBase? : Option Rat := none
  IO.eprintln
    s!"info: logit-diff base={showOpt baseComponents.base?} \
    tight={showOpt baseComponents.tight?} \
    weighted={showOpt baseComponents.weighted?} \
    exact={showOpt exactBase?} \
    best={showOpt baseComponents.best?}"
  match valBoundsArr? with
  | some valBoundsArr =>
      let valBounds : Fin seq → Rat × Rat := fun k => valBoundsArr[k.1]!
      let modelComponents :=
        logitDiffLowerBoundComponentsWithActiveValBounds active cert valBounds
      let exact? :=
        logitDiffLowerBoundExactWithActiveValBounds?
          active weightsPresent cert.weights valBounds
      let bestModel? :=
        match modelComponents.best?, exact? with
        | some a, some b => some (max a b)
        | some a, none => some a
        | none, some b => some b
        | none, none => none
      IO.eprintln
        s!"info: logit-diff-model \
        tight={showOpt modelComponents.tight?} \
        weighted={showOpt modelComponents.weighted?} \
        exact={showOpt exact?} \
        best={showOpt bestModel?}"
  | none =>
    IO.eprintln "info: logit-diff-model skipped (missing model value bounds)"

/--
Tight logit-diff lower bound using external per-key value lower bounds.

This is useful when model-anchored value bounds are available and tighter than
the certificate's internal bounds.
-/
def logitDiffLowerBoundTightWithActiveValBounds? (active : Finset (Fin seq))
    (cert : Circuit.InductionHeadCert seq)
    (valBounds : Fin seq → Rat × Rat) : Option Rat :=
  (logitDiffLowerBoundComponentsWithActiveValBounds active cert valBounds).best?

/-- Tight logit-diff lower bound: max of global-interval and per-key lower bound variants. -/
def logitDiffLowerBoundTight? (cert : Circuit.InductionHeadCert seq) : Option Rat :=
  logitDiffLowerBoundTightWithActive? cert.active cert

end InductionHeadCert

end IO

end Nfp
