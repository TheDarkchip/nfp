-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.IO.Pure
import Nfp.IO.NfptPure
import Nfp.Circuit.Cert.LogitDiff
import Nfp.Circuit.Cert.DownstreamLinear
import Nfp.Circuit.Cert.ResidualBound
import Nfp.Circuit.Cert.ResidualInterval
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Induction

/-!
IO wrappers for loading and checking induction certificates.
-/

namespace Nfp

namespace IO

open Nfp.Circuit

/-- Load a softmax-margin certificate from disk. -/
def loadSoftmaxMarginCert (path : System.FilePath) :
    IO (Except String (Sigma SoftmaxMarginCert)) := do
  let data ← IO.FS.readFile path
  return Pure.parseSoftmaxMarginCert data

/-- Load raw softmax-margin inputs from disk. -/
def loadSoftmaxMarginRaw (path : System.FilePath) :
    IO (Except String (Sigma Pure.SoftmaxMarginRaw)) := do
  let data ← IO.FS.readFile path
  return Pure.parseSoftmaxMarginRaw data

/-- Load a value-range certificate from disk. -/
def loadValueRangeCert (path : System.FilePath) :
    IO (Except String (Sigma ValueRangeCert)) := do
  let data ← IO.FS.readFile path
  return Pure.parseValueRangeCert data

/-- Load a downstream linear certificate from disk. -/
def loadDownstreamLinearCert (path : System.FilePath) :
    IO (Except String DownstreamLinearCert) := do
  let data ← IO.FS.readFile path
  return Pure.parseDownstreamLinearCert data

/-- Load a downstream matrix payload from disk. -/
def loadDownstreamMatrixRaw (path : System.FilePath) :
    IO (Except String (Sigma (fun rows =>
      Sigma (fun cols => Pure.DownstreamMatrixRaw rows cols)))) := do
  let data ← IO.FS.readFile path
  return Pure.parseDownstreamMatrixRaw data

/-- Load a residual-bound certificate from disk. -/
def loadResidualBoundCert (path : System.FilePath) :
    IO (Except String (Sigma (fun n => ResidualBoundCert n))) := do
  let data ← IO.FS.readFile path
  return Pure.parseResidualBoundCert data

/-- Load a residual-interval certificate from disk. -/
def loadResidualIntervalCert (path : System.FilePath) :
    IO (Except String (Sigma (fun n => ResidualIntervalCert n))) := do
  let data ← IO.FS.readFile path
  return Pure.parseResidualIntervalCert data

/-- Load raw value-range inputs from disk. -/
def loadValueRangeRaw (path : System.FilePath) :
    IO (Except String (Sigma Pure.ValueRangeRaw)) := do
  let data ← IO.FS.readFile path
  return Pure.parseValueRangeRaw data

/-- Load induction head inputs from disk. -/
def loadInductionHeadInputs (path : System.FilePath) :
    IO (Except String (Sigma (fun seq =>
      Sigma (fun dModel => Sigma (fun dHead => Model.InductionHeadInputs seq dModel dHead))))) := do
  let data ← IO.FS.readFile path
  return Pure.parseInductionHeadInputs data

private def checkSoftmaxMargin (seq : Nat) (cert : SoftmaxMarginCert seq) :
    IO (Except String Unit) :=
  match seq with
  | 0 => return Except.error "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      let ok := Circuit.checkSoftmaxMarginCert cert
      if ok then
        return Except.ok ()
      else
        return Except.error "softmax-margin certificate rejected"

private def checkValueRange (seq : Nat) (cert : ValueRangeCert seq) :
    IO (Except String Unit) :=
  match seq with
  | 0 => return Except.error "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      let ok := Circuit.checkValueRangeCert cert
      if ok then
        return Except.ok ()
      else
        return Except.error "value-range certificate rejected"

private def parseRatOpt (label : String) (raw? : Option String) :
    Except String (Option Rat) :=
  match raw? with
  | none => Except.ok none
  | some raw =>
      match Pure.parseRat raw with
      | Except.ok v => Except.ok (some v)
      | Except.error msg => Except.error s!"invalid {label}: {msg}"

/-- Check induction certificates and print a short status line. -/
def runInductionCertify (scoresPath : System.FilePath)
    (valuesPath? : Option System.FilePath) (minActive? : Option Nat)
    (minLogitDiffStr? : Option String) (minMarginStr? : Option String)
    (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? => do
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (1 / 2 : Rat)
      if minLogitDiff?.isSome && valuesPath?.isNone then
        IO.eprintln "error: min-logit-diff requires --values"
        return 2
      let parsedScores ← loadSoftmaxMarginCert scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          let scoresOk ← checkSoftmaxMargin seq cert
          match scoresOk with
          | Except.error msg =>
              IO.eprintln s!"error: {msg}"
              return 2
          | Except.ok () =>
              let activeCount := cert.active.card
              let defaultMinActive := max 1 (seq / 8)
              let minActive := minActive?.getD defaultMinActive
              if activeCount < minActive then
                IO.eprintln
                  s!"error: active queries {activeCount} below minimum {minActive}"
                return 2
              if cert.margin < minMargin then
                IO.eprintln
                  s!"error: margin {cert.margin} below minimum {minMargin}"
                return 2
              if maxEps < cert.eps then
                IO.eprintln
                  s!"error: eps {cert.eps} above maximum {maxEps}"
                return 2
              match valuesPath? with
              | none =>
                  IO.println
                    s!"ok: softmax-margin certificate accepted \
                    (seq={seq}, active={activeCount})"
                  return 0
              | some valuesPath =>
                  let parsedValues ← loadValueRangeCert valuesPath
                  match parsedValues with
                  | Except.error msg =>
                      IO.eprintln s!"error: {msg}"
                      return 1
                  | Except.ok ⟨seqVals, certVals⟩ =>
                      if hseq : seqVals ≠ seq then
                        IO.eprintln s!"error: seq mismatch (scores={seq}, values={seqVals})"
                        return 2
                      else
                        have hseq' : seqVals = seq := by
                          exact (not_ne_iff).1 hseq
                        let certVals' : ValueRangeCert seq := by
                          simpa [hseq'] using certVals
                        let valuesOk ← checkValueRange seq certVals'
                        match valuesOk with
                        | Except.error msg =>
                            IO.eprintln s!"error: {msg}"
                            return 2
                        | Except.ok () =>
                            let tol := cert.eps * (certVals'.hi - certVals'.lo)
                            let logitDiffLB? :=
                              Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
                                certVals'.lo certVals'.hi certVals'.vals
                            let effectiveMinLogitDiff :=
                              match minLogitDiff?, certVals'.direction with
                              | some v, _ => some v
                              | none, some _ => some (0 : Rat)
                              | none, none => none
                            match logitDiffLB? with
                            | none =>
                                IO.eprintln "error: empty active set for logit-diff bound"
                                return (2 : UInt32)
                            | some logitDiffLB =>
                                let violation? : Option Rat :=
                                  match effectiveMinLogitDiff with
                                  | none => none
                                  | some minLogitDiff =>
                                      if logitDiffLB < minLogitDiff then
                                        some minLogitDiff
                                      else
                                        none
                                match violation? with
                                | some minLogitDiff =>
                                    IO.eprintln
                                      s!"error: logitDiffLB {logitDiffLB} \
                                      below minimum {minLogitDiff}"
                                    return (2 : UInt32)
                                | none =>
                                IO.println
                                  s!"ok: induction bound certified \
                                  (seq={seq}, active={activeCount}, tol={tol}, \
                                  logitDiffLB={logitDiffLB})"
                                return 0

/-- Build and check induction certificates from raw scores/values. -/
def runInductionCertifySound (scoresPath : System.FilePath)
    (valuesPath : System.FilePath) (minActive? : Option Nat)
    (minLogitDiffStr? : Option String) (minMarginStr? : Option String)
    (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? =>
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (1 / 2 : Rat)
      let parsedScores ← loadSoftmaxMarginRaw scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, raw⟩ =>
          match seq with
          | 0 =>
              IO.eprintln "error: seq must be positive"
              return 2
          | Nat.succ n =>
              let seq := Nat.succ n
              let _ : NeZero seq := ⟨by simp⟩
              match Sound.buildSoftmaxMarginCert? raw.active raw.prev raw.scores raw.weights with
              | none =>
                  IO.eprintln "error: softmax-margin inputs rejected"
                  return 2
              | some ⟨cert, _⟩ =>
                  let activeCount := cert.active.card
                  let defaultMinActive := max 1 (seq / 8)
                  let minActive := minActive?.getD defaultMinActive
                  if activeCount < minActive then
                    IO.eprintln
                      s!"error: active queries {activeCount} below minimum {minActive}"
                    return 2
                  if cert.margin < minMargin then
                    IO.eprintln
                      s!"error: margin {cert.margin} below minimum {minMargin}"
                    return 2
                  if maxEps < cert.eps then
                    IO.eprintln
                      s!"error: eps {cert.eps} above maximum {maxEps}"
                    return 2
                  let parsedValues ← loadValueRangeRaw valuesPath
                  match parsedValues with
                  | Except.error msg =>
                      IO.eprintln s!"error: {msg}"
                      return 1
                  | Except.ok ⟨seqVals, rawVals⟩ =>
                      if hseq : seqVals ≠ seq then
                        IO.eprintln
                          s!"error: seq mismatch (scores={seq}, values={seqVals})"
                        return 2
                      else
                        have hseq' : seqVals = seq := by
                          exact (not_ne_iff).1 hseq
                        let rawVals' : Pure.ValueRangeRaw seq := by
                          simpa [hseq'] using rawVals
                        match Sound.buildValueRangeCert? rawVals'.vals rawVals'.direction with
                        | none =>
                            IO.eprintln "error: value-range inputs rejected"
                            return 2
                        | some ⟨certVals, _⟩ =>
                            let tol := cert.eps * (certVals.hi - certVals.lo)
                            let logitDiffLB? :=
                              Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
                                certVals.lo certVals.hi certVals.vals
                            let effectiveMinLogitDiff :=
                              match minLogitDiff?, certVals.direction with
                              | some v, _ => some v
                              | none, some _ => some (0 : Rat)
                              | none, none => none
                            match logitDiffLB? with
                            | none =>
                                IO.eprintln "error: empty active set for logit-diff bound"
                                return 2
                            | some logitDiffLB =>
                                let violation? : Option Rat :=
                                  match effectiveMinLogitDiff with
                                  | none => none
                                  | some minLogitDiff =>
                                      if logitDiffLB < minLogitDiff then
                                        some minLogitDiff
                                      else
                                        none
                                match violation? with
                                | some minLogitDiff =>
                                    IO.eprintln
                                      s!"error: logitDiffLB {logitDiffLB} \
                                      below minimum {minLogitDiff}"
                                    return 2
                                | none =>
                                IO.println
                                  s!"ok: induction bound certified \
                                  (seq={seq}, active={activeCount}, \
                                  tol={tol}, logitDiffLB={logitDiffLB})"
                                return 0

/-- Check end-to-end induction certificates with a downstream error bound. -/
def runInductionCertifyEndToEnd (scoresPath : System.FilePath)
    (valuesPath : System.FilePath) (downstreamPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? => do
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (1 / 2 : Rat)
      let parsedScores ← loadSoftmaxMarginCert scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          let scoresOk ← checkSoftmaxMargin seq cert
          match scoresOk with
          | Except.error msg =>
              IO.eprintln s!"error: {msg}"
              return 2
          | Except.ok () =>
              let activeCount := cert.active.card
              let defaultMinActive := max 1 (seq / 8)
              let minActive := minActive?.getD defaultMinActive
              if activeCount < minActive then
                IO.eprintln
                  s!"error: active queries {activeCount} below minimum {minActive}"
                return 2
              if cert.margin < minMargin then
                IO.eprintln
                  s!"error: margin {cert.margin} below minimum {minMargin}"
                return 2
              if maxEps < cert.eps then
                IO.eprintln
                  s!"error: eps {cert.eps} above maximum {maxEps}"
                return 2
              let parsedValues ← loadValueRangeCert valuesPath
              match parsedValues with
              | Except.error msg =>
                  IO.eprintln s!"error: {msg}"
                  return 1
              | Except.ok ⟨seqVals, certVals⟩ =>
                  if hseq : seqVals ≠ seq then
                    IO.eprintln s!"error: seq mismatch (scores={seq}, values={seqVals})"
                    return 2
                  else
                    have hseq' : seqVals = seq := by
                      exact (not_ne_iff).1 hseq
                    let certVals' : ValueRangeCert seq := by
                      simpa [hseq'] using certVals
                    let valuesOk ← checkValueRange seq certVals'
                    match valuesOk with
                    | Except.error msg =>
                        IO.eprintln s!"error: {msg}"
                        return 2
                    | Except.ok () =>
                        let logitDiffLB? :=
                          Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
                            certVals'.lo certVals'.hi certVals'.vals
                        let effectiveMinLogitDiff :=
                          match minLogitDiff?, certVals'.direction with
                          | some v, _ => some v
                          | none, some _ => some (0 : Rat)
                          | none, none => none
                        match logitDiffLB? with
                        | none =>
                            IO.eprintln "error: empty active set for logit-diff bound"
                            return (2 : UInt32)
                        | some logitDiffLB =>
                            let parsedDownstream ← loadDownstreamLinearCert downstreamPath
                            match parsedDownstream with
                            | Except.error msg =>
                                IO.eprintln s!"error: {msg}"
                                return 1
                            | Except.ok downstream =>
                                let downstreamOk := Circuit.checkDownstreamLinearCert downstream
                                if downstreamOk then
                                  let finalLB := logitDiffLB - downstream.error
                                  let violation? : Option Rat :=
                                    match effectiveMinLogitDiff with
                                    | none => none
                                    | some minLogitDiff =>
                                        if finalLB < minLogitDiff then
                                          some minLogitDiff
                                        else
                                          none
                                  match violation? with
                                  | some minLogitDiff =>
                                      IO.eprintln
                                        s!"error: end-to-end logitDiffLB {finalLB} \
                                        below minimum {minLogitDiff}"
                                      return (2 : UInt32)
                                  | none =>
                                      IO.println
                                        s!"ok: end-to-end induction bound certified \
                                        (seq={seq}, active={activeCount}, \
                                        logitDiffLB={logitDiffLB}, \
                                        downstreamError={downstream.error}, \
                                        finalLB={finalLB})"
                                  return 0
                                else
                                  IO.eprintln "error: downstream certificate rejected"
                                  return 2

/-- Check end-to-end induction certificates with a downstream matrix. -/
def runInductionCertifyEndToEndMatrix (scoresPath : System.FilePath)
    (valuesPath : System.FilePath) (matrixPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? => do
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (1 / 2 : Rat)
      let parsedScores ← loadSoftmaxMarginCert scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          let scoresOk ← checkSoftmaxMargin seq cert
          match scoresOk with
          | Except.error msg =>
              IO.eprintln s!"error: {msg}"
              return 2
          | Except.ok () =>
              let activeCount := cert.active.card
              let defaultMinActive := max 1 (seq / 8)
              let minActive := minActive?.getD defaultMinActive
              if activeCount < minActive then
                IO.eprintln
                  s!"error: active queries {activeCount} below minimum {minActive}"
                return 2
              if cert.margin < minMargin then
                IO.eprintln
                  s!"error: margin {cert.margin} below minimum {minMargin}"
                return 2
              if maxEps < cert.eps then
                IO.eprintln
                  s!"error: eps {cert.eps} above maximum {maxEps}"
                return 2
              let parsedValues ← loadValueRangeCert valuesPath
              match parsedValues with
              | Except.error msg =>
                  IO.eprintln s!"error: {msg}"
                  return 1
              | Except.ok ⟨seqVals, certVals⟩ =>
                  if hseq : seqVals ≠ seq then
                    IO.eprintln s!"error: seq mismatch (scores={seq}, values={seqVals})"
                    return 2
                  else
                    have hseq' : seqVals = seq := by
                      exact (not_ne_iff).1 hseq
                    let certVals' : ValueRangeCert seq := by
                      simpa [hseq'] using certVals
                    let valuesOk ← checkValueRange seq certVals'
                    match valuesOk with
                    | Except.error msg =>
                        IO.eprintln s!"error: {msg}"
                        return 2
                    | Except.ok () =>
                        let logitDiffLB? :=
                          Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
                            certVals'.lo certVals'.hi certVals'.vals
                        let effectiveMinLogitDiff :=
                          match minLogitDiff?, certVals'.direction with
                          | some v, _ => some v
                          | none, some _ => some (0 : Rat)
                          | none, none => none
                        match logitDiffLB? with
                        | none =>
                            IO.eprintln "error: empty active set for logit-diff bound"
                            return (2 : UInt32)
                        | some logitDiffLB =>
                            let parsedMatrix ← loadDownstreamMatrixRaw matrixPath
                            match parsedMatrix with
                            | Except.error msg =>
                                IO.eprintln s!"error: {msg}"
                                return 1
                            | Except.ok ⟨rows, ⟨cols, raw⟩⟩ =>
                                let inputBound := raw.inputBound
                                if hneg : inputBound < 0 then
                                  IO.eprintln
                                    s!"error: input-bound {inputBound} must be nonnegative"
                                  return 2
                                else
                                  have hinput : 0 ≤ inputBound := by
                                    exact le_of_not_gt hneg
                                  let W : Matrix (Fin rows) (Fin cols) Rat := raw.entries
                                  let downstream :=
                                    (Sound.Bounds.buildDownstreamLinearCert W inputBound hinput).1
                                  let finalLB := logitDiffLB - downstream.error
                                  let violation? : Option Rat :=
                                    match effectiveMinLogitDiff with
                                    | none => none
                                    | some minLogitDiff =>
                                        if finalLB < minLogitDiff then
                                          some minLogitDiff
                                        else
                                          none
                                  match violation? with
                                  | some minLogitDiff =>
                                      IO.eprintln
                                        s!"error: end-to-end logitDiffLB {finalLB} \
                                        below minimum {minLogitDiff}"
                                      return (2 : UInt32)
                                  | none =>
                                      IO.println
                                        s!"ok: end-to-end induction bound certified \
                                        (seq={seq}, active={activeCount}, \
                                        logitDiffLB={logitDiffLB}, \
                                        downstreamError={downstream.error}, \
                                        finalLB={finalLB})"
                                      return 0

/-- Check end-to-end induction certificates using a model file and residual bounds. -/
def runInductionCertifyEndToEndModel (scoresPath : System.FilePath)
    (valuesPath : System.FilePath) (modelPath : System.FilePath)
    (residualIntervalPath : System.FilePath) (minActive? : Option Nat)
    (minLogitDiffStr? : Option String) (minMarginStr? : Option String)
    (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? => do
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (1 / 2 : Rat)
      let parsedScores ← loadSoftmaxMarginCert scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          let scoresOk ← checkSoftmaxMargin seq cert
          match scoresOk with
          | Except.error msg =>
              IO.eprintln s!"error: {msg}"
              return 2
          | Except.ok () =>
              let activeCount := cert.active.card
              let defaultMinActive := max 1 (seq / 8)
              let minActive := minActive?.getD defaultMinActive
              if activeCount < minActive then
                IO.eprintln
                  s!"error: active queries {activeCount} below minimum {minActive}"
                return 2
              if cert.margin < minMargin then
                IO.eprintln
                  s!"error: margin {cert.margin} below minimum {minMargin}"
                return 2
              if maxEps < cert.eps then
                IO.eprintln
                  s!"error: eps {cert.eps} above maximum {maxEps}"
                return 2
              let parsedValues ← loadValueRangeCert valuesPath
              match parsedValues with
              | Except.error msg =>
                  IO.eprintln s!"error: {msg}"
                  return 1
              | Except.ok ⟨seqVals, certVals⟩ =>
                  if hseq : seqVals ≠ seq then
                    IO.eprintln s!"error: seq mismatch (scores={seq}, values={seqVals})"
                    return 2
                  else
                    have hseq' : seqVals = seq := by
                      exact (not_ne_iff).1 hseq
                    let certVals' : ValueRangeCert seq := by
                      simpa [hseq'] using certVals
                    let valuesOk ← checkValueRange seq certVals'
                    match valuesOk with
                    | Except.error msg =>
                        IO.eprintln s!"error: {msg}"
                        return 2
                    | Except.ok () =>
                        let logitDiffLB? :=
                          Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
                            certVals'.lo certVals'.hi certVals'.vals
                        let effectiveMinLogitDiff :=
                          match minLogitDiff?, certVals'.direction with
                          | some v, _ => some v
                          | none, some _ => some (0 : Rat)
                          | none, none => none
                        match logitDiffLB? with
                        | none =>
                            IO.eprintln "error: empty active set for logit-diff bound"
                            return (2 : UInt32)
                        | some logitDiffLB =>
                            match certVals'.direction with
                            | none =>
                                IO.eprintln
                                  "error: value-range certificate missing direction \
                                  metadata"
                                return 2
                            | some dirSpec =>
                                let data ← IO.FS.readBinFile modelPath
                                match NfptPure.parseHeader data with
                                | Except.error msg =>
                                    IO.eprintln s!"error: {msg}"
                                    return 1
                                | Except.ok ⟨header, start⟩ =>
                                    let parsedResidual ←
                                      loadResidualIntervalCert residualIntervalPath
                                    match parsedResidual with
                                    | Except.error msg =>
                                        IO.eprintln s!"error: {msg}"
                                        return 1
                                    | Except.ok ⟨dim, residualCert⟩ =>
                                        if hdim : dim = header.modelDim then
                                          let residualCert' :
                                              ResidualIntervalCert header.modelDim := by
                                            simpa [hdim] using residualCert
                                          let residualOk :=
                                            Circuit.checkResidualIntervalCert residualCert'
                                          if residualOk then
                                            let dirPos := dirSpec.target
                                            let dirNeg := dirSpec.negative
                                            match
                                              NfptPure.readUnembedColumn data start header dirPos
                                            with
                                            | Except.error msg =>
                                                IO.eprintln s!"error: {msg}"
                                                return 1
                                            | Except.ok colTarget =>
                                                match
                                                  NfptPure.readUnembedColumn
                                                    data start header dirNeg
                                                with
                                                | Except.error msg =>
                                                    IO.eprintln s!"error: {msg}"
                                                    return 1
                                                | Except.ok colNeg =>
                                                    let dirVec :
                                                        Fin header.modelDim → Rat :=
                                                      fun i => colTarget i - colNeg i
                                                    let downstreamError :=
                                                      Sound.Bounds.dotIntervalAbsBound
                                                        dirVec residualCert'.lo residualCert'.hi
                                                    let finalLB := logitDiffLB - downstreamError
                                                    let violation? : Option Rat :=
                                                      match effectiveMinLogitDiff with
                                                      | none => none
                                                      | some minLogitDiff =>
                                                          if finalLB < minLogitDiff then
                                                            some minLogitDiff
                                                          else
                                                            none
                                                    match violation? with
                                                    | some minLogitDiff =>
                                                        IO.eprintln
                                                          s!"error: end-to-end logitDiffLB \
                                                          {finalLB} below minimum \
                                                          {minLogitDiff}"
                                                        return (2 : UInt32)
                                                    | none =>
                                                        IO.println
                                                          s!"ok: end-to-end induction \
                                                          bound certified (seq={seq}, \
                                                          active={activeCount}, \
                                                          logitDiffLB={logitDiffLB}, \
                                                          downstreamError={downstreamError}, \
                                                          finalLB={finalLB})"
                                                        return 0
                                          else
                                            IO.eprintln
                                              "error: residual-interval certificate rejected"
                                            return 2
                                        else
                                          IO.eprintln
                                            s!"error: residual interval dim {dim} \
                                            does not match model dim {header.modelDim}"
                                          return 2

private def checkInductionHeadInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (minActive? : Option Nat) (minLogitDiff? : Option Rat)
    (minMargin maxEps : Rat) : IO UInt32 := do
  match seq with
  | 0 =>
      IO.eprintln "error: seq must be positive"
      return 2
  | Nat.succ n =>
      let seq := Nat.succ n
      let _ : NeZero seq := ⟨by simp⟩
      match Sound.buildInductionCertFromHead? inputs with
      | none =>
          IO.eprintln "error: head inputs rejected"
          return 2
      | some ⟨cert, _hcert⟩ =>
          let activeCount := cert.active.card
          let defaultMinActive := max 1 (seq / 8)
          let minActive := minActive?.getD defaultMinActive
          if activeCount < minActive then
            IO.eprintln
              s!"error: active queries {activeCount} below minimum {minActive}"
            return 2
          if cert.margin < minMargin then
            IO.eprintln
              s!"error: margin {cert.margin} below minimum {minMargin}"
            return 2
          if maxEps < cert.eps then
            IO.eprintln
              s!"error: eps {cert.eps} above maximum {maxEps}"
            return 2
          let tol := cert.eps * (cert.values.hi - cert.values.lo)
          let logitDiffLB? :=
            Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
              cert.values.lo cert.values.hi cert.values.vals
          let effectiveMinLogitDiff :=
            match minLogitDiff? with
            | some v => some v
            | none => some (0 : Rat)
          match logitDiffLB? with
          | none =>
              IO.eprintln "error: empty active set for logit-diff bound"
              return 2
          | some logitDiffLB =>
              let violation? : Option Rat :=
                match effectiveMinLogitDiff with
                | none => none
                | some minLogitDiff =>
                    if logitDiffLB < minLogitDiff then
                      some minLogitDiff
                    else
                      none
              match violation? with
              | some minLogitDiff =>
                  IO.eprintln
                    s!"error: logitDiffLB {logitDiffLB} \
                    below minimum {minLogitDiff}"
                  return 2
              | none =>
                  IO.println
                    s!"ok: induction bound certified \
                    (seq={seq}, active={activeCount}, \
                    tol={tol}, logitDiffLB={logitDiffLB})"
                  return 0

/-- Build and check induction certificates from exact head inputs. -/
def runInductionCertifyHead (inputsPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? =>
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (1 / 2 : Rat)
      let parsedInputs ← loadInductionHeadInputs inputsPath
      match parsedInputs with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨_seq, ⟨_dModel, ⟨_dHead, inputs⟩⟩⟩ =>
          checkInductionHeadInputs inputs minActive? minLogitDiff? minMargin maxEps

/-- Build and check induction certificates from a model binary. -/
def runInductionCertifyHeadModel (modelPath : System.FilePath)
    (layer head period dirTarget dirNegative : Nat)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? =>
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (1 / 2 : Rat)
      let data ← IO.FS.readBinFile modelPath
      match NfptPure.parseHeader data with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨header, start⟩ =>
          match
            NfptPure.readInductionHeadInputs
              data start header layer head period dirTarget dirNegative
          with
          | Except.error msg =>
              IO.eprintln s!"error: {msg}"
              return 1
          | Except.ok inputs =>
              checkInductionHeadInputs inputs minActive? minLogitDiff? minMargin maxEps

end IO

end Nfp
