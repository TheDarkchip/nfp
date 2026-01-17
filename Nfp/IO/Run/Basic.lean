-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Checks
public import Nfp.IO.Derive
public import Nfp.IO.HeadScore
public import Nfp.IO.InductionHead
public import Nfp.IO.Loaders
public import Nfp.IO.NfptPure
public import Nfp.IO.Timing
public import Nfp.IO.Util
public import Nfp.Circuit.Cert.DownstreamLinear
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.Circuit.Cert.ResidualBound
public import Nfp.Circuit.Cert.ResidualInterval
public import Nfp.Sound.Bounds.MatrixNorm
public import Nfp.Sound.Bounds.Transformer
public import Nfp.Sound.Induction
public import Nfp.Sound.Induction.HeadBounds
public import Nfp.Sound.Induction.LogitDiff
public import Nfp.Sound.Linear.FinFold

/-!
IO entrypoints used by the CLI.
-/

public section

namespace Nfp
namespace IO
open Nfp.Circuit

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
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      if minLogitDiff?.isSome && valuesPath?.isNone then
        IO.eprintln "error: min-logit-diff requires --values"
        return 2
      let parsedScores ← timePhase "load softmax cert" <|
        loadSoftmaxMarginCert scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          let scoresOk ← timePhase "check softmax cert" <|
            checkSoftmaxMargin seq cert
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
  warnDeprecated
    "certify_sound builds certificates from raw scores/values; use explicit certs \
    via `nfp induction certify` or `nfp induction head_cert_check`."
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
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
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
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let parsedScores ← timePhase "load softmax cert" <|
        loadSoftmaxMarginCert scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          let scoresOk ← timePhase "check softmax cert" <|
            checkSoftmaxMargin seq cert
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
              let parsedValues ← timePhase "load value cert" <|
                loadValueRangeCert valuesPath
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
                    let valuesOk ← timePhase "check value cert" <|
                      checkValueRange seq certVals'
                    match valuesOk with
                    | Except.error msg =>
                        IO.eprintln s!"error: {msg}"
                        return 2
                    | Except.ok () =>
                        let logitDiffLB? ← timePure "logit-diff lower bound" (fun () =>
                          Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
                            certVals'.lo certVals'.hi certVals'.vals)
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
  warnDeprecated
    "certify_end_to_end_matrix builds downstream bounds from a raw matrix payload; \
    use a downstream cert instead."
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
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let parsedScores ← timePhase "load softmax cert" <|
        loadSoftmaxMarginCert scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          let scoresOk ← timePhase "check softmax cert" <|
            checkSoftmaxMargin seq cert
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
              let parsedValues ← timePhase "load value cert" <|
                loadValueRangeCert valuesPath
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
                    let valuesOk ← timePhase "check value cert" <|
                      checkValueRange seq certVals'
                    match valuesOk with
                    | Except.error msg =>
                        IO.eprintln s!"error: {msg}"
                        return 2
                    | Except.ok () =>
                        let logitDiffLB? ← timePure "logit-diff lower bound" (fun () =>
                          Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
                            certVals'.lo certVals'.hi certVals'.vals)
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
/-- Check end-to-end induction certificates using a model file and residual bounds
    (loaded from disk or derived from the model). -/
def runInductionCertifyEndToEndModel (scoresPath : System.FilePath)
    (valuesPath : System.FilePath) (modelPath : System.FilePath)
    (residualIntervalPath? : Option System.FilePath)
    (layer? : Option Nat) (head? : Option Nat) (period? : Option Nat)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  warnDeprecated
    "certify_end_to_end_model derives residual bounds from a model file; \
    use an explicit residual-interval cert instead."
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
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let parsedScores ← timePhase "load softmax cert" <|
        loadSoftmaxMarginCert scoresPath
      match parsedScores with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          let scoresOk ← timePhase "check softmax cert" <|
            checkSoftmaxMargin seq cert
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
              let parsedValues ← timePhase "load value cert" <|
                loadValueRangeCert valuesPath
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
                    let valuesOk ← timePhase "check value cert" <|
                      checkValueRange seq certVals'
                    match valuesOk with
                    | Except.error msg =>
                        IO.eprintln s!"error: {msg}"
                        return 2
                    | Except.ok () =>
                        let logitDiffLB? ← timePure "logit-diff lower bound" (fun () =>
                          Circuit.logitDiffLowerBound cert.active cert.prev cert.eps
                            certVals'.lo certVals'.hi certVals'.vals)
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
                                let data ← timePhase "read model file" <|
                                  IO.FS.readBinFile modelPath
                                let headerE ← timePure "parse model header" (fun () =>
                                  NfptPure.parseHeader data)
                                match headerE with
                                | Except.error msg =>
                                    IO.eprintln s!"error: {msg}"
                                    return 1
                                | Except.ok ⟨header, start⟩ =>
                                    if hseq : header.seqLen = seq then
                                      let active? : Option (Finset (Fin header.seqLen)) :=
                                        if hactive : cert.active.Nonempty then
                                          some (by simpa [hseq] using cert.active)
                                        else
                                          none
                                      let residualCertE : Except String
                                          (ResidualIntervalCert header.modelDim) ←
                                        match residualIntervalPath? with
                                        | some residualIntervalPath => do
                                            let parsedResidual ←
                                              timePhase "load residual interval" <|
                                                loadResidualIntervalCert residualIntervalPath
                                            match parsedResidual with
                                            | Except.error msg => pure (Except.error msg)
                                            | Except.ok ⟨dim, residualCert⟩ =>
                                                if hdim : dim = header.modelDim then
                                                  let residualCert' :
                                                      ResidualIntervalCert header.modelDim := by
                                                    simpa [hdim] using residualCert
                                                  pure (Except.ok residualCert')
                                                else
                                                  pure (Except.error
                                                    s!"residual interval dim {dim} \
                                                    does not match model dim {header.modelDim}")
                                        | none =>
                                            deriveResidualIntervalFromModel data start header
                                              active?
                                      match residualCertE with
                                      | Except.error msg =>
                                          IO.eprintln s!"error: {msg}"
                                          return 1
                                      | Except.ok residualCert' =>
                                          let residualOk ←
                                            timePure "check residual interval" (fun () =>
                                              Circuit.checkResidualIntervalCert residualCert')
                                          if residualOk then
                                            let dirPos := dirSpec.target
                                            let dirNeg := dirSpec.negative
                                            if layer?.isSome != head?.isSome then
                                              IO.eprintln
                                                "error: --layer and --head must be provided \
                                                together"
                                              return 2
                                            let headChoice? : Option (Nat × Nat) :=
                                              match layer?, head? with
                                              | some layer, some head => some (layer, head)
                                              | _, _ => none
                                            if period?.isSome && headChoice?.isNone then
                                              IO.eprintln
                                                "warning: --period ignored without \
                                                --layer/--head"
                                            let colTargetE ←
                                              timePure "read unembed column target" (fun () =>
                                                NfptPure.readUnembedColumn
                                                  data start header dirPos)
                                            match colTargetE with
                                            | Except.error msg =>
                                                IO.eprintln s!"error: {msg}"
                                                return 1
                                            | Except.ok colTarget =>
                                                let colNegE ←
                                                  timePure "read unembed column negative" (fun () =>
                                                    NfptPure.readUnembedColumn
                                                      data start header dirNeg)
                                                match colNegE with
                                                | Except.error msg =>
                                                    IO.eprintln s!"error: {msg}"
                                                    return 1
                                                | Except.ok colNeg =>
                                                    let dirVec :
                                                        Fin header.modelDim → Rat :=
                                                      fun i => colTarget i - colNeg i
                                                    let dotIntervalAbs :=
                                                      Sound.Bounds.dotIntervalAbsBound
                                                    let intervalErrorFromHead? :
                                                        Model.InductionHeadInputs
                                                          seq header.modelDim header.headDim →
                                                        ResidualIntervalCert header.modelDim →
                                                        Option Rat :=
                                                      fun inputs residual => by
                                                        classical
                                                        match hseq0 : seq with
                                                        | 0 => exact none
                                                        | Nat.succ n =>
                                                            let _ : NeZero seq := by
                                                              exact ⟨by simp [hseq0]⟩
                                                            match
                                                              Sound.buildHeadOutputIntervalFromHead?
                                                                inputs with
                                                            | none => exact none
                                                            | some result =>
                                                                exact some
                                                                  (dotIntervalAbs
                                                                    dirVec
                                                                    (fun i =>
                                                                      residual.lo i -
                                                                        result.cert.hi i)
                                                                    (fun i =>
                                                                      residual.hi i -
                                                                        result.cert.lo i))
                                                    let downstreamError ←
                                                      timePure "downstream error" (fun () =>
                                                        dotIntervalAbs
                                                          dirVec
                                                          residualCert'.lo
                                                          residualCert'.hi)
                                                    let finalLB := logitDiffLB - downstreamError
                                                    let intervalError? ←
                                                      match headChoice? with
                                                      | none => pure none
                                                      | some (layer, head) => do
                                                          let inputsE ←
                                                            timePure "read head inputs" (fun () =>
                                                              NfptPure.readInductionHeadInputs
                                                                data start header layer head
                                                                dirPos dirNeg period? false)
                                                          match inputsE with
                                                          | Except.error msg =>
                                                              IO.eprintln s!"warning: {msg}"
                                                              pure none
                                                          | Except.ok inputs =>
                                                              let inputs' :
                                                                  Model.InductionHeadInputs
                                                                    seq header.modelDim
                                                                    header.headDim := by
                                                                simpa [hseq] using inputs
                                                              let inputsAligned :
                                                                  Model.InductionHeadInputs
                                                                    seq header.modelDim
                                                                    header.headDim :=
                                                                { inputs' with
                                                                  active := cert.active
                                                                  prev := cert.prev }
                                                              let intervalError? ←
                                                                timePure
                                                                  "head output interval"
                                                                  (fun () =>
                                                                    intervalErrorFromHead?
                                                                      inputsAligned
                                                                      residualCert')
                                                              match intervalError? with
                                                              | none =>
                                                                  IO.eprintln
                                                                    "warning: head output interval \
                                                                    rejected"
                                                                  pure none
                                                              | some intervalError =>
                                                                  pure (some intervalError)
                                                    let intervalLB? :=
                                                      intervalError?.map (fun err =>
                                                        logitDiffLB - err)
                                                    let effectiveLB :=
                                                      match intervalLB? with
                                                      | some intervalLB => max finalLB intervalLB
                                                      | none => finalLB
                                                    let violation? : Option Rat :=
                                                      match effectiveMinLogitDiff with
                                                      | none => none
                                                      | some minLogitDiff =>
                                                          if effectiveLB < minLogitDiff then
                                                            some minLogitDiff
                                                          else
                                                            none
                                                    match violation? with
                                                    | some minLogitDiff =>
                                                        IO.eprintln
                                                          s!"error: end-to-end bound \
                                                          {effectiveLB} below minimum \
                                                          {minLogitDiff}"
                                                        return (2 : UInt32)
                                                    | none =>
                                                        match intervalLB? with
                                                        | none =>
                                                            IO.println
                                                              s!"ok: end-to-end induction \
                                                              bound certified (seq={seq}, \
                                                              active={activeCount}, \
                                                              logitDiffLB={logitDiffLB}, \
                                                              downstreamError={downstreamError}, \
                                                              finalLB={finalLB})"
                                                        | some intervalLB =>
                                                            let intervalError :=
                                                              logitDiffLB - intervalLB
                                                            IO.println
                                                              s!"ok: end-to-end induction \
                                                              bound certified (seq={seq}, \
                                                              active={activeCount}, \
                                                              logitDiffLB={logitDiffLB}, \
                                                              downstreamError={downstreamError}, \
                                                              finalLB={finalLB}, \
                                                              intervalError={intervalError}, \
                                                              intervalLB={intervalLB}, \
                                                              effectiveLB={effectiveLB})"
                                                        return 0
                                          else
                                            IO.eprintln
                                              "error: residual-interval certificate rejected"
                                            return 2
                                    else
                                      IO.eprintln
                                        s!"error: model seq {header.seqLen} \
                                        does not match cert seq {seq}"
                                      return 2
end IO
end Nfp
