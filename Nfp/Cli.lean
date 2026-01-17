-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Cli
import Nfp.IO

/-!
Minimal CLI surface for the NFP rewrite.
-/

public section

open Cli

namespace Nfp

/-- Human-readable version string for the CLI. -/
def versionString : String := "0.1.0-tabula"

/-- Print the CLI version. -/
def runVersion (_p : Parsed) : IO UInt32 := do
  IO.println s!"nfp version {versionString}"
  return 0

/-- The version subcommand. -/
def versionCmd : Cmd := `[Cli|
  version VIA runVersion;
  "Print the NFP version."
]

private def parseDirectionSpec (raw : String) : Except String (Nat × Nat) := do
  let partsComma := raw.splitOn ","
  let parts := if partsComma.length = 2 then partsComma else raw.splitOn ":"
  match parts with
  | [targetRaw, negativeRaw] =>
      match targetRaw.toNat?, negativeRaw.toNat? with
      | some target, some negative => pure (target, negative)
      | _, _ => throw s!"direction must be two natural numbers (got '{raw}')"
  | _ =>
      throw s!"direction must look like \"target,negative\" (got '{raw}')"

private def parseSplitPreset (raw : String) :
    Except String (Option Nat × Option Nat × Option Nat × Option Nat) := do
  let key := raw.toLower
  match key with
  | "balanced" | "default" => pure (none, none, none, none)
  | "fast" => pure (some 0, some 0, some 0, some 0)
  | "tight" => pure (some 4, some 4, some 2, some 16)
  | _ =>
      throw s!"unknown preset '{raw}' (expected: fast, balanced, tight)"

private def toZeroBased (label : String) (idx : Nat) (zeroBased : Bool) :
    Except String Nat := do
  if zeroBased then
    pure idx
  else
    if idx = 0 then
      throw s!"{label} must be >= 1 for 1-based indexing (use --zero-based for 0-based)"
    else
      pure (idx - 1)

private def runInductionCertifyUnified (requireNonvacuous : Bool) (p : Parsed) : IO UInt32 := do
  let inputsPath? := (p.flag? "inputs").map (·.as! String)
  let modelPath? := (p.flag? "model").map (·.as! String)
  let layer? := (p.flag? "layer").map (·.as! Nat)
  let head? := (p.flag? "head").map (·.as! Nat)
  let period? := (p.flag? "period").map (·.as! Nat)
  let prevShift := p.hasFlag "prev-shift"
  let directionStr? := (p.flag? "direction").map (·.as! String)
  let presetStr? := (p.flag? "preset").map (·.as! String)
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let timing? := (p.flag? "timing").map (·.as! Nat)
  let heartbeatMs? := (p.flag? "heartbeat-ms").map (·.as! Nat)
  let skipLogitDiff := p.hasFlag "skip-logit-diff"
  let zeroBased := p.hasFlag "zero-based"
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  let presetE :=
    match presetStr? with
    | none => Except.ok (none, none, none, none)
    | some raw => parseSplitPreset raw
  let directionE :=
    match directionStr? with
    | none => Except.ok none
    | some raw => (parseDirectionSpec raw).map some
  match presetE, directionE with
  | Except.error msg, _ => fail msg
  | _, Except.error msg => fail msg
  | Except.ok ⟨splitBudgetQ?, splitBudgetK?, splitBudgetDiffBase?, splitBudgetDiffRefined?⟩,
      Except.ok direction? =>
      match inputsPath?, modelPath? with
      | some inputsPath, none =>
          if layer?.isSome || head?.isSome || period?.isSome || prevShift then
            fail "--layer/--head/--period/--prev-shift are only valid with --model"
          else if direction?.isSome then
            fail "--direction is only valid with --model"
          else if requireNonvacuous && skipLogitDiff then
            fail "--skip-logit-diff is not allowed with certify_nonvacuous"
          else if requireNonvacuous then
            IO.runInductionCertifyHeadNonvacuous inputsPath minActive? minLogitDiffStr?
              minMarginStr? maxEpsStr? timing? heartbeatMs?
              splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
          else
            IO.runInductionCertifyHead inputsPath minActive? minLogitDiffStr?
              minMarginStr? maxEpsStr? timing? heartbeatMs?
              splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
              skipLogitDiff
      | none, some modelPath =>
          match layer?, head? with
          | some layer, some head =>
              let layerE := toZeroBased "layer" layer zeroBased
              let headE := toZeroBased "head" head zeroBased
              match layerE, headE with
              | Except.error msg, _ => fail msg
              | _, Except.error msg => fail msg
              | Except.ok layer', Except.ok head' =>
                match direction? with
                | some ⟨dirTarget, dirNegative⟩ =>
                    if requireNonvacuous && skipLogitDiff then
                      fail "--skip-logit-diff is not allowed with certify_nonvacuous"
                    else if requireNonvacuous then
                      IO.runInductionCertifyHeadModelNonvacuous modelPath layer' head' dirTarget
                        dirNegative period? prevShift minActive? minLogitDiffStr? minMarginStr?
                        maxEpsStr?
                        timing? heartbeatMs?
                        splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
                    else
                      IO.runInductionCertifyHeadModel modelPath layer' head' dirTarget dirNegative
                        period? prevShift minActive? minLogitDiffStr? minMarginStr? maxEpsStr?
                        timing? heartbeatMs?
                        splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
                        skipLogitDiff
                | none =>
                    if requireNonvacuous && skipLogitDiff then
                      fail "--skip-logit-diff is not allowed with certify_nonvacuous"
                    else if requireNonvacuous then
                      IO.runInductionCertifyHeadModelAutoNonvacuous modelPath layer' head' period?
                        prevShift minActive? minLogitDiffStr? minMarginStr? maxEpsStr?
                        timing? heartbeatMs?
                        splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
                    else
                      IO.runInductionCertifyHeadModelAuto modelPath layer' head' period? prevShift
                        minActive? minLogitDiffStr? minMarginStr? maxEpsStr?
                        timing? heartbeatMs?
                        splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
                        skipLogitDiff
          | _, _ =>
              fail "--layer and --head are required with --model"
      | none, none =>
          fail "provide exactly one of --inputs or --model"
      | some _, some _ =>
          fail "provide exactly one of --inputs or --model"

private def runInductionCertifySimple (p : Parsed) : IO UInt32 :=
  runInductionCertifyUnified false p

private def runInductionCertifyNonvacuousSimple (p : Parsed) : IO UInt32 :=
  runInductionCertifyUnified true p

private def runInductionIntervalSimple (p : Parsed) : IO UInt32 := do
  let inputsPath? := (p.flag? "inputs").map (·.as! String)
  let modelPath? := (p.flag? "model").map (·.as! String)
  let layer? := (p.flag? "layer").map (·.as! Nat)
  let head? := (p.flag? "head").map (·.as! Nat)
  let period? := (p.flag? "period").map (·.as! Nat)
  let prevShift := p.hasFlag "prev-shift"
  let directionStr? := (p.flag? "direction").map (·.as! String)
  let outPath? := (p.flag? "out").map (·.as! String)
  let zeroBased := p.hasFlag "zero-based"
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  let directionE :=
    match directionStr? with
    | none => Except.ok none
    | some raw => (parseDirectionSpec raw).map some
  match directionE with
  | Except.error msg => fail msg
  | Except.ok direction? =>
      match inputsPath?, modelPath? with
      | some inputsPath, none =>
          if layer?.isSome || head?.isSome || period?.isSome || prevShift then
            fail "--layer/--head/--period/--prev-shift are only valid with --model"
          else if direction?.isSome then
            fail "--direction is only valid with --model"
          else
            IO.runInductionHeadInterval inputsPath outPath?
      | none, some modelPath =>
          match layer?, head?, direction? with
          | some layer, some head, some ⟨dirTarget, dirNegative⟩ =>
              let layerE := toZeroBased "layer" layer zeroBased
              let headE := toZeroBased "head" head zeroBased
              match layerE, headE with
              | Except.error msg, _ => fail msg
              | _, Except.error msg => fail msg
              | Except.ok layer', Except.ok head' =>
                  IO.runInductionHeadIntervalModel modelPath layer' head' dirTarget dirNegative
                    period? prevShift outPath?
          | _, _, none =>
              fail "--direction is required with --model (use \"target,negative\")"
          | _, _, _ =>
              fail "--layer and --head are required with --model"
      | none, none =>
          fail "provide exactly one of --inputs or --model"
      | some _, some _ =>
          fail "provide exactly one of --inputs or --model"

/-- `nfp induction certify` subcommand (streamlined). -/
def inductionCertifySimpleCmd : Cmd := `[Cli|
  certify VIA runInductionCertifySimple;
  "Check induction head certificates from inputs or a model file."
  FLAGS:
    inputs : String; "Path to the induction head input file (use either --inputs or --model)."
    model : String; "Path to the NFP_BINARY_V1 model file (use either --inputs or --model)."
    layer : Nat; "Layer index for the induction head (1-based, required with --model)."
    head : Nat; "Head index for the induction head (1-based, required with --model)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Optional prompt period override (model only; default: derive from tokens)."
    "prev-shift"; "Use shifted prev/active (prev = q - period + 1) for model inputs."
    direction : String; "Optional logit-diff direction as \"target,negative\" (model only). \
                          When omitted with --model, direction is derived from tokens."
    preset : String; "Split-budget preset: fast | balanced | tight (default: balanced)."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; default: 0)."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    "skip-logit-diff"; "Skip logit-diff lower bound computation."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
]

/-- `nfp induction certify_nonvacuous` subcommand (streamlined). -/
def inductionCertifyNonvacuousSimpleCmd : Cmd := `[Cli|
  certify_nonvacuous VIA runInductionCertifyNonvacuousSimple;
  "Require a strictly positive logit-diff bound from inputs or a model file."
  FLAGS:
    inputs : String; "Path to the induction head input file (use either --inputs or --model)."
    model : String; "Path to the NFP_BINARY_V1 model file (use either --inputs or --model)."
    layer : Nat; "Layer index for the induction head (1-based, required with --model)."
    head : Nat; "Head index for the induction head (1-based, required with --model)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Optional prompt period override (model only; default: derive from tokens)."
    "prev-shift"; "Use shifted prev/active (prev = q - period + 1) for model inputs."
    direction : String; "Optional logit-diff direction as \"target,negative\" (model only). \
                          When omitted with --model, direction is derived from tokens."
    preset : String; "Split-budget preset: fast | balanced | tight (default: balanced)."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; default: 0)."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
]

/-- `nfp induction interval` subcommand (streamlined). -/
def inductionIntervalSimpleCmd : Cmd := `[Cli|
  interval VIA runInductionIntervalSimple;
  "Build head-output interval bounds from inputs or a model file."
  FLAGS:
    inputs : String; "Path to the induction head input file (use either --inputs or --model)."
    model : String; "Path to the NFP_BINARY_V1 model file (use either --inputs or --model)."
    layer : Nat; "Layer index for the induction head (1-based, required with --model)."
    head : Nat; "Head index for the induction head (1-based, required with --model)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Optional prompt period override (model only; default: derive from tokens)."
    direction : String; "Required logit-diff direction as \"target,negative\" (model only)."
    out : String; "Optional path to write the residual-interval certificate."
]

/-- Check induction certificates for induction heads. -/
def runInductionCertify (p : Parsed) : IO UInt32 := do
  let scoresPath := p.flag! "scores" |>.as! String
  let valuesPath? := (p.flag? "values").map (·.as! String)
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertify scoresPath valuesPath? minActive? minLogitDiffStr?
    minMarginStr? maxEpsStr?

/-- `nfp induction certify` subcommand. -/
def inductionCertifyCmd : Cmd := `[Cli|
  certify VIA runInductionCertify;
  "Check induction certificates for induction heads."
  FLAGS:
    scores : String; "Path to the softmax-margin certificate file."
    values : String; "Optional path to a value-range certificate file."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; requires --values). Defaults \
                                to 0 when direction metadata is present."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
]

/-- `nfp induction certify-sound` subcommand. -/
def runInductionCertifySound (p : Parsed) : IO UInt32 := do
  let scoresPath := p.flag! "scores" |>.as! String
  let valuesPath := p.flag! "values" |>.as! String
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertifySound scoresPath valuesPath minActive? minLogitDiffStr?
    minMarginStr? maxEpsStr?

/-- `nfp induction certify_sound` subcommand. -/
def inductionCertifySoundCmd : Cmd := `[Cli|
  certify_sound VIA runInductionCertifySound;
  "Check induction certificates from raw scores/values."
  FLAGS:
    scores : String; "Path to the raw scores/weights file."
    values : String; "Path to the raw value entries file."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0 when \
                                direction metadata is present."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
]

/-- `nfp induction certify_end_to_end` subcommand. -/
def runInductionCertifyEndToEnd (p : Parsed) : IO UInt32 := do
  let scoresPath := p.flag! "scores" |>.as! String
  let valuesPath := p.flag! "values" |>.as! String
  let downstreamPath := p.flag! "downstream" |>.as! String
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertifyEndToEnd scoresPath valuesPath downstreamPath
    minActive? minLogitDiffStr? minMarginStr? maxEpsStr?

/-- `nfp induction certify_end_to_end` subcommand. -/
def inductionCertifyEndToEndCmd : Cmd := `[Cli|
  certify_end_to_end VIA runInductionCertifyEndToEnd;
  "Check end-to-end induction bounds with a downstream error certificate."
  FLAGS:
    scores : String; "Path to the softmax-margin certificate file."
    values : String; "Path to the value-range certificate file."
    downstream : String; "Path to the downstream linear certificate file."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0 when \
                                direction metadata is present."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
]

/-- `nfp induction certify_end_to_end_matrix` subcommand. -/
def runInductionCertifyEndToEndMatrix (p : Parsed) : IO UInt32 := do
  let scoresPath := p.flag! "scores" |>.as! String
  let valuesPath := p.flag! "values" |>.as! String
  let matrixPath := p.flag! "matrix" |>.as! String
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertifyEndToEndMatrix scoresPath valuesPath matrixPath
    minActive? minLogitDiffStr? minMarginStr? maxEpsStr?

/-- `nfp induction certify_end_to_end_matrix` subcommand. -/
def inductionCertifyEndToEndMatrixCmd : Cmd := `[Cli|
  certify_end_to_end_matrix VIA runInductionCertifyEndToEndMatrix;
  "Check end-to-end induction bounds using a downstream matrix payload."
  FLAGS:
    scores : String; "Path to the softmax-margin certificate file."
    values : String; "Path to the value-range certificate file."
    matrix : String; "Path to the downstream matrix payload file."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0 when \
                                direction metadata is present."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
]

/-- `nfp induction certify_end_to_end_model` subcommand. -/
def runInductionCertifyEndToEndModel (p : Parsed) : IO UInt32 := do
  let scoresPath := p.flag! "scores" |>.as! String
  let valuesPath := p.flag! "values" |>.as! String
  let modelPath := p.flag! "model" |>.as! String
  let residualIntervalPath? := (p.flag? "residual-interval").map (·.as! String)
  let layer? := (p.flag? "layer").map (·.as! Nat)
  let head? := (p.flag? "head").map (·.as! Nat)
  let period? := (p.flag? "period").map (·.as! Nat)
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertifyEndToEndModel scoresPath valuesPath modelPath residualIntervalPath?
    layer? head? period? minActive? minLogitDiffStr? minMarginStr? maxEpsStr?

/-- `nfp induction certify_end_to_end_model` subcommand. -/
def inductionCertifyEndToEndModelCmd : Cmd := `[Cli|
  certify_end_to_end_model VIA runInductionCertifyEndToEndModel;
  "Check end-to-end induction bounds using a model file for the downstream matrix."
  FLAGS:
    scores : String; "Path to the softmax-margin certificate file."
    values : String; "Path to the value-range certificate file."
    model : String; "Path to the NFP_BINARY_V1 model file."
    "residual-interval" : String; "Optional path to a residual-interval certificate file \
                                    (defaults to deriving from the model)."
    layer : Nat; "Optional layer index for a head-output interval bound (requires --head)."
    head : Nat; "Optional head index for a head-output interval bound (requires --layer)."
    period : Nat; "Optional prompt period override when reading head inputs."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0 when \
                                direction metadata is present."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
]

/-- `nfp induction certify_head` subcommand. -/
def runInductionCertifyHead (p : Parsed) : IO UInt32 := do
  let inputsPath := p.flag! "inputs" |>.as! String
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let timing? := (p.flag? "timing").map (·.as! Nat)
  let heartbeatMs? := (p.flag? "heartbeat-ms").map (·.as! Nat)
  let splitBudgetQ? := (p.flag? "split-budget-q").map (·.as! Nat)
  let splitBudgetK? := (p.flag? "split-budget-k").map (·.as! Nat)
  let splitBudgetDiffBase? := (p.flag? "split-budget-diff-base").map (·.as! Nat)
  let splitBudgetDiffRefined? := (p.flag? "split-budget-diff-refined").map (·.as! Nat)
  let skipLogitDiff := p.hasFlag "skip-logit-diff"
  IO.runInductionCertifyHead inputsPath minActive? minLogitDiffStr?
    minMarginStr? maxEpsStr? timing? heartbeatMs?
    splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined? skipLogitDiff

/-- `nfp induction certify_head_nonvacuous` subcommand. -/
def runInductionCertifyHeadNonvacuous (p : Parsed) : IO UInt32 := do
  let inputsPath := p.flag! "inputs" |>.as! String
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let timing? := (p.flag? "timing").map (·.as! Nat)
  let heartbeatMs? := (p.flag? "heartbeat-ms").map (·.as! Nat)
  let splitBudgetQ? := (p.flag? "split-budget-q").map (·.as! Nat)
  let splitBudgetK? := (p.flag? "split-budget-k").map (·.as! Nat)
  let splitBudgetDiffBase? := (p.flag? "split-budget-diff-base").map (·.as! Nat)
  let splitBudgetDiffRefined? := (p.flag? "split-budget-diff-refined").map (·.as! Nat)
  IO.runInductionCertifyHeadNonvacuous inputsPath minActive? minLogitDiffStr?
    minMarginStr? maxEpsStr? timing? heartbeatMs?
    splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?

/-- `nfp induction certify_head` subcommand. -/
def inductionCertifyHeadCmd : Cmd := `[Cli|
  certify_head VIA runInductionCertifyHead;
  "Check induction certificates from exact head inputs."
  FLAGS:
    inputs : String; "Path to the induction head input file."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
    "split-budget-q" : Nat; "Split-budget for query dims in sign-splitting bounds (default: 2)."
    "split-budget-k" : Nat; "Split-budget for key dims in sign-splitting bounds (default: 2)."
    "split-budget-diff-base" : Nat; "Split-budget for base diff dims in sign-splitting bounds \
                                      (default: 0)."
    "split-budget-diff-refined" : Nat; "Split-budget for refined diff dims in sign-splitting \
                                         bounds (default: 12)."
    "skip-logit-diff" : Bool; "Skip logit-diff lower bound computation."
]

/-- `nfp induction certify_head_nonvacuous` subcommand. -/
def inductionCertifyHeadNonvacuousCmd : Cmd := `[Cli|
  certify_head_nonvacuous VIA runInductionCertifyHeadNonvacuous;
  "Require a strictly positive logit-diff bound from exact head inputs."
  FLAGS:
    inputs : String; "Path to the induction head input file."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; default: 0)."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
    "split-budget-q" : Nat; "Split-budget for query dims in sign-splitting bounds (default: 2)."
    "split-budget-k" : Nat; "Split-budget for key dims in sign-splitting bounds (default: 2)."
    "split-budget-diff-base" : Nat; "Split-budget for base diff dims in sign-splitting bounds \
                                      (default: 0)."
    "split-budget-diff-refined" : Nat; "Split-budget for refined diff dims in sign-splitting \
                                         bounds (default: 12)."
    "skip-logit-diff" : Bool; "Skip logit-diff lower bound computation."
]

/-- `nfp induction certify_head_model` subcommand. -/
def runInductionCertifyHeadModel (p : Parsed) : IO UInt32 := do
  let modelPath := p.flag! "model" |>.as! String
  let layer := p.flag! "layer" |>.as! Nat
  let head := p.flag! "head" |>.as! Nat
  let period? := (p.flag? "period").map (·.as! Nat)
  let prevShift := p.hasFlag "prev-shift"
  let dirTarget := p.flag! "direction-target" |>.as! Nat
  let dirNegative := p.flag! "direction-negative" |>.as! Nat
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let timing? := (p.flag? "timing").map (·.as! Nat)
  let heartbeatMs? := (p.flag? "heartbeat-ms").map (·.as! Nat)
  let splitBudgetQ? := (p.flag? "split-budget-q").map (·.as! Nat)
  let splitBudgetK? := (p.flag? "split-budget-k").map (·.as! Nat)
  let splitBudgetDiffBase? := (p.flag? "split-budget-diff-base").map (·.as! Nat)
  let splitBudgetDiffRefined? := (p.flag? "split-budget-diff-refined").map (·.as! Nat)
  let skipLogitDiff := p.hasFlag "skip-logit-diff"
  let zeroBased := p.hasFlag "zero-based"
  let layerE := toZeroBased "layer" layer zeroBased
  let headE := toZeroBased "head" head zeroBased
  match layerE, headE with
  | Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok layer', Except.ok head' =>
      IO.runInductionCertifyHeadModel modelPath layer' head' dirTarget dirNegative period?
        prevShift minActive? minLogitDiffStr? minMarginStr? maxEpsStr? timing? heartbeatMs?
        splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined? skipLogitDiff

/-- `nfp induction certify_head_model_nonvacuous` subcommand. -/
def runInductionCertifyHeadModelNonvacuous (p : Parsed) : IO UInt32 := do
  let modelPath := p.flag! "model" |>.as! String
  let layer := p.flag! "layer" |>.as! Nat
  let head := p.flag! "head" |>.as! Nat
  let period? := (p.flag? "period").map (·.as! Nat)
  let prevShift := p.hasFlag "prev-shift"
  let dirTarget := p.flag! "direction-target" |>.as! Nat
  let dirNegative := p.flag! "direction-negative" |>.as! Nat
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let timing? := (p.flag? "timing").map (·.as! Nat)
  let heartbeatMs? := (p.flag? "heartbeat-ms").map (·.as! Nat)
  let splitBudgetQ? := (p.flag? "split-budget-q").map (·.as! Nat)
  let splitBudgetK? := (p.flag? "split-budget-k").map (·.as! Nat)
  let splitBudgetDiffBase? := (p.flag? "split-budget-diff-base").map (·.as! Nat)
  let splitBudgetDiffRefined? := (p.flag? "split-budget-diff-refined").map (·.as! Nat)
  let zeroBased := p.hasFlag "zero-based"
  let layerE := toZeroBased "layer" layer zeroBased
  let headE := toZeroBased "head" head zeroBased
  match layerE, headE with
  | Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok layer', Except.ok head' =>
      IO.runInductionCertifyHeadModelNonvacuous modelPath layer' head' dirTarget dirNegative
        period? prevShift minActive? minLogitDiffStr? minMarginStr? maxEpsStr?
        timing? heartbeatMs?
        splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?

/-- `nfp induction certify_head_model_auto` subcommand. -/
def runInductionCertifyHeadModelAuto (p : Parsed) : IO UInt32 := do
  let modelPath := p.flag! "model" |>.as! String
  let layer := p.flag! "layer" |>.as! Nat
  let head := p.flag! "head" |>.as! Nat
  let period? := (p.flag? "period").map (·.as! Nat)
  let prevShift := p.hasFlag "prev-shift"
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let timing? := (p.flag? "timing").map (·.as! Nat)
  let heartbeatMs? := (p.flag? "heartbeat-ms").map (·.as! Nat)
  let splitBudgetQ? := (p.flag? "split-budget-q").map (·.as! Nat)
  let splitBudgetK? := (p.flag? "split-budget-k").map (·.as! Nat)
  let splitBudgetDiffBase? := (p.flag? "split-budget-diff-base").map (·.as! Nat)
  let splitBudgetDiffRefined? := (p.flag? "split-budget-diff-refined").map (·.as! Nat)
  let skipLogitDiff := p.hasFlag "skip-logit-diff"
  let zeroBased := p.hasFlag "zero-based"
  let layerE := toZeroBased "layer" layer zeroBased
  let headE := toZeroBased "head" head zeroBased
  match layerE, headE with
  | Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok layer', Except.ok head' =>
      IO.runInductionCertifyHeadModelAuto modelPath layer' head' period? prevShift
        minActive? minLogitDiffStr? minMarginStr? maxEpsStr? timing? heartbeatMs?
        splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined? skipLogitDiff

/-- `nfp induction certify_head_model_auto_nonvacuous` subcommand. -/
def runInductionCertifyHeadModelAutoNonvacuous (p : Parsed) : IO UInt32 := do
  let modelPath := p.flag! "model" |>.as! String
  let layer := p.flag! "layer" |>.as! Nat
  let head := p.flag! "head" |>.as! Nat
  let period? := (p.flag? "period").map (·.as! Nat)
  let prevShift := p.hasFlag "prev-shift"
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let timing? := (p.flag? "timing").map (·.as! Nat)
  let heartbeatMs? := (p.flag? "heartbeat-ms").map (·.as! Nat)
  let splitBudgetQ? := (p.flag? "split-budget-q").map (·.as! Nat)
  let splitBudgetK? := (p.flag? "split-budget-k").map (·.as! Nat)
  let splitBudgetDiffBase? := (p.flag? "split-budget-diff-base").map (·.as! Nat)
  let splitBudgetDiffRefined? := (p.flag? "split-budget-diff-refined").map (·.as! Nat)
  let zeroBased := p.hasFlag "zero-based"
  let layerE := toZeroBased "layer" layer zeroBased
  let headE := toZeroBased "head" head zeroBased
  match layerE, headE with
  | Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok layer', Except.ok head' =>
      IO.runInductionCertifyHeadModelAutoNonvacuous modelPath layer' head' period? prevShift
        minActive? minLogitDiffStr? minMarginStr? maxEpsStr? timing? heartbeatMs?
        splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?

/-! `nfp induction certify_circuit_model` subcommand. -/
/-- CLI entrypoint for `nfp induction certify_circuit_model`. -/
def runInductionCertifyCircuitModel (p : Parsed) : IO UInt32 := do
  let modelPath := p.flag! "model" |>.as! String
  let prevLayer := p.flag! "prev-layer" |>.as! Nat
  let prevHead := p.flag! "prev-head" |>.as! Nat
  let indLayer := p.flag! "ind-layer" |>.as! Nat
  let indHead := p.flag! "ind-head" |>.as! Nat
  let period? := (p.flag? "period").map (·.as! Nat)
  let dirTarget := p.flag! "direction-target" |>.as! Nat
  let dirNegative := p.flag! "direction-negative" |>.as! Nat
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let timing? := (p.flag? "timing").map (·.as! Nat)
  let heartbeatMs? := (p.flag? "heartbeat-ms").map (·.as! Nat)
  let splitBudgetQ? := (p.flag? "split-budget-q").map (·.as! Nat)
  let splitBudgetK? := (p.flag? "split-budget-k").map (·.as! Nat)
  let splitBudgetDiffBase? := (p.flag? "split-budget-diff-base").map (·.as! Nat)
  let splitBudgetDiffRefined? := (p.flag? "split-budget-diff-refined").map (·.as! Nat)
  let skipLogitDiff := p.hasFlag "skip-logit-diff"
  let zeroBased := p.hasFlag "zero-based"
  match period? with
  | none =>
      IO.eprintln "error: --period is required for circuit certification"
      return 2
  | some period =>
      if period = 0 then
        IO.eprintln "error: --period must be positive for circuit certification"
        return 2
      let prevLayerE := toZeroBased "prev-layer" prevLayer zeroBased
      let prevHeadE := toZeroBased "prev-head" prevHead zeroBased
      let indLayerE := toZeroBased "ind-layer" indLayer zeroBased
      let indHeadE := toZeroBased "ind-head" indHead zeroBased
      match prevLayerE, prevHeadE, indLayerE, indHeadE with
      | Except.error msg, _, _, _ =>
          IO.eprintln s!"error: {msg}"
          return 2
      | _, Except.error msg, _, _ =>
          IO.eprintln s!"error: {msg}"
          return 2
      | _, _, Except.error msg, _ =>
          IO.eprintln s!"error: {msg}"
          return 2
      | _, _, _, Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 2
      | Except.ok prevLayer', Except.ok prevHead', Except.ok indLayer', Except.ok indHead' =>
          IO.runInductionCertifyCircuitModel modelPath prevLayer' prevHead' indLayer' indHead'
            dirTarget dirNegative period
            minActive? minLogitDiffStr? minMarginStr? maxEpsStr? timing? heartbeatMs?
            splitBudgetQ? splitBudgetK? splitBudgetDiffBase? splitBudgetDiffRefined?
            skipLogitDiff

/-- `nfp induction certify_head_model` subcommand. -/
def inductionCertifyHeadModelCmd : Cmd := `[Cli|
  certify_head_model VIA runInductionCertifyHeadModel;
  "Check induction certificates by reading a model binary directly."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    layer : Nat; "Layer index for the induction head (1-based)."
    head : Nat; "Head index for the induction head (1-based)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Optional prompt period override (default: derive from tokens)."
    "prev-shift"; "Use shifted prev/active (prev = q - period + 1)."
    "direction-target" : Nat; "Target token id for logit-diff direction."
    "direction-negative" : Nat; "Negative token id for logit-diff direction."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
    "split-budget-q" : Nat; "Split-budget for query dims in sign-splitting bounds (default: 2)."
    "split-budget-k" : Nat; "Split-budget for key dims in sign-splitting bounds (default: 2)."
    "split-budget-diff-base" : Nat; "Split-budget for base diff dims in sign-splitting bounds \
                                      (default: 0)."
    "split-budget-diff-refined" : Nat; "Split-budget for refined diff dims in sign-splitting \
                                         bounds (default: 12)."
    "skip-logit-diff" : Bool; "Skip logit-diff lower bound computation."
]

/-- `nfp induction certify_head_model_nonvacuous` subcommand. -/
def inductionCertifyHeadModelNonvacuousCmd : Cmd := `[Cli|
  certify_head_model_nonvacuous VIA runInductionCertifyHeadModelNonvacuous;
  "Require a strictly positive logit-diff bound from a model binary."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    layer : Nat; "Layer index for the induction head (1-based)."
    head : Nat; "Head index for the induction head (1-based)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Optional prompt period override (default: derive from tokens)."
    "prev-shift"; "Use shifted prev/active (prev = q - period + 1)."
    "direction-target" : Nat; "Target token id for logit-diff direction."
    "direction-negative" : Nat; "Negative token id for logit-diff direction."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; default: 0)."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
    "split-budget-q" : Nat; "Split-budget for query dims in sign-splitting bounds (default: 2)."
    "split-budget-k" : Nat; "Split-budget for key dims in sign-splitting bounds (default: 2)."
    "split-budget-diff-base" : Nat; "Split-budget for base diff dims in sign-splitting bounds \
                                      (default: 0)."
    "split-budget-diff-refined" : Nat; "Split-budget for refined diff dims in sign-splitting \
                                         bounds (default: 12)."
]

/-- `nfp induction certify_head_model_auto` subcommand. -/
def inductionCertifyHeadModelAutoCmd : Cmd := `[Cli|
  certify_head_model_auto VIA runInductionCertifyHeadModelAuto;
  "Check induction certificates by reading a model binary and deriving the direction \
  from the prompt tokens."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    layer : Nat; "Layer index for the induction head (1-based)."
    head : Nat; "Head index for the induction head (1-based)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Optional prompt period override (default: derive from tokens)."
    "prev-shift"; "Use shifted prev/active (prev = q - period + 1)."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
    "split-budget-q" : Nat; "Split-budget for query dims in sign-splitting bounds (default: 2)."
    "split-budget-k" : Nat; "Split-budget for key dims in sign-splitting bounds (default: 2)."
    "split-budget-diff-base" : Nat; "Split-budget for base diff dims in sign-splitting bounds \
                                      (default: 0)."
    "split-budget-diff-refined" : Nat; "Split-budget for refined diff dims in sign-splitting \
                                         bounds (default: 12)."
]

/-- `nfp induction certify_head_model_auto_nonvacuous` subcommand. -/
def inductionCertifyHeadModelAutoNonvacuousCmd : Cmd := `[Cli|
  certify_head_model_auto_nonvacuous VIA runInductionCertifyHeadModelAutoNonvacuous;
  "Require a strictly positive logit-diff bound from a model binary, with the direction \
  derived from the prompt tokens."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    layer : Nat; "Layer index for the induction head (1-based)."
    head : Nat; "Head index for the induction head (1-based)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Optional prompt period override (default: derive from tokens)."
    "prev-shift"; "Use shifted prev/active (prev = q - period + 1)."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; default: 0)."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
    "split-budget-q" : Nat; "Split-budget for query dims in sign-splitting bounds (default: 2)."
    "split-budget-k" : Nat; "Split-budget for key dims in sign-splitting bounds (default: 2)."
    "split-budget-diff-base" : Nat; "Split-budget for base diff dims in sign-splitting bounds \
                                      (default: 0)."
    "split-budget-diff-refined" : Nat; "Split-budget for refined diff dims in sign-splitting \
                                         bounds (default: 12)."
]

/-- `nfp induction certify_circuit_model` subcommand. -/
def inductionCertifyCircuitModelCmd : Cmd := `[Cli|
  certify_circuit_model VIA runInductionCertifyCircuitModel;
  "Check a two-head induction circuit by reading a model binary directly \
  (induction head uses shifted prev)."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    "prev-layer" : Nat; "Layer index for the previous-token head (1-based)."
    "prev-head" : Nat; "Head index for the previous-token head (1-based)."
    "ind-layer" : Nat; "Layer index for the induction head (1-based)."
    "ind-head" : Nat; "Head index for the induction head (1-based)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Prompt period override (required)."
    "direction-target" : Nat; "Target token id for logit-diff direction."
    "direction-negative" : Nat; "Negative token id for logit-diff direction."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
    "skip-logit-diff"; "Skip logit-diff lower bound computation."
    timing : Nat; "Emit timing output to stdout (0=off, 1=on)."
    "heartbeat-ms" : Nat; "Emit progress heartbeat every N ms (0 disables)."
    "split-budget-q" : Nat; "Split-budget for query dims in sign-splitting bounds (default: 2)."
    "split-budget-k" : Nat; "Split-budget for key dims in sign-splitting bounds (default: 2)."
    "split-budget-diff-base" : Nat; "Split-budget for base diff dims in sign-splitting bounds \
                                      (default: 0)."
    "split-budget-diff-refined" : Nat; "Split-budget for refined diff dims in sign-splitting \
                                         bounds (default: 12)."
]

/-- `nfp induction head_interval` subcommand. -/
def runInductionHeadInterval (p : Parsed) : IO UInt32 := do
  let inputsPath := p.flag! "inputs" |>.as! String
  let outPath? := (p.flag? "out").map (·.as! String)
  IO.runInductionHeadInterval inputsPath outPath?

/-- `nfp induction head_interval` subcommand. -/
def inductionHeadIntervalCmd : Cmd := `[Cli|
  head_interval VIA runInductionHeadInterval;
  "Build head-output interval bounds from exact head inputs."
  FLAGS:
    inputs : String; "Path to the induction head input file."
    out : String; "Optional path to write the residual-interval certificate."
]

/-- `nfp induction head_interval_model` subcommand. -/
def runInductionHeadIntervalModel (p : Parsed) : IO UInt32 := do
  let modelPath := p.flag! "model" |>.as! String
  let layer := p.flag! "layer" |>.as! Nat
  let head := p.flag! "head" |>.as! Nat
  let period? := (p.flag? "period").map (·.as! Nat)
  let prevShift := p.hasFlag "prev-shift"
  let dirTarget := p.flag! "direction-target" |>.as! Nat
  let dirNegative := p.flag! "direction-negative" |>.as! Nat
  let outPath? := (p.flag? "out").map (·.as! String)
  let zeroBased := p.hasFlag "zero-based"
  let layerE := toZeroBased "layer" layer zeroBased
  let headE := toZeroBased "head" head zeroBased
  match layerE, headE with
  | Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok layer', Except.ok head' =>
      IO.runInductionHeadIntervalModel modelPath layer' head' dirTarget dirNegative period?
        prevShift outPath?

/-- `nfp induction head_interval_model` subcommand. -/
def inductionHeadIntervalModelCmd : Cmd := `[Cli|
  head_interval_model VIA runInductionHeadIntervalModel;
  "Build head-output interval bounds by reading a model binary directly."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    layer : Nat; "Layer index for the induction head (1-based)."
    head : Nat; "Head index for the induction head (1-based)."
    "zero-based"; "Interpret --layer/--head as zero-based indices (legacy)."
    period : Nat; "Optional prompt period override (default: derive from tokens)."
    "prev-shift"; "Use shifted prev/active (prev = q - period + 1)."
    "direction-target" : Nat; "Target token id for logit-diff direction."
    "direction-negative" : Nat; "Negative token id for logit-diff direction."
    out : String; "Optional path to write the residual-interval certificate."
]

/-- Advanced induction-head subcommands (full flag surface). -/
def inductionAdvancedCmd : Cmd := `[Cli|
  advanced NOOP;
  "Advanced induction-head utilities (full flag set)."
  SUBCOMMANDS:
    inductionCertifyCmd;
    inductionCertifySoundCmd;
    inductionCertifyEndToEndCmd;
    inductionCertifyEndToEndMatrixCmd;
    inductionCertifyEndToEndModelCmd;
    inductionCertifyHeadCmd;
    inductionCertifyHeadNonvacuousCmd;
    inductionCertifyHeadModelCmd;
    inductionCertifyHeadModelNonvacuousCmd;
    inductionCertifyHeadModelAutoCmd;
    inductionCertifyHeadModelAutoNonvacuousCmd;
    inductionCertifyCircuitModelCmd;
    inductionHeadIntervalCmd;
    inductionHeadIntervalModelCmd
]

/-- Induction-head subcommands. -/
def inductionCmd : Cmd := `[Cli|
  induction NOOP;
  "Induction-head utilities (streamlined). Use `nfp induction advanced --help` for full options."
  SUBCOMMANDS:
    inductionCertifySimpleCmd;
    inductionCertifyNonvacuousSimpleCmd;
    inductionIntervalSimpleCmd;
    inductionAdvancedCmd
]

/-- The root CLI command. -/
def nfpCmd : Cmd := `[Cli|
  nfp NOOP;
  "NFP: Neural Formal Pathways (rewrite in progress)."
  SUBCOMMANDS:
    versionCmd;
    inductionCmd
]

/-- Main entry point for the CLI. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--version" then
    IO.println s!"nfp version {versionString}"
    return 0
  nfpCmd.validate args

end Nfp

end
