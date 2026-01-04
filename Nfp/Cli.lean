-- SPDX-License-Identifier: AGPL-3.0-or-later

import Cli
import Nfp.IO

/-!
Minimal CLI surface for the NFP rewrite.
-/

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
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertifyEndToEndModel scoresPath valuesPath modelPath residualIntervalPath?
    minActive? minLogitDiffStr? minMarginStr? maxEpsStr?

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
  IO.runInductionCertifyHead inputsPath minActive? minLogitDiffStr?
    minMarginStr? maxEpsStr?

/-- `nfp induction certify_head_nonvacuous` subcommand. -/
def runInductionCertifyHeadNonvacuous (p : Parsed) : IO UInt32 := do
  let inputsPath := p.flag! "inputs" |>.as! String
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertifyHeadNonvacuous inputsPath minActive? minLogitDiffStr?
    minMarginStr? maxEpsStr?

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
]

/-- `nfp induction certify_head_model` subcommand. -/
def runInductionCertifyHeadModel (p : Parsed) : IO UInt32 := do
  let modelPath := p.flag! "model" |>.as! String
  let layer := p.flag! "layer" |>.as! Nat
  let head := p.flag! "head" |>.as! Nat
  let period? := (p.flag? "period").map (·.as! Nat)
  let dirTarget := p.flag! "direction-target" |>.as! Nat
  let dirNegative := p.flag! "direction-negative" |>.as! Nat
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertifyHeadModel modelPath layer head dirTarget dirNegative period?
    minActive? minLogitDiffStr? minMarginStr? maxEpsStr?

/-- `nfp induction certify_head_model_nonvacuous` subcommand. -/
def runInductionCertifyHeadModelNonvacuous (p : Parsed) : IO UInt32 := do
  let modelPath := p.flag! "model" |>.as! String
  let layer := p.flag! "layer" |>.as! Nat
  let head := p.flag! "head" |>.as! Nat
  let period? := (p.flag? "period").map (·.as! Nat)
  let dirTarget := p.flag! "direction-target" |>.as! Nat
  let dirNegative := p.flag! "direction-negative" |>.as! Nat
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  IO.runInductionCertifyHeadModelNonvacuous modelPath layer head dirTarget dirNegative period?
    minActive? minLogitDiffStr? minMarginStr? maxEpsStr?

/-- `nfp induction certify_head_model` subcommand. -/
def inductionCertifyHeadModelCmd : Cmd := `[Cli|
  certify_head_model VIA runInductionCertifyHeadModel;
  "Check induction certificates by reading a model binary directly."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    layer : Nat; "Layer index for the induction head."
    head : Nat; "Head index for the induction head."
    period : Nat; "Optional prompt period override (default: derive from tokens)."
    "direction-target" : Nat; "Target token id for logit-diff direction."
    "direction-negative" : Nat; "Negative token id for logit-diff direction."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal). Defaults to 0."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
]

/-- `nfp induction certify_head_model_nonvacuous` subcommand. -/
def inductionCertifyHeadModelNonvacuousCmd : Cmd := `[Cli|
  certify_head_model_nonvacuous VIA runInductionCertifyHeadModelNonvacuous;
  "Require a strictly positive logit-diff bound from a model binary."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    layer : Nat; "Layer index for the induction head."
    head : Nat; "Head index for the induction head."
    period : Nat; "Optional prompt period override (default: derive from tokens)."
    "direction-target" : Nat; "Target token id for logit-diff direction."
    "direction-negative" : Nat; "Negative token id for logit-diff direction."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; default: 0)."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
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
  let dirTarget := p.flag! "direction-target" |>.as! Nat
  let dirNegative := p.flag! "direction-negative" |>.as! Nat
  let outPath? := (p.flag? "out").map (·.as! String)
  IO.runInductionHeadIntervalModel modelPath layer head dirTarget dirNegative period? outPath?

/-- `nfp induction head_interval_model` subcommand. -/
def inductionHeadIntervalModelCmd : Cmd := `[Cli|
  head_interval_model VIA runInductionHeadIntervalModel;
  "Build head-output interval bounds by reading a model binary directly."
  FLAGS:
    model : String; "Path to the NFP_BINARY_V1 model file."
    layer : Nat; "Layer index for the induction head."
    head : Nat; "Head index for the induction head."
    period : Nat; "Optional prompt period override (default: derive from tokens)."
    "direction-target" : Nat; "Target token id for logit-diff direction."
    "direction-negative" : Nat; "Negative token id for logit-diff direction."
    out : String; "Optional path to write the residual-interval certificate."
]

/-- Induction-head subcommands. -/
def inductionCmd : Cmd := `[Cli|
  induction NOOP;
  "Induction-head utilities."
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
    inductionHeadIntervalCmd;
    inductionHeadIntervalModelCmd
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
