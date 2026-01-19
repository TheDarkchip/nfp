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

private def runInductionVerifySimple (p : Parsed) : IO UInt32 := do
  let certPath? := (p.flag? "cert").map (·.as! String)
  let batchPath? := (p.flag? "batch").map (·.as! String)
  let stripeCertPath? := (p.flag? "stripe-cert").map (·.as! String)
  let stripeBatchPath? := (p.flag? "stripe-batch").map (·.as! String)
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let tokensPath? := (p.flag? "tokens").map (·.as! String)
  let minStripeMeanStr? := (p.flag? "min-stripe-mean").map (·.as! String)
  let minStripeTop1Str? := (p.flag? "min-stripe-top1").map (·.as! String)
  let minInductionMeanStr? := (p.flag? "min-induction-mean").map (·.as! String)
  let minInductionTop1Str? := (p.flag? "min-induction-top1").map (·.as! String)
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  let selections :=
    [certPath?.isSome, batchPath?.isSome, stripeCertPath?.isSome, stripeBatchPath?.isSome]
  if selections.count true != 1 then
    return (← fail "provide exactly one of --cert, --batch, --stripe-cert, or --stripe-batch")
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  match certPath?, batchPath?, stripeCertPath?, stripeBatchPath? with
  | some certPath, none, none, none =>
      IO.runInductionHeadCertCheck certPath minActive? minLogitDiffStr?
        minMarginStr? maxEpsStr? tokensPath? minStripeMeanStr? minStripeTop1Str?
        minInductionMeanStr? minInductionTop1Str?
  | none, some batchPath, none, none =>
      IO.runInductionHeadBatchCheck batchPath
  | none, none, some stripeCertPath, none =>
      IO.runStripeCertCheck stripeCertPath minStripeMeanStr? minStripeTop1Str?
  | none, none, none, some stripeBatchPath =>
      IO.runStripeBatchCheck stripeBatchPath
  | _, _, _, _ =>
      fail "provide exactly one of --cert, --batch, --stripe-cert, or --stripe-batch"

/-- `nfp induction verify` subcommand (streamlined). -/
def inductionVerifySimpleCmd : Cmd := `[Cli|
  verify VIA runInductionVerifySimple;
  "Verify induction or stripe certificates (choose exactly one input flag)."
  FLAGS:
    cert : String; "Path to the induction head certificate file."
    batch : String; "Path to the induction batch file."
    "stripe-cert" : String; "Path to the stripe certificate file."
    "stripe-batch" : String; "Path to the stripe batch file."
    tokens : String; "Optional path to a token list to verify prev/active."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; default: 0)."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0). \
                             Applies to onehot-approx only."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2). \
                          Applies to onehot-approx only."
    "min-stripe-mean" : String; "Optional minimum stripe-mean (rational literal). \
                                 Applies to induction-aligned or stripe certs."
    "min-stripe-top1" : String; "Optional minimum stripe-top1 (rational literal). \
                                 Applies to induction-aligned or stripe certs."
    "min-induction-mean" : String; "Optional minimum induction-mean (rational literal). \
                                    Applies to induction-aligned only."
    "min-induction-top1" : String; "Optional minimum induction-top1 (rational literal). \
                                    Applies to induction-aligned only."
]


/-- Induction-head subcommands. -/
def inductionCmd : Cmd := `[Cli|
  induction NOOP;
  "Induction-head utilities (streamlined)."
  SUBCOMMANDS:
    inductionVerifySimpleCmd
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
