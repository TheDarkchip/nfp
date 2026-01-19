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

private def runInductionCertifySimple (p : Parsed) : IO UInt32 := do
  let certPath? := (p.flag? "cert").map (·.as! String)
  let minActive? := (p.flag? "min-active").map (·.as! Nat)
  let minLogitDiffStr? := (p.flag? "min-logit-diff").map (·.as! String)
  let minMarginStr? := (p.flag? "min-margin").map (·.as! String)
  let maxEpsStr? := (p.flag? "max-eps").map (·.as! String)
  let tokensPath? := (p.flag? "tokens").map (·.as! String)
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  match certPath? with
  | none => fail "provide --cert"
  | some certPath =>
      IO.runInductionHeadCertCheck certPath minActive? minLogitDiffStr?
        minMarginStr? maxEpsStr? tokensPath?

/-- `nfp induction certify` subcommand (streamlined). -/
def inductionCertifySimpleCmd : Cmd := `[Cli|
  certify VIA runInductionCertifySimple;
  "Check induction head certificates from an explicit cert."
  FLAGS:
    cert : String; "Path to the induction head certificate file."
    tokens : String; "Optional path to a token list to verify prev/active."
    "min-active" : Nat; "Optional minimum number of active queries required \
                          (default: max 1 (seq/8))."
    "min-logit-diff" : String; "Optional minimum logit-diff lower bound \
                                (rational literal; default: 0)."
    "min-margin" : String; "Optional minimum score margin (rational literal; default: 0)."
    "max-eps" : String; "Optional maximum eps tolerance (rational literal; default: 1/2)."
]

private def runInductionCertifyBatch (p : Parsed) : IO UInt32 := do
  let batchPath? := (p.flag? "batch").map (·.as! String)
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  match batchPath? with
  | none => fail "provide --batch"
  | some batchPath =>
      IO.runInductionHeadBatchCheck batchPath

/-- `nfp induction certifyBatch` subcommand (streamlined). -/
def inductionCertifyBatchCmd : Cmd := `[Cli|
  certifyBatch VIA runInductionCertifyBatch;
  "Check a batch of induction head certificates from a batch file."
  FLAGS:
    batch : String; "Path to the batch file listing certs and tokens."
]

private def runStripeCertify (p : Parsed) : IO UInt32 := do
  let certPath? := (p.flag? "cert").map (·.as! String)
  let minMeanStr? := (p.flag? "min-stripe-mean").map (·.as! String)
  let minTop1Str? := (p.flag? "min-stripe-top1").map (·.as! String)
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  match certPath? with
  | none => fail "provide --cert"
  | some certPath =>
      IO.runStripeCertCheck certPath minMeanStr? minTop1Str?

/-- `nfp induction stripeCertify` subcommand (streamlined). -/
def inductionStripeCertifyCmd : Cmd := `[Cli|
  stripeCertify VIA runStripeCertify;
  "Check stripe-attention certificates from an explicit cert."
  FLAGS:
    cert : String; "Path to the stripe certificate file."
    "min-stripe-mean" : String; "Optional minimum stripe-mean (rational literal)."
    "min-stripe-top1" : String; "Optional minimum stripe-top1 (rational literal)."
]

private def runStripeCertifyBatch (p : Parsed) : IO UInt32 := do
  let batchPath? := (p.flag? "batch").map (·.as! String)
  let fail (msg : String) : IO UInt32 := do
    IO.eprintln s!"error: {msg}"
    return 2
  match batchPath? with
  | none => fail "provide --batch"
  | some batchPath =>
      IO.runStripeBatchCheck batchPath

/-- `nfp induction stripeBatch` subcommand (streamlined). -/
def inductionStripeBatchCmd : Cmd := `[Cli|
  stripeBatch VIA runStripeCertifyBatch;
  "Check a batch of stripe-attention certificates from a batch file."
  FLAGS:
    batch : String; "Path to the stripe batch file."
]

/-- Induction-head subcommands. -/
def inductionCmd : Cmd := `[Cli|
  induction NOOP;
  "Induction-head utilities (streamlined)."
  SUBCOMMANDS:
    inductionCertifySimpleCmd;
    inductionCertifyBatchCmd;
    inductionStripeCertifyCmd;
    inductionStripeBatchCmd
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
