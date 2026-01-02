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

/-- Check a softmax-margin certificate for induction heads. -/
def runInductionCertify (p : Parsed) : IO UInt32 := do
  let scoresPath := p.flag! "scores" |>.as! String
  IO.runInductionCertify scoresPath

/-- `nfp induction certify` subcommand. -/
def inductionCertifyCmd : Cmd := `[Cli|
  certify VIA runInductionCertify;
  "Check a softmax-margin certificate for induction heads."
  FLAGS:
    scores : String; "Path to the softmax-margin certificate file."
]

/-- Induction-head subcommands. -/
def inductionCmd : Cmd := `[Cli|
  induction NOOP;
  "Induction-head utilities."
  SUBCOMMANDS:
    inductionCertifyCmd
]

/-- The root CLI command. -/
def nfpCmd : Cmd := `[Cli|
  nfp NOOP;
  "NFP: Neural Formal Pathways (rewrite in progress)."
  SUBCOMMANDS:
    versionCmd
    inductionCmd
]

/-- Main entry point for the CLI. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--version" then
    IO.println s!"nfp version {versionString}"
    return 0
  nfpCmd.validate args

end Nfp
