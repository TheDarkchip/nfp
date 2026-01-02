-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Cli

/-- CLI entry point. -/
def main (args : List String) : IO UInt32 :=
  Nfp.main args
