-- SPDX-License-Identifier: AGPL-3.0-or-later

module

import Nfp.Cli

public section

/-- CLI entry point. -/
def main (args : List String) : IO UInt32 :=
  Nfp.main args

end
