-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Basic
import Nfp.Circuit.Combinators
import Nfp.Circuit.Interface
import Nfp.Circuit.Semantics
import Nfp.Circuit.WellFormed
import Nfp.Circuit.Cert
import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange
import Nfp.Circuit.Typed
import Nfp.Circuit.Compose
import Nfp.Circuit.Gates
import Nfp.Circuit.Tensor
import Nfp.Circuit.Layers

/-!
Circuit definitions, semantics, and equivalence checking.
-/
