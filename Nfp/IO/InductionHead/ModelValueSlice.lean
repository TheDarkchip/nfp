-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/-- Value-path inputs for anchoring per-key values. -/
structure ModelValueSlice where
  /-- Hidden dimension. -/
  dModel : Nat
  /-- Head dimension. -/
  headDim : Nat
  /-- Layer index (metadata). -/
  layer : Nat
  /-- Head index (metadata). -/
  head : Nat
  /-- Value projection slice (d_model × head_dim). -/
  wv : Fin dModel → Fin headDim → Rat
  /-- Value projection bias (head_dim). -/
  bv : Fin headDim → Rat
  /-- Output projection slice (d_model × head_dim). -/
  wo : Fin dModel → Fin headDim → Rat
  /-- Attention output bias (d_model). -/
  attnBias : Fin dModel → Rat

end InductionHeadCert

end IO

end Nfp
