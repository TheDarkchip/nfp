-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/-- Direction inputs anchored to unembedding rows. -/
structure ModelDirectionSlice where
  /-- Hidden dimension. -/
  dModel : Nat
  /-- Target token index (metadata). -/
  target : Nat
  /-- Negative token index (metadata). -/
  negative : Nat
  /-- Unembedding row for the target token. -/
  unembedTarget : Fin dModel → Rat
  /-- Unembedding row for the negative token. -/
  unembedNegative : Fin dModel → Rat
  /-- Direction vector in model space. -/
  direction : Fin dModel → Rat

end InductionHeadCert

end IO

end Nfp
