-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/-- LayerNorm inputs for anchoring post-LN residuals. -/
structure ModelLnSlice (seq : Nat) where
  /-- Hidden dimension. -/
  dModel : Nat
  /-- LayerNorm epsilon. -/
  lnEps : Rat
  /-- Nonnegative slack for LayerNorm bounds. -/
  lnSlack : Rat
  /-- Optional LayerNorm sqrt bound scale (positive when present). -/
  lnScale? : Option Nat
  /-- Optional fast-path toggle for fixed-denominator checks. -/
  lnFast? : Option Bool
  /-- Optional decimal precision for fixed-denominator checks. -/
  lnDecimals? : Option Nat
  /-- LayerNorm gamma. -/
  lnGamma : Fin dModel → Rat
  /-- LayerNorm beta. -/
  lnBeta : Fin dModel → Rat
  /-- Pre-LN embeddings/residual stream. -/
  embed : Fin seq → Fin dModel → Rat

end InductionHeadCert

end IO

end Nfp
