-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Matrix.Mul

/-!
Linear and affine gate combinators built from `Matrix.mulVec`.
-/

public section

namespace Nfp

namespace Circuit

namespace Gates

universe u v

variable {Row : Type u} {Col : Type u} {Val : Type v}

/-- Linear map on vectors defined by a matrix. -/
def linear [Fintype Row] [Fintype Col] [NonUnitalNonAssocSemiring Val]
    (W : Matrix Row Col Val) (x : Col → Val) : Row → Val :=
  Matrix.mulVec W x

/-- Affine map on vectors defined by a matrix and bias. -/
def affine [Fintype Row] [Fintype Col] [NonUnitalNonAssocSemiring Val]
    (W : Matrix Row Col Val) (b : Row → Val) (x : Col → Val) : Row → Val :=
  linear W x + b

end Gates

end Circuit

end Nfp
