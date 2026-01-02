-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Matrix.Basic

/-!
Typed tensor indices and tensor aliases.
-/

namespace Nfp

namespace Circuit

namespace Tensor

universe u v

/-- Index type for a length-`n` vector. -/
abbrev VecIndex (n : Nat) : Type := Fin n

/-- Index type for an `m × n` matrix. -/
abbrev MatIndex (m n : Nat) : Type := Fin m × Fin n

/-- Index type for a 3D tensor. -/
abbrev Tensor3Index (a b c : Nat) : Type := Fin a × Fin b × Fin c

/-- Index type for a 4D tensor. -/
abbrev Tensor4Index (a b c d : Nat) : Type := Fin a × Fin b × Fin c × Fin d

/-- A length-`n` vector of values. -/
abbrev Vec (n : Nat) (Val : Type v) : Type v := VecIndex n → Val

/-- An `m × n` matrix of values. -/
abbrev Mat (m n : Nat) (Val : Type v) : Type v := Matrix (VecIndex m) (VecIndex n) Val

/-- A 3D tensor of values. -/
abbrev Tensor3 (a b c : Nat) (Val : Type v) : Type v := Tensor3Index a b c → Val

/-- A 4D tensor of values. -/
abbrev Tensor4 (a b c d : Nat) (Val : Type v) : Type v := Tensor4Index a b c d → Val

end Tensor

end Circuit

end Nfp
