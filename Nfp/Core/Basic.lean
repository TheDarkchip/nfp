-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.Order.Group.Unbundled.Abs
public import Mathlib.Data.NNReal.Defs
public import Mathlib.Data.NNReal.Basic
public import Mathlib.Data.Rat.Cast.Lemmas
public import Mathlib.Data.Rat.Cast.Order
public import Mathlib.Data.Real.Basic

/-!
Basic shared definitions for the NFP rewrite.
-/

@[expose] public section

namespace Nfp

/-- Nonnegative mass used for probabilities and weights. -/
abbrev Mass := NNReal

/-- Default precision placeholder retained for API compatibility. -/
def defaultRatPrec : Int := 48

/-- Round a rational down (identity in the exact-rational refactor). -/
def ratRoundDown (q : Rat) (_prec : Int := defaultRatPrec) : Rat :=
  q

/-- Round a rational up (identity in the exact-rational refactor). -/
def ratRoundUp (q : Rat) (_prec : Int := defaultRatPrec) : Rat :=
  q

/-- Real cast of a rational value. -/
def ratToReal (x : Rat) : Real :=
  (x : Real)

@[simp] theorem ratToReal_zero : ratToReal 0 = 0 := by
  simp [ratToReal]

@[simp] theorem ratToReal_one : ratToReal 1 = 1 := by
  simp [ratToReal]

theorem ratRoundDown_le (q : Rat) (_prec : Int := defaultRatPrec) :
    (ratRoundDown q _prec : Rat) ≤ q := by
  simp [ratRoundDown]

theorem ratRoundUp_ge (q : Rat) (_prec : Int := defaultRatPrec) :
    q ≤ (ratRoundUp q _prec : Rat) := by
  simp [ratRoundUp]

theorem ratRoundDown_le_real (q : Rat) (_prec : Int := defaultRatPrec) :
    (ratRoundDown q _prec : Real) ≤ (q : Real) := by
  simp [ratRoundDown]

theorem real_le_ratRoundUp (q : Rat) (_prec : Int := defaultRatPrec) :
    (q : Real) ≤ (ratRoundUp q _prec : Real) := by
  simp [ratRoundUp]

theorem ratRoundDown_nonneg {q : Rat} (hq : 0 ≤ q) (_prec : Int := defaultRatPrec) :
    0 ≤ ratRoundDown q _prec := by
  simpa [ratRoundDown] using hq

theorem ratRoundUp_nonneg {q : Rat} (hq : 0 ≤ q) (_prec : Int := defaultRatPrec) :
    0 ≤ ratRoundUp q _prec := by
  simpa [ratRoundUp] using hq

theorem ratRoundUp_pos {q : Rat} (hq : 0 < q) (_prec : Int := defaultRatPrec) :
    0 < ratRoundUp q _prec := by
  simpa [ratRoundUp] using hq

/-- Interpret `n * 2^{-prec}` as a rational (matching power-of-two scaling). -/
def ratOfIntWithPrec (n : Int) (prec : Int) : Rat :=
  let pow2 (k : Nat) : Rat := Rat.ofInt (Int.ofNat (Nat.pow 2 k))
  if _h : 0 ≤ prec then
    Rat.ofInt n / pow2 (Int.toNat prec)
  else
    Rat.ofInt n * pow2 (Int.toNat (-prec))

/-- Rational division with downward rounding (exact for rationals). -/
def ratDivDown (x y : Rat) (_prec : Int := defaultRatPrec) : Rat :=
  if y = 0 then
    0
  else
    x / y

/-- Rational division with upward rounding (exact for rationals). -/
def ratDivUp (x y : Rat) (_prec : Int := defaultRatPrec) : Rat :=
  if y = 0 then
    0
  else
    x / y

theorem ratDivUp_ge (x y : Rat) (hy : y ≠ 0) :
    (x / y : Rat) ≤ (ratDivUp x y : Rat) := by
  simp [ratDivUp, hy]

theorem ratDivUp_ge_real (x y : Rat) (hy : y ≠ 0) :
    (x : Real) / (y : Real) ≤ ratToReal (ratDivUp x y) := by
  simp [ratDivUp, ratToReal, hy]

@[simp] theorem ratToReal_add (x y : Rat) :
    ratToReal (x + y) = ratToReal x + ratToReal y := by
  simp [ratToReal]

@[simp] theorem ratToReal_sub (x y : Rat) :
    ratToReal (x - y) = ratToReal x - ratToReal y := by
  simp [ratToReal]

@[simp] theorem ratToReal_mul (x y : Rat) :
    ratToReal (x * y) = ratToReal x * ratToReal y := by
  simp [ratToReal]

@[simp] theorem ratToReal_neg (x : Rat) :
    ratToReal (-x) = -ratToReal x := by
  simp [ratToReal]

@[simp] theorem ratToReal_if {p : Prop} [Decidable p] (a b : Rat) :
    ratToReal (if p then a else b) =
      if p then ratToReal a else ratToReal b := by
  by_cases hp : p <;> simp [hp, ratToReal]

theorem ratToReal_le_iff {x y : Rat} :
    ratToReal x ≤ ratToReal y ↔ x ≤ y := by
  simp [ratToReal]

/-- Rational order implies real order after casting. -/
theorem ratToReal_le_of_le {x y : Rat} (h : x ≤ y) :
    ratToReal x ≤ ratToReal y := by
  simpa [ratToReal_le_iff] using h

theorem ratToReal_lt_iff {x y : Rat} :
    ratToReal x < ratToReal y ↔ x < y := by
  simp [ratToReal]

theorem ratToReal_nonneg_iff {x : Rat} :
    0 ≤ ratToReal x ↔ 0 ≤ x := by
  simp [ratToReal]

theorem ratToReal_nonneg_of_nonneg {x : Rat} (h : 0 ≤ x) :
    0 ≤ ratToReal x := by
  simpa [ratToReal_nonneg_iff] using h

theorem ratToReal_nonpos_iff {x : Rat} :
    ratToReal x ≤ 0 ↔ x ≤ 0 := by
  simp [ratToReal]

@[simp] theorem ratToReal_abs (x : Rat) :
    ratToReal |x| = |ratToReal x| := by
  simp [ratToReal]

theorem ratToReal_abs_le_of_le {x y : Rat} (h : |x| ≤ y) :
    |ratToReal x| ≤ ratToReal y := by
  simpa [ratToReal_abs] using ratToReal_le_of_le h

@[simp] theorem ratToReal_max (x y : Rat) :
    ratToReal (max x y) = max (ratToReal x) (ratToReal y) := by
  simp [ratToReal]

@[simp] theorem ratToReal_min (x y : Rat) :
    ratToReal (min x y) = min (ratToReal x) (ratToReal y) := by
  simp [ratToReal]

end Nfp

end
