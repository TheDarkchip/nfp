-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.NNReal.Basic
import Mathlib.Algebra.Order.Floor.Defs
import Mathlib.Data.Rat.Cast.Lemmas
import Mathlib.Data.Rat.Cast.Order
import Init.Data.Dyadic
import Init.Data.Dyadic.Inv
import Init.Data.Dyadic.Round

/-!
Basic shared definitions for the NFP rewrite.
-/

namespace Nfp

/-- Nonnegative mass used for probabilities and weights. -/
abbrev Mass := NNReal

instance : ToString Dyadic :=
  ⟨fun x => toString x.toRat⟩

/-- Default dyadic precision (binary digits after the point). -/
def defaultDyadicPrec : Int := 48

/-- One ulp at the given dyadic precision. -/
def dyadicUlp (prec : Int := defaultDyadicPrec) : Dyadic :=
  Dyadic.ofIntWithPrec 1 prec

/-- Round a rational down to dyadic precision. -/
def dyadicOfRatDown (q : Rat) (prec : Int := defaultDyadicPrec) : Dyadic :=
  Rat.toDyadic q prec

/-- Round a rational up to dyadic precision. -/
def dyadicOfRatUp (q : Rat) (prec : Int := defaultDyadicPrec) : Dyadic :=
  Rat.toDyadic q prec + dyadicUlp prec

instance : Coe Dyadic Rat := ⟨Dyadic.toRat⟩

/-- Real cast of a dyadic value via `Rat`. -/
def dyadicToReal (x : Dyadic) : Real :=
  (x.toRat : Real)

instance : Coe Dyadic Real := ⟨dyadicToReal⟩

@[simp] theorem dyadicToReal_zero : dyadicToReal 0 = 0 := by
  simp [dyadicToReal]

@[simp] theorem dyadicToReal_one : dyadicToReal 1 = 1 := by
  change ((1 : Dyadic).toRat : Real) = 1
  have h : (1 : Dyadic).toRat = (1 : Rat) := Dyadic.toRat_natCast 1
  simp [h]

@[simp] theorem dyadicToRat_one : (Dyadic.toRat 1 : Rat) = 1 := by
  exact Dyadic.toRat_natCast 1

@[simp] theorem dyadicOfInt_zero : Dyadic.ofInt 0 = 0 := by
  simp [Dyadic.ofInt, Dyadic.ofIntWithPrec]

@[simp] theorem dyadicOfInt_toRat (i : Int) : (Dyadic.ofInt i).toRat = i := by
  change ((i : Dyadic).toRat = i)
  exact Dyadic.toRat_intCast (x := i)

@[simp] theorem dyadicOfInt_succ (n : Nat) : Dyadic.ofInt (n + 1) = Dyadic.ofInt n + 1 := by
  apply (Dyadic.toRat_inj).1
  calc
    (Dyadic.ofInt (n + 1)).toRat = (n + 1 : Int) := by
      simp
    _ = (n : Int) + 1 := by
      simp
    _ = (Dyadic.ofInt n).toRat + 1 := by
      simp
    _ = (Dyadic.ofInt n + 1).toRat := by
      simp [Dyadic.toRat_add]

theorem dyadicOfRatDown_le (q : Rat) (prec : Int := defaultDyadicPrec) :
    (dyadicOfRatDown q prec : Rat) ≤ q := by
  simpa [dyadicOfRatDown] using (Rat.toRat_toDyadic_le (x := q) (prec := prec))

theorem dyadicOfRatUp_ge (q : Rat) (prec : Int := defaultDyadicPrec) :
    q ≤ (dyadicOfRatUp q prec : Rat) := by
  have hlt := Rat.lt_toRat_toDyadic_add (x := q) (prec := prec)
  exact le_of_lt (by simpa [dyadicOfRatUp, dyadicUlp] using hlt)

theorem dyadicOfRatDown_le_real (q : Rat) (prec : Int := defaultDyadicPrec) :
    (dyadicOfRatDown q prec : Real) ≤ (q : Real) := by
  have h :=
    (Rat.cast_le (K := Real) (p := (dyadicOfRatDown q prec : Rat)) (q := q)).2
      (dyadicOfRatDown_le q prec)
  simpa [dyadicToReal] using h

theorem real_le_dyadicOfRatUp (q : Rat) (prec : Int := defaultDyadicPrec) :
    (q : Real) ≤ (dyadicOfRatUp q prec : Real) := by
  have h :=
    (Rat.cast_le (K := Real) (p := q) (q := (dyadicOfRatUp q prec : Rat))).2
      (dyadicOfRatUp_ge q prec)
  simpa [dyadicToReal] using h

theorem dyadicOfRatDown_lt_add_ulp (q : Rat) (prec : Int := defaultDyadicPrec) :
    q < (dyadicOfRatDown q prec : Rat) + (dyadicUlp prec : Rat) := by
  have h := Rat.lt_toRat_toDyadic_add (x := q) (prec := prec)
  simpa [dyadicOfRatDown, dyadicUlp, Dyadic.toRat_add] using h

theorem dyadicOfRatDown_sub_ulp_lt (q : Rat) (prec : Int := defaultDyadicPrec) :
    q - (dyadicUlp prec : Rat) < (dyadicOfRatDown q prec : Rat) := by
  exact (sub_lt_iff_lt_add).2 (dyadicOfRatDown_lt_add_ulp q prec)

theorem dyadicOfRatUp_le_add_ulp (q : Rat) (prec : Int := defaultDyadicPrec) :
    (dyadicOfRatUp q prec : Rat) ≤ q + (dyadicUlp prec : Rat) := by
  have hdown : (dyadicOfRatDown q prec : Rat) ≤ q := dyadicOfRatDown_le q prec
  have hsum : (dyadicOfRatDown q prec : Rat) + (dyadicUlp prec : Rat) ≤
      q + (dyadicUlp prec : Rat) := by
    exact Rat.add_le_add_right.2 hdown
  simpa [dyadicOfRatUp, dyadicUlp, dyadicOfRatDown, Dyadic.toRat_add] using hsum

theorem dyadicOfRatDown_lt_add_ulp_real (q : Rat) (prec : Int := defaultDyadicPrec) :
    (q : Real) < (dyadicOfRatDown q prec : Real) + (dyadicUlp prec : Real) := by
  have h :=
    (Rat.cast_lt (K := Real) (p := q)
      (q := (dyadicOfRatDown q prec : Rat) + (dyadicUlp prec : Rat))).2
      (dyadicOfRatDown_lt_add_ulp q prec)
  simpa [dyadicToReal] using h

theorem dyadicOfRatUp_le_add_ulp_real (q : Rat) (prec : Int := defaultDyadicPrec) :
    (dyadicOfRatUp q prec : Real) ≤ (q : Real) + (dyadicUlp prec : Real) := by
  have h :=
    (Rat.cast_le (K := Real)
      (p := (dyadicOfRatUp q prec : Rat))
      (q := q + (dyadicUlp prec : Rat))).2
      (dyadicOfRatUp_le_add_ulp q prec)
  simpa [dyadicToReal] using h

theorem dyadicOfRatDown_nonneg {q : Rat} (hq : 0 ≤ q) (prec : Int := defaultDyadicPrec) :
    0 ≤ dyadicOfRatDown q prec := by
  apply (Dyadic.toRat_le_toRat_iff (x := 0) (y := dyadicOfRatDown q prec)).1
  have hrat :
      (dyadicOfRatDown q prec : Rat) =
        (q * 2 ^ prec).floor / 2 ^ prec := by
    simpa [dyadicOfRatDown] using (Rat.toRat_toDyadic (x := q) (prec := prec))
  have hpow_pos : (0 : Rat) < (2 : Rat) ^ prec := by
    exact Rat.zpow_pos (by decide : (0 : Rat) < 2)
  have hmul_nonneg : (0 : Rat) ≤ q * 2 ^ prec := by
    exact mul_nonneg hq (le_of_lt hpow_pos)
  have hfloor_nonneg : 0 ≤ (q * 2 ^ prec).floor := by
    exact (Int.floor_nonneg (a := q * 2 ^ prec)).2 hmul_nonneg
  have hfloor_nonneg_rat : (0 : Rat) ≤ ((q * 2 ^ prec).floor : Rat) := by
    exact_mod_cast hfloor_nonneg
  have hdiv_nonneg :
      (0 : Rat) ≤ ((q * 2 ^ prec).floor : Rat) / (2 : Rat) ^ prec := by
    exact div_nonneg hfloor_nonneg_rat (le_of_lt hpow_pos)
  simpa [hrat] using hdiv_nonneg

private lemma dyadicUlp_rat (prec : Int) :
    (dyadicUlp prec : Rat) = (1 : Rat) * 2 ^ (-prec) := by
  simp [dyadicUlp, Dyadic.toRat_ofIntWithPrec_eq_mul_two_pow]

theorem dyadicUlp_nonneg (prec : Int := defaultDyadicPrec) : 0 ≤ dyadicUlp prec := by
  apply (Dyadic.toRat_le_toRat_iff (x := 0) (y := dyadicUlp prec)).1
  have hrat : (dyadicUlp prec : Rat) = (1 : Rat) * 2 ^ (-prec) := dyadicUlp_rat prec
  have hpow_pos : (0 : Rat) < (2 : Rat) ^ (-prec) := by
    exact Rat.zpow_pos (by decide : (0 : Rat) < 2)
  have hnonneg : (0 : Rat) ≤ (1 : Rat) * 2 ^ (-prec) := by
    exact mul_nonneg (by decide : (0 : Rat) ≤ 1) (le_of_lt hpow_pos)
  simpa [hrat] using hnonneg

theorem dyadicUlp_pos (prec : Int := defaultDyadicPrec) : 0 < dyadicUlp prec := by
  apply (Dyadic.toRat_lt_toRat_iff (x := 0) (y := dyadicUlp prec)).1
  have hrat : (dyadicUlp prec : Rat) = (1 : Rat) * 2 ^ (-prec) := dyadicUlp_rat prec
  have hpow_pos : (0 : Rat) < (2 : Rat) ^ (-prec) := by
    exact Rat.zpow_pos (by decide : (0 : Rat) < 2)
  have hpos : (0 : Rat) < (1 : Rat) * 2 ^ (-prec) := by
    exact mul_pos (by decide : (0 : Rat) < 1) hpow_pos
  simpa [hrat] using hpos

theorem dyadicOfRatUp_nonneg {q : Rat} (hq : 0 ≤ q) (prec : Int := defaultDyadicPrec) :
    0 ≤ dyadicOfRatUp q prec := by
  have hdown : 0 ≤ dyadicOfRatDown q prec := dyadicOfRatDown_nonneg hq prec
  have hulp : 0 ≤ dyadicUlp prec := dyadicUlp_nonneg prec
  apply (Dyadic.toRat_le_toRat_iff (x := 0) (y := dyadicOfRatUp q prec)).1
  have hdown_rat : (0 : Rat) ≤ (dyadicOfRatDown q prec : Rat) :=
    (Dyadic.toRat_le_toRat_iff (x := 0) (y := dyadicOfRatDown q prec)).2 hdown
  have hulp_rat : (0 : Rat) ≤ (dyadicUlp prec : Rat) :=
    (Dyadic.toRat_le_toRat_iff (x := 0) (y := dyadicUlp prec)).2 hulp
  have hsum : (0 : Rat) ≤ (dyadicOfRatDown q prec : Rat) + (dyadicUlp prec : Rat) := by
    exact Rat.add_nonneg hdown_rat hulp_rat
  simpa [dyadicOfRatUp, dyadicUlp, dyadicOfRatDown, Dyadic.toRat_add] using hsum

theorem dyadicOfRatUp_pos {q : Rat} (hq : 0 < q) (prec : Int := defaultDyadicPrec) :
    0 < dyadicOfRatUp q prec := by
  apply (Dyadic.toRat_lt_toRat_iff (x := 0) (y := dyadicOfRatUp q prec)).1
  have hdown_nonneg : (0 : Rat) ≤ (dyadicOfRatDown q prec : Rat) := by
    have hdown : 0 ≤ dyadicOfRatDown q prec := dyadicOfRatDown_nonneg hq.le prec
    exact (Dyadic.toRat_le_toRat_iff (x := 0) (y := dyadicOfRatDown q prec)).2 hdown
  have hulp_pos : (0 : Rat) < (dyadicUlp prec : Rat) := by
    exact (Dyadic.toRat_lt_toRat_iff (x := 0) (y := dyadicUlp prec)).2 (dyadicUlp_pos prec)
  have hsum : (0 : Rat) < (dyadicOfRatDown q prec : Rat) + (dyadicUlp prec : Rat) := by
    nlinarith
  simpa [dyadicOfRatUp, dyadicUlp, dyadicOfRatDown, Dyadic.toRat_add] using hsum

-- TODO: use a tighter upper rounding when needed.

/-- Dyadic division with downward rounding at the chosen precision. -/
def dyadicDivDown (x y : Dyadic) (prec : Int := defaultDyadicPrec) : Dyadic :=
  if y = 0 then
    0
  else
    dyadicOfRatDown (x.toRat / y.toRat) prec

/-- Dyadic division with upward rounding at the chosen precision. -/
def dyadicDivUp (x y : Dyadic) (prec : Int := defaultDyadicPrec) : Dyadic :=
  if y = 0 then
    0
  else
    dyadicOfRatUp (x.toRat / y.toRat) prec

theorem dyadicDivUp_ge (x y : Dyadic) (hy : y ≠ 0) :
    (x.toRat / y.toRat : Rat) ≤ (dyadicDivUp x y : Rat) := by
  have hrat : (x.toRat / y.toRat : Rat) ≤ (dyadicOfRatUp (x.toRat / y.toRat) : Rat) :=
    dyadicOfRatUp_ge (x.toRat / y.toRat)
  simpa [dyadicDivUp, hy] using hrat

theorem dyadicDivUp_ge_real (x y : Dyadic) (hy : y ≠ 0) :
    (x.toRat : Real) / (y.toRat : Real) ≤ dyadicToReal (dyadicDivUp x y) := by
  have hrat : (x.toRat / y.toRat : Rat) ≤ (dyadicDivUp x y : Rat) :=
    dyadicDivUp_ge x y hy
  have hrat' : ((x.toRat / y.toRat : Rat) : Real) ≤ ((dyadicDivUp x y : Rat) : Real) :=
    (Rat.cast_le (K := Real) (p := _) (q := _)).2 hrat
  simpa [dyadicToReal] using hrat'

@[simp] theorem dyadicToReal_add (x y : Dyadic) :
    dyadicToReal (x + y) = dyadicToReal x + dyadicToReal y := by
  simp [dyadicToReal, Dyadic.toRat_add]

@[simp] theorem dyadicToReal_sub (x y : Dyadic) :
    dyadicToReal (x - y) = dyadicToReal x - dyadicToReal y := by
  simp [dyadicToReal, Dyadic.toRat_sub]

@[simp] theorem dyadicToReal_mul (x y : Dyadic) :
    dyadicToReal (x * y) = dyadicToReal x * dyadicToReal y := by
  simp [dyadicToReal, Dyadic.toRat_mul]

@[simp] theorem dyadicToReal_neg (x : Dyadic) :
    dyadicToReal (-x) = -dyadicToReal x := by
  simp [dyadicToReal, Dyadic.toRat_neg]

@[simp] theorem dyadicToReal_if {p : Prop} [Decidable p] (a b : Dyadic) :
    dyadicToReal (if p then a else b) =
      if p then dyadicToReal a else dyadicToReal b := by
  by_cases hp : p <;> simp [hp]

theorem dyadicToReal_le_iff {x y : Dyadic} :
    dyadicToReal x ≤ dyadicToReal y ↔ x ≤ y := by
  constructor
  · intro h
    have h' : x.toRat ≤ y.toRat := by
      have h'' : (x.toRat : Real) ≤ (y.toRat : Real) := by
        simpa [dyadicToReal] using h
      exact (Rat.cast_le (K := Real) (p := x.toRat) (q := y.toRat)).1 h''
    exact (Dyadic.toRat_le_toRat_iff).1 h'
  · intro h
    have h' : x.toRat ≤ y.toRat := (Dyadic.toRat_le_toRat_iff).2 h
    have h'' : (x.toRat : Real) ≤ (y.toRat : Real) :=
      (Rat.cast_le (K := Real) (p := x.toRat) (q := y.toRat)).2 h'
    simpa [dyadicToReal] using h''

/-- Dyadic order implies real order after casting. -/
theorem dyadicToReal_le_of_le {x y : Dyadic} (h : x ≤ y) :
    dyadicToReal x ≤ dyadicToReal y :=
  (dyadicToReal_le_iff (x := x) (y := y)).2 h

theorem dyadicToReal_lt_iff {x y : Dyadic} :
    dyadicToReal x < dyadicToReal y ↔ x < y := by
  constructor
  · intro h
    have h' : x.toRat < y.toRat := by
      have h'' : (x.toRat : Real) < (y.toRat : Real) := by
        simpa [dyadicToReal] using h
      exact (Rat.cast_lt (K := Real) (p := x.toRat) (q := y.toRat)).1 h''
    exact (Dyadic.toRat_lt_toRat_iff).1 h'
  · intro h
    have h' : x.toRat < y.toRat := (Dyadic.toRat_lt_toRat_iff).2 h
    have h'' : (x.toRat : Real) < (y.toRat : Real) :=
      (Rat.cast_lt (K := Real) (p := x.toRat) (q := y.toRat)).2 h'
    simpa [dyadicToReal] using h''

theorem dyadicToReal_nonneg_iff {x : Dyadic} :
    0 ≤ dyadicToReal x ↔ 0 ≤ x := by
  simpa [dyadicToReal] using (dyadicToReal_le_iff (x := 0) (y := x))

theorem dyadicToReal_nonneg_of_nonneg {x : Dyadic} (h : 0 ≤ x) :
    0 ≤ dyadicToReal x :=
  (dyadicToReal_nonneg_iff (x := x)).2 h

theorem dyadicToReal_nonpos_iff {x : Dyadic} :
    dyadicToReal x ≤ 0 ↔ x ≤ 0 := by
  simpa [dyadicToReal_zero] using (dyadicToReal_le_iff (x := x) (y := 0))

instance : LinearOrder Dyadic where
  le := (· ≤ ·)
  lt := (· < ·)
  le_refl := Dyadic.le_refl
  le_trans := by intro a b c hab hbc; exact Dyadic.le_trans hab hbc
  le_antisymm := by intro a b hab hba; exact Dyadic.le_antisymm hab hba
  le_total := Dyadic.le_total
  toDecidableLE := inferInstance
  toDecidableEq := inferInstance
  toDecidableLT := inferInstance
  lt_iff_le_not_ge := by
    intro a b
    have hlt : a < b ↔ a.toRat < b.toRat :=
      (Dyadic.toRat_lt_toRat_iff (x := a) (y := b)).symm
    have hle : a ≤ b ↔ a.toRat ≤ b.toRat :=
      (Dyadic.toRat_le_toRat_iff (x := a) (y := b)).symm
    have hge : b ≤ a ↔ b.toRat ≤ a.toRat :=
      (Dyadic.toRat_le_toRat_iff (x := b) (y := a)).symm
    have hrat : a.toRat < b.toRat ↔ a.toRat ≤ b.toRat ∧ ¬ b.toRat ≤ a.toRat := by
      simpa using (Rat.lt_iff_le_not_ge (a := a.toRat) (b := b.toRat))
    calc
      a < b ↔ a.toRat < b.toRat := hlt
      _ ↔ a.toRat ≤ b.toRat ∧ ¬ b.toRat ≤ a.toRat := hrat
      _ ↔ a ≤ b ∧ ¬ b ≤ a := by
            constructor
            · intro h
              refine ⟨(hle.mpr h.1), ?_⟩
              intro hba
              exact h.2 (hge.mp hba)
            · intro h
              refine ⟨(hle.mp h.1), ?_⟩
              intro hba
              exact h.2 (hge.mpr hba)
  min_def := by intro a b; rfl
  max_def := by intro a b; rfl
  compare_eq_compareOfLessAndEq := by intro a b; rfl

instance : AddMonoid Dyadic where
  add := (· + ·)
  zero := 0
  add_assoc := Dyadic.add_assoc
  zero_add := Dyadic.zero_add
  add_zero := Dyadic.add_zero
  nsmul := nsmulRec
  nsmul_zero := by intro x; rfl
  nsmul_succ := by intro n x; rfl

instance : AddCommMonoid Dyadic :=
  { (inferInstance : AddMonoid Dyadic) with
    add_comm := Dyadic.add_comm }

instance : AddMonoidWithOne Dyadic where
  add := (· + ·)
  zero := 0
  add_assoc := Dyadic.add_assoc
  zero_add := Dyadic.zero_add
  add_zero := Dyadic.add_zero
  nsmul := nsmulRec
  nsmul_zero := by intro x; rfl
  nsmul_succ := by intro n x; rfl
  one := 1
  natCast := fun n => Dyadic.ofInt n
  natCast_zero := by
    simp [dyadicOfInt_zero]
  natCast_succ := by
    intro n
    simp [dyadicOfInt_succ n]

instance : AddCommMonoidWithOne Dyadic where
  add := (· + ·)
  zero := 0
  add_assoc := Dyadic.add_assoc
  zero_add := Dyadic.zero_add
  add_zero := Dyadic.add_zero
  nsmul := nsmulRec
  nsmul_zero := by intro x; rfl
  nsmul_succ := by intro n x; rfl
  one := 1
  natCast := fun n => Dyadic.ofInt n
  natCast_zero := by
    simp [dyadicOfInt_zero]
  natCast_succ := by
    intro n
    simp [dyadicOfInt_succ n]
  add_comm := Dyadic.add_comm

instance : AddGroup Dyadic where
  add := (· + ·)
  zero := 0
  add_assoc := Dyadic.add_assoc
  zero_add := Dyadic.zero_add
  add_zero := Dyadic.add_zero
  nsmul := nsmulRec
  nsmul_zero := by intro x; rfl
  nsmul_succ := by intro n x; rfl
  neg := Neg.neg
  sub := fun a b => a + -b
  sub_eq_add_neg := by intro a b; rfl
  zsmul := zsmulRec
  zsmul_zero' := by intro a; rfl
  zsmul_succ' := by intro n a; rfl
  zsmul_neg' := by intro n a; rfl
  neg_add_cancel := Dyadic.neg_add_cancel

instance : AddCommGroup Dyadic :=
  { (inferInstance : AddGroup Dyadic) with
    add_comm := Dyadic.add_comm }

instance : IsOrderedAddMonoid Dyadic where
  add_le_add_left a b h c := by
    have hrat : a.toRat ≤ b.toRat :=
      (Dyadic.toRat_le_toRat_iff (x := a) (y := b)).2 h
    have hrat' : a.toRat + c.toRat ≤ b.toRat + c.toRat := by
      exact Rat.add_le_add_right.2 hrat
    have hrat'' :
        (a + c).toRat ≤ (b + c).toRat := by
      simpa [Dyadic.toRat_add] using hrat'
    exact (Dyadic.toRat_le_toRat_iff (x := a + c) (y := b + c)).1 hrat''

instance : ExistsAddOfLE Dyadic where
  exists_add_of_le {a b} h := by
    refine ⟨b - a, ?_⟩
    simp [sub_eq_add_neg]

instance : Monoid Dyadic where
  mul := (· * ·)
  one := 1
  mul_assoc := Dyadic.mul_assoc
  one_mul := Dyadic.one_mul
  mul_one := Dyadic.mul_one

instance : MonoidWithZero Dyadic :=
  { (inferInstance : Monoid Dyadic) with
    zero := 0
    zero_mul := Dyadic.zero_mul
    mul_zero := Dyadic.mul_zero }

instance : Distrib Dyadic where
  left_distrib := Dyadic.mul_add
  right_distrib := Dyadic.add_mul

instance : Semiring Dyadic where
  add := (· + ·)
  zero := 0
  add_assoc := Dyadic.add_assoc
  zero_add := Dyadic.zero_add
  add_zero := Dyadic.add_zero
  add_comm := Dyadic.add_comm
  nsmul := nsmulRec
  nsmul_zero := by intro x; rfl
  nsmul_succ := by intro n x; rfl
  one := 1
  natCast := fun n => Dyadic.ofInt n
  natCast_zero := by
    simp [dyadicOfInt_zero]
  natCast_succ := by
    intro n
    simp [dyadicOfInt_succ n]
  mul := (· * ·)
  mul_assoc := Dyadic.mul_assoc
  one_mul := Dyadic.one_mul
  mul_one := Dyadic.mul_one
  left_distrib := Dyadic.mul_add
  right_distrib := Dyadic.add_mul
  zero_mul := Dyadic.zero_mul
  mul_zero := Dyadic.mul_zero

instance : CommSemiring Dyadic where
  toSemiring := (inferInstance : Semiring Dyadic)
  mul_comm := by intro a b; exact Dyadic.mul_comm a b

instance : AddGroupWithOne Dyadic :=
  { (inferInstance : AddMonoidWithOne Dyadic),
    (inferInstance : AddGroup Dyadic) with
    intCast := Int.castDef
    intCast_ofNat := by
      intro n
      apply (Dyadic.toRat_inj).1
      have hleft : (Int.castDef (R := Dyadic) (n : ℤ)).toRat = (n : Rat) := by
        simp [Int.castDef, Dyadic.toRat_natCast]
      have hright : (n : Dyadic).toRat = (n : Rat) := Dyadic.toRat_natCast n
      exact hleft.trans hright.symm
    intCast_negSucc := by
      intro n
      apply (Dyadic.toRat_inj).1
      simp [Int.castDef, Dyadic.toRat_natCast, Dyadic.toRat_neg, Dyadic.toRat_add] }

instance : Ring Dyadic :=
  { (inferInstance : Semiring Dyadic),
    (inferInstance : AddCommGroup Dyadic),
    (inferInstance : AddGroupWithOne Dyadic) with }

instance : CommRing Dyadic :=
  { (inferInstance : Ring Dyadic), (inferInstance : CommSemiring Dyadic) with }

instance : Nontrivial Dyadic :=
  ⟨0, 1, by
    intro h
    have hrat : (0 : Rat) = (1 : Rat) := by
      simpa [Dyadic.toRat_zero, dyadicToRat_one] using congrArg Dyadic.toRat h
    exact (zero_ne_one (α := Rat)) hrat⟩

instance : ZeroLEOneClass Dyadic where
  zero_le_one := by
    have hrat : (0 : Rat) ≤ (1 : Rat) := by decide
    exact (Dyadic.toRat_le_toRat_iff (x := (0 : Dyadic)) (y := (1 : Dyadic))).1 hrat

instance : PosMulMono Dyadic where
  mul_le_mul_of_nonneg_left {a} ha {b c} hbc := by
    have ha' : (0 : Dyadic) ≤ a := by simpa using ha
    have ha'' : (0 : Rat) ≤ a.toRat :=
      (Dyadic.toRat_le_toRat_iff (x := 0) (y := a)).2 ha'
    have hbc' : b.toRat ≤ c.toRat :=
      (Dyadic.toRat_le_toRat_iff (x := b) (y := c)).2 hbc
    have hrat : a.toRat * b.toRat ≤ a.toRat * c.toRat := by
      exact Rat.mul_le_mul_of_nonneg_left hbc' ha''
    have hrat' :
        (a * b).toRat ≤ (a * c).toRat := by
      simpa [Dyadic.toRat_mul] using hrat
    exact (Dyadic.toRat_le_toRat_iff (x := a * b) (y := a * c)).1 hrat'

instance : MulPosMono Dyadic where
  mul_le_mul_of_nonneg_right {a} ha {b c} hbc := by
    have h := (PosMulMono.mul_le_mul_of_nonneg_left (a := a) ha hbc)
    simpa [mul_comm, mul_left_comm, mul_assoc] using h

instance : IsOrderedRing Dyadic :=
  IsOrderedRing.of_mul_nonneg (R := Dyadic) (mul_nonneg := by
    intro a b ha hb
    have ha' : (0 : Dyadic) ≤ a := by simpa using ha
    have hb' : (0 : Dyadic) ≤ b := by simpa using hb
    have ha'' : (0 : Rat) ≤ a.toRat :=
      (Dyadic.toRat_le_toRat_iff (x := 0) (y := a)).2 ha'
    have hb'' : (0 : Rat) ≤ b.toRat :=
      (Dyadic.toRat_le_toRat_iff (x := 0) (y := b)).2 hb'
    have hrat : (0 : Rat) ≤ a.toRat * b.toRat := Rat.mul_nonneg ha'' hb''
    have hrat' : (0 : Rat) ≤ (a * b).toRat := by
      simpa [Dyadic.toRat_mul] using hrat
    exact (Dyadic.toRat_le_toRat_iff (x := 0) (y := a * b)).1 hrat')

@[simp] theorem dyadicToReal_abs (x : Dyadic) :
    dyadicToReal |x| = |dyadicToReal x| := by
  by_cases hx : 0 ≤ x
  · have hx' : 0 ≤ dyadicToReal x := (dyadicToReal_nonneg_iff).2 hx
    calc
      dyadicToReal |x| = dyadicToReal x := by simp [abs_of_nonneg hx]
      _ = |dyadicToReal x| := (abs_of_nonneg hx').symm
  · have hx' : x ≤ 0 := le_of_not_ge hx
    have hx'' : dyadicToReal x ≤ 0 := by
      simpa [dyadicToReal_zero] using dyadicToReal_le_of_le hx'
    calc
      dyadicToReal |x| = dyadicToReal (-x) := by simp [abs_of_nonpos hx']
      _ = -dyadicToReal x := dyadicToReal_neg x
      _ = |dyadicToReal x| := (abs_of_nonpos hx'').symm

theorem dyadicToReal_abs_le_of_le {x y : Dyadic} (h : |x| ≤ y) :
    |dyadicToReal x| ≤ dyadicToReal y := by
  have h' : dyadicToReal |x| ≤ dyadicToReal y :=
    dyadicToReal_le_of_le h
  simpa [dyadicToReal_abs] using h'

@[simp] theorem dyadicToReal_max (x y : Dyadic) :
    dyadicToReal (max x y) = max (dyadicToReal x) (dyadicToReal y) := by
  by_cases hxy : x ≤ y
  · have hxy' : dyadicToReal x ≤ dyadicToReal y :=
      dyadicToReal_le_of_le hxy
    calc
      dyadicToReal (max x y) = dyadicToReal y := by simp [max_eq_right hxy]
      _ = max (dyadicToReal x) (dyadicToReal y) := by
        symm
        exact max_eq_right hxy'
  · have hyx : y ≤ x := le_of_not_ge hxy
    have hyx' : dyadicToReal y ≤ dyadicToReal x :=
      dyadicToReal_le_of_le hyx
    calc
      dyadicToReal (max x y) = dyadicToReal x := by simp [max_eq_left hyx]
      _ = max (dyadicToReal x) (dyadicToReal y) := by
        exact (max_eq_left hyx').symm

@[simp] theorem dyadicToReal_min (x y : Dyadic) :
    dyadicToReal (min x y) = min (dyadicToReal x) (dyadicToReal y) := by
  by_cases hxy : x ≤ y
  · have hxy' : dyadicToReal x ≤ dyadicToReal y :=
      dyadicToReal_le_of_le hxy
    calc
      dyadicToReal (min x y) = dyadicToReal x := by simp [min_eq_left hxy]
      _ = min (dyadicToReal x) (dyadicToReal y) := by
        symm
        exact min_eq_left hxy'
  · have hyx : y ≤ x := le_of_not_ge hxy
    have hyx' : dyadicToReal y ≤ dyadicToReal x :=
      dyadicToReal_le_of_le hyx
    calc
      dyadicToReal (min x y) = dyadicToReal y := by simp [min_eq_right hyx]
      _ = min (dyadicToReal x) (dyadicToReal y) := by
        exact (min_eq_right hyx').symm

end Nfp
