-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Data.Matrix.Mul
public import Mathlib.Data.Rat.BigOperators
public import Batteries.Data.Fin.Fold
public import Nfp.Core.Basic

/-!
Tail-recursive folds and sums over `Fin`.

These helpers keep sound computations stack-safe while remaining explicit.
-/

public section

namespace Nfp

namespace Sound

namespace Linear

variable {α : Type _}

/-- Tail-recursive fold over `Fin n`. -/
def foldlFin (n : Nat) (f : α → Fin n → α) (init : α) : α :=
  Fin.dfoldl n (fun _ => α) (fun i acc => f acc i) init

/-- `foldlFin` matches `Fin.foldl`. -/
theorem foldlFin_eq_foldl (n : Nat) (f : α → Fin n → α) (init : α) :
    foldlFin n f init = Fin.foldl n f init := by
  simpa [foldlFin] using
    (Fin.dfoldl_eq_foldl (n := n) (f := fun i acc => f acc i) (x := init))

/-- Tail-recursive sum over `Fin n` (Rat-valued). -/
def sumFin (n : Nat) (f : Fin n → Rat) : Rat :=
  foldlFin n (fun acc i => acc + f i) 0

/-- Tail-recursive sum over `Fin n` (alias for `sumFin`). -/
def sumFinCommonDen (n : Nat) (f : Fin n → Rat) : Rat :=
  sumFin n f

/-- `sumFin` as a left fold over the finite range list. -/
theorem sumFin_eq_list_foldl (n : Nat) (f : Fin n → Rat) :
    sumFin n f = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
  simpa [sumFin, foldlFin_eq_foldl] using
    (Fin.foldl_eq_foldl_finRange (f := fun acc i => acc + f i) (x := (0 : Rat)) (n := n))

/-- `sumFin` agrees with the `Finset.univ` sum. -/
theorem sumFin_eq_sum_univ {n : Nat} (f : Fin n → Rat) :
    sumFin n f = ∑ i, f i := by
  classical
  have hsum_list :
      ((List.finRange n).map f).sum = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
    have hmap :
        ((List.finRange n).map f).foldl (fun acc x : Rat => acc + x) 0 =
          (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
      simpa using
        (List.foldl_map (f := f) (g := fun acc x : Rat => acc + x)
          (l := List.finRange n) (init := (0 : Rat)))
    calc
      ((List.finRange n).map f).sum
          = ((List.finRange n).map f).foldl (fun acc x : Rat => acc + x) 0 := by
            simpa using (List.sum_eq_foldl (l := (List.finRange n).map f))
      _ = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
            exact hmap
  have hsum_univ : ((List.finRange n).map f).sum = ∑ i, f i := by
    simpa using (Fin.sum_univ_def f).symm
  calc
    sumFin n f
        = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
            simpa using sumFin_eq_list_foldl n f
    _ = ((List.finRange n).map f).sum := hsum_list.symm
    _ = ∑ i, f i := hsum_univ

/-- Casting a `Finset.univ` rational sum to `Real` commutes with summation. -/
theorem ratToReal_sum_univ {n : Nat} (f : Fin n → Rat) :
    ratToReal (∑ i, f i) = ∑ i, ratToReal (f i) := by
  classical
  simp [ratToReal_def]

/-- Casting a rational `sumFin` to `Real` commutes with summation. -/
theorem ratToReal_sumFin {n : Nat} (f : Fin n → Rat) :
    ratToReal (sumFin n f) = ∑ i, ratToReal (f i) := by
  classical
  simpa [sumFin_eq_sum_univ] using ratToReal_sum_univ (f := f)

/-- `sumFinCommonDen` agrees with `sumFin`. -/
theorem sumFinCommonDen_eq_sumFin (n : Nat) (f : Fin n → Rat) :
    sumFinCommonDen n f = sumFin n f := by
  simp [sumFinCommonDen]

/-- Dot product over `Fin n` (Rat-valued). -/
def dotFin (n : Nat) (x y : Fin n → Rat) : Rat :=
  sumFin n (fun i => x i * y i)

/-- Unfolding lemma for `dotFin`. -/
theorem dotFin_def (n : Nat) (x y : Fin n → Rat) :
    dotFin n x y = sumFin n (fun i => x i * y i) := by
  simp [dotFin]

/-- `dotFin` matches `dotProduct`. -/
theorem dotFin_eq_dotProduct (n : Nat) (x y : Fin n → Rat) :
    dotFin n x y = dotProduct x y := by
  simp [dotFin_def, sumFin_eq_sum_univ, dotProduct]

/-- Right-distribute dot products over subtraction (Real-valued). -/
theorem dotProduct_sub_right {n : Nat} (x y z : Fin n → Real) :
    dotProduct x (fun i => y i - z i) = dotProduct x y - dotProduct x z := by
  classical
  simp only [dotProduct, mul_sub, Finset.sum_sub_distrib]

/-- Right-distribute dot products over addition (Real-valued). -/
theorem dotProduct_add_right {n : Nat} (x y z : Fin n → Real) :
    dotProduct x (fun i => y i + z i) = dotProduct x y + dotProduct x z := by
  classical
  simp only [dotProduct, mul_add, Finset.sum_add_distrib]

/-- Pull a constant factor out of the right-hand side of a dot product. -/
theorem dotProduct_mul_right {n : Nat} (x y : Fin n → Real) (a : Real) :
    dotProduct x (fun i => y i * a) = dotProduct x y * a := by
  classical
  simp only [dotProduct, mul_assoc, Finset.sum_mul]

end Linear

end Sound

end Nfp
