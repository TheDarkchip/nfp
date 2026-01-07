-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Matrix.Mul
import Batteries.Data.Fin.Fold
import Nfp.Core.Basic

/-!
Tail-recursive folds and sums over `Fin`.

These helpers keep sound computations stack-safe while remaining explicit.
-/

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

/-- Tail-recursive sum over `Fin n` (Dyadic-valued). -/
def sumFin (n : Nat) (f : Fin n → Dyadic) : Dyadic :=
  foldlFin n (fun acc i => acc + f i) 0

/-- Tail-recursive sum over `Fin n` (alias for `sumFin`). -/
def sumFinCommonDen (n : Nat) (f : Fin n → Dyadic) : Dyadic :=
  sumFin n f

/-- `sumFin` as a left fold over the finite range list. -/
theorem sumFin_eq_list_foldl (n : Nat) (f : Fin n → Dyadic) :
    sumFin n f = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
  simpa [sumFin, foldlFin_eq_foldl] using
    (Fin.foldl_eq_foldl_finRange (f := fun acc i => acc + f i) (x := (0 : Dyadic)) (n := n))

/-- `sumFin` agrees with the `Finset.univ` sum. -/
theorem sumFin_eq_sum_univ {n : Nat} (f : Fin n → Dyadic) :
    sumFin n f = ∑ i, f i := by
  classical
  have hfold :
      sumFin n f = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
    simpa using sumFin_eq_list_foldl n f
  have hmap :
      ((List.finRange n).map f).foldl (fun acc x : Dyadic => acc + x) 0 =
        (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
    simpa using
      (List.foldl_map (f := f) (g := fun acc x : Dyadic => acc + x)
        (l := List.finRange n) (init := (0 : Dyadic)))
  let _ : Std.Commutative (fun a b : Dyadic => a + b) :=
    ⟨by intro a b; exact add_comm _ _⟩
  let _ : Std.Associative (fun a b : Dyadic => a + b) :=
    ⟨by intro a b c; exact add_assoc _ _ _⟩
  have hfoldr :
      ((List.finRange n).map f).foldl (fun acc x : Dyadic => acc + x) 0 =
        ((List.finRange n).map f).foldr (fun x acc => x + acc) 0 := by
    simpa using
      (List.foldl_eq_foldr (f := fun acc x : Dyadic => acc + x)
        (a := 0) (l := (List.finRange n).map f))
  have hsum_list :
      ((List.finRange n).map f).sum = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
    calc
      ((List.finRange n).map f).sum
          = ((List.finRange n).map f).foldr (fun x acc => x + acc) 0 := by
            rfl
      _ = ((List.finRange n).map f).foldl (fun acc x : Dyadic => acc + x) 0 := by
            exact hfoldr.symm
      _ = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
            exact hmap
  have hsum_univ : ((List.finRange n).map f).sum = ∑ i, f i := by
    exact (Fin.sum_univ_def f).symm
  calc
    sumFin n f
        = (List.finRange n).foldl (fun acc i => acc + f i) 0 := hfold
    _ = ((List.finRange n).map f).sum := hsum_list.symm
    _ = ∑ i, f i := hsum_univ

/-- Casting a `Finset.univ` dyadic sum to `Real` commutes with summation. -/
theorem dyadicToReal_sum_univ {n : Nat} (f : Fin n → Dyadic) :
    ((∑ i, f i : Dyadic) : Real) = ∑ i, (f i : Real) := by
  classical
  refine Finset.induction_on (Finset.univ : Finset (Fin n)) ?_ ?_
  · simp
  · intro a s ha hs
    simp [Finset.sum_insert, ha, hs, dyadicToReal_add]

/-- Casting a dyadic `sumFin` to `Real` commutes with summation. -/
theorem dyadicToReal_sumFin {n : Nat} (f : Fin n → Dyadic) :
    (sumFin n f : Real) = ∑ i, (f i : Real) := by
  classical
  have hsum : sumFin n f = ∑ i, f i := sumFin_eq_sum_univ (f := f)
  have hcast : ((∑ i, f i : Dyadic) : Real) = ∑ i, (f i : Real) :=
    dyadicToReal_sum_univ (f := f)
  simpa [hsum] using hcast

/-- `sumFinCommonDen` agrees with `sumFin`. -/
theorem sumFinCommonDen_eq_sumFin (n : Nat) (f : Fin n → Dyadic) :
    sumFinCommonDen n f = sumFin n f := rfl

/-- Dot product over `Fin n` (Dyadic-valued). -/
def dotFin (n : Nat) (x y : Fin n → Dyadic) : Dyadic :=
  sumFin n (fun i => x i * y i)

/-- Unfolding lemma for `dotFin`. -/
theorem dotFin_def (n : Nat) (x y : Fin n → Dyadic) :
    dotFin n x y = sumFin n (fun i => x i * y i) := rfl

/-- `dotFin` matches `dotProduct`. -/
theorem dotFin_eq_dotProduct (n : Nat) (x y : Fin n → Dyadic) :
    dotFin n x y = dotProduct x y := by
  simp [dotFin_def, sumFin_eq_sum_univ, dotProduct]

end Linear

end Sound

end Nfp
