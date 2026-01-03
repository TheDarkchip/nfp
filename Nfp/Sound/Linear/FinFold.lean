-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Rat
import Batteries.Data.Fin.Fold

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

/-- Tail-recursive sum over `Fin n` (Rat-valued). -/
def sumFin (n : Nat) (f : Fin n → Rat) : Rat :=
  foldlFin n (fun acc i => acc + f i) 0

/-- `sumFin` as a left fold over the finite range list. -/
theorem sumFin_eq_list_foldl (n : Nat) (f : Fin n → Rat) :
    sumFin n f = (List.finRange n).foldl (fun acc i => acc + f i) 0 := by
  simpa [sumFin, foldlFin_eq_foldl] using
    (Fin.foldl_eq_foldl_finRange (f := fun acc i => acc + f i) (x := (0 : Rat)) (n := n))

/-- Dot product over `Fin n` (Rat-valued). -/
def dotFin (n : Nat) (x y : Fin n → Rat) : Rat :=
  sumFin n (fun i => x i * y i)

/-- Unfolding lemma for `dotFin`. -/
theorem dotFin_def (n : Nat) (x y : Fin n → Rat) :
    dotFin n x y = sumFin n (fun i => x i * y i) := rfl

end Linear

end Sound

end Nfp
