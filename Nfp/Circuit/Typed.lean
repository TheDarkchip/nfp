-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Combinators
import Nfp.Circuit.Cert

/-!
Typed circuit wrappers and typed equivalence checking.
-/

namespace Nfp

universe u v u' u_in u_out

namespace Circuit

/-- A circuit bundled with a typed input/output interface. -/
structure TypedCircuit (ι : Type u) [Fintype ι] [DecidableEq ι] (α : Type v)
    (ι_in : Type u_in) (ι_out : Type u_out) where
  /-- The underlying circuit. -/
  circuit : Circuit ι α
  /-- Typed input/output interface for `circuit`. -/
  interface : Interface circuit ι_in ι_out

namespace TypedCircuit

variable {ι : Type u} [Fintype ι] [DecidableEq ι]
variable {α : Type v} {ι_in : Type u_in} {ι_out : Type u_out}

/-- Evaluate a typed circuit on a typed input. -/
def eval (T : TypedCircuit ι α ι_in ι_out) (input : ι_in → α) : ι_out → α :=
  T.interface.eval input

/-- Decide equivalence by enumerating typed inputs. -/
def checkEquiv (T₁ T₂ : TypedCircuit ι α ι_in ι_out)
    [Fintype ι_in] [DecidableEq ι_in] [Fintype ι_out]
    [Fintype α] [DecidableEq α] : Bool :=
  Circuit.checkEquivOnInterface T₁.circuit T₂.circuit T₁.interface T₂.interface

/-- `checkEquiv` is sound and complete for `EquivOnInterface`. -/
theorem checkEquiv_eq_true_iff (T₁ T₂ : TypedCircuit ι α ι_in ι_out)
    [Fintype ι_in] [DecidableEq ι_in] [Fintype ι_out]
    [Fintype α] [DecidableEq α] :
    checkEquiv T₁ T₂ = true ↔
      EquivOnInterface T₁.circuit T₂.circuit T₁.interface T₂.interface := by
  simpa [checkEquiv] using
    (checkEquivOnInterface_eq_true_iff T₁.circuit T₂.circuit T₁.interface T₂.interface)

variable {ι' : Type u'} [Fintype ι'] [DecidableEq ι']

/-- Relabel the nodes of a typed circuit. -/
def relabel (T : TypedCircuit ι α ι_in ι_out) (e : ι ≃ ι') :
    TypedCircuit ι' α ι_in ι_out :=
  { circuit := T.circuit.relabel e
    interface := T.interface.relabel e }

end TypedCircuit

end Circuit

end Nfp
