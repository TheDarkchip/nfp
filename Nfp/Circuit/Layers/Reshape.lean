-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Logic.Equiv.Prod
public import Nfp.Circuit.Typed

/-!
Reshape combinators for product-typed circuit interfaces.
-/

public section

namespace Nfp

namespace Circuit

namespace Layers

universe u v u_in u_out u_in' u_out'

variable {Node : Type u} [Fintype Node] [DecidableEq Node]
variable {Val : Type v}
variable {α β γ : Type u_in}
variable {δ ε ζ : Type u_out}
variable {Input : Type u_in} {Output : Type u_out}
variable {Input' : Type u_in'} {Output' : Type u_out'}

/-- Reassociate the input/output product structure of a typed circuit. -/
def reassoc3
    (T : TypedCircuit Node Val ((α × β) × γ) ((δ × ε) × ζ)) :
    TypedCircuit Node Val (α × β × γ) (δ × ε × ζ) :=
  { circuit := T.circuit
    interface :=
      { inputs := (_root_.Equiv.prodAssoc α β γ).symm.trans T.interface.inputs
        outputs := (_root_.Equiv.prodAssoc δ ε ζ).symm.trans T.interface.outputs } }

/-- Swap the two factors of the input/output product structure. -/
def swap12
    (T : TypedCircuit Node Val (α × β) (δ × ε)) :
    TypedCircuit Node Val (β × α) (ε × δ) :=
  { circuit := T.circuit
    interface :=
      { inputs := (_root_.Equiv.prodComm α β).symm.trans T.interface.inputs
        outputs := (_root_.Equiv.prodComm δ ε).symm.trans T.interface.outputs } }

/-- Apply equivalences to the input/output labels of a typed circuit. -/
def mapInterface
    (T : TypedCircuit Node Val Input Output)
    (eIn : _root_.Equiv Input Input') (eOut : _root_.Equiv Output Output') :
    TypedCircuit Node Val Input' Output' :=
  { circuit := T.circuit
    interface :=
      { inputs := eIn.symm.trans T.interface.inputs
        outputs := eOut.symm.trans T.interface.outputs } }

end Layers

end Circuit

end Nfp
