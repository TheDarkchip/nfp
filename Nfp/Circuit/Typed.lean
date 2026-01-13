-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Combinators
public import Nfp.Circuit.Cert.Basic

/-!
Typed circuit wrappers and typed equivalence checking.
-/

public section

namespace Nfp

universe u v u' u_in u_out

namespace Circuit

/-- A circuit bundled with a typed input/output interface. -/
structure TypedCircuit (Node : Type u) [Fintype Node] [DecidableEq Node] (Val : Type v)
    (Input : Type u_in) (Output : Type u_out) where
  /-- The underlying circuit. -/
  circuit : Circuit Node Val
  /-- Typed input/output interface for `circuit`. -/
  interface : Interface circuit Input Output

namespace TypedCircuit

variable {Node : Type u} [Fintype Node] [DecidableEq Node]
variable {Val : Type v} {Input : Type u_in} {Output : Type u_out}

/-- Evaluate a typed circuit on a typed input. -/
@[expose] def eval (T : TypedCircuit Node Val Input Output) (input : Input → Val) : Output → Val :=
  T.interface.eval input

/-- Decide equivalence by enumerating typed inputs. -/
def checkEquiv (T1 T2 : TypedCircuit Node Val Input Output)
    [Fintype Input] [DecidableEq Input] [Fintype Output]
    [Fintype Val] [DecidableEq Val] : Bool :=
  Circuit.checkEquivOnInterface T1.circuit T2.circuit T1.interface T2.interface

/-- `checkEquiv` is sound and complete for `EquivOnInterface`. -/
theorem checkEquiv_eq_true_iff (T1 T2 : TypedCircuit Node Val Input Output)
    [Fintype Input] [DecidableEq Input] [Fintype Output]
    [Fintype Val] [DecidableEq Val] :
    checkEquiv T1 T2 = true ↔
      EquivOnInterface T1.circuit T2.circuit T1.interface T2.interface := by
  simpa [checkEquiv] using
    (checkEquivOnInterface_eq_true_iff T1.circuit T2.circuit T1.interface T2.interface)

variable {Node' : Type u'} [Fintype Node'] [DecidableEq Node']

/-- Relabel the nodes of a typed circuit. -/
def relabel (T : TypedCircuit Node Val Input Output) (e : _root_.Equiv Node Node') :
    TypedCircuit Node' Val Input Output :=
  { circuit := T.circuit.relabel e
    interface := T.interface.relabel e }

end TypedCircuit

end Circuit

end Nfp
