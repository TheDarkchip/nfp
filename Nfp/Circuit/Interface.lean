-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Semantics

/-!
Typed input/output interfaces for circuits.
-/

namespace Nfp

universe u v u_in u_out

namespace Circuit

variable {ι : Type u} [Fintype ι] [DecidableEq ι]
variable {α : Type v}

/-- A typed input/output interface for a circuit. -/
structure Interface (C : Circuit ι α) (ι_in : Type u_in) (ι_out : Type u_out) where
  /-- Input labels correspond exactly to the circuit's input nodes. -/
  inputs : ι_in ≃ { i // i ∈ C.inputs }
  /-- Output labels correspond exactly to the circuit's output nodes. -/
  outputs : ι_out ≃ { i // i ∈ C.outputs }

namespace Interface

variable {C : Circuit ι α} {ι_in : Type u_in} {ι_out : Type u_out}

/-- Convert a typed input assignment into an input-node assignment. -/
def toInputAssignment (I : Interface C ι_in ι_out) (input : ι_in → α) : C.InputAssignment :=
  fun i => input (I.inputs.symm i)

/-- Evaluate a circuit on a typed interface. -/
def eval (I : Interface C ι_in ι_out) (input : ι_in → α) : ι_out → α :=
  fun o => evalInput C (I.toInputAssignment input) (I.outputs o).1

/-- Unfolding equation for `Interface.eval`. -/
theorem eval_eq (I : Interface C ι_in ι_out) (input : ι_in → α) (o : ι_out) :
    I.eval input o = evalInput C (I.toInputAssignment input) (I.outputs o).1 :=
  rfl

end Interface

end Circuit

end Nfp
