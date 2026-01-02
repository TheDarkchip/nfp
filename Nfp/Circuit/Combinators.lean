-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Image
import Mathlib.Logic.Equiv.Basic
import Nfp.Circuit.Interface

/-!
Circuit combinators such as relabeling.
-/

namespace Nfp

universe u v u' u_in u_out

namespace Circuit

variable {ι : Type u} [Fintype ι]
variable {ι' : Type u'} [Fintype ι']
variable {α : Type v}

/-- Relabel the nodes of a circuit along an equivalence. -/
def relabel (C : Circuit ι α) (e : ι ≃ ι') : Circuit ι' α := by
  refine
    { dag := C.dag.relabel e
      inputs := C.inputs.map e.toEmbedding
      outputs := C.outputs.map e.toEmbedding
      gate := ?_ }
  intro i rec
  refine C.gate (e.symm i) ?_
  intro j h
  refine rec (e j) ?_
  change C.dag.rel (e.symm (e j)) (e.symm i)
  simpa using h

namespace Interface

variable {ι : Type u} [Fintype ι] [DecidableEq ι]
variable {ι' : Type u'} [Fintype ι'] [DecidableEq ι']
variable {α : Type v}
variable {ι_in : Type u_in} {ι_out : Type u_out}
variable {C : Circuit ι α}

/-- Relabel a circuit interface along an equivalence of nodes. -/
def relabel (I : Interface C ι_in ι_out) (e : ι ≃ ι') :
    Interface (C.relabel e) ι_in ι_out := by
  refine { inputs := ?_, outputs := ?_ }
  · refine I.inputs.trans ?_
    refine (e.subtypeEquiv ?_)
    intro a
    simp [Circuit.relabel]
  · refine I.outputs.trans ?_
    refine (e.subtypeEquiv ?_)
    intro a
    simp [Circuit.relabel]

end Interface

end Circuit

end Nfp
