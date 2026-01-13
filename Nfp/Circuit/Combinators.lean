-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Image
public import Mathlib.Logic.Equiv.Basic
public import Nfp.Circuit.Interface

/-!
Circuit combinators such as relabeling.
-/

@[expose] public section

namespace Nfp

universe u v u' u_in u_out

namespace Circuit

variable {Node : Type u} [Fintype Node]
variable {Node' : Type u'} [Fintype Node']
variable {Val : Type v}

/-- Relabel the nodes of a circuit along an equivalence. -/
def relabel (C : Circuit Node Val) (e : _root_.Equiv Node Node') : Circuit Node' Val := by
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

variable {Node : Type u} [Fintype Node] [DecidableEq Node]
variable {Node' : Type u'} [Fintype Node'] [DecidableEq Node']
variable {Val : Type v}
variable {Input : Type u_in} {Output : Type u_out}
variable {C : Circuit Node Val}

/-- Relabel a circuit interface along an equivalence of nodes. -/
def relabel (I : Interface C Input Output) (e : _root_.Equiv Node Node') :
    Interface (C.relabel e) Input Output := by
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
