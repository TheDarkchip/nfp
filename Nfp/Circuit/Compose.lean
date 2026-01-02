-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Disjoint
import Mathlib.Data.Fintype.Sum
import Mathlib.Data.Sum.Order
import Mathlib.Logic.Embedding.Basic
import Nfp.Circuit.Typed

/-!
Combinators for composing typed circuits (sequential and residual wiring).
-/

namespace Nfp

universe u v u' u_in u_mid u_out

namespace Circuit

open Function

section SumEquiv

variable {Left : Type u} {Right : Type u'}

/-- Embed a finset subtype into the left injection of a sum. -/
def inlSubtypeEquiv (s : Finset Left) :
    { i // i ∈ s } ≃ { i // i ∈ s.map (Embedding.inl : Left ↪ Left ⊕ Right) } :=
  { toFun := fun i =>
      ⟨Sum.inl i.1, by
        refine Finset.mem_map.2 ?_
        exact ⟨i.1, i.2, rfl⟩⟩
    invFun := fun i =>
      match i with
      | ⟨Sum.inl a, ha⟩ =>
          let h' : a ∈ s := by
            rcases (Finset.mem_map.1 ha) with ⟨a', ha', h⟩
            cases h
            exact ha'
          ⟨a, h'⟩
      | ⟨Sum.inr b, hb⟩ =>
          False.elim <| by
            rcases (Finset.mem_map.1 hb) with ⟨a, _ha, h⟩
            cases h
    left_inv := by
      intro i
      rfl
    right_inv := by
      intro i
      cases i with
      | mk s hs =>
          cases s with
          | inl a =>
              rfl
          | inr b =>
              have : False := by
                rcases (Finset.mem_map.1 hs) with ⟨a, _ha, h⟩
                cases h
              cases this }

/-- Embed a finset subtype into the right injection of a sum. -/
def inrSubtypeEquiv (s : Finset Right) :
    { i // i ∈ s } ≃ { i // i ∈ s.map (Embedding.inr : Right ↪ Left ⊕ Right) } :=
  { toFun := fun i =>
      ⟨Sum.inr i.1, by
        refine Finset.mem_map.2 ?_
        exact ⟨i.1, i.2, rfl⟩⟩
    invFun := fun i =>
      match i with
      | ⟨Sum.inr a, ha⟩ =>
          let h' : a ∈ s := by
            rcases (Finset.mem_map.1 ha) with ⟨a', ha', h⟩
            cases h
            exact ha'
          ⟨a, h'⟩
      | ⟨Sum.inl b, hb⟩ =>
          False.elim <| by
            rcases (Finset.mem_map.1 hb) with ⟨a, _ha, h⟩
            cases h
    left_inv := by
      intro i
      rfl
    right_inv := by
      intro i
      cases i with
      | mk s hs =>
          cases s with
          | inr a =>
              rfl
          | inl b =>
              have : False := by
                rcases (Finset.mem_map.1 hs) with ⟨a, _ha, h⟩
                cases h
              cases this }

end SumEquiv

section Seq

variable {Node₁ : Type u} [Fintype Node₁]
variable {Node₂ : Type u'} [Fintype Node₂]
variable {Val : Type v}
variable {Input : Type u_in} {Mid : Type u_mid} {Output : Type u_out}
variable {C1 : Circuit Node₁ Val} {C2 : Circuit Node₂ Val}
variable {I1 : Interface C1 Input Mid} {I2 : Interface C2 Mid Output}

/-- Bridge edges from the outputs of `C1` to the inputs of `C2`. -/
def seqBridge (j : Node₁) (i : Node₂) : Prop :=
  ∃ h : i ∈ C2.inputs,
    j = (I1.outputs (I2.inputs.symm ⟨i, h⟩)).1

/-- Fixing an input witness reduces `seqBridge` to an equality. -/
theorem seqBridge_iff_eq {j : Node₁} {i : Node₂} (hmem : i ∈ C2.inputs) :
    seqBridge (C1 := C1) (C2 := C2) (I1 := I1) (I2 := I2) j i ↔
      j = (I1.outputs (I2.inputs.symm ⟨i, hmem⟩)).1 := by
  constructor
  · rintro ⟨h, hEq⟩
    have hSubtype :
        (⟨i, h⟩ : { i // i ∈ C2.inputs }) = ⟨i, hmem⟩ := by
      apply Subtype.ext
      rfl
    have hMid :
        I2.inputs.symm ⟨i, h⟩ = I2.inputs.symm ⟨i, hmem⟩ := by
      exact congrArg I2.inputs.symm hSubtype
    have hOut :
        (I1.outputs (I2.inputs.symm ⟨i, h⟩)).1 =
          (I1.outputs (I2.inputs.symm ⟨i, hmem⟩)).1 := by
      exact congrArg Subtype.val (congrArg I1.outputs hMid)
    exact hEq.trans hOut
  · intro hEq
    exact ⟨hmem, hEq⟩

/-- Adjacency for sequentially composed circuits. -/
def seqAdj : Node₁ ⊕ Node₂ → Node₁ ⊕ Node₂ → Prop
  | Sum.inl j, Sum.inl i => C1.dag.rel j i
  | Sum.inl j, Sum.inr i =>
      seqBridge (C1 := C1) (C2 := C2) (I1 := I1) (I2 := I2) j i
  | Sum.inr j, Sum.inr i => C2.dag.rel j i
  | _, _ => False

variable [DecidableEq Node₁] [DecidableEq Node₂]

/-- DAG for sequentially composed circuits. -/
def seqDag : Dag (Node₁ ⊕ Node₂) :=
  { graph := { Adj := seqAdj (C1 := C1) (C2 := C2) (I1 := I1) (I2 := I2) }
    decAdj := by
      intro j i
      cases j with
      | inl j =>
          cases i with
          | inl i =>
              exact (inferInstance : Decidable (C1.dag.rel j i))
          | inr i =>
              by_cases hmem : i ∈ C2.inputs
              · by_cases hEq :
                    j = (I1.outputs (I2.inputs.symm ⟨i, hmem⟩)).1
                · exact isTrue ⟨hmem, hEq⟩
                · exact isFalse (by
                    intro h
                    have hEq' :
                        j = (I1.outputs (I2.inputs.symm ⟨i, hmem⟩)).1 :=
                      (seqBridge_iff_eq (C1 := C1) (C2 := C2) (I1 := I1) (I2 := I2)
                        (j := j) (i := i) hmem).1 h
                    exact hEq hEq')
              · exact isFalse (by
                  intro h
                  exact hmem h.1)
      | inr j =>
          cases i with
          | inl _ =>
              exact isFalse (by intro h; cases h)
          | inr i =>
              exact (inferInstance : Decidable (C2.dag.rel j i))
    wf := by
      have hsub :
          Subrelation
            (seqAdj (C1 := C1) (C2 := C2) (I1 := I1) (I2 := I2))
            (Sum.Lex C1.dag.rel C2.dag.rel) := by
        intro j i h
        cases j with
        | inl j =>
            cases i with
            | inl i =>
                exact Sum.Lex.inl h
            | inr i =>
                exact Sum.Lex.sep _ _
        | inr j =>
            cases i with
            | inl _ =>
                exact False.elim h
            | inr i =>
                exact Sum.Lex.inr h
      have hwf : WellFounded (Sum.Lex C1.dag.rel C2.dag.rel) :=
        Sum.lex_wf C1.dag.wf C2.dag.wf
      exact Subrelation.wf hsub hwf }

/-- Sequential composition of circuits at the node level. -/
def seqCircuit : Circuit (Node₁ ⊕ Node₂) Val :=
  { dag := seqDag (C1 := C1) (C2 := C2) (I1 := I1) (I2 := I2)
    inputs := C1.inputs.map (Embedding.inl : Node₁ ↪ Node₁ ⊕ Node₂)
    outputs := C2.outputs.map (Embedding.inr : Node₂ ↪ Node₁ ⊕ Node₂)
    gate := by
      intro i rec
      cases i with
      | inl i =>
          exact C1.gate i (fun j h =>
            rec (Sum.inl j) (by
              change C1.dag.rel j i
              exact h))
      | inr i =>
          by_cases hinput : i ∈ C2.inputs
          · let mid : Mid := I2.inputs.symm ⟨i, hinput⟩
            let out : Node₁ := (I1.outputs mid).1
            exact rec (Sum.inl out) (by
              refine ⟨hinput, rfl⟩)
          · exact C2.gate i (fun j h =>
              rec (Sum.inr j) (by
                change C2.dag.rel j i
                exact h)) }

/-- Interface for sequentially composed circuits. -/
def seqInterface :
    Interface
      (seqCircuit (C1 := C1) (C2 := C2) (I1 := I1) (I2 := I2)) Input Output :=
  { inputs :=
      I1.inputs.trans (inlSubtypeEquiv (s := C1.inputs))
    outputs :=
      I2.outputs.trans (inrSubtypeEquiv (s := C2.outputs)) }

end Seq

section Residual

variable {Node : Type u}
variable {Input : Type u_in} [Fintype Input]

/-- Output nodes for residual wiring. -/
def residualOutputs : Finset (Node ⊕ Input) :=
  (Finset.univ : Finset Input).map (Embedding.inr : Input ↪ Node ⊕ Input)

/-- Output equivalence for residual wiring. -/
def residualOutputEquiv :
    Input ≃ { i // i ∈ residualOutputs (Node := Node) (Input := Input) } :=
  { toFun := fun o =>
      ⟨Sum.inr o, by
        refine Finset.mem_map.2 ?_
        exact ⟨o, by simp, rfl⟩⟩
    invFun := fun i =>
      match i with
      | ⟨Sum.inr o, _⟩ => o
      | ⟨Sum.inl _, h⟩ =>
          False.elim <| by
            rcases (Finset.mem_map.1 h) with ⟨o, _ho, ho⟩
            cases ho
    left_inv := by
      intro o
      rfl
    right_inv := by
      intro i
      cases i with
      | mk s hs =>
          cases s with
          | inr o =>
              rfl
          | inl s =>
              have : False := by
                rcases (Finset.mem_map.1 hs) with ⟨o, _ho, ho⟩
                cases ho
              cases this }

variable [Fintype Node]
variable {Val : Type v} [Add Val]
variable {C : Circuit Node Val}
variable {I : Interface C Input Input}

/-- Adjacency for residual wiring on a typed circuit. -/
def residualAdj : Node ⊕ Input → Node ⊕ Input → Prop
  | Sum.inl j, Sum.inl i => C.dag.rel j i
  | Sum.inl j, Sum.inr o => j = (I.inputs o).1 ∨ j = (I.outputs o).1
  | _, _ => False

variable [DecidableEq Node]

/-- DAG for residual wiring on a typed circuit. -/
def residualDag : Dag (Node ⊕ Input) :=
  { graph := { Adj := residualAdj (C := C) (I := I) }
    decAdj := by
      intro j i
      cases j with
      | inl j =>
          cases i with
          | inl i =>
              exact (inferInstance : Decidable (C.dag.rel j i))
          | inr o =>
              exact (inferInstance :
                Decidable (j = (I.inputs o).1 ∨ j = (I.outputs o).1))
      | inr _ =>
          cases i with
          | inl _ =>
              exact isFalse (by intro h; cases h)
          | inr _ =>
              exact isFalse (by intro h; cases h)
    wf := by
      have hsub :
          Subrelation (residualAdj (C := C) (I := I))
            (Sum.Lex C.dag.rel (fun _ _ : Input => False)) := by
        intro j i h
        cases j with
        | inl j =>
            cases i with
            | inl i =>
                exact Sum.Lex.inl h
            | inr _ =>
                exact Sum.Lex.sep _ _
        | inr _ =>
            cases i with
            | inl _ =>
                exact False.elim h
            | inr _ =>
                exact False.elim h
      have hfalse : WellFounded (fun _ _ : Input => False) := by
        refine ⟨?_⟩
        intro a
        refine Acc.intro a ?_
        intro b h
        cases h
      have hwf : WellFounded (Sum.Lex C.dag.rel (fun _ _ : Input => False)) :=
        Sum.lex_wf C.dag.wf hfalse
      exact Subrelation.wf hsub hwf }

/-- Circuit that adds a residual connection to a typed circuit. -/
def residualCircuit : Circuit (Node ⊕ Input) Val :=
  { dag := residualDag (C := C) (I := I)
    inputs := C.inputs.map (Embedding.inl : Node ↪ Node ⊕ Input)
    outputs := residualOutputs (Node := Node) (Input := Input)
    gate := by
      intro i rec
      cases i with
      | inl i =>
          exact C.gate i (fun j h =>
            rec (Sum.inl j) (by simpa [residualAdj] using h))
      | inr o =>
          let inNode : Node := (I.inputs o).1
          let outNode : Node := (I.outputs o).1
          let inVal := rec (Sum.inl inNode) (by
            change inNode = (I.inputs o).1 ∨ inNode = (I.outputs o).1
            exact Or.inl rfl)
          let outVal := rec (Sum.inl outNode) (by
            change outNode = (I.inputs o).1 ∨ outNode = (I.outputs o).1
            exact Or.inr rfl)
          exact inVal + outVal }

/-- Interface for residual wiring. -/
def residualInterface :
    Interface (residualCircuit (C := C) (I := I)) Input Input :=
  { inputs := I.inputs.trans (inlSubtypeEquiv (s := C.inputs))
    outputs := residualOutputEquiv (Node := Node) (Input := Input) }

end Residual

namespace TypedCircuit

variable {Node₁ : Type u} [Fintype Node₁] [DecidableEq Node₁]
variable {Node₂ : Type u'} [Fintype Node₂] [DecidableEq Node₂]
variable {Val : Type v}
variable {Input : Type u_in} {Mid : Type u_mid} {Output : Type u_out}

/-- Sequential composition of typed circuits. -/
def seq (T₁ : TypedCircuit Node₁ Val Input Mid)
    (T₂ : TypedCircuit Node₂ Val Mid Output) :
    TypedCircuit (Node₁ ⊕ Node₂) Val Input Output :=
  { circuit := seqCircuit (C1 := T₁.circuit) (C2 := T₂.circuit)
      (I1 := T₁.interface) (I2 := T₂.interface)
    interface := seqInterface (C1 := T₁.circuit) (C2 := T₂.circuit)
      (I1 := T₁.interface) (I2 := T₂.interface) }

variable [Add Val]
variable [Fintype Input]
variable [DecidableEq Input]

/-- Add a residual connection to a typed circuit. -/
def residual (T : TypedCircuit Node₁ Val Input Input) :
    TypedCircuit (Node₁ ⊕ Input) Val Input Input :=
  { circuit := residualCircuit (C := T.circuit) (I := T.interface)
    interface := residualInterface (C := T.circuit) (I := T.interface) }

end TypedCircuit

end Circuit

end Nfp
