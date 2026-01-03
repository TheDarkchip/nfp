-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Data.Vector.Defs
import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange
import Nfp.Circuit.Layers.Softmax
import Nfp.Model.InductionHead
import Nfp.Sound.Linear.FinFold

/-!
Sound builders for induction certificates.

These builders recompute certificate bounds inside Lean from exact inputs and
return proof-carrying results. The head-input path derives softmax tolerances
from score margins rather than trusting external weight dumps.
-/

namespace Nfp

namespace Sound

open scoped BigOperators

open Nfp.Circuit

variable {seq : Nat}

/-- Cached query projections for head inputs (opaque to avoid kernel reduction). -/
private opaque qVecVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector (Vector Rat dHead) seq :=
  Vector.ofFn (fun q : Fin seq =>
    Vector.ofFn (fun d : Fin dHead =>
      Linear.dotFin dModel (fun j => inputs.embed q j) (fun j => inputs.wq j d)))

/-- Cached key projections for head inputs (opaque to avoid kernel reduction). -/
private opaque kVecVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector (Vector Rat dHead) seq :=
  Vector.ofFn (fun q : Fin seq =>
    Vector.ofFn (fun d : Fin dHead =>
      Linear.dotFin dModel (fun j => inputs.embed q j) (fun j => inputs.wk j d)))

/-- Cached value projections for head inputs (opaque to avoid kernel reduction). -/
private opaque vVecVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector (Vector Rat dHead) seq :=
  Vector.ofFn (fun q : Fin seq =>
    Vector.ofFn (fun d : Fin dHead =>
      Linear.dotFin dModel (fun j => inputs.embed q j) (fun j => inputs.wv j d)))

/-- Cached attention scores for head inputs (opaque to avoid kernel reduction). -/
private opaque scoresVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (qVecVec kVecVec : Vector (Vector Rat dHead) seq) : Vector (Vector Rat seq) seq :=
  Vector.ofFn (fun q : Fin seq =>
    Vector.ofFn (fun k : Fin seq =>
      let qVec : Fin dHead → Rat := fun d => (qVecVec.get q).get d
      let kVec : Fin dHead → Rat := fun d => (kVecVec.get k).get d
      inputs.scale * (Linear.dotFin dHead (fun d => qVec d) (fun d => kVec d))))

/-- Cached direction head for head inputs (opaque to avoid kernel reduction). -/
private opaque dirHeadVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector Rat dHead :=
  Vector.ofFn (fun d : Fin dHead =>
    Linear.dotFin dModel (fun j => inputs.wo j d) (fun j => inputs.direction j))

/-- Cached value projections for head inputs (opaque to avoid kernel reduction). -/
private opaque valsVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vVecVec : Vector (Vector Rat dHead) seq) (dirHeadVec : Vector Rat dHead) :
    Vector Rat seq :=
  Vector.ofFn (fun k : Fin seq =>
    let vVec : Fin dHead → Rat := fun d => (vVecVec.get k).get d
    let dirHead : Fin dHead → Rat := fun d => dirHeadVec.get d
    Linear.dotFin dHead (fun d => vVec d) (fun d => dirHead d))

/-- Sound induction-certificate payload built from exact head inputs. -/
structure InductionHeadCert (seq : Nat) where
  /-- Weight tolerance. -/
  eps : Rat
  /-- Score margin used to justify the weight tolerance. -/
  margin : Rat
  /-- Active queries for which bounds are required. -/
  active : Finset (Fin seq)
  /-- `prev` selector for induction-style attention. -/
  prev : Fin seq → Fin seq
  /-- Score matrix entries. -/
  scores : Fin seq → Fin seq → Rat
  /-- Value-range certificate for the direction values. -/
  values : ValueRangeCert seq

/-- Soundness predicate for `InductionHeadCert`. -/
structure InductionHeadCertSound [NeZero seq] (c : InductionHeadCert seq) : Prop where
  /-- Softmax weights respect the derived margin bounds. -/
  softmax_bounds :
    Layers.SoftmaxMarginBoundsOn (Val := Real) (c.eps : Real) (c.margin : Real)
      (fun q => q ∈ c.active) c.prev
      (fun q k => (c.scores q k : Real))
      (fun q k => Circuit.softmax (fun j => (c.scores q j : Real)) k)
  /-- Value-range bounds hold for the certificate values. -/
  value_bounds :
    Layers.ValueRangeBounds (Val := Rat) c.values.lo c.values.hi c.values.vals

/-- Build and certify a softmax-margin certificate from exact scores/weights. -/
def buildSoftmaxMarginCert? [NeZero seq]
    (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (scores : Fin seq → Fin seq → Rat)
    (weights : Fin seq → Fin seq → Rat) :
    Option {c : SoftmaxMarginCert seq // checkSoftmaxMarginCert c = true} := by
  classical
  let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (prev q)
  let epsAt : Fin seq → Rat := fun q =>
    let other := otherKeys q
    let maxOther :=
      if h : other.Nonempty then
        other.sup' h (fun k => weights q k)
      else
        (0 : Rat)
    let deficit := (1 : Rat) - weights q (prev q)
    max maxOther deficit
  let marginAt : Fin seq → Rat := fun q =>
    let other := otherKeys q
    if h : other.Nonempty then
      other.inf' h (fun k => scores q (prev q) - scores q k)
    else
      (0 : Rat)
  let eps :=
    if h : active.Nonempty then
      active.sup' h epsAt
    else
      (0 : Rat)
  let margin :=
    if h : active.Nonempty then
      active.inf' h marginAt
    else
      (0 : Rat)
  let cert : SoftmaxMarginCert seq :=
    { eps := eps
      margin := margin
      active := active
      prev := prev
      scores := scores
      weights := weights }
  if h : checkSoftmaxMarginCert cert = true then
    exact some ⟨cert, h⟩
  else
    exact none

/-- Build and certify a value-range certificate from exact values. -/
def buildValueRangeCert? [NeZero seq]
    (vals : Fin seq → Rat)
    (direction : Option DirectionSpec) :
    Option {c : ValueRangeCert seq // checkValueRangeCert c = true} := by
  classical
  let _ : Nonempty (Fin seq) := by
    refine ⟨⟨0, ?_⟩⟩
    exact Nat.pos_of_ne_zero (NeZero.ne seq)
  let univ : Finset (Fin seq) := Finset.univ
  let hnonempty : univ.Nonempty := Finset.univ_nonempty
  let lo := univ.inf' hnonempty vals
  let hi := univ.sup' hnonempty vals
  let cert : ValueRangeCert seq :=
    { lo := lo
      hi := hi
      vals := vals
      direction := direction }
  if h : checkValueRangeCert cert = true then
    exact some ⟨cert, h⟩
  else
    exact none

/-- Build and certify induction certificates from exact head inputs. -/
def buildInductionCertFromHead? [NeZero seq] {dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option {c : InductionHeadCert seq // InductionHeadCertSound c} := by
  classical
  let qVecVec := qVecVecOfInputs inputs
  let kVecVec := kVecVecOfInputs inputs
  let vVecVec := vVecVecOfInputs inputs
  let scoresVec := scoresVecOfInputs inputs qVecVec kVecVec
  let scores : Fin seq → Fin seq → Rat := fun q k =>
    (scoresVec.get q).get k
  let otherKeys : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
  let marginAt : Fin seq → Rat := fun q =>
    let other := otherKeys q
    if h : other.Nonempty then
      other.inf' h (fun k => scores q (inputs.prev q) - scores q k)
    else
      (0 : Rat)
  let margin : Rat :=
    if h : inputs.active.Nonempty then
      inputs.active.inf' h marginAt
    else
      (0 : Rat)
  let eps : Rat :=
    if margin < 0 then
      (1 : Rat)
    else
      (seq - 1 : Rat) / (1 + margin)
  have hseq : (1 : Nat) ≤ seq :=
    Nat.succ_le_iff.mpr (Nat.pos_of_ne_zero (NeZero.ne seq))
  let dirHeadVec := dirHeadVecOfInputs inputs
  let valsVec := valsVecOfInputs inputs vVecVec dirHeadVec
  let vals : Fin seq → Rat := fun k => valsVec.get k
  exact
    match buildValueRangeCert? vals (some inputs.directionSpec) with
    | none => none
    | some ⟨valCert, hval⟩ =>
      let cert : InductionHeadCert seq :=
        { eps := eps
          margin := margin
          active := inputs.active
          prev := inputs.prev
          scores := scores
          values := valCert }
      have hvalues : Layers.ValueRangeBounds (Val := Rat) valCert.lo valCert.hi valCert.vals :=
        Circuit.checkValueRangeCert_sound valCert hval
      let scoresReal : Fin seq → Fin seq → Real := fun q k => (scores q k : Real)
      let weights : Fin seq → Fin seq → Real := fun q k =>
        Circuit.softmax (scoresReal q) k
      let others : Fin seq → Finset (Fin seq) := fun q =>
        (Finset.univ : Finset (Fin seq)).erase (inputs.prev q)
      have hscore_margin :
          ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
            scores q k + margin ≤ scores q (inputs.prev q) := by
        intro q hq k hk
        by_cases hactive : inputs.active.Nonempty
        · have hmargin_le : margin ≤ marginAt q := by
            have hle : margin ≤ inputs.active.inf' hactive marginAt := by
              simp [margin, hactive]
            have hle_all :=
              (Finset.le_inf'_iff (s := inputs.active) (H := hactive) (f := marginAt)
                (a := margin)).1 hle
            exact hle_all q hq
          have hother : (otherKeys q).Nonempty := ⟨k, by simp [otherKeys, hk]⟩
          have hgap_le :
              marginAt q ≤ scores q (inputs.prev q) - scores q k := by
            have hle : marginAt q ≤
                (otherKeys q).inf' hother
                  (fun k => scores q (inputs.prev q) - scores q k) := by
              simp [marginAt, hother]
            have hle_all :=
              (Finset.le_inf'_iff (s := otherKeys q) (H := hother)
                (f := fun k => scores q (inputs.prev q) - scores q k)
                (a := marginAt q)).1 hle
            exact hle_all k (by simp [otherKeys, hk])
          have hgap : margin ≤ scores q (inputs.prev q) - scores q k :=
            le_trans hmargin_le hgap_le
          have hgap' :=
            add_le_add_left hgap (scores q k)
          simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using hgap'
        · exact (hactive ⟨q, hq⟩).elim
      have hscore_margin_real :
          ∀ q, q ∈ inputs.active → ∀ k, k ≠ inputs.prev q →
            scoresReal q k + (margin : Real) ≤ scoresReal q (inputs.prev q) := by
        intro q hq k hk
        have hrat := hscore_margin q hq k hk
        have hreal :
            ((scores q k + margin : Rat) : Real) ≤ scores q (inputs.prev q) := by
          exact (Rat.cast_le (K := Real)).2 hrat
        simpa [scoresReal, Rat.cast_add] using hreal
      have hsoftmax_bounds :
          Layers.SoftmaxMarginBoundsOn (Val := Real) (eps : Real) (margin : Real)
            (fun q => q ∈ inputs.active) inputs.prev scoresReal weights := by
        classical
        refine
          { score_margin := ?_
            nonneg := ?_
            sum_one := ?_
            prev_large := ?_
            other_le := ?_ }
        · intro q hq k hk
          exact hscore_margin_real q hq k hk
        · intro q _ k
          simpa [weights] using
            (Circuit.softmax_nonneg (scores := scoresReal q) k)
        · intro q _
          simpa [weights] using
            (Circuit.softmax_sum_one (scores := scoresReal q))
        · intro q hq
          have hsum_others_le : (∑ k ∈ others q, weights q k) ≤ (eps : Real) := by
            by_cases hneg : margin < 0
            · have heps : (eps : Real) = 1 := by
                simp [eps, hneg]
              have hsubset : others q ⊆ (Finset.univ : Finset (Fin seq)) := by
                intro k hk
                simp
              have hnonneg :
                  ∀ k ∈ (Finset.univ : Finset (Fin seq)), 0 ≤ weights q k := by
                intro k _
                simpa [weights] using
                  (Circuit.softmax_nonneg (scores := scoresReal q) k)
              have hsum_le :
                  (∑ k ∈ others q, weights q k) ≤
                    ∑ k ∈ (Finset.univ : Finset (Fin seq)), weights q k :=
                Finset.sum_le_sum_of_subset_of_nonneg hsubset (by
                  intro k hk _; exact hnonneg k hk)
              have hsum_one : (∑ k, weights q k) = 1 := by
                simpa [weights] using
                  (Circuit.softmax_sum_one (scores := scoresReal q))
              have hsum_le' : (∑ k ∈ others q, weights q k) ≤ 1 := by
                simpa [hsum_one] using hsum_le
              simpa [heps] using hsum_le'
            · have hnonneg : 0 ≤ margin := le_of_not_gt hneg
              have hnonneg_real : 0 ≤ (margin : Real) := by
                exact (Rat.cast_nonneg (K := Real)).2 hnonneg
              have hbound :
                  ∀ k ∈ others q,
                    weights q k ≤ (1 + (margin : Real))⁻¹ := by
                intro k hk
                have hkne : k ≠ inputs.prev q := (Finset.mem_erase.mp hk).1
                have hscore := hscore_margin_real q hq k hkne
                simpa [weights] using
                  (Circuit.softmax_other_le_inv_one_add (scores := scoresReal q)
                    (prev := inputs.prev q) (k := k) (m := (margin : Real))
                    hnonneg_real hscore)
              have hsum_le :
                  (∑ k ∈ others q, weights q k) ≤
                    ∑ k ∈ others q, (1 + (margin : Real))⁻¹ :=
                Finset.sum_le_sum hbound
              have hsum_const :
                  (∑ k ∈ others q, (1 + (margin : Real))⁻¹) =
                    (others q).card * (1 + (margin : Real))⁻¹ := by
                simp
              have hcard : (others q).card = seq - 1 := by
                simp [others, Finset.card_erase_of_mem]
              have hsum_le' :
                  (∑ k ∈ others q, weights q k) ≤
                    (seq - 1 : Real) * (1 + (margin : Real))⁻¹ := by
                have hsum_le'' := hsum_le.trans_eq hsum_const
                have hsum_le''' := hsum_le''
                simp only [hcard, Nat.cast_sub hseq, Nat.cast_one] at hsum_le'''
                exact hsum_le'''
              have heps :
                  (eps : Real) = (seq - 1 : Real) * (1 + (margin : Real))⁻¹ := by
                simp [eps, hneg, Rat.cast_add, div_eq_mul_inv]
              simpa [heps] using hsum_le'
          have hsum_eq :
              weights q (inputs.prev q) + ∑ k ∈ others q, weights q k = 1 := by
            have hsum' :
                weights q (inputs.prev q) + ∑ k ∈ others q, weights q k =
                  ∑ k, weights q k := by
              simp [others]
            have hsum_one : (∑ k, weights q k) = 1 := by
              simpa [weights] using
                (Circuit.softmax_sum_one (scores := scoresReal q))
            calc
              weights q (inputs.prev q) + ∑ k ∈ others q, weights q k =
                  ∑ k, weights q k := hsum'
              _ = 1 := hsum_one
          have hsum_le' :
              weights q (inputs.prev q) + ∑ k ∈ others q, weights q k ≤
                weights q (inputs.prev q) + (eps : Real) := by
            have hsum_le'' := add_le_add_left hsum_others_le (weights q (inputs.prev q))
            simpa [add_comm, add_left_comm, add_assoc] using hsum_le''
          have hprev :
              1 ≤ weights q (inputs.prev q) + (eps : Real) := by
            simpa [hsum_eq] using hsum_le'
          exact hprev
        · intro q hq k hk
          have hsum_others_le : (∑ j ∈ others q, weights q j) ≤ (eps : Real) := by
            by_cases hneg : margin < 0
            · have heps : (eps : Real) = 1 := by
                simp [eps, hneg]
              have hsubset : others q ⊆ (Finset.univ : Finset (Fin seq)) := by
                intro j hj
                simp
              have hnonneg :
                  ∀ j ∈ (Finset.univ : Finset (Fin seq)), 0 ≤ weights q j := by
                intro j _
                simpa [weights] using
                  (Circuit.softmax_nonneg (scores := scoresReal q) j)
              have hsum_le :
                  (∑ j ∈ others q, weights q j) ≤
                    ∑ j ∈ (Finset.univ : Finset (Fin seq)), weights q j :=
                Finset.sum_le_sum_of_subset_of_nonneg hsubset (by
                  intro j hj _; exact hnonneg j hj)
              have hsum_one : (∑ j, weights q j) = 1 := by
                simpa [weights] using
                  (Circuit.softmax_sum_one (scores := scoresReal q))
              have hsum_le' : (∑ j ∈ others q, weights q j) ≤ 1 := by
                simpa [hsum_one] using hsum_le
              simpa [heps] using hsum_le'
            · have hnonneg : 0 ≤ margin := le_of_not_gt hneg
              have hnonneg_real : 0 ≤ (margin : Real) := by
                exact (Rat.cast_nonneg (K := Real)).2 hnonneg
              have hbound :
                  ∀ j ∈ others q,
                    weights q j ≤ (1 + (margin : Real))⁻¹ := by
                intro j hj
                have hjne : j ≠ inputs.prev q := (Finset.mem_erase.mp hj).1
                have hscore := hscore_margin_real q hq j hjne
                simpa [weights] using
                  (Circuit.softmax_other_le_inv_one_add (scores := scoresReal q)
                    (prev := inputs.prev q) (k := j) (m := (margin : Real))
                    hnonneg_real hscore)
              have hsum_le :
                  (∑ j ∈ others q, weights q j) ≤
                    ∑ j ∈ others q, (1 + (margin : Real))⁻¹ :=
                Finset.sum_le_sum hbound
              have hsum_const :
                  (∑ j ∈ others q, (1 + (margin : Real))⁻¹) =
                    (others q).card * (1 + (margin : Real))⁻¹ := by
                simp
              have hcard : (others q).card = seq - 1 := by
                simp [others, Finset.card_erase_of_mem]
              have hsum_le' :
                  (∑ j ∈ others q, weights q j) ≤
                    (seq - 1 : Real) * (1 + (margin : Real))⁻¹ := by
                have hsum_le'' := hsum_le.trans_eq hsum_const
                have hsum_le''' := hsum_le''
                simp only [hcard, Nat.cast_sub hseq, Nat.cast_one] at hsum_le'''
                exact hsum_le'''
              have heps :
                  (eps : Real) = (seq - 1 : Real) * (1 + (margin : Real))⁻¹ := by
                simp [eps, hneg, Rat.cast_add, div_eq_mul_inv]
              simpa [heps] using hsum_le'
          have hk' : k ∈ others q := by
            simp [others, hk]
          have hnonneg :
              ∀ j ∈ others q, 0 ≤ weights q j := by
            intro j _
            simpa [weights] using
              (Circuit.softmax_nonneg (scores := scoresReal q) j)
          have hle :
              weights q k ≤ ∑ j ∈ others q, weights q j := by
            have h := Finset.single_le_sum hnonneg hk'
            simpa using h
          exact hle.trans hsum_others_le
      some ⟨cert, { softmax_bounds := hsoftmax_bounds, value_bounds := hvalues }⟩

end Sound

end Nfp
