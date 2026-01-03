-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Data.Vector.Defs
import Nfp.Circuit.Cert.ResidualInterval
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
      Linear.dotFin dModel (fun j => inputs.embed q j) (fun j => inputs.wq j d) +
        inputs.bq d))

/-- Cached key projections for head inputs (opaque to avoid kernel reduction). -/
private opaque kVecVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector (Vector Rat dHead) seq :=
  Vector.ofFn (fun q : Fin seq =>
    Vector.ofFn (fun d : Fin dHead =>
      Linear.dotFin dModel (fun j => inputs.embed q j) (fun j => inputs.wk j d) +
        inputs.bk d))

/-- Cached value projections for head inputs (opaque to avoid kernel reduction). -/
private opaque vVecVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead) : Vector (Vector Rat dHead) seq :=
  Vector.ofFn (fun q : Fin seq =>
    Vector.ofFn (fun d : Fin dHead =>
      Linear.dotFin dModel (fun j => inputs.embed q j) (fun j => inputs.wv j d) +
        inputs.bv d))

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

/-- Cached per-key head outputs in model space (opaque to avoid kernel reduction). -/
private opaque headValueVecOfInputs {seq dModel dHead : Nat}
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (vVecVec : Vector (Vector Rat dHead) seq) : Vector (Vector Rat dModel) seq :=
  Vector.ofFn (fun k : Fin seq =>
    Vector.ofFn (fun i : Fin dModel =>
      let vVec : Fin dHead → Rat := fun d => (vVecVec.get k).get d
      Linear.dotFin dHead (fun d => vVec d) (fun d => inputs.wo i d)))

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

section HeadOutputInterval

variable {seq dModel dHead : Nat}

noncomputable section

/-- Real-valued head output using explicit score inputs. -/
def headOutputWithScores (scores : Fin seq → Fin seq → Rat)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (fun j => (scores q j : Real)) k
  let vVecVec := vVecVecOfInputs inputs
  let headValuesVec := headValueVecOfInputs inputs vVecVec
  let vals : Fin seq → Real := fun k => (headValuesVec.get k).get i
  dotProduct (weights q) vals

/-- Unfolding lemma for `headOutputWithScores`. -/
theorem headOutputWithScores_def (scores : Fin seq → Fin seq → Rat)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutputWithScores scores inputs q i =
      let weights : Fin seq → Fin seq → Real := fun q k =>
        Circuit.softmax (fun j => (scores q j : Real)) k
      let vVecVec := vVecVecOfInputs inputs
      let headValuesVec := headValueVecOfInputs inputs vVecVec
      let vals : Fin seq → Real := fun k => (headValuesVec.get k).get i
      dotProduct (weights q) vals := rfl

/-- Real-valued head output for a query and model dimension. -/
def headOutput (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  let qVecVec := qVecVecOfInputs inputs
  let kVecVec := kVecVecOfInputs inputs
  let scoresVec := scoresVecOfInputs inputs qVecVec kVecVec
  let scores : Fin seq → Fin seq → Rat := fun q k => (scoresVec.get q).get k
  headOutputWithScores scores inputs q i

/-- Unfolding lemma for `headOutput`. -/
theorem headOutput_def (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutput inputs q i =
      let qVecVec := qVecVecOfInputs inputs
      let kVecVec := kVecVecOfInputs inputs
      let scoresVec := scoresVecOfInputs inputs qVecVec kVecVec
      let scores : Fin seq → Fin seq → Rat := fun q k => (scoresVec.get q).get k
      headOutputWithScores scores inputs q i := rfl

/-- Soundness predicate for head-output interval bounds. -/
structure HeadOutputIntervalSound [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (scores : Fin seq → Fin seq → Rat)
    (active : Finset (Fin seq))
    (c : Circuit.ResidualIntervalCert dModel) : Prop where
  /-- Interval bounds are ordered coordinatewise. -/
  bounds : Circuit.ResidualIntervalBounds c
  /-- Active-query outputs lie inside the interval bounds. -/
  output_mem :
    ∀ q, q ∈ active → ∀ i,
      (c.lo i : Real) ≤ headOutputWithScores scores inputs q i ∧
        headOutputWithScores scores inputs q i ≤ (c.hi i : Real)

/-- Certified head-output interval data for a specific active set. -/
structure HeadOutputIntervalResult [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) where
  /-- Scores used to derive softmax weights. -/
  scores : Fin seq → Fin seq → Rat
  /-- Active queries covered by the interval bounds. -/
  active : Finset (Fin seq)
  /-- Residual-interval certificate for head outputs. -/
  cert : Circuit.ResidualIntervalCert dModel
  /-- Soundness proof for the interval bounds. -/
  sound : HeadOutputIntervalSound inputs scores active cert

/-- Build residual-interval bounds for head outputs on active queries. -/
def buildHeadOutputIntervalFromHead? [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (HeadOutputIntervalResult inputs) := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      cases hbuild : buildInductionCertFromHead? inputs with
      | none => exact none
      | some certWithProof =>
          rcases certWithProof with ⟨cert, hcert⟩
          let vVecVec := vVecVecOfInputs inputs
          let headValuesVec := headValueVecOfInputs inputs vVecVec
          let headValue : Fin (Nat.succ n) → Fin dModel → Rat := fun k i =>
            (headValuesVec.get k).get i
          let scores : Fin (Nat.succ n) → Fin (Nat.succ n) → Rat := cert.scores
          let scoresReal : Fin (Nat.succ n) → Fin (Nat.succ n) → Real :=
            fun q k => (scores q k : Real)
          let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := fun q k =>
            Circuit.softmax (scoresReal q) k
          let activeSet : Finset (Fin (Nat.succ n)) := cert.active
          let univ : Finset (Fin (Nat.succ n)) := Finset.univ
          have huniv : univ.Nonempty := Finset.univ_nonempty
          let loVal : Fin dModel → Rat := fun i =>
            univ.inf' huniv (fun k => headValue k i)
          let hiVal : Fin dModel → Rat := fun i =>
            univ.sup' huniv (fun k => headValue k i)
          have hvalsBounds :
              ∀ i, Layers.ValueRangeBounds (Val := Rat) (loVal i) (hiVal i)
                (fun k => headValue k i) := by
            intro i
            refine { lo_le_hi := ?_, lo_le := ?_, le_hi := ?_ }
            · rcases huniv with ⟨k0, hk0⟩
              have hlo :=
                Finset.inf'_le (s := univ) (f := fun k => headValue k i) hk0
              have hhi :=
                Finset.le_sup' (s := univ) (f := fun k => headValue k i) hk0
              exact le_trans hlo hhi
            · intro k
              exact Finset.inf'_le (s := univ) (f := fun k => headValue k i) (by simp [univ])
            · intro k
              exact Finset.le_sup' (s := univ) (f := fun k => headValue k i) (by simp [univ])
          have hvalsBoundsReal :
              ∀ i, Layers.ValueRangeBounds (Val := Real)
                (loVal i : Real) (hiVal i : Real)
                (fun k => (headValue k i : Real)) := by
            intro i
            have hvals := hvalsBounds i
            refine { lo_le_hi := ?_, lo_le := ?_, le_hi := ?_ }
            · exact (Rat.cast_le (K := Real)).2 hvals.lo_le_hi
            · intro k
              exact (Rat.cast_le (K := Real)).2 (hvals.lo_le k)
            · intro k
              exact (Rat.cast_le (K := Real)).2 (hvals.le_hi k)
          have hsoftmax :
              Layers.SoftmaxMarginBoundsOn (Val := Real) (cert.eps : Real) (cert.margin : Real)
                (fun q => q ∈ activeSet) cert.prev scoresReal weights := by
            simpa [scoresReal, weights, activeSet] using hcert.softmax_bounds
          have hweights :
              Layers.OneHotApproxBoundsOnActive (Val := Real) (cert.eps : Real)
                (fun q => q ∈ activeSet) cert.prev weights :=
            Layers.oneHotApproxBoundsOnActive_of_softmaxMargin
              (Val := Real)
              (ε := (cert.eps : Real))
              (margin := (cert.margin : Real))
              (active := fun q => q ∈ activeSet)
              (prev := cert.prev)
              (scores := scoresReal)
              (weights := weights)
              hsoftmax
          have happrox :
              ∀ i, Layers.InductionSpecApproxOn (Val := Real) (n := n)
                ((cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)))
                (fun q => q ∈ activeSet) cert.prev
                (fun q => dotProduct (weights q) (fun k => (headValue k i : Real)))
                (fun k => (headValue k i : Real)) := by
            intro i
            exact
              Layers.inductionSpecApproxOn_of_oneHotApprox_valueRange
                (Val := Real)
                (n := n)
                (ε := (cert.eps : Real))
                (lo := (loVal i : Real))
                (hi := (hiVal i : Real))
                (active := fun q => q ∈ activeSet)
                (prev := cert.prev)
                (weights := weights)
                (vals := fun k => (headValue k i : Real))
                (hweights := hweights)
                (hvals := hvalsBoundsReal i)
          let delta : Fin dModel → Rat := fun i => hiVal i - loVal i
          let boundLoRat : Fin (Nat.succ n) → Fin dModel → Rat := fun q i =>
            headValue (cert.prev q) i - cert.eps * delta i
          let boundHiRat : Fin (Nat.succ n) → Fin dModel → Rat := fun q i =>
            headValue (cert.prev q) i + cert.eps * delta i
          let loOut : Fin dModel → Rat := fun i =>
            if h : activeSet.Nonempty then
              activeSet.inf' h (fun q => boundLoRat q i)
            else
              0
          let hiOut : Fin dModel → Rat := fun i =>
            if h : activeSet.Nonempty then
              activeSet.sup' h (fun q => boundHiRat q i)
            else
              0
          have hout :
              ∀ q, q ∈ activeSet → ∀ i,
                (loOut i : Real) ≤ headOutputWithScores scores inputs q i ∧
                  headOutputWithScores scores inputs q i ≤ (hiOut i : Real) := by
            intro q hq i
            have hactive : activeSet.Nonempty := ⟨q, hq⟩
            have hspec := (happrox i) q hq
            have hout_def :
                headOutputWithScores scores inputs q i =
                  dotProduct (weights q) (fun k => (headValue k i : Real)) := by
              simp [headOutputWithScores, scoresReal, weights, headValue, headValuesVec, vVecVec]
            have hupper :
                headOutputWithScores scores inputs q i ≤ (boundHiRat q i : Real) := by
              have hupper' :=
                (happrox i) q hq |>.1
              simpa [hout_def, boundHiRat, delta] using hupper'
            have hlower :
                (boundLoRat q i : Real) ≤ headOutputWithScores scores inputs q i := by
              have hlower' :
                  (headValue (cert.prev q) i : Real) -
                      (cert.eps : Real) * ((hiVal i : Real) - (loVal i : Real)) ≤
                    dotProduct (weights q) (fun k => (headValue k i : Real)) := by
                exact (sub_le_iff_le_add).2 hspec.2
              simpa [hout_def, boundLoRat, delta] using hlower'
            have hlo :
                (loOut i : Real) ≤ (boundLoRat q i : Real) := by
              have hloRat : loOut i ≤ boundLoRat q i := by
                simpa [loOut, hactive] using
                  (Finset.inf'_le (s := activeSet) (f := fun q => boundLoRat q i) hq)
              exact (Rat.cast_le (K := Real)).2 hloRat
            have hhi :
                (boundHiRat q i : Real) ≤ (hiOut i : Real) := by
              have hhiRat : boundHiRat q i ≤ hiOut i := by
                simpa [hiOut, hactive] using
                  (Finset.le_sup' (s := activeSet) (f := fun q => boundHiRat q i) hq)
              exact (Rat.cast_le (K := Real)).2 hhiRat
            exact ⟨le_trans hlo hlower, le_trans hupper hhi⟩
          have hbounds : Circuit.ResidualIntervalBounds { lo := loOut, hi := hiOut } := by
            refine { lo_le_hi := ?_ }
            intro i
            by_cases hactive : activeSet.Nonempty
            · rcases hactive with ⟨q, hq⟩
              have hout_i := hout q hq i
              have hleReal : (loOut i : Real) ≤ (hiOut i : Real) :=
                le_trans hout_i.1 hout_i.2
              exact (Rat.cast_le (K := Real)).1 hleReal
            · simp [loOut, hiOut, hactive]
          let certOut : Circuit.ResidualIntervalCert dModel := { lo := loOut, hi := hiOut }
          exact some
            { scores := scores
              active := activeSet
              cert := certOut
              sound :=
                { bounds := hbounds
                  output_mem := by
                    intro q hq i
                    exact hout q hq i } }

end

end HeadOutputInterval

end Sound

end Nfp
