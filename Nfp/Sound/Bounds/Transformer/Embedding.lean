-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Nfp.Core.Basic

/-!
Embedding interval bounds for transformer stacks.

This module isolates per-position and per-set embedding bounds.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

private lemma fin_univ_nonempty (seq : Nat) [NeZero seq] :
    (Finset.univ : Finset (Fin seq)).Nonempty := by
  classical
  refine ⟨⟨0, ?_⟩, by simp⟩
  exact Nat.pos_of_ne_zero (NeZero.ne (n := seq))

/-- Interval bounds across tokens for an embedding map. -/
def embeddingIntervalBounds {seq dModel : Nat} [NeZero seq]
    (x : Fin seq → Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let h : (Finset.univ : Finset (Fin seq)).Nonempty := fin_univ_nonempty (seq := seq)
  (fun i => (Finset.univ).inf' h (fun q => x q i),
   fun i => (Finset.univ).sup' h (fun q => x q i))

/-- `embeddingIntervalBounds` bounds embeddings coordinatewise. -/
theorem embeddingIntervalBounds_spec {seq dModel : Nat} [NeZero seq]
    (x : Fin seq → Fin dModel → Rat) :
    let bounds := embeddingIntervalBounds x
    ∀ q i,
      (bounds.1 i : Real) ≤ (x q i : Real) ∧
        (x q i : Real) ≤ (bounds.2 i : Real) := by
  classical
  intro bounds q i
  have hloRat : bounds.1 i ≤ x q i := by
    have h :=
      Finset.inf'_le (s := (Finset.univ : Finset (Fin seq)))
        (f := fun k => x k i) (b := q) (by simp)
    simpa [bounds, embeddingIntervalBounds, fin_univ_nonempty] using h
  have hhiRat : x q i ≤ bounds.2 i := by
    have h :=
      Finset.le_sup' (s := (Finset.univ : Finset (Fin seq)))
        (f := fun k => x k i) (b := q) (by simp)
    simpa [bounds, embeddingIntervalBounds, fin_univ_nonempty] using h
  constructor
  · exact ratToReal_le_of_le hloRat
  · exact ratToReal_le_of_le hhiRat

/-- Interval bounds across a finite set of positions for an embedding map. -/
def embeddingIntervalBoundsOn {seq dModel : Nat} [NeZero seq]
    (positions : Finset (Fin seq)) (hpos : positions.Nonempty)
    (x : Fin seq → Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  (fun i => positions.inf' hpos (fun q => x q i),
   fun i => positions.sup' hpos (fun q => x q i))

/-- `embeddingIntervalBoundsOn` bounds embeddings on the chosen positions. -/
theorem embeddingIntervalBoundsOn_spec {seq dModel : Nat} [NeZero seq]
    (positions : Finset (Fin seq)) (hpos : positions.Nonempty)
    (x : Fin seq → Fin dModel → Rat) :
    let bounds := embeddingIntervalBoundsOn positions hpos x
    ∀ q, q ∈ positions → ∀ i,
      (bounds.1 i : Real) ≤ (x q i : Real) ∧
        (x q i : Real) ≤ (bounds.2 i : Real) := by
  classical
  intro bounds q hq i
  have hloRat : bounds.1 i ≤ x q i := by
    have h :=
      Finset.inf'_le (s := positions)
        (f := fun k => x k i) (b := q) hq
    simpa [bounds, embeddingIntervalBoundsOn] using h
  have hhiRat : x q i ≤ bounds.2 i := by
    have h :=
      Finset.le_sup' (s := positions)
        (f := fun k => x k i) (b := q) hq
    simpa [bounds, embeddingIntervalBoundsOn] using h
  constructor
  · exact ratToReal_le_of_le hloRat
  · exact ratToReal_le_of_le hhiRat

/-- Collapse per-position interval bounds over a finite set of positions. -/
def intervalBoundsOn {seq dModel : Nat} [NeZero seq]
    (positions : Finset (Fin seq)) (hpos : positions.Nonempty)
    (lo hi : Fin seq → Fin dModel → Rat) : (Fin dModel → Rat) × (Fin dModel → Rat) :=
  (fun i => positions.inf' hpos (fun q => lo q i),
   fun i => positions.sup' hpos (fun q => hi q i))

/-- `intervalBoundsOn` soundness for bounds on the chosen positions. -/
theorem intervalBoundsOn_spec {seq dModel : Nat} [NeZero seq]
    (positions : Finset (Fin seq)) (hpos : positions.Nonempty)
    (lo hi : Fin seq → Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hlo : ∀ q, q ∈ positions → ∀ i, (lo q i : Real) ≤ x q i)
    (hhi : ∀ q, q ∈ positions → ∀ i, x q i ≤ (hi q i : Real)) :
    let bounds := intervalBoundsOn positions hpos lo hi
    ∀ q, q ∈ positions → ∀ i,
      (bounds.1 i : Real) ≤ x q i ∧
        x q i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds q hq i
  have hmin : bounds.1 i ≤ lo q i := by
    have h :=
      Finset.inf'_le (s := positions)
        (f := fun k => lo k i) (b := q) hq
    simpa [bounds, intervalBoundsOn] using h
  have hmax : hi q i ≤ bounds.2 i := by
    have h :=
      Finset.le_sup' (s := positions)
        (f := fun k => hi k i) (b := q) hq
    simpa [bounds, intervalBoundsOn] using h
  have hlo' := hlo q hq i
  have hhi' := hhi q hq i
  constructor
  · have hmin_real :
        (bounds.1 i : Real) ≤ (lo q i : Real) := by
      exact ratToReal_le_of_le hmin
    exact le_trans hmin_real hlo'
  · have hmax_real :
        (hi q i : Real) ≤ (bounds.2 i : Real) := by
      exact ratToReal_le_of_le hmax
    exact le_trans hhi' hmax_real

end Bounds

end Sound

end Nfp
