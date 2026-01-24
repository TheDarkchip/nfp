-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Core.Basic
public import Nfp.Sound.Induction.OneHot

/-!
Score-gap bounds derived from score intervals.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Circuit

variable {seq : Nat}

/-- Score-gap lower bound obtained from per-key score intervals. -/
def scoreGapLoOfBounds (prev : Fin seq → Fin seq)
    (scoreLo scoreHi : Fin seq → Fin seq → Rat) : Fin seq → Fin seq → Rat :=
  fun q k => scoreLo q (prev q) - scoreHi q k

/--
Weight-bound formula derived from score gaps.
-/
def weightBoundAtOfScoreGap (prev : Fin seq → Fin seq)
    (scoreGapLo : Fin seq → Fin seq → Rat) : Fin seq → Fin seq → Rat :=
  fun q k =>
    if k = prev q then
      0
    else
      if scoreGapLo q k < 0 then
        (1 : Rat)
      else
        ratDivUp 1 (1 + scoreGapLo q k)

/-- Unfold `weightBoundAtOfScoreGap` on non-`prev` keys. -/
theorem weightBoundAtOfScoreGap_eq {prev : Fin seq → Fin seq}
    {scoreGapLo : Fin seq → Fin seq → Rat} {q k : Fin seq} (hk : k ≠ prev q) :
    weightBoundAtOfScoreGap prev scoreGapLo q k =
      if scoreGapLo q k < 0 then
        (1 : Rat)
      else
        ratDivUp 1 (1 + scoreGapLo q k) := by
  simp [weightBoundAtOfScoreGap, hk]

/--
Per-query epsilon from per-key weight bounds.
-/
def epsAtOfWeightBoundAt (prev : Fin seq → Fin seq)
    (weightBoundAt : Fin seq → Fin seq → Rat) : Fin seq → Rat :=
  fun q =>
    min (1 : Rat)
      ((Finset.univ : Finset (Fin seq)).erase (prev q) |>.sum (fun k =>
        weightBoundAt q k))

/-- Unfolding lemma for `epsAtOfWeightBoundAt`. -/
theorem epsAtOfWeightBoundAt_def (prev : Fin seq → Fin seq)
    (weightBoundAt : Fin seq → Fin seq → Rat) (q : Fin seq) :
    epsAtOfWeightBoundAt prev weightBoundAt q =
      min (1 : Rat)
        ((Finset.univ : Finset (Fin seq)).erase (prev q) |>.sum (fun k =>
          weightBoundAt q k)) := by
  rfl

/--
Score-gap lower bounds from score intervals are sound for real scores.
-/
theorem scoreGapLoOfBounds_real_at
    (prev : Fin seq → Fin seq)
    (scoresReal : Fin seq → Fin seq → Real)
    (scoreLo scoreHi : Fin seq → Fin seq → Rat)
    (hlo : ∀ q k, (scoreLo q k : Real) ≤ scoresReal q k)
    (hhi : ∀ q k, scoresReal q k ≤ (scoreHi q k : Real)) :
    ∀ q k,
      scoresReal q k + (scoreGapLoOfBounds prev scoreLo scoreHi q k : Real) ≤
        scoresReal q (prev q) := by
  intro q k
  have hlo_prev := hlo q (prev q)
  have hhi_k := hhi q k
  calc
    scoresReal q k + (scoreGapLoOfBounds prev scoreLo scoreHi q k : Real)
        =
        scoresReal q k + (scoreLo q (prev q) : Real) - (scoreHi q k : Real) := by
          simp [scoreGapLoOfBounds, sub_eq_add_neg, add_assoc]
    _ ≤ scoresReal q (prev q) := by
          linarith

/--
Per-key weight bounds from score intervals, via score-gap bounds.
-/
theorem weight_bound_at_of_scoreBounds [NeZero seq]
    (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (scoresReal : Fin seq → Fin seq → Real)
    (scoreLo scoreHi : Fin seq → Fin seq → Rat)
    (weightBoundAt : Fin seq → Fin seq → Rat)
    (hweightBoundAt :
      ∀ q k, k ≠ prev q →
        weightBoundAt q k =
          if scoreGapLoOfBounds prev scoreLo scoreHi q k < 0 then
            (1 : Rat)
          else
            ratDivUp 1 (1 + scoreGapLoOfBounds prev scoreLo scoreHi q k))
    (hscore_bounds :
      ∀ q k, (scoreLo q k : Real) ≤ scoresReal q k ∧
        scoresReal q k ≤ (scoreHi q k : Real)) :
    ∀ q, q ∈ active → ∀ k, k ≠ prev q →
      Circuit.softmax (scoresReal q) k ≤ (weightBoundAt q k : Real) := by
  intro q hq k hk
  let scoreGapLo := scoreGapLoOfBounds prev scoreLo scoreHi
  have hscore_gap_real_at :
      ∀ q, q ∈ active → ∀ k, k ≠ prev q →
        scoresReal q k + (scoreGapLo q k : Real) ≤ scoresReal q (prev q) := by
    intro q' hq' k' hk'
    have hgap :=
      scoreGapLoOfBounds_real_at
        (prev := prev)
        (scoresReal := scoresReal)
        (scoreLo := scoreLo)
        (scoreHi := scoreHi)
        (hlo := fun a b => (hscore_bounds a b).1)
        (hhi := fun a b => (hscore_bounds a b).2)
    exact hgap q' k'
  have hweightBoundAt' :
      ∀ q k, k ≠ prev q →
        weightBoundAt q k =
          if scoreGapLo q k < 0 then
            (1 : Rat)
          else
            ratDivUp 1 (1 + scoreGapLo q k) := by
    intro q' k' hk'
    simpa [scoreGapLo] using hweightBoundAt q' k' hk'
  have hbound :=
    weight_bound_at_of_scoreGapLo
      (active := active)
      (prev := prev)
      (scoresReal := scoresReal)
      (scoreGapLo := scoreGapLo)
      (weightBoundAt := weightBoundAt)
      hweightBoundAt'
      hscore_gap_real_at
  exact hbound q hq k hk

end Sound

end Nfp
