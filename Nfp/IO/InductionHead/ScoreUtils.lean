-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Model.InductionPrompt

/-!
Scoring helpers for induction-head certificate checks.
-/

public section

namespace Nfp

namespace IO

/-- Render a rational value as a string. -/
def ratToString (x : Rat) : String :=
  toString x

/-- Check if tokens repeat with a fixed period. -/
def tokensPeriodic {seq : Nat} (period : Nat) (tokens : Fin seq → Nat) : Bool :=
  (List.finRange seq).all (fun q =>
    if period ≤ q.val then
      decide (tokens q = tokens (Model.prevOfPeriod (seq := seq) period q))
    else
      true)

/-- Sum a matrix over all query/key pairs. -/
private def sumOver {seq : Nat} (f : Fin seq → Fin seq → Rat) : Rat :=
  (List.finRange seq).foldl (fun acc q =>
    acc + (List.finRange seq).foldl (fun acc' k => acc' + f q k) 0) 0

/-- Boolean TL induction pattern indicator (duplicate head shifted right). -/
private def tlPatternBool {seq : Nat} (tokens : Fin seq → Nat) (q k : Fin seq) : Bool :=
  (List.finRange seq).any (fun r =>
    decide (r < q) && decide (tokens r = tokens q) && decide (k.val = r.val + 1))

/-- Numeric TL induction pattern indicator (0/1). -/
private def tlPatternIndicator {seq : Nat} (tokens : Fin seq → Nat) (q k : Fin seq) : Rat :=
  if tlPatternBool (seq := seq) tokens q k then 1 else 0

/-- TL mul numerator with TL-style masking. -/
private def tlMulNumeratorMasked {seq : Nat} (excludeBos excludeCurrent : Bool)
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat) : Rat :=
  sumOver (seq := seq) (fun q k =>
    Model.tlMaskedWeights (seq := seq) excludeBos excludeCurrent weights q k *
      tlPatternIndicator (seq := seq) tokens q k)

/-- TL mul score with TL-style masking. -/
def tlMulScoreMasked {seq : Nat} (excludeBos excludeCurrent : Bool)
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat) : Rat :=
  let masked := Model.tlMaskedWeights (seq := seq) excludeBos excludeCurrent weights
  let denom := sumOver (seq := seq) masked
  if denom = 0 then
    0
  else
    tlMulNumeratorMasked (seq := seq) excludeBos excludeCurrent weights tokens / denom

/-- Copying score utility (ratio scaled to [-1, 1]). -/
def copyScore {seq : Nat} (weights copyLogits : Fin seq → Fin seq → Rat) : Option Rat :=
  let total := sumOver (seq := seq) copyLogits
  if total = 0 then
    none
  else
    let weighted := sumOver (seq := seq) (fun q k => weights q k * copyLogits q k)
    let ratio := weighted / total
    some ((4 : Rat) * ratio - 1)

end IO

end Nfp
