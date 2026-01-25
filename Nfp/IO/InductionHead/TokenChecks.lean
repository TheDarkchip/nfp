-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.InductionHead
public import Nfp.Model.InductionPrompt
public import Nfp.IO.InductionHead.ScoreUtils

/-!
Token-derived sanity checks for induction-head certificates.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/-- Check periodicity for induction-aligned prompts. -/
def checkPeriodicTokens? {seq : Nat} (period? : Option Nat) (tokens : Fin seq → Nat) :
    Except String Unit := do
  match period? with
  | none =>
      throw "missing period entry for induction-aligned"
  | some period =>
      if tokensPeriodic (seq := seq) period tokens then
        pure ()
      else
        throw "tokens are not periodic for induction-aligned period"

/-- Check that the certificate's active set is contained in token repeats. -/
def activeTokensSuperset {seq : Nat} (cert : Circuit.InductionHeadCert seq)
    (tokens : Fin seq → Nat) : Bool :=
  decide (cert.active ⊆ Model.activeOfTokens (seq := seq) tokens)

/-- Check that the certificate's prev map matches token repeats on active queries. -/
def prevMapMatchesTokens {seq : Nat} (cert : Circuit.InductionHeadCert seq)
    (tokens : Fin seq → Nat) : Bool :=
  let prevTokens := Model.prevOfTokens (seq := seq) tokens
  (List.finRange seq).all (fun q =>
    if decide (q ∈ cert.active) then
      decide (prevTokens q = cert.prev q)
    else
      true)

/-- Check that the certificate's prev map matches shifted-token repeats on active queries. -/
def prevMapMatchesTokensShift {seq : Nat} (cert : Circuit.InductionHeadCert seq)
    (tokens : Fin seq → Nat) : Bool :=
  let prevTokens := Model.prevOfTokensShift (seq := seq) tokens
  (List.finRange seq).all (fun q =>
    if decide (q ∈ cert.active) then
      decide (prevTokens q = cert.prev q)
    else
      true)

/-- Check direction metadata against the shifted-token continuation at `q`. -/
def checkDirectionQ {seq : Nat} (tokens : Fin seq → Nat) (qNat : Nat)
    (cert : Circuit.InductionHeadCert seq) : Except String Unit := do
  if hq : qNat < seq then
    let q : Fin seq := ⟨qNat, hq⟩
    let activeTokens := Model.activeOfTokensShift (seq := seq) tokens
    if !decide (q ∈ activeTokens) then
      throw "direction-q is not active under shifted-token rules"
    match cert.values.direction with
    | none =>
        throw "direction-q requires direction metadata"
    | some dir =>
        let expected :=
          Model.directionSpecOfTokensShift (seq := seq) tokens q
        if dir.target != expected.target ||
            dir.negative != expected.negative then
          throw "direction metadata does not match shifted-token direction"
  else
    throw "direction-q out of range for tokens"

/-- Check direction-q metadata, requiring a tokens file. -/
def checkDirectionQWithTokens? {seq : Nat} (tokensOpt : Option (Fin seq → Nat)) (qNat : Nat)
    (cert : Circuit.InductionHeadCert seq) : Except String Unit := do
  match tokensOpt with
  | none =>
      throw "direction-q requires a tokens file"
  | some tokens' =>
      checkDirectionQ (seq := seq) tokens' qNat cert

/-- Check token semantics (and optional direction-q metadata) against a certificate. -/
def checkTokensAndDirection? {seq : Nat} (kind : String) (period? : Option Nat)
    (directionQ? : Option Nat) (cert : Circuit.InductionHeadCert seq)
    (tokensOpt : Option (Fin seq → Nat)) : Except String Unit := do
  match tokensOpt with
  | none =>
      if directionQ?.isSome then
        throw "direction-q requires a tokens file"
      else
        pure ()
  | some tokens' =>
      if kind = "induction-aligned" then
        checkPeriodicTokens? (seq := seq) period? tokens'
      else
        if !activeTokensSuperset (seq := seq) cert tokens' then
          throw "active set not contained in token repeats"
        let prevOk :=
          if directionQ?.isSome then
            prevMapMatchesTokensShift (seq := seq) cert tokens'
          else
            prevMapMatchesTokens (seq := seq) cert tokens'
        if !prevOk then
          throw "prev map does not match tokens on active queries"
      if let some qNat := directionQ? then
        checkDirectionQ (seq := seq) tokens' qNat cert

end InductionHeadCert

end IO

end Nfp
