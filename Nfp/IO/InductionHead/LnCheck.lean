-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.Bounds.LayerNorm

/-!
LayerNorm residual checks for induction-head certificates.
-/

public section

namespace Nfp

namespace IO

namespace InductionHeadCert

/-- Check that residual entries fall within LayerNorm bounds from pre-LN inputs. -/
def residWithinLayerNormBounds {seq : Nat} (slice : ModelLnSlice seq)
    (resid : Fin seq → Fin slice.dModel → Rat) : Bool :=
  let pow10 (d : Nat) : Nat := Nat.pow 10 d
  let ratScaledNum? (den : Nat) (x : Rat) : Option Int :=
    if x.den ∣ den then
      let mul := den / x.den
      some (x.num * (Int.ofNat mul))
    else
      none
  let ratMulNatFloor (x : Rat) (m : Nat) : Int :=
    let num := x.num
    let den := (x.den : Int)
    let prod := num * (Int.ofNat m)
    if prod ≥ 0 then
      prod / den
    else
      (prod - (den - 1)) / den
  let ratMulNatCeil (x : Rat) (m : Nat) : Int :=
    let num := x.num
    let den := (x.den : Int)
    let prod := num * (Int.ofNat m)
    if prod ≥ 0 then
      (prod + (den - 1)) / den
    else
      prod / den
  let fracLe (aNum aDen bNum bDen : Nat) : Bool :=
    aNum * bDen ≤ bNum * aDen
  let maxFrac (aNum aDen bNum bDen : Nat) : Nat × Nat :=
    if fracLe aNum aDen bNum bDen then (bNum, bDen) else (aNum, aDen)
  let minFrac (aNum aDen bNum bDen : Nat) : Nat × Nat :=
    if fracLe aNum aDen bNum bDen then (aNum, aDen) else (bNum, bDen)
  let sqrtLowerNumDen (numAbs den scale : Nat) : Nat × Nat :=
    let aBase := Nat.sqrt numAbs
    let bBase := Nat.sqrt den + 1
    let aAlt := Nat.sqrt (numAbs * den)
    let bAlt := den
    let aScaled := Nat.sqrt (numAbs * den * scale * scale)
    let bScaled := den * scale
    let m1 := maxFrac aBase bBase aAlt bAlt
    maxFrac m1.1 m1.2 aScaled bScaled
  let sqrtUpperNumDen (numAbs den scale : Nat) : Nat × Nat :=
    let aBase := Nat.sqrt numAbs + 1
    let bBase := Nat.sqrt den
    let aAlt := Nat.sqrt (numAbs * den) + 1
    let bAlt := den
    let aScaled := Nat.sqrt (numAbs * den * scale * scale) + 1
    let bScaled := den * scale
    let m1 := minFrac aBase bBase aAlt bAlt
    minFrac m1.1 m1.2 aScaled bScaled
  let residWithinFixed : Option Bool :=
    match slice.lnFast? with
    | some true => do
        let d ← slice.lnDecimals?
        let den := pow10 d
        if den = 0 then
          none
        else
          let gamma ←
            (List.finRange slice.dModel).foldlM (m := Option)
              (fun arr i => do
                let v ← ratScaledNum? den (slice.lnGamma i)
                return arr.push v)
              (Array.mkEmpty slice.dModel)
          let beta ←
            (List.finRange slice.dModel).foldlM (m := Option)
              (fun arr i => do
                let v ← ratScaledNum? den (slice.lnBeta i)
                return arr.push v)
              (Array.mkEmpty slice.dModel)
          let slackNum ← ratScaledNum? den slice.lnSlack
          let n := slice.dModel
          let nInt := Int.ofNat n
          let denNat := den
          let coeffDen : Nat := denNat * denNat * n
          let varDen : Nat := denNat * denNat * n * n * n
          let scale := slice.lnScale?.getD Bounds.sqrtLowerScale
          let mut ok := true
          for q in List.finRange seq do
            if ok then
              let rowEmbed? :=
                (List.finRange n).foldlM (m := Option)
                  (fun arr i => do
                    let v ← ratScaledNum? den (slice.embed q i)
                    return arr.push v)
                  (Array.mkEmpty n)
              let rowResid? :=
                (List.finRange n).foldlM (m := Option)
                  (fun arr i => do
                    let v ← ratScaledNum? den (resid q i)
                    return arr.push v)
                  (Array.mkEmpty n)
              match rowEmbed?, rowResid? with
              | some rowEmbed, some rowResid =>
                  let sum := rowEmbed.foldl (fun acc v => acc + v) 0
                  let varNum :=
                    rowEmbed.foldl (fun acc v =>
                      let centered := v * nInt - sum
                      acc + centered * centered) 0
                  let epsLo := ratMulNatFloor slice.lnEps varDen
                  let epsHi := ratMulNatCeil slice.lnEps varDen
                  let varLo := varNum + epsLo
                  let varHi := varNum + epsHi
                  let varLoAbs := Int.natAbs varLo
                  let varHiAbs := Int.natAbs varHi
                  let epsLoAbs := Int.natAbs epsLo
                  let sqrtLowerEps := sqrtLowerNumDen epsLoAbs varDen scale
                  let sqrtLowerVar := sqrtLowerNumDen varLoAbs varDen scale
                  let sqrtLowerBound :=
                    maxFrac sqrtLowerEps.1 sqrtLowerEps.2
                      sqrtLowerVar.1 sqrtLowerVar.2
                  let sqrtUpperBound := sqrtUpperNumDen varHiAbs varDen scale
                  let invUpperNum : Int :=
                    if sqrtLowerBound.1 = 0 then 0 else Int.ofNat sqrtLowerBound.2
                  let invUpperDen : Nat :=
                    if sqrtLowerBound.1 = 0 then 1 else sqrtLowerBound.1
                  let invLowerNum : Int :=
                    if sqrtUpperBound.1 = 0 then 0 else Int.ofNat sqrtUpperBound.2
                  let invLowerDen : Nat :=
                    if sqrtUpperBound.1 = 0 then 1 else sqrtUpperBound.1
                  for i in List.finRange n do
                    if ok then
                      let v := rowEmbed[i.1]!
                      let centered := v * nInt - sum
                      let coeffNum := (gamma[i.1]!) * centered
                      let residNum := rowResid[i.1]!
                      let betaNum := beta[i.1]!
                      let loInvNum := if coeffNum ≥ 0 then invLowerNum else invUpperNum
                      let loInvDen := if coeffNum ≥ 0 then invLowerDen else invUpperDen
                      let hiInvNum := if coeffNum ≥ 0 then invUpperNum else invLowerNum
                      let hiInvDen := if coeffNum ≥ 0 then invUpperDen else invLowerDen
                      let lhsLo :=
                        (residNum - betaNum + slackNum) *
                          (Int.ofNat coeffDen) * (Int.ofNat loInvDen)
                      let rhsLo :=
                        coeffNum * loInvNum * (Int.ofNat denNat)
                      let lhsHi :=
                        (residNum - betaNum - slackNum) *
                          (Int.ofNat coeffDen) * (Int.ofNat hiInvDen)
                      let rhsHi :=
                        coeffNum * hiInvNum * (Int.ofNat denNat)
                      if lhsLo < rhsLo || lhsHi > rhsHi then
                        ok := false
              | _, _ => ok := false
          return ok
    | _ => none
  match residWithinFixed with
  | some ok => ok
  | none =>
      (List.finRange seq).all (fun q =>
        let bounds :=
          match slice.lnScale? with
          | some scale =>
              Bounds.layerNormBoundsWithScale scale
                slice.lnEps slice.lnGamma slice.lnBeta (slice.embed q)
          | none =>
              Bounds.layerNormBounds slice.lnEps slice.lnGamma slice.lnBeta (slice.embed q)
        (List.finRange slice.dModel).all (fun i =>
          let lo := bounds.1 i - slice.lnSlack
          let hi := bounds.2 i + slice.lnSlack
          decide (lo ≤ resid q i) && decide (resid q i ≤ hi)))

end InductionHeadCert

end IO

end Nfp
