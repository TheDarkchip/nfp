-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.Interval
public import Nfp.Bounds.LayerNorm
public import Nfp.Bounds.Transformer

/-!
End-to-end bounds for transformer stacks using interval propagation.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Bounds

variable {seq dModel dHead numHeads hidden : Nat}

/--
Bound the direction logit for a transformer stack by propagating interval bounds
through the stack and final LayerNorm.
-/
theorem transformerStackLogitBounds_spec {seq dModel dHead numHeads hidden : Nat} [NeZero seq]
    (eps : Rat)
    (layers : List (Bounds.TransformerLayerData seq dModel dHead numHeads hidden))
    (finalGamma finalBeta : Fin dModel → Rat)
    (direction : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let stackBounds :=
      Bounds.transformerStackBounds eps (layers.map (fun layer => layer.params)) lo hi
    let absBound := Bounds.intervalAbsBound stackBounds.1 stackBounds.2
    let lnBounds := Bounds.layerNormAbsBounds eps finalGamma finalBeta absBound
    ∀ q,
      (Bounds.dotIntervalLower direction lnBounds.1 lnBounds.2 : Real) ≤
        dotProduct (fun i => (direction i : Real))
          (fun i =>
            layerNormRealOfReal eps finalGamma finalBeta
              (fun j => Bounds.transformerStackReal eps layers x q j) i) ∧
      dotProduct (fun i => (direction i : Real))
          (fun i =>
            layerNormRealOfReal eps finalGamma finalBeta
              (fun j => Bounds.transformerStackReal eps layers x q j) i) ≤
        (Bounds.dotIntervalUpper direction lnBounds.1 lnBounds.2 : Real) := by
  classical
  intro stackBounds absBound lnBounds q
  have hstack :=
    Bounds.transformerStackBounds_spec (seq := seq) (dModel := dModel) (dHead := dHead)
      (numHeads := numHeads) (hidden := hidden) eps layers lo hi x hne heps hsqrt hlo hhi
  have hloStack : ∀ i,
      (stackBounds.1 i : Real) ≤ Bounds.transformerStackReal eps layers x q i := by
    intro i
    have h := (hstack q i).1
    simpa [stackBounds] using h
  have hhiStack : ∀ i,
      Bounds.transformerStackReal eps layers x q i ≤ (stackBounds.2 i : Real) := by
    intro i
    have h := (hstack q i).2
    simpa [stackBounds] using h
  have habs : ∀ i,
      |Bounds.transformerStackReal eps layers x q i| ≤ (absBound : Real) := by
    intro i
    have hbound :
        |Bounds.transformerStackReal eps layers x q i| ≤
          max |(stackBounds.1 i : Real)| |(stackBounds.2 i : Real)| :=
      abs_le_max_abs_abs_of_interval_real (hloStack i) (hhiStack i)
    have hsup : max |stackBounds.1 i| |stackBounds.2 i| ≤
        intervalAbsBound stackBounds.1 stackBounds.2 :=
      max_abs_le_intervalAbsBound stackBounds.1 stackBounds.2 i
    have hsup_real :
        max |(stackBounds.1 i : Real)| |(stackBounds.2 i : Real)| ≤ (absBound : Real) := by
      have hsup' :
          ratToReal (max |stackBounds.1 i| |stackBounds.2 i|) ≤ ratToReal absBound :=
        ratToReal_le_of_le hsup
      simpa [ratToReal_abs, ratToReal_max, ratToReal_def] using hsup'
    exact le_trans hbound hsup_real
  have hln :
      ∀ i,
        (lnBounds.1 i : Real) ≤
            layerNormRealOfReal eps finalGamma finalBeta
              (fun j => Bounds.transformerStackReal eps layers x q j) i ∧
          layerNormRealOfReal eps finalGamma finalBeta
              (fun j => Bounds.transformerStackReal eps layers x q j) i ≤
            (lnBounds.2 i : Real) := by
    have hln' :=
      layerNormAbsBounds_spec_real eps finalGamma finalBeta absBound
        (x := fun j => Bounds.transformerStackReal eps layers x q j) hne heps hsqrt habs
    simpa [lnBounds] using hln'
  have hloDot :=
    dotIntervalLower_le_dotProduct_real (v := direction) (lo := lnBounds.1) (hi := lnBounds.2)
      (x := fun i =>
        layerNormRealOfReal eps finalGamma finalBeta
          (fun j => Bounds.transformerStackReal eps layers x q j) i)
      (fun i => (hln i).1) (fun i => (hln i).2)
  have hhiDot :=
    dotProduct_le_dotIntervalUpper_real (v := direction) (lo := lnBounds.1) (hi := lnBounds.2)
      (x := fun i =>
        layerNormRealOfReal eps finalGamma finalBeta
          (fun j => Bounds.transformerStackReal eps layers x q j) i)
      (fun i => (hln i).1) (fun i => (hln i).2)
  exact ⟨hloDot, hhiDot⟩

end Sound

end Nfp
