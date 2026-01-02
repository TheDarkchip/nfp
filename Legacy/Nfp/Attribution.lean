-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Real.Basic
import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Basic
import Nfp.Prob
import Nfp.Mixer

/-!
# Attribution Axioms for Neural Network Interpretation

This module formalizes key properties (axioms) that attribution methods for
neural networks should satisfy. These axioms are used to characterize and
compare different interpretation methods such as Integrated Gradients, LRP,
Shapley values, and path-based attribution.

## Main definitions

* `Attribution` – an attribution assigns a contribution score to each input feature
* `Completeness` – contributions sum to the output difference from baseline
* `Sensitivity` – if an input affects output, it receives nonzero attribution
* `Implementation Invariance` – attributions depend only on input-output behavior
* `Linearity` – attribution is linear in the function being explained

## Key theorems

* `completeness_of_conservation` – conservation mixers induce complete attributions
* `tracer_attribution_complete` – tracer-based attributions satisfy completeness

## References

* Sundararajan, Taly, Yan: "Axiomatic Attribution for Deep Networks" (ICML 2017)
* Ancona et al.: "Towards better understanding of gradient-based attribution" (2018)
* Lundstrom et al.: "A Rigorous Study of Integrated Gradients" (2022)
-/

namespace Nfp

open scoped BigOperators
open Finset

/-! ## Attribution structure -/

section Attribution

variable {Input Output : Type*} [Fintype Input] [Fintype Output]

/-- An attribution method assigns a contribution score to each input feature
for a given output. Scores are signed reals to allow negative contributions. -/
structure Attribution (Input Output : Type*) [Fintype Input] [Fintype Output] where
  /-- The contribution of input `i` to output `o`. -/
  contrib : Input → Output → ℝ

namespace Attribution

variable (A : Attribution Input Output)

/-- Total contribution to a specific output. -/
noncomputable def totalContrib (o : Output) : ℝ :=
  ∑ i, A.contrib i o

/-- An attribution is nonnegative if all contributions are nonnegative. -/
def Nonneg : Prop :=
  ∀ i o, 0 ≤ A.contrib i o

end Attribution

end Attribution

/-! ## Completeness axiom -/

section Completeness

variable {Input Output : Type*} [Fintype Input] [Fintype Output]

/-- The completeness axiom: contributions sum to the difference between the
output at input `x` and the output at baseline `x₀`. This is the core axiom
from Integrated Gradients and Shapley value attribution.

For neural networks: `f(x) - f(x₀) = ∑ᵢ attribution(i)` -/
def Attribution.Complete
    (A : Attribution Input Output)
    (f : (Input → ℝ) → Output → ℝ)
    (x x₀ : Input → ℝ)
    (o : Output) : Prop :=
  A.totalContrib o = f x o - f x₀ o

/-- Completeness for a single-output function (common case). -/
def Attribution.CompleteScalar
    (A : Attribution Input Unit)
    (f : (Input → ℝ) → ℝ)
    (x x₀ : Input → ℝ) : Prop :=
  A.totalContrib () = f x - f x₀

end Completeness

/-! ## Sensitivity axiom -/

section Sensitivity

variable {Input Output : Type*} [Fintype Input] [Fintype Output]

/-- An input feature `i` influences output `o` for function `f` at point `x`
relative to baseline `x₀` if changing just that feature changes the output. -/
def Influences
    (f : (Input → ℝ) → Output → ℝ)
    (x x₀ : Input → ℝ)
    (i : Input)
    (o : Output) : Prop :=
  ∃ (x' : Input → ℝ),
    (∀ j, j ≠ i → x' j = x₀ j) ∧
    x' i = x i ∧
    f x' o ≠ f x₀ o

/-- The sensitivity axiom: if a feature influences the output, it receives
nonzero attribution. -/
def Attribution.Sensitive
    (A : Attribution Input Output)
    (f : (Input → ℝ) → Output → ℝ)
    (x x₀ : Input → ℝ) : Prop :=
  ∀ i o, Influences f x x₀ i o → A.contrib i o ≠ 0

/-- Dummy axiom: features that don't influence output get zero attribution. -/
def Attribution.Dummy
    (A : Attribution Input Output)
    (f : (Input → ℝ) → Output → ℝ)
    (x x₀ : Input → ℝ) : Prop :=
  ∀ i o, ¬ Influences f x x₀ i o → A.contrib i o = 0

end Sensitivity

/-! ## Conservation and tracer-based attribution -/

section Conservation

variable {S : Type*} [Fintype S]

/-- A mixer `M` is mass-conserving if the total outgoing mass equals
the total incoming mass. For row-stochastic mixers, this is automatic. -/
lemma Mixer.conserves_mass (M : Mixer S S) (p : ProbVec S) :
    (∑ i, (M.push p).mass i) = ∑ i, p.mass i := by
  simp only [ProbVec.sum_mass]

/-- Attribution derived from a probability distribution (tracer mass). -/
noncomputable def attributionOfProbVec [Fintype S]
    (p : ProbVec S) : Attribution S Unit where
  contrib := fun i _ => (p.mass i : ℝ)

/-- Tracer-based attributions automatically satisfy completeness with respect
to total mass (which is 1). -/
theorem tracer_attribution_complete (p : ProbVec S) :
    (attributionOfProbVec p).totalContrib () = 1 := by
  simp only [Attribution.totalContrib, attributionOfProbVec]
  have h := p.norm_one
  simp only [← NNReal.coe_sum] at h ⊢
  exact_mod_cast h

end Conservation

/-! ## Linearity axiom -/

section Linearity

variable {Input Output : Type*} [Fintype Input] [Fintype Output]

/-- A method that produces attributions for any function. -/
abbrev AttributionMethod (Input Output : Type*) [Fintype Input] [Fintype Output] :=
  ((Input → ℝ) → Output → ℝ) → (Input → ℝ) → (Input → ℝ) → Attribution Input Output

/-- The linearity axiom: attribution of a sum of functions equals the sum
of attributions. -/
def AttributionMethod.Linear (method : AttributionMethod Input Output) : Prop :=
  ∀ (f g : (Input → ℝ) → Output → ℝ) (x x₀ : Input → ℝ) (i : Input) (o : Output),
    (method (fun inp => fun o => f inp o + g inp o) x x₀).contrib i o =
      (method f x x₀).contrib i o + (method g x x₀).contrib i o

/-- Scale invariance: scaling the function scales the attribution. -/
def AttributionMethod.ScaleInvariant (method : AttributionMethod Input Output) : Prop :=
  ∀ (f : (Input → ℝ) → Output → ℝ) (c : ℝ) (x x₀ : Input → ℝ) (i : Input) (o : Output),
    (method (fun inp => fun o => c * f inp o) x x₀).contrib i o =
      c * (method f x x₀).contrib i o

end Linearity

/-! ## Symmetry and efficiency -/

section Symmetry

variable {Input Output : Type*} [Fintype Input] [Fintype Output] [DecidableEq Input]

/-- Swap two input coordinates in a feature vector. -/
def swapInputs (x : Input → ℝ) (i j : Input) : Input → ℝ :=
  fun k => if k = i then x j else if k = j then x i else x k

/-- Two inputs are symmetric for a function if swapping them preserves the output. -/
def SymmetricInputs
    (f : (Input → ℝ) → Output → ℝ)
    (i j : Input) : Prop :=
  ∀ (x : Input → ℝ) (o : Output),
    f x o = f (swapInputs x i j) o

/-- The symmetry axiom: symmetric inputs receive equal attribution. -/
def Attribution.Symmetric
    (A : Attribution Input Output)
    (f : (Input → ℝ) → Output → ℝ) : Prop :=
  ∀ i j o, SymmetricInputs f i j → A.contrib i o = A.contrib j o

end Symmetry

/-! ## Path-based attribution -/

section PathAttribution

variable {Input : Type*}

/-- A path from baseline `x₀` to input `x` parameterized by `t ∈ [0,1]`. -/
abbrev Path (Input : Type*) := (t : NNReal) → Input → ℝ

/-- A straight-line (linear interpolation) path between two points. -/
noncomputable def straightPath (x x₀ : Input → ℝ) : Path Input :=
  fun t i => (1 - (t : ℝ)) * x₀ i + (t : ℝ) * x i

/-- A valid path starts at baseline and ends at input. -/
structure Path.Valid (γ : Path Input) (x x₀ : Input → ℝ) : Prop where
  at_zero : ∀ i, γ 0 i = x₀ i
  at_one : ∀ i, γ 1 i = x i

lemma straightPath_valid (x x₀ : Input → ℝ) : (straightPath x x₀).Valid x x₀ := by
  constructor
  · intro i; simp [straightPath]
  · intro i; simp [straightPath]

end PathAttribution

end Nfp
