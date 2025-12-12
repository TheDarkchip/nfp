import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Prod
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.DeriveFintype
import Nfp.Prob
import Nfp.Mixer

/-
Core influence specifications and helpers that sit below mixers.  Influence
specs carry raw, non-normalized edge capacities; `Mixer.ofInfluenceSpec` turns
them into row-stochastic mixers with a small fallback for empty rows.  This
file also introduces lightweight sign and hierarchy indices (`Chan`, `HSite`,
`BigSite`) together with convenience projections for tracers living on the
combined site.
-/

namespace Nfp

open scoped BigOperators
open Finset

/-! ## Influence specs -/

/-- A raw influence description: adjacency plus nonnegative edge capacities. -/
structure InfluenceSpec (Site : Type*) where
  adj : Site → Site → Prop
  κ : ∀ {s t}, adj s t → NNReal

/-- A family of influence specs indexed by different “views”. -/
structure InfluenceSpecFamily (Site : Type*) where
  View : Type*
  spec : View → InfluenceSpec Site

namespace InfluenceSpecFamily

variable {Site : Type*}

/-- Convexly combine multiple influence specs using nonnegative weights `α`.
Specs with zero weight do not contribute to the merged adjacency. -/
noncomputable def combine (F : InfluenceSpecFamily Site) [Fintype F.View]
    (α : F.View → NNReal) (_hα : ∑ v, α v = 1) : InfluenceSpec Site :=
  {
    adj := fun s t => ∃ v, α v ≠ 0 ∧ (F.spec v).adj s t
    κ := by
      intro s t _
      classical
      exact ∑ v, α v * (if h : (F.spec v).adj s t then (F.spec v).κ h else 0)
  }

end InfluenceSpecFamily

namespace InfluenceSpec

variable {Site : Type*} [Fintype Site] [DecidableEq Site]

/-- Sum of outgoing raw capacities from a site. -/
noncomputable def rowTotal (I : InfluenceSpec Site) (s : Site) : NNReal := by
  classical
  exact ∑ t, (if h : I.adj s t then I.κ h else 0)

/-! ### Row scaling helpers -/

/-- Scale all raw capacities in a single row of an influence spec. -/
noncomputable def scaleRow (I : InfluenceSpec Site) (s0 : Site) (c : NNReal) :
    InfluenceSpec Site :=
  {
    adj := I.adj
    κ := by
      intro s t h
      by_cases hs : s = s0
      · subst hs
        exact c * I.κ h
      · exact I.κ h
  }

lemma rowTotal_scaleRow_self (I : InfluenceSpec Site) (s0 : Site) (c : NNReal) :
    InfluenceSpec.rowTotal (Site := Site) (scaleRow (Site := Site) I s0 c) s0 =
      c * InfluenceSpec.rowTotal (Site := Site) I s0 := by
  classical
  have hrewrite :
      InfluenceSpec.rowTotal (Site := Site) (scaleRow (Site := Site) I s0 c) s0 =
        ∑ t : Site, c * (if h : I.adj s0 t then I.κ h else 0) := by
    simp [InfluenceSpec.rowTotal, scaleRow]
  calc
    InfluenceSpec.rowTotal (Site := Site) (scaleRow (Site := Site) I s0 c) s0 =
        ∑ t : Site, c * (if h : I.adj s0 t then I.κ h else 0) := hrewrite
    _ = c * InfluenceSpec.rowTotal (Site := Site) I s0 := by
          -- pull the scalar outside the sum
          simpa [InfluenceSpec.rowTotal] using
            (Finset.mul_sum (s := (Finset.univ : Finset Site))
              (f := fun t : Site => (if h : I.adj s0 t then I.κ h else 0))
              (a := c)).symm

lemma rowTotal_scaleRow_other (I : InfluenceSpec Site) {s s0 : Site} (c : NNReal)
    (hs : s ≠ s0) :
    InfluenceSpec.rowTotal (Site := Site) (scaleRow (Site := Site) I s0 c) s =
      InfluenceSpec.rowTotal (Site := Site) I s := by
  classical
  simp [InfluenceSpec.rowTotal, scaleRow, hs]

end InfluenceSpec

namespace InfluenceSpecFamily

variable {Site : Type*}

lemma combine_adj_iff (F : InfluenceSpecFamily Site) [Fintype F.View]
    (α : F.View → NNReal) (hα : ∑ v, α v = 1) (s t : Site) :
    (combine (F:=F) α hα).adj s t ↔ ∃ v, α v ≠ 0 ∧ (F.spec v).adj s t :=
  Iff.rfl

end InfluenceSpecFamily

/-! ## Mixers derived from influence specs -/

namespace Mixer

variable {Site : Type*} [Fintype Site] [DecidableEq Site]

/-- Canonical mixer obtained by normalizing each row of an influence spec.
If a row has zero total capacity, we fall back to an identity row on that site,
interpreting the site as an absorbing state. -/
noncomputable def ofInfluenceSpec (I : InfluenceSpec Site) : Mixer Site Site := by
  classical
  let Z : Site → NNReal := InfluenceSpec.rowTotal (Site := Site) I
  refine
    {
      w := fun s t =>
        if hZ : Z s = 0 then
          if hst : s = t then 1 else 0
        else
          if hAdj : I.adj s t then I.κ hAdj / Z s else 0
      row_sum_one := by
        intro s
        by_cases hZ : Z s = 0
        · have hdiag :
            (∑ t : Site, (if s = t then (1 : NNReal) else 0)) = 1 := by
            classical
            simp
          simp [Z, hZ, hdiag]
        · have hZne : Z s ≠ 0 := hZ
          have hnormalize :
              (∑ t : Site, (if h : I.adj s t then I.κ h / Z s else 0)) =
                (∑ t : Site, (if h : I.adj s t then I.κ h else 0)) * (1 / Z s) := by
            classical
            have hrewrite :
                (∑ t : Site, (if h : I.adj s t then I.κ h / Z s else 0)) =
                  (∑ t : Site, (if h : I.adj s t then I.κ h else 0) * (1 / Z s)) := by
              refine Finset.sum_congr rfl ?_
              intro t _ht
              by_cases hAdj : I.adj s t <;> simp [hAdj, div_eq_mul_inv, mul_comm]
            have hfactor :
                (∑ t : Site, (if h : I.adj s t then I.κ h else 0) * (1 / Z s)) =
                  (∑ t : Site, (if h : I.adj s t then I.κ h else 0)) * (1 / Z s) := by
              simpa using
                (Finset.sum_mul (s := (Finset.univ : Finset Site))
                  (f := fun t : Site => if h : I.adj s t then I.κ h else 0)
                  (a := (1 / Z s))).symm
            exact hrewrite.trans hfactor
          have hrow : (∑ t : Site, (if h : I.adj s t then I.κ h else 0)) = Z s := rfl
          have hnormalized :
              (∑ t : Site, (if h : I.adj s t then I.κ h / Z s else 0)) = 1 := by
            have hdiv :
                (∑ t : Site, (if h : I.adj s t then I.κ h else 0)) * (1 / Z s) =
                  (1 : NNReal) := by
              simp [hrow, div_eq_mul_inv, hZne]
            exact hnormalize.trans hdiv
          simpa [Z, hZ] using hnormalized
    }

lemma ofInfluenceSpec_zero_of_not_adj (I : InfluenceSpec Site)
    {s t : Site} (hZ : InfluenceSpec.rowTotal (Site := Site) I s ≠ 0)
    (hAdj : ¬ I.adj s t) :
    (ofInfluenceSpec (Site := Site) I).w s t = 0 := by
  classical
  simp [ofInfluenceSpec, hZ, hAdj]

lemma ofInfluenceSpec_adj_weight (I : InfluenceSpec Site)
    {s t : Site} (hZ : InfluenceSpec.rowTotal (Site := Site) I s ≠ 0)
    (hAdj : I.adj s t) :
    (ofInfluenceSpec (Site := Site) I).w s t =
      I.κ hAdj / InfluenceSpec.rowTotal (Site := Site) I s := by
  classical
  simp [ofInfluenceSpec, hZ, hAdj]

lemma ofInfluenceSpec_supported (I : InfluenceSpec Site)
    (hZ : ∀ s, InfluenceSpec.rowTotal (Site := Site) I s ≠ 0) :
    Mixer.supported (S := Site) (T := Site) (ofInfluenceSpec (Site := Site) I) I.adj := by
  intro s t hAdj
  exact ofInfluenceSpec_zero_of_not_adj (Site := Site) (I:=I) (hZ s) hAdj

lemma ofInfluenceSpec_row_scaling (I : InfluenceSpec Site) (s0 : Site) {c : NNReal}
    (hc : c ≠ 0) (t : Site) :
    (ofInfluenceSpec (Site := Site) (InfluenceSpec.scaleRow (Site := Site) I s0 c)).w s0 t =
      (ofInfluenceSpec (Site := Site) I).w s0 t := by
  classical
  by_cases hZ : InfluenceSpec.rowTotal (Site := Site) I s0 = 0
  · have hZ' :
      InfluenceSpec.rowTotal (Site := Site)
          (InfluenceSpec.scaleRow (Site := Site) I s0 c) s0 = 0 := by
      simp [InfluenceSpec.rowTotal_scaleRow_self (I:=I) (s0:=s0) (c:=c), hZ]
    simp [ofInfluenceSpec, hZ, hZ']
  · have hrow :
      InfluenceSpec.rowTotal (Site := Site) (InfluenceSpec.scaleRow (Site := Site) I s0 c)
          s0 = c * InfluenceSpec.rowTotal (Site := Site) I s0 :=
      InfluenceSpec.rowTotal_scaleRow_self (I:=I) (s0:=s0) (c:=c)
    have hZ' :
      InfluenceSpec.rowTotal (Site := Site)
          (InfluenceSpec.scaleRow (Site := Site) I s0 c) s0 ≠ 0 := by
      have hne : c * InfluenceSpec.rowTotal (Site := Site) I s0 ≠ 0 :=
        mul_ne_zero hc hZ
      simpa [hrow] using hne
    by_cases hAdj : I.adj s0 t
    · have hκ :
        (InfluenceSpec.scaleRow (Site := Site) I s0 c).κ hAdj = c * I.κ hAdj := by
        simp [InfluenceSpec.scaleRow]
      have hleft :
          (ofInfluenceSpec (Site := Site) (InfluenceSpec.scaleRow (Site := Site) I s0 c)).w s0 t =
            (InfluenceSpec.scaleRow (Site := Site) I s0 c).κ hAdj /
              InfluenceSpec.rowTotal (Site := Site)
                (InfluenceSpec.scaleRow (Site := Site) I s0 c) s0 :=
        ofInfluenceSpec_adj_weight (I := InfluenceSpec.scaleRow (Site := Site) I s0 c) hZ' hAdj
      have hright :
          (ofInfluenceSpec (Site := Site) I).w s0 t =
            I.κ hAdj / InfluenceSpec.rowTotal (Site := Site) I s0 :=
        ofInfluenceSpec_adj_weight (I := I) hZ hAdj
      have hcancel :
          (InfluenceSpec.scaleRow (Site := Site) I s0 c).κ hAdj /
              InfluenceSpec.rowTotal
                (Site := Site) (InfluenceSpec.scaleRow (Site := Site) I s0 c) s0 =
            c * I.κ hAdj / (c * InfluenceSpec.rowTotal (Site := Site) I s0) := by
        simp [hκ, hrow]
      calc
        (ofInfluenceSpec (Site := Site) (InfluenceSpec.scaleRow (Site := Site) I s0 c)).w s0 t =
            c * I.κ hAdj / (c * InfluenceSpec.rowTotal (Site := Site) I s0) := by
              simp [hcancel] at hleft
              simpa using hleft
        _ = I.κ hAdj / InfluenceSpec.rowTotal (Site := Site) I s0 := by
              have hc' : (c : NNReal) ≠ 0 := hc
              simpa [mul_comm, mul_left_comm, mul_assoc] using
                (mul_div_mul_left (a := I.κ hAdj)
                  (b := InfluenceSpec.rowTotal (Site := Site) I s0)
                  (c := c) (hc := hc'))
        _ = (ofInfluenceSpec (Site := Site) I).w s0 t := hright.symm
    · have hleft :
        (ofInfluenceSpec (Site := Site) (InfluenceSpec.scaleRow (Site := Site) I s0 c)).w s0 t =
          0 :=
        ofInfluenceSpec_zero_of_not_adj
          (I := InfluenceSpec.scaleRow (Site := Site) I s0 c) hZ' hAdj
      have hright :
        (ofInfluenceSpec (Site := Site) I).w s0 t = 0 :=
        ofInfluenceSpec_zero_of_not_adj (I := I) hZ hAdj
      simp [hleft, hright]

end Mixer

/-! ## Sign channels and hierarchical sites -/

/-- Explicit sign channels; sign is tracked structurally, not by negative mass. -/
inductive Chan
  | pos
  | neg
deriving DecidableEq, Fintype

/-- Hierarchical site: either a base node or a group-level aggregate. -/
inductive HSite (Base Group : Type*) : Type _
  | base : Base → HSite Base Group
  | group : Group → HSite Base Group
deriving DecidableEq

namespace HSite

variable {Base Group : Type*}

/-- An equivalence to a sum type, used to reuse existing finite instances. -/
def equivSum : HSite Base Group ≃ Sum Base Group :=
  {
    toFun := fun
      | base b => Sum.inl b
      | group g => Sum.inr g
    invFun := fun
      | Sum.inl b => base b
      | Sum.inr g => group g
    left_inv := by
      intro x
      cases x <;> rfl
    right_inv := by
      intro x
      cases x <;> rfl
  }

instance [Fintype Base] [Fintype Group] : Fintype (HSite Base Group) :=
  Fintype.ofEquiv (Sum Base Group) (equivSum (Base:=Base) (Group:=Group)).symm

end HSite

/-- Combined site carrying an objective, a base node with sign, or a group. -/
abbrev BigSite (Obj Node Group : Type*) :=
  Obj × HSite (Node × Chan) Group

/-! ## Tracer view helpers on `BigSite` -/

namespace TracerViews

section

variable {Obj Node Group : Type*} [Fintype Obj] [Fintype Node] [Fintype Group]
variable (p : ProbVec (BigSite Obj Node Group))

/-- Positive-channel tracer mass for a specific objective/node pair. -/
noncomputable def posTracer (o : Obj) (n : Node) : NNReal :=
  p.mass (o, HSite.base (n, Chan.pos))

/-- Negative-channel tracer mass for a specific objective/node pair. -/
noncomputable def negTracer (o : Obj) (n : Node) : NNReal :=
  p.mass (o, HSite.base (n, Chan.neg))

/-- Net tracer mass (`pos - neg`) for a specific objective/node pair. -/
noncomputable def netTracer (o : Obj) (n : Node) : ℝ :=
  (posTracer (p:=p) o n : ℝ) - (negTracer (p:=p) o n : ℝ)

/-- Group-level tracer mass for an objective and group. -/
noncomputable def groupTracer (o : Obj) (g : Group) : NNReal :=
  p.mass (o, HSite.group g)

@[simp] lemma posTracer_eval (o : Obj) (n : Node) :
    posTracer (p:=p) o n = p.mass (o, HSite.base (n, Chan.pos)) := by
  rfl

@[simp] lemma negTracer_eval (o : Obj) (n : Node) :
    negTracer (p:=p) o n = p.mass (o, HSite.base (n, Chan.neg)) := by
  rfl

@[simp] lemma groupTracer_eval (o : Obj) (g : Group) :
    groupTracer (p:=p) o g = p.mass (o, HSite.group g) := by
  rfl

end

end TracerViews

open TracerViews (posTracer negTracer netTracer groupTracer)
export TracerViews (posTracer negTracer netTracer groupTracer)

end Nfp
