-- SPDX-License-Identifier: AGPL-3.0-or-later

module

import Nfp

/-!
Axioms used by key definitions/lemmas.
These `#print axioms` lines help ensure we only depend on a small set of axioms
(ideally a subset of: `propext`, `Classical.choice`, `Quot.sound`).
-/

public section

#print axioms Nfp.ProbVec.sum_mass
#print axioms Nfp.ProbVec.pure
#print axioms Nfp.ProbVec.mix
#print axioms Nfp.Mixer.push
#print axioms Nfp.Mixer.comp
#print axioms Nfp.Mixer.id
#print axioms Nfp.Dag.parents
#print axioms Nfp.LocalSystem.toMixer
#print axioms Nfp.LocalSystem.eval
#print axioms Nfp.LocalSystem.eval_eq
#print axioms Nfp.Circuit.eval
#print axioms Nfp.Circuit.evalInput
#print axioms Nfp.Circuit.Interface.eval
#print axioms Nfp.Circuit.checkEquiv
#print axioms Nfp.Circuit.checkEquivOnInterface

/-- Entrypoint for the axiom report build target. -/
def main : IO Unit := pure ()

end
