-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Core
import Nfp.Prob
import Nfp.Mixer
import Nfp.System

/-!
Top-level reexports and trust dashboard for the NFP rewrite.
-/

/-!
Axioms used by key definitions/lemmas.
These `#print axioms` lines help ensure we only depend on a small set of axioms
(ideally a subset of: `propext`, `Classical.choice`, `Quot.sound`).
-/

#print axioms Nfp.ProbVec.sum_mass
#print axioms Nfp.ProbVec.pure
#print axioms Nfp.ProbVec.mix
#print axioms Nfp.Mixer.push
#print axioms Nfp.Mixer.comp
#print axioms Nfp.Mixer.id
#print axioms Nfp.Dag.parents
#print axioms Nfp.LocalSystem.toMixer
