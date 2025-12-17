-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Prob
import Nfp.PCC
import Nfp.Uniqueness
import Nfp.Appendix
import Nfp.Mixer
import Nfp.Reroute.Heat
import Nfp.Reroute.Partition
import Nfp.Influence
import Nfp.MixerLocalSystem
import Nfp.Attribution
import Nfp.Layers
import Nfp.SignedMixer
import Nfp.Linearization
import Nfp.Abstraction
import Nfp.Induction
import Nfp.Discovery
import Nfp.Verification
import Nfp.IO
import Nfp.Sound.IO
import Nfp.Sound.Demo

/-!
Axioms used by key theorems/definitions
These `#print axioms` lines help ensure we only depend on a small set of axioms
(ideally a subset of: `propext`, `Classical.choice`, `Quot.sound`).
-/

-- From Nfp.Prob
#print axioms Nfp.ProbVec.sum_mass
#print axioms Nfp.ProbVec.pure
#print axioms Nfp.ProbVec.mix

-- From Nfp.PCC
#print axioms Nfp.tracerOfContrib
#print axioms Nfp.tracerOfContrib_mass
#print axioms Nfp.sum_monotone_chain
#print axioms Nfp.monotone_removed_mass
-- From Nfp.Uniqueness
#print axioms Nfp.LocalSystem.tracer_unique
-- From Nfp.Appendix (PCC-focused lemmas)
#print axioms Nfp.drop_eq_mass
#print axioms Nfp.pcc_upper_bound_for_any_mask
#print axioms Nfp.normalized_sum_monotone
-- Residuals
#print axioms Nfp.lambdaEC
#print axioms Nfp.lambdaEC_sum_one
#print axioms Nfp.residual_lambda_from_norm
-- Greedy optimality
#print axioms Nfp.greedy_topk_optimal
-- Budgeted PCC sup
#print axioms Nfp.normMass
#print axioms Nfp.normMass_union
#print axioms Nfp.normMass_partitionUnion
#print axioms Nfp.normMass_partition_eq_one
#print axioms Nfp.ReroutePlan.masks
#print axioms Nfp.ReroutePlan.normMass_sum_one
#print axioms Nfp.unionParts
#print axioms Nfp.disjoint_unionParts
#print axioms Nfp.feasible
#print axioms Nfp.pccArg
#print axioms Nfp.pccMax
#print axioms Nfp.pccMax_le_tau
#print axioms Nfp.pccMax_monotone
#print axioms Nfp.pccMax_dominates
#print axioms Nfp.greedy_topmass_optimal

-- PCC(t) alias and properties
#print axioms Nfp.PCC_monotone
#print axioms Nfp.PCC_upper_bounds_masks

-- Normalization utilities and uniqueness
#print axioms Nfp.normalizeOn_outside
#print axioms Nfp.normalizeOn_inside
#print axioms Nfp.normalizeOn_sum_one
#print axioms Nfp.proportional_row_unique
-- Residual characterization (Appendix A.3)
#print axioms Nfp.lambdaEC_scale_invariant_global

-- Appendix A.1 consolidated wrappers and packaged theorem
#print axioms Nfp.A1_residual_unique
#print axioms Nfp.A1_normalize_unique
#print axioms Nfp.A1_global_tracer_unique
#print axioms Nfp.A1

-- Forward mixers (Appendix A.1)
#print axioms Nfp.Mixer.push
#print axioms Nfp.Mixer.row
#print axioms Nfp.Mixer.push_pure
#print axioms Nfp.Mixer.push_mix
#print axioms Nfp.Mixer.comp

-- Appendix A.2 wrappers and packaged theorem
#print axioms Nfp.A2_graph_faithful_comp
#print axioms Nfp.A2_residual_energy_consistent
#print axioms Nfp.A2_residual_unique
#print axioms Nfp.A2_normalize_row_sum_one
#print axioms Nfp.A2

-- From Nfp.Mixer (support restriction)
#print axioms Nfp.Mixer.supported_comp
#print axioms Nfp.Mixer.supp_push_subset_image

-- From Nfp.Influence
#print axioms Nfp.Mixer.ofInfluenceSpec
#print axioms Nfp.Mixer.ofInfluenceSpec_supported

-- From Nfp.MixerLocalSystem
#print axioms Nfp.LocalSystem.ofMixerIdx
#print axioms Nfp.LocalSystem.ofMixer

-- From Nfp.Reroute.Partition
#print axioms Nfp.ReroutePlan.increments
#print axioms Nfp.ReroutePlan.increments_pairwise
#print axioms Nfp.sum_unionParts_eq

-- From Nfp.Reroute.Heat
#print axioms Nfp.WeightedReroutePlan.rerouteHeat
#print axioms Nfp.WeightedReroutePlan.heatRaw_sum_increment
#print axioms Nfp.WeightedReroutePlan.rerouteHeat_sum_increment

-- From Nfp.Attribution (attribution axioms for NN interpretation)
#print axioms Nfp.Attribution.Complete
#print axioms Nfp.Attribution.CompleteScalar
#print axioms Nfp.Attribution.Sensitive
#print axioms Nfp.Attribution.Dummy
#print axioms Nfp.Attribution.Symmetric
#print axioms Nfp.attributionOfProbVec
#print axioms Nfp.tracer_attribution_complete

-- From Nfp.Layers (NN layer mixers)
#print axioms Nfp.Mixer.identity
#print axioms Nfp.Mixer.identity_comp
#print axioms Nfp.Mixer.comp_identity
#print axioms Nfp.Mixer.comp_assoc
#print axioms Nfp.Mixer.attention
#print axioms Nfp.Mixer.selfAttention
#print axioms Nfp.Mixer.residual
#print axioms Nfp.Mixer.residual_one
#print axioms Nfp.Mixer.residual_zero
#print axioms Nfp.Mixer.comp3
#print axioms Nfp.Mixer.comp3_path_decomposition
#print axioms Nfp.Mixer.transformerBlock
#print axioms Nfp.attentionFlow
#print axioms Nfp.attentionFlow_singleton
#print axioms Nfp.attentionFlow_nil
#print axioms Nfp.effectiveAttention_normalized
#print axioms Nfp.pathContrib_sum
#print axioms Nfp.Mixer.push_preserves_total_mass
#print axioms Nfp.Mixer.push_comp
#print axioms Nfp.transformer_attribution_unique

-- Residual stream analysis
#print axioms Nfp.Mixer.residual_decomposition
#print axioms Nfp.Mixer.residual_skip_dominance
#print axioms Nfp.Mixer.residual_off_diag_bound
#print axioms Nfp.Mixer.residual_off_diag_scaling

-- Attention concentration
#print axioms Nfp.Mixer.maxWeight
#print axioms Nfp.Mixer.weight_le_maxWeight
#print axioms Nfp.Mixer.push_concentration_bound

-- Ablation analysis
#print axioms Nfp.Mixer.maskFn
#print axioms Nfp.Mixer.maskFn_blocked
#print axioms Nfp.Mixer.maskFn_unblocked
#print axioms Nfp.blockedContribution
#print axioms Nfp.unblockedContribution
#print axioms Nfp.Mixer.ablation_decomposition

-- Composition depth and reachability
#print axioms Nfp.Mixer.reachable
#print axioms Nfp.Mixer.reach_comp
#print axioms Nfp.Mixer.path_contrib_le_comp
#print axioms Nfp.Mixer.comp_reachable_of_path

-- Information bounds
#print axioms Nfp.Mixer.supportSize
#print axioms Nfp.Mixer.exists_nonzero
#print axioms Nfp.Mixer.supportSize_pos

-- Gradient correspondence (chain rule)
#print axioms Nfp.Mixer.chain_rule_analog
#print axioms Nfp.Mixer.chain_rule_three

-- Multi-head attention
#print axioms Nfp.Mixer.multiHead
#print axioms Nfp.Mixer.multiHead_head_contrib_bound
#print axioms Nfp.Mixer.multiHead_zero_weight
#print axioms Nfp.Mixer.multiHead_single_head
#print axioms Nfp.Mixer.multiHead_convex

-- Causal (autoregressive) attention
#print axioms Nfp.isCausal
#print axioms Nfp.causal_reachable_dir
#print axioms Nfp.causal_comp
#print axioms Nfp.causal_future_attention_zero
#print axioms Nfp.causal_past_attention_one
#print axioms Nfp.causal_first_token_self

-- Attention head analysis
#print axioms Nfp.Mixer.attentionConcentration
#print axioms Nfp.Mixer.attentionConcentration_upper_bound
#print axioms Nfp.Mixer.isSparseAt
#print axioms Nfp.Mixer.isDiffuseAt
#print axioms Nfp.Mixer.attentionConcentration_one_hot

-- Residual dominance analysis
#print axioms Nfp.residual_diagonal_lower
#print axioms Nfp.residual_offdiag_scale
#print axioms Nfp.residual_offdiag_sum_bound

-- Deep composition analysis
#print axioms Nfp.comp_weight_le_one
#print axioms Nfp.comp_term_bound

-- Cross-attention for encoder-decoder models
#print axioms Nfp.Mixer.crossAttention
#print axioms Nfp.Mixer.crossAttention_normalized

-- Layer-wise attribution analysis
#print axioms Nfp.layerWiseAttribution
#print axioms Nfp.layerWiseAttribution_nil
#print axioms Nfp.layerWiseAttribution_singleton
#print axioms Nfp.layerWiseAttribution_le_one
#print axioms Nfp.layerWiseAttribution_sum_one

-- From Nfp.SignedMixer (generalized signed weight mixers)
#print axioms Nfp.SignedMixer
#print axioms Nfp.SignedMixer.comp
#print axioms Nfp.SignedMixer.comp_assoc
#print axioms Nfp.SignedMixer.identity
#print axioms Nfp.SignedMixer.positivePart
#print axioms Nfp.SignedMixer.negativePart
#print axioms Nfp.SignedMixer.decompose
#print axioms Nfp.SignedMixer.rowSum
#print axioms Nfp.SignedMixer.IsRowStochastic
#print axioms Nfp.SignedMixer.toMixer
#print axioms Nfp.SignedMixer.ofMixer
#print axioms Nfp.SignedMixer.ofMixer_isRowStochastic
#print axioms Nfp.SignedMixer.influence
#print axioms Nfp.SignedMixer.totalInfluenceFrom
#print axioms Nfp.SignedMixer.totalInfluenceOn
#print axioms Nfp.SignedMixer.apply
#print axioms Nfp.SignedMixer.apply_comp
#print axioms Nfp.SignedMixer.jacobianEntry_eq_weight

-- AffineMixer (linear + bias)
#print axioms Nfp.AffineMixer
#print axioms Nfp.AffineMixer.linear
#print axioms Nfp.AffineMixer.bias
#print axioms Nfp.AffineMixer.apply
#print axioms Nfp.AffineMixer.comp

-- From Nfp.Linearization (linearizing non-linear ops)
#print axioms Nfp.Linearization
#print axioms Nfp.Linearization.comp
#print axioms Nfp.relu
#print axioms Nfp.reluGrad
#print axioms Nfp.reluLinearization
#print axioms Nfp.reluLinearization_diagonal
#print axioms Nfp.reluLinearization_diag_binary
#print axioms Nfp.gelu
#print axioms Nfp.geluGrad
#print axioms Nfp.geluLinearization
#print axioms Nfp.layerNorm
#print axioms Nfp.layerNormJacobian
#print axioms Nfp.layerNorm_translation_invariant
#print axioms Nfp.softmax
#print axioms Nfp.softmaxJacobian
#print axioms Nfp.softmax_nonneg
#print axioms Nfp.softmax_sum_one
#print axioms Nfp.softmaxJacobian_diag_pos
#print axioms Nfp.softmaxJacobian_off_diag_neg
#print axioms Nfp.softmax_translation_invariant
#print axioms Nfp.ropeJacobian
#print axioms Nfp.rope
#print axioms Nfp.ropeJacobian_cross_pos
#print axioms Nfp.rope_operatorNormBound_le_two
#print axioms Nfp.gradientTimesInput
#print axioms Nfp.gradientTimesInput_complete
#print axioms Nfp.composed_attribution
#print axioms Nfp.integratedGradientsLinear
#print axioms Nfp.integratedGradients_linear_complete
