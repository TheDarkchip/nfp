This repo is a Lean 4 formalization of an `Nfp`-style system:
finite probability on finite types, row-stochastic "mixers", influence specifications, and uniqueness results on DAGs.

You are an automated agent extending **a math library**, not an app.

The design is allowed to evolve. If a current abstraction is insufferable, you may refactor it — but keep the core philosophy and invariants.

---

## 1. Core Philosophy

- **Finite + nonnegative by default**
  - Use `NNReal` for masses, capacities, probabilities.
  - Work over finite types (`[Fintype ι]`) whenever reasonable.

- **No fake proofs, no new axioms**
  - Do **not** introduce `sorry`.
  - Do **not** introduce new nontrivial axioms beyond what `mathlib` already uses.

- **Proofs should be readable**
  - Prefer `simp`, `rw`, small `calc` blocks, and light `aesop` usage with helper lemmas.

- **Linting is sacred**
  - Never disable linters globally or locally.
  - In particular, options like
    `set_option linter.unnecessarySimpa false`
    or any `set_option linter.* false`
    are **forbidden**.
  - Fix or rewrite code to satisfy linters instead.

- **Build must be clean (warnings = errors)**
  - `lake build -q --wfail` **must succeed**.
  - Any compiler warning is treated as an error and must be resolved, not ignored.

---

## 2. Module Roles (Where Things Live)

Use this as a *default* structure, not a prison. You may reshuffle if a better design emerges, but document it.

> **Always keep this section up to date.**  
> If you add/remove/rename/repurpose modules, update this list in the **same commit** so it stays an accurate map of the codebase.

- **`Prob.lean`**  
  Probability vectors (`ProbVec`) on finite types: nonnegative mass functions summing to 1.

- **`Mixer.lean`**  
  Row-stochastic operators ("mixers"), their composition (`comp`), pushforward (`push`),
  and support restriction (`supported`, `compSupport`).

- **`Reroute/Partition.lean`**  
  Partitions of finite types, `unionParts`, disjointness helpers, and `ReroutePlan` structures
  with incremental mask computation (`increments`).

- **`Reroute/Heat.lean`**  
  Weighted reroute plans (`WeightedReroutePlan`) and the induced "heat" probability vector
  (`rerouteHeat`), distributing weights uniformly over each incremental mask block.

- **`Influence.lean`**  
  Influence specifications (`InfluenceSpec`, `InfluenceSpecFamily`), edge capacities, row scaling,
  and conversion to mixers via `Mixer.ofInfluenceSpec`. Also defines sign channels (`Chan`),
  hierarchical sites (`HSite`), and tracer view helpers (`posTracer`, `negTracer`, etc.).

- **`PCC.lean`**  
  Contribution/tracer utilities (`tracerOfContrib`), monotonicity helpers for mask chains,
  and discrete AUC interval machinery (`AInterval`, `AUC`, `intervalsFromWeights`).

- **`Uniqueness.lean`**  
  `LocalSystem` structure for finite DAG mixing systems; the core uniqueness theorem
  `tracer_unique` proving that tracer families satisfying the same recurrence must coincide.

- **`MixerLocalSystem.lean`**  
  Bridge from mixers on DAGs to `LocalSystem`: `ofMixer` and `ofMixerIdx` interpreters
  that convert a mixer with a topological ordering into a `LocalSystem`.

- **`Appendix.lean`**  
  Appendix-style results: normalized mass (`normMass`), feasibility (`feasible`), PCC curve
  (`pccArg`, `pccMax`, `PCC`), residual coefficients (`lambdaEC`), normalization (`normalizeOn`),
  uniqueness wrappers (`A1`, `A2`), and greedy optimality lemmas.

- **`Attribution.lean`**  
  Attribution axioms for neural network interpretation: `Complete` (contributions sum to output),
  `Sensitive` (influential features get nonzero attribution), `Dummy` (non-influential features
  get zero), `Symmetric` (symmetric inputs receive equal attribution). Bridges tracer-based
  attributions to classical interpretability axioms via `tracer_attribution_complete`.

- **`Layers.lean`**  
  Neural network layer operations formalized as mixers with comprehensive theorems for interpretation:
  - **Layer operations**: `identity`, `attention`, `selfAttention`, `residual`, `transformerBlock`
  - **Algebraic laws**: `identity_comp`, `comp_identity`, `comp_assoc` (mixer forms a monoid)
  - **Attention flow**: `attentionFlow` (attention rollout), `effectiveAttention_normalized`
  - **Path decomposition**: `comp_path_decomposition`, `comp3_path_decomposition`, `pathContrib_sum`
  - **Conservation**: `push_preserves_total_mass`, `push_comp` (mass conservation under composition)
  - **Residual analysis**: `residual_decomposition`, `residual_skip_dominance`, `residual_off_diag_scaling`
    (quantifies how skip connections "protect" self-information)
  - **Attention concentration**: `maxWeight`, `push_concentration_bound` (bounds on information spread)
  - **Ablation**: `maskFn`, `ablation_decomposition` (formalizes causal intervention/masking)
  - **Reachability**: `reachable`, `reach_comp`, `comp_reachable_of_path` (how influence spreads)
  - **Information bounds**: `supportSize`, `supportSize_pos`, `exists_nonzero` (attention sparsity)
  - **Gradient correspondence**: `chain_rule_analog`, `chain_rule_three` (discrete chain rule)
  - **Uniqueness bridge**: `transformer_attribution_unique` connects to `LocalSystem.tracer_unique`
  - **Multi-head attention**: `multiHead`, `multiHead_head_contrib_bound`, `multiHead_single_head`
    (how individual attention heads combine, single-head dominance)
  - **Causal (autoregressive) attention**: `isCausal`, `causal_comp`, `causal_future_attention_zero`,
    `causal_first_token_self` (GPT-style decoder-only attention constraints)
  - **Attention head analysis**: `attentionConcentration`, `isSparseAt`, `isDiffuseAt`,
    `attentionConcentration_one_hot` (measuring and classifying attention patterns)
  - **Residual dominance**: `residual_diagonal_lower`, `residual_offdiag_scale`,
    `residual_offdiag_sum_bound` (quantifying skip connection vs attention strength)
  - **Deep composition**: `comp_weight_le_one`, `comp_term_bound` (bounds on composed attention)
  - **Cross-attention**: `crossAttention`, `crossAttention_normalized` (encoder-decoder models)
  - **Layer-wise attribution**: `layerWiseAttribution`, `layerWiseAttribution_sum_one`
    (tracking attribution through layer stacks with conservation)
  
  The core insights for practitioners:
  1. Attention-based attribution is uniquely determined by attention patterns + boundary conditions
  2. Residual connections quantifiably protect self-information (diagonal ≥ c, off-diagonal ≤ (1-c)×original)
  3. Ablation effects decompose cleanly into blocked vs unblocked contributions
  4. Composition creates compounding reachability through nonzero paths
  5. Causal attention forces information to flow only backward in sequence position
  6. Multi-head attention is a convex combination; zero-weight heads contribute nothing
  7. First token in causal models has self-attention weight 1 (cannot attend elsewhere)

- **`SignedMixer.lean`**  
  Generalization of mixers to support **signed (real) weights** for modeling full neural network layers:
  - **Core structure**: `SignedMixer` with `w : S → T → ℝ` (allows negative weights)
  - **Affine maps**: `AffineMixer` for y = Wx + b (linear + bias)
  - **Algebraic structure**: `Zero`, `Add`, `Neg`, `Sub`, `SMul` instances (vector space structure)
  - **Decomposition**: `positivePart`, `negativePart` decompose M = M⁺ - M⁻
  - **Row properties**: `rowSum`, `IsRowStochastic`, `rowAbsSum`, `totalInfluence`
  - **Conversions**: `toMixer`, `ofMixer` (embed/extract nonnegative row-stochastic matrices)
  - **Application**: `apply` applies mixer to real-valued input vectors, with `apply_comp`
  - **Influence metrics**: `influence` (|w_{ij}|), `influenceSign`, `totalInfluenceFrom/On`
  - **Gradient connection**: `gradient_is_weight` (∂(Mx)_j/∂x_i = M.w i j)
  - **AffineMixer composition**: `AffineMixer.comp` for (W₂(W₁x + b₁) + b₂)
  
  Key insight: This bridges the gap between the row-stochastic `Mixer` (suitable for attention)
  and real neural network layers (MLPs, value projections) that can have:
  - Negative weights (inhibitory connections)
  - Non-unit row sums (scaling effects)
  - Biases (affine transformations)

- **`Linearization.lean`**  
  Linearization of non-linear operations for real neural network interpretation:
  - **Core structure**: `Linearization` record of (input, output, Jacobian as SignedMixer)
  - **ReLU**: `reluLinearization` - diagonal 0/1 Jacobian, `reluLinearization_exact` shows
    piecewise-linear networks have exact local linearizations
  - **GeLU**: `geluLinearization` - diagonal Jacobian with `geluGrad` entries
  - **LayerNorm**: `layerNormJacobian` - **dense** Jacobian (every output depends on every input!),
    `layerNorm_translation_invariant` proves translation invariance
  - **Softmax**: `softmaxJacobian` - diagonal positive, off-diagonal negative (`softmaxJacobian_off_diag_neg`),
    `softmax_translation_invariant`, `softmax_sum_one`
  - **Attribution**: `gradientTimesInput` (Gradient × Input), `gradientTimesInput_complete`
  - **Composition**: `Linearization.comp` implements chain rule, `composed_attribution`
  - **Integrated Gradients**: `integratedGradientsLinear`, `integratedGradients_linear_complete`
  - **Attention Jacobian Decomposition**:
    - `FullAttentionLayer`: W_Q, W_K, W_V, W_O projection matrices
    - `AttentionForwardState`: all intermediate values (queries, keys, values, scores, weights)
    - `AttentionLinearization`: extended structure with fullJacobian
    - `valueTerm`: A_{qk} · (W_V · W_O) - the "Attention Rollout" component
    - `patternTerm`: fullJacobian - valueTerm - gradient through attention patterns
    - `attention_jacobian_decomposition`: fullJacobian = valueTerm + patternTerm
    - `attentionApproximationError`, `valueTermNorm`, `relativeApproximationError`
    - `isAttentionRolloutFaithful`: criterion for when attention rollout is valid
    - `positionValueTerm_proportional_to_attention`: position-collapsed value term ∝ attention
  - **Transformer Layer Structure**:
    - `TransformerLayerLinearization`: attention + MLP with residual connections
    - `transformer_layer_jacobian_structure`: (I+A)·(I+M) = I + A + M + A·M
  - **Deep Linearization (Certified Virtual Attention)**:
    - `DeepLinearization`: Multi-layer transformer structure (numLayers, per-layer AttentionLinearization,
      mlpJacobians, composedJacobian)
    - `VirtualHead`: Composition of valueTerm across layers (e.g., induction heads)
    - `PositionVirtualHead`: Position-collapsed virtual attention flow
    - `DeepValueTerm`: "Attention Rollout" composition through deep networks
    - `DeepPatternTerm`: composedJacobian - DeepValueTerm (approximation error)
    - `frobeniusNorm`, `operatorNormBound`: Error metrics for matrices
    - `layerError`, `totalLayerError`: Per-layer and accumulated error
    - `isLayerFaithful`, `isDeepFaithful`: ε-faithfulness predicates
    - `amplificationFactor`: Product of (1 + ‖layer Jacobian‖) for error propagation
    - `deep_jacobian_decomposition`: composedJacobian = DeepValueTerm + DeepPatternTerm
    - `deep_error_bounded_by_layer_errors`: Existence of finite error bound
    - `two_layer_faithfulness_composition`: ε₁-faithful + ε₂-faithful → (C·ε₁ + C·ε₂ + ε₁·ε₂)-faithful
  - **N-Layer Faithfulness Composition**:
    - `layerNormBounds`, `layerFaithfulness`: Per-layer operator norms and pattern term bounds
    - `suffixAmplification`: ∏_{j≥start} (1 + Cⱼ) - how errors get amplified by subsequent layers
    - `totalAmplifiedError`: Σᵢ εᵢ · suffixAmplification(i+1) - the full error formula
    - `n_layer_faithfulness_composition`: **Central theorem** - if each layer i is εᵢ-faithful with
      norm bound Cᵢ, then ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ)
    - `n_layer_uniform_bound`: Simplified bound ε · L · (1+C)^{L-1} for uniform constants
    - `n_layer_geometric_bound`: Geometric series bound ε · ((1+C)^L - 1) / C
    - `n_layer_zero_norm_bound`: Linear bound ε · L when amplification is 1
    - `totalLayerError_eq_n_layer_no_amplification`: Connection to simple error sum
    - `suffixAmplification_nonneg`, `totalAmplifiedError_nonneg`: Nonnegativity lemmas
    - `isCertifiedVirtualHead`, `virtual_head_certification`: Certified approximation bounds
    - `InductionHeadPattern`: Formalization of induction head mechanism
    - `induction_head_certified`: Induction patterns valid within error bounds
    - `isInterpretabilityIllusion`, `isGenuineMechanism`: Pattern vs value dominance
    - `mechanism_trichotomy`: Every mechanism is genuine OR illusion (decidable)
    - `isDeepGenuineMechanism`, `deep_genuine_implies_bounded`: Deep network certification
  
  Key insights:
  1. Attention Jacobian = Value Term + Pattern Term (exact decomposition)
  2. Value Term = A ⊗ (W_V·W_O) is what "attention rollout" computes
  3. Pattern Term captures how attention patterns shift with input perturbations
  4. When Pattern Term is small relative to Value Term, attention rollout is faithful
  5. Position-collapsed Value Term is proportional to attention weights
  6. Transformer layers decompose into 4 terms: identity, attention, MLP, interaction
  7. **Deep networks**: Errors compose multiplicatively with amplification factor
  8. **Induction heads**: Cross-layer attention composition is certifiably approximated
  9. **Mechanism certification**: Distinguishes genuine circuits from "interpretability illusions"
  10. **N-Layer Composition**: Early layer errors compound more (amplified by subsequent layers);
      ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ) enables layer-by-layer verification of deep networks
  
  This provides mathematical justification for when attention visualization is valid:
  - Attention rollout is exact when Pattern Term = 0
  - Relative error metric `relativeApproximationError` quantifies approximation quality
  - The `isAttentionRolloutFaithful` predicate formalizes when rollout is ε-faithful
  - **Deep network certification**: `isDeepFaithful` with explicit error bounds through layers

- **`Abstraction.lean`**  
  Causal consistency of circuit abstractions—bridges real networks to abstract causal graphs:
  - **Ablation**: `SignedMixer.ablate`, `DeepLinearization.ablateJacobian`, `ablateValueTerm`
    (zero out contributions from blocked positions for causal intervention)
  - **Ablation error**: `ablationError`, `ablationError_eq_ablatedPatternTerm`
    (error from ablating full Jacobian vs value term = ablated pattern term)
  - **Projection**: `positionMixer`, `positionFlowMagnitude`, `DeepLinearization.toLocalSystem`
    (extract causal DAG from network's `DeepValueTerm`)
  - **Discrepancy**: `ablationDiscrepancy` measures |real ablation - abstract ablation|
  - **Causal consistency theorems**:
    - `causal_consistency_bound`: pointwise bound by pattern term weights
    - `causal_consistency_frobenius_simple`: Σⱼ(discrepancy_j)² ≤ ‖v‖² · ‖PatternTerm‖²_F
    - `causal_consistency_frobenius`: if ‖PatternTerm‖_F ≤ ε then error ≤ ε²·‖v‖²
  - **Mechanism certification**:
    - `isCausallyCertified`: pattern term below threshold
    - `certified_ablation_faithful`: certified circuits have bounded ablation error
    - `VerifiedMechanism`: structure bundling threshold, certification, and description
    - `VerifiedMechanism.causal_consistency`, `VerifiedMechanism.rms_bound`
  
  Key insight: This transforms the library into a **verification engine**. Given real
  model weights, it provides mathematical certificates that discovered circuits are
  **genuine mechanisms** (not interpretability illusions). When `DeepPatternTerm` is
  small, interventions on the abstract `LocalSystem` accurately predict real network
  behavior—enabling rigorous circuit analysis.

- **`Discovery.lean`**  
  Executable circuit discovery for induction heads and generic circuits—bridges theoretical
  bounds to practical weight analysis:
  - **Concrete structures**: `ConcreteMatrix` (row-major Float storage with `get`, `matmul`,
    `frobeniusNorm`, `transpose`, `add`, `scale`, `map`, `addBias`), `ConcreteAttentionLayer`
    (W_Q, W_K, W_V, W_O with dimension constraints), `ConcreteAttentionWeights` (attention
    matrix with softmax computation), `ConcreteMLPLayer` (W_in, W_out, biases with hiddenDim)
  - **Operator norms**: `operatorNormBound` (power iteration for spectral norm), `maxAbsRowSum`
    (L∞ operator norm), `maxAbsColSum` (L1 operator norm)—for computing Cⱼ bounds
  - **Precomputation cache** (PERFORMANCE OPTIMIZATION):
    - `PrecomputedHeadData`: Cached attention patterns (A), projections (V·W_O, Q·K^T), norms
    - `PrecomputedCache`: Model-wide cache with `headData[layerIdx][headIdx]` and `layerNormBounds`
    - `PrecomputedCache.build`: One-time precomputation of all attention patterns and norms
    - Reduces O(L²H²) redundant computations to O(LH) for GPT-2 Small (20,736 → 144 calls)
  - **Forward pass implementation**:
    - `ConcreteAttentionLayer.forward`: Compute attention output Y = A·V·W_O for given input
    - `ConcreteAttentionLayer.computeAttentionWeights`: Compute A = softmax(Q·K^T/√d)
    - `ConcreteMLPLayer.forward`: Compute Y = W_out · GeLU(W_in · X + b_in) + b_out
    - `ConcreteModel.runForward`: Full forward pass accumulating residual stream
    - `ForwardPassResult`: Layer-wise inputs, attention outputs, MLP outputs, final output
  - **Efficient bound calculation**: `computeValueTermNorm` uses ‖A‖_F · ‖W_V·W_O‖_F factorization,
    `computePatternTermBound` uses (2/scale) · ‖X‖_F · ‖W_Q·W_K^T‖_F · ‖W_V·W_O‖_F (avoids
    expanding full N²×N² Jacobian)
  - **N-layer error amplification** (bridges `Linearization.lean` theorems to executable code):
    - `computeSuffixAmplification`: ∏_{j≥start} (1 + Cⱼ) - how errors get amplified
    - `computeTotalAmplifiedError`: Σᵢ εᵢ · suffixAmplification(i+1) - the N-layer formula
    - `estimateAttentionLayerNorm`: Layer Jacobian norm estimation from weights
    - `estimateLayerPatternBound`, `estimateLayerAblationError`: Per-layer error components
    - `DeepCircuitCandidate`: Induction head candidate with N-layer amplified error
    - `findDeepCircuitCandidates`: Discover induction heads using proper N-layer bounds
    - `LayerErrorMetrics`, `DeepCircuitVerification`: Per-layer and total error breakdown
    - `verifyDeepCircuit`: Main verification function using N-layer composition formula
    - `certifyDeepCircuit`, `VerifiedDeepCircuit`: Certified circuits that meet threshold
  - **Induction head discovery** (OPTIMIZED):
    - `findInductionHeadCandidates`: Legacy single-layer analysis (uses raw embeddings)
    - `findInductionHeadCandidatesDeep`: **Correct multi-layer analysis** using accumulated
      residual stream—layer N sees output of layers 0..N-1. **OPTIMIZED** to use
      `PrecomputedCache` for O(LH) instead of O(L²H²) attention computations
    - `findDeepCircuitCandidates`: **OPTIMIZED** N-layer circuit discovery using cached
      attention patterns, projections, and operator norm bounds
    - `findVerifiedInductionHeads`, `findVerifiedInductionHeadsDeep`: Filter by threshold
    - `checkPrevTokenPattern`, `checkInductionPattern`: Pattern detection helpers
  - **Generic circuit discovery**:
    - `ComponentId`: Identifier for model components (attention heads via `head`, MLP neurons via `mlpNeuron`)
    - `ConcreteCircuit`: Sparse subgraph mask for both `includedHeads` and `includedNeurons`
    - `ComponentImportance`: Metrics for ranking components (valueTermNorm, patternTermBound,
      faithfulnessRatio)
    - `CircuitError`: Error breakdown (patternTermError + ablationError = totalError)
    - `estimateCircuitFaithfulness`: Compute faithfulness error for any circuit mask (heads + neurons)
    - `discoverCircuit`: Greedy pruning algorithm—starts with full model, iteratively removes
      least important component (by valueTermNorm) until error exceeds threshold
    - `discoverCircuitVerbose`: Same algorithm with step-by-step logging
    - `VerifiedCircuit`: Certified circuit that meets threshold
  - **Target-aware circuit discovery (Logit Lens)**:
    - `ConcreteModel.unembedding`: Optional decoder matrix (modelDim × vocabSize) for logit computation
    - `TargetDirection`: Target vector in residual stream space (e.g., logit difference)
    - `TargetDirection.fromLogitDiff`: Create `u = W_U[:, correct] - W_U[:, incorrect]`
    - `TargetDirection.fromSingleToken`: Target direction for promoting a specific token
    - `TargetAwareImportance`: Like `ComponentImportance` but with `targetProjection` field
    - `computeHeadTargetImportance`: `‖A‖_F · ‖(W_V·W_O)·u‖` instead of `‖W_V·W_O‖_F`
    - `computeNeuronTargetImportance`: `‖W_in[:,i]‖ · |W_out[i,:]·u|` instead of `‖W_out[i,:]‖`
    - `computeAllTargetImportance`: Target-aware metrics for all components
    - `discoverTargetedCircuit`: Greedy pruning using target projections
    - `discoverTargetedCircuitVerbose`: Same with step-by-step logging
    - `discoverLogitDiffCircuit`: Convenience wrapper for logit difference analysis
    - `rankComponentsByTargetImportance`, `topKTargetComponents`: Target-based ranking
  - **Ablated forward pass (causal intervention)**:
    - `ConcreteMLPLayer.forwardAblated`: MLP forward with neuron-level masking
    - `ConcreteModel.runAblatedForward`: Full forward pass with circuit mask—excluded heads/neurons
      contribute zero to residual stream, enabling causal intervention experiments
    - `AblationResult`: Captures full vs ablated output, empirical error, relative error
    - `computeAblationDiscrepancy`: Measure `‖full - ablated‖_F` for a circuit
    - `VerificationResult`: Compare empirical error to theoretical bound
    - `verifyCircuitFaithfulness`: Assert empirical ≤ theoretical (the verification bridge)
    - `discoverAndVerify`: End-to-end: discover circuit + empirically verify
    - `discoverTargetedAndVerify`: Same for target-aware circuits
    - `analyzeCircuitFaithfulness`, `analyzeTargetedCircuitFaithfulness`: Full analysis pipeline
  - **MLP neuron analysis**:
    - `ConcreteMLPLayer.neuronInfluence`: ‖W_in[:,i]‖ · ‖W_out[i,:]‖ bounds information flow
    - `computeNeuronImportance`: Calculate importance metrics for individual MLP neurons
    - Pattern term bound = 0 for ReLU (piecewise linear when active)
  - **MLP Interval Bound Propagation (IBP)** — sound activation stability analysis:
    - `NeuronIntervalBound`: Per-neuron interval bounds on pre-activations [z-Δ, z+Δ]
    - `NeuronIntervalBound.isStable`, `isStablyActive`, `isStablyInactive`: Stability predicates
    - `NeuronIntervalBound.patternTermBound`: Error bound for unstable (flip-prone) neurons
    - `MLPIntervalAnalysis`: Layer-wide stability analysis (stable/unstable counts, total error)
    - `ConcreteMLPLayer.computePreActivations`: z = W_in^T · x + b for all neurons
    - `ConcreteMLPLayer.computeIntervalBounds`: Propagate perturbation ε through weights
    - `ConcreteMLPLayer.analyzeIntervalBounds`: Full IBP analysis for a layer
    - `ConcreteMLPLayer.neuronPatternTermBoundIBP`: Single-neuron IBP query
    - `computeNeuronImportanceIBP`: IBP-aware importance with rigorous pattern term bounds
    - `CircuitErrorIBP`: Extended error with `unstableNeuronCount`, `mlpInstabilityError`
    - `estimateCircuitFaithfulnessIBP`: **Sound** circuit verification accounting for MLP flips
    - `analyzeModelMLPStability`: Model-wide MLP stability summary
    
    **Mathematical Foundation:**
    - Pre-activation interval: z' ∈ [z - ε·‖W_in‖, z + ε·‖W_in‖] by Cauchy-Schwarz
    - Stable neuron: interval doesn't cross zero → pattern term = 0 (exact linearization)
    - Unstable neuron: interval crosses zero → may flip activation state
    - Flip error bound: ‖W_out[i,:]‖ · max(|z_lower|, |z_upper|)
    
    **Key Insight:** Standard circuit discovery assumes MLP neurons are locally linear
    (pattern term = 0), which is **unsound** when ablations cause activation flips.
    IBP provides rigorous bounds by detecting which neurons may flip and bounding
    the error from each flip. This converts the tool from a heuristic scanner into
    a **sound verifier**.
  - **Sparse Autoencoder (SAE) support** — feature-level circuit discovery:
    - `ConcreteSAE`: SAE weights (W_enc, W_dec, b_enc, b_dec) with dimension constraints
    - `ConcreteSAE.encode`: f = ReLU(W_enc^T · x + b_enc) → sparse feature activations
    - `ConcreteSAE.decode`: x' = W_dec^T · f + b_dec → reconstructed input
    - `ConcreteSAE.forward`: encode then decode (autoencoder pipeline)
    - `ConcreteSAE.reconstructionError`: ‖x - forward(x)‖ (SAE approximation quality)
    - `ConcreteSAE.featureInfluence`: ‖W_enc[:,k]‖ · ‖W_dec[k,:]‖ (feature importance)
    - `ConcreteSAE.activeFeatures`, `numActiveFeatures`: Sparsity analysis
    - `SAEFeatureIntervalBound`: IBP for SAE features (stability under perturbations)
    - `SAEIntervalAnalysis`: Layer-wide SAE stability analysis
    - `ComponentId.saeFeature`: New case for SAE feature components
    - `SAECircuit`: Circuit mask operating on SAE features (not MLP neurons)
    - `SAEEnhancedModel`: Model with SAEs attached at each layer's MLP
    - `SAEFeatureImportance`: Importance metrics with reconstructionError field
    - `SAECircuitError`: Error breakdown including reconstruction error
    - `estimateSAECircuitFaithfulness`: Sound verification with SAE approximation error
    - `discoverSAECircuit`: Greedy pruning over (heads + SAE features)
    - `discoverSAECircuitWithResult`: Discovery with full result details
    
    **Key Insight:** SAEs decompose MLP activations into interpretable features,
    enabling circuit discovery in terms of monosemantic concepts rather than
    polysemantic neurons. The total error now includes:
    1. Pattern term error (attention + feature activation instability)
    2. Ablation error (excluded components' information loss)
    3. Reconstruction error (SAE's approximation of MLP behavior)
    
    This enables finding circuits like "feature #42 (the 'cat' feature) flows through
    attention head 5.2 to produce the 'cat' logit"—a much more interpretable description
    than "neuron 1847 activates and flows through..."
  - **Sparsity-aware pattern term bounds**:
    - `softmaxRowJacobianNorm`: `sqrt(1 - Σᵢ p_i²)` for probability vector p
    - `avgSoftmaxJacobianNorm`, `maxSoftmaxJacobianNorm`, `softmaxJacobianFrobeniusNorm`:
      Compute data-dependent softmax Jacobian norms across attention rows
    - `computePatternTermBound`: Uses actual attention sparsity instead of worst-case 2.0
    - `computePatternTermBoundPessimistic`: Old constant-factor bound for comparison
  - **Analysis utilities**: `rankComponentsByImportance`, `topKComponents`, `reliableComponents`,
    `analyzeCircuit` (comprehensive circuit analysis with compression ratio)
  
  Key insights:
  1. The Frobenius norm factorization ‖A ⊗ M‖_F = ‖A‖_F · ‖M‖_F enables O(N²+D²) bound
     computation instead of O(N²D²)
  2. **Error model**: Total Error = Σ(included) patternBound + Σ(excluded) valueNorm
     - Included components contribute approximation error (pattern term)
     - Excluded (ablated) components contribute information loss (value term)
  3. **Greedy pruning**: Remove smallest-valueNorm component at each step, stop when
     threshold would be exceeded. O(n²) complexity for n components.
  4. This module is **executable**: designed for export from PyTorch/JAX and runtime verification
  5. **Deep analysis**: `runForward` computes the residual stream through all layers,
     enabling `findInductionHeadCandidatesDeep` to correctly detect induction heads
     where layer 2 attends to information created by layer 1
  6. **MLP neurons**: Neuron importance = product of input/output weight norms; ReLU has
     pattern term = 0 (locally linear), enabling efficient full-circuit analysis
  7. **Sparsity exploitation**: Sharp attention heads (like induction heads) have near-zero
     softmax Jacobian norm, drastically reducing pattern term bounds and enabling verification
     of circuits that fail under pessimistic constant bounds
  8. **N-layer verification**: `verifyDeepCircuit` implements the rigorous N-layer composition
     formula ε_total = Σᵢ εᵢ · ∏_{j>i} (1 + Cⱼ), accounting for error amplification through
     subsequent layers—early layer errors compound more than late layer errors
  9. **Performance optimization**: `PrecomputedCache` eliminates O(L²H²) redundant attention
     computations in nested discovery loops. For GPT-2 Small: 12 layers × 12 heads = 144 heads,
     naive nested loops = L² × H² = 144² = 20,736 attention computations, optimized = L × H = 144
     (144× speedup). All attention patterns, projections, and norms computed once per head.
  10. **Target-aware discovery**: `discoverTargetedCircuit` finds circuits for *specific predictions*
     (e.g., "why does the model predict 'cat' over 'dog'?") by projecting component outputs onto
     target directions. This yields much smaller circuits than generic discovery, isolating the
     mechanism responsible for a particular behavior rather than "everything the model does."
  11. **Empirical verification**: `runAblatedForward` executes the discovered circuit, and
      `verifyCircuitFaithfulness` confirms that empirical error ≤ theoretical bound, closing the
      loop between static weight analysis and dynamic execution.
  12. **SAE-based discovery**: `discoverSAECircuit` enables circuit discovery using interpretable
      SAE features instead of polysemantic MLP neurons. The error model adds reconstruction error
      (how well the SAE approximates the MLP) to the standard pattern term + ablation errors.
      This enables discovering circuits described in terms of monosemantic concepts.

- **`IO.lean`**  
  Model loading and CLI integration—bridges external model weights to verified analysis:
  - **Text format parsing**: `parseFloat`, `parseNat` for robust number parsing
  - **Matrix construction**: `buildMatrix` with safe `Array.ofFn` construction and size proofs
  - **Layer construction**: `mkAttentionLayer`, `mkMLPLayer` with dimension constraint proofs
  - **Model loading**: `loadFromText` parses NFP Text Format (`.nfpt`) files into `ConcreteModel`
  - **Tokenization**: `Tokenizer` structure with vocabulary lookup for prompt processing
  - **Analysis reports**: `AnalysisReport`, `analyzeModel`, `analyzeAndVerify` for human-readable output
  - **File format**: `.nfpt` is a simple text format with headers (`NFPT_VERSION`, `num_layers`, etc.)
    followed by weight matrices in row-major order
  
  Key insight: This module is **IO-only**—it contains no proofs, just executable code for
  loading weights and formatting results. The verified analysis happens in `Discovery.lean`;
  this module just provides the bridge to the filesystem and CLI.
  
  Note: Due to Mathlib initialization issues in compiled executables, full analysis runs via
  `lake env lean --run scripts/analyze.lean` rather than a compiled binary.

- **`Nfp.lean`**  
  Top-level re-exports and `#print axioms` verification that all theorems depend only on
  `propext`, `Classical.choice`, and `Quot.sound`.

If you introduce a new conceptual layer, either extend the closest existing file or add a new module with a clear name and docstring, and then reflect it here.

---

## 3. Refactoring & Extension Rules

You **may** perform nontrivial refactors when needed. Use these guardrails:

- **Preserve invariants**
  - Probability vectors sum to `1`.
  - Mixers remain row-stochastic where intended.
  - Capacities stay nonnegative.
  - DAG / acyclicity assumptions stay valid.

- **Respect the narrative, but update it if needed**
  - Each major definition/theorem should have a short docstring explaining:
    - what it is, and
    - roughly which informal concept / section it corresponds to.
  - If you change a design substantially, update comments/docstrings to match.

- **API stability is a soft preference, not a hard rule**
  - Try not to gratuitously rename core theorems/defs.
  - If a rename/type change is needed for a cleaner design:
    - update all call sites,
    - leave a brief comment explaining the rationale.

- **Lean options & style**
  - Do not weaken linters or change global options to "make things pass".
  - Use small, composable lemmas rather than huge opaque proofs.

- **Keep AGENTS.md aligned with the code**
  - After refactors that affect module layout or responsibilities, update
    Section 2 ("Module Roles") so future agents see the correct structure.

---

## 4. Checklist Before You Commit Changes

- [ ] `lake build -q --wfail` succeeds (no warnings, no errors).
- [ ] No `sorry` remains.
- [ ] No new axioms were added.
- [ ] Linters remain fully enabled; no `set_option linter.* false` anywhere.
- [ ] Section **2. Module Roles** accurately reflects the current module layout.
- [ ] New nontrivial definitions/theorems have short, accurate docstrings.
- [ ] Core invariants (nonnegativity, normalization, finiteness, acyclicity) are preserved and, where possible, explicitly proved.
- [ ] Refactors that change the conceptual design are accompanied by updated comments/docstrings.

If you must choose between:
- a slightly breaking but conceptually clean redesign vs.
- preserving an awkward design forever,

prefer the **clean redesign**, but do it consciously and documented.
