# Module Map (Where Things Live)

This is a *map*, not a prison. You may reshuffle if a better design emerges,
but you **must** update this list in the same commit.

## Core types
- `Nfp/Core/Basic.lean`
  - `Mass` alias for nonnegative weights used throughout the rewrite.
- `Nfp/Core.lean`
  - Aggregator for core shared definitions.

## Probability vectors
- `Nfp/Prob/Basic.lean`
  - `ProbVec` definition + invariants.
- `Nfp/Prob/Operations.lean`
  - `pure`, `mix`, and basic lemmas.
- `Nfp/Prob.lean`
  - Aggregator for probability modules.

## Mixers
- `Nfp/Mixer/Basic.lean`
  - `Mixer` structure and row-stochastic invariant.
- `Nfp/Mixer/Operations.lean`
  - `push`, `comp`, and `id` mixers.
- `Nfp/Mixer.lean`
  - Aggregator for mixer modules.

## Systems (DAG + local mixing)
- `Nfp/System/Dag.lean`
  - DAG relation + parent/child sets.
- `Nfp/System/LocalSystem.lean`
  - `LocalSystem` with edge support, row-stochastic predicate, and evaluation semantics.
- `Nfp/System.lean`
  - Aggregator for system modules.

## Circuits (certification core)
- `Nfp/Circuit/Basic.lean`
  - DAG-based circuit structure with inputs/outputs and gate semantics.
- `Nfp/Circuit/Combinators.lean`
  - Core circuit combinators (relabeling, interface transport).
- `Nfp/Circuit/Interface.lean`
  - Typed input/output interfaces and interface-based evaluation.
- `Nfp/Circuit/Semantics.lean`
  - Well-founded evaluation semantics for circuits.
- `Nfp/Circuit/WellFormed.lean`
  - Basic well-formedness conditions for circuit inputs.
- `Nfp/Circuit/Cert.lean`
  - Equivalence definition and finite checker.
- `Nfp/Circuit/Cert/SoftmaxMargin.lean`
  - Softmax-margin certificate payloads and checker soundness.
- `Nfp/Circuit/Cert/ValueRange.lean`
  - Value-range certificate payloads and checker soundness.
- `Nfp/Circuit/Cert/LogitDiff.lean`
  - Logit-diff lower-bound computation for induction certificates.
- `Nfp/Circuit/Cert/DownstreamLinear.lean`
  - Downstream linear error certificates for end-to-end induction bounds.
- `Nfp/Circuit/Cert/ResidualBound.lean`
  - Residual-stream bound certificates for downstream error computation.
- `Nfp/Circuit/Cert/ResidualInterval.lean`
  - Residual-stream interval certificates for downstream dot-product bounds.
- `Nfp/Circuit/Typed.lean`
  - Typed circuit wrapper and interface-level equivalence checker.
- `Nfp/Circuit/Compose.lean`
  - Sequential composition and residual wiring for typed circuits.
- `Nfp/Circuit/Gates/Basic.lean`
  - Basic gate combinators for aggregating parent values.
- `Nfp/Circuit/Gates/Linear.lean`
  - Linear and affine gate combinators built from `Matrix.mulVec`.
- `Nfp/Circuit/Gates.lean`
  - Aggregator for gate combinator modules.
- `Nfp/Circuit/Tensor.lean`
  - Typed tensor indices and tensor aliases.
- `Nfp/Circuit/Layers/Linear.lean`
  - Linear/affine layer circuits with typed interfaces.
- `Nfp/Circuit/Layers/Tensor.lean`
  - Batched linear/affine layer circuits for tensor-shaped data.
- `Nfp/Circuit/Layers/Reshape.lean`
  - Reshape combinators for product-typed circuit interfaces.
- `Nfp/Circuit/Layers/Heads.lean`
  - Head split/merge combinators for transformer-shaped indices.
- `Nfp/Circuit/Layers/Softmax.lean`
  - Softmax helpers and margin-based bounds for layer reasoning.
- `Nfp/Circuit/Layers/Attention.lean`
  - Q/K/V, output projection wiring, and attention score/mixing core.
- `Nfp/Circuit/Layers/Induction.lean`
  - Induction-head weight specs and attention-core output lemmas.
- `Nfp/Circuit/Layers/TransformerBlock.lean`
  - GPT-style transformer block wiring from LN/attention/MLP circuits.
- `Nfp/Circuit/Layers.lean`
  - Aggregator for circuit layer modules.
- `Nfp/Circuit.lean`
  - Aggregator for circuit modules.

## CLI surface
- `Nfp/IO/Pure.lean`
  - Aggregator for pure parsing helpers.
- `Nfp/IO/Pure/Basic.lean`
  - Shared parsing helpers (`Nat`/`Int`/`Rat`, token cleanup).
- `Nfp/IO/Pure/InductionHead.lean`
  - Induction-head input payload parsing from text/bytes.
- `Nfp/IO/Pure/InductionHead/Bytes.lean`
  - Byte-level parser for induction-head input payloads.
- `Nfp/IO/Pure/SoftmaxMargin.lean`
  - Aggregator for softmax-margin parsing helpers.
- `Nfp/IO/Pure/SoftmaxMargin/Shared.lean`
  - Shared parsing helpers for softmax-margin payloads.
- `Nfp/IO/Pure/SoftmaxMargin/Cert.lean`
  - Softmax-margin certificate parser.
- `Nfp/IO/Pure/SoftmaxMargin/Raw.lean`
  - Softmax-margin raw-input parser.
- `Nfp/IO/Pure/ValueRange.lean`
  - Aggregator for value-range parsing helpers.
- `Nfp/IO/Pure/ValueRange/Shared.lean`
  - Shared parsing helpers for value-range payloads.
- `Nfp/IO/Pure/ValueRange/Cert.lean`
  - Value-range certificate parser.
- `Nfp/IO/Pure/ValueRange/Raw.lean`
  - Value-range raw-input parser.
- `Nfp/IO/Pure/Downstream.lean`
  - Downstream linear and matrix payload parsers.
- `Nfp/IO/Pure/Residual.lean`
  - Residual-bound and residual-interval payload parsers.
- `Nfp/IO/NfptPure.lean`
  - Pure parsing helpers for `NFP_BINARY_V1` model slices.
- `Nfp/IO/HeadScore.lean`
  - Pure task-based cache builder for head score dot-abs bounds.
- `Nfp/IO/Loaders.lean`
  - IO loaders for certificates and raw inputs.
- `Nfp/IO/Checks.lean`
  - IO checks for certificate validity.
- `Nfp/IO/Derive.lean`
  - IO derivations building certificates from model binaries.
- `Nfp/IO/Timing.lean`
  - IO timing helpers with microsecond reporting and phase wrappers.
- `Nfp/IO/Util.lean`
  - Small CLI parsing utilities shared across IO entrypoints.
- `Nfp/IO/InductionHead.lean`
  - Induction-head IO pipeline with timing instrumentation.
- `Nfp/IO/Bench/Rational.lean`
  - Microbenchmarks for rational arithmetic and caching.
- `Nfp/IO/Bench/InductionCore.lean`
  - Benchmark helpers for induction-head core certification.
- `Nfp/IO/Bench/InductionCounts.lean`
  - Call-count instrumentation for induction-head computations.
- `Nfp/IO.lean`
  - IO-only wrappers for loading inputs and running checks.
- `Nfp/Cli.lean`
  - CLI commands and `main` implementation.
- `Main.lean`
  - Thin entrypoint delegating to `Nfp.Cli.main`.
  - Benchmark entrypoint for rational microbenchmarks.
- `Nfp.lean`
  - Top-level reexports.
- `TheoremAxioms.lean`
  - Axiom dashboard for `theorem-axioms` build target (`#print axioms`).

## Sound certification
- `Nfp/Sound/Induction.lean`
  - Aggregator for induction soundness modules.
- `Nfp/Sound/Induction/Core.lean`
  - Sound builders and core proofs for induction certificates from exact inputs.
- `Nfp/Sound/Induction/CoreSound.lean`
  - Soundness proof for `buildInductionCertFromHeadCore?`.
- `Nfp/Sound/Induction/CoreSound/Values.lean`
  - Helper lemmas for value-direction projections in the core soundness proof.
- `Nfp/Sound/Induction/CoreDefs.lean`
  - Core definitions and soundness predicates for induction certificates.
- `Nfp/Sound/Induction/HeadOutput.lean`
  - Head-output interval certificates built from induction head inputs.
- `Nfp/Sound/Induction/HeadBounds.lean`
  - Helper bounds used to stage head-induction certificate construction.
- `Nfp/Sound/Induction/LogitDiff.lean`
  - Logit-diff bounds derived from induction certificates.
- `Nfp/Sound/Induction/OneHot.lean`
  - Per-query one-hot bounds derived from score margins.
- `Nfp/Sound/Bounds/Cache.lean`
  - Cached bound evaluators (thunk/task backed) for interval computations.
- `Nfp/Sound/Bounds/MatrixNorm.lean`
  - Row-sum matrix norms and downstream linear certificate builders.
- `Nfp/Sound/Bounds/MatrixNorm/Interval.lean`
  - Dot-product and matrix-vector interval bounds (rational and real).
- `Nfp/Sound/Bounds/LayerNorm.lean`
  - LayerNorm interval bounds and end-to-end soundness lemmas.
- `Nfp/Sound/Bounds/LayerNorm/MeanVariance.lean`
  - Mean/variance helpers for LayerNorm bounds.
- `Nfp/Sound/Bounds/LayerNorm/InvStd.lean`
  - Inverse-standard-deviation bounds for LayerNorm.
- `Nfp/Sound/Bounds/UnnormRat.lean`
  - Unnormalized rational helpers for deferred normalization in bounds kernels.
- `Nfp/Sound/Bounds/Gelu.lean`
  - Tanh-GELU bounds for interval propagation through MLPs.
- `Nfp/Sound/Bounds/Mlp.lean`
  - Interval bounds for GPT-2 MLP blocks and LayerNorm composition.
- `Nfp/Sound/Bounds/Attention.lean`
  - Interval bounds for multi-head attention and transformer layers.
- `Nfp/Sound/Bounds/Transformer.lean`
  - Interval bounds for transformer stacks and final LayerNorm outputs.
- `Nfp/Sound/Bounds/Transformer/Embedding.lean`
  - Embedding interval bounds and position-restricted bounds.
- `Nfp/Sound/Linear/FinFold.lean`
  - Tail-recursive folds and sums for sound linear computations.
- `Nfp/Sound/Gpt2/HeadInputs.lean`
  - Sound construction of GPT-2 induction head inputs.
- `Nfp/Sound.lean`
  - Aggregator for sound certification modules.

## Model inputs
- `Nfp/Model/InductionHead.lean`
  - Exact induction-head input payloads (embeddings and projection weights).
- `Nfp/Model/InductionPrompt.lean`
  - Prompt utilities (`prev` map and active set for periodic prompts).
- `Nfp/Model/Gpt2.lean`
  - Exact GPT-2 head-slice data, layer/MLP/LayerNorm parameters, and embedding helpers.
- `Nfp/Model.lean`
  - Aggregator for model input modules.

If you introduce a new conceptual layer:
- either extend the closest existing file,
- or add a new module with a clear name + top docstring,
- and update this map in the same commit.
