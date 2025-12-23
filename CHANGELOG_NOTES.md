# CHANGELOG / NOTES

## 2025-12-24
- Added attention pattern-term coefficients using max `W_Q/W_K` row-sum norms and a conservative
  LayerNorm output magnitude bound; updated layer cert formulas and reports accordingly.
- Added `modelDim`/`headDim` metadata to sound certificates and threaded through the checker.

## 2025-12-23
- Added margin-derived softmax max-probability and Jacobian bounds for best-match pattern certificates.
- Added effort-indexed exp lower bounds (scaled Taylor + squaring) and wired them into best-match softmax bounds.
- Extended best-match head pattern certs with a recorded softmax Jacobian upper bound and wired untrusted computation to populate it.
- Noted that the exp lower-bound correctness is not yet formalized in Lean.
- Layer-level sound certificates now use a portfolio softmax Jacobian bound field, with margin-based
  tightening available when margins are supplied (defaults remain worst-case today).
- Added `nfp certify --softmaxMargin/--softmaxExpEffort` flags and report fields to pass margin
  evidence into layer-level softmax portfolio bounds.

## 2025-12-22
- Optimized induction head discovery by caching per-head induction scores and per-layer input norms, eliminating redundant pattern scans and repeated Frobenius norm computations.
- Tightened induction error bounds by using data-dependent V norms (Frobenius/op) in pattern-term calculations.
- Tightened per-head weight operator-norm bounds using Brauer/moment Gram candidates, improving pattern-term bounds.

## 2025-12-21
- Updated sound certificate algebra to include the attn*mlp cross term and surfaced it in output.
- Added CLAIMS.md and clarified soundness limitations and reproducibility documentation.
- Added operator-norm bound lemmas for SignedMixers, including a residual composition bound that takes external operator-norm bounds.
- Added a helper lemma to extract the `C` identity from `LayerAmplificationCert.Valid`.
- Added a bridge lemma to bound residual composition from component bounds plus the cast `C` identity.
- Added cast and `Valid`-based bridge lemmas to move from certificate validity to the residual bound.
- Added a `DeepLinearization` lemma that turns per-component operator-norm bounds into a layer residual bound.
- Added a certificate-to-Jacobian bridge lemma tying layer certificates to residual bounds (under component-bound assumptions).
- Added attention/MLP component bound lemmas that connect certificate identities to operator-norm bounds.
- Added an assumption-based bridge theorem combining component bounds into a full layer residual bound.
- Added a `LayerComponentNormAssumptions` structure to package the remaining component-norm obligations.
- Added an operator-norm bound lemma for diagonal mixers based on uniform entry bounds.
- Added an operator-norm bound lemma for `A ∘ diag(d) ∘ B` from component bounds.
- Replaced the MLP component assumption with a factored-Jacobian assumption and derived the MLP bound from it.
- Added `MLPFactorization` and `mlpFactors` to `DeepLinearization`, with MLP Jacobians derived from the factorization data.
- Added `mlpWinBound`/`mlpWoutBound` fields to the sound certificate and wired the bridge to use them for MLP coefficient bounds.
- Updated README tiny demo instructions to use the binary fixture directly and document the optional scripted path.
- Added a helper to patch missing `gelu_kind` in legacy `.nfpt` headers and wired it into the GPT-2 sound demo script.
- Made demo scripts fall back to `python3` when `python` is unavailable.
- Documented the GPT-2 header patch behavior in README.
- Updated the GPT-2 induction demo to patch legacy headers and use the current Python executable when generating rigorous models.
- Documented GPT-2 induction scan runtime notes in README.
- Added `--fast`, `--jobs`, and `--nfp-bin` options to the induction scan script for faster runs.
