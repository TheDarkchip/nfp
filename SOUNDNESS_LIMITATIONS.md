## SOUNDNESS upgrade status

This file tracks **current limitations** and **remaining work** for the rigorous
soundness upgrade. It is intentionally brief and human-readable.

### Current limitations
- The bridge theorem in `Nfp/Sound/Bridge.lean` links `LayerAmplificationCert` bounds to
  `DeepLinearization` residual Jacobians, but it requires external operator-norm assumptions
  (LN Jacobians, attention full Jacobian, and MLP factors). The trusted checker does not yet
  discharge those assumptions from model weights.
- `partitionDepth > 0` is rejected with an explicit error (no partitioning logic yet).
- Affine arithmetic is only a scaffold (`Nfp/Sound/Affine.lean`) and not wired into SOUND certification.
- Softmax Jacobian bounds in the standard `certify` path still use the worst-case probability
  interval `[0,1]`; direct `--softmaxMargin` is rejected because margin evidence is unverified.
- Best-match margin tightening is now available via `nfp certify --bestMatchMargins` (binary + local
  inputs with EMBEDDINGS). It runs a full best-match sweep across heads and query positions, which
  can be expensive and will fail if coverage is incomplete.
- Per-head best-match tightening (used by head-pattern/induction certs) is still separate from
  model-level certification unless `--bestMatchMargins` is used.
- Best-match pattern certificates now use a margin-derived softmax Jacobian bound with an
  effort-indexed `expLB` (scaled Taylor + squaring). The lower-bound correctness of `expLB`
  is not yet formalized in Lean.
- GeLU derivative bounds are conservative envelopes; the exact interval supremum is not computed yet.
- Attention Jacobian bounds now include an explicit pattern-term coefficient using max `W_Q/W_K`
  row-sum norms and a conservative LayerNorm output magnitude bound (`max|gamma|*sqrt(d)+max|beta|`),
  but this is still very conservative and only connected to the Lean Jacobian theorems
  under the external norm assumptions above.

### Remaining work
- Implement input-space partitioning in the SOUND local path and plumb it through the certify pipeline.
- Replace or augment interval propagation with affine forms to preserve correlations.
- Add sound probability interval extraction for softmax (requires sound exp/log-sum-exp bounds).
- Verify or compute margin evidence in the trusted path so margin-derived softmax tightening can be
  enabled without a best-match sweep and without rejecting `--softmaxMargin`.
- Tighten GeLU derivative envelopes to the exact interval supremum if desired.
- Discharge the bridge theorem’s component-norm assumptions from certificates/model weights, and
  connect the resulting statement to the `Linearization` Jacobian theorems.
