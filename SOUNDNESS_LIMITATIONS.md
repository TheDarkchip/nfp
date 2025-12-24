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
- Softmax Jacobian bounds are enforced to use the worst-case probability interval `[0,1]` in
  trusted IO. Margin-derived tightening is computed by the untrusted path, but trusted IO
  currently **rejects nonzero** `softmaxMarginLowerBound` because margin evidence is unverified.
- Local per-head contribution bounds can now be tightened using a best-match pattern certificate,
  but this tightening does **not** propagate to layer-level ModelCert bounds.
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
- Verify or compute margin evidence in the trusted path so margin-derived softmax tightening can be enabled.
- Tighten GeLU derivative envelopes to the exact interval supremum if desired.
- Discharge the bridge theorem’s component-norm assumptions from certificates/model weights, and
  connect the resulting statement to the `Linearization` Jacobian theorems.
