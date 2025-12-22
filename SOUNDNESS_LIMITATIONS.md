## SOUNDNESS upgrade status

This file tracks **current limitations** and **remaining work** for the rigorous
soundness upgrade. It is intentionally brief and human-readable.

### Current limitations
- The sound certificate checker verifies internal arithmetic consistency only; there is not yet
  a Lean theorem linking `LayerAmplificationCert` bounds to the Jacobian objects defined in `Nfp/Linearization.lean`
  without additional component-norm assumptions.
- `partitionDepth > 0` is rejected with an explicit error (no partitioning logic yet).
- Affine arithmetic is only a scaffold (`Nfp/Sound/Affine.lean`) and not wired into SOUND certification.
- Softmax Jacobian bounds use probability intervals defaulted to `[0,1]`, so they reduce to worst-case.
- GeLU derivative bounds are conservative envelopes; the exact interval supremum is not computed yet.
- Attention Jacobian bounds currently omit an explicit pattern-term bound with `W_Q/W_K` contributions and
  activation-magnitude factors.
- The current bridge theorem assumes external operator-norm bounds for:
  `ln1Jacobian`, `ln2Jacobian`, and `fullJacobian`. The MLP side now uses an explicit
  factorization (Win ∘ diag(deriv) ∘ Wout) and records `mlpWinBound`/`mlpWoutBound` in
  the certificate, but still assumes those bounds match the true operator norms.

### Remaining work
- Implement input-space partitioning in the SOUND local path and plumb it through the certify pipeline.
- Replace or augment interval propagation with affine forms to preserve correlations.
- Add sound probability interval extraction for softmax (requires sound exp/log-sum-exp bounds).
- Tighten GeLU derivative envelopes to the exact interval supremum if desired.
- Prove a bridge theorem linking certificate validity to the Lean Jacobian bounds it is intended to certify.
