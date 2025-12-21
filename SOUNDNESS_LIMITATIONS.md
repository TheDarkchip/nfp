## SOUNDNESS upgrade status

This file tracks **current limitations** and **remaining work** for the rigorous
soundness upgrade. It is intentionally brief and human-readable.

### Current limitations
- `partitionDepth > 0` is rejected with an explicit error (no partitioning logic yet).
- Affine arithmetic is only a scaffold (`Nfp/Sound/Affine.lean`) and not wired into SOUND certification.
- Softmax Jacobian bounds use probability intervals defaulted to `[0,1]`, so they reduce to worst-case.
- GeLU derivative bounds are conservative envelopes; the exact interval supremum is not computed yet.

### Remaining work
- Implement input-space partitioning in the SOUND local path and plumb it through the certify pipeline.
- Replace or augment interval propagation with affine forms to preserve correlations.
- Add sound probability interval extraction for softmax (requires sound exp/log-sum-exp bounds).
- Tighten GeLU derivative envelopes to the exact interval supremum if desired.
