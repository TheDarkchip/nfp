# CLAIMS

This file lists what is formally proven in Lean, what is soundly checked by the trusted checker,
what is heuristic, and what is not yet proven.

| Claim | Status | Where |
| --- | --- | --- |
| Definitions of mixers/signed mixers and linearizations (ReLU, GeLU, LayerNorm, softmax) with basic lemmas (composition, diagonality, etc.) | Proven in Lean | `Nfp/SignedMixer.lean`, `Nfp/Linearization.lean` |
| Model-level SOUND certificate checker validates internal arithmetic consistency and recomputes weight-derived bounds from model files | Soundly checked (Lean) | `Nfp/Sound/Cert.lean`, `Nfp/Sound/IO.lean`, `Nfp/Sound/BinaryPure.lean`, `Nfp/Sound/TextPure.lean` |
| Per-head contribution, head-pattern, and induction-head certificates (including best-match variants) have internal consistency checks | Soundly checked (Lean) | `Nfp/Sound/HeadCert.lean`, `Nfp/Sound/IO.lean` |
| Sound bound formulas use exact `Rat` arithmetic (LayerNorm/softmax/GeLU envelopes); witness values are produced in untrusted code and then checked | Soundly checked formulas; untrusted witnesses | `Nfp/Sound/Bounds.lean`, `Nfp/Untrusted/SoundCompute.lean`, `Nfp/Untrusted/SoundBinary.lean` |
| Best-match margin tightening uses untrusted logit bounds; verification checks only internal margin/softmax consistency | Partially checked (internal consistency only) | `Nfp/Sound/HeadCert.lean`, `Nfp/Sound/IO.lean`, `Nfp/Untrusted/SoundCompute.lean` |
| Heuristic discovery and ranking of induction-style candidates | Heuristic | `Nfp/Discovery.lean`, CLI `induction` |
| Empirical causal verification via head ablation (competence/control/energy checks) | Heuristic | `Nfp/Verification.lean`, CLI `analyze --verify` / `induction --verify` |
| End-to-end statement that certificate validity implies `||layerJacobian - I|| <= C` for Lean-defined Jacobians | Not yet proven | See `SOUNDNESS_LIMITATIONS.md` |
