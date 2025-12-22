# CLAIMS

This file lists what is formally proven in Lean, what is soundly checked by the trusted checker,
what is heuristic, and what is not yet proven.

| Claim | Status | Where |
| --- | --- | --- |
| Definitions of mixers/signed mixers and linearizations (ReLU, GeLU, LayerNorm, softmax) with basic lemmas (composition, diagonality, etc.) | Proven in Lean | `Nfp/SignedMixer.lean`, `Nfp/Linearization.lean` |
| The sound certificate checker validates internal arithmetic consistency for layer bounds and the total amplification factor | Soundly checked (Lean) | `Nfp/Sound/Cert.lean` |
| Sound bound formulas use exact `Rat` arithmetic (LayerNorm/softmax/GeLU envelopes); witness values are produced in untrusted code and then checked | Soundly checked formulas; untrusted witnesses | `Nfp/Sound/Bounds.lean`, `Nfp/Untrusted/SoundCompute.lean` |
| Heuristic discovery and ranking of induction-style candidates | Heuristic | `Nfp/Discovery.lean`, CLI `induction` |
| End-to-end statement that certificate validity implies `||layerJacobian - I|| <= C` for Lean-defined Jacobians | Not yet proven | See `SOUNDNESS_LIMITATIONS.md` |
