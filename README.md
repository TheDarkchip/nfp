# nfp (Lean 4)

This repository is a Lean 4 formalization of an “Nfp-style” framework:
finite probability on finite types, row-stochastic mixers, influence specifications,
and uniqueness-style results on finite DAGs, with an executable bridge for circuit analysis.

## Quickstart

- Build (warnings as errors): `lake build -q --wfail`
- Build the CLI executable: `lake build nfp`
- Run on the included toy model:
  - `./.lake/build/bin/nfp analyze test_model.nfpt`
  - `./.lake/build/bin/nfp analyze test_model.nfpt --verify --verbose`

## Repo structure

Most of the library lives in `Nfp/`. The top-level `Nfp.lean` re-exports modules and includes
`#print axioms` checks for key theorems/definitions.

- Proof-centric core: `Nfp/Prob.lean`, `Nfp/Mixer.lean`, `Nfp/Influence.lean`,
  `Nfp/Uniqueness.lean`, `Nfp/MixerLocalSystem.lean`, `Nfp/PCC.lean`, `Nfp/Reroute/*`.
- Neural interpretation layer: `Nfp/Layers.lean`, `Nfp/Attribution.lean`.
- Real-weight layer: `Nfp/SignedMixer.lean`, `Nfp/Linearization.lean`, `Nfp/Abstraction.lean`.
- Executable bridge (Floats/Arrays/IO): `Nfp/Discovery.lean`, `Nfp/IO.lean`, and `Main.lean`.

## Conventions (important)

`SignedMixer` and related code uses a **row-vector** convention:

- Weights are written `w i j` (“from input `i` to output `j`”).
- Application is `out j = ∑ i, v i * w i j`.
- Composition `M.comp N` means “first apply `M`, then `N`” (matrix product order `M` then `N`).

This is the convention used throughout the executable and proof layers.

