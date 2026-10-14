# friction-cliff-bench

**Status: planning. No code and no results yet.**

A feasibility-audited, parity-controlled benchmark of how a car's controller copes with **sudden tire–road
friction loss** (asphalt → ice). It compares an RMA-style teacher–student PPO policy, which infers friction
from its own recent history, against strong adaptive MPPI baselines (LLA-MPC-style model bank, learned
μ-estimator) and known-friction oracles. Every scenario is first labelled by what is physically possible:
*certified infeasible*, *preview-required*, *reactive-feasible*, or *undetermined*. Results are therefore
reported as "where each method wins", not as a single number.

**Thesis.** At abrupt friction loss, whether learned policy-side adaptation or model-side adaptive MPC wins
depends on whether the scenario is physically recoverable and on how far reality departs from the
controller's model family. A feasibility-audited benchmark can map those regimes on one laptop GPU.

## Planned components
- Small GPU vehicle simulator in NVIDIA Warp: dynamic bicycle, combined-slip Pacejka, wheel spin,
  per-axle μ fields, actuator lag. This is infrastructure, validated by property tests and cross-checked
  against an independent implementation.
- Feasibility labelling: a friction-circle certificate plus reactive and clairvoyant oracle MPPI.
- **C++17/CUDA MPPI library** (CMake, GoogleTest, Python bindings), checked for equivalence against a
  reference implementation. The confirmatory sweeps run on it.
- PPO teacher (privileged *current* per-axle μ), RMA student, and ablations.
- Pre-registered hypotheses, paired scenarios, multi-seed runs, and confidence intervals on every reported number.

## Documents
| File | Contents |
|---|---|
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Dated design decisions and why |
