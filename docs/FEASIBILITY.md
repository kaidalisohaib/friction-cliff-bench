# FEASIBILITY: labelling what is physically possible

**Why this exists.** Without it, "who wins at the ice patch" mostly measures how the scenarios were sampled.
Example: at μ = 0.1, 15 m/s, R = 50 m, the car needs v²/R = 4.5 m/s² of lateral acceleration but has
μg ≈ 0.98 m/s². With a 3.52 m/s² shortfall the lateral error grows ≈ ½·3.52·t², so the car crosses the
half-lane (1.75 m) in ≈ 1.0 s (15 m) and the full lane in ≈ 1.4 s (21 m). The max speed on that curve on ice is
√(μgR) ≈ 7.0 m/s, and shedding 8 m/s at 0.98 m/s² takes ≈ 90 m of *straight-line* braking. No controller can
survive a long patch there. Published adaptation papers do not separate such cells: LLA-MPC's known-friction
oracle still leaves the track for 0.72 s after its 40% drop. We found no vehicle-adaptation benchmark that
labels scenarios by feasibility (as of 2026-10-13).

## 1. Labels (per scenario)
| Label | Definition |
|---|---|
| **certified-infeasible** | A *sound* argument (§2) shows that no admissible trajectory stays within the success corridor |
| **preview-required** | The clairvoyant oracle succeeds; the reactive oracle fails (all oracle seeds) |
| **reactive-feasible** | The reactive oracle succeeds (≥ 1 of N oracle seeds) |
| **undetermined** | Not certified infeasible, and no oracle succeeds. Reported separately, never merged |

Within reactive-feasible, a **marginal** sub-band is defined by the reactive oracle's minimum lane clearance
(e.g. < 0.3 m; ⚠ fix in prereg). This avoids headline results being driven by knife-edge cells.

"Success" (shared with `EVAL_PROTOCOL.md`): no footprint corner beyond the lane edge, no loss of control (|β| ≤ 20°),
no spin, over the whole episode.

## 2. Layer 1: analytic certificate (sound infeasibility only)

**Relaxation argument (to be written out formally in P1):** every tire force satisfies |F_i| ≤ μ_i·F_z,i (gate V4/S4).
So the total horizontal force on the car is ≤ μ_max(t)·m·g, where μ_max(t) is the larger of the per-axle μ values
at time t. The car's CoG therefore follows a **point-mass** trajectory with |a(t)| ≤ μ_max(t)·g. If *no* point-mass
trajectory with that acceleration bound keeps the CoG inside the corridor shrunk by the footprint's inner
half-width, the scenario is infeasible for the full model. This is a necessary condition. It can certify
infeasibility but never feasibility.

**Computing it (⚠ choose in P1, in this order of preference):**
1. **Closed-form bound** for the dominant case (constant curvature κ across a patch longer than the escape
   distance): the best achievable lateral deviation under |a| ≤ μ1·g given entry speed v0. Derive it, prove that it
   is a bound, and unit-test it against brute-force point-mass optimization.
2. **Convex relaxation / grid search**, if the closed form is too loose: a point-mass minimum-peak-deviation
   problem over a discretized time grid (convex in a path-parameterized formulation; solve with a QP/SOCP solver),
   reported as a lower bound.
3. **HJ reachability** on a reduced lateral model (`optimized_dp` / `hj_reachability`): stretch only, offline.

Requirement: every certificate method has a written soundness argument in this file *before* any scenario is
labelled infeasible with it.

## 3. Layer 2: constructive oracles (feasibility witnesses)
- **Reactive oracle:** MPPI with the *exact* truth simulator as its rollout model and the true *current* per-axle
  μ + surface, held constant over its horizon (it does not know what is ahead).
- **Clairvoyant oracle:** the same, but its rollouts use the true μ/surface map ahead.
- Oracle settings are generous: long horizon (≈ 3 s), many samples (≈ 16k), best of N = 5 noise seeds per
  scenario. Settings are frozen before labelling.
- A success by either oracle is a **witness** that the scenario is feasible at that information level.

## 4. Sanity gate S1
No oracle may succeed in a certified-infeasible scenario. If one does, it is a bug (simulator, certificate, or
metric) and must be fixed before any results are reported.

## 5. Outputs
- Labelled scenario sets (tuning, test) with per-scenario labels + oracle metrics.
- Feasibility-map figures: label share over (v0 × R) for each surface × patch length.
- The *undetermined* share is reported. If it is large, improve the certificate (Layer 1 option 2/3) or the oracle,
  but never re-label without a recorded reason.
