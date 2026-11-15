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
(< 0.30 m, best O-R seed; prereg V3). This avoids headline results being driven by knife-edge cells.

"Success" (shared with `EVAL_PROTOCOL.md`): no footprint corner beyond the lane edge, no loss of control (|β| ≤ 20°),
no spin, over the whole episode.

## 2. Layer 1: analytic certificate (sound infeasibility only)

**Relaxation argument (to be written out formally in P1):** every tire force satisfies |F_i| ≤ μ_i·F_z,i (gate V4/S4).
So the total horizontal force on the car is ≤ μ_max(t)·m·g, where μ_max(t) is the larger of the per-axle μ values
at time t. The car's CoG therefore follows a **point-mass** trajectory with |a(t)| ≤ μ_max(t)·g. If *no* point-mass
trajectory with that acceleration bound keeps the CoG inside the corridor shrunk by the footprint's inner
half-width, the scenario is infeasible for the full model. This is a necessary condition. It can certify
infeasibility but never feasibility.

**Computing it (options in order of preference; P1 implemented option 1, the closed-form bound):**
1. **Closed-form bound** for the dominant case (constant curvature κ across a patch longer than the escape
   distance): the best achievable lateral deviation under |a| ≤ μ1·g given entry speed v0. Derive it, prove that it
   is a bound, and unit-test it against brute-force point-mass optimization.
2. **Convex relaxation / grid search**, if the closed form is too loose: a point-mass minimum-peak-deviation
   problem over a discretized time grid (convex in a path-parameterized formulation; solve with a QP/SOCP solver),
   reported as a lower bound.
3. **HJ reachability** on a reduced lateral model (`optimized_dp` / `hj_reachability`): stretch only, offline.

Requirement: every certificate method has a written soundness argument in this file *before* any scenario is
labelled infeasible with it.

### 2.1 Implemented certificate v1 (P1): statement and soundness argument
Code: `feasibility/certificate.py` (`certify_arc`, `certify_scenario`). Tests: `tests/test_certificate.py`.

**Claim.** Consider a friction drop on a constant-radius arc R. At t = 0 the front axle reaches the start of the
transition with speed v_a, lateral offset |e| ≤ e0, lateral velocity |ė| ≤ ė0. If `certify_arc` returns
certified, then **every** trajectory of the simulator (any controller, any information) has a footprint corner
beyond the lane edge at some 20 ms lane-check instant before the low-friction window ends.

**Facts about the simulator used (each enforced in code and tested):**
1. Each axle force satisfies |F_i| ≤ μ_i·Fz_i: the tire curves peak at D = μ·Fz (V4/S4), the applied force is
   clipped to the friction circle after tire relaxation, and friction-field ramps blend μ and shape linearly.
2. Fzf + Fzr = m·g exactly (the longitudinal load transfer clamps the acceleration, not the loads).
3. Other horizontal forces: rolling resistance ≤ f_rr·m·g, drag ≤ ½ρC_dA·v².
4. Above |v| = v_hi the low-speed kinematic blend has weight exactly 1 (smoothstep on the *speed*).
5. Semi-implicit Euler: velocity changes are impulses at the old position; between impulses the CoG moves in a
   straight line.
Hence the CoG acceleration obeys |a| ≤ A(t) = μ_max(t)·g + f_rr·g + ½ρC_dA·v_ub²/m, with μ_max(t) ≤ μ0 always and
≤ μ1 while both axles are on the patch.

**Geometry.** No footprint corner is beyond the edge only if the CoG lateral deviation is ≤ w = 1.75 − W/2: the
disc of radius W/2 around the CoG lies inside the footprint rectangle (the CoG is ≥ W/2 from every edge), so a
CoG deviation > w puts a footprint point, hence a corner, beyond the edge. On the arc the CoG corridor is the
annulus R − w ≤ r ≤ R + w around the arc centre.

**Dynamics bound.** In polar coordinates around the arc centre, with h = r²θ̇: r̈ = h²/r³ + a_r and
ḣ = r·a_θ (exact for the continuous motion; for straight-line motion between impulses r̈ = h²/r³ and h is constant;
an impulse changes ṙ by ≥ −A·dt and h by ≥ −r·A·dt). Suppose the CoG stays in the annulus on [0, T]. Then
h(t) ≥ h_lb(t) = h0 − (R + w)∫A and r̈ ≥ g(t) = max(h_lb, 0)²/(R + w)³ − A(t), so ρ = r − R ≥ ρ_lb(t) =
ρ0 + ρ̇0·t + ∬g with ρ0 = −e0, ρ̇0 = −ė0, h0 = (R − e0)·√(v_a² − ė0²) (the worst case for the car), each reduced by
one extra physics step of worst-case change (discretization). ρ_lb is a piecewise polynomial computed in closed
form.

**Phases.** A = A(μ0) until the rear axle is surely past the entry ramp (CoG progress ≥ L + lt + 0.1 m; rear-axle
s ≥ CoG s − lr whatever the heading); progress has the lower bound R·∫h_lb/(R + w)² (θ̇ = h/r²). After that,
A = A(μ1). The window ends when the front axle could reach the exit ramp or the CoG could leave the arc
(progress ≤ R·∫h_ub/(R − w)², h_ub = (R + w)(v_a + ∫A)).

**Decision.** Certified if at some grid time t*: ρ_lb(t*) > w + 1 mm, ρ̇_lb(t*) ≥ 0, g ≥ 0 on [t*, t* + 20 ms] (so
ρ_lb stays above w until the next lane check), t* + 20 ms is inside the window, the speed lower bound
v_a − ∫A stays ≥ v_hi (fact 4 applies) and the speed upper bound stays within the drag headroom used in A.
This contradicts "stays in the annulus on [0, T]", so a lane-check instant with a corner beyond the edge exists.

**Scope (v1).** Arcs only, with the CoG already on the arc at arrival (apex and exit drops). Straight drops are
never infeasible for the point mass (brake in a straight line). Entry drops (clothoid) are not covered yet; they
are labelled by the oracles or stay *undetermined*.

**Conservatism (measured, not a result).** The bound lets the car brake and turn with the full budget at once.
A brute-force point-mass search (`feasibility/pointmass.py`) fails at arrival speeds about 30–40% below the
certified threshold, so arrival speeds in that band are *undetermined*, not certified. Possible tightening
(future): bound the yaw angle in the dry phase to use per-axle grip; a coupled-budget Lyapunov bound; HJ
reachability.

**Checks in the test suite:** the G1 classic cell is certified; never certified when steady circling is possible;
monotone in speed; Monte Carlo random point-mass trajectories never beat ρ_lb; the point-mass search never
survives a certified case; the full simulator with strong heuristic controllers leaves the lane in certified
cases while being in lane at arrival (S1 preview).

### 2.2 What "arrival state" means for labels and S1 (DECIDED 2026-11-08, D16; prereg amendment A3)
The certificate is a function of the arrival state. A clairvoyant controller may brake *before* the patch and
arrive slower, so a scenario certified at the nominal arrival state (v0, on the centreline) can still be
survived by O-C. Proposed: (a) label *certified-infeasible* = the certificate holds at the nominal arrival state
(i.e. "nothing can save a car that arrives at the reference speed without having seen the patch"; a certified
form of reactive infeasibility); (b) check S1 per oracle run at that run's *actual* logged arrival state
(speed, |e|, |ė|), which is sharp and sound for both oracles; (c) O-C successes in certified scenarios are
expected (pre-braking) and are reported, not treated as bugs. The prereg's S1 wording ("no oracle succeeds in a
certified-infeasible scenario") would be refined by a dated amendment before any test-split labelling.
**Update 2026-11-08:** adopted as prereg amendment A3, which also fixes the label
precedence (reactive-feasible > preview-required > certified-infeasible > undetermined) and the
*certified-reactive-infeasible* flag.

## 3. Layer 2: constructive oracles (feasibility witnesses)
- **Reactive oracle:** MPPI with the *exact* truth simulator as its rollout model and the true *current* per-axle
  μ + surface, held constant over its horizon (it does not know what is ahead).
- **Clairvoyant oracle:** the same, but its rollouts use the true μ/surface map ahead.
- Oracle settings are generous: long horizon (≈ 3 s), many samples (≈ 16k), best of N = 5 noise seeds per
  scenario. Settings are frozen before labelling.
- A success by either oracle is a **witness** that the scenario is feasible at that information level.

## 4. Sanity gate S1
(Refined by prereg amendment A3, 2026-11-08: checked per run at the actual arrival state; see §2.2.)
No oracle may succeed in a certified-infeasible scenario. If one does, it is a bug (simulator, certificate, or
metric) and must be fixed before any results are reported.

## 5. Outputs
- Labelled scenario sets (tuning, test) with per-scenario labels + oracle metrics.
- Feasibility-map figures: label share over (v0 × R) for each surface × patch length.
- The *undetermined* share is reported. If it is large, improve the certificate (Layer 1 option 2/3) or the oracle,
  but never re-label without a recorded reason.
