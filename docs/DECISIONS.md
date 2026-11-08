# Decision log

Dated entries. Never edit an old entry's substance; supersede it with a new dated entry.
Statuses: **DECIDED (date)** or **OPEN**.

## D1: Modify the original idea; don't replace it. DECIDED 2026-10-14
The robotics-blueprint "P2 extreme-dynamics slip-aware planner" keeps its question (learned vs model-based
adaptation at sudden friction loss) and drops its headline ("RL beats MPC"). Ranked against a pure C++ MPPI
project (3.80), PufferDrive self-play (3.50), a kinematic→physics fidelity ladder (3.15), and the original as written
(2.70), the modified version scored 4.30 (report §"The modified original ranks first"). Its descope floor *is* the
C++ MPPI project. The old blueprint snippets are treated as nonexistent; everything is
built from this spec.

## D2: Fair baselines, not a strawman. DECIDED 2026-10-14
Constant-μ MPC is a context row only. The real baselines are bank-adaptive MPPI (LLA-MPC style) and
learned-estimator MPPI, with matched information, objective and tuning budget (`BASELINES.md`). Why: LLA-MPC
already comes within ~0.07 s of a known-friction oracle after a 40% drop.

## D3: Per-scenario feasibility labelling with oracles. DECIDED 2026-10-14
Labels: certified-infeasible / preview-required / reactive-feasible / undetermined (`FEASIBILITY.md`). Why: the
original headline cell (μ = 0.1, 15 m/s, R = 50 m) is infeasible for any controller, and published oracles still
show unexplained violations.

## D4: Teacher sees current per-axle μ only. DECIDED 2026-10-14
Preview belongs to the clairvoyant oracle (labelling only). The preview teacher is kept as the H3 ablation.
Why: a proprioceptive student cannot infer friction it hasn't reached; a preview teacher teaches blanket caution.

## D5: Pre-registered hypotheses replace target numbers. DECIDED 2026-10-14
H1–H3 with falsifiers (`PREREGISTRATION.md`, tagged `prereg-v1` at the end of P0). No result numbers anywhere
before the harness produces them.

## D6: Build a small simulator in Warp; JAX gyms are references only. DECIDED 2026-10-14
Why: f1tenth_gym dev-jax and INI jax_pacejka are 1:10 scale with no documented friction randomization, and they
would put JAX next to P1's PyTorch stack. Warp passes data to torch without copies and underpins Newton/Isaac Lab
3.0. The simulator is infrastructure, not a claimed contribution, and throughput is not a headline.

## D7: C++ purpose = CUDA + vehicle controls, in one load-bearing MPPI library. DECIDED 2026-10-14
Supersedes the 2026-10-13 wording ("C++ is a gap"). I already had substantial C++/ROS 2 experience (AUV perception on
Jetson). What was missing was CUDA backed by a project, optimal/sampling-based control, and car-domain work. So C++ goes into
a C++17/CUDA MPPI library (~20% of hours) that the confirmatory sweeps run on. Out: ROS 2 node, TensorRT, C++ RL env.

## D8: Full-size passenger car, flat ground; terramechanics cut. DECIDED 2026-10-14
Cut: heightmap/grade, Bekker soil/terramechanics, CBF monitor (now a post-MVP stretch: a μ-adaptive safety
filter). Why: off-road is a separate field (full-scale labs, Chrono-based sims that aren't GPU-batched), the
question is fully answered on paved surfaces + ice/snow/wet, and bachelor's-level postings rarely mention
terramechanics. Full-size parameters match the permissive real data (comma MIT, MCity-WSPI CC-BY-4.0).

## D9: Public timing and code license. OPEN
Options: public at G2 (Nov 30) if citing it on summer-2027 applications, otherwise at the MVP. License: MIT suggested.

## D10: Own Python environment. DECIDED 2026-10-14
`uv`, Python 3.12, a PyTorch CUDA ≥ 12.8 build, `warp-lang`, pinned `rsl-rl-lib`. Never symlink or reuse P1's IsaacLab venv.

## D11: Surface changes tire *shape*, not only μ. DECIDED 2026-10-14
Each surface has its own (B, C, D = μ, E). Why: real ice/snow change the slip curve's shape. A μ-only model would
make model-bank MPPI artificially well-specified.

## D12: V8 cross-check uses our own scalar NumPy reference; jax_pacejka is optional and partial. DECIDED 2026-10-14
Supersedes the P1 task wording "vs jax_pacejka (or a scalar NumPy reference if the license blocks it)". The license
does not block it (top-level MIT), but the P0 check showed INI's jax_pacejka is a lateral-only Magic Formula with
no combined slip and no wheel-spin states, and its own JAX-vs-ODE parity tests fail at HEAD `6a8a3d4`. So it cannot
cross-check most of our model. V8 = an independent scalar NumPy implementation written from `DESIGN.md` §2 (not
from the Warp code), compared on matched parameters. jax_pacejka may be called as an external black box for the
pure-lateral subset only, and is never vendored (`DATA_AND_LICENSES.md`).

## D13: Vehicle parameters are frozen at G1, not at the prereg tag. DECIDED 2026-10-14
`prereg-v1` is still tagged at the end of P0 (hypotheses, metrics, statistics, budget, sampler spec, seeds). The
`DESIGN.md` §2.3 parameters and the surface MF coefficients are justified with sources in P1 and frozen by one
dated prereg amendment at G1, before any test-split scenario is labelled or evaluated. Why: the parameter research is
P1 work, and freezing before any test-split use keeps the confirmatory analysis clean.

## D14: Actuator delay stays in RL domain randomization; MPPI gets the same preparation. DECIDED 2026-10-14
Conflict: RL DR randomized steering delay 0–40 ms, but delay is H2 mismatch axis 3, and H2 must test conditions
neither side was prepared for. Options were (A) drop delay from DR, or (B) keep it and prepare the MPPI side
equally. Chosen: **B**. RL DR uses extra actuator delay 0–40 ms on both channels; the M-B bank adds delay variants
{0, 20, 40} ms; the M-L estimator also regresses the extra delay. H2 axis 3 moves to +{60, 100, 140} ms, beyond
what either side has seen. Why: keeps the realistic robustness of DR without rigging H2 in RL's
favour. Cost: M-B bank is ~3× larger. Supersedes the "remove delay from DR" proposal.

## D15: Tire surfaces re-sourced; separate longitudinal and lateral curves; understeer via axle stiffness. DECIDED 2026-10-14 (proposed values; frozen at G1 with D13) → approved and frozen 2026-11-08 (prereg A1)
Found in P1: the DESIGN "typical MF coefficients" (MathWorks Simscape table) describe *longitudinal* force only.
Applied to lateral slip they put the ice lateral peak at ~21 deg (measured: 2-4 deg, CRREL). And with equal tire
coefficients on both axles a bicycle model is exactly neutral-steer. Changes:
1. Each surface has separate pure-slip curves (Bx, Cx, Ex) and (By, Cy, Ey), combined by the normalized
   slip-vector rule Fx = (sx/s) Fx0(s), Fy = (sy/s) Fy0(s). This keeps |F| <= mu Fz exactly (S4) and reduces to
   each pure curve; friction-ellipse H2 variant unchanged in spirit.
2. Longitudinal shapes: least-squares fit to Burckhardt (1993) mu-slip curves (`scripts/fit_surfaces.py`,
   max error ~3% of peak). Lateral: dry/wet with Pacejka's typical C = 1.3, C_alpha/Fz = 17/rad, peaks at 7/6 deg;
   ice C = 1.6 with peak at 3 deg and marked drop-off, snow C = 1.0 rising to ~15 deg (CRREL measurements).
3. Peak mu keeps the frozen prereg V7 ranges; only shapes change.
4. Per-axle lateral stiffness scales 0.80 / 1.52 give an understeer gradient of ~2.0 deg/g on dry (typical 2-3
   deg/g); to be sanity-checked on comma data in P5.
Consequences: dry/wet steady cornering limit within 3% of sqrt(mu g R) (V2b); ice now ~0.96, snow ~0.92 (by
design: snow lateral force approaches its peak only at large angles). Sources still to strengthen: primary CRREL report text, a published lateral MF parameter set for a passenger tire.

## D16: Certificate labels use the arrival state; S1 checked per run at the actual arrival state. OPEN (proposed 2026-10-14) → DECIDED 2026-11-08 (prereg A3)
See `FEASIBILITY.md` §2.2. Needed because a clairvoyant oracle can pre-brake and arrive slower than the reference
speed, so "no oracle succeeds in a certified-infeasible scenario" is not the right test for an arrival-state
certificate. Requires a dated prereg amendment (S1 wording) before any test-split labelling.
**Update 2026-11-08:** approved. Adopted as prereg amendment A3, which adds a label precedence rule
(reactive-feasible > preview-required > certified-infeasible > undetermined) and the *certified-reactive-infeasible*
flag, so a scenario certified at the nominal arrival state but survived by a pre-braking O-C has exactly one label.

## D17: G1 freezes are enforced by hashes in CI. DECIDED 2026-11-08
Prereg A1/A2 record SHA-256 hashes of the frozen parameters (`sim.params.fingerprint()`) and of the committed
scenario files (`splits/`). `tests/test_frozen.py` checks them on every CI run, so a frozen value cannot change
without a failing test and a new dated amendment. The committed split files are the authority (not regeneration),
because a future numpy could change the random stream. `.gitattributes` marks them `-text` so line endings never
change their bytes.

## D18: GPU throughput measurement and the oracle compute budget move to the start of P2. DECIDED 2026-11-08
Prereg V2/V7 ask for P1 throughput measurements (PPO run length; oracle labelling cost). The GPU was busy with an
another project eval run during the G1 session (project rule 1), and the oracle cost depends on the MPPI kernel that
P2 builds. Both measurements are made at the start of P2, and any K/horizon (V7) or G (V2) amendment is dated
before any labelling or tuning starts. Planning projection (not a result): oracle labelling as specified
(2 oracles × 5 seeds × 16k samples × 3 s horizon on 4,000 scenarios, 1 ms substeps in every rollout) is on the
order of 10^15 substeps, which is very likely too large for one laptop GPU. Options to weigh then: start MPPI only
shortly before the drop (steady tracking before it is not informative), shorter episodes after the patch, fewer
samples, or a coarser rollout substep for the oracles (which would need its own validation).
Same for the V4 note "above the simulated IMU noise floor (to confirm in P1)": the observation model does not
exist yet, so it is built and checked in P2. P2 gets both as its first items.


## D19: Shared prefix driver and a handover 4 s before the drop. PROPOSED 2026-11-08 (needs a prereg amendment before test labelling)
Every scenario starts at s = 0 (sampler header), but drops lie up to ~700 m down the road (up to ~70 s at 10 m/s).
Running oracle MPPI for the whole dry approach is the dominant cost and tells us nothing. Proposal: a fixed,
deterministic path follower (`control/episode.py: prefix_action`, curvature feedforward + PD, curvature speed
profile at 0.7·μ0·g with 0.4 g braking) drives every scenario on dry road until the front axle is 4 s (at the
current speed) from the drop. That handover state is shared by every method (paired). 4 s exceeds the 3 s oracle
horizon, so no controller could have seen the drop before handover. Side benefit: scenarios whose v0 is above the
*dry* curve limit (e.g. 25 m/s on R50) arrive at a speed a competent driver would have; without the profile they
crash on dry road before the ice. Episode end, success and termination rules are unchanged (DESIGN §3.2/§4).
Cost of not doing it: the oracle labelling bill grows roughly 3–5×.

## D20: The path follower W-PD counts as a witness of reactive feasibility. PROPOSED 2026-11-08
Found in P2: the plain prefix path follower (no friction information) survives some tuning scenarios that O-R
fails. O-R assumes the current μ persists over its horizon. On short patches that makes it over-react (hard
braking into a slide), and on ice it over-steers past the tire peak, which spins it when grip returns. Any
controller with at most reactive information that survives is a sound witness that the scenario is feasible
without preview. Proposal: reactive-feasible = O-R succeeds (≥ 1 of 5 seeds) **or** W-PD succeeds. O-R stays the
regret reference. Both label variants are computed and logged (`label`, `label_d20`). Effect on the tuning split
at the proposed settings: 1 of 200 labels changes (preview-required → reactive-feasible). Alternative: keep A3's
O-R-only rule and report W-PD separately.

## D21: Oracle settings and compute (resolves D18 if approved). PROPOSED 2026-11-08
See `docs/AMENDMENT_A4_DRAFT.md`, chosen on the tuning split only:
- H = 3 s, K = 1024, σ = (0.05, 0.3), λ = 10, PD ancillary sample;
- ρ = 0 for O-R and ρ = 0.9 for O-C;
- O-C only where no reactive witness exists.

K-check: K = 4096 changed 2 of 200 labels (one each way). Projected test-pool labelling ≈ 3 h on the shared laptop
GPU. Also proposes the observation-noise levels (`PARAMETERS.md` §7). Becomes prereg amendment A4 once approved;
nothing in the test pool is labelled before that.

