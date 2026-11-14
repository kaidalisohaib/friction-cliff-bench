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


## D19–D21 resolved. DECIDED 2026-11-08 (in-session
- **D19 handover: approved with the v_eff fix.** The P2 self-review (LOG 2026-11-08) found that v0 is unreachable
  on dry road in about 10% of cells. A scenario's effective reference speed is therefore
  v_eff = min(v0, √(0.7·μ0·g·R)) (`scenarios/sampler.py: v_eff`). v_eff is the objective's v_ref, the
  certificate's nominal arrival speed, and the speed axis of figures. v0 stays the sampled cell value.
  Rejected alternative: dropping those cells, which would edit the frozen pool.
- **D20 W-PD witness: approved.** It is a sound existence proof. The official `label` counts W-PD, and
  `label_or_only` (the A3 rule) is logged next to it.
- **D21 oracle settings: approved after the K = 16k check** on the non-RF tuning scenarios (D22). Result and
  final K: prereg A4.4.

## D22: K = 16k check before freezing the oracle K. DECIDED 2026-11-08
"Preview-required" only means "O-R failed N times", so a weak O-R inflates that class. Before freezing K, O-R and
O-C are rerun at K = 16384 (the prereg V7 number) on every tuning scenario that is not reactive-feasible at
K = 1024.
- Rule fixed in advance: if 16k turns at most 2 of them reactive-feasible, keep K = 1024.
- Otherwise freeze the larger K, if the test-pool compute stays ≤ 12 GPU-h; if not, freeze 4096.

## D23: S1 probe set. DECIDED 2026-11-08
`scripts/s1_probe.py`: 96 apex snow/ice scenarios from their own seed, not part of any split. A dry driver at
0.9·μ0·g carries cars into the drop fast, and W-PD, O-R and M-N (2 seeds) run on them. S1 passes only if at least
30 runs arrive in a certified state and none of those succeeds.
- First run 2026-11-08: 480 runs, 278 certified arrivals, 0 successes → PASS. This is engineering, not a result.
- The probe is rerun with every labelling run.

## D24: State information for MPPI methods; vy in the observations. DECIDED 2026-11-08
All MPPI methods (O-R, O-C, M-N, M-B, M-L) start their rollouts from the true vehicle state. This is perfect state
estimation, as in LLA-MPC; state estimation is out of scope.
- The adaptive MPPIs infer friction and delay only from the noisy observation history.
- So that the policies are not disadvantaged on sideslip, every non-privileged method also observes a noisy
  lateral velocity vy (0.1 m/s; `PARAMETERS.md` §7).
- The residual asymmetry (MPPI sees the exact rest of the state) favours the MPPI side. The report states it.

## D25: Domain randomization of tire shape. DECIDED 2026-11-08
BASELINES §4 says "tire shape (B, C, E) ±15%". The simulator exposes only a per-axle B scale as a per-car
parameter, so training DR randomizes B_scale_f and B_scale_r independently by ±15%; C and E are not randomized.
The other DR ranges are as written.

## D26: Own minimal PPO instead of rsl-rl-lib; RL arm details. DECIDED 2026-11-08
`rl/ppo.py` is a small PPO (GAE, clipped objective, asymmetric critic) on the GPU env `rl/env.py`. Reasons: it
avoids a pinned dependency and its API churn, keeps everything zero-copy with Warp on one CUDA stream, and every
line is tested and explainable.
- Reward = −(V1 objective with the exact footprint violation) · reward_scale, with −C_fail on termination. No
  shaping.
- Time-limit truncations are treated as terminal. The episode end is a calm post-patch drive, so the bias is
  small.
- Teacher latent z is 8-dimensional. The student is a GRU over the W-step feature history; its z is regressed onto
  the teacher's z with the actor frozen, on the student's own rollouts.
- Budget split inside P-S (and P-SP): 14 teacher trials + 6 student trials = 20 trials, and GPU-h pro rata.
  P-DR, M-B and M-L get 20 trials each.
- Each PPO trial is fixed at 200M environment steps (raised from 60M after pilot G, LOG 2026-11-08).
- Allowed history windows for every adaptive method (P-S, P-SP, M-B, M-L): W ∈ {5, 12, 25, 50} steps
  (0.1–1.0 s, the E1 set), tuned within each method's budget (prereg V8: "W tuned within the shared allowed set").
- M-B and M-L use the oracles' MPPI compute (K = 1024, H = 150) and tune σ, λ, ρ and W. M-N gets the best M-B
  MPPI settings (V2).

## D27: Behavior-cloning warm start for every PPO actor. DECIDED 2026-11-08
Found in P3 pilots, which ran on the training library only: PPO from scratch never completed an episode in 21M
steps. The cause is the objective itself. Once the car is off-centre, the lane-violation cost (up to 100 per step)
makes crashing early cheaper than surviving, so random exploration never learns to drive a curve.
- The same GPU env driven by the friction-blind path-follower law succeeds in 66% of training episodes. So the env
  is solvable and the issue is exploration.
- Fix: before PPO, every actor (P-T, P-TP, P-DR; the students inherit their teacher's actor) is regressed for 2,000
  steps onto the noise-free shared path follower, on states that follower visits with action noise
  (`rl/ppo.py: bc_warm_start`). Then PPO trains with the unchanged V1 reward.
- The follower has no friction information, so the warm start teaches lane keeping only. All friction adaptation
  is learned by PPO.
- Identical for every PPO variant, so it is not a fairness asymmetry; the MPPI side gets the same follower as its
  ancillary sample (D20/P2). Disclosed in the report.

## D28: Tuning resume must not repeat a configuration; a repeat costs GPU-h but is not a trial. DECIDED 2026-11-11
Found on the 2026-11-11 resume. Replaying the ledger restarted the TPE sampler's RNG, so the first new trial after
each resume redrew an earlier configuration: M-L trial 14 repeats trial 10, and P-T trial 1 repeats trial 0 (the
PPO run reproduced the same objective exactly, 0.0113, which is how it showed up).
- Fix (`eval/tuning.py`): after replay, the sampler seed is offset by the number of replayed trials, so the
  sequence is still deterministic but does not restart. Two regression tests.
- Budget rule (prereg V2 says 20 trials = 20 evaluated configurations): a repeated configuration still counts its
  GPU-seconds toward the GPU-hour cap, but not toward the trial count. M-L and P-T therefore each get one
  replacement trial; their duplicate lines stay in the ledger.
- Applies to every method equally (M-B never resumed, so it has 20 distinct configurations already).

## D29: C++17/CUDA MPPI backend: C ABI + ctypes, Warp-identical RNG, S3 tolerances, used for every MPPI method in the confirmatory runs. DECIDED 2026-11-11
Built in `mppi_cpp/` (BASELINES §5), for the confirmatory runs.
- **Shared source for CPU and GPU.** The dynamics, track projection, friction field, cost, path follower and every
  MPPI kernel body are `__host__ __device__` functions (`mppi_cpp/include/fcb/`). The CUDA driver maps them onto
  threads, and the CPU driver onto loops for GoogleTest. The weights and update steps use block reductions on the
  GPU when K ≥ 256.
- **Bindings: a C ABI loaded with ctypes** (`control/mppi_cpp.py`), not nanobind/pybind11 as BASELINES §5
  suggested. The library reads Warp arrays in place via raw pointers, with no copies, and launches on Warp's
  current stream. This needs no build-time Python dependency. `CppMPPI` subclasses the Warp `MPPI` and replaces
  only `act()`.
- **RNG.** The port copies Warp's PCG hash and Box-Muller normal bit for bit (checked against `wp.randn`). With the
  same seeds, both backends therefore draw identical samples, and S3's "fixed noise" holds by construction.
- **S3 tolerances (the prereg names none):**
  - first control step from identical inputs: the random samples must be bit-identical, the action within 1e-4,
    the plan within 1e-3;
  - second step, whose inputs already differ by rounding: action 2e-3, plan 2e-2.
  - Measured on 1,000 perturbed tuning-handover states (O-R and O-C settings), the first-step action differences
    were ≤ 1.3e-5. The remaining differences are nvcc-vs-NVRTC rounding; both compile with fused multiply-add on and
    no fast math.
- **Closed-loop consequence:** rounding differences grow over an episode. On 32 tuning scenarios, O-R had the same
  outcome on both backends (32/32), with min clearance differing by up to 4.4 cm. So a comparison must run every
  method on one backend.
- **Rule:**
  - All MPPI-based methods (O-R, O-C, M-N, M-B, M-L) in the confirmatory test-split evaluation use the C++ backend
    (`MPPI_BACKEND=cpp`), as the project design states.
  - The P2 feasibility labels stay as computed by the Warp oracles: labelling is not a method comparison, and S3
    shows the two implement the same controller.
  - Tuning (already running on Warp) is not redone. The tuned parameters carry over because the controller is the
    same.

## D30: Evaluation plan fixed before test evaluation (prereg A5). DECIDED 2026-11-11
What A5 adds to V6/V7/EVAL_PROTOCOL and why:
- **H2 subset of 500 reactive-feasible scenarios.** The full set would be 9 levels × ~3,000 scenarios × 5 methods
  × 5 seeds, i.e. hundreds of GPU-hours. With 500 scenarios × 5 seeds, a 5 pp regret difference is resolvable, and
  every level uses the same instances, so the slope is paired.
- **"Better of M-B/M-L" chosen on the test set by lower departure.** The alternative was choosing on the tuning
  objective. Taking the minimum can only shrink (better MPPI − P-S), so it cannot help falsify H1. The tuning-based
  choice would also have picked M-B (0.039 vs 0.0725), so the two rules usually agree.
- **Dry set built from the H2 subset by removing the drop.** EVAL_PROTOCOL defines dry pace on "dry-only
  scenarios" without defining that set. Reusing the H2 instances keeps the pairing, and the geometry and episode
  timing are unchanged.
- **p-values in the claimed direction** (1 otherwise), so a wrong-signed effect never counts toward Holm.
- **Recovery latency** is moved to an exploratory pass. It needs per-step traces of every run, and it is not in
  the Holm family.


## D31: H2 scenarios the shared driver cannot hand over are excluded per axis (prereg A6). DECIDED 2026-11-11
- **Found by:** the evaluation dry run on the tuning split (first 60 scenarios, stand-in checkpoints). At +140 ms
  delay, `run_prefix` did not reach handover for one scenario (U-00032, R400 exit, 24.9 m/s). The PD + feedforward
  driver was never designed for 160 ms of total delay at that speed.
- **Options considered:**
  1. **Drop the scenario from that level only.** Rejected: the levels would no longer share one scenario list, and
     the paired slope bootstrap (A5.6) needs one.
  2. **Count it as a failure for every method.** Rejected: it adds zeros to every regret difference and hides the
     fact that the scenario never reached the methods.
  3. **Hand over on the nominal truth, then switch the plant.** Rejected: it changes A5.6 ("the handover drive
     uses the mismatched truth"). It also needs the delay buffer resized mid-run, which is a new code path whose
     only purpose would be this case.
  4. **Exclude per axis (chosen).** Paired, decided before any method runs, reported. Expected to be rare (1/43
     at the worst delay level on tuning data).
- **Implementation:** `control/episode.py: run_prefix(strict=False)`, `eval/runner.py: axis_exclusions`. The analysis
  asserts that every level of an axis has the same scenario list. `scripts/eval_split.py` requires A6 in the
  prereg before a test evaluation.

## D32: Separate adaptation from caution with pre-declared exploratory analyses (prereg A7). DECIDED 2026-11-11
- **Found by:** running the tuned P-T and P-S checkpoints through the evaluation runner on tuning data and adding
  an arrival-speed column. The policies arrive at ~0.65 × v_eff; O-R arrives at ~1.0 and M-B at ~0.9. They
  depart less on RF scenarios, and they survive about half of the preview-required/undetermined ones, where every
  MPPI method fails.
- **Why it happens:** every episode in the training library and in the benchmark has a friction drop ~4 s after
  handover. RL learns the prior "a drop is coming". The MPPI cost is the same, but a planner with a dry-road (or
  current-friction) model sees no reason to slow down.
- **Options considered:**
  1. **Retrain the policies with drop-free episodes.** Rejected: tuning is finished, and changing the training
     distribution now would void the logged budget. The test distribution itself always has a drop, so the
     prior is not wrong for this benchmark.
  2. **Change H1 to a speed-adjusted metric.** Rejected: that edits a confirmatory hypothesis after seeing
     tuning behaviour.
  3. **Keep H1–H3 and add exploratory, pre-declared analyses (chosen).** E6 measures caution. E7 gives the bank
     MPPI the same speed, with c fixed from tuning data, and states the interpretation rule in advance.
- **Note for the write-up:** this is the kind of artefact the benchmark exists to expose. A "RL wins" headline
  without E6/E7 would be misleading.

## D33: S2 failed on two collapsed PPO seeds; add collapse rollback, retrain all PPO seeds (prereg A8). DECIDED 2026-11-11
- **Evidence:**
  - S2 departure was 0.195 pooled; the per-seed values were 0.010 / 0.182 / 0.000 / 0.039 / 0.746.
  - The logs of seeds 1 and 4 show the known P3 failure mode: success falls to ~0 around iterations 100–130 of
    381 with exploding returns (−4,700, −9,400), and it never comes back. Target-KL stopping, gradient clipping,
    LR decay and best-checkpoint selection were all already on.
- **Options considered:**
  1. **Drop or replace the bad seeds.** Rejected: that is best-of selection (EVAL_PROTOCOL §3).
  2. **Go ahead with S2 failed.** Rejected while a cheap, principled fix exists. V5 calls this a training bug.
  3. **Retune with multi-seed scoring.** Rejected: it would double the RL tuning budget after the fact, and the
     MPPI arm got no such second round.
  4. **Collapse rollback (chosen).** A standard divergence guard, applied identically to every PPO run. It never
     triggers on a healthy run, so tuning results carry over.
- **Fairness note.** This adds engineering effort to the learned arm after seeing tuning data. The MPPI arm also
  had bugs fixed after tuning-data runs (e.g. D28). Human-hours are logged, not capped (V2). The trial and
  GPU-hour budgets are unchanged.

## D34: The test evaluation runs at batch = 64, the tuning batch size (engineering, no amendment). DECIDED 2026-11-11
- **Finding:** before launching the test evaluation, a check on 16 tuning scenarios compared batch 4 against
  batch 16.
  - O-R (pure MPPI) records are identical: the MPPI noise is seeded per car (`run_seed`).
  - M-B and P-S records differ. The observation-noise generator in `rl/features.py` is seeded once per batch,
    so a car's noise realisation depends on which cars share its batch.
  - Repeating any method at the same batch size reproduces it bit for bit.
- **Why this is not a validity problem:** the noise is i.i.d. per car with the same distribution in any batch.
  No method sees a different noise *law*; only the realisation changes, like a different seed.
- **Decision:** keep the code. Run the test evaluation at the default batch = 64, the batch size tuning used
  (`tune_method.py`). Results are then reproducible from the commit, the checkpoints and that batch size.
  - Making the noise per car would change the noise realisations relative to tuning. It would also mean editing
    controller code just before the test run, for no gain in validity.
- **Write-up:** the reproducibility statement names the batch size.

## D35: RL follow-up phase: diagnose and fix PPO training on training/tuning data only, then a pre-registered re-test on a fresh pool (A9). DECIDED 2026-11-13
- **Goal:** improve the learned arm, so that the RL result is a solved problem rather than an
  unstable one.
- **What the confirmatory record already holds:** H1–H3 and E1–E7 on test v1 stand as reported. Nothing in this
  phase changes or re-analyses them.
- **Rules for this phase:**
  - diagnosis and recipe work use only the training library and the tuning split (`scripts/rl_diag.py`,
    `scripts/rl_eval_tuning.py`; outputs in `runs/rl_fix/`);
  - test v1 is never used again for a decision;
  - a new claim needs amendment A9: hypotheses, recipe, comparator and a fresh test pool (v2). A9 is committed
    before any v2 evaluation; it is tagged `prereg-v2`.
- **Parity:** extra engineering on the RL arm is logged. The MPPI arm gets the matching extra budget under A9.
