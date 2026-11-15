# PREREGISTRATION: **DRAFT** (values approved 2026-10-14; frozen when tagged `prereg-v1`)

Freeze procedure (end of P0): resolve every open value, fill the test-split spec, commit, `git tag prereg-v1`, push
the tag.
After tagging, changes go in a dated amendment section at the bottom. Original text is never edited.

## Research questions
- **RQ1.** On reactive-feasible friction drops, does RMA-style policy adaptation beat adaptive MPPI when both get the
  same information and tuning budget?
- **RQ2.** How does each degrade when the truth's tire/actuator *structure* leaves the MPPI model family (and the RL
  training distribution)?
- **RQ3.** What privileged information should the teacher see: current friction only, or a preview?

## Confirmatory hypotheses
| ID | Hypothesis | Falsifier |
|---|---|---|
| **H1** | On reactive-feasible, in-family scenarios, P-S's lane-departure rate is **not lower than the better of M-B/M-L by more than 5 pp** (equivalence-or-worse) | The paired difference (best adaptive MPPI − P-S) has a 95% CI entirely above +5 pp, aggregated over ≥ 5 training seeds |
| **H2** | As the truth departs from the nominal family along the pre-specified mismatch axes, adaptive-MPPI regret grows faster than P-S/P-DR regret, and **the curves cross within the swept range** | No crossing up to the largest mismatch level, or the student's regret slope ≥ MPPI's |
| **H3** | A student distilled from the current-μ-only teacher (P-S) matches or beats the preview-teacher student (P-SP) on departure rate and dry pace | P-SP has a lower departure rate (CI excludes 0) while losing < 2% dry progress |

Note on H1: we *predict* that the "RL wins" headline does not survive a fair adaptive baseline. A clear RL win
would falsify H1 and is a reportable result.

## H2 mismatch axes and levels (values and test rule in "Frozen values" V6)
1. Combined-slip formulation: slip-vector method (nominal) → friction-ellipse scaling, as a blend α ∈ {0, 0.25, 0.5, 1}.
2. Tire relaxation length: 0 (nominal) → {0.2, 0.4, 0.6} m.
3. Actuator delay: nominal → +{60, 100, 140} ms (D14).
Every H2 test level lies outside both the RL domain-randomization support and the MPPI bank: axes 1–2 are absent
from both; extra delay is covered by both only up to +40 ms (D14).

## Exploratory (no confirmatory claims)
- E1: history window W ∈ {0.1, 0.25, 0.5, 1.0} s for P-S, M-B, M-L.
- E2: performance vs per-step compute budget.
- E3: RMA latent regression vs DAgger action distillation.
- E4: does recovery latency track the axle transit time L/v (front-axle slip as a "free preview" for the rear)?
- E5 (stretch): the sim-trained μ estimator on MCity-WSPI logs vs MARWIS friction.

## Sanity gates (failures = bugs, not results)
- S1: no oracle succeeds in a certified-infeasible scenario.
- S2: the teacher is within 5 pp of O-R's departure rate (V5) on reactive-feasible tuning scenarios.
- S3: C++ MPPI ≡ Warp MPPI (fixed noise) on 1,000 random states.
- S4: |F| ≤ μF_z on the dense slip grid for every surface.

## Fixed in advance
- Metrics and statistics: `EVAL_PROTOCOL.md` as of the tag.
- Objective weights: V1 below (also in `DESIGN.md` §4).
- Tuning budget per method: N = 20 Optuna trials and G = 15 GPU-hours (V2).
- Marginal band 0.30 m (V3) and recovery-latency band (V4).
- Test split: sampler spec + seeds in V7; SHA-256 of the generated set added by dated amendment in P1, before any
  test-split labelling or evaluation; ≥ 1,000 reactive-feasible scenarios (headline).
- Vehicle parameters and MF coefficients: frozen by dated amendment at G1 (D13).

## Frozen values (settled 2026-10-14; frozen at the `prereg-v1` tag)
Proposed and settled 2026-10-14. Nothing here is a result.

**V1. Objective weights** (`DESIGN.md` §4; e_y and lane_violation in m, u normalized to [-1, 1], per 20 ms step).
Rule: each term is O(0.1–1) per step at a "meaningful" magnitude, and failure dominates any recoverable excursion.
| Weight | Value | One-line justification |
|---|---|---|
| w_ey | 1.0 m⁻² | e_y = 0.5 m (60% of the 0.83 m in-lane slack) costs 0.25/step: noticeable, not dominant |
| w_lane | 100 m⁻² | 0.1 m beyond the edge costs 1.0/step (4× the in-lane cost at e_y = 0.5 m); 0.5 m costs 25/step |
| w_v | 2.0 | a 20% speed deficit costs 0.08/step: slowing is allowed and cheap, but not free |
| w_du | 0.1 | full-scale action jump costs 0.1/step; normal smooth driving (Δu ≈ 0.05) is negligible |
| w_prog | 0.1 m⁻¹ | at 20 m/s (0.4 m/step) worth 0.04/step: only stops "park and wait" from being optimal |
| C_fail | 2000 | ≥ 2× the cost of a deep recoverable excursion (0.5 m out for 0.75 s ≈ 940), so recovery always beats termination |
Consequence to accept explicitly: blanket 30% slowdown for a 10 s episode costs ≈ 95, so a controller should prefer
caution whenever it believes departure risk > ~5%. This ratio (C_fail vs w_v) sets the risk–pace trade-off for
every method, and the dry pace tax metric exposes it.

**V2. Tuning budget** (per controller ID with tunables: M-B, M-L, P-S, P-DR, P-SP; O-R/O-C use frozen generous
settings, M-N gets the M-B MPPI settings). **N = 20 Optuna (TPE) trials and G = 15 GPU-hours, whichever binds first.**
Rationale: TPE needs ≈ 20 trials to beat random search meaningfully; 15 GPU-h ≈ 20 PPO runs of ≈ 45 min. P-S's
budget includes its teacher's trials; M-L's includes estimator training. If P1 throughput shows one PPO run > 0.75
GPU-h, amend G (dated, before any tuning starts). Human-hours are logged, not capped.

**V3. Marginal band** (`FEASIBILITY.md` §1): reactive-feasible with **O-R min lane clearance < 0.30 m**
(best O-R seed). Justification: ≈ 36% of the 0.83 m slack between car side and lane edge on a straight; small enough
that most reactive-feasible scenarios stay non-marginal, large enough to exclude knife-edge cells.

**V4. Recovery-latency band** (`EVAL_PROTOCOL.md` §1): **|e_y − e_y^{O-R}| ≤ 0.15 m and |r − r^{O-R}| ≤ 0.05 rad/s,
sustained 0.5 s**, time-aligned to the same scenario instance. 0.15 m = half the marginal band; 0.05 rad/s ≈ 3°/s
≈ 25% of the yaw rate at 20 m/s on R = 100 m, and above the simulated IMU noise floor (to confirm in P1).

**V5. S2 margin:** teacher P-T departure rate ≤ O-R's **+ 5 pp** (point estimate, pooled over ≥ 5 seeds, on ≥ 200
reactive-feasible tuning scenarios). Same scale as the H1 margin; P-T has O-R's information, so a larger gap
means the policy class or training is broken, not that adaptation is hard.

**V6. H2 levels** (all strictly outside RL DR and the M-B bank; axis 3 revised per D14):
1. Combined slip: force = (1 − α)·F_slip-vector + α·F_friction-ellipse, **α ∈ {0, 0.25, 0.5, 1}**. A convex blend of
   two vectors inside the friction circle stays inside it, so S4/V4 and the certificate remain sound.
2. Tire relaxation length: **0 → {0.2, 0.4, 0.6} m** (`DESIGN.md` range).
3. Actuator: extra pure delay on both channels **+{60, 100, 140} ms** (RL DR, the M-B bank and the M-L estimator
   all cover 0–40 ms, so both sides are equally prepared and every test level is a surprise to both).
H2 test (per axis): regret = paired departure-rate difference vs O-R (O-R always uses the true model). Axis supported
if (a) the regret slope over level index (best adaptive MPPI − P-S) > 0 with 95% bootstrap CI excluding 0, **and**
(b) at the largest level, regret(best adaptive MPPI) − regret(P-S) > 0 with CI excluding 0. **H2 supported if ≥ 2 of 3
axes pass.** p-values for Holm: per axis, p_axis = max(p_a, p_b) (intersection–union, both criteria required);
for H2, p_H2 = min(1, 2 × second-smallest p_axis). The factor 2 is needed because, if only one axis truly has an
effect, either of the two null axes can supply the false second pass.
Delay handling follows D14: delay stays in RL DR, the MPPI side gets the same 0–40 ms preparation, H2 tests only beyond it.

**V7. Test-split sampler spec (v1).**
- Cells (full factorial, 1,872): v0 ∈ {10, 15, 20, 25} m/s × geometry {straight; R ∈ {50, 100, 200, 400} m × drop at
  {entry, apex, exit}} (13) × post-drop surface {wet, snow, ice} × patch length {10, 30, 100, ∞} m × transition
  {step, ramp 2 m, ramp 5 m}.
- Within a cell (continuous jitter): v0 + U[−1, 1] m/s; μ0 ~ U[0.85, 1.0] (dry shape); μ1 ~ U[wet 0.5–0.8 | snow
  0.20–0.35 | ice 0.08–0.15] with that surface's nominal (B, C, E); both axles see the same field (front meets it
  first); initial condition = steady-state tracking at v0 on dry; truth vehicle parameters = nominal (`DESIGN.md`
  §2.3, no DR at test time); H2 sets reuse the same instances with the mismatch applied.
- Pool: **4,000 scenarios**, cells drawn uniformly at random with replacement, then jitter. Top-up blocks of 1,000
  (seed + 1, + 2, + 3, pre-declared) are added until reactive-feasible ≥ 1,000; max 3 top-ups; whatever n the other
  classes reach is reported with Wilson CIs. Headline = all reactive-feasible scenarios pooled; per-stratum curves.
- **Seeds (verifiable, nothing-up-my-sleeve):** test = int(sha256("friction-cliff-bench test split v1")[:8], 16) =
  **4019369836**; tuning = int(sha256("friction-cliff-bench tuning split v1")[:8], 16) = **4210175585**. RL training
  scenarios use the tuning sampler with seeds derived the same way ("… train v1"); instance overlap with the test set
  is measure-zero (continuous jitter) and is checked by hash anyway.
- SHA-256 of the generated set: recorded in a dated amendment in P1 when the generator exists, **before** any test
  scenario is labelled or evaluated (the spec and seed are frozen by the tag, so the hash adds no freedom).
- Compute note: oracle labelling (2 oracles × 5 seeds × 16k samples × 3 s horizon) on 4,000+ scenarios is the
  biggest GPU job in the project; measure throughput in P1 and, if needed, amend the oracle K/horizon *before*
  labelling (never after).

**V8. Other choices fixed at the tag:**
- **Drivetrain: RWD** default. Drive-induced rear saturation is the hardest and most adaptation-sensitive mode, and
  the dynamic-bicycle models in the closest work (LLA-MPC, Kalaria) are rear-driven. FWD/AWD exploratory only.
- **M-B selection: argmin** (LLA-MPC Eq. 7), not softmin: keeps M-B the published method; W tuned within the
  shared allowed set.
- **Parameter-freeze timing (D13).** `DESIGN.md` said parameters and MF coefficients are
  "justified in P1 and frozen before `prereg-v1`", but the tag is at the end of P0. Resolution: tag at P0 as planned;
  `DESIGN.md` §2.3 parameters + MF coefficients are frozen by one dated amendment at G1, before any test-split
  labelling (D13).

## Amendments (post-tag, dated)

### A1 (2026-11-08): vehicle parameters and tire surfaces frozen (G1, D13, D15)
Adopted 2026-11-08, before any test-split scenario was labelled or evaluated.
- Frozen: every `VehicleParams` default, the four `Surface` tables (separate longitudinal/lateral Magic Formula
  curves, D15), `MU_RANGES` (unchanged from V7), and the numerical constants (g, air density, 20 ms control step with
  20 × 1 ms substeps, max delay 8 steps), exactly as in `sim/params.py` at this commit.
- Fingerprint: SHA-256 of `json.dumps(sim.params.frozen_payload(), sort_keys=True)` =
  **b1e0ddceb9246f5b682c475def8a2bf9e178a4d48a0e7e91432ba89716d8cc21** (checked in CI by `tests/test_frozen.py`).
- Sources and plausible ranges for every value: `docs/PARAMETERS.md`.
- What this does *not* freeze: the truth-model switches stay at their nominal values (σ_relax = 0, α = 0) except
  in H2 sets, where V6 applies; DR ranges for training are in `BASELINES.md` §4.

### A2 (2026-11-08): scenario sets generated and hashed (V7)
Generated by `scripts/setup/make_splits.py` from the V7 spec and seeds; committed under `splits/`. No scenario in them has
been labelled or evaluated. SHA-256 of the committed files:
| File | Content | SHA-256 |
|---|---|---|
| `test_v1_T0.jsonl` | test pool, 4,000 | e742af7f058c322c461fb929d6b1d23f4a2d99aa9d8d0fee3e10c782f51a82fc |
| `test_v1_T1.jsonl` | top-up block 1, 1,000 (used only if needed, in order) | bb003d36c03235e61763a12510a4cc0b1fe2bcf482fe811a6227989de7a379bb |
| `test_v1_T2.jsonl` | top-up block 2, 1,000 | 9deb033a547ae50c824a1d340046e988056b80c356cbcb361d26f75c39879cb6 |
| `test_v1_T3.jsonl` | top-up block 3, 1,000 | 2f1c4f4693216af3f2aa37a58b660c164b08757104a554c5c6acfaf351ba612e |
| `tuning_v1.jsonl` | tuning split, 200 | 5dd1e7e7d8038c1ceb2aa53423f8e1f70946d72b21df293f505afa5cd0b14a63 |
The committed files are the authority. If a dependency update changes the regenerated stream, the files are not
regenerated. The tuning set shares no instance with any test block (checked).

### A3 (2026-11-08): arrival-state semantics for feasibility labels and gate S1 (D16)
Adopted 2026-11-08. Refines the S1 wording; the intent ("an oracle beating the certificate is a bug")
is unchanged.
- **Label.** *Certified-infeasible* means the certificate holds at the scenario's *nominal* arrival state: speed v0,
  on the centreline, no lateral velocity, i.e. a car that has not seen the patch and arrives at the reference
  speed cannot be saved. This is a certified form of reactive infeasibility.
- **S1 (refined).** For every oracle run, the certificate is evaluated at that run's *actual logged* arrival state
  (speed, |e|, |ė| when the front axle reaches the transition). If it certifies infeasibility and the run
  nevertheless stays in the lane, that is a bug (simulator, certificate, or metric) and is fixed before any result is
  reported.
- **Consequence.** A clairvoyant-oracle success in a certified-infeasible scenario is *expected* when that oracle
  slowed down before the patch (its own arrival state is then not certified). Such successes are reported, not
  treated as bugs, and they are what makes a scenario *preview-required*.
- **Precedence** (so every scenario gets exactly one label): reactive-feasible if O-R succeeds (≥ 1 seed); else
  preview-required if O-C succeeds; else certified-infeasible if the certificate holds at the nominal arrival
  state; else undetermined. A preview-required scenario that is also certified at the nominal arrival state carries
  the flag *certified-reactive-infeasible* (proven to need preview). Because of S1, O-R can succeed in a
  certified scenario only by arriving in a non-certified state, and such runs are listed for inspection.

### A4 (2026-11-08): episode handover, effective speed, witnesses, oracle settings, information, tuning blocks (D19–D27)
Decided 2026-11-08. Adopted **before any test-split scenario was labelled or evaluated**. Everything below was chosen on the
tuning split only. Rationale: `DECISIONS.md` D19–D27 and the LOG entries of 2026-11-08.
- **A4.1 Handover (D19).** Every scenario is driven from s = 0 by the shared deterministic dry driver
  (`control/episode.py: prefix_action`; curvature feedforward + PD; speed profile at 0.7·μ0·g lateral, 0.4 g
  braking) until the front axle is 4.0 s (at the current speed) from the drop. That state is the common starting
  state of every method. Episode end, success and termination are unchanged.
- **A4.2 Effective reference speed (D19).** v_eff = min(v0, √(0.7·μ0·g·R)) (v_eff = v0 on the straight).
  - v_eff replaces v0 as the objective's v_ref (DESIGN §4) and as the speed of the certificate's nominal arrival
    state (refines A3: "certified-infeasible" = certified at v_eff, on the centreline).
  - Figures use v_eff; v0 stays the sampled cell value and is reported too.
- **A4.3 Reactive witnesses (D20).** Reactive-feasible = O-R succeeds in ≥ 1 of 5 seeds, **or** W-PD succeeds. W-PD
  is the shared dry driver continuing after handover, with no friction information. The O-R-only label (A3) is
  logged as `label_or_only`. O-R remains the regret reference.
- **A4.4 Oracle settings (D21, D22).** H = 150 steps (3 s), σ = (0.05, 0.3), λ = 10 (fixed), ancillary PD sample,
  5 seeds. O-R: ρ = 0, current true per-axle surfaces. O-C: ρ = 0.9, true friction field. O-C runs only where O-R
  has no success, until its first success. **K = 1024**. K-check per D22: O-R and O-C rerun at K = 16384 on the 43 tuning scenarios that are not reactive-feasible at K = 1024. 2 became reactive-feasible (31 stayed preview-required, 10 undetermined), within the pre-set rule (≤ 2), so K = 1024 is kept. Read: about 5% of non-RF labels may reflect oracle compute rather than physics; the report states this.
  This replaces "≈ 16k samples" in V7; the test-pool compute is reported with the labels.
- **A4.5 Adaptation-required stratum (D22 in the review; pre-declared here).** Reactive-feasible scenarios where
  W-PD fails. H1 is unchanged (pooled reactive-feasible headline). The same H1 test (TOST at ±5 pp, the same
  statistics) is also reported on this stratum, with equal prominence, as a **pre-declared secondary analysis**
  (not in the Holm family).
- **A4.6 S1 probe (D23).** In addition to S1 on every labelled run, S1 is run on the probe set
  (`scripts/setup/s1_probe.py`, own seed 1423445753, 96 scenarios, not part of any split). It must have ≥ 30 runs
  arriving in a certified state and none of them succeeding.
- **A4.7 Second tuning block.** `splits/tuning_v1_b2.jsonl`, 200 scenarios, seed
  int(sha256("friction-cliff-bench tuning split v1 block 2")[:8], 16) = 3838336887, SHA-256
  42fac623f33c41b2b4caa26746951a6deec18e84f5a034ec6abd77b3b28f1862. Tuning and S2 use both tuning blocks, because
  S2 needs ≥ 200 reactive-feasible tuning scenarios (V5). It shares no instance with the test blocks (checked in CI).
- **A4.8 Information (D24).** Every MPPI method starts its rollouts from the true vehicle state (perfect state
  estimation). Non-privileged methods observe the channels of `sim/observe.py`, which now include a noisy lateral
  velocity (0.1 m/s). Observation noise levels: `PARAMETERS.md` §7.
- **A4.9 Learned-method details (D25–D27).**
  - DR randomizes only the per-axle lateral B scale (±15%) for tire shape.
  - Own PPO.
  - Every PPO actor is warm-started by behavior cloning the friction-blind shared dry driver.
  - Budget split: P-S = 14 teacher + 6 student trials (P-SP likewise); P-DR, M-B and M-L get 20 trials each. The
    V2 GPU-hour cap applies pro rata.
  - History windows: W ∈ {5, 12, 25, 50} control steps for every adaptive method.

### A5 (2026-11-11): evaluation plan (D29, D30)
Decided 2026-11-11. Adopted **before any method was evaluated on a test-split scenario**. It fixes what V6, V7
and `EVAL_PROTOCOL.md` leave open. Code: `eval/runner.py`, `eval/analysis.py`, `scripts/evaluate/eval_split.py`.
- **A5.1 Implementation.** Every MPPI-based method (O-R, O-C, M-N, M-B, M-L) runs on the C++/CUDA MPPI in the
  confirmatory runs (D29; S3 passed on 1,000 states). The labels stay as computed by the Warp oracles.
- **A5.2 Gates first.** No test-split evaluation before S2 passes on the final P-T seeds (`scripts/evaluate/s2_gate.py`) and
  the test-pool labels pass S1 and the A4.6 probe.
- **A5.3 Runs.**
  - Every method runs on every labelled scenario of the main set (all classes; tables by class).
  - MPPI methods (O-R, O-C, M-N, M-B): 5 noise seeds per scenario.
  - Learned methods (P-T, P-TP, P-DR, P-S, P-SP): the 5 final training seeds, one episode per scenario each, with
    observation-noise seed = training seed.
  - M-L: estimator training seed s with MPPI noise seed s, s = 0…4.
  - O-C runs on all scenarios here, unlike in labelling.
  - M-N uses M-B's tuned MPPI settings (V2).
- **A5.4 "The better of M-B/M-L"** is the one with the lower pooled departure rate on the reactive-feasible test
  scenarios. Taking the minimum makes the H1 falsifier harder to meet, so the choice is conservative. The same
  method is "the best adaptive MPPI" in H2 at every level.
- **A5.5 H1 statistics.** d = departure(better MPPI) − departure(P-S), paired, hierarchical bootstrap (10,000;
  seeds per method, then scenarios jointly).
  - Falsified iff the 95% percentile CI lies entirely above +0.05.
  - The p-value entering Holm is the bootstrap P(d ≤ 0.05).
  - TOST at ±5 pp is reported alongside.
  - The adaptation-required stratum (A4.5) gets the same numbers as a secondary analysis.
- **A5.6 H2 sets.**
  - Scenarios: 500 reactive-feasible test scenarios drawn without replacement with seed 2027 from the
    reactive-feasible list in split-file order, the same for every axis and level. This bounds compute: 9
    mismatch levels × 5 methods × 5 seeds.
  - Level 0 is the main-set runs on those scenarios.
  - Levels follow V6. The delay levels are +3/+5/+7 control steps, i.e. +60/100/140 ms on top of the nominal 20 ms.
  - The handover drive uses the mismatched truth. O-R plans with the true (mismatched) model; every other method
    keeps its nominal model.
  - Methods: O-R, M-B, M-L, P-S, P-DR.
  - Regret = departure − O-R's departure on the same scenarios. Δ(level) = regret(best adaptive MPPI) −
    regret(P-S).
  - (a) The least-squares slope of Δ over the centred level index 0…3 must be > 0, with its bootstrap 95% CI
    excluding 0. (b) Δ at level 3 must be > 0, with its CI excluding 0.
  - p_a and p_b are two-sided bootstrap p-values, set to 1 if the estimate points the other way. p_axis and p_H2
    are as in V6.
- **A5.7 H3.**
  - Departure on the reactive-feasible main set.
  - Dry progress on the **dry set**: the A5.6 scenarios with the drop removed (μ1 = μ0, dry surface), nominal truth.
  - Falsified iff (i) the 95% CI of departure(P-S) − departure(P-SP) lies above 0, and (ii) the upper 95% CI bound
    of P-SP's relative dry-progress loss vs P-S is < 2%.
  - p_H3 = max(p of (i) as in A5.6, bootstrap P(loss ≥ 2%)).
- **A5.8 Holm family.** {H1: P(d ≤ 0.05), H2: p_H2, H3: p_H3} at α = 0.05.
- **A5.9 Secondary.**
  - Regret vs O-R in max |e_y|, time out of lane, progress and loss of control, paired on reactive-feasible
    scenarios.
  - Dry pace tax = 1 − progress/progress(O-R) on the dry set.
  - Recovery latency (V4) needs full traces and is computed in a separate pass labelled exploratory.
  - All are reported with bootstrap 95% CIs and are outside the Holm family.
- **A5.10 Top-ups and failures.**
  - If the main pool has < 1,000 reactive-feasible scenarios, the V7 top-up blocks are labelled and added in order
    before any evaluation.
  - Diverged or failed training runs count as runs (EVAL_PROTOCOL §3).


### A6 (2026-11-11): H2 scenarios the shared driver cannot hand over (D31)
Written after a pipeline dry run on the **tuning** split. No test scenario has been evaluated by any method.
- **Problem.** A5.6 runs the handover drive (the shared dry driver, A4) on the mismatched truth. At +140 ms
  delay, the driver itself failed to reach the handover state in 1 of 43 tuning-split H2 scenarios
  (U-00032: R400 curve exit, 24.9 m/s). A5 did not say what happens to such a scenario.
- **Rule.** For each mismatch axis, a scenario of the H2 subset is excluded if the shared driver fails to reach
  handover under **any** level of that axis.
  - It is excluded from every level of that axis, level 0 (the main-set runs) included, and for every method. So
    Δ(level) and the slope stay paired on one scenario list per axis.
  - Exclusion is decided from the handover drive alone, before any method runs on the set. It cannot depend on
    any method's outcome.
  - The excluded scenario ids and their number are reported per axis (`h2_excluded_<axis>.json`).
- **Unchanged:** the H2 subset draw (500, seed 2027), the levels, the methods, the tests and the p-values of A5.6.

**A6.2 Test labels frozen (G2).**
- The test pool `splits/test_v1_T0.jsonl` was labelled with the frozen A4 oracle settings (config hash
  7050c39f6a2a, 5 seeds). Result: 3,009 reactive-feasible, 755 preview-required, 29 certified-infeasible and
  207 undetermined, with 363 adaptation-required. S1: 0 violations.
- 3,009 ≥ 1,000 reactive-feasible, so no top-up block is labelled (A5.10).
- The label file `labels/test_v1_T0.jsonl` is frozen by SHA-256 `7671fb0f…13d6` (`tests/test_frozen.py`).
- No method has been run on any test scenario.

### A7 (2026-11-11): adaptation or caution? Exploratory E6/E7, declared before any test evaluation (D32)
Written after tuning-split runs of the tuned teacher and student. No method has been run on any test scenario.
- **Observation (tuning data, engineering).**
  - The learned policies arrive at the drop at about 0.65 × v_eff; the MPPI methods arrive at about 0.9–1.0.
  - Every training and evaluation episode contains a drop about 4 s after handover. A policy trained on that
    distribution can learn to slow down pre-emptively; the MPPI methods plan with the current or estimated
    friction and have no such prior.
  - V1 accepted this risk–pace trade-off for every method, and the dry pace tax reports its cost. But a lower
    departure rate bought by slowing down would look like successful adaptation in H1.
- **Unchanged.** H1–H3, their tests and the Holm family. The following are **exploratory** and outside the Holm
  family.
- **E6: arrival speed.** On reactive-feasible test scenarios, per method:
  - the mean of (front-axle speed at the drop) / v_eff, with a hierarchical-bootstrap 95% CI;
  - the departure rate within arrival-ratio bins [0, 0.8), [0.8, 0.9), [0.9, 1.0), [1.0, ∞), with Wilson 95%
    CIs.
  - Runs that never reach the drop are counted separately.
- **E7: speed-matched bank MPPI (M-B-c).** M-B with its reference speed scaled by c.
  - c = the mean arrival ratio of the five final P-S seeds on the reactive-feasible **tuning** scenarios
    (`scripts/setup/caution_c.py`). It is frozen in `ledgers/caution_c.json` before any test evaluation.
  - M-B-c otherwise keeps the tuned M-B settings and noise seeds. It runs on the main and dry sets.
  - Reported: departure (main) and dry pace tax against P-S, paired, with 95% CIs.
  - Interpretation, fixed now: if M-B-c's departure is not higher than P-S's (CI of the difference includes 0 or
    favours M-B-c), the student's H1 advantage is attributed to caution, not to adaptation.

### A8 (2026-11-11): S2 failed; PPO collapse rollback (D33)
No method has been run on any test scenario.
- **S2 result (tuning data).** The final P-T at the tuned configuration, 5 seeds, departed on 0.195 of the RF
  tuning runs (N = 1,535; O-R + 5 pp is the limit). S2 failed.
  - Per seed: 0.010, 0.182, 0.000, 0.039, 0.746.
  - Seeds 1 and 4 collapsed mid-training: windowed training success went from ~0.5–0.6 to ~0 and never
    recovered. The kept best checkpoint is an early, weak one.
  - V5 reads a gap this large as "training is broken". Tuning scored each configuration on seed 0 only, so it
    could not see the instability.
- **Fix, for every PPO method (P-T, P-TP, P-DR, so also P-S and P-SP via their teachers) and every seed.** When the
  windowed training success stays below 0.5 × its best so far for 3 consecutive log points (30 iterations),
  restore the best checkpoint (actor, critic, std), reset the optimisers and halve the actor learning rate. At
  most 3 times per run.
  - A run that never collapses trains exactly as before. That includes the tuning runs, so the tuned
    configurations stay valid.
- **Unchanged:** the tuned hyperparameters and the training length. Tuning trials are not rerun.
- **All seeds of the new procedure are reported** (never best-of). The pre-A8 final checkpoints are archived in
  `runs/final_v1_prefix_A8/` with their S2 numbers (LOG). c for E7 is recomputed from the new P-S seeds.
- **If S2 still fails after A8:** the policy class is reported as failing S2, and H1–H3 are run and reported with
  that caveat, not withheld.

### A9 (2026-11-14): RL follow-up on test pool v2 (D35, D36)
Written and committed **before** test pool v2 is labelled or any method runs on it. This commit is tagged
`prereg-v2`. Test v1 results (H1–H3, E1–E7) are unchanged and not re-analysed.
- **Why.** On v1 the RL student's result depended on the training seed (S2 failed, A8). A diagnosis on training,
  tuning and dev data only (LOG 2026-11-13/14) found two causes:
  1. Training kept the best checkpoint by *stochastic* training success, but evaluation uses the mean action. A
     kept checkpoint can be noise-robust yet fail deterministically.
  2. Some current-friction-teacher seeds learn poor lane keeping.
  - Fix: select checkpoints by deterministic success on a held-out validation library.
  - With it, the preview-teacher student was reliable on all 10 dev seeds.
- **Test pool v2** = the pre-declared V7 top-up blocks `test_v1_T1/T2/T3` (A2 hashes; never labelled or evaluated).
  - Labelled with the frozen A4 oracles (`scripts/setup/label_split.py`, 5 seeds), RF scenarios only.
  - `main` (nominal truth) and `dry` (drop removed, for the pace tax) sets; batch 64; C++ MPPI backend.
  - Runner: `scripts/evaluate/eval_v2.py`. Analysis: `eval/followup.py` (tested; committed with this amendment).
- **Frozen inputs** (`ledgers/A9_frozen.json`, SHA-256 of every checkpoint and data file; `eval_v2.py` refuses to
  run on any mismatch):
  - **P-SP2** (the follow-up recipe), 10 seeds:
    - teacher P-TP at its tuned A8 configuration plus deterministic validation every 10 iterations and
      checkpoint selection on it (`val_every=10`, `select_on_val`; validation library `runs/rl/lib_val_v1.npz`,
      2,048 scenarios, new seed, nominal parameters);
    - student distilled with the tuned P-SP configuration (`scripts/train/rl_distill.py`).
  - **P-S2**: the same, from the P-T teacher (10 seeds).
  - **P-S**: the A8 recipe, 10 seeds.
    - Seeds 0–4 are the confirmatory checkpoints.
    - Seeds 5–9 were trained during the follow-up by the same code and configuration (training is
      deterministic per seed: seeds 0–4 reproduce bit for bit).
  - **P-SP**: A8, 5 seeds.
  - **O-R and M-B**: as A4/A5, 5 seeds.
  - **M-Bc**: M-B with v_ref × c'. c' is interpolated on the **dev split** from the M-B@c pace-tax curve at
    P-SP2's mean dev pace tax (`scripts/setup/freeze_A9.py`); 5 seeds.
- **H4 (the one confirmatory test).** P-SP2 is not worse than the bank MPPI at matched pace.
  - Test: departure(P-SP2) − departure(M-Bc) on v2 RF, upper 95% hierarchical-bootstrap bound < +2 pp.
  - Margin: 2 pp is under half the H1 equivalence margin (5 pp) and about M-B's own departure at full speed on
    dev.
  - Pace match: |dry pace tax difference| ≤ 3 pp (point estimate); otherwise H4 is "pace not matched" and not
    interpreted.
  - Superiority (upper bound < 0) is reported as secondary.
  - **Prediction: non-inferior; not superior.**
- **H5 (seed reliability, a rule, no p-value).** Gate S2 (V5: departure ≤ O-R + 5 pp, point estimate) applied to
  each seed separately. It holds if all 10 P-SP2 seeds pass. It is reported for every learned method.
  - **Prediction: holds for P-SP2; fails for at least one P-S seed.**
- **Exploratory, pre-declared:**
  - P-S − P-S2 (the selection fix on the current-friction student). Prediction: lower departure, but the interval
    may include 0, because one P-S2 dev seed still has poor lane keeping;
  - P-S − P-SP2 and P-SP − P-SP2;
  - the P-SP2 seeds of the five best teachers by validation success (`P-SP2_top5_by_validation` in the ledger);
  - per-seed tables;
  - learned-policy latency (`scripts/bench/bench_policy.py`) next to the MPPI latency table.
- **Data use.** Dev split (`splits/dev_v1.jsonl`, new seed, 737 RF, labelled with the frozen oracles) and tuning:
  recipe choices and c'. Test v1: no decisions. Test v2: this evaluation only.
- **Parity.** The RL arm's extra engineering (diagnosis runs, about 15 GPU-h) is logged in LOG. The MPPI arm's
  extra degree of freedom is c', set on dev. H4 is a matched-pace comparison, so neither arm gains from caution
  alone.
- **All seeds reported**; no seed dropped.
