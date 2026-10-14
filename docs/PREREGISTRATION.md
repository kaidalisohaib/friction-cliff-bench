# PREREGISTRATION: **DRAFT** (values approved 2026-10-14; frozen when tagged `prereg-v1`)

Freeze procedure (end of P0): resolve every ⚠, fill the test-split spec, commit, `git tag prereg-v1`, push the tag.
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

## Frozen values (all approved 2026-10-14; frozen at the `prereg-v1` tag)
Proposed 2026-10-14, settled the same day. Nothing here is a result.

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
_(none)_
