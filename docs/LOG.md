# LOG (dated notebook; newest first)

Format per entry: date, phase, hours, what was done, and the verdict for any run (prediction → result).

## Hours
| Week of | Phase | Hours | Running total |
|---|---|---|---|
| 2026-10-12 | planning | — | — |
| 2026-10-12 | P0 | ≈ 4.5 | ≈ 4.5 |
| 2026-10-12 | P1 | ≈ 6 | ≈ 10.5 |
| 2026-11-02 | P1/G1 | ≈ 1.5 | ≈ 12 |
| 2026-11-02 | P2 | ≈ 6 | ≈ 18 |

## Tuning ledger (per method: trials, GPU-h, human-h)
- Oracles (not a tuned method; settings pilot on the tuning split, FEASIBILITY §3): 24 grid configs on 40 scenarios +
  K-check (2 × 200 scenarios × 5 seeds) + 2 hard-case variant scans; ≈ 1.0 GPU-h of oracle runs (shared GPU). Logs in `runs/`.
- Tunable methods (2026-11-08, paused; GPU-h = wall time on a GPU shared between jobs, which over-counts):
  - M-B: 20/20 trials, 2.81 GPU-h, done;
  - M-L: 14/20, 3.06 GPU-h;
  - P-T: 1/14, 1.26 GPU-h;
  - P-DR: 0/20.
  Line by line: `ledgers/tuning_ledger.jsonl`.
  PPO pilots before tuning (not counted in any method's budget, which is a design pilot like the oracle pilot):
  ≈ 1.5 GPU-h.

## Entries

### 2026-11-08 late: run paused (GPU needed for another project)
- Every job was stopped cleanly at about 23:00.
- **Test-pool labelling had not finished and left no output**, so it must be rerun. No test label exists, and no
  test scenario has been evaluated by any method.
- Tuning is resumable: `eval/tuning.py` now replays finished trials from the ledger (tested), and the ledger stores
  each trial's checkpoint tag.
- M-B tuning is complete, M-L is at 14/20, and P-T has 1 trial. To resume, rerun the same tuning commands.

### 2026-11-08 evening: long run, G2 → P3; engineering checks, not results
- **Decisions D19–D27, prereg A4** (see DECISIONS).
  - Every check below used tuning/training data only.
  - The test pool was labelled only after A4 was committed (997a231), with the frozen settings.
- **v_eff:** tuning block 1 relabelled. Prediction: the 3 previously certified scenarios lose certification and
  the dry-road O-R failure goes away → **confirmed**.
  - RF 157 / PR 32 / UD 11 / CI 0.
  - S1: 0 violations.
  - The W-PD witness no longer changes any label.
  - Adaptation-required: 17.
- **S1 probe:** prediction "≥ 30 certified arrivals, 0 survivors" → **confirmed**.
  - 480 runs, 278 certified arrivals, 0 survivors.
- **Tuning block 2** (200, new seed, hashed): RF 150 / PR 39 / UD 10 / CI 1, S1 0.
  - With both blocks there are 307 RF tuning scenarios, enough for S2's ≥ 200.
- **K = 16k check** on the 43 non-RF block-1 scenarios: 2 → RF, within the pre-set rule, so K = 1024 is frozen.
- **S2 reference** on the 307 RF tuning scenarios (5 O-R seeds, 1,535 runs):
  - O-R departure 0.037, so the S2 threshold is 0.087;
  - W-PD departure 0.143.
- **PPO pilots** (training library):
  - from scratch: 0 successes in 21M steps;
  - BC warm start, then PPO: collapses as soon as the actor updates. With the actor frozen it stays at ~0.5
    train success, and a toy test shows the PPO update is correct;
  - large batch with low lr and small KL: peaks at 0.69, then collapses;
  - + separate actor/critic optimizers, return normalization, linear LR decay, best training checkpoint
    (pilot F, 60M steps): ends at 0.68 train success, and 0.778 success / 0.221 departure on the 307 RF tuning
    scenarios. Far from the S2 threshold.
  - **Pilot G** (same settings, 200M steps; best training checkpoint at it 360, train-success 0.90 under
    exploration noise and DR): **0.990 success / 0.010 departure** on the 307 RF tuning scenarios, which is below
    O-R's 0.037. Longer training was the missing piece, so PPO trials are fixed at 200M steps (D26 updated here).
- **M-L estimator** (W = 25, first trial): μ MAE ≈ 0.07, surface accuracy ≈ 0.8. Extra-delay accuracy stays at
  chance (≈ 0.36 for 3 classes): 0–40 ms of extra delay is not identifiable from these observations by this
  estimator.

### 2026-11-08 (later): P2 self-review (tuning split only; engineering checks, not results)
- **Correction to the entry below:** "S1: 0 violations" is true but vacuous. Re-certifying every tuning run at its
  logged arrival state: no O-R, O-C or W-PD run arrived in a certified state, so S1 was never exercised.
- **Speed-cap artefact (D19 × A3):** the prefix caps speed at √(0.7·μ0·g·R). On 21 of 200 tuning scenarios the prefix
  speed at the drop is below v0. All 3 scenarios certified at v0 are among them, and none is certified at the
  prefix speed. So the single certified-infeasible label and both certified-reactive-infeasible flags rest on an
  unreachable state. Also, with v_ref = v0 unreachable, O-R left the road on dry in 4 of 5 seeds on U-00084
  (25.6 m/s, R50 exit).
- **RF composition:** W-PD (no friction information) succeeds on 139 of 152 tuning RF scenarios. The pooled-RF H1
  is dominated by scenarios where adaptation is not needed.
- Proposed responses (to decide): v_eff for v_ref, the certificate and the strata; an S1 probe set; a K = 16k
  check on the non-RF tuning scenarios; a pre-declared adaptation-required stratum. Nothing is labelled.
- **2026-11-08 night (P2, GPU shared with another project's eval):**
  - **Built and tested** (134 CPU tests + GPU slow tests):
    - batched Warp MPPI (rollout = truth sim to 1e-5, AR(1) noise, n_iter, ESS-targeted λ, PD ancillary sample with
      preview for O-C);
    - O-R / O-C / M-N;
    - episode runner with prefix handover (D19, proposed) and exact footprint metrics;
    - labelling pipeline with A3 labels + D20 variant + S1;
    - stats (Wilson, paired hierarchical bootstrap, TOST, Holm), observation model, Optuna harness with ledger.
  - **Throughput:** sim 3.2e8 control steps/s; MPPI rollouts ≈ 2e8 (fixed) / 1.3e8 (field) control steps/s.
  - **Prediction:** "MPPI oracle beats a plain PD follower" → **refuted** at first (cold start: noise around a
    straight plan never finds the curve; too much steering noise; weight collapse). Fixed by the PD ancillary
    sample. Some PD-only survivals remain, because O-R is certainty-equivalent (D20).
  - **Pilot** (tuning split only): ρ = 0 best for O-R, ρ = 0.9 for O-C.
  - **K-check:** K = 1024 vs 4096 → 2/200 labels differ (one each way).
  - **S1:** 0 violations in every run.
  - **Proposed:** D19, D20, D21 / A4 draft. **Test pool NOT labelled** (waits for review).
- **2026-11-08 (G1 passed):** Approved D15 and D16. Parameters sourced with plausible ranges and confidence
  levels (`PARAMETERS.md`; e.g. normalized yaw inertia 0.93 within the published 0.85–1.05 band, CdA from the
  0.85·W·H frontal-area rule on Camry dimensions, brake share = ideal at ≈ 0.56 g). Prereg amendments A1
  (parameter fingerprint), A2 (SHA-256 of the test pool, the three top-up blocks and the tuning split, committed
  under `splits/`), A3 (arrival-state S1, label precedence). `tests/test_frozen.py` enforces both freezes in CI (D17).
  Prediction: regenerating the sets from the frozen seeds reproduces the files, and tuning/test share no instance
  → **confirmed** (114 tests pass). Not done: GPU throughput (another project's eval was running, so no GPU job was
  started) → moved to P2 start with the oracle compute budget (D18).
- **2026-10-14 (P1, scenarios + certificate):** Tracks (5 shared, clothoid–arc–clothoid 90° curves), Frenet
  projection, per-axle per-substep friction field, prereg V7 sampler (deterministic, canonical JSONL; SHA not yet
  recorded). Analytic certificate v1 (arc drops, arrival-state based) with the soundness argument written in
  FEASIBILITY §2.1 before any labelling. Prediction: the classic cell (15 m/s, R 50, ice, continuing arc) is
  certified → **confirmed**. Checks: never certifies when circling is possible; Monte Carlo point-mass never beats
  the bound; brute-force point-mass never survives a certified case; full sim leaves the lane in certified cases.
  Measured conservatism: brute-force point mass fails 30–40% below the certified speed (that band = undetermined).
  On the real 90° tracks only apex drops get certified (exit: arc too short; entry: not covered yet). Fixed a
  realism bug: low-speed blend keyed on |vx| (a sideways slide counted as "slow"); now |v|. D16 (arrival-state
  semantics for labels and S1) proposed, OPEN.
- **2026-10-14 (P1, simulator core):** Built and tested `sim/` (tire, NumPy reference, Warp sim, steady-turn
  solver). Gates green: V1, V2 (reformulated), V3, V4/S4, V5, V6 (1e8 GPU env-steps, no NaN), V7, V8 (Warp vs
  reference ~1e-4 m over 2 s). Findings: (1) equal tire coefficients make the car exactly neutral-steer → per-axle
  stiffness scales; (2) the DESIGN MF table is longitudinal-only and put the ice lateral peak at ~21° → D15
  re-sourcing (Burckhardt + CRREL + Pacejka); (3) cornering drag and rolling resistance consume rear (drive)
  grip, so the steady RWD limit is below √(μgR), most on low-μ surfaces; (4) my first V5 hand calc assumed the
  iced axle jumps to its peak force; it stays on the linear part of the ice curve (sim matched the corrected
  calc). V2 now = (a) no steady turn above √(μgR) [soundness], (b) dry/wet within 3%, (c) snow/ice reported,
  (d) closed-loop dry circle.
- **2026-10-14 (G0 passed):** Tagged and pushed `prereg-v1` (at `d640081`). Env verified, no prior head-to-head
  found, prereg frozen. From now on the prereg changes only through dated amendments. P0 complete; next is P1.
- **2026-10-14 (P0, prereg values approved):** Approved the remaining proposals (tuning budget, marginal and
  recovery bands, S2 margin, H2 levels + test rule, test-split spec + seeds, RWD, M-B argmin) after a detailed
  walkthrough. Values moved into a "Frozen values" section (V1–V8) and propagated to DESIGN, FEASIBILITY,
  EVAL_PROTOCOL, BASELINES. Also fixed: WSPI μ calibration (P5) can no longer change the frozen test split; it is an
  appendix check. Remaining before the tag: a final review, then `git tag prereg-v1`.
- **2026-10-14 (P0, decisions):** Approved the objective weights (prereg P1) and chose option B for the
  delay conflict (D14: delay stays in RL DR, the MPPI bank/estimator cover the same 0–40 ms, H2 delay levels
  +{60, 100, 140} ms). Remaining prereg proposals (P2–P8) still pending approval.
- **2026-10-14 (P0, follow-up):** Downloaded the Poznań extended-abstract PDF; read in full. Confirmed: no MPC
  baseline, no RMA baseline run, smooth (not abrupt) friction, multi-lap memory. Scoop verdict unchanged. I
  approved the G1 parameter-freeze timing (D13).
- **2026-10-14 (P0, prereg drafting):** Proposed values for every remaining ⚠ written into
  `PREREGISTRATION.md` § "PROPOSED values" (objective weights, tuning budget, marginal and recovery bands, S2 margin,
  H2 levels + test, test-split sampler spec + verifiable seeds, drivetrain, M-B argmin). Found two inconsistencies:
  RL DR randomizes steering delay although delay is an H2 axis, and DESIGN says parameters are frozen "before
  prereg-v1" but justified in P1. Fixes proposed. **Awaiting review; not tagged.**
  Hours: ≈ 4.5 h equivalent (prior work ≈ 2.5, repos ≈ 1, env ≈ 0.5,
  prereg ≈ 0.5).
- **2026-10-14 (P0, timeboxed repo checks, ~1 h of the 4 h box):** Both cloned and run only under `~/p0_scratch/`,
  each in its own venv. **LLA-MPC** (`dfe09e9`, MIT): runs with Py 3.10 + modern deps after disabling LaTeX/video;
  the full sudden-drop closed-loop run completes (only a final plot call fails on new matplotlib). μ̂ follows
  the step qualitatively. `fixros.py` runs `sudo dpkg -r`, never run it. **INI f1tenth_development_gym** (`6a8a3d4`,
  MIT top level; LGPL-3.0 vendored utility; unlicensed submodules): imports and runs on CPU JAX, but its own
  JAX-vs-ODE parity tests fail 3/3 at HEAD, and the model is lateral-only. Verdicts in `DATA_AND_LICENSES.md`;
  consequence logged as D12 (V8 uses our own NumPy reference).
- **2026-10-14 (P0, prior-work check):** Full texts read: LLA-MPC (2505.19512), AnyCar (2409.15783), Kalaria et al.
  (2410.06565), OptCar (2607.13319, now CoRL 2026), plus LLA-MPPI (2610.10465, found in the search). Poznań
  extended abstract located (XPlore @ ICRA 2026, OpenReview aSdoatNZc1) but only the abstract was readable (PDF
  blocked to scripted access). Web + arXiv-API searches recorded in `RELATED_WORK.md`. Prediction: no RMA-class vs
  adaptive-MPC head-to-head exists. **Verdict: confirmed, no head-to-head and no feasibility-labelled benchmark
  found as of 2026-10-14.** No scoop; stay on the main plan. Watch: LLA group (now does MPPI-vs-DR-PPO on legged
  robots) and the Poznań group. Open follow-ups: read the Poznań PDF in a browser; IEEE Xplore was
  not queried directly.
- **2026-10-14 (P0, env check):** `uv sync` → Python 3.12.13, torch 2.14.1+cu130 (arch list includes sm_120;
  `cuda.is_available()` True; 4k×4k matmul OK), warp-lang 1.18.0 (Apache-2.0; bundles CUDA toolkit 13.4, driver
  13.0). Warp saxpy kernel runs on `cpu` and `cuda:0` (CPU exact; GPU max abs err ~1e-6 vs NumPy, consistent with
  FMA contraction; matters for V7 tolerance only at the 1e-6 level). `wp.from_torch`/`wp.to_torch` share memory.
  **Trap confirmed on 1.18.0:** `wp.struct` field defaults (`m: float = 1600.0`) are silently ignored: a
  default-constructed struct reads 0 on both CPU and GPU. Always set every field explicitly. CUDA/CMake: nvcc
  13.3.73 (at `/usr/local/cuda/bin`, **not on PATH**; pass `-DCMAKE_CUDA_COMPILER` or add to PATH), CMake 3.28.3,
  GCC 14.2.0. A hello-world saxpy built with `CMAKE_CUDA_ARCHITECTURES=120` gives native sm_120 SASS and runs
  correctly (runtime 13.3 on driver 13.0). Verification scripts live outside the repo (`~/p0_scratch/`).
  Verdict: env OK.
- **2026-10-14:** Plan solidified. Docs: PLAN, DESIGN, FEASIBILITY, BASELINES, EVAL_PROTOCOL, PREREGISTRATION
  (draft), RELATED_WORK, DATA_AND_LICENSES. D1–D8, D10, D11 decided; D9 open. Next: P0 checklist.
- **2026-10-13:** Project created (docs only) after a literature and design survey. Verdict on the original blueprint
  P2 idea: modify (fair baselines, feasibility labelling, current-μ teacher, C++/CUDA MPPI; terramechanics and
  ROS 2 cut).
