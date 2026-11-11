# RELATED_WORK

Status of the novelty claim (updated 2026-10-14, after the P0 full-text check, §"P0 prior-work check" below):
we found **no paper or public repo that trains an RMA-class (privileged teacher → history student) driving policy
and compares it head-to-head with fast adaptive MPC at abrupt friction drops**, and **no vehicle-adaptation
benchmark with per-scenario feasibility labels**. This is "no prior head-to-head found as of 2026-10-14", not a
claim of being first. Caveat: IEEE Xplore could not be queried directly (web search only). The Poznań extended abstract was read in
full from a manually downloaded PDF.
**Never claim "first".**

## Closest work ("how we differ" confirmed in the P0 full-text check, 2026-10-14)
Columns: **RMA?** = does it compare an RMA-class policy (privileged teacher → history student) head-to-head with
fast adaptive MPC at abrupt μ drops? **Feas.?** = does it label scenarios by physical feasibility?

| Work | What it does (from the full text unless noted) | RMA? | Feas.? | How we differ |
|---|---|---|---|---|
| LLA-MPC, arXiv 2505.19512 v1 (CMU, Dolan lab; README says IROS 2025); code DRIVE-LAB-CMU/LLA-MPC | Bank of 10k–20k dynamic-bicycle models (B, C, D per axle + rolling resistance + drag, sampled ±150%), look-back W = 0.2 s, NMPC H = 0.4 s; μ̂ = (D_f + D_r)/2mg, exp-smoothed, picks a speed profile. 1:43 numeric sim (ETHZ, ETHZMobil) + CARLA (gradual decay only). Exp 2/3: sudden 40% μ drop; 10 runs averaged; baselines APACRace (online ELM) + a known-μ oracle MPC | No. Its "learning-based approaches … struggle" claim is tested only against APACRace (gradual online learning), never an offline-trained adaptive policy | No. Its own oracle also leaves the track after the drop, unexplained | We add the RMA-class policy arm, a learned-estimator MPPI arm, matched information/tuning, full-size car with surface-dependent tire *shape* (its bank varies shape too, but truth and bank share one family), feasibility labels, and many paired scenarios with CIs |
| AnyCar, arXiv 2409.15783 v1 (CMU, Shi/Dolan) | Transformer in-context dynamics model (100M sim steps from 4 simulators + 10 min real), MPPI (CoVO-MPC, 600 samples). 1/10 and 1/16 RC; perturbations: towing, payload, plastic wheels, SLAM noise. Baselines: pure pursuit, sys-ID'd DBM, AnyCar w/o fine-tuning | No. RL explicitly "outside the scope of this work"; RMA appears only in related work | No | Model-side learning only; no mid-run abrupt μ step; no policy arm. We test policy-side vs model-side adaptation at abrupt steps |
| Kalaria et al., ICRA 2025, arXiv 2410.06565 v2 (CMU) | MAML-initialised LSTM dynamics + online gradient adaptation + ensemble-uncertainty MPPI. Numeric sim, Unity (incl. friction *decay* 1 → 0.7, mud, go-kart), RC hardware. Baselines: no adaptation, APACRace, ablations | No. Cites RMA [5] only to argue such policies fail outside the training distribution; no RL baseline | No | Friction change is gradual, not abrupt; no policy arm. Their out-of-distribution argument against RMA is exactly what our H2 tests |
| OptCar, arXiv 2607.13319 v2 (UT Austin AMRL + ARL; CoRL 2026) | FiLM-conditioned transformer FKD model (history context token, M = 250 steps) inside MPPI; generalist pretraining + targeted sim+real fine-tuning. 1/5-scale off-road truck, figure-8 at 2/4/6 m/s on road, wet grass, vegetation + dirt, cart pulling. Baselines: DBM-MPC, IKD, AnyCar, specialist | No. End-to-end policies dismissed in related work; cites RMA only | No (it picks the figure-8 size "so that 6 m/s tracking remains feasible", informally) | Off-road, mixed terrain rather than controlled abrupt μ steps; model-side only |
| Poznań memory-policy extended abstract: Czechmanowski, Węgrzynowski, Kicki, Walas, "Beyond Reactive Adaptation: Long-Horizon Memory for Autonomous Racing via State Space Models", XPlore workshop @ ICRA 2026 (OpenReview aSdoatNZc1) | **Full text read 2026-10-14** (3 pp, manually downloaded PDF). Mamba PPO policy, obs = vehicle state + Fourier features of track progress, so it memorizes slippery zones across laps. Sim F1TENTH; friction is a smooth spline (2–5 control points, μ 0.5–1.0) along the track, not an abrupt step. Baselines: MLP, RNN, and an oracle MLP trained on the exact test friction. Metric: lap time over consecutive laps | No. No MPC baseline at all; RMA is discussed as the short-horizon alternative but not run | No | Still **most likely to scoop** (same group: RL-vs-MPC at IROS 2025, LP-MPPI, HyperMPC; future work = real F1TENTH). Differs on every axis we care about: multi-lap memory vs single-encounter abrupt drops, smooth vs step friction, no model-based baselines, no feasibility labels |
| Czechmanowski et al., IROS 2025, arXiv 2504.02420 | RL beats MPC on real F1TENTH | No | No | Single-condition racing; not abrupt μ steps; MPC tuning effort unreported |
| LLA-MPPI, arXiv 2610.10465 v1 (7 Oct 2026; CMU, AL-Sunni/Dolan + Manchester/Liu) | LLA bank selection over MJWarp worlds + MPPI for **legged** whole-body control (Go2): payload, leg lock, box mass, slope+friction (friction *decreases* during the episode). Section V-B compares with domain-randomized PPO (no history, no teacher-student) on the payload task | No. Legged; RL baseline is DR-PPO, not RMA; no vehicles | No | Different domain. **Scoop-watch signal:** the LLA group now runs MPPI-vs-RL comparisons and uses Warp; a vehicle version is plausible. Re-check monthly |
| UNLV NSF REU repo, hawaii-nc/NSF-REU-UNLV-Smart-Cities-2026 (Summer 2026, no license, unpublished) | RMA (PPO base policy on LiDAR + motion history, adaptation module infers tire-grip latent) on F1TENTH/Traxxas; sim + real across friction conditions (per README; self-reported, unverified) | No. Compared with a friction oracle and a non-adapting policy; no MPC baseline; friction varies per condition, not as mid-run steps (per README) | No | Shows RMA-for-cars itself is not novel (an undergrad did it). Our contribution is the fair head-to-head + feasibility labelling, not RMA on a car |
| Friction-adaptive stochastic NMPC, arXiv 2305.03798 (MERL) | Online μ learning in SNMPC | No | No | Model-based only |
| Online slip detection + μ estimation, arXiv 2509.15423 | Training-free detection, 1:10 racing | No | No | Estimator, not a controller comparison |
| Djeumou et al., CoRL 2024 (PMLR 270) "One Model to Drift Them All" | Conditional diffusion over physics-model params, in MPC, real cars | No | No | Model-side adaptation at drifting; real hardware |
| Djeumou et al., arXiv 2410.20990 | Reference-free RL drifting from a neural-SDE sim, zero-shot real | No | No | RL at limits, not a fair adaptation benchmark |
| Safe RL + predictive safety filter, arXiv 2506.22894 | Drifting, CarSim | No | No | Safety filter not μ-adaptive |
| SceneFactory, arXiv 2605.08528 | Isaac Lab PhysX vehicles; kinematic→physics transfer gap | No | No | Planning-level; motivates dynamics realism |

## Methodology anchors
- Song et al., Science Robotics 2023, arXiv 2310.10943: RL vs optimal control. RL wins by optimizing a better objective.
- Cart-pole NMPC-vs-RL break-even under model error, arXiv 1910.12047 (H2 analogue).
- Legged RL-vs-MPC benchmark, arXiv 2501.16590: the winner depends on the perturbation set.
- Feasibility in scenario generation: ScenePilot σ score (arXiv 2605.21168), CARS (arXiv 2605.13751).
- RMA, arXiv 2107.04034. MPPI: Williams et al., ICRA 2017 (AutoRally).
- Residual MPC with GPU-batched MPC in the RL loop, arXiv 2510.12717 (P1 link).
- Imitable experts in driving IL: LEAD / TransFuser v6, arXiv 2512.20563 (supports teacher-information design, H3).

## P0 prior-work check (2026-10-14): search record
- **Full texts read** (arXiv PDFs → `pdftotext`, read in full except references): 2505.19512 v1, 2409.15783 v1,
  2410.06565 v2, 2607.13319 v2, and 2610.10465 v1 (found during the search; read for its RL comparison).
- **Poznań abstract:** located via the XPlore @ ICRA 2026 accepted-papers page. Scripted access to OpenReview was
  blocked (HTTP 403); I downloaded the PDF and it was read in full the same day: no MPC baseline.
- **Web searches** (standard/extended): RMA teacher-student vs adaptive MPC friction driving; IEEE IV / T-IV / ITSC
  RL vs MPC low-friction / ice / sudden friction change; privileged teacher-student vehicle friction; "rapid motor
  adaptation" + racing/vehicle/F1TENTH; 2026 benchmark learned policy vs adaptive MPPI abrupt friction. IEEE Xplore
  itself was not queried (no access from this environment); IEEE coverage relies on web-search indexing.
- **arXiv API queries** (sorted by date, all years): friction ∧ "reinforcement learning" ∧ (MPC ∨ MPPI ∨
  "predictive control"); "motor adaptation" ∧ (vehicle ∨ car ∨ racing ∨ driving); (teacher-student ∨ privileged) ∧
  friction; friction ∧ (sudden ∨ abrupt) ∧ (policy ∨ RL); friction ∧ MPPI ∧ adapt*. No vehicle hit with an RMA-class
  policy vs adaptive MPC. Closest off-target hits: 2607.20973 (RL-guided MPC for racing overtakes), 2509.18466
  (RL-augmented adaptive MPC, bipedal).
- Other hits checked and ruled out: PPO lateral stability on low-adhesion / split-μ roads vs MPC and SMC (PMC12654942;
  search snippet only, full text not read; nothing in the snippet suggests an
  RMA-class policy or adaptive MPC, to confirm); Lin et al. T-IV 2021 DRL vs MPC for ACC (car-following, no μ steps);
  adaptive-MPC review 2608.17902 (abstract: no RL comparison); hawaii-nc/f1tenth_benchmarks (copy of the Evans et
  al. F1TENTH benchmark: fixed-model MPCC + SAC/TD3).
- **Verdict:** no head-to-head found; no scoop. Do not switch to replicate-and-audit (PLAN §6).

## Scoop watch log (weekly arXiv alert; newest first)
- **2026-11-11:** The P0 arXiv queries were rerun (sorted by date) plus (vehicle ∨ racing ∨ car) ∧ friction ∧ adapt*.
  None of the new papers is a head-to-head of an RMA-class policy against adaptive MPC at abrupt drops, and none
  labels feasibility.
  - LLA-MPC on embedded hardware (2610.03616, Liao/AL-Sunni/Dolan): F1TENTH, model-side only. Cited.
  - Context-conditioned HJ reachability (2610.08115, Bansal lab): a learned friction-conditioned safety value
    function. Related to our feasibility layer and to the safety-filter extension. Cited.
  - Indirect-adaptive robust control under unknown friction (2610.22675): model-based only.
  - Verdict: no scoop.
- **2026-10-14:** LLA-MPPI (2610.10465, 7 Oct 2026): legged, DR-PPO baseline; not a scoop, but the LLA group is the
  most likely to publish a vehicle MPPI-vs-RL comparison. Poznań group remains the other top risk. UNLV REU RMA
  racing repo (Summer 2026): RMA on cars without MPC baselines.
