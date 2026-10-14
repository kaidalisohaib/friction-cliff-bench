# RELATED_WORK

Status of the novelty claim (2026-10-13): about ten targeted searches found **no paper training an RMA-style
privileged-teacher → history-student driving policy and benchmarking it head-to-head against fast adaptive MPC
at abrupt friction drops**, and **no vehicle-adaptation benchmark with per-scenario feasibility labels**.
Not yet checked: the full texts of AnyCar / Kalaria / OptCar, and IEEE-only venues (IV, ITSC, T-IV). That is P0's job.
**Never claim "first".**

## Closest work (fill "how we differ" after the P0 full-text check)
| Work | What it does | How we differ (⚠ confirm in P0) |
|---|---|---|
| LLA-MPC, arXiv 2505.19512 (Dolan lab, CMU); code github.com/DRIVE-LAB-CMU/LLA-MPC | Bank of ~20k bicycle models, look-back selection; sudden 40% μ drop: 0.65 s track violation vs 0.72 s known-μ oracle vs 2.12 s APACRace | Its claim that learning struggles at abrupt changes was tested only against gradual online learning, not an RMA-class policy; no feasibility labels |
| AnyCar, arXiv 2409.15783 | In-context transformer dynamics + MPPI, cross-vehicle | Model-side learning; no RMA policy baseline (per abstract) |
| Kalaria et al., ICRA 2025, arXiv 2410.06565 | Meta-learned dynamics + uncertainty-aware MPPI | Same |
| OptCar, arXiv 2607.13319 (UT Austin AMRL) | Learned dynamics, off-road MPPI | Same; off-road |
| Poznań memory-policy extended abstract (Czechmanowski, Węgrzynowski, Kicki, Walas) | Mamba policy in sim F1TENTH under friction changes; vs MLP/RNN/oracle | No MPC baselines; names RMA as the thing to beat. **Most likely to scoop us** |
| Czechmanowski et al., IROS 2025, arXiv 2504.02420 | RL beats MPC on real F1TENTH | Single-condition racing; not abrupt μ steps; MPC tuning effort unreported |
| Friction-adaptive stochastic NMPC, arXiv 2305.03798 (MERL) | Online μ learning in SNMPC | Model-based only |
| Online slip detection + μ estimation, arXiv 2509.15423 | Training-free detection, 1:10 racing | Estimator, not a controller comparison |
| Djeumou et al., CoRL 2024 (PMLR 270) "One Model to Drift Them All" | Conditional diffusion over physics-model params, in MPC, real cars | Model-side adaptation at drifting; real hardware |
| Djeumou et al., arXiv 2410.20990 | Reference-free RL drifting from a neural-SDE sim, zero-shot real | RL at limits, not a fair adaptation benchmark |
| Safe RL + predictive safety filter, arXiv 2506.22894 | Drifting, CarSim | Safety filter not μ-adaptive |
| SceneFactory, arXiv 2605.08528 | Isaac Lab PhysX vehicles; kinematic→physics transfer 47.3% vs 99.5% reverse | Planning-level; motivates dynamics realism |

## Methodology anchors
- Song et al., Science Robotics 2023, arXiv 2310.10943: RL vs optimal control. RL wins by optimizing a better objective.
- Cart-pole NMPC-vs-RL break-even under model error, arXiv 1910.12047 (H2 analogue).
- Legged RL-vs-MPC benchmark, arXiv 2501.16590: the winner depends on the perturbation set.
- Feasibility in scenario generation: ScenePilot σ score (arXiv 2605.21168), CARS (arXiv 2605.13751).
- RMA, arXiv 2107.04034. MPPI: Williams et al., ICRA 2017 (AutoRally).
- Residual MPC with GPU-batched MPC in the RL loop, arXiv 2510.12717 (P1 link).
- Imitable experts in driving IL: LEAD / TransFuser v6, arXiv 2512.20563 (supports teacher-information design, H3).

## Scoop watch log (weekly arXiv alert; newest first)
_(none yet)_
