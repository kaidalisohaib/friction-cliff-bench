# BASELINES: controllers, information, fairness, implementation

## 1. Controller set
| ID | Controller | Friction information at test time | History window | Role |
|---|---|---|---|---|
| O-R | Reactive-oracle MPPI | true current per-axle μ + surface (no preview) | — | Upper bound for reactive methods; defines *adaptation regret* |
| O-C | Clairvoyant-oracle MPPI | full true μ/surface map ahead | — | Feasibility labelling only, never in the head-to-head |
| M-N | Nominal MPPI | assumes dry | — | Context row (the classic strawman), not a real baseline |
| M-B | **Bank-adaptive MPPI** (LLA-MPC style) | model selected from a bank by look-back prediction error | W | Strongest classical adaptive baseline |
| M-L | **Learned-estimator MPPI** | μ̂ (+ surface class) from a GRU on history | W | Same network class as the RMA encoder, feeding a planner instead of a policy |
| P-T | PPO teacher | privileged true current per-axle μ (+ DR params) via latent z | — | Upper bound for the policy class |
| **P-S** | **RMA student** | latent ẑ regressed from history | W | **Method under test** |
| P-DR | PPO, domain-randomized, no history | none | — | Tests whether "robust" = blanket caution |
| P-SP | Student of a 30 m *preview* teacher | none (teacher had preview) | W | H3 ablation |
| (stretch) M-D | Learned-dynamics MPPI (AnyCar-style in-context model) | implicit | W | Fourth adaptation style |

## 2. Fairness checklist (all mandatory)
1. **Feasibility oracle per scenario** (`FEASIBILITY.md`). Head-to-head claims are made per class.
2. **Sweeps, not single points:** results are shown as curves over the sweep axes.
3. **Matched objective:** the same cost (`DESIGN.md` §4) is the MPPI running cost and the PPO reward.
4. **Matched information:** every adaptive method (M-B, M-L, P-S, P-SP) gets the same observation channels and the
   same history window W. Privileged true μ goes only to O-R, O-C and P-T.
   Observation channels for non-privileged methods: IMU (ax, ay, r with noise/bias), wheel speeds, steering
   angle, last actions, path-relative state (e_y, e_ψ, s, κ preview of the *geometry*, not friction), speed.
5. **Equal tuning budget:** the same number of Optuna trials on the tuning split and the same GPU-hour cap per
   method. For PPO, a "trial" = one training run with sampled hyperparameters (lr, entropy coef, history
   length within the allowed W, network width). Ledger in `LOG.md`: trials, GPU-hours, human-hours.
6. **Reported compute:** per-step latency on the same GPU, MPPI samples × horizon, policy FLOPs.
7. **Multiple seeds with CIs** (`EVAL_PROTOCOL.md`).
8. **Same actuators and delays** for everyone (lag, rate limits, 1-step delay).

## 3. MPPI (reference implementation in Warp, P2)
- Rollout model: the truth simulator's dynamics kernel (in-family) at control dt (20 ms) with physics substeps.
  For H2 mismatch tests, the rollout model stays in the *nominal family* while the truth changes.
- Initial settings (tuned within budget): horizon 1.5–2.5 s, K = 1k–4k samples, temperature λ, noise Σ on (u_δ, u_a),
  warm start from the shifted previous plan, cost per `DESIGN.md` §4 + soft constraints for limits.
- **M-B (bank):** a bank of candidate (surface params, μ, extra actuator delay ∈ {0, 20, 40} ms) models spanning the DR range (D14) (LLA-MPC uses ~20k dynamic
  bicycle variants; start with ~50–200 and justify the size). Each step, score candidates by one-step prediction
  error over the look-back window W and pick the argmin (or a softmin mixture, ⚠ decide). MPPI uses the selected model.
- **M-L (estimator):** a GRU trained offline (supervised, sim data under DR) to predict per-axle μ (+ surface class
  + extra actuator delay, D14) from the W-window history. MPPI uses μ̂ and the estimated delay.
- Oracles: same MPPI code with true information and the generous settings in `FEASIBILITY.md` §3.

## 4. RL arm (P3)
- PPO (rsl-rl-lib, pinned version) on the Warp env via zero-copy torch tensors. Actions: (u_δ, u_a) at 50 Hz.
- **Teacher P-T:** obs = non-privileged obs + privileged current per-axle μ, surface one-hot, DR params (mass, Iz,
  tire shape scale, delays) → encoder → latent z. Asymmetric critic may see more (true forces, vy).
- **Student P-S (RMA phase 2):** history encoder (GRU or 1D-conv over W) regresses z, then is frozen into the
  teacher policy head. Variant E3: DAgger action distillation instead of latent regression.
- **P-DR:** same obs as the student minus history, same DR.
- **P-SP:** the teacher additionally sees a μ/surface preview at 0–30 m ahead; a student is distilled from it
  exactly like P-S.
- Curriculum: dry only → mild patches → full scenario distribution. Training scenarios are sampled from the
  *tuning-split sampler* only, never the test split.
- DR (train): mass ±15%, Iz ±15%, CoG h ±10%, tire shape (B, C, E) ±15% within the surface family, μ within the
  surface ranges, extra actuator delay 0–40 ms on both channels (the M-B bank and M-L estimator cover the same
  range, D14), sensor noise/bias. **H2 test levels lie outside the DR support** (axes 1–2 absent; delay only beyond 40 ms).

## 5. C++17/CUDA MPPI library (P4)
- `mppi_cpp/include/vehicle.hpp`: `__host__ __device__` dynamics step, identical equations/params to the Warp sim
  (shared parameter file, e.g. YAML → generated header or runtime struct).
- CUDA kernel: batched rollouts over (env, sample), cost accumulation, weighted update, warm start.
- CMake build, GoogleTest (dynamics vs the NumPy reference; MPPI update math; determinism with fixed noise).
- nanobind (or pybind11) bindings that accept PyTorch CUDA tensors without copies.
- **S3 equivalence:** with fixed noise tensors, C++ and Warp MPPI produce the same control sequence within tolerance on
  1,000 random states.
- Latency table: samples ∈ {256 … 16k} × horizon ∈ {50 … 150} steps, on the RTX 5090 Laptop GPU.
- Out of scope: ROS 2 node, TensorRT, C++ RL environment.
