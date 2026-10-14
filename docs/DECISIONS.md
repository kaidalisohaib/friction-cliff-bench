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
