# Decision log

Dated entries. Never edit an old entry silently; supersede it with a new one.
Status **PROPOSED (2026-10-13)** means pending. Change it to **DECIDED (date)** once approved.

## D1: Modify the original idea instead of replacing it. PROPOSED
The original "extreme-dynamics slip-aware planner" (robotics-blueprint P2) keeps its question (learned vs
model-based adaptation at sudden friction loss) and loses its headline ("RL beats MPC"). Ranked against a
pure C++ MPPI project, PufferDrive self-play, and a kinematic→physics fidelity ladder, the modified version
scored highest (report §"The modified original ranks first"). Its descope floor *is* the C++ MPPI project.

## D2: Fair baselines, not a strawman. PROPOSED
Constant-μ MPC is a context row only. The real baselines are bank-adaptive MPPI (LLA-MPC style) and
learned-estimator MPPI, under matched information (same history window), a matched objective, and an
equal logged tuning budget. Why: LLA-MPC already comes within ~0.07 s of a known-friction oracle after a
40% friction drop.

## D3: Per-scenario feasibility labelling with oracles. PROPOSED
Labels: certified-infeasible / preview-required / reactive-feasible / undetermined. They come from a
friction-circle certificate plus reactive and clairvoyant oracle MPPI. Why: the original headline cell
(μ = 0.1, 15 m/s, R = 50 m) is infeasible for any controller. The car crosses the half-lane in ~1.0 s, and
slowing to √(μgR) ≈ 7 m/s would take ~90 m of straight braking.

## D4: Teacher sees current per-axle μ only. PROPOSED
No friction preview for the teacher. Preview belongs to the clairvoyant oracle, which is used for labelling
only. Tested explicitly as H3 (current-μ teacher vs preview teacher).

## D5: Pre-registered hypotheses replace target numbers. PROPOSED
H1: on reactive-feasible, in-family scenarios, the RMA student does *not* beat the best adaptive MPPI by
more than 5 pp (TOST equivalence). H2: under structural model mismatch, the regret curves cross. H3: as in D4.
Falsifiers and statistics are in the report, and are finalized in `PREREGISTRATION.md` (tag `prereg-v1`).

## D6: Build a small simulator in Warp; use the JAX gyms as references. PROPOSED
Why: f1tenth_gym dev-jax and INI's jax_pacejka are 1:10 scale with no documented friction randomization,
and they would put JAX next to P1's PyTorch stack. Warp passes data to PyTorch without copies and underpins
Newton/Isaac Lab 3.0. The simulator is infrastructure, not a claimed contribution. Throughput is not a headline.

## D7: C++ goes into one load-bearing C++17/CUDA MPPI library, about 20% of hours. PROPOSED
No ROS 2 node, no TensorRT, no C++ RL environment. Why: undergrad-eligible AV postings stress C++,
planning/controls, MPPI, and validation, while RoboSub already demonstrates ROS 2/Jetson/Docker. The
confirmatory sweeps run on the C++ library, so it carries real weight in the results.

## D8: Full-size passenger-car parameters; 1:10 file kept for later. PROPOSED
This matches the permissive real-data sources: commaSteeringControl (MIT) and MCity-WSPI winter runs
(CC-BY-4.0). Cut from scope: heightmap, Bekker soil, and the CBF monitor (now a post-MVP extension).

## D9: Repo stays private until G2 or the MVP; license TBD. PROPOSED
Make it public at G2 (Nov 30) if citing it on summer-2027 applications, otherwise at the MVP.
