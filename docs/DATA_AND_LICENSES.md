# DATA_AND_LICENSES

Rule: **nothing NC-licensed is vendored or used for published claims.** Link to it instead of copying it.
Own code: license ⚠ (MIT suggested), decided before going public (`DECISIONS.md` D9).

| Resource | Use here | License / terms | Notes |
|---|---|---|---|
| NVIDIA Warp (`warp-lang`) | Simulator, reference MPPI | ⚠ check the current license at install | Verify every API against the installed version |
| PyTorch (CUDA ≥ 12.8 build) | PPO, estimators | BSD-style | RTX 5090 Laptop = sm_120 needs CUDA 12.8+ wheels |
| rsl-rl-lib | PPO | BSD-3 (check) | Pin the version |
| Optuna | Tuning harness | MIT | — |
| CUDA, CMake, GoogleTest, nanobind/pybind11 | C++ MPPI | permissive | — |
| **commaSteeringControl** (HF `commaai/commaSteeringControl`) | P5 sanity check: realistic lateral accel vs steering angle/speed (understeer-gradient range) for passenger cars | **MIT** | ~12,500 h, hundreds of car models; fields incl. `latAccelSteeringAngle`, `roll`, `vEgo` |
| **MCity-WSPI winter physics** (HF `AD-Assurance-Lab/winter-driving-dataset`; GitHub `AD-Assurance-Lab/dataset--winter-driving--main`) | P5: μ ranges and transition sharpness from MARWIS; E5 stretch | **CC-BY-4.0** (HF card); cite Tye et al., "MI-Snow1000", SAE Int. J. CAV 9(2), 2026 | ~85 GB; new repo (Jun 2026), inconsistent names; 25 Hz nearest-neighbour resampling; wheel speeds in rad/s; MARWIS = optical road-sensor proxy, **not** tire μ; download a subset only |
| comma2k19 | Optional CAN wheel speeds | MIT | Old tooling (Py 2.7/3.6) |
| comma `controls_challenge` | Reference only | **No license file**, so link, don't copy | Archived |
| INI `f1tenth_development_gym` (jax_pacejka) | V8 cross-check reference | ⚠ check in P0 | Don't vendor until checked |
| RoboRacer `f1tenth_gym` dev-jax | Possible upstream contribution (friction fields) | MIT | 1:10 scale |
| LLA-MPC repo (DRIVE-LAB-CMU) | Reference for M-B | ⚠ check in P0; does it run? 4 h timebox | Reimplement natively if blocked |
| BeamNG.tech | Stretch: out-of-family truth sim | Academic, non-commercial | Publish own code/results only |
| RACECAR (IAC), CADC | **Avoid** | CC BY-NC 4.0 | RACECAR also lacks wheel/steering channels |
