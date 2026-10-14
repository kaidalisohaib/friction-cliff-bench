# DESIGN: simulator, tracks, scenarios, shared task objective

Status: **design spec, pre-implementation.** Parameter values are *initial*: they must be justified with a
source (or a sanity check) in P1 and frozen by a dated prereg amendment at G1, before any test-split
labelling (D13). Anything marked ⚠ is an open choice.

---

## 1. Scope of the model

The model is a planar full-size passenger car on flat ground. It deliberately excludes heightmaps, grade, soil
or terramechanics, aero downforce, suspension/roll dynamics, and lateral load transfer (all out of scope, see
`DECISIONS.md` D8). The model must be rich enough to (a) reproduce limit-handling behaviour (sliding, spin,
plough), and (b) offer *structural* mismatch axes for H2.

## 2. Vehicle dynamics (truth simulator)

### 2.1 States, inputs
- Body: `X, Y, ψ, vx, vy, r` (world position, yaw; body-frame velocities; yaw rate).
- Wheels (per axle, lumped): `ωf, ωr`.
- Tire lag (relaxation length, mismatch axis): `Fyf_lag, Fyr_lag`.
- Actuators: `δ_act` (road-wheel angle), `T_act` (net drive/brake torque command after lag).
- Inputs (policy/MPPI action, normalized to [-1, 1], 50 Hz): `u_δ` (steering *target*) and `u_a` (longitudinal:
  >0 drive, <0 brake).

### 2.2 Equations (per substep)
- **Load transfer** (quasi-static, longitudinal only): `Fzf = m(g·lr − ax·h)/L`, `Fzr = m(g·lf + ax·h)/L`, with
  `ax` the previous substep's *inertial* longitudinal acceleration (`v̇x − r·vy`), clamped so Fz ≥ 5% of static.
- **Slip angles:** `αf = δ − atan2(vy + lf·r, |vx|)`, `αr = −atan2(vy − lr·r, |vx|)`. Below `v_min ≈ 1 m/s`, blend
  to a kinematic bicycle (tanh blend). Clamp α to ±π/2 − ε before any `tan`.
- **Slip ratios:** `κi = (R·ωi − v_xi) / max(|R·ωi|, |v_xi|, ε)`, where `v_xi` is the hub speed *in the wheel frame*
  (front is steered).
- **Tire forces:** Pacejka Magic Formula per axle, `F = D·sin(C·atan(Bs − E(Bs − atan(Bs))))`, with `D = μ·Fz`.
  **Combined slip** by the normalized slip-vector method (σx = κ/(1+κ), σy = tan α/(1+κ), σ = ‖σ‖, force direction
  along σ, magnitude MF(σ)). Mismatch variant: friction-ellipse scaling of pure-slip curves.
- **Surface-dependent shape**, not only μ. Each surface has its own (B, C, D=μ, E). Initial values come from the
  widely quoted "typical MF coefficients" table: dry (10, 1.9, 1.0, 0.97), wet (12, 2.3, 0.82, 1.0), snow
  (5, 2.0, 0.3, 1.0), ice (4, 2.0, 0.1, 1.0). ⚠ Find and cite a primary source in P1. The surface is sampled per axle.
- **Tire relaxation** (mismatch axis): `dFy_lag/dt = (|vx|/σ_relax)(Fy − Fy_lag)`, with σ_relax ∈ {0 (off), 0.3–0.6 m}.
- **Body:** planar rigid body with `Fxf, Fyf` rotated by δ; rolling resistance `f_rr·Fz`; aero drag `½ρCdA·vx|vx|`.
- **Wheels:** `Jw·ω̇ = T_drive_axle − T_brake_axle − R·Fx − b·ω`. Brakes cannot reverse wheel direction
  (zero-crossing clamp).
- **Actuators:** steering first-order lag τδ + rate limit + saturation; longitudinal first-order lag τT; pure
  delay of `d` control steps (DR / mismatch axis).
- **Integration:** semi-implicit Euler, physics dt = 1 ms, 20 substeps per 20 ms control step. ⚠ Confirm stability
  of the wheel-spin states at dt = 1 ms with the chosen Jw (V6).

### 2.3 Initial parameters (generic mid-size sedan; ⚠ justify in P1)
| Param | Value | Param | Value |
|---|---|---|---|
| m | 1600 kg | Iz | 2700 kg·m² |
| lf / lr | 1.25 / 1.45 m (L = 2.70 m) | h (CoG) | 0.55 m |
| R (wheel) | 0.33 m | Jw (per axle) | 2.0 kg·m² |
| footprint | 4.8 × 1.85 m | drivetrain | RWD (prereg V8); FWD/AWD as config, exploratory only |
| δ max / rate | 0.55 rad / 0.7 rad/s (road wheel) | τδ | 0.08 s |
| drive force max | ≈ 0.4·m·g at wheels | brake max | ≈ 1.0·m·g total, bias 0.65 front |
| τT | 0.10 s | control delay d | 1 step (20 ms) |
| f_rr | 0.012 | CdA | 0.7 m² |

### 2.4 Implementation notes
- Warp kernels, one thread per env. Parameters are passed as **explicitly constructed** structs. Never rely on Python
  default field values in `wp.struct` (they are not applied; this zeroes the parameters silently).
- Torch interop through zero-copy views (`wp.to_torch` / `wp.from_torch`). Seeded RNG in-kernel (`wp.rand_init`, `wp.randf`).
- Every API call must be verified against the installed Warp version's docs. Write tests first.
- A scalar NumPy reference implementation of the tire + one dynamics step is used for unit tests and CPU/GPU parity.

## 3. Tracks and scenarios

### 3.1 Track
- A Frenet reference path built from primitives: straight, constant-radius arc, clothoid transitions (no curvature
  jumps). Lane width **3.5 m**; the reference is the lane centre.
- The friction field is defined along the path, `μ(s)` with an optional lateral band, and sampled at each axle's
  projected `s`. The front axle meets a patch `L/v` before the rear.

### 3.2 Scenario parameters (sampled; strata define reporting cells)
| Factor | Range / levels (initial) |
|---|---|
| Entry speed v0 (= reference speed) | {10, 15, 20, 25} m/s strata, continuous within ±1 m/s |
| Geometry at drop | straight; arc R ∈ {50, 100, 200, 400} m |
| Drop location | straight / corner entry / apex / exit |
| Pre-drop surface | dry, μ0 ~ U[0.85, 1.0] |
| Post-drop surface | wet (μ≈0.5–0.8), snow (0.2–0.35), ice (0.08–0.15), with surface-specific shape params |
| Patch length | {10, 30, 100, ∞} m |
| Transition | step, or linear ramp over {2, 5} m |
| Initial condition | steady-state tracking at v0 on dry, small DR perturbation |
- An episode starts ≥ 2 s before the drop and ends at the patch exit + 3 s (or 8 s after the drop if the patch is ∞),
  or on termination.
- **Splits:** tuning split (~200 scenarios, any seed) and a frozen **test split** (≥ 1,000 per feasibility class for
  headline cells). The sampler spec + seed + SHA-256 of the generated test set are recorded in `PREREGISTRATION.md`.
- The literature μ ranges above are frozen in the test-split spec (prereg V7). MCity-WSPI MARWIS statistics (P5) are
  reported as an appendix realism check and never change the frozen test split.

## 4. Shared task objective (identical for MPPI cost and PPO reward)

Per control step: `c = w_lane·lane_violation² + w_ey·e_y² + w_v·(vx − v_ref)²/v_ref² + w_du·‖Δu‖² − w_prog·Δs`,
plus a terminal/termination penalty `C_fail` on departure or loss of control.
- `lane_violation` = distance of the furthest footprint corner beyond the lane edge (0 inside the lane).
- `v_ref` = entry speed. Controllers are *allowed* to slow down; the speed term is soft.
- **Weights (prereg V1):** w_ey = 1.0 m⁻², w_lane = 100 m⁻², w_v = 2.0, w_du = 0.1 (u normalized to [-1, 1]),
  w_prog = 0.1 m⁻¹, C_fail = 2000; per 20 ms control step.
- Weights are fixed once, before tuning, and are **not** a tuned hyperparameter of any method. PPO-only shaping
  terms (if any) are disclosed and kept out of evaluation.
- Episode terminations (training *and* eval): footprint departure beyond lane + 1.0 m (severe departure),
  |β| > 30°, or spin (|ψ − ψ_path| > 90°).

## 5. Simulator validation suite (gate G1, runs in CI on CPU)
- **V1 linear regime:** steady-state yaw rate vs `v·δ/(L + K_us·v²)` within 2% at small δ (K_us from parameters).
- **V2 limit cornering:** max steady speed on radius R vs `√(μgR)` within 3% (the surface peak equals D = μFz).
- **V3 tire function:** pure-slip curves equal the analytic MF to 1e-6. Combined slip reduces to pure slip on each axis.
- **V4 friction bound (S4):** |F| ≤ μ·Fz (+1e-6) on a dense (κ, α, Fz, μ) grid for every surface.
- **V5 μ-step transients:** rear-axle-first ice → oversteer (β grows with sign opposite to r). Front-axle-only ice →
  understeer. Signs and rough magnitudes checked against hand calculations.
- **V6 robustness:** zero NaN/Inf over 1e8 random-action env-steps including vx → 0, reverse, wheel lock, and deep slides.
- **V7 determinism/parity:** seeded GPU replay is bit-identical; CPU vs GPU trajectories agree within 1e-4 relative over 2 s.
- **V8 cross-implementation:** vs the independent scalar NumPy reference on matched parameters (D12; optional
  pure-lateral comparison vs jax_pacejka); document the error.
- Throughput is measured and logged but **not** a claimed result.
