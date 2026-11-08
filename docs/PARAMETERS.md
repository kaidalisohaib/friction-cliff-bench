# PARAMETERS: sources and plausibility of the frozen vehicle (G1, D13)

The truth vehicle is a **generic mid-size RWD sedan**, not a replica of one model. The aim is that each value sits inside the
range of real cars of that class and that we can say where the range comes from. Frozen by prereg amendment A1
(2026-11-08); the code values are in `sim/params.py`, guarded by `fingerprint()` and `tests/test_frozen.py`.

Reference car for cross-checks: **2020 Toyota Camry** (a common mid-size sedan with public specs). Values used
below: wheelbase 111.2 in = 2.82 m, curb weight 3,241–3,530 lb = 1,470–1,600 kg by trim, width 72.4 in = 1.84 m,
height 56.9 in = 1.45 m, Cd 0.32 (spec aggregators, see Sources); length ≈ 4.9 m (manufacturer spec, not re-checked here). It is FWD; we use RWD (prereg V8),
which matters only for the drive/brake split, not for the values below.

Confidence: **S** = sourced (a primary or reputable secondary source gives the range), **C** = cross-checked by
a physics or consistency argument, **E** = engineering choice (no single source; covered by DR or an H2 axis).

## 1. Mass and geometry
| Param | Value | Plausible range | Why / source | Conf. |
|---|---|---|---|---|
| m | 1600 kg | 1,450–1,750 kg | Camry curb weight 1,470–1,600 kg + one occupant ≈ 75 kg | S |
| lf / lr | 1.25 / 1.45 m (L = 2.70 m) | L 2.65–2.85 m | Camry L = 2.82 m, Civic/Corolla ≈ 2.70 m. Static front-axle share lr/L = 54%: in the 50–55% band typical of front-engine RWD sedans (FWD sedans ≈ 58–62%) | S/C |
| Iz | 2700 kg·m² | normalized Iz/(m·lf·lr) ≈ 0.85–1.05 | NHTSA's measured database (Heydinger et al. 1999) shows the normalized yaw inertia Iz/(m·a·b) is similar across passenger cars; the cited range is 0.85–1.05 (via Sakai 2016; Garrott et al. 1988 give regressions). Ours: 2700 / (1600·1.25·1.45) = **0.93** | S/C |
| h (CoG) | 0.55 m | 0.50–0.58 m | NHTSA static stability factor SSF = T/(2h) is ≈ 1.4–1.5 for sedans; with a mid-size track T ≈ 1.6 m that gives h ≈ 0.53–0.57 m. Large-sedan examples converted from published SSF tables ≈ 0.55–0.57 m | S/C |
| R_w | 0.33 m | 0.31–0.34 m | 215/55R17: rim radius 0.216 m + sidewall 0.55·0.215 m = 0.334 m (unloaded; loaded ≈ 0.32 m) | C |
| Jw | 2.0 kg·m² per axle (1.0 per wheel) | 1.0–1.5 per wheel (+ driveline on the driven axle) | A ≈ 20 kg 17" assembly with the tire mass near r ≈ 0.3 m gives ≈ 1–1.5 kg·m² (order-of-magnitude estimate; measured data: Metz et al., SAE 900760, paywalled). Low end; affects only how fast wheelspin/lockup develop, and V6 confirmed stability at 1 ms | E |
| footprint | 4.8 × 1.85 m | 4.7–4.9 × 1.8–1.9 m | Camry ≈ 4.9 × 1.84 m | S |

## 2. Resistances
| Param | Value | Plausible range | Why / source | Conf. |
|---|---|---|---|---|
| f_rr | 0.012 | 0.010–0.015 | Engineering tables for car tires on asphalt/concrete (0.010–0.015); typical tires "very close to 0.01" | S |
| CdA | 0.70 m² | 0.55–0.75 m² | Frontal area ≈ 0.85·width·height (standard estimate for sedans) = 0.85·1.84·1.45 = 2.26 m²; × Cd 0.32 = 0.72 m² | S/C |

## 3. Actuators
| Param | Value | Plausible range | Why / source | Conf. |
|---|---|---|---|---|
| δ max | 0.55 rad (31.5°, road wheel) | 30–38° | Bicycle-equivalent angle (between inner and outer wheel). Turning circle check: R ≈ L/tan δ ≈ 4.4 m at the rear axle → outer front wheel at ≈ 5.9 m → ≈ 11.7 m kerb-to-kerb diameter, the same order as typical mid-size sedans (≈ 11–12 m) | C |
| δ rate | 0.7 rad/s (road wheel) | — | × steering ratio ≈ 15 → ≈ 600°/s at the hand wheel: an emergency-capable actuator (EPS motors are specified up to ≈ 800°/s; 99.5% of normal driving stays below 90°/s, patent data). Deliberately not comfort-limited: the benchmark is about emergencies | S/E |
| τδ, τT | 0.08 s, 0.10 s | 0.05–0.15 s | First-order lags standing in for actuator bandwidth (≈ 2 Hz / 1.6 Hz). No single source | E |
| delay d | 1 step (20 ms) | 0–60 ms | Typical computation + bus latency; extra 0–40 ms is randomized for every method (D14); +60/100/140 ms is the H2 axis | E |
| drive force max | 0.4·m·g = 6.3 kN at the wheels | — | At 20 m/s = 126 kW at the wheels: plausible for a 150–220 kW sedan. Slightly below the dry rear-axle traction limit (≈ 0.46·m·g), so on dry the engine binds first and on low μ the tires do | C |
| brake force max | 1.0·m·g total | ≥ μ·m·g | Brakes can lock the wheels on any surface (real cars: well above 1 g of brake torque) | C |
| brake front share | 0.65 | 0.60–0.75 | The ideal share for deceleration a is (lr + h·a/g)/L; 0.65 is ideal at a ≈ 0.56 g, mid-range for the wet/snow regime this benchmark is about | C |
| drivetrain | RWD | — | prereg V8 | — |

## 4. Tires and handling (D15)
| Param | Value | Why / source | Conf. |
|---|---|---|---|
| Longitudinal (Bx, Cx, Ex) per surface | `sim/params.py` | Least-squares fit to Burckhardt (1993) μ-slip curves (`scripts/fit_surfaces.py`, max error ≈ 3% of peak) | S |
| Lateral dry/wet (By, Cy, Ey) | `sim/params.py` | Pacejka's typical lateral shape C = 1.3, cornering stiffness C_α/Fz = 17 /rad, peak at 7° / 6° | S |
| Lateral snow/ice | `sim/params.py` | CRREL winter-surface measurements: ice peaks at 2–4° then drops (C = 1.6, peak 3°); snow keeps rising to ≈ 15° (C = 1.0) | S |
| μ ranges | dry 0.85–1.0, wet 0.5–0.8, snow 0.20–0.35, ice 0.08–0.15 | prereg V7 (frozen at the tag); WSPI statistics appear in an appendix only (P5) | S |
| B_scale_f / B_scale_r | 0.80 / 1.52 | Understeer gradient ≈ 2.0°/g on dry; typical passenger cars 2–3°/g at the road wheel. To be sanity-checked on comma data in P5 | S/C |

## 5. Numerical constants (not physical; frozen so nothing drifts)
G = 9.81, ρ_air = 1.225, control dt 20 ms with 20 substeps of 1 ms, max delay 8 steps; low-speed blend
v_lo = 0.5 / v_hi = 2.0 m/s on |v|, τ_kin = 0.05 s, ε_v = 0.1 m/s; switches σ_relax = 0 and α = 0 (the nominal
model; H2 changes them).

## 7. Observation noise (P2, `sim/observe.py`; not part of the A1 freeze, which covers the truth vehicle)
| Channel | White-noise std | Per-episode bias std | Why |
|---|---|---|---|
| ax, ay | 0.05 m/s² | 0.05 m/s² | Consumer-grade automotive MEMS accelerometer at 50 Hz after the usual filtering | E |
| r (yaw rate) | 0.005 rad/s | 0.002 rad/s | MEMS gyro (≈ 0.3°/s noise, ≈ 0.1°/s residual bias) | E |
| wheel speeds | 0.1 rad/s | — | Production wheel-speed sensor quantization at 50 Hz | E |
| steering angle | 0.001 rad | — | Steering-angle sensor resolution ≈ 0.1° at the hand wheel / ratio | E |
| e_y, e_ψ, speed | 0.05 m, 0.005 rad, 0.05 m/s | — | Lane-level localization (camera + map) | E |
| vy (lateral velocity) | 0.1 m/s | — | Sideslip estimate (GNSS/INS fusion class); added by D24 so the policies see what the MPPI rollouts start from | E |
Check (prereg V4 note): the recovery band |Δr| ≤ 0.05 rad/s is 10× the gyro noise and |Δe_y| ≤ 0.15 m is 3× the
localization noise; the band is evaluated on the true state anyway, so noise cannot move it. These levels go
into a dated amendment together with the oracle settings before any test-split use.

## 6. What is weak, honestly
- Jw, τδ, τT and the delay are engineering choices. They are reasonable, and the delay is covered by DR and H2,
  but no source pins them.
- h and Iz are checked by ratios (SSF, normalized inertia) rather than by one measured car.
- Dry/wet lateral shapes are "typical", not one measured tire (D15). The comma check in P5 is the planned sanity test.
- None of this decides the comparison: every method faces the same truth car, and the H2 axes test model mismatch
  explicitly.

## Sources
- NHTSA inertial database: Heydinger et al., "Measured Vehicle Inertial Parameters: NHTSA's Data Through
  November 1998", SAE 1999-01-1336, https://saemobilus.sae.org/papers/measured-vehicle-inertial-parameters-nhtsas-data-november-1998-1999-01-1336
- Garrott et al., "Vehicle Inertial Parameters: Measured Values and Approximations", SAE 881767,
  https://saemobilus.sae.org/papers/vehicle-inertial-parameters-measured-values-approximations-881767
- Normalized yaw inertia 0.85–1.05 (citing a 1999 estimate): Sakai 2016, https://kindai.repo.nii.ac.jp/record/18525/files/AA12517632-20170600-0111.pdf ;
  similarity across vehicles: https://folk.ntnu.no/skoge/prost/proceedings/acc08/data/papers/0310.pdf
- SSF definition and sedan values: NHTSA SSF test procedure, https://webarchive.library.unt.edu/web/20161101030520/http://www.safercar.gov/staticfiles/safercar/NCAP/SSF_Test_Procedure-March2013.pdf ;
  https://isdacenter.org/vehicle-static-stability-factor/
- Rolling resistance: https://engineeringtoolbox.com/rolling-friction-resistance-d_1303.html ;
  https://pub.pg.edu.pl/en/download/146943-decreasing-co2-emissions-by-reducing-tire-rolling-resistance
- Frontal-area factor 0.85: https://fchartsoftware.com/ees/drag_coefficients/hs2230.htm
- Camry 2020 specs: https://www.iseecars.com/car/2020-toyota-camry-specs ; https://www.motormatchup.com/catalog/Toyota/Camry/2020/L-4DR-FWD-Sedan
- Steering rates (patent data, illustrative): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8862322 ;
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9969425
- Tire inertia measurements: Metz et al., SAE 900760, https://saemobilus.sae.org/papers/moments-inertia-mounted-unmounted-passenger-car-motorcycle-tires-900760
- Tire shapes: Burckhardt (1993); Pacejka, *Tire and Vehicle Dynamics*; CRREL winter-tire reports (see D15 and
  `scripts/fit_surfaces.py`).
