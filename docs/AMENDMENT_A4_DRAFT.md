# DRAFT prereg amendment A4: oracle settings, episode handover, witnesses, observation noise

**Status: DRAFT for a final review. Not part of the pre-registration until approved. Once approved it
is copied, dated, into `PREREGISTRATION.md` § Amendments.** It must be adopted before any test-split scenario is
labelled or evaluated (FEASIBILITY §3: oracle settings are frozen before labelling; prereg V7 compute note).
Everything below was chosen on the **tuning split only** (pilot logs in `runs/oracle_pilot*`, summarized in LOG).

## A4.1 Episode handover (D19)
Every scenario is driven from s = 0 by the deterministic prefix follower (`control/episode.py`) on dry road, until
the front axle is 4.0 s (at current speed) from the drop. The resulting state is the shared starting state for
every method. Episode end, success and termination are unchanged.

## A4.2 Oracle MPPI settings (FEASIBILITY §3, replaces "≈ 16k samples, ≈ 3 s, N = 5")
| Setting | O-R | O-C | Why |
|---|---|---|---|
| horizon H | 150 steps (3.0 s) | 150 steps (3.0 s) | as in FEASIBILITY §3 |
| samples K | 1024 | 1024 | K-check on the full tuning split (200 × 5 seeds): K = 4096 changed 2 of 200 labels (one each way) at 3.5× the cost |
| noise σ (steer, long) | 0.05, 0.3 | 0.05, 0.3 | pilot grid (16 + 8 configs, 40 tuning scenarios) |
| AR(1) ρ | 0.0 | 0.9 | pilot: white noise gave O-R the most witnesses, correlated noise gave O-C the most |
| temperature λ | 10 (fixed) | 10 (fixed) | pilot grid; ESS-targeting gave no gain in the hard-case check (4 scenarios × 2 seeds) |
| ancillary PD sample | yes (current μ) | yes (friction map ahead) | cold-start fix (P2_REPORT §12) |
| seeds N | 5 | up to 5, stop at first success | O-C is run only where no reactive witness exists |

## A4.3 Reactive witnesses (D20)
A scenario is *reactive-feasible* if O-R succeeds (≥ 1 of N seeds) **or** the prefix follower W-PD, which has no
friction information, succeeds when it keeps driving after the handover. Rationale: any controller with at most
reactive information that survives is a valid witness. O-R remains the regret reference (BASELINES §1).
Both label variants are logged.

## A4.4 Observation noise (BASELINES §2, `PARAMETERS.md` §7)
IMU accelerations 0.05 m/s² white + 0.05 m/s² bias; yaw rate 0.005 rad/s + 0.002 rad/s bias; wheel speeds 0.1 rad/s;
steering angle 0.001 rad; e_y 0.05 m, e_ψ 0.005 rad, speed 0.05 m/s. Identical for every non-privileged method.

## A4.5 Compute budget (D18)
Measured on the RTX 5090 Laptop (shared): ≈ 3 ms per car per control step for O-R at K = 4096, H = 150, ≈ 1.6× that
for O-C. At the chosen settings the full tuning split (200 scenarios, 5 O-R seeds + conditional O-C) took
≈ 9 GPU-min (550 s wall); the 4,000-scenario test pool is projected at ≈ 3 h wall, plus top-up blocks if needed.
This replaces "≈ 16k samples"; total samples per scenario (5 seeds × 1024) are reported with the labels.
