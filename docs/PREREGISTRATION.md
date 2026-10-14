# PREREGISTRATION: **DRAFT** (not frozen)

Freeze procedure (end of P0): resolve every ⚠, fill the test-split spec, commit, `git tag prereg-v1`, push the tag.
After tagging, changes go in a dated amendment section at the bottom. Original text is never edited.

## Research questions
- **RQ1.** On reactive-feasible friction drops, does RMA-style policy adaptation beat adaptive MPPI when both get the
  same information and tuning budget?
- **RQ2.** How does each degrade when the truth's tire/actuator *structure* leaves the MPPI model family (and the RL
  training distribution)?
- **RQ3.** What privileged information should the teacher see: current friction only, or a preview?

## Confirmatory hypotheses
| ID | Hypothesis | Falsifier |
|---|---|---|
| **H1** | On reactive-feasible, in-family scenarios, P-S's lane-departure rate is **not lower than the better of M-B/M-L by more than 5 pp** (equivalence-or-worse) | The paired difference (best adaptive MPPI − P-S) has a 95% CI entirely above +5 pp, aggregated over ≥ 5 training seeds |
| **H2** | As the truth departs from the nominal family along the pre-specified mismatch axes, adaptive-MPPI regret grows faster than P-S/P-DR regret, and **the curves cross within the swept range** | No crossing up to the largest mismatch level, or the student's regret slope ≥ MPPI's |
| **H3** | A student distilled from the current-μ-only teacher (P-S) matches or beats the preview-teacher student (P-SP) on departure rate and dry pace | P-SP has a lower departure rate (CI excludes 0) while losing < 2% dry progress |

Note on H1: we *predict* that the "RL wins" headline does not survive a fair adaptive baseline. A clear RL win
would falsify H1 and is a reportable result.

## H2 mismatch axes and levels (⚠ finalize)
1. Combined-slip formulation: slip-vector method (nominal) → friction-ellipse scaling.
2. Tire relaxation length: 0 (nominal) → {0.2, 0.4, 0.6} m.
3. Actuator lag/delay: nominal → +{20, 40, 80} ms.
These axes are excluded from RL domain randomization *and* from the MPPI bank.

## Exploratory (no confirmatory claims)
- E1: history window W ∈ {0.1, 0.25, 0.5, 1.0} s for P-S, M-B, M-L.
- E2: performance vs per-step compute budget.
- E3: RMA latent regression vs DAgger action distillation.
- E4: does recovery latency track the axle transit time L/v (front-axle slip as a "free preview" for the rear)?
- E5 (stretch): the sim-trained μ estimator on MCity-WSPI logs vs MARWIS friction.

## Sanity gates (failures = bugs, not results)
- S1: no oracle succeeds in a certified-infeasible scenario.
- S2: the teacher is within ⚠ X pp of O-R's departure rate on reactive-feasible tuning scenarios.
- S3: C++ MPPI ≡ Warp MPPI (fixed noise) on 1,000 random states.
- S4: |F| ≤ μF_z on the dense slip grid for every surface.

## Fixed in advance
- Metrics and statistics: `EVAL_PROTOCOL.md` as of the tag.
- Objective weights: `DESIGN.md` §4 values as of the tag (⚠ fill numbers).
- Tuning budget per method: ⚠ N Optuna trials and ⚠ G GPU-hours.
- Marginal band threshold and recovery-latency band: ⚠.
- Test split: sampler spec ⚠, seed ⚠, SHA-256 ⚠, ≥ 1,000 scenarios per feasibility class in headline cells.

## Amendments (post-tag, dated)
_(none)_
