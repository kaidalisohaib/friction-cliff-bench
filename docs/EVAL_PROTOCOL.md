# EVAL_PROTOCOL: metrics, splits, statistics, reporting

## 1. Metrics (per episode)
**Primary:** **lane departure.** 1 if any footprint corner crosses the lane edge at any time, else 0. Reported as
a rate per feasibility class (headline: reactive-feasible).

**Secondary:**
- *Adaptation regret* vs the reactive oracle O-R on the same scenario instance: Δ max |e_y|, Δ time out of lane,
  Δ progress over [t_drop, t_drop + 5 s].
- *Recovery latency:* time from the first axle entering the patch until |e_y| and |r − r_ref| return to within a
  band (⚠ fix in prereg) of O-R's trace, sustained 0.5 s. Censored at episode end.
- *Loss of control:* |β| > 20° or spin.
- Peak |β|, peak |r − r_ref|, RMS jerk (comfort), μ-estimate error (M-B, M-L, latent probe for P-S).
- Per-step compute latency (same GPU).
- *Dry pace tax:* progress deficit vs O-R on dry-only scenarios (does adaptation cost speed when nothing happens?).

## 2. Splits and seeds
- Tuning split: ~200 scenarios plus anything sampled during training. Test split: frozen; generated from the
  prereg sampler spec + seed; SHA-256 recorded. **No method is ever tuned or trained on the test split.**
- Learned methods: ≥ 5 independent training seeds (10 for H1 if one run < 1 h). MPPI methods: 5 noise seeds per scenario.
- **Paired design:** every method runs on identical scenario instances.

## 3. Statistics
- Binary rates: Wilson 95% intervals. With n = 1,000 per class, the half-width is ≈ ±3.1 pp at p = 0.5 and ≈ ±1.9 pp at p = 0.1.
- Paired differences and continuous metrics: **hierarchical bootstrap** (10,000 resamples; resample training seeds,
  then scenarios).
- H1 uses a TOST equivalence check at the pre-registered ±5 pp margin (see `PREREGISTRATION.md`).
- Confirmatory family H1–H3: Holm–Bonferroni at α = 0.05.
- All seeds reported (never best-of). Failed/diverged training runs count as runs.

## 4. Reporting rules
- Always break results down by feasibility class; show curves over the sweep axes, not single numbers.
- Every reported number has N and a CI. No number appears in README or report unless produced by the harness.
- Exploratory analyses are labelled *exploratory*.
- One script regenerates every figure and table from raw JSON logs (gate G4).
- Logs: one JSON line per (method, seed, scenario) with all metrics + config hash + git SHA.
