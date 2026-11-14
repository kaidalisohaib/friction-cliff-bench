# RL follow-up on test pool v2 (prereg A9; harness output of `eval/followup.py`)

Reactive-feasible v2 scenarios: N = 2,233; seeds per method as listed. 95% hierarchical-bootstrap CIs.

| method | seeds | departure [95% CI] | per seed | dry pace tax [95% CI] |
|---|---|---|---|---|
| O-R | 5 | 0.0400 [0.0325, 0.0474] | 0.038, 0.044, 0.038, 0.042, 0.038 | 0.0% [-0.0, 0.0] |
| M-B | 5 | 0.0325 [0.0263, 0.0393] | 0.036, 0.034, 0.030, 0.034, 0.030 | 4.2% [4.1, 4.3] |
| M-Bc | 5 | 0.0055 [0.0029, 0.0084] | 0.008, 0.004, 0.004, 0.004, 0.007 | 13.8% [13.5, 14.1] |
| P-S | 10 | 0.1026 [0.0146, 0.2448] | 0.000, 0.026, 0.160, 0.058, 0.704, 0.041, 0.002, 0.000, 0.001, 0.034 | 18.7% [14.9, 22.7] |
| P-SP | 5 | 0.0131 [0.0013, 0.0261] | 0.001, 0.000, 0.029, 0.004, 0.030 | 13.3% [7.6, 17.8] |
| P-S2 | 10 | 0.0236 [0.0050, 0.0508] | 0.000, 0.013, 0.135, 0.014, 0.041, 0.028, 0.005, 0.000, 0.000, 0.000 | 18.8% [15.7, 21.8] |
| P-SP2 | 10 | 0.0085 [0.0037, 0.0142] | 0.001, 0.001, 0.010, 0.004, 0.021, 0.008, 0.001, 0.003, 0.011, 0.023 | 14.1% [12.4, 16.0] |

**H4** (non-inferior to M-B at matched pace, margin 2 pp): departure(P-SP2) − departure(M-Bc) = +0.3 pp [-0.3, +0.9]; dry pace tax difference +0.3 pp [-1.4, +2.2]; non-inferior; superiority not shown (bootstrap p < 1/10,000).

**H5** (seed reliability: gate S2 per seed, departure ≤ O-R + 5 pp = 0.090): P-S 8/10 seeds pass; P-SP 5/5 seeds pass; P-S2 9/10 seeds pass; P-SP2 10/10 seeds pass. Holds for P-SP2.

## Exploratory (pre-declared in A9)
- The checkpoint-selection fix on the current-friction student: departure(P-S) − departure(P-S2) = +7.9 pp [-1.5, +22.4].
- Whole recipe (preview teacher + validation selection): departure(P-S) − departure(P-SP2) = +9.4 pp [+0.6, +23.7].
- The fix on the preview-teacher student: departure(P-SP) − departure(P-SP2) = +0.5 pp [-0.8, +1.8].
- Top-5-by-validation P-SP2 seeds [7, 1, 0, 6, 3]: departure 0.0022 [0.0006, 0.0043]; minus M-Bc -0.3 pp [-0.7, -0.0].
- *Post hoc* (added after the v2 results, not pre-declared): the top-5 subset's dry pace tax is 16.0% [13.6, 18.1], +2.2 pp [-0.2, +4.2] relative to M-Bc, so the top-5 comparison above is within the A9 pace tolerance (3 pp).
