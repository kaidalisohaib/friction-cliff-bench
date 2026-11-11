# Feasibility classes: test_T0

Harness output (`scripts/fig_labels.py`), N = 4000 scenarios.

| class | n | share | Wilson 95% |
|---|---|---|---|
| reactive-feasible | 3009 | 0.752 | [0.739, 0.765] |
| preview-required | 755 | 0.189 | [0.177, 0.201] |
| certified-infeasible | 29 | 0.007 | [0.005, 0.010] |
| undetermined | 207 | 0.052 | [0.045, 0.059] |

| flag | n |
|---|---|
| adaptation_required | 363 |
| marginal | 230 |
| certified_reactive_infeasible | 18 |
| certified_nominal | 47 |

O-R-only rule (prereg A3, logged for comparison): reactive-feasible 2977, preview-required 787, certified-infeasible 29, undetermined 207
