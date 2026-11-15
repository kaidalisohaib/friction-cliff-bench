# Exploratory E1-E3 (harness output of `scripts/evaluate/exploratory.py`)

Pre-declared 500-scenario reactive-feasible test subset (N = 500 scenarios × 5 seeds), nominal truth. Departure rate with a hierarchical-bootstrap 95% CI; differences are paired against the tuned setting. Exploratory: outside the Holm family, nothing tuned on these results.

| variant | departure [95% CI] | per seed |
|---|---|---|
| M-B_K256 | 0.062 [0.044, 0.082] | 0.066, 0.066, 0.052, 0.070, 0.058 |
| M-B_K4096 | 0.020 [0.009, 0.032] | 0.024, 0.016, 0.024, 0.020, 0.016 |
| M-B_W12 | 0.035 [0.023, 0.048] | 0.030, 0.038, 0.040, 0.032, 0.036 |
| M-B_W25 | 0.044 [0.030, 0.060] | 0.042, 0.044, 0.042, 0.046, 0.048 |
| M-B_W5 | 0.033 [0.021, 0.046] | 0.036, 0.030, 0.034, 0.034, 0.032 |
| M-B_W50 | 0.050 [0.034, 0.067] | 0.054, 0.052, 0.048, 0.040, 0.056 |
| M-L_K256 | 0.069 [0.050, 0.089] | 0.062, 0.074, 0.072, 0.066, 0.072 |
| M-L_K4096 | 0.070 [0.049, 0.093] | 0.074, 0.090, 0.060, 0.056, 0.068 |
| M-L_W12 | 0.066 [0.047, 0.087] | 0.072, 0.060, 0.060, 0.080, 0.060 |
| M-L_W25 | 0.068 [0.048, 0.090] | 0.074, 0.086, 0.060, 0.052, 0.068 |
| M-L_W5 | 0.071 [0.052, 0.092] | 0.062, 0.074, 0.068, 0.068, 0.082 |
| M-L_W50 | 0.090 [0.069, 0.114] | 0.094, 0.096, 0.076, 0.096, 0.090 |
| P-S-act | 0.173 [0.023, 0.404] | 0.000, 0.028, 0.160, 0.054, 0.622 |
| P-S_W12 | 0.190 [0.025, 0.454] | 0.000, 0.034, 0.156, 0.054, 0.706 |
| P-S_W25 | 0.185 [0.023, 0.447] | 0.016, 0.010, 0.148, 0.056, 0.696 |
| P-S_W5 | 0.189 [0.014, 0.463] | 0.000, 0.006, 0.158, 0.058, 0.722 |
| P-S_W50 | 0.188 [0.026, 0.451] | 0.000, 0.036, 0.146, 0.056, 0.704 |

| paired difference | departure [95% CI] |
|---|---|
| M-B_W12 - M-B_W5 | +0.2 pp [-0.7, +1.2] |
| M-B_W25 - M-B_W5 | +1.1 pp [+0.0, +2.3] |
| M-B_W50 - M-B_W5 | +1.7 pp [+0.2, +3.2] |
| M-L_W12 - M-L_W5 | -0.4 pp [-2.0, +1.1] |
| M-L_W25 - M-L_W5 | -0.3 pp [-2.0, +1.4] |
| M-L_W50 - M-L_W5 | +2.0 pp [+0.2, +3.9] |
| P-S_W5 - P-S_W12 | -0.1 pp [-30.5, +31.4] |
| P-S_W25 - P-S_W12 | -0.5 pp [-30.3, +29.9] |
| P-S_W50 - P-S_W12 | -0.2 pp [-29.8, +30.1] |
| M-B_K256 - M-B_W5 (K=1024) | +2.9 pp [+1.4, +4.5] |
| M-B_K4096 - M-B_W5 (K=1024) | -1.3 pp [-2.4, -0.3] |
| M-L_K256 - M-L_W5 (K=1024) | -0.2 pp [-1.5, +1.2] |
| M-L_K4096 - M-L_W5 (K=1024) | -0.1 pp [-1.8, +1.7] |
| P-S-act - P-S_W12 | -1.7 pp [-30.5, +26.7] |

Student variants share their teacher's seed, so they are also paired by seed. Resampling seeds jointly (added after the first analysis; the independent-seed intervals above stay the reported default):

| paired difference (seeds paired) | departure [95% CI] |
|---|---|
| P-S_W5 - P-S_W12 | -0.1 pp [-1.7, +1.4] |
| P-S_W25 - P-S_W12 | -0.5 pp [-2.0, +0.9] |
| P-S_W50 - P-S_W12 | -0.2 pp [-1.1, +0.8] |
| P-S-act - P-S_W12 | -1.7 pp [-5.3, +0.2] |
