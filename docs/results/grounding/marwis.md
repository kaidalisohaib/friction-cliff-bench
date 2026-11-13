# MARWIS friction proxy vs simulator mu ranges (harness output of `scripts/wspi_marwis.py`)

MARWIS reports an optical grip proxy (0.82 = dry), not tire peak friction; compare ratios to dry. Medians with 95% cluster-bootstrap CIs over drives.

| class | samples | drives | median [95% CI] | 5–95% | ratio to dry median [95% CI] |
|---|---|---|---|---|---|
| dry | 147 | 1 | 0.820 [0.820, 0.820] | 0.820–0.820 | 1.00 [1.00, 1.00] |
| snow | 2248 | 3 | 0.250 [0.242, 0.284] | 0.216–0.295 | 0.30 [0.30, 0.35] |
| snow/ice | 1436 | 3 | 0.254 [0.248, 0.278] | 0.213–0.304 | 0.31 [0.30, 0.34] |

Simulator (sampled mu ranges and their ratio to dry): dry 0.85–1.00 (ratio 0.85–1.18); wet 0.50–0.80 (ratio 0.50–0.94); snow 0.20–0.35 (ratio 0.20–0.41); ice 0.08–0.15 (ratio 0.08–0.18).

Logged every 0.42 s (median); the reported value changes every 10.0 s (median time between changes).

Kalamazoo 2022–23 winter drives (unlabelled; 1972 samples, 21 drives): quantiles 0.01: 0.19, 0.05: 0.28, 0.25: 0.38, 0.5: 0.68, 0.75: 0.79, 0.95: 0.82, min 0.10; fraction below 0.4: 0.270 [0.102, 0.373].
