# Learned-policy latency per control step (harness output of `scripts/bench_policy.py`)

Student (GRU over W = 12 feature vectors + MLP actor), `runs/rl_fix/stu_tpval_s0.pt`. Median and 2.5–97.5% range of N = 300 timed calls; GPU shared during the run: True.

| device | threads | cars | median [ms] | 2.5–97.5% [ms] |
|---|---|---|---|---|
| cuda | — | 1 | 0.158 | 0.147–1.282 |
| cuda | — | 64 | 0.188 | 0.172–1.293 |
| cpu | 1 | 1 | 0.194 | 0.192–0.200 |
| cpu | 1 | 64 | 0.896 | 0.889–0.905 |
| cpu | 24 | 1 | 0.195 | 0.192–0.202 |
| cpu | 24 | 64 | 203.010 | 165.962–334.178 |

All-thread CPU cells contend with any other CPU load on the machine (here: another project's GPU job was running); the single-thread cells are the deployment-relevant ones.
