# friction-cliff-bench

**Status: planning (P0). No code and no results yet.**

A feasibility-audited, parity-controlled benchmark of how a car's controller copes with **sudden tire–road
friction loss** (dry asphalt → wet / snow / ice).

1. **Label every scenario by what is physically possible:** certified infeasible, preview-required,
   reactive-feasible, or undetermined. This uses a friction-circle certificate plus oracle controllers that know
   the true friction.
2. **On the reactive-feasible scenarios**, compare an **RMA-style teacher → student PPO policy** (infers friction
   from its own recent history) against **adaptive MPPI** (model-bank and learned-estimator variants), with equal
   information and an equal, logged tuning budget.
3. Run the confirmatory sweeps on a **C++17/CUDA MPPI library**, verified against a reference implementation.

The output is a regime map (*where each method wins, where nobody can win, and why*), not "RL beats MPC".

**Thesis.** At abrupt friction loss, whether learned policy-side adaptation or model-side adaptive MPC wins depends
on whether the scenario is physically recoverable and on how far reality departs from the controller's model
family. A feasibility-audited benchmark can map those regimes on one laptop GPU.

## Documents
| File | Contents |
|---|---|
| [the project rules](the project rules) | Orientation + working rules for agents (and humans) |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Vehicle model, parameters, tracks, scenario space, shared objective, simulator validation suite |
| [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) | Labels, analytic certificate (soundness), oracles, sanity gate S1 |
| [`docs/BASELINES.md`](docs/BASELINES.md) | Controller set, fairness checklist, MPPI/RL/C++ implementation specs |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | Metrics, splits, seeds, statistics, reporting rules |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | RQs, H1–H3 + falsifiers, E1–E5, sanity gates (**draft until tag `prereg-v1`**) |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Closest work, "how we differ", scoop-watch log |
| [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md) | Tools/datasets and what may be published |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Dated design decisions and why |
| [`docs/LOG.md`](docs/LOG.md) | Dated notebook, run verdicts, hours + tuning ledger |

## Planned code layout (arrives with its phase)
```
sim/         Warp vehicle simulator + NumPy reference          (P1)
scenarios/   track/Frenet utils, scenario sampler, splits      (P1)
feasibility/ analytic certificate, oracle labelling            (P1–P2)
mppi/        Warp reference MPPI (+ bank, estimator variants)  (P2–P3)
rl/          PPO teacher/student/ablations                     (P3)
mppi_cpp/    C++17/CUDA MPPI, GoogleTest, bindings             (P4)
eval/        harness, metrics, statistics, figures             (P2–P4)
configs/  scripts/  tests/
```
