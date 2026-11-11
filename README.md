# Friction-Cliff-Bench

**When the road suddenly turns to ice, can a learned policy that adapts from its own history keep a car on the
road better than an adaptive model-predictive controller, under equal information and an equal tuning budget?**
And in which scenarios can *nothing* save the car?

![Two MPPI controllers on a tuning scenario where the curve entry turns to ice. Left: the planner assumes dry road; its sampled plans (orange) curve sharply and the car spins off. Right: the planner is told the current friction and stays in lane.](docs/media/hero.gif)

*Tuning scenario, not a result. Orange lines are the planner's sampled plans, simulated with the planner's own
friction model: the friction-blind planner keeps believing it can turn.*

**Status (Nov 2026):** simulator, feasibility labelling, oracles, adaptive MPPI, RL arm and the C++/CUDA MPPI are
built and tested. The 4,000-scenario test pool is labelled and frozen
([class breakdown](docs/figures/labels_test_T0.md)), and the methods are being tuned on the tuning split.
**No controller results yet.** The protocol was
[pre-registered](docs/PREREGISTRATION.md) (`prereg-v1`) before any test scenario was labelled or evaluated; later
changes are dated amendments.

## What it is
A benchmark of vehicle control under **abrupt tire–road friction loss** (dry → wet / snow / ice), built so that the
comparison cannot quietly favour one side:

1. **Every scenario is labelled by what is physically possible.** A friction-circle certificate proves when no
   controller can stay in lane; oracle controllers that know the true friction show when recovery *is* possible,
   and whether it needs a preview of the road ahead. Classes: reactive-feasible, preview-required,
   certified-infeasible, undetermined. Controllers are only ranked where saving the car is possible.
2. **The contenders, with matched information and an equal, logged tuning budget:**
   - adaptive MPPI with a **model bank** (LLA-MPC style) or a **learned friction estimator**;
   - an **RMA-style teacher → student PPO** policy (the teacher sees friction; the student infers it from history);
   - a domain-randomized PPO policy, and a student of a teacher that previews the road ahead (H3).
3. **Known-friction oracles** (reactive and clairvoyant MPPI) give the regret reference on every scenario.
4. **Confirmatory runs use our own C++17/CUDA MPPI**, checked against the reference implementation.

The intended output is a regime map: *where each method wins, where nobody can win, and why*. Not "RL beats MPC".

## How it is built
```
sim/          GPU vehicle simulator in NVIDIA Warp: dynamic bicycle, combined-slip Pacejka tires, wheel spin,
              relaxation, actuator lag/delay, per-axle friction fields; float64 NumPy reference model
scenarios/    tracks (straight, clothoid curves R 50-400 m), scenario sampler, frozen hashed splits
feasibility/  analytic friction-circle certificate (soundness argument in docs/FEASIBILITY.md)
labelling/    oracle labelling + sanity gate S1
control/      batched Warp MPPI, oracles, model-bank and learned-estimator MPPI, C++ backend binding
rl/           GPU RL environment, PPO teacher / RMA student / DR policy, training library
mppi_cpp/     C++17/CUDA MPPI: __host__ __device__ dynamics and MPPI kernels, CPU + CUDA drivers, GoogleTest
eval/         episode runner, tuning harness with budget ledger, evaluation runner, statistics, analysis
scripts/      labelling, tuning, training, gates, benchmarks, figures
tests/        pytest suite (simulator vs reference, certificate soundness, MPPI, RL env, statistics, C++ equivalence)
```

**Verification habits:** simulator physics written test-first and cross-checked against an independent float64
model; the C++ MPPI reproduces Warp's random number generator bit for bit, so both implementations draw the same
samples and are compared on whole control steps (gate S3); frozen values (parameters, splits, oracle settings) are
fingerprinted in CI; tuning trials and GPU-hours are logged per method.

## What is pre-registered
Hypotheses H1–H3 with falsifiers, metrics and statistics (hierarchical bootstrap, TOST at a ±5 pp margin, Holm),
the scenario sampler and seeds, the tuning budget, the oracle settings, and the evaluation plan. See
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md); every later change is a dated amendment (A1–A5) with its
reason in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Run it
```bash
uv sync                                   # Python 3.12, PyTorch (CUDA 12.8+), warp-lang
uv run pytest -q                          # fast suite (GPU tests need a CUDA device); slow: -m slow
cmake -S mppi_cpp -B mppi_cpp/build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build mppi_cpp/build -j && ./mppi_cpp/build/fcb_tests      # GoogleTest
MPPI_BACKEND=cpp PYTHONPATH=. uv run pytest -q tests/test_mppi_cpp.py   # C++ vs Warp equivalence
```

## Grounding against real cars
The simulated car's steady-state understeer is compared with ten passenger-car platforms from openpilot driving logs
([commaSteeringControl](https://huggingface.co/datasets/commaai/commaSteeringControl), MIT). The steering ratio is
fitted, not assumed. Script, table and figure: [`docs/results/grounding/`](docs/results/grounding/comma_understeer.md).

## Limitations (by design)
Simulation only, with a planar single-track (bicycle) model; one vehicle; friction drops along the lane only.
Tuning, training and the benchmark run on one laptop GPU.

## Documents
| File | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Vehicle model, tracks, scenario space, shared objective, simulator validation |
| [`docs/PARAMETERS.md`](docs/PARAMETERS.md) | Source and plausible range of every vehicle/tire value |
| [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md) | Labels, the certificate and its soundness argument, oracles, gate S1 |
| [`docs/BASELINES.md`](docs/BASELINES.md) | Controller set, fairness checklist, MPPI / RL / C++ specs |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | Metrics, splits, seeds, statistics, reporting rules |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | Research questions, hypotheses, gates, amendments |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Closest work and how this differs |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Dated design decisions and why |
| [`docs/LOG.md`](docs/LOG.md) | Dated notebook, run verdicts, tuning ledger |
