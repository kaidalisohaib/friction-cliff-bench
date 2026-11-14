"""Exploratory analyses E1-E3 (PREREGISTRATION.md; outside the Holm family, labelled exploratory everywhere).

  E1  history window W in {5, 12, 25, 50} steps (0.1, 0.24, 0.5, 1.0 s; 12 is the nearest step count to 0.25 s and
      the student's tuned value) for M-B (bank look-back), M-L (estimator window) and P-S (student history).
  E2  MPPI sample budget K in {256, 1024, 4096} for M-B and M-L at their tuned settings (compute vs performance).
  E3  RMA latent regression (P-S) vs DAgger action distillation (P-S-act): same teachers, same history length,
      same data collection (on-policy student rollouts), loss on the teacher's actions instead of its latent.

Evaluation set: the pre-declared 500-scenario reactive-feasible subset of the test split (the H2/dry/trace subset,
prereg A5), nominal truth, 5 seeds, batch 64, C++ MPPI backend. Nothing is tuned here; every non-varied setting is
the tuned one. Training uses the training library only; the extra checkpoints go to runs/exploratory/ckpt/.

  PYTHONPATH=. uv run python scripts/exploratory.py train
  MPPI_BACKEND=cpp PYTHONPATH=. uv run python scripts/exploratory.py eval [--variants M-B_W12,...]
  PYTHONPATH=. uv run python scripts/exploratory.py analyze        # docs/results/exploratory.md, figure, tex
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path("runs/eval_test/exploratory")
CKPT = Path("runs/exploratory/ckpt")
FINAL = Path("runs/final")
SEEDS = (0, 1, 2, 3, 4)
WINDOWS = (5, 12, 25, 50)
KS = (256, 1024, 4096)
TUNED_W = {"M-B": 5, "M-L": 5, "P-S": 12}
TUNED_K = 1024


def variants() -> dict[str, dict]:
    """name -> spec. K = 1024 runs are the tuned-W runs of E1 (same configuration), so they are aliases."""
    v = {}
    for m in ("M-B", "M-L", "P-S"):
        for w in WINDOWS:
            v[f"{m}_W{w}"] = {"exp": "E1", "method": m, "W": w}
    for m in ("M-B", "M-L"):
        for k in KS:
            if k != TUNED_K:
                v[f"{m}_K{k}"] = {"exp": "E2", "method": m, "K": k}
    v["P-S-act"] = {"exp": "E3", "method": "P-S-act"}
    return v


def alias(name: str) -> str:
    for m in ("M-B", "M-L"):
        if name == f"{m}_K{TUNED_K}":
            return f"{m}_W{TUNED_W[m]}"
    return name


# --- training ------------------------------------------------------------------------------------------------------
def train(device: str) -> None:
    import torch

    from control.estimator import train_estimator
    from rl import library as Lb
    from rl.ppo import ActorCritic, DistillConfig, distill, save_policy
    from scripts.tune_method import DISTILL_STEPS
    from eval.runner import best
    CKPT.mkdir(parents=True, exist_ok=True)
    lib = Lb.load(Path("runs/rl/lib_train_v1.npz"))
    bp_s, bp_l = best("P-S")["best_params"], best("M-L")["best_params"]
    jobs = [("P-S", w, "latent") for w in WINDOWS if w != TUNED_W["P-S"]] + [("P-S-act", TUNED_W["P-S"], "action")] \
        + [("M-L", w, None) for w in WINDOWS if w != TUNED_W["M-L"]]
    for m, w, target in jobs:
        for seed in SEEDS:
            path = CKPT / (f"{m}_s{seed}.pt" if m == "P-S-act" else f"{m}_W{w}_s{seed}.pt")
            if path.exists():
                continue
            t0 = time.time()
            if m.startswith("P-S"):
                tpath = FINAL / f"P-T_s{seed}.pt"
                ck = torch.load(tpath, weights_only=False)
                teacher = ActorCritic(ck["meta"]["kind"], width=ck["meta"]["cfg"]["width"]).to(device)
                teacher.load_state_dict(ck["state_dict"])
                dcfg = DistillConfig(hist_len=w, hidden=bp_s["hidden"], lr=bp_s["lr"], total_steps=DISTILL_STEPS,
                                     target=target)
                model, hist = distill(teacher, lib, dcfg, seed=seed, device=device)
                meta = {"kind": "student", "teacher": str(tpath), "cfg": asdict(dcfg), "seed": seed, "exploratory": m}
            else:
                model, hist = train_estimator(lib, w, hidden=bp_l["hidden"], seed=seed, device=device)
                meta = {"kind": "estimator", "W": w, "hidden": bp_l["hidden"], "seed": seed, "exploratory": m}
            save_policy(path, model, meta)
            with open(CKPT / "ledger.jsonl", "a") as f:
                f.write(json.dumps({"ckpt": str(path), "gpu_s": time.time() - t0}) + "\n")
            print(f"{path.name} in {time.time() - t0:.0f}s", flush=True)


# --- evaluation ----------------------------------------------------------------------------------------------------
def factory(spec: dict, seed: int):
    from control.adaptive import BankMPPI, LearnedMPPI
    from control.mppi import MPPIConfig
    from eval.runner import load_estimator, load_policy, mppi_cfg
    from rl.policy import policy_controller
    m = spec["method"]
    if m in ("M-B", "M-L"):
        cfg = mppi_cfg(m)
        if "K" in spec:
            cfg = MPPIConfig(**{**cfg.__dict__, "K": spec["K"]})
        w = spec.get("W", TUNED_W[m])
        if m == "M-B":
            return lambda sim, sub, ss, sd: BankMPPI(sim, sub, cfg, sd, ss, w)
        p = FINAL / f"M-L_s{seed}.pt" if w == TUNED_W["M-L"] else CKPT / f"M-L_W{w}_s{seed}.pt"
        est = load_estimator(p)
        return lambda sim, sub, ss, sd: LearnedMPPI(sim, sub, cfg, sd, ss, est)
    if m == "P-S":
        w = spec["W"]
        p = FINAL / f"P-S_s{seed}.pt" if w == TUNED_W["P-S"] else CKPT / f"P-S_W{w}_s{seed}.pt"
    else:  # P-S-act
        p = CKPT / f"P-S-act_s{seed}.pt"
    model, kind, hl = load_policy(p)
    return lambda sim, sub, ss, sd: policy_controller(sim, sub, model, kind, hl, sd)


def run_eval(names: list[str], device: str, batch: int) -> None:
    from eval.runner import handover, h2_subset, load_labels, run_controller
    from labelling.label import git_sha
    from scenarios import sampler as S
    from scenarios import track as T
    from sim.params import VehicleParams
    labels = load_labels(["labels/test_v1_T0.jsonl"])
    scs = h2_subset(labels, [sc for sc in S.load_jsonl("splits/test_v1_T0.jsonl")[1] if sc.sid in labels])
    truth, tracks = VehicleParams(), T.all_tracks()
    ROOT.mkdir(parents=True, exist_ok=True)
    snap = handover(scs, tracks, truth, device, cache=ROOT / "handover.npz")
    (ROOT / "meta.json").write_text(json.dumps({"n": len(scs), "git": git_sha(), "batch": batch,
                                                 "subset": "prereg A5 H2 subset"}, indent=1))
    V = variants()
    for name in names:
        name = alias(name)
        # the controller's noise seeds follow its base method, so variants of one method share them
        method_id = V[name]["method"] if V[name]["method"] != "P-S-act" else "P-S"
        for seed in SEEDS:
            path = ROOT / f"{name}_s{seed}.jsonl"
            if path.exists() and sum(1 for _ in open(path)) == len(scs):
                continue
            t0 = time.time()
            recs = run_controller(scs, snap, factory(V[name], seed), method_id, seed, device, batch, tracks, truth)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for sc in scs:
                    f.write(json.dumps({**asdict(recs[sc.sid]), "variant": name}) + "\n")
            tmp.rename(path)
            print(f"[exploratory] {name} seed {seed}: {len(scs)} runs in {time.time() - t0:.0f}s", flush=True)


# --- analysis ------------------------------------------------------------------------------------------------------
def analyze(out_md: Path, out_fig: Path, out_tex: Path | None) -> dict:
    from eval import analysis as A
    from eval import stats as St
    V = variants()
    names = sorted({alias(n) for n in V} | {f"{m}_W{TUNED_W[m]}" for m in TUNED_W})
    d = A.load_set(ROOT.parent, ROOT.name, names)
    res = {}
    for n in names:
        if n not in d or len(d[n]["recs"]) < len(SEEDS):
            continue
        a = A.mat(d, n, "departed")
        bm = A.boot_means({"x": a}, n_boot=2000, seed=11)["x"]
        res[n] = {"departure": A.summary(bm, float(np.nanmean(a))),
                  "per_seed": [float(x) for x in np.nanmean(a, axis=1)], "n_scenarios": int(a.shape[1]),
                  "n_runs": int(np.isfinite(a).sum())}

    def diff(x, ref, pair_seeds=False):  # paired x - ref, hierarchical bootstrap
        if x not in res or ref not in res:
            return None
        dd = St.hier_bootstrap_diff(A.mat(d, x, "departed"), A.mat(d, ref, "departed"), 4000,
                                    np.random.default_rng(12), pair_seeds=pair_seeds)
        return A.summary(dd, res[x]["departure"]["estimate"] - res[ref]["departure"]["estimate"])

    out = {"variants": res, "diffs": {}}
    for m in ("M-B", "M-L", "P-S"):
        ref = f"{m}_W{TUNED_W[m]}"
        for w in WINDOWS:
            if w != TUNED_W[m]:
                out["diffs"][f"{m}_W{w} - {ref}"] = diff(f"{m}_W{w}", ref)
    for m in ("M-B", "M-L"):
        ref = f"{m}_W{TUNED_W[m]}"
        for k in KS:
            if k != TUNED_K:
                out["diffs"][f"{m}_K{k} - {ref} (K={TUNED_K})"] = diff(f"{m}_K{k}", ref)
    out["diffs"]["P-S-act - P-S_W12"] = diff("P-S-act", "P-S_W12")
    out["diffs"] = {k: v for k, v in out["diffs"].items() if v is not None}
    # students of the same teacher seed are paired by seed (added 2026-11-13 after the first analysis showed the
    # independent-seed intervals to be conservative; reported next to them, not instead of them)
    out["diffs_seed_paired"] = {k: diff(k.split(" - ")[0], k.split(" - ")[1].split(" ")[0], pair_seeds=True)
                                for k in out["diffs"] if k.startswith("P-S")}
    write_md(out, out_md)
    write_fig(out, out_fig)
    if out_tex:
        write_tex(out, out_tex)
    return out


def label(name: str) -> str:
    """Human-readable variant name for the report."""
    if name == "P-S-act":
        return "P-S, action distillation"
    m, v = name.rsplit("_", 1)
    return f"{m}, W = {int(v[1:]) * 0.02:g} s" if v[0] == "W" else f"{m}, K = {int(v[1:]):,}"


def _fmt(e, k=3):
    return f"{e['estimate']:.{k}f} [{e['ci95'][0]:.{k}f}, {e['ci95'][1]:.{k}f}]"


def _pp(e):
    return f"{100 * e['estimate']:+.1f} pp [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]"


def write_md(out: dict, path: Path) -> None:
    res = out["variants"]
    n = next(iter(res.values()))["n_scenarios"] if res else 0
    L = ["# Exploratory E1-E3 (harness output of `scripts/exploratory.py`)", "",
         f"Pre-declared 500-scenario reactive-feasible test subset (N = {n} scenarios × 5 seeds), nominal truth. "
         "Departure rate with a hierarchical-bootstrap 95% CI; differences are paired against the tuned setting. "
         "Exploratory: outside the Holm family, nothing tuned on these results.", "",
         "| variant | departure [95% CI] | per seed |", "|---|---|---|"]
    for k, r in res.items():
        L.append(f"| {k} | {_fmt(r['departure'])} | {', '.join(f'{x:.3f}' for x in r['per_seed'])} |")
    L += ["", "| paired difference | departure [95% CI] |", "|---|---|"]
    L += [f"| {k} | {_pp(v)} |" for k, v in out["diffs"].items()]
    if out.get("diffs_seed_paired"):
        L += ["", "Student variants share their teacher's seed, so they are also paired by seed. Resampling seeds "
              "jointly (added after the first analysis; the independent-seed intervals above stay the reported "
              "default):", "", "| paired difference (seeds paired) | departure [95% CI] |", "|---|---|"]
        L += [f"| {k} | {_pp(v)} |" for k, v in out["diffs_seed_paired"].items()]
    path.write_text("\n".join(L) + "\n")


def write_fig(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = out["variants"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    col = {"M-B": "#4C72B0", "M-L": "#76B7B2", "P-S": "#59A14F"}
    ax = axs[0]
    for m in ("M-B", "M-L", "P-S"):
        pts = [(w * 0.02, res[f"{m}_W{w}"]["departure"]) for w in WINDOWS if f"{m}_W{w}" in res]
        if pts:
            x = [p[0] for p in pts]
            y = [p[1]["estimate"] for p in pts]
            lo = [p[1]["estimate"] - p[1]["ci95"][0] for p in pts]
            hi = [p[1]["ci95"][1] - p[1]["estimate"] for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=3, color=col[m], label=m)
    ax.set_xscale("log")
    ax.set_xticks([w * 0.02 for w in WINDOWS])
    ax.set_xticklabels([f"{w * 0.02:g}" for w in WINDOWS])
    ax.set_xlabel("history window W [s]")
    ax.set_ylabel("departure rate (95% CI)")
    ax.set_title("E1: history window")
    ax.legend(fontsize=8)
    ax = axs[1]
    for m in ("M-B", "M-L"):
        pts = [(k, res[alias(f"{m}_K{k}")]["departure"]) for k in KS if alias(f"{m}_K{k}") in res]
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1]["estimate"] for p in pts],
                        yerr=[[p[1]["estimate"] - p[1]["ci95"][0] for p in pts],
                              [p[1]["ci95"][1] - p[1]["estimate"] for p in pts]],
                        marker="o", capsize=3, color=col[m], label=m)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(KS))
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("MPPI samples K")
    ax.set_title("E2: sample budget")
    ax.legend(fontsize=8)
    ax = axs[2]
    for i, (n, lab) in enumerate((("P-S_W12", "latent regression\n(RMA, P-S)"), ("P-S-act", "action distillation\n(DAgger)"))):
        if n in res:
            ps = res[n]["per_seed"]
            ax.scatter([i] * len(ps), ps, color="#59A14F", alpha=0.6, s=18)
            e = res[n]["departure"]
            ax.errorbar([i + 0.15], [e["estimate"]], yerr=[[e["estimate"] - e["ci95"][0]], [e["ci95"][1] - e["estimate"]]],
                        marker="o", color="k", capsize=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["latent regression\n(RMA, P-S)", "action distillation\n(DAgger)"], fontsize=8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("E3: distillation target (dots = seeds)")
    fig.suptitle("Exploratory (500-scenario reactive-feasible test subset, 5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_tex(out: dict, path: Path) -> None:
    res, D = out["variants"], out["diffs"]
    n = next(iter(res.values()))["n_scenarios"] if res else 0
    parts = [f"\\paragraph{{Exploratory E1--E3.}} On the {n}-scenario reactive-feasible subset (5 seeds; paired "
             "differences against the tuned setting, 95\\% CIs; Figure~\\ref{fig:fig_exploratory}). "]
    e1 = [f"{label(k.split(' - ')[0])}: {_pp(v)}" for k, v in D.items() if "_W" in k.split(" - ")[0]]
    if e1:
        parts.append("E1, history windows other than the tuned one: " + "; ".join(e1).replace(" pp", "\\,pp") + ". ")
    e2 = [f"{label(k.split(' - ')[0])}: {_pp(v)}" for k, v in D.items() if "_K" in k.split(" - ")[0]]
    if e2:
        parts.append("E2, sample budgets other than $K = 1{,}024$: " + "; ".join(e2).replace(" pp", "\\,pp") + ". ")
    if "P-S-act - P-S_W12" in D and "P-S-act" in res:
        r = res["P-S-act"]
        parts.append(f"E3: DAgger action distillation departs {_fmt(r['departure'])} (per seed "
                     f"{', '.join(f'{x:.3f}' for x in r['per_seed'])}); difference to latent regression "
                     f"{_pp(D['P-S-act - P-S_W12'])}.".replace(" pp", "\\,pp"))
        sp = out.get("diffs_seed_paired", {})
        parts.append(" The student variants share their teachers' seeds, but the bootstrap above resamples seeds "
                     "independently per variant, so the student intervals are conservative. Seed by seed, the pattern "
                     "repeats: in every student variant the same teacher seed gives the worst student and the same two "
                     "give the best."
                     + (" Resampling seeds jointly (a pairing noted after the first analysis): "
                        + "; ".join(f"{label(k.split(' - ')[0])}: {_pp(v)}" for k, v in sp.items()).replace(" pp", "\\,pp")
                        + "." if sp else ""))
    txt = "".join(parts) + "\n\n\\begin{figure}[t]\\centering\\includegraphics[width=\\linewidth]{fig_exploratory.png}" \
        "\\caption{Exploratory E1--E3 on the reactive-feasible test subset.}\\label{fig:fig_exploratory}\\end{figure}\n"
    path.write_text(txt.replace("_", "\\_").replace("\\_exploratory", "_exploratory")
                    .replace("fig:fig\\_", "fig:fig_").replace("fig\\_exploratory.png", "fig_exploratory.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "analyze"])
    ap.add_argument("--variants", default="", help="comma-separated (eval); default all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tex", default="paper/gen/exploratory.tex")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.device)
    elif a.cmd == "eval":
        names = a.variants.split(",") if a.variants else list(variants())
        run_eval(names, a.device, a.batch)
    else:
        out = analyze(Path("docs/results/exploratory.md"), Path("docs/results/fig_exploratory.png"),
                      Path(a.tex) if a.tex else None)
        Path("docs/results/exploratory.json").write_text(json.dumps(out, indent=1))
        print(Path("docs/results/exploratory.md").read_text())


if __name__ == "__main__":
    main()
