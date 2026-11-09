"""G2 figures: feasibility-class composition of a labelled scenario set (labels from labelling/label.py).

Writes docs/figures/labels_<name>.png (class shares per post-drop surface x patch length, and per v_eff band x
geometry family) and docs/figures/labels_<name>.md (class counts with Wilson 95% intervals; harness output).

PYTHONPATH=. uv run python scripts/fig_labels.py splits/test_v1_T0.jsonl runs/label_test_T0/labels.jsonl --name test_T0
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.stats import wilson  # noqa: E402
from scenarios import sampler as S  # noqa: E402

CLASSES = ["reactive-feasible", "preview-required", "certified-infeasible", "undetermined"]
COLORS = {"reactive-feasible": "#4C9F70", "preview-required": "#E0A33A", "certified-infeasible": "#C4473A",
          "undetermined": "#9A9A9A"}


def bars(ax, groups, title):
    keys = list(groups)
    bottom = np.zeros(len(keys))
    for c in CLASSES:
        v = np.array([groups[k][c] / max(sum(groups[k].values()), 1) for k in keys])
        ax.bar(range(len(keys)), v, bottom=bottom, color=COLORS[c], label=c, width=0.8)
        bottom += v
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n(n={sum(groups[k].values())})" for k in keys], fontsize=7)
    ax.set_ylabel("share of scenarios")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("labels")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    _, scs = S.load_jsonl(a.split)
    sc = {s.sid: s for s in scs}
    labs = [json.loads(line) for line in open(a.labels)]
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    g1, g2 = {}, {}
    for lab in labs:
        s = sc[lab["sid"]]
        pl = "∞" if s.patch_len >= 1e8 else f"{int(s.patch_len)}"
        k1 = f"{s.surface} {pl} m"
        ve = lab["v_eff"]
        band = "≤12" if ve <= 12 else "12–17" if ve <= 17 else "17–22" if ve <= 22 else ">22"
        fam = "straight" if s.R == 0 else f"R{int(s.R)}"
        k2 = f"{fam} {band}"
        g1.setdefault(k1, Counter())[lab["label"]] += 1
        g2.setdefault(k2, Counter())[lab["label"]] += 1
    g1 = dict(sorted(g1.items(), key=lambda kv: (["wet", "snow", "ice"].index(kv[0].split()[0]), kv[0])))
    order = ["straight", "R50", "R100", "R200", "R400"]
    bands = ["≤12", "12–17", "17–22", ">22"]
    g2 = {k: g2[k] for k in sorted(g2, key=lambda k: (order.index(k.split()[0]), bands.index(k.split()[1])))}
    fig, axs = plt.subplots(2, 1, figsize=(13, 8))
    bars(axs[0], g1, "by post-drop surface and patch length")
    bars(axs[1], g2, "by geometry and effective speed v_eff [m/s]")
    axs[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    fig.suptitle(f"Feasibility classes, {a.name} (N = {len(labs)}; oracle labels, prereg A4)")
    fig.tight_layout()
    fig.savefig(out / f"labels_{a.name}.png", dpi=130)
    n = len(labs)
    cnt = Counter(lab["label"] for lab in labs)
    lines = [f"# Feasibility classes: {a.name}", "", f"Harness output (`scripts/fig_labels.py`), N = {n} scenarios.",
             "", "| class | n | share | Wilson 95% |", "|---|---|---|---|"]
    for c in CLASSES:
        lo, hi = wilson(cnt[c], n)
        lines.append(f"| {c} | {cnt[c]} | {cnt[c] / n:.3f} | [{lo:.3f}, {hi:.3f}] |")
    flags = {f: sum(bool(lab.get(f)) for lab in labs) for f in
             ("adaptation_required", "marginal", "certified_reactive_infeasible", "certified_nominal")}
    lines += ["", "| flag | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in flags.items()]
    alt = Counter(lab["label_or_only"] for lab in labs)
    lines += ["", "O-R-only rule (prereg A3, logged for comparison): " +
              ", ".join(f"{c} {alt[c]}" for c in CLASSES)]
    (out / f"labels_{a.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
