import json

import numpy as np

from eval import followup as F


def _write(root, set_name, m, seeds, dep_p, prog, n=300, rng=None):
    d = root / set_name
    d.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        p = dep_p[s] if isinstance(dep_p, (list, tuple)) else dep_p
        with open(d / f"{m}_s{s}.jsonl", "w") as f:
            for j in range(n):
                f.write(json.dumps({"sid": f"T1-{j:05d}", "departed": bool(rng.random() < p),
                                    "progress": float(prog * (1 + 0.01 * rng.standard_normal()))}) + "\n")


def _fixture(tmp_path, psp2=0.01, mbc=0.01, prog_psp2=86.0, prog_mbc=86.0):
    rng = np.random.default_rng(0)
    spec = {"O-R": (0.04, 100.0), "M-B": (0.03, 96.0), "M-Bc": (mbc, prog_mbc), "P-S": ([0.0, 0.03, 0.15, 0.05, 0.7, 0.1, 0.0, 0.0, 0.02, 0.0], 80.0),
            "P-SP": (0.012, 87.0), "P-S2": (0.03, 82.0), "P-SP2": (psp2, prog_psp2)}
    for m, (p, prog) in spec.items():
        for st in ("main", "dry"):
            _write(tmp_path, st, m, F.SEEDS[m], p if st == "main" else 0.0, prog, rng=rng)
    return tmp_path


def test_h5_seed_gate_and_exploratory_contrasts(tmp_path):
    res = F.analyze(_fixture(tmp_path), top5=[0, 1, 2, 3, 4], n_boot=2000)
    g = res["H5"]["gate"]
    assert g["P-SP2"]["n_pass"] == 10 and res["H5"]["holds"]
    assert g["P-S"]["n_pass"] < 10  # the 0.7 and 0.15 seeds fail O-R + 5 pp
    assert {"P-S_minus_P-S2", "P-S_minus_P-SP2", "P-SP_minus_P-SP2", "P-SP2_top5",
            "post_hoc_P-SP2_top5_minus_MBc_pace_tax"} <= set(res["exploratory"])
    assert abs(res["exploratory"]["post_hoc_P-SP2_top5_minus_MBc_pace_tax"]["estimate"]) < 0.01  # same pace in fixture
    assert len(res["methods"]["P-SP2"]["per_seed"]) == 10
    md = F.markdown(res)
    assert "H4" in md and "H5" in md
    tx = F.tex(res)
    assert "Test pool v2" in tx and "H4" in tx and "\\\\" in tx and "\\%" in tx


def test_h4_non_inferior_at_matched_pace_and_not_when_pace_differs(tmp_path):
    res = F.analyze(_fixture(tmp_path / "a"), n_boot=2000)
    assert res["H4"]["pace_matched"] and res["H4"]["non_inferior"] and not res["H4"]["superior"]
    res = F.analyze(_fixture(tmp_path / "b", prog_psp2=80.0, prog_mbc=86.0), n_boot=2000)
    assert not res["H4"]["pace_matched"] and not res["H4"]["non_inferior"]  # 6 pp pace gap: not interpreted
    res = F.analyze(_fixture(tmp_path / "c", psp2=0.06, mbc=0.01), n_boot=2000)
    assert res["H4"]["pace_matched"] and not res["H4"]["non_inferior"]


def test_pace_tax_is_relative_to_O_R(tmp_path):
    res = F.analyze(_fixture(tmp_path), n_boot=1000)
    assert abs(res["methods"]["M-B"]["dry_pace_tax"]["estimate"] - 0.04) < 0.005


def test_abstract_and_readme_fragments_follow_the_verdicts(tmp_path):
    res = F.analyze(_fixture(tmp_path / "a"), n_boot=1000)
    ab = F.abstract_tex(res)
    assert "non-inferior" in ab and "all 10 training seeds" in ab and "\\,pp" in ab and " pp" not in ab
    rd = "\n".join(F.readme_lines(res))
    assert "| P-SP2 | 10 |" in rd and "**H4** (" in rd and "**H5** (" in rd
    res = F.analyze(_fixture(tmp_path / "b", prog_psp2=80.0, prog_mbc=86.0), n_boot=1000)
    assert "not interpreted" in F.abstract_tex(res)
    assert F.abstract_tex({"methods": {}}) == "" and F.readme_lines({"methods": {}}) == []


def test_figure_renders(tmp_path):
    res = F.analyze(_fixture(tmp_path / "f"), n_boot=500)
    F.figure(res, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 10_000
