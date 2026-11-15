"""MARWIS summary on synthetic logs with known values (scripts/report/wspi_marwis.py)."""

import numpy as np

from scripts.report import wspi_marwis as W
from sim.params import MU_RANGES


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("timestamp_local,dew_point,ambient_temp,relative_humidity,surface_temp,road_condition,friction,"
                "ice_percent,water_film_height\n")
        for i, (cls, mu) in enumerate(rows):
            f.write(f"2026-01-28 10:{i // 60:02d}:{i % 60:02d}.000,0,0,0,0,{cls},{mu},0,0\n")


def test_mu_ranges_match_the_simulator():
    assert W.MU_RANGES == MU_RANGES


def test_ratios_and_counts(tmp_path):
    root = tmp_path / "raw"
    _write(root / "mcity_wspi/csv/d1/marwis_data.csv", [("dry", 0.82)] * 10 + [("snow covered", 0.246)] * 20)
    _write(root / "mcity_wspi/csv/d2/marwis_data.csv", [("snow covered", 0.246)] * 20 + [("snow/ice covered", 0.164)] * 5)
    res = W.summarize(root)
    assert res["classes"]["snow"]["n_samples"] == 40 and res["classes"]["snow"]["n_drives"] == 2
    assert abs(res["classes"]["snow"]["ratio_to_dry_median"][0] - 0.3) < 1e-9
    assert abs(res["classes"]["snow/ice"]["ratio_to_dry_median"][0] - 0.2) < 1e-9
    assert res["log_interval_s_median"] == 1.0
    W.write_md(res, tmp_path / "m.md")
    W.write_tex(res, tmp_path / "m.tex")
    assert "0.30" in (tmp_path / "m.tex").read_text()
