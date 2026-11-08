"""Label precedence (prereg A3), the proposed D20 variant, and the S1 check, on synthetic run records."""

import numpy as np

from control.episode import RunRecord
from labelling.label import assign_label, s1_check
from scenarios import sampler as S
from scenarios import track as T

TRACKS = T.all_tracks()
TR = TRACKS[1]
# 17 m/s: on the real 90-degree R50 track the certificate covers this apex drop (P1 test uses the same case)
CLASSIC = S.Scenario("c", 0, 15, "R50-apex", 50.0, "apex", "ice", 1e9, 0.0, 17.0, 0.95, 0.1, 1, TR.s_apex)
EASY = S.Scenario("e", 0, 10, "straight", 0.0, "straight", "wet", 30.0, 0.0, 10.0, 0.95, 0.7, 0, T.LEAD_IN)


def rec(method, ok, v=15.0, clear=0.5):
    return RunRecord("x", method, 0, ok, not ok, False, "", clear if ok else -1.0, 0.1, 0.0, 1.0, v, 0.0, 0.0, 9.0,
                     100.0)


def test_precedence():
    from sim.params import VehicleParams
    p = VehicleParams()
    assert assign_label(EASY, [rec("O-R", True)], [], TRACKS, p)["label"] == "reactive-feasible"
    assert assign_label(EASY, [rec("O-R", False)], [rec("O-C", True)], TRACKS, p)["label"] == "preview-required"
    assert assign_label(EASY, [rec("O-R", False)], [rec("O-C", False)], TRACKS, p)["label"] == "undetermined"
    lab = assign_label(CLASSIC, [rec("O-R", False)], [rec("O-C", False)], TRACKS, p)
    assert lab["label"] == "certified-infeasible" and lab["certified_nominal"]
    lab = assign_label(CLASSIC, [rec("O-R", False)], [rec("O-C", True, v=6.0)], TRACKS, p)
    assert lab["label"] == "preview-required" and lab["certified_reactive_infeasible"]
    # an O-R success always wins (and is then checked by S1 at its own arrival state)
    assert assign_label(CLASSIC, [rec("O-R", True, v=6.0)], [], TRACKS, p)["label"] == "reactive-feasible"


def test_marginal_flag_uses_best_or_seed():
    from sim.params import VehicleParams
    p = VehicleParams()
    lab = assign_label(EASY, [rec("O-R", True, clear=0.1), rec("O-R", True, clear=0.25)], [], TRACKS, p)
    assert lab["marginal"] and np.isclose(lab["or_best_clearance"], 0.25)
    lab = assign_label(EASY, [rec("O-R", True, clear=0.1), rec("O-R", True, clear=0.4)], [], TRACKS, p)
    assert not lab["marginal"]


def test_d20_label_counts_the_pd_witness():
    from sim.params import VehicleParams
    p = VehicleParams()
    lab = assign_label(EASY, [rec("O-R", False)], [rec("O-C", True)], TRACKS, p, pd_rec=rec("W-PD", True))
    assert lab["label"] == "preview-required" and lab["label_d20"] == "reactive-feasible"


def test_s1_flags_a_survivor_whose_arrival_state_is_certified():
    from sim.params import VehicleParams
    p = VehicleParams()
    assert s1_check(CLASSIC, rec("O-C", True, v=17.0), TRACKS, p) is not None  # impossible survivor -> bug
    assert s1_check(CLASSIC, rec("O-C", True, v=6.0), TRACKS, p) is None  # slowed down first -> fine
    assert s1_check(CLASSIC, rec("O-R", False, v=17.0), TRACKS, p) is None  # failures are never violations
