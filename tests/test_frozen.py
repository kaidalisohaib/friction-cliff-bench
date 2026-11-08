"""G1 freezes (prereg amendments A1, A2). A failure here means a frozen value changed: that is only allowed by a
new dated prereg amendment that records the new hash, never by editing these constants silently."""

import hashlib
import json
from pathlib import Path

import pytest

from scenarios import sampler as S
from sim.params import fingerprint

# prereg amendment A1 (2026-11-08): vehicle parameters, surface tables, mu ranges, numerical constants
FROZEN_PARAMS_SHA = "b1e0ddceb9246f5b682c475def8a2bf9e178a4d48a0e7e91432ba89716d8cc21"

# prereg amendment A2 (2026-11-08): SHA-256 of the committed scenario files
FROZEN_SPLITS = {
    "test_v1_T0.jsonl": "e742af7f058c322c461fb929d6b1d23f4a2d99aa9d8d0fee3e10c782f51a82fc",
    "test_v1_T1.jsonl": "bb003d36c03235e61763a12510a4cc0b1fe2bcf482fe811a6227989de7a379bb",
    "test_v1_T2.jsonl": "9deb033a547ae50c824a1d340046e988056b80c356cbcb361d26f75c39879cb6",
    "test_v1_T3.jsonl": "2f1c4f4693216af3f2aa37a58b660c164b08757104a554c5c6acfaf351ba612e",
    "tuning_v1.jsonl": "5dd1e7e7d8038c1ceb2aa53423f8e1f70946d72b21df293f505afa5cd0b14a63",
}
MAKERS = {"test_v1_T0.jsonl": lambda: S.test_pool(0), "test_v1_T1.jsonl": lambda: S.test_pool(1),
          "test_v1_T2.jsonl": lambda: S.test_pool(2), "test_v1_T3.jsonl": lambda: S.test_pool(3),
          "tuning_v1.jsonl": S.tuning_split}
SPLITS = Path(__file__).resolve().parents[1] / "splits"


def test_parameters_match_the_g1_freeze():
    assert fingerprint() == FROZEN_PARAMS_SHA


@pytest.mark.parametrize("name", sorted(FROZEN_SPLITS))
def test_committed_split_file_matches_recorded_hash(name):
    assert hashlib.sha256((SPLITS / name).read_bytes()).hexdigest() == FROZEN_SPLITS[name]


@pytest.mark.parametrize("name", sorted(FROZEN_SPLITS))
def test_regenerating_gives_the_same_scenarios(name):
    """Compares scenario lines only: the header records the numpy version, which may legitimately change.
    If this fails after a dependency update, the RNG stream changed: keep using the committed files."""
    committed = (SPLITS / name).read_text().splitlines()[1:]
    regenerated = S.to_jsonl(MAKERS[name]()).splitlines()[1:]
    assert committed == regenerated


def test_tuning_and_test_sets_share_no_instance():
    def keys(name):
        rows = [json.loads(line) for line in (SPLITS / name).read_text().splitlines()[1:]]
        return {(r["cell"], r["v0"], r["mu0"], r["mu1"]) for r in rows}
    tune = keys("tuning_v1.jsonl")
    for name in FROZEN_SPLITS:
        if name.startswith("test"):
            assert not (tune & keys(name)), name
