"""Generate the frozen scenario sets (prereg V7; hashes recorded in prereg amendment A2) into splits/.

  T0 = 4,000-scenario test pool; T1..T3 = the pre-declared top-up blocks (used only if reactive-feasible < 1,000);
  U  = 200-scenario tuning split.
Generating a set is not labelling or evaluating it. The committed files are the authority: tests/test_frozen.py
checks their SHA-256 against the amendment and that regenerating gives the same scenarios.

Run:  PYTHONPATH=. uv run python scripts/make_splits.py          (writes splits/*.jsonl, prints the hashes)
"""

from pathlib import Path

from scenarios import sampler as S

SETS = {
    "test_v1_T0.jsonl": lambda: S.test_pool(0),
    "test_v1_T1.jsonl": lambda: S.test_pool(1),
    "test_v1_T2.jsonl": lambda: S.test_pool(2),
    "test_v1_T3.jsonl": lambda: S.test_pool(3),
    "tuning_v1.jsonl": S.tuning_split,
}


def main():
    out = Path("splits")
    out.mkdir(exist_ok=True)
    for name, make in SETS.items():
        text = S.to_jsonl(make())
        (out / name).write_bytes(text.encode())
        print(f"{name}  {S.sha256(text)}")


if __name__ == "__main__":
    main()
