"""Generate the frozen scenario sets (prereg V7; hashes recorded in prereg amendment A2) into splits/.

  T0 = 4,000-scenario test pool; T1..T3 = the pre-declared top-up blocks (used only if reactive-feasible < 1,000);
  U  = 200-scenario tuning split.
Generating a set is not labelling or evaluating it. The committed files are the authority: tests/test_frozen.py
checks their SHA-256 against the amendment and that regenerating gives the same scenarios.

Run:  PYTHONPATH=. uv run python scripts/setup/make_splits.py          (writes splits/*.jsonl, prints the hashes)
"""

from pathlib import Path

from scenarios import sampler as S

SETS = {
    "test_v1_T0.jsonl": lambda: S.test_pool(0),
    "test_v1_T1.jsonl": lambda: S.test_pool(1),
    "test_v1_T2.jsonl": lambda: S.test_pool(2),
    "test_v1_T3.jsonl": lambda: S.test_pool(3),
    "tuning_v1.jsonl": S.tuning_split,
    "tuning_v1_b2.jsonl": S.tuning_block2,  # prereg A4.7
    "dev_v1.jsonl": S.dev_split,  # RL follow-up development split (D35); not frozen by A2
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
