#!/bin/bash
# After the confirmatory test evaluation: check that every run file exists, then generate every number-bearing
# artefact from the harness output (results JSON/Markdown, figures, report fragments, README block, PDF).
# Usage: bash scripts/finalize.sh [--allow-partial]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=runs/eval_test
n() { { ls "$1"/*_s[0-4].$2 2>/dev/null || true; } | wc -l; }
missing=0
check() { local got; got=$(n "$1" "$3"); if [ "$got" -lt "$2" ]; then echo "INCOMPLETE $1: $got/$2"; missing=1; fi; }
check $ROOT/main 55 jsonl          # 11 methods x 5 seeds
check $ROOT/dry 55 jsonl
for d in $(PYTHONPATH=. uv run python -c "from eval.runner import h2_set_names; print(' '.join(h2_set_names()))"); do
    check $ROOT/$d 25 jsonl        # 5 H2 methods x 5 seeds
done
check $ROOT/trace 15 npz           # O-R, M-B, M-L x 5 seeds (exploratory)
if [ "$missing" = 1 ] && [ "${1:-}" != "--allow-partial" ]; then
    echo "evaluation incomplete; rerun the eval_split.py jobs (they resume) or pass --allow-partial"; exit 1
fi
PYTHONPATH=. uv run python scripts/make_figures.py --root $ROOT --labels labels/test_v1_T0.jsonl \
    --split splits/test_v1_T0.jsonl --out docs/results --tex paper/gen
PYTHONPATH=. uv run python scripts/compute_tex.py
PYTHONPATH=. uv run python scripts/readme_results.py
if [ "$(n runs/eval_v2/main jsonl)" -gt 0 ]; then   # follow-up (prereg A9; eval/followup.py)
    PYTHONPATH=. uv run python -m eval.followup > /dev/null
fi
if [ "$(n runs/eval_test/exploratory jsonl)" -gt 0 ]; then   # exploratory E1-E3 (scripts/exploratory.py)
    PYTHONPATH=. uv run python scripts/exploratory.py analyze > /dev/null
fi
(cd paper && tectonic main.tex)
echo "done: docs/results/results.md, docs/results/*.png, paper/gen/*.tex, README results block, paper/main.pdf"
