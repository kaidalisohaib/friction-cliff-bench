"""Write the GoogleTest fixtures from the Python side (run from the repo root):

  PYTHONPATH=. uv run python mppi_cpp/tests/make_fixtures.py

- substep_cases.txt: the 7 V8 maneuvers of tests/test_warp_sim.py run through the float64 NumPy reference
  (sim/reference.py). Per control step: the applied (delayed) action, the scaled per-axle surfaces, and the
  reference state after the step. The C++ dynamics must track them within the V8 tolerances.
- warp_randn.txt: normals drawn by Warp's wp.randn for a few (seed, offset) pairs; the C++ RNG must reproduce them.
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
import warp as wp

from sim.params import VehicleParams
from sim.reference import ReferenceSim, scaled_surfaces
from sim.warp_sim import surface_vec
from tests.test_warp_sim import maneuvers

OUT = Path(__file__).parent / "data"


def write_substep_cases(path: Path) -> None:
    lines = []
    for name, p, x0, actions, surfaces in maneuvers():
        ref = ReferenceSim(p, x0)
        lines.append(f"case {name} {len(actions)}")
        lines.append("params " + " ".join(repr(float(getattr(p, f.name))) for f in fields(VehicleParams)))
        lines.append("x0 " + " ".join(repr(float(v)) for v in x0))
        for a, (sf, sr) in zip(actions, surfaces):
            # the action applied this step: buf[delay] after the shift = buf[delay - 1] before it
            u = np.asarray(a, float) if p.delay_steps == 0 else ref.buf[p.delay_steps - 1].copy()
            sfs, srs = scaled_surfaces(sf, sr, p)
            ref.step(a, sf, sr)
            vals = [*u, *surface_vec(sfs), *surface_vec(srs), *ref.x]
            lines.append("step " + " ".join(repr(float(v)) for v in vals))
    path.write_text("\n".join(lines) + "\n")


@wp.kernel
def _randn_kernel(seeds: wp.array(dtype=int), offs: wp.array(dtype=int), out: wp.array2d(dtype=float)):
    i = wp.tid()
    st = wp.rand_init(seeds[i], offs[i])
    for j in range(out.shape[1]):
        out[i, j] = wp.randn(st)


def write_randn(path: Path) -> None:
    pairs = [(0, 0), (1, 0), (12345, 7), (2**29 + 11, 1023), (-5, 3)]
    seeds = wp.array(np.array([s for s, _ in pairs], np.int32), dtype=int, device="cpu")
    offs = wp.array(np.array([o for _, o in pairs], np.int32), dtype=int, device="cpu")
    out = wp.zeros((len(pairs), 6), dtype=float, device="cpu")
    wp.launch(_randn_kernel, dim=len(pairs), inputs=[seeds, offs, out], device="cpu")
    o = out.numpy()
    path.write_text("".join(f"{s} {off} " + " ".join(repr(float(v)) for v in o[i]) + "\n"
                            for i, (s, off) in enumerate(pairs)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write_substep_cases(OUT / "substep_cases.txt")
    write_randn(OUT / "warp_randn.txt")
    print("fixtures written to", OUT)
