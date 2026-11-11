"""Render a side-by-side top-down GIF of two controllers on one *tuning* scenario (never a test scenario: prereg
rule 4), with each MPPI's sampled rollouts drawn as it plans. For the README / project page; carries no numbers.

The fan shows what the planner *believes*: its samples are re-simulated from the true state with the planner's own
model (its assumed per-axle surfaces, its delay), exactly as its rollouts do. A friction-blind planner's fan curves
nicely through the bend on ice; the car then does not follow it.

PYTHONPATH=. uv run python scripts/render_episode.py --sid U2-00074 --left M-N --right O-R --out docs/media/hero.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from eval import runner as R  # noqa: E402
from eval.methods import handover, tuning_sets  # noqa: E402
from scenarios import track as T  # noqa: E402
from sim.params import CONTROL_DT, VehicleParams  # noqa: E402
from sim.warp_sim import ScenarioSim, WarpSim  # noqa: E402

TITLES = {"M-N": "MPPI assuming dry road", "O-R": "MPPI told the current friction",
          "M-B": "MPPI with a model bank (adaptive)", "M-L": "MPPI with a learned friction estimator",
          "P-S": "RL student (history only)", "O-C": "MPPI that sees the friction ahead"}
N_FAN = 48


def run(method: str, sc, snap, tracks, p, device, t_before=1.0, t_after=8.0, stride=3):
    sim = ScenarioSim([sc], p, tracks, device)
    sim.load(snap)
    ctrl = R.factory(method, 0, p)(sim, [sc], snap, 0)
    mp = getattr(ctrl, "mppi", None)  # MPPI controllers (mppi_controller, BankMPPI, LearnedMPPI) expose it
    frames, k, t_drop = [], 0, None
    rng = np.random.default_rng(0)
    while True:
        acts = ctrl(sim, k)
        st = sim.get_state()[0]
        fr = sim.get_frenet()[0]
        if t_drop is None and fr[0] + p.lf >= sc.s_drop:
            t_drop = k * CONTROL_DT
        t = k * CONTROL_DT
        show = (t_drop is None and sc.s_drop - fr[0] < 12.0 + st[3] * t_before) or (t_drop is not None)
        if show and k % stride == 0:
            fan = replay_fan(mp, sim, p, rng, device) if mp is not None else None
            frames.append({"state": st.copy(), "fan": fan, "e": fr[1]})
        sim.step(acts)
        k += 1
        if t_drop is not None and (t - t_drop > t_after or abs(sim.get_frenet()[0][1]) > 3.5):
            break
        if k > 3000:
            break
    return frames


def replay_fan(m, sim, p, rng, device):
    """Positions along N_FAN of the planner's samples (highest weights + random), simulated with its model."""
    w = m.w.numpy()[0]
    V = m.V.numpy()[0]
    top = np.argsort(-w)[: N_FAN // 2]
    rest = rng.choice(np.setdiff1d(np.arange(len(w)), top), N_FAN - len(top), replace=False)
    idx = np.concatenate([top, rest])
    mp = VehicleParams()  # planner model params (nominal) for every method rendered here
    ws = WarpSim(len(idx), mp, device)
    ws.reset(np.repeat(sim.get_state()[:1].astype(np.float32), len(idx), 0))
    ws.buf.assign(wp.array(np.repeat(sim.buf.numpy()[:1], len(idx), 0), dtype=wp.float32, device=device))
    ws.set_surfaces(np.repeat(m.sf_fix.numpy().reshape(-1, 8)[:1], len(idx), 0),
                    np.repeat(m.sr_fix.numpy().reshape(-1, 8)[:1], len(idx), 0))
    H = V.shape[1]  # V: (K, H, 2)
    xy = np.zeros((len(idx), H, 2))
    for h in range(H):
        ws.step(np.ascontiguousarray(V[idx, h, :]))
        xy[:, h] = ws.get_state()[:, :2]
    return {"xy": xy, "w": w[idx] / max(w.max(), 1e-12)}


def road_polys(tr, sc):
    nx, ny = -np.sin(tr.theta), np.cos(tr.theta)
    hw = T.LANE_WIDTH / 2
    left = np.stack([tr.x + hw * nx, tr.y + hw * ny], 1)
    right = np.stack([tr.x - hw * nx, tr.y - hw * ny], 1)
    s_end = sc.s_drop + sc.transition + min(sc.patch_len, 1e4)
    sel = (tr.s >= sc.s_drop) & (tr.s <= s_end)
    road = np.concatenate([left, right[::-1]])
    patch = np.concatenate([left[sel], right[sel][::-1]]) if sel.any() else None
    return road, left, right, patch


def car_poly(st, p):
    X, Y, psi = st[0], st[1], st[2]
    c, s = np.cos(psi), np.sin(psi)
    hw = p.width / 2
    pts = np.array([[p.foot_front, hw], [p.foot_front, -hw], [-p.foot_rear, -hw], [-p.foot_rear, hw]])
    return np.stack([X + pts[:, 0] * c - pts[:, 1] * s, Y + pts[:, 0] * s + pts[:, 1] * c], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="U2-00074")
    ap.add_argument("--left", default="M-N")
    ap.add_argument("--right", default="O-R")
    ap.add_argument("--out", default="docs/media/hero.gif")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()
    scs, _ = tuning_sets()
    sc = next(s for s in scs if s.sid == a.sid)  # tuning split only
    tracks = T.all_tracks()
    tr = tracks[sc.track_id]
    p = VehicleParams()
    snap = handover([sc], tracks, p, a.device)
    runs = {m: run(m, sc, snap, tracks, p, a.device) for m in (a.left, a.right)}
    n = max(len(v) for v in runs.values())
    road, left, right, patch = road_polys(tr, sc)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=80)
    fig.patch.set_facecolor("white")
    arts = []
    for ax, m in zip(axs, (a.left, a.right)):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Polygon(road, closed=True, fc="#d9d9d9", ec="none", zorder=0))
        if patch is not None:
            ax.add_patch(Polygon(patch, closed=True, fc="#9fd3f5", ec="none", alpha=0.9, zorder=1))
        ax.plot(left[:, 0], left[:, 1], color="white", lw=2, zorder=2)
        ax.plot(right[:, 0], right[:, 1], color="white", lw=2, zorder=2)
        ax.plot(tr.x, tr.y, color="white", lw=1, ls=(0, (4, 6)), zorder=2)
        ax.set_title(TITLES.get(m, m), fontsize=12)
        fan_lines = [ax.plot([], [], color="#e8711a", lw=0.8, alpha=0.0, zorder=3)[0] for _ in range(N_FAN)]
        trail, = ax.plot([], [], color="#333333", lw=1.2, zorder=4)
        car = Polygon(np.zeros((4, 2)), closed=True, fc="#1f3b73", ec="black", lw=0.8, zorder=5)
        ax.add_patch(car)
        arts.append((ax, m, fan_lines, trail, car))
    fig.text(0.5, 0.03, "Tuning scenario: the road turns to ice (blue) at the curve entry. Orange = sampled plans, "
             "simulated with each planner's own friction model.", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    def draw(i):
        out = []
        for ax, m, fan_lines, trail, car in arts:
            fr = runs[m][min(i, len(runs[m]) - 1)]
            st = fr["state"]
            hist = np.array([f["state"][:2] for f in runs[m][: min(i, len(runs[m]) - 1) + 1]])
            trail.set_data(hist[:, 0], hist[:, 1])
            car.set_xy(car_poly(st, p))
            fan = fr["fan"]
            for j, ln in enumerate(fan_lines):
                if fan is None:
                    ln.set_alpha(0.0)
                    continue
                ln.set_data(fan["xy"][j, :, 0], fan["xy"][j, :, 1])
                ln.set_alpha(float(0.12 + 0.75 * fan["w"][j]))
            ax.set_xlim(st[0] - 22, st[0] + 34)
            ax.set_ylim(st[1] - 16, st[1] + 16)
            out += [trail, car, *fan_lines]
        return out

    anim = animation.FuncAnimation(fig, draw, frames=n, blit=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(a.out, writer=animation.PillowWriter(fps=a.fps))
    print("wrote", a.out, Path(a.out).stat().st_size // 1024, "KB,", n, "frames")


if __name__ == "__main__":
    main()
