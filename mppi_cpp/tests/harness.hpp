// Test helpers: owned storage for an FcbMppiArgs on a synthetic track (straight or constant-radius arc).
#pragma once

#include <cmath>
#include <vector>

#include "fcb/capi.h"

namespace fcbtest {

inline fcb::Params default_params() {
    // sim/params.py VehicleParams() defaults (same values as the first line of tests/data/substep_cases.txt)
    fcb::Params p{};
    p.m = 1600.f; p.Iz = 2700.f; p.lf = 1.25f; p.lr = 1.45f; p.h = 0.55f; p.R_w = 0.33f; p.Jw = 2.0f; p.b_w = 0.f;
    p.foot_front = 2.2f; p.foot_rear = 2.6f; p.width = 1.85f; p.delta_max = 0.55f; p.delta_rate = 0.7f;
    p.tau_delta = 0.08f; p.tau_T = 0.1f; p.drive_force_max = 6278.4f; p.brake_force_max = 15696.f;
    p.brake_front_frac = 0.65f; p.drive_front_frac = 0.f; p.delay_steps = 1; p.B_scale_f = 0.8f;
    p.B_scale_r = 1.52f; p.f_rr = 0.012f; p.CdA = 0.7f; p.sigma_relax = 0.f; p.combined_alpha = 0.f;
    p.v_lo = 0.5f; p.v_hi = 2.0f; p.tau_kin = 0.05f; p.eps_v = 0.1f;
    return p;
}

inline fcb::Surf dry() { return fcb::Surf{{1.0f, 13.071f, 1.526f, 0.715f, 13.077f, 1.3f, -1.742f, 0.f}}; }
inline fcb::Surf ice() { return fcb::Surf{{0.1f, 150.150f, 1.611f, 1.000f, 28.557f, 1.6f, 0.f, 0.f}}; }

struct Problem {
    int n, K, H;
    std::vector<fcb::Params> mparams;
    std::vector<float> state, buf, frenet, sf, sr, prof, base, tx, ty, tth, tk, v_ref;
    std::vector<int32_t> hints, track_id, track_n, seeds;
    std::vector<float> U, V, costs, xend, w, info, Unew, action;
    fcb::FcbMppiArgs args{};

    // n cars at speed v on one track: radius R (0 = straight), lateral offsets e_i spread in [-0.5, 0.5] m.
    Problem(int n_, int K_, int H_, float v = 20.f, float R = 0.f, int npts = 4000)
        : n(n_), K(K_), H(H_) {
        const float ds = 0.25f;
        for (int j = 0; j < npts; ++j) {
            float s = j * ds;
            if (R == 0.f) {
                tx.push_back(s); ty.push_back(0.f); tth.push_back(0.f); tk.push_back(0.f);
            } else {
                float th = s / R;
                tx.push_back(R * std::sin(th)); ty.push_back(R * (1.f - std::cos(th)));
                tth.push_back(th); tk.push_back(1.f / R);
            }
        }
        track_n = {npts};
        fcb::Surf d = dry(), ic = ice();
        base.assign(d.v, d.v + 8);
        for (int i = 0; i < 3; ++i) base.insert(base.end(), ic.v, ic.v + 8);  // wet/snow slots unused here
        for (int e = 0; e < n; ++e) {
            fcb::Params p = default_params();
            mparams.push_back(p);
            float off = n > 1 ? -0.5f + e * (1.0f / (n - 1)) : 0.f;
            // place the car at s = 20 m, offset `off` to the left of the track
            int j = 80;
            float nx = -std::sin(tth[j]), ny = std::cos(tth[j]);
            float x[13] = {tx[j] + off * nx, ty[j] + off * ny, tth[j], v, 0.f, R == 0.f ? 0.f : v / R,
                           v / p.R_w, v / p.R_w, 0.f, 0.f, R == 0.f ? 0.f : (p.lf + p.lr) / R, 0.f, 0.f};
            state.insert(state.end(), x, x + 13);
            hints.insert(hints.end(), {j, j + 5, j - 6});
            frenet.insert(frenet.end(), {j * ds, off});
            sf.insert(sf.end(), d.v, d.v + 8);
            sr.insert(sr.end(), d.v, d.v + 8);
            float pr[8] = {60.f, 0.f, 50.f, 1.0f, 0.1f, 3.f, 0.f, 0.f};  // ice patch from s = 60 m, 50 m long
            prof.insert(prof.end(), pr, pr + 8);
            track_id.push_back(0);
            v_ref.push_back(v);
            seeds.push_back(1000 + 17 * e);
        }
        buf.assign(static_cast<size_t>(n) * fcb::BUF_LEN * 2, 0.f);
        U.assign(static_cast<size_t>(n) * H * 2, 0.f);
        Unew = U;
        V.assign(static_cast<size_t>(n) * K * H * 2, 0.f);
        costs.assign(static_cast<size_t>(n) * K, 0.f);
        w = costs;
        xend.assign(static_cast<size_t>(n) * K * 13, 0.f);
        info.assign(static_cast<size_t>(n) * 3, 0.f);
        action.assign(static_cast<size_t>(n) * 2, 0.f);
        fcb::FcbMppiArgs& a = args;
        a.n = n; a.K = K; a.H = H; a.nsub = fcb::N_SUBSTEPS; a.mode = fcb::MODE_FIXED; a.pd_sample = 1;
        a.n_iter = 1; a.t = 0; a.track_stride = npts; a.buf_len = fcb::BUF_LEN;
        a.sigma_steer = 0.05f; a.sigma_long = 0.3f; a.rho = 0.f; a.lam = 10.f; a.ess_target = 0.f;
        a.dt = fcb::CONTROL_DT / fcb::N_SUBSTEPS; a.ds = ds;
        a.cw = fcb::CostW{100.f, 1.f, 2.f, 0.1f, 0.1f, 2000.f, 1.f, 0.5235987756f, 1.5707963268f};
        bind();
    }
    void bind() {
        fcb::FcbMppiArgs& a = args;
        a.mparams = mparams.data(); a.state = state.data(); a.buf = buf.data(); a.hints = hints.data();
        a.frenet = frenet.data(); a.sf_fix = sf.data(); a.sr_fix = sr.data(); a.prof = prof.data();
        a.base = base.data(); a.track_id = track_id.data(); a.track_x = tx.data(); a.track_y = ty.data();
        a.track_th = tth.data(); a.track_k = tk.data(); a.track_n = track_n.data(); a.v_ref = v_ref.data();
        a.seeds = seeds.data(); a.U = U.data(); a.V = V.data(); a.costs = costs.data(); a.xend = xend.data();
        a.w = w.data(); a.info = info.data(); a.Unew = Unew.data(); a.action = action.data();
    }
};

}  // namespace fcbtest
