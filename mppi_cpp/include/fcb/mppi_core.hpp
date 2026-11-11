// MPPI building blocks shared by the CUDA kernels and the CPU reference (port of control/mppi.py).
// One function per Warp kernel body; the drivers (src/) only map them onto threads or loops.
#pragma once

#include "common.hpp"
#include "rng.hpp"
#include "track.hpp"
#include "vehicle.hpp"

namespace fcb {

constexpr int MODE_FIXED = 0;
constexpr int MODE_FIELD = 1;

// Plain-old-data argument block of the C API (mirrored by a ctypes.Structure in control/mppi_cpp.py). All arrays
// are row-major float32/int32 with the shapes of the Warp arrays in control/mppi.py and sim/warp_sim.py.
struct FcbMppiArgs {
    int32_t n, K, H, nsub, mode, pd_sample, n_iter, t, track_stride, buf_len;
    float sigma_steer, sigma_long, rho, lam, ess_target, dt, ds;
    CostW cw;
    const Params* mparams;   // (n)            model parameters per car
    const float* state;      // (n, 13)        true current state
    const float* buf;        // (n, buf_len, 2) action delay buffer, index j = action from j steps ago
    const int32_t* hints;    // (n, 3)         projection hints (centre, front axle, rear axle)
    const float* frenet;     // (n, 2)         current (s, e)
    const float* sf_fix;     // (n, 8)         MODE_FIXED surfaces (unscaled)
    const float* sr_fix;     // (n, 8)
    const float* prof;       // (n, 8)         friction-field profile per car (MODE_FIELD)
    const float* base;       // (4, 8)         surface shapes dry/wet/snow/ice
    const int32_t* track_id; // (n)
    const float* track_x;    // (n_tracks, track_stride)
    const float* track_y;
    const float* track_th;
    const float* track_k;
    const int32_t* track_n;  // (n_tracks)
    const float* v_ref;      // (n)
    const int32_t* seeds;    // (n)
    float* U;                // (n, H, 2)      plan (in: warm start; out: shifted plan)
    float* V;                // (n, K, H, 2)   sampled controls
    float* costs;            // (n, K)
    float* xend;             // (n, K, 13)     rollout end states
    float* w;                // (n, K)         normalized weights
    float* info;             // (n, 3)         (min cost, effective sample size, lambda)
    float* Unew;             // (n, H, 2)
    float* action;           // (n, 2)
};

FCB_HD Tracks tracks_of(const FcbMppiArgs& a) {
    return Tracks{a.track_x, a.track_y, a.track_th, a.track_k, a.track_n, a.track_stride, a.ds};
}
FCB_HD const Surf& surf_at(const float* p, int i) { return reinterpret_cast<const Surf*>(p)[i]; }
FCB_HD const State& state_at(const float* p, int i) { return reinterpret_cast<const State*>(p)[i]; }

FCB_HD float wrap(float a) { return atan2f(sinf(a), cosf(a)); }

struct F2c {
    float c, term;
};

// (running cost, terminated flag) of one control step; footprint in the small-angle Frenet approximation.
FCB_HD F2c step_cost(const State& xs, float s, float e, int idx, int tid, float s_prev, const Tracks& T,
                     const Params& p, float v_ref, float du2, const CostW& cw) {
    const float* x = xs.v;
    float epsi = wrap(x[2] - T.TH(tid, idx));
    float k = T.K(tid, idx);
    float sp = sinf(epsi), cp = cosf(epsi);
    float worst = 0.0f;
    for (int a = 0; a < 2; ++a) {
        float lon = a == 1 ? -p.foot_rear : p.foot_front;
        for (int b = 0; b < 2; ++b) {
            float lat = 0.5f * p.width;
            if (b == 1) lat = -lat;
            float ec = e + lon * sp + lat * cp - 0.5f * k * lon * lon;
            worst = fmax_(worst, fabsf(ec));
        }
    }
    float viol = fmax_(worst - HALF_LANE, 0.0f);
    float dv = (x[3] - v_ref) / v_ref;
    float c = cw.w_lane * viol * viol + cw.w_ey * e * e + cw.w_v * dv * dv + cw.w_du * du2 - cw.w_prog * (s - s_prev);
    float term = 0.0f;
    float speed = sqrtf(x[3] * x[3] + x[4] * x[4]);
    float beta = 0.0f;
    if (speed > 1.0f) beta = atan2f(x[4], fabsf(x[3]));
    if (viol > cw.severe || fabsf(beta) > cw.beta_term || fabsf(epsi) > cw.spin_term) {
        c = c + cw.C_fail;
        term = 1.0f;
    }
    return {c, term};
}

// Closed-loop path follower (control/episode.py prefix_action): curvature feedforward + PD + yaw-rate damping.
FCB_HD float pd_steer(const State& xs, F3 q, int tid, const Tracks& T, const Params& p) {
    const float* x = xs.v;
    int idx = static_cast<int>(q.idx);
    float k = T.K(tid, idx);
    float epsi = wrap(x[2] - T.TH(tid, idx));
    float delta = (p.lf + p.lr) * k - 0.15f * q.e - 0.8f * epsi - 0.1f * (x[5] - x[3] * k);
    return clampf(delta / p.delta_max, -1.0f, 1.0f);
}

FCB_HD float pd_long(const State& xs, F3 q, int tid, const Tracks& T, int npts, float v_ref, float mu, int mode,
                     const Surf& pr, const Surf* base) {
    const float* x = xs.v;
    const float g = 9.81f;
    int i0 = static_cast<int>(q.idx);
    int ia = clampi(i0 + static_cast<int>(fmax_(x[3], 1.0f) * 0.5f / T.ds), 0, npts - 1);
    float ka = fmax_(fabsf(T.K(tid, ia)), 1.0e-9f);
    float vt = fmin_(v_ref, sqrtf(0.7f * mu * g / ka));
    if (mode == MODE_FIELD) {
        float ab = 0.5f * mu * g;
        for (int j = 0; j < 40; ++j) {
            float dsj = static_cast<float>(j) * 2.0f;
            int ij = clampi(i0 + static_cast<int>(dsj / T.ds), 0, npts - 1);
            float mu_j = field_surface(q.s + dsj, pr, base).v[0];
            float kj = fmax_(fabsf(T.K(tid, ij)), 1.0e-9f);
            float va = sqrtf(0.7f * mu_j * g / kj);
            vt = fmin_(vt, sqrtf(va * va + 2.0f * ab * dsj));
        }
    }
    return clampf(0.3f * (vt - x[3]), -1.0f, 1.0f);
}

// --- per-thread bodies ----------------------------------------------------------------------------------------
FCB_HD size_t vidx(const FcbMppiArgs& a, int e, int k, int h) {
    return ((static_cast<size_t>(e) * a.K + k) * a.H + h) * 2;
}

// AR(1) noise along the horizon around U; sample k = 0 is noise-free. t_rng = (control step) * 16 + iteration.
FCB_HD void sample_one(const FcbMppiArgs& a, int e, int k, int t_rng) {
    int32_t seed = static_cast<int32_t>(static_cast<uint32_t>(a.seeds[e]) + static_cast<uint32_t>(t_rng) * 7919u);
    uint32_t st = rand_init(seed, k);
    float q = sqrtf(1.0f - a.rho * a.rho);
    float n0 = randn(st);
    float n1 = randn(st);
    for (int h = 0; h < a.H; ++h) {
        if (h > 0) {
            n0 = a.rho * n0 + q * randn(st);
            n1 = a.rho * n1 + q * randn(st);
        }
        float a0 = a.sigma_steer * n0;
        float a1 = a.sigma_long * n1;
        if (k == 0) {
            a0 = 0.0f;
            a1 = 0.0f;
        }
        size_t i = vidx(a, e, k, h);
        a.V[i] = clampf(a.U[(e * a.H + h) * 2] + a0, -1.0f, 1.0f);
        a.V[i + 1] = clampf(a.U[(e * a.H + h) * 2 + 1] + a1, -1.0f, 1.0f);
    }
}

FCB_HD void rollout_one(const FcbMppiArgs& a, int e, int k) {
    const Tracks T = tracks_of(a);
    const Surf* base = reinterpret_cast<const Surf*>(a.base);
    const Params& p = a.mparams[e];
    State x = state_at(a.state, e);
    int tid = a.track_id[e];
    int npts = a.track_n[tid];
    const Surf& pr = surf_at(a.prof, e);
    int hc = a.hints[e * 3 + 0], hf = a.hints[e * 3 + 1], hr = a.hints[e * 3 + 2];
    float s_prev = a.frenet[e * 2 + 0];
    Surf sf = surf_at(a.sf_fix, e);
    Surf sr = surf_at(a.sr_fix, e);
    sf.v[4] = sf.v[4] * p.B_scale_f;
    sr.v[4] = sr.v[4] * p.B_scale_r;
    int d = p.delay_steps;
    const float* buf = a.buf + static_cast<size_t>(e) * a.buf_len * 2;
    float u_prev_s = buf[0], u_prev_l = buf[1];
    float total = 0.0f;
    bool is_pd = a.pd_sample == 1 && k == 1;
    F3 q{s_prev, a.frenet[e * 2 + 1], static_cast<float>(hc)};
    for (int h = 0; h < a.H; ++h) {
        size_t vi = vidx(a, e, k, h);
        if (is_pd) {
            a.V[vi] = pd_steer(x, q, tid, T, p);
            a.V[vi + 1] = pd_long(x, q, tid, T, npts, a.v_ref[e], sf.v[0], a.mode, pr, base);
        }
        float us, ul;
        if (h >= d) {
            size_t vd = vidx(a, e, k, h - d);
            us = a.V[vd];
            ul = a.V[vd + 1];
        } else {
            us = buf[(d - 1 - h) * 2];
            ul = buf[(d - 1 - h) * 2 + 1];
        }
        for (int j = 0; j < a.nsub; ++j) {
            if (a.mode == MODE_FIELD) {
                float c = cosf(x.v[2]), s_ = sinf(x.v[2]);
                F3 qf = project_local(T, tid, npts, x.v[0] + p.lf * c, x.v[1] + p.lf * s_, hf);
                F3 qr = project_local(T, tid, npts, x.v[0] - p.lr * c, x.v[1] - p.lr * s_, hr);
                hf = static_cast<int>(qf.idx);
                hr = static_cast<int>(qr.idx);
                sf = field_surface(qf.s, pr, base);
                sr = field_surface(qr.s, pr, base);
                sf.v[4] = sf.v[4] * p.B_scale_f;
                sr.v[4] = sr.v[4] * p.B_scale_r;
            }
            x = substep(x, us, ul, p, sf, sr, a.dt);
        }
        q = project_local(T, tid, npts, x.v[0], x.v[1], hc);
        hc = static_cast<int>(q.idx);
        float vs = a.V[vi], vl = a.V[vi + 1];
        float du2 = (vs - u_prev_s) * (vs - u_prev_s) + (vl - u_prev_l) * (vl - u_prev_l);
        u_prev_s = vs;
        u_prev_l = vl;
        F2c ct = step_cost(x, q.s, q.e, hc, tid, s_prev, T, p, a.v_ref[e], du2, a.cw);
        s_prev = q.s;
        total = total + ct.c;
        if (ct.term > 0.0f) break;
    }
    a.costs[static_cast<size_t>(e) * a.K + k] = total;
    reinterpret_cast<State*>(a.xend)[static_cast<size_t>(e) * a.K + k] = x;
}

FCB_HD float ess_at(const float* costs, int K, float cmin, float lam) {
    float z = 0.0f, z2 = 0.0f;
    for (int k = 0; k < K; ++k) {
        float wk = expf(-(costs[k] - cmin) / lam);
        z = z + wk;
        z2 = z2 + wk * wk;
    }
    return z * z / z2;
}

// softmin(costs / lambda) for one car; lambda by bisection on the effective sample size if ess_target > 0.
FCB_HD void weights_one(const FcbMppiArgs& a, int e) {
    const int K = a.K;
    const float* c = a.costs + static_cast<size_t>(e) * K;
    float* w = a.w + static_cast<size_t>(e) * K;
    float cmin = c[0];
    for (int k = 0; k < K; ++k) cmin = fmin_(cmin, c[k]);
    float lam_e = a.lam;
    if (a.ess_target > 0.0f) {
        float lo = -6.0f, hi = 5.0f;
        for (int it = 0; it < 24; ++it) {
            float mid = 0.5f * (lo + hi);
            if (ess_at(c, K, cmin, powf(10.0f, mid)) < a.ess_target)
                lo = mid;
            else
                hi = mid;
        }
        lam_e = powf(10.0f, 0.5f * (lo + hi));
    }
    float z = 0.0f;
    for (int k = 0; k < K; ++k) {
        float wk = expf(-(c[k] - cmin) / lam_e);
        w[k] = wk;
        z = z + wk;
    }
    float s2 = 0.0f;
    for (int k = 0; k < K; ++k) {
        float wk = w[k] / z;
        w[k] = wk;
        s2 = s2 + wk * wk;
    }
    a.info[e * 3 + 0] = cmin;
    a.info[e * 3 + 1] = 1.0f / s2;
    a.info[e * 3 + 2] = lam_e;
}

FCB_HD void update_one(const FcbMppiArgs& a, int e, int h) {
    float a0 = 0.0f, a1 = 0.0f;
    for (int k = 0; k < a.K; ++k) {
        float wk = a.w[static_cast<size_t>(e) * a.K + k];
        size_t i = vidx(a, e, k, h);
        a0 = a0 + wk * a.V[i];
        a1 = a1 + wk * a.V[i + 1];
    }
    a.Unew[(e * a.H + h) * 2] = a0;
    a.Unew[(e * a.H + h) * 2 + 1] = a1;
}

FCB_HD void copy_plan_one(const FcbMppiArgs& a, int e, int h) {
    a.U[(e * a.H + h) * 2] = a.Unew[(e * a.H + h) * 2];
    a.U[(e * a.H + h) * 2 + 1] = a.Unew[(e * a.H + h) * 2 + 1];
}

// act = Unew[:, 0]; warm start U[h] = Unew[h + 1] (last entry repeated)
FCB_HD void shift_one(const FcbMppiArgs& a, int e, int h) {
    int j = h + 1 < a.H - 1 ? h + 1 : a.H - 1;
    a.U[(e * a.H + h) * 2] = a.Unew[(e * a.H + j) * 2];
    a.U[(e * a.H + h) * 2 + 1] = a.Unew[(e * a.H + j) * 2 + 1];
    if (h == 0) {
        a.action[e * 2] = a.Unew[e * a.H * 2];
        a.action[e * 2 + 1] = a.Unew[e * a.H * 2 + 1];
    }
}

}  // namespace fcb
