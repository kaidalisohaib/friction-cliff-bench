// Track projection and the friction field along a track (port of sim/warp_sim.py project_local / field_*).
#pragma once

#include "common.hpp"

namespace fcb {

constexpr int WIN = 8;  // local projection search window [table points]

// Tabulated tracks, row-major (n_tracks, stride); row tid has n[tid] valid points.
struct Tracks {
    const float* x;
    const float* y;
    const float* th;
    const float* k;
    const int32_t* n;
    int stride;
    float ds;
    FCB_HD float X(int t, int j) const { return x[t * stride + j]; }
    FCB_HD float Y(int t, int j) const { return y[t * stride + j]; }
    FCB_HD float TH(int t, int j) const { return th[t * stride + j]; }
    FCB_HD float K(int t, int j) const { return k[t * stride + j]; }
};

struct F3 {
    float s, e, idx;
};

// Frenet (s, e, idx) of a point by local search around `hint` (left of travel: e > 0).
FCB_HD F3 project_local(const Tracks& T, int tid, int npts, float px, float py, int hint) {
    const float ds = T.ds;
    int best_j = hint;
    float best_d = 1.0e30f;
    for (int k = 0; k < 2 * WIN + 1; ++k) {
        int j = clampi(hint - WIN + k, 0, npts - 1);
        float dx = T.X(tid, j) - px;
        float dy = T.Y(tid, j) - py;
        float d = dx * dx + dy * dy;
        if (d < best_d) {
            best_d = d;
            best_j = j;
        }
    }
    float s_out = static_cast<float>(best_j) * ds;
    float e_out = 0.0f;
    float best_seg = 1.0e30f;
    for (int q = 0; q < 2; ++q) {
        int a = best_j - 1 + q;
        if (a >= 0 && a + 1 < npts) {
            float tx = T.X(tid, a + 1) - T.X(tid, a);
            float ty = T.Y(tid, a + 1) - T.Y(tid, a);
            float seg2 = tx * tx + ty * ty;
            float t = clampf(((px - T.X(tid, a)) * tx + (py - T.Y(tid, a)) * ty) / seg2, 0.0f, 1.0f);
            float qx = T.X(tid, a) + t * tx;
            float qy = T.Y(tid, a) + t * ty;
            float d = (px - qx) * (px - qx) + (py - qy) * (py - qy);
            if (d < best_seg) {
                best_seg = d;
                s_out = (static_cast<float>(a) + t) * ds;
                e_out = (tx * (py - T.Y(tid, a)) - ty * (px - T.X(tid, a))) / sqrtf(seg2);
            }
        }
    }
    return {s_out, e_out, static_cast<float>(best_j)};
}

FCB_HD float field_weight(float s, float s_drop, float lt, float Lp) {
    float s_end = s_drop + lt + Lp;
    float up = 0.0f, down = 0.0f;
    if (lt > 0.0f) {
        up = clampf((s - s_drop) / lt, 0.0f, 1.0f);
        down = clampf(1.0f - (s - s_end) / lt, 0.0f, 1.0f);
    } else {
        if (s >= s_drop) up = 1.0f;
        if (s < s_end) down = 1.0f;
    }
    return fmin_(up, down);
}

// prof = (s_drop, lt, Lp, mu0, mu1, patch_surface_index, -, -); base[0] = dry shape
FCB_HD Surf field_surface(float s, const Surf& prof, const Surf* base) {
    float w = field_weight(s, prof.v[0], prof.v[1], prof.v[2]);
    Surf a = base[0];
    Surf b = base[static_cast<int>(prof.v[5])];
    a.v[0] = prof.v[3];
    b.v[0] = prof.v[4];
    Surf o;
    for (int i = 0; i < 8; ++i) o.v[i] = a.v[i] * (1.0f - w) + b.v[i] * w;
    return o;
}

}  // namespace fcb
