// Shared types and constants of the C++17/CUDA port (BASELINES.md §5).
// Every function is __host__ __device__ so that the CPU build (GoogleTest, reference) and the CUDA kernels run the
// same source. The equations mirror sim/warp_sim.py and control/mppi.py line by line, in float32.
#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define FCB_HD __host__ __device__ __forceinline__
#else
#define FCB_HD inline
#endif

namespace fcb {

constexpr int NX = 13;          // state size (sim/reference.py NX)
constexpr int BUF_LEN = 9;      // action delay buffer length (sim/params.py BUF_LEN)
constexpr int N_SUBSTEPS = 20;  // physics substeps per control step
constexpr float CONTROL_DT = 0.02f;

constexpr float G = 9.81f;
constexpr float RHO = 1.225f;
constexpr float ALPHA_MAX = static_cast<float>(1.5707963267948966 - 0.01);  // sim/tire.py
constexpr float OPK_MIN = 0.01f;
constexpr float S_EPS = 1e-9f;
constexpr float FD_DOMEGA = 0.05f;
constexpr float HALF_LANE = 1.75f;
constexpr float PI_F = 3.14159265358979323846f;

// Same field order and types as the Warp struct sim/warp_sim.py:Params (all 4-byte fields, no padding), so a
// Warp array of Params can be read in place.
struct Params {
    float m, Iz, lf, lr, h, R_w, Jw, b_w, foot_front, foot_rear, width;
    float delta_max, delta_rate, tau_delta, tau_T, drive_force_max, brake_force_max, brake_front_frac,
        drive_front_frac;
    int32_t delay_steps;
    float B_scale_f, B_scale_r, f_rr, CdA, sigma_relax, combined_alpha, v_lo, v_hi, tau_kin, eps_v;
};
static_assert(sizeof(Params) == 30 * 4, "Params must match the Warp struct layout");

struct State {
    float v[NX];
};
// per-axle surface: (mu, Bx, Cx, Ex, By, Cy, Ey, unused)
struct Surf {
    float v[8];
};

// Shared objective weights (prereg V1), control/mppi.py:CostWeights
struct CostW {
    float w_lane, w_ey, w_v, w_du, w_prog, C_fail, severe, beta_term, spin_term;
};

FCB_HD float fmin_(float a, float b) { return fminf(a, b); }
FCB_HD float fmax_(float a, float b) { return fmaxf(a, b); }
FCB_HD float clampf(float x, float a, float b) { return fminf(fmaxf(a, x), b); }
FCB_HD int clampi(int x, int a, int b) { return x < a ? a : (x > b ? b : x); }

}  // namespace fcb
