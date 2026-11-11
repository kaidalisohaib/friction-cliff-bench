// CPU drivers: map the per-thread bodies of mppi_core.hpp onto loops (reference + GoogleTest).
#include <cstring>

#include "fcb/capi.h"

namespace fcb {
const char* g_last_error = "";
}

extern "C" {

int fcb_mppi_act_host(const fcb::FcbMppiArgs* args) {
    const fcb::FcbMppiArgs& a = *args;
    for (int it = 0; it < a.n_iter; ++it) {
        for (int e = 0; e < a.n; ++e)
            for (int k = 0; k < a.K; ++k) fcb::sample_one(a, e, k, a.t * 16 + it);
        for (int e = 0; e < a.n; ++e)
            for (int k = 0; k < a.K; ++k) fcb::rollout_one(a, e, k);
        for (int e = 0; e < a.n; ++e) fcb::weights_one(a, e);
        for (int e = 0; e < a.n; ++e)
            for (int h = 0; h < a.H; ++h) fcb::update_one(a, e, h);
        if (it < a.n_iter - 1)
            for (int e = 0; e < a.n; ++e)
                for (int h = 0; h < a.H; ++h) fcb::copy_plan_one(a, e, h);
    }
    for (int e = 0; e < a.n; ++e)
        for (int h = 0; h < a.H; ++h) fcb::shift_one(a, e, h);
    return 0;
}

int fcb_substep_host(const fcb::Params* p, const float* x, float u_steer, float u_long, const float* sf,
                     const float* sr, float dt, float* out) {
    fcb::State xs;
    std::memcpy(xs.v, x, sizeof(xs.v));
    fcb::Surf a, b;
    std::memcpy(a.v, sf, sizeof(a.v));
    std::memcpy(b.v, sr, sizeof(b.v));
    fcb::State o = fcb::substep(xs, u_steer, u_long, *p, a, b, dt);
    std::memcpy(out, o.v, sizeof(o.v));
    return 0;
}

int fcb_randn_host(int seed, int offset, int count, float* out) {
    uint32_t st = fcb::rand_init(seed, offset);
    for (int i = 0; i < count; ++i) out[i] = fcb::randn(st);
    return 0;
}

int fcb_params_echo_host(const fcb::Params* p, float* out30) {
    std::memcpy(out30, p, sizeof(fcb::Params));
    out30[19] = static_cast<float>(p->delay_steps);  // the int field, converted
    return 0;
}

const char* fcb_last_error() { return fcb::g_last_error; }
int fcb_version() { return 1; }
}
