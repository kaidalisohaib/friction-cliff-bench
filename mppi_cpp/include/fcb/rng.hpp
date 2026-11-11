// Bit-exact copy of NVIDIA Warp's random number generator (warp/native/rand.h, warp-lang 1.18.0: PCG hash of
// Jarzynski & Olano 2020, Box-Muller normal). Matching it lets the C++ MPPI draw exactly the samples the Warp MPPI
// draws, so the two implementations can be compared on whole control steps (S3).
#pragma once

#include "common.hpp"

namespace fcb {

constexpr float RANDN_EPSILON = 5.96e-8f;

FCB_HD uint32_t rand_pcg(uint32_t state) {
    uint32_t b = state * 747796405u + 2891336453u;
    uint32_t c = ((b >> ((b >> 28u) + 4u)) ^ b) * 277803737u;
    return (c >> 22u) ^ c;
}

FCB_HD uint32_t rand_init(int32_t seed, int32_t offset) {
    return rand_pcg(static_cast<uint32_t>(seed) + rand_pcg(static_cast<uint32_t>(offset)));
}

FCB_HD float randf(uint32_t& state) {
    state = rand_pcg(state);
    return static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
}

// Warp evaluates sqrt(-2 log(randf + eps)) * cos(2 pi randf) with the log draw first (checked against Warp in
// tests/test_mppi_cpp.py::test_rng_matches_warp).
FCB_HD float randn(uint32_t& state) {
    float u1 = randf(state);
    float u2 = randf(state);
    return sqrtf(-2.0f * logf(u1 + RANDN_EPSILON)) * cosf(2.0f * PI_F * u2);
}

}  // namespace fcb
