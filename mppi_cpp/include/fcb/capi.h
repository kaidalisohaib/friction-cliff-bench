// C API of libfcb_mppi (loaded from Python with ctypes; see control/mppi_cpp.py). Returns 0 on success.
#pragma once

#include "mppi_core.hpp"

extern "C" {
// One MPPI control step (n_iter iterations + shift) on device pointers, launched on `stream` (cudaStream_t).
int fcb_mppi_act_cuda(const fcb::FcbMppiArgs* args, void* stream);
// The same on host pointers (CPU reference; also used by the GoogleTest suite).
int fcb_mppi_act_host(const fcb::FcbMppiArgs* args);
// Test hooks.
int fcb_substep_host(const fcb::Params* p, const float* x, float u_steer, float u_long, const float* sf,
                     const float* sr, float dt, float* out);
int fcb_randn_host(int seed, int offset, int count, float* out);
int fcb_params_echo_host(const fcb::Params* p, float* out30);
const char* fcb_last_error();
int fcb_version();
}
