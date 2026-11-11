// CUDA driver vs CPU driver on the same problem: identical samples up to libm rounding, same plan and action.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "fcb/capi.h"
#include "harness.hpp"

namespace {

bool have_gpu() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

template <class T>
T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

struct DeviceProblem {
    fcb::FcbMppiArgs a;
    std::vector<void*> allocs;
    template <class T>
    T* up(const std::vector<T>& v) {
        T* p = upload(v);
        allocs.push_back(p);
        return p;
    }
    explicit DeviceProblem(const fcbtest::Problem& P) : a(P.args) {
        a.mparams = up(P.mparams); a.state = up(P.state); a.buf = up(P.buf); a.hints = up(P.hints);
        a.frenet = up(P.frenet); a.sf_fix = up(P.sf); a.sr_fix = up(P.sr); a.prof = up(P.prof); a.base = up(P.base);
        a.track_id = up(P.track_id); a.track_x = up(P.tx); a.track_y = up(P.ty); a.track_th = up(P.tth);
        a.track_k = up(P.tk); a.track_n = up(P.track_n); a.v_ref = up(P.v_ref); a.seeds = up(P.seeds);
        a.U = up(P.U); a.V = up(P.V); a.costs = up(P.costs); a.xend = up(P.xend); a.w = up(P.w);
        a.info = up(P.info); a.Unew = up(P.Unew); a.action = up(P.action);
    }
    ~DeviceProblem() {
        for (void* p : allocs) cudaFree(p);
    }
};

void compare_modes(int mode, float rho, int n_iter) {
    fcbtest::Problem P(4, 256, 60);
    P.args.mode = mode;
    P.args.rho = rho;
    P.args.n_iter = n_iter;
    DeviceProblem D(P);
    for (int t = 0; t < 3; ++t) {
        P.args.t = D.a.t = t;
        ASSERT_EQ(0, fcb_mppi_act_host(&P.args));
        ASSERT_EQ(0, fcb_mppi_act_cuda(&D.a, nullptr)) << fcb_last_error();
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    }
    auto act = download(D.a.action, P.action.size());
    auto U = download(D.a.U, P.U.size());
    for (size_t i = 0; i < act.size(); ++i) EXPECT_NEAR(act[i], P.action[i], 2e-3f) << i;
    double maxdu = 0;
    for (size_t i = 0; i < U.size(); ++i) maxdu = std::max(maxdu, static_cast<double>(std::fabs(U[i] - P.U[i])));
    EXPECT_LT(maxdu, 1e-2);
}

TEST(Cuda, MatchesCpuFixed) {
    if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
    compare_modes(fcb::MODE_FIXED, 0.f, 1);
}
TEST(Cuda, MatchesCpuFieldCorrelatedTwoIters) {
    if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
    compare_modes(fcb::MODE_FIELD, 0.9f, 2);
}

}  // namespace
