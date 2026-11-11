// CUDA drivers: one thread per (car, sample) for sampling and rollouts, one per car for the weights, one per
// (car, horizon step) for the update and the shift, as in control/mppi.py. All launches go on the caller's stream.
#include <cuda_runtime.h>

#include "fcb/capi.h"

namespace fcb {
extern const char* g_last_error;

__global__ void k_sample(FcbMppiArgs a, int t_rng) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<long long>(a.n) * a.K) return;
    sample_one(a, static_cast<int>(i / a.K), static_cast<int>(i % a.K), t_rng);
}
__global__ void k_rollout(FcbMppiArgs a) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<long long>(a.n) * a.K) return;
    rollout_one(a, static_cast<int>(i / a.K), static_cast<int>(i % a.K));
}
__global__ void k_weights(FcbMppiArgs a) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= a.n) return;
    weights_one(a, e);
}
__global__ void k_update(FcbMppiArgs a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n * a.H) return;
    update_one(a, i / a.H, i % a.H);
}
__global__ void k_copy_plan(FcbMppiArgs a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n * a.H) return;
    copy_plan_one(a, i / a.H, i % a.H);
}
__global__ void k_shift(FcbMppiArgs a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.n * a.H) return;
    shift_one(a, i / a.H, i % a.H);
}

// --- block-parallel weights and update (one block per car / per (car, h)) ------------------------------------
// Same math as weights_one / update_one, but the sums over the K samples are tree reductions in shared memory
// instead of one thread looping over K. Only the summation order differs (float rounding; S3 tolerances hold).
constexpr int RB = 256;  // reduction block size (power of two)

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = RB / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_min(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = RB / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] = fminf(sh[threadIdx.x], sh[threadIdx.x + o]);
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

__global__ void k_weights_block(FcbMppiArgs a) {
    __shared__ float sh[RB];
    const int e = blockIdx.x, K = a.K;
    const float* c = a.costs + static_cast<size_t>(e) * K;
    float* w = a.w + static_cast<size_t>(e) * K;
    float m = 1.0e38f;
    for (int k = threadIdx.x; k < K; k += RB) m = fminf(m, c[k]);
    const float cmin = block_min(m, sh);
    float lam_e = a.lam;
    if (a.ess_target > 0.0f) {
        float lo = -6.0f, hi = 5.0f;
        for (int it = 0; it < 24; ++it) {
            float mid = 0.5f * (lo + hi);
            float lam = powf(10.0f, mid), z = 0.0f, z2 = 0.0f;
            for (int k = threadIdx.x; k < K; k += RB) {
                float wk = expf(-(c[k] - cmin) / lam);
                z += wk;
                z2 += wk * wk;
            }
            z = block_sum(z, sh);
            z2 = block_sum(z2, sh);
            if (z * z / z2 < a.ess_target)
                lo = mid;
            else
                hi = mid;
        }
        lam_e = powf(10.0f, 0.5f * (lo + hi));
    }
    float z = 0.0f;
    for (int k = threadIdx.x; k < K; k += RB) {
        float wk = expf(-(c[k] - cmin) / lam_e);
        w[k] = wk;
        z += wk;
    }
    z = block_sum(z, sh);
    float s2 = 0.0f;
    for (int k = threadIdx.x; k < K; k += RB) {
        float wk = w[k] / z;
        w[k] = wk;
        s2 += wk * wk;
    }
    s2 = block_sum(s2, sh);
    if (threadIdx.x == 0) {
        a.info[e * 3 + 0] = cmin;
        a.info[e * 3 + 1] = 1.0f / s2;
        a.info[e * 3 + 2] = lam_e;
    }
}

__global__ void k_update_block(FcbMppiArgs a) {
    __shared__ float sh[RB];
    const int e = blockIdx.x / a.H, h = blockIdx.x % a.H;
    float a0 = 0.0f, a1 = 0.0f;
    for (int k = threadIdx.x; k < a.K; k += RB) {
        float wk = a.w[static_cast<size_t>(e) * a.K + k];
        size_t i = vidx(a, e, k, h);
        a0 += wk * a.V[i];
        a1 += wk * a.V[i + 1];
    }
    a0 = block_sum(a0, sh);
    a1 = block_sum(a1, sh);
    if (threadIdx.x == 0) {
        a.Unew[(e * a.H + h) * 2] = a0;
        a.Unew[(e * a.H + h) * 2 + 1] = a1;
    }
}

static unsigned blocks(long long n, int bs) { return static_cast<unsigned>((n + bs - 1) / bs); }
}  // namespace fcb

extern "C" int fcb_mppi_act_cuda(const fcb::FcbMppiArgs* args, void* stream) {
    using namespace fcb;
    const FcbMppiArgs a = *args;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const long long nk = static_cast<long long>(a.n) * a.K;
    const long long nh = static_cast<long long>(a.n) * a.H;
    constexpr int BS_ROLL = 128, BS = 256;
    for (int it = 0; it < a.n_iter; ++it) {
        k_sample<<<blocks(nk, BS), BS, 0, s>>>(a, a.t * 16 + it);
        k_rollout<<<blocks(nk, BS_ROLL), BS_ROLL, 0, s>>>(a);
        if (a.K >= RB) {  // block reductions pay off once there is a block's worth of samples per car
            k_weights_block<<<a.n, RB, 0, s>>>(a);
            k_update_block<<<static_cast<unsigned>(nh), RB, 0, s>>>(a);
        } else {
            k_weights<<<blocks(a.n, 64), 64, 0, s>>>(a);
            k_update<<<blocks(nh, BS), BS, 0, s>>>(a);
        }
        if (it < a.n_iter - 1) k_copy_plan<<<blocks(nh, BS), BS, 0, s>>>(a);
    }
    k_shift<<<blocks(nh, BS), BS, 0, s>>>(a);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        g_last_error = cudaGetErrorString(err);
        return 1;
    }
    return 0;
}
