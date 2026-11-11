// MPPI update math, sampling and determinism on the CPU reference driver.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>

#include "fcb/capi.h"
#include "harness.hpp"

namespace {

TEST(Rng, MatchesWarpRandn) {
    // tests/data/warp_randn.txt was drawn by wp.randn (make_fixtures.py); the port must be bit-compatible up to
    // libm rounding of log/cos/sqrt (host libm vs Warp's CPU build): 1e-6 absolute.
    std::ifstream f(std::string(FCB_TEST_DATA) + "/warp_randn.txt");
    std::string line;
    int rows = 0;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        int seed, off;
        is >> seed >> off;
        float out[6];
        fcb_randn_host(seed, off, 6, out);
        for (int j = 0; j < 6; ++j) {
            double ref;
            is >> ref;
            EXPECT_NEAR(out[j], ref, 1e-6) << "seed " << seed << " offset " << off << " draw " << j;
        }
        ++rows;
    }
    EXPECT_EQ(rows, 5);
}

TEST(Rng, StandardNormalMoments) {
    std::vector<float> v(200000);
    fcb_randn_host(42, 0, static_cast<int>(v.size()), v.data());
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double s2 = 0;
    for (float x : v) s2 += (x - m) * (x - m);
    s2 /= v.size();
    EXPECT_NEAR(m, 0.0, 0.01);
    EXPECT_NEAR(s2, 1.0, 0.01);
}

TEST(Weights, SoftminOfKnownCosts) {
    fcbtest::Problem P(1, 4, 5);
    float c[4] = {3.f, 1.f, 2.f, 1.f};
    std::copy(c, c + 4, P.costs.begin());
    P.args.lam = 1.f;
    fcb::weights_one(P.args, 0);
    double z = 2 + std::exp(-1.0) + std::exp(-2.0);  // exp(-(c - min c)) for c = 3, 1, 2, 1
    EXPECT_NEAR(P.w[1], 1 / z, 1e-6);
    EXPECT_NEAR(P.w[3], 1 / z, 1e-6);
    EXPECT_NEAR(P.w[2], std::exp(-1.0) / z, 1e-6);
    EXPECT_NEAR(P.w[0], std::exp(-2.0) / z, 1e-6);
    EXPECT_FLOAT_EQ(P.info[0], 1.f);  // min cost
    double ess = z * z / (2 + std::exp(-2.0) + std::exp(-4.0));
    EXPECT_NEAR(P.info[1], ess, 1e-4);
}

TEST(Weights, EssTargetBisection) {
    fcbtest::Problem P(1, 256, 5);
    for (int k = 0; k < 256; ++k) P.costs[k] = static_cast<float>(k);
    P.args.ess_target = 20.f;
    fcb::weights_one(P.args, 0);
    EXPECT_NEAR(P.info[1], 20.f, 0.5f);
    float s = std::accumulate(P.w.begin(), P.w.end(), 0.f);
    EXPECT_NEAR(s, 1.f, 1e-5f);
}

TEST(Update, WeightedAverageOfSamples) {
    fcbtest::Problem P(1, 3, 2);
    float w[3] = {0.5f, 0.25f, 0.25f};
    std::copy(w, w + 3, P.w.begin());
    for (int k = 0; k < 3; ++k)
        for (int h = 0; h < 2; ++h) {
            P.V[((k * 2) + h) * 2] = static_cast<float>(k + h);
            P.V[((k * 2) + h) * 2 + 1] = -static_cast<float>(k);
        }
    for (int h = 0; h < 2; ++h) fcb::update_one(P.args, 0, h);
    EXPECT_NEAR(P.Unew[0], 0.75f, 1e-6f);   // h = 0: 0.5*0 + 0.25*1 + 0.25*2
    EXPECT_NEAR(P.Unew[2], 1.75f, 1e-6f);   // h = 1
    EXPECT_NEAR(P.Unew[1], -0.75f, 1e-6f);
}

TEST(Mppi, SampleZeroIsTheNominalPlanAndNoiseIsClipped) {
    fcbtest::Problem P(2, 64, 30);
    for (size_t i = 0; i < P.U.size(); ++i) P.U[i] = (i % 2) ? 0.95f : -0.2f;
    P.args.sigma_long = 0.5f;
    for (int e = 0; e < 2; ++e)
        for (int k = 0; k < 64; ++k) fcb::sample_one(P.args, e, k, 0);
    for (int e = 0; e < 2; ++e)
        for (int h = 0; h < 30; ++h) {
            EXPECT_EQ(P.V[((e * 64 + 0) * 30 + h) * 2], P.U[(e * 30 + h) * 2]);
            EXPECT_EQ(P.V[((e * 64 + 0) * 30 + h) * 2 + 1], P.U[(e * 30 + h) * 2 + 1]);
        }
    for (float v : P.V) {
        EXPECT_LE(v, 1.f);
        EXPECT_GE(v, -1.f);
    }
}

TEST(Mppi, DeterministicGivenSeeds) {
    fcbtest::Problem A(3, 128, 50), B(3, 128, 50);
    for (int t = 0; t < 3; ++t) {
        A.args.t = B.args.t = t;
        ASSERT_EQ(0, fcb_mppi_act_host(&A.args));
        ASSERT_EQ(0, fcb_mppi_act_host(&B.args));
    }
    EXPECT_EQ(A.action, B.action);
    EXPECT_EQ(A.U, B.U);
}

TEST(Mppi, OffsetCarSteersBackTowardsCentre) {
    // car 0 is 0.5 m right of the centre line (e < 0), car 2 is 0.5 m left: the first steering action must have
    // the sign that reduces |e| (left = positive steer).
    fcbtest::Problem P(3, 512, 100);
    ASSERT_EQ(0, fcb_mppi_act_host(&P.args));
    EXPECT_GT(P.action[0], 0.f);
    EXPECT_LT(P.action[4], 0.f);
    for (int e = 0; e < 3; ++e) {
        EXPECT_GT(P.info[e * 3 + 1], 1.f);    // effective sample size
        EXPECT_LE(P.info[e * 3 + 1], 512.f);
    }
}

TEST(Mppi, ClosedLoopOnArcStaysInLane) {
    // closed loop on a 100 m radius arc at 15 m/s with the CPU driver and the port's own dynamics
    fcbtest::Problem P(1, 256, 75, 15.f, 100.f);
    const fcb::Params p = P.mparams[0];
    fcb::Surf d = fcbtest::dry();
    fcb::Surf ds = d, dr = d;
    ds.v[4] *= p.B_scale_f;
    dr.v[4] *= p.B_scale_r;
    fcb::Tracks T{P.tx.data(), P.ty.data(), P.tth.data(), P.tk.data(), P.track_n.data(), 4000, 0.25f};
    float max_e = 0.f;
    for (int t = 0; t < 100; ++t) {  // 2 s
        P.args.t = t;
        ASSERT_EQ(0, fcb_mppi_act_host(&P.args));
        // step the true car (delay 1: apply last step's action), shift the buffer
        for (int j = fcb::BUF_LEN - 1; j > 0; --j) {
            P.buf[j * 2] = P.buf[(j - 1) * 2];
            P.buf[j * 2 + 1] = P.buf[(j - 1) * 2 + 1];
        }
        P.buf[0] = P.action[0];
        P.buf[1] = P.action[1];
        fcb::State x;
        std::copy(P.state.begin(), P.state.begin() + 13, x.v);
        for (int j = 0; j < fcb::N_SUBSTEPS; ++j) x = fcb::substep(x, P.buf[2], P.buf[3], p, ds, dr, P.args.dt);
        std::copy(x.v, x.v + 13, P.state.begin());
        fcb::F3 q = fcb::project_local(T, 0, 4000, x.v[0], x.v[1], P.hints[0]);
        P.hints[0] = static_cast<int>(q.idx);
        P.hints[1] = P.hints[0] + 5;
        P.hints[2] = P.hints[0] - 6;
        P.frenet[0] = q.s;
        P.frenet[1] = q.e;
        max_e = std::max(max_e, std::fabs(q.e));
    }
    EXPECT_LT(max_e, 0.6f);
    EXPECT_NEAR(P.state[3], 15.f, 1.5f);
}

}  // namespace
