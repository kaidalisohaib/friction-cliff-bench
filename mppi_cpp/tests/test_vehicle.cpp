// Dynamics vs the float64 NumPy reference (sim/reference.py) on the 7 V8 maneuvers, with the V8 tolerances of
// tests/test_warp_sim.py: position within 2 cm and vx, vy, r within 1e-3 relative over 2 s.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fcb/capi.h"
#include "harness.hpp"

namespace {

struct Case {
    std::string name;
    std::vector<float> params;
    std::vector<float> x0;
    std::vector<std::vector<double>> steps;  // u(2), sf(8), sr(8), x_ref(13)
};

std::vector<double> nums(std::istringstream& is) {
    std::vector<double> v;
    double d;
    while (is >> d) v.push_back(d);
    return v;
}

std::vector<Case> load_cases() {
    std::ifstream f(std::string(FCB_TEST_DATA) + "/substep_cases.txt");
    std::vector<Case> cases;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string tag;
        is >> tag;
        if (tag == "case") {
            cases.emplace_back();
            is >> cases.back().name;
        } else if (tag == "params") {
            for (double d : nums(is)) cases.back().params.push_back(static_cast<float>(d));
        } else if (tag == "x0") {
            for (double d : nums(is)) cases.back().x0.push_back(static_cast<float>(d));
        } else if (tag == "step") {
            cases.back().steps.push_back(nums(is));
        }
    }
    return cases;
}

fcb::Params params_from(const std::vector<float>& v) {
    // field order of VehicleParams == field order of fcb::Params (delay_steps is the 20th field)
    fcb::Params p{};
    float* raw = reinterpret_cast<float*>(&p);
    for (int i = 0; i < 30; ++i) raw[i] = v[i];
    p.delay_steps = static_cast<int32_t>(v[19]);
    return p;
}

TEST(Vehicle, FixturesPresent) {
    auto cases = load_cases();
    ASSERT_EQ(cases.size(), 7u);
    for (auto& c : cases) {
        EXPECT_EQ(c.params.size(), 30u);
        EXPECT_EQ(c.x0.size(), 13u);
        EXPECT_EQ(c.steps.size(), 100u);
    }
}

TEST(Vehicle, DefaultParamsMatchFixture) {
    fcb::Params a = fcbtest::default_params();
    fcb::Params b = params_from(load_cases()[0].params);
    EXPECT_EQ(0, std::memcmp(&a, &b, sizeof(a)));
}

class V8 : public ::testing::TestWithParam<int> {};

TEST_P(V8, MatchesNumpyReference) {
    const Case c = load_cases()[GetParam()];
    const fcb::Params p = params_from(c.params);
    fcb::State x;
    std::copy(c.x0.begin(), c.x0.end(), x.v);
    const float dt = fcb::CONTROL_DT / fcb::N_SUBSTEPS;
    double pos_err = 0, scale[3] = {0.1, 0.1, 0.1}, err[3] = {0, 0, 0};
    const int idx[3] = {3, 4, 5};
    for (const auto& st : c.steps) {
        fcb::Surf sf, sr;
        for (int i = 0; i < 8; ++i) {
            sf.v[i] = static_cast<float>(st[2 + i]);
            sr.v[i] = static_cast<float>(st[10 + i]);
        }
        for (int j = 0; j < fcb::N_SUBSTEPS; ++j)
            x = fcb::substep(x, static_cast<float>(st[0]), static_cast<float>(st[1]), p, sf, sr, dt);
        const double* xr = &st[18];
        for (int i = 0; i < 13; ++i) ASSERT_TRUE(std::isfinite(x.v[i])) << c.name;
        pos_err = std::max(pos_err, std::hypot(xr[0] - x.v[0], xr[1] - x.v[1]));
        for (int q = 0; q < 3; ++q) {
            scale[q] = std::max(scale[q], std::fabs(xr[idx[q]]));
            err[q] = std::max(err[q], std::fabs(xr[idx[q]] - x.v[idx[q]]));
        }
    }
    EXPECT_LT(pos_err, 0.02) << c.name;
    for (int q = 0; q < 3; ++q) EXPECT_LT(err[q] / scale[q], 1e-3) << c.name << " state " << idx[q];
    RecordProperty("max_pos_err_m", std::to_string(pos_err));
}

INSTANTIATE_TEST_SUITE_P(AllManeuvers, V8, ::testing::Range(0, 7));

TEST(Vehicle, StandstillStaysPut) {
    fcb::State x{};
    fcb::Params p = fcbtest::default_params();
    fcb::Surf d = fcbtest::dry();
    for (int j = 0; j < 1000; ++j) x = fcb::substep(x, 0.f, 0.f, p, d, d, 1e-3f);
    for (int i = 0; i < 13; ++i) EXPECT_NEAR(x.v[i], 0.f, 1e-6f) << i;
}

TEST(Vehicle, BrakingOnIceTakesLongerThanOnDry) {
    fcb::Params p = fcbtest::default_params();
    auto stop_dist = [&](const fcb::Surf& s) {
        fcb::State x{};
        x.v[3] = 15.f;
        x.v[6] = x.v[7] = 15.f / p.R_w;
        for (int j = 0; j < 60000 && x.v[3] > 0.05f; ++j) x = fcb::substep(x, 0.f, -1.f, p, s, s, 1e-3f);
        return x.v[0];
    };
    float d_dry = stop_dist(fcbtest::dry()), d_ice = stop_dist(fcbtest::ice());
    EXPECT_GT(d_dry, 5.f);
    EXPECT_GT(d_ice, 4.f * d_dry);  // mu 0.1 vs 1.0: roughly 10x longer
}

}  // namespace
