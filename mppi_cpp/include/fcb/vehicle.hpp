// Vehicle dynamics: dynamic bicycle, combined-slip Pacejka, wheel spin, relaxation, actuators (DESIGN.md §2).
// Line-by-line port of sim/warp_sim.py (tire, slip_ratio, wheel_update, substep).
#pragma once

#include "common.hpp"

namespace fcb {

FCB_HD float mf(float x, float B, float C, float D, float E) {
    float bx = B * x;
    return D * sinf(C * atanf(bx - E * (bx - atanf(bx))));
}

struct F2 {
    float x, y;
};

FCB_HD F2 tire(float kappa, float alpha, float Fz, const Surf& s, float ca) {
    float a = clampf(alpha, -ALPHA_MAX, ALPHA_MAX);
    float opk = fmax_(1.0f + kappa, OPK_MIN);
    float sx = kappa / opk;
    float sy = tanf(a) / opk;
    float D = s.v[0] * Fz;
    float sn = sqrtf(sx * sx + sy * sy);
    float fx = 0.0f, fy = 0.0f;
    if (sn > S_EPS) {
        fx = mf(sn, s.v[1], s.v[2], D, s.v[3]) * sx / sn;
        fy = mf(sn, s.v[4], s.v[5], D, s.v[6]) * sy / sn;
    }
    if (ca != 0.0f) {
        float fx_e = mf(sx, s.v[1], s.v[2], D, s.v[3]);
        float ratio = 0.0f;
        if (D > 0.0f) ratio = fx_e / D;
        float fy_e = mf(tanf(a), s.v[4], s.v[5], D, s.v[6]) * sqrtf(clampf(1.0f - ratio * ratio, 0.0f, 1.0f));
        fx = (1.0f - ca) * fx + ca * fx_e;
        fy = (1.0f - ca) * fy + ca * fy_e;
    }
    return {fx, fy};
}

FCB_HD float slip_ratio(float R, float omega, float vxw, float eps_v) {
    return (R * omega - vxw) / fmax_(fmax_(fabsf(R * omega), fabsf(vxw)), eps_v);
}

FCB_HD float smoothstep(float t) {
    float u = clampf(t, 0.0f, 1.0f);
    return u * u * (3.0f - 2.0f * u);
}

FCB_HD F2 clip_circle(F2 f, float fmax) {
    float n = sqrtf(f.x * f.x + f.y * f.y);
    if (n > fmax) {
        float k = fmax / n;
        return {f.x * k, f.y * k};
    }
    return f;
}

FCB_HD float wheel_update(float omega, float T_drive, float T_brake, float Fx, float vxw, float alpha, float Fz,
                          const Surf& srf, const Params& p, float dt) {
    float Fx_h = tire(slip_ratio(p.R_w, omega + FD_DOMEGA, vxw, p.eps_v), alpha, Fz, srf, p.combined_alpha).x;
    float dFx = (Fx_h - Fx) / FD_DOMEGA;
    float k = fmax_((p.R_w * dFx + p.b_w) / p.Jw, 0.0f);
    float denom = 1.0f + dt * k;
    float om = omega + dt * (T_drive - p.R_w * Fx - p.b_w * omega) / p.Jw / denom;
    float db = dt * T_brake / p.Jw / denom;
    if (fabsf(om) <= db) return 0.0f;
    if (om > 0.0f) return om - db;
    return om + db;
}

// One 1 ms physics substep. sf, sr: per-axle surfaces with the lateral B already scaled by B_scale_{f,r}.
FCB_HD State substep(const State& xin, float u_steer, float u_long, const Params& p, const Surf& sf, const Surf& sr,
                     float dt) {
    const float* x = xin.v;
    float vx = x[3], vy = x[4], r = x[5], wf = x[6], wr = x[7], fyf_l = x[8], fyr_l = x[9], delta = x[10],
          a_act = x[11];
    float L = p.lf + p.lr;
    const float one = 1.0f, zero = 0.0f;

    float d_target = clampf(u_steer, -one, one) * p.delta_max;
    float d_dot = clampf((d_target - delta) / p.tau_delta, -p.delta_rate, p.delta_rate);
    delta = clampf(delta + dt * d_dot, -p.delta_max, p.delta_max);
    a_act = a_act + dt / p.tau_T * (clampf(u_long, -one, one) - a_act);

    float T_drive = fmax_(a_act, zero) * p.drive_force_max * p.R_w;
    float T_brake = fmax_(-a_act, zero) * p.brake_force_max * p.R_w;
    float Tdf = T_drive * p.drive_front_frac;
    float Tdr = T_drive * (one - p.drive_front_frac);
    float Tbf = T_brake * p.brake_front_frac;
    float Tbr = T_brake * (one - p.brake_front_frac);

    float ax = clampf(x[12], -0.95f * G * p.lf / p.h, 0.95f * G * p.lr / p.h);
    float Fzf = p.m * (G * p.lr - ax * p.h) / L;
    float Fzr = p.m * (G * p.lf + ax * p.h) / L;

    float c = cosf(delta), s = sinf(delta);
    float vyf_body = vy + p.lf * r;
    float vxf = vx * c + vyf_body * s;
    float vyf = -vx * s + vyf_body * c;
    float vxr = vx;
    float vyr = vy - p.lr * r;
    float alpha_f = -atan2f(vyf, fabsf(vxf));
    float alpha_r = -atan2f(vyr, fabsf(vxr));

    F2 Ff = tire(slip_ratio(p.R_w, wf, vxf, p.eps_v), alpha_f, Fzf, sf, p.combined_alpha);
    F2 Fr = tire(slip_ratio(p.R_w, wr, vxr, p.eps_v), alpha_r, Fzr, sr, p.combined_alpha);

    wf = wheel_update(wf, Tdf, Tbf, Ff.x, vxf, alpha_f, Fzf, sf, p, dt);
    wr = wheel_update(wr, Tdr, Tbr, Fr.x, vxr, alpha_r, Fzr, sr, p, dt);

    if (p.sigma_relax > zero) {
        fyf_l = fyf_l + fmin_(dt * fabsf(vxf) / p.sigma_relax, one) * (Ff.y - fyf_l);
        fyr_l = fyr_l + fmin_(dt * fabsf(vxr) / p.sigma_relax, one) * (Fr.y - fyr_l);
    } else {
        fyf_l = Ff.y;
        fyr_l = Fr.y;
    }
    F2 Faf = clip_circle({Ff.x, fyf_l}, sf.v[0] * Fzf);
    F2 Far = clip_circle({Fr.x, fyr_l}, sr.v[0] * Fzr);

    float Fx_fb = Faf.x * c - Faf.y * s;
    float Fy_fb = Faf.x * s + Faf.y * c;
    float F_rr = -p.f_rr * (Fzf + Fzr) * tanhf(vx / 0.1f);
    float F_aero = -0.5f * RHO * p.CdA * vx * fabsf(vx);
    float SFx = Fx_fb + Far.x + F_rr + F_aero;
    float SFy = Fy_fb + Far.y;
    float Mz = p.lf * Fy_fb - p.lr * Far.y;

    float dvx = SFx / p.m + r * vy;
    float dvy_dyn = SFy / p.m - r * vx;
    float dr_dyn = Mz / p.Iz;
    float wgt = smoothstep((sqrtf(vx * vx + vy * vy) - p.v_lo) / (p.v_hi - p.v_lo));
    float r_kin = vx * tanf(delta) / L;
    float vy_kin = p.lr * r_kin;
    float dvy = wgt * dvy_dyn + (one - wgt) * (vy_kin - vy) / p.tau_kin;
    float dr = wgt * dr_dyn + (one - wgt) * (r_kin - r) / p.tau_kin;

    vx = vx + dt * dvx;
    vy = vy + dt * dvy;
    r = r + dt * dr;
    float psi = x[2] + dt * r;
    float X = x[0] + dt * (vx * cosf(psi) - vy * sinf(psi));
    float Y = x[1] + dt * (vx * sinf(psi) + vy * cosf(psi));

    State o;
    o.v[0] = X;
    o.v[1] = Y;
    o.v[2] = psi;
    o.v[3] = vx;
    o.v[4] = vy;
    o.v[5] = r;
    o.v[6] = wf;
    o.v[7] = wr;
    o.v[8] = fyf_l;
    o.v[9] = fyr_l;
    o.v[10] = delta;
    o.v[11] = a_act;
    o.v[12] = SFx / p.m;
    return o;
}

}  // namespace fcb
