"""Vehicle and surface parameters (DESIGN.md §2.3).

Single source of truth: the NumPy reference and the Warp simulator both read these dataclasses.
Values are the *initial* generic mid-size sedan from DESIGN.md; they get sources and are frozen by a
dated prereg amendment at G1 (D13).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

G = 9.81  # gravity [m/s^2]
RHO_AIR = 1.225  # air density [kg/m^3]

CONTROL_DT = 0.02  # control period [s] (50 Hz)
N_SUBSTEPS = 20  # physics substeps per control step
PHYSICS_DT = CONTROL_DT / N_SUBSTEPS  # 1 ms

MAX_DELAY_STEPS = 8  # max pure delay [control steps]: 1 nominal + up to 7 extra (140 ms, prereg V6)
BUF_LEN = MAX_DELAY_STEPS + 1  # action buffer: index j holds the action from j steps ago

# Drivetrain: fraction of drive torque sent to the front axle.
RWD, FWD, AWD = 0.0, 1.0, 0.5


@dataclass(frozen=True)
class VehicleParams:
    m: float = 1600.0  # mass [kg]
    Iz: float = 2700.0  # yaw inertia [kg m^2]
    lf: float = 1.25  # CoG to front axle [m]
    lr: float = 1.45  # CoG to rear axle [m]
    h: float = 0.55  # CoG height [m]
    R_w: float = 0.33  # wheel radius [m]
    Jw: float = 2.0  # wheel inertia per axle (two wheels lumped) [kg m^2]
    b_w: float = 0.0  # wheel bearing damping per axle [N m s]
    # footprint relative to the CoG (front/rear overhang from DESIGN's 4.8 x 1.85 m box)
    foot_front: float = 2.2  # CoG to front bumper [m]
    foot_rear: float = 2.6  # CoG to rear bumper [m]
    width: float = 1.85  # [m]
    # actuators
    delta_max: float = 0.55  # road-wheel angle limit [rad]
    delta_rate: float = 0.7  # road-wheel rate limit [rad/s]
    tau_delta: float = 0.08  # steering lag [s]
    tau_T: float = 0.10  # longitudinal lag [s]
    drive_force_max: float = 0.4 * 1600.0 * G  # at the wheels [N]
    brake_force_max: float = 1.0 * 1600.0 * G  # total [N]
    brake_front_frac: float = 0.65
    drive_front_frac: float = RWD  # prereg V8: RWD
    delay_steps: int = 1  # pure action delay in control steps (nominal 20 ms)
    # per-axle cornering-stiffness scale on the surface B (1.0 = DESIGN.md). With equal tire coefficients
    # a bicycle model is exactly neutral-steer (C_alpha ~ Fz ~ the other axle's lever arm); real sedans
    # understeer, so the G1 parameter freeze (D13) must pick B_scale_r > B_scale_f from data (comma, P5).
    B_scale_f: float = 1.0
    B_scale_r: float = 1.0
    # resistances
    f_rr: float = 0.012  # rolling resistance coefficient
    CdA: float = 0.7  # drag area [m^2]
    # model switches (H2 mismatch axes, prereg V6)
    sigma_relax: float = 0.0  # tire relaxation length [m]; 0 = off (nominal)
    combined_alpha: float = 0.0  # 0 = slip-vector combined slip (nominal), 1 = friction ellipse
    # numerics
    v_lo: float = 0.5  # below: kinematic bicycle [m/s]
    v_hi: float = 2.0  # above: fully dynamic model [m/s]
    tau_kin: float = 0.05  # relaxation time towards kinematic vy, r at low speed [s]
    eps_v: float = 0.1  # floor on slip-ratio denominator [m/s]

    @property
    def L(self) -> float:
        return self.lf + self.lr

    @property
    def lane_slack(self) -> float:
        """Half-width the CoG may deviate in a 3.5 m lane before a footprint point must cross the edge."""
        return 3.5 / 2 - self.width / 2

    def with_(self, **kw) -> "VehicleParams":
        return replace(self, **kw)


@dataclass(frozen=True)
class Surface:
    """Magic Formula shape per surface: D = mu * Fz (DESIGN.md §2.2, D11)."""

    name: str
    B: float
    C: float
    mu: float
    E: float

    def with_mu(self, mu: float) -> "Surface":
        return replace(self, mu=mu)


# Initial "typical MF coefficients" (DESIGN.md §2.2; primary source to be cited in P1, D13).
DRY = Surface("dry", 10.0, 1.9, 1.0, 0.97)
WET = Surface("wet", 12.0, 2.3, 0.82, 1.0)
SNOW = Surface("snow", 5.0, 2.0, 0.3, 1.0)
ICE = Surface("ice", 4.0, 2.0, 0.1, 1.0)
SURFACES = {s.name: s for s in (DRY, WET, SNOW, ICE)}

# Sampling ranges of the post-drop mu (prereg V7).
MU_RANGES = {"dry": (0.85, 1.0), "wet": (0.5, 0.8), "snow": (0.20, 0.35), "ice": (0.08, 0.15)}


def param_names() -> list[str]:
    return [f.name for f in fields(VehicleParams)]
