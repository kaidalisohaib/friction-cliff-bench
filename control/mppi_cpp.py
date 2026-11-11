"""C++17/CUDA MPPI backend (mppi_cpp/, BASELINES.md §5): a drop-in for control.mppi.MPPI.

`CppMPPI` keeps every buffer of the Warp MPPI (plan U, samples V, costs, weights, ...) and only replaces `act()`
by one call into libfcb_mppi.so. The library reads the simulator's and the planner's Warp arrays in place (raw
device pointers, no copies) and launches on Warp's current stream, so it orders correctly with the Warp kernels
around it. It reproduces Warp's random number generator, so with the same seeds both backends draw the same samples
(S3 equivalence: tests/test_mppi_cpp.py).

Build once:  cmake -S mppi_cpp -B mppi_cpp/build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
             cmake --build mppi_cpp/build -j
Use:         MPPI_BACKEND=cpp (environment) makes control.controllers / control.adaptive build CppMPPI.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import warp as wp

from control.mppi import MPPI
from sim.params import BUF_LEN, CONTROL_DT, N_SUBSTEPS

LIB_PATH = Path(__file__).resolve().parent.parent / "mppi_cpp" / "build" / "libfcb_mppi.so"
_lib = None


class CostWC(ctypes.Structure):
    _fields_ = [(k, ctypes.c_float) for k in
                ("w_lane", "w_ey", "w_v", "w_du", "w_prog", "C_fail", "severe", "beta_term", "spin_term")]


_P = ctypes.c_void_p


class Args(ctypes.Structure):
    """Mirror of fcb::FcbMppiArgs (mppi_cpp/include/fcb/mppi_core.hpp); field order and types must match."""
    _fields_ = ([(k, ctypes.c_int32) for k in ("n", "K", "H", "nsub", "mode", "pd_sample", "n_iter", "t",
                                                "track_stride", "buf_len")]
                + [(k, ctypes.c_float) for k in ("sigma_steer", "sigma_long", "rho", "lam", "ess_target", "dt",
                                                  "ds")]
                + [("cw", CostWC)]
                + [(k, _P) for k in ("mparams", "state", "buf", "hints", "frenet", "sf_fix", "sr_fix", "prof",
                                     "base", "track_id", "track_x", "track_y", "track_th", "track_k", "track_n",
                                     "v_ref", "seeds", "U", "V", "costs", "xend", "w", "info", "Unew", "action")])


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not LIB_PATH.exists():
            raise FileNotFoundError(f"{LIB_PATH} not built; see control/mppi_cpp.py for the cmake commands")
        L = ctypes.CDLL(str(LIB_PATH))
        L.fcb_mppi_act_cuda.argtypes = [ctypes.POINTER(Args), ctypes.c_void_p]
        L.fcb_mppi_act_host.argtypes = [ctypes.POINTER(Args)]
        L.fcb_substep_host.argtypes = [_P, _P, ctypes.c_float, ctypes.c_float, _P, _P, ctypes.c_float, _P]
        L.fcb_randn_host.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _P]
        L.fcb_params_echo_host.argtypes = [_P, _P]
        L.fcb_last_error.restype = ctypes.c_char_p
        _lib = L
    return _lib


def available() -> bool:
    return LIB_PATH.exists()


def use_cpp() -> bool:
    """MPPI backend switch for the method factories (environment variable MPPI_BACKEND=cpp)."""
    return os.environ.get("MPPI_BACKEND", "warp").lower() == "cpp"


class CppMPPI(MPPI):
    """Same constructor, state and interface as control.mppi.MPPI; act() runs the C++/CUDA implementation."""

    def _args(self) -> Args:
        s, c = self.sim, self.cfg
        a = Args()
        a.n, a.K, a.H, a.nsub, a.mode = s.n, c.K, c.H, N_SUBSTEPS, int(c.mode)
        a.pd_sample, a.n_iter, a.t = int(c.pd_sample), int(c.n_iter), int(self.t)
        a.track_stride, a.buf_len = int(s.track_x.shape[1]), BUF_LEN
        a.sigma_steer, a.sigma_long, a.rho, a.lam = c.sigma_steer, c.sigma_long, c.rho, c.lam
        a.ess_target, a.dt, a.ds = c.ess_target, CONTROL_DT / N_SUBSTEPS, s.ds
        a.cw = CostWC(*(float(getattr(self.cw, k)) for k, _ in CostWC._fields_))
        for name, arr in (("mparams", self.mparams), ("state", s.state), ("buf", s.buf), ("hints", s.hints),
                          ("frenet", s.frenet), ("sf_fix", self.sf_fix), ("sr_fix", self.sr_fix), ("prof", s.prof),
                          ("base", s.base), ("track_id", s.track_id), ("track_x", s.track_x),
                          ("track_y", s.track_y), ("track_th", s.track_theta), ("track_k", s.track_kappa),
                          ("track_n", s.track_n), ("v_ref", self.v_ref), ("seeds", self.seeds), ("U", self.U),
                          ("V", self.V), ("costs", self.costs), ("xend", self.xend), ("w", self.w),
                          ("info", self.info), ("Unew", self.Unew), ("action", self.action)):
            assert arr.is_contiguous, name
            setattr(a, name, arr.ptr)
        return a

    def act(self) -> wp.array:
        a = self._args()
        L = lib()
        if self.sim.device == "cpu" or str(self.sim.device).startswith("cpu"):
            rc = L.fcb_mppi_act_host(ctypes.byref(a))
        else:
            stream = wp.get_stream(self.sim.device)
            rc = L.fcb_mppi_act_cuda(ctypes.byref(a), ctypes.c_void_p(stream.cuda_stream))
        if rc != 0:
            raise RuntimeError(f"fcb_mppi_act failed: {L.fcb_last_error().decode()}")
        self.t += 1
        return self.action


def make_mppi(*args, **kw) -> MPPI:
    """MPPI factory honouring the backend switch."""
    return (CppMPPI if use_cpp() else MPPI)(*args, **kw)
