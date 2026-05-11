"""
Importable terminal ProNav simulation module.

This module is designed to be called from a higher-level mission script after
Dymos has produced a terminal handoff state.  It contains no top-level mission
execution, so importing it will not run a simulation or create plots.

State convention for the PN simulation
--------------------------------------
The internal PN state vector is flat-earth / local Cartesian:

    y = [x_m, y_m, z_m, v, gamma, psi, x_t, y_t, z_t]

where:
    x_m, y_m, z_m : vehicle position [m]
    v             : vehicle speed [m/s]
    gamma         : flight-path angle [rad]
    psi           : heading angle [rad]
    x_t, y_t, z_t : target position [m]

This is intentionally separate from the Dymos spherical state convention
(theta, phi, h).  Helper functions are provided for converting a Dymos-style
state dictionary into local Cartesian coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.integrate import solve_ivp


EARTH_RADIUS_M = 6_371_000.0


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a unit vector, or zeros if the norm is too small."""
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


@dataclass
class ProNavConfig:
    """Configuration for the terminal aerodynamic ProNav model."""

    # Guidance / target motion
    N: float = 3.0
    target_speed: float = 0.0
    target_heading_deg: float = 0.0

    # Vehicle / environment
    gravity: float = 9.81
    mass: float = 340.1943
    area_ref: float = 0.2919

    # Aero coefficients matching the simplified Dymos aero model
    cl_alpha: float = 1.5658
    cd_alpha2: float = 1.6537
    cd0: float = 0.0612

    # Control limits
    alpha_max_deg: float = 30.0
    sigma_max_deg: float = 180.0

    # Atmosphere
    density0: float = 1.225
    scale_height: float = 7500.0

    # Integration / event settings
    intercept_radius_m: float = 5.0
    t_final: float = 30.0
    num_time_points: int = 1000
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: Optional[float] = None
    dense_output: bool = True


@dataclass
class ProNavInitialState:
    """Initial state for the terminal PN simulation in local Cartesian coordinates."""

    x: float
    y: float
    z: float
    v: float
    gamma: float
    psi: float


@dataclass
class ProNavTargetState:
    """Initial target position for the terminal PN simulation in local Cartesian coordinates."""

    x: float
    y: float
    z: float = 0.0


class ProNav3DAeroScenario:
    """
    Terminal 3D PN model with simplified aerodynamic coupling.

    Guidance:
      - PN computes a desired normal acceleration vector in inertial coordinates.
      - The command is projected onto flight-path basis directions e_gamma and e_psi.
      - Angle of attack alpha is chosen to generate the required total lift magnitude.
      - Bank angle sigma splits that lift between gamma and psi channels.

    Aerodynamics:
      CL = cl_alpha * alpha
      CD = cd_alpha2 * alpha**2 + cd0
    """

    def __init__(self, cfg: ProNavConfig):
        self.cfg = cfg
        self.N = cfg.N
        self.gravity = cfg.gravity
        self.mass = cfg.mass
        self.area_ref = cfg.area_ref
        self.cl_alpha = cfg.cl_alpha
        self.cd_alpha2 = cfg.cd_alpha2
        self.cd0 = cfg.cd0
        self.alpha_max = np.radians(cfg.alpha_max_deg)
        self.sigma_max = np.radians(cfg.sigma_max_deg)
        self.density0 = cfg.density0
        self.scale_height = cfg.scale_height
        self.intercept_radius_m = cfg.intercept_radius_m

        target_heading = np.radians(cfg.target_heading_deg)
        self.v_t = np.array([
            cfg.target_speed * np.cos(target_heading),
            cfg.target_speed * np.sin(target_heading),
            0.0,
        ])

    def atmosphere_density(self, z: float) -> float:
        z_eff = max(float(z), 0.0)
        return self.density0 * np.exp(-z_eff / self.scale_height)

    def aero_forces(self, alpha: float, speed: float, rho: float) -> tuple[float, float]:
        q = 0.5 * rho * speed**2
        cl = self.cl_alpha * alpha
        cd = self.cd_alpha2 * alpha**2 + self.cd0
        lift = q * self.area_ref * cl
        drag = q * self.area_ref * cd
        return lift, drag

    def command_and_aero(self, y: np.ndarray) -> dict[str, float | np.ndarray]:
        """Compute PN command, alpha/sigma, lift/drag, and diagnostic quantities."""
        x_m, y_m, z_m, v, gamma, psi = y[0:6]
        r_m = np.array([x_m, y_m, z_m])
        r_t = y[6:9]

        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)
        cos_p = np.cos(psi)
        sin_p = np.sin(psi)

        e_v = np.array([cos_g * cos_p, cos_g * sin_p, sin_g])
        e_gamma = np.array([-sin_g * cos_p, -sin_g * sin_p, cos_g])
        e_psi = np.array([-sin_p, cos_p, 0.0])
        v_m = max(v, 0.0) * e_v

        r_rel = r_t - r_m
        v_rel = self.v_t - v_m
        r_norm = np.linalg.norm(r_rel)

        if r_norm < 1.0 or v < 1.0:
            a_pn = np.zeros(3)
            Vc = 0.0
            los_rate_vec = np.zeros(3)
        else:
            r_hat = r_rel / r_norm
            los_rate_vec = np.cross(r_rel, v_rel) / (r_norm**2)
            Vc = -np.dot(r_hat, v_rel)
            a_pn = self.N * Vc * np.cross(los_rate_vec, e_v)
            # Force purely normal-to-velocity command.
            a_pn = a_pn - np.dot(a_pn, e_v) * e_v

        a_gamma_cmd = float(np.dot(a_pn, e_gamma))
        a_psi_cmd = float(np.dot(a_pn, e_psi))
        a_lift_req = float(np.sqrt(a_gamma_cmd**2 + a_psi_cmd**2))

        rho = self.atmosphere_density(z_m)
        qS = 0.5 * rho * max(v, 1.0)**2 * self.area_ref
        if qS > 1e-9:
            alpha_unsat = (self.mass * a_lift_req) / (qS * self.cl_alpha)
        else:
            alpha_unsat = 0.0
        alpha_cmd = float(np.clip(alpha_unsat, 0.0, self.alpha_max))

        lift, drag = self.aero_forces(alpha_cmd, max(v, 1.0), rho)
        a_lift_avail = lift / self.mass

        if a_lift_req < 1e-9 or a_lift_avail < 1e-9:
            sigma_unsat = 0.0
            sigma_cmd = 0.0
        else:
            sigma_unsat = float(np.arctan2(a_psi_cmd, a_gamma_cmd))
            sigma_cmd = float(np.clip(sigma_unsat, -self.sigma_max, self.sigma_max))

        return {
            "r_norm": float(r_norm),
            "Vc": float(Vc),
            "los_rate_vec": los_rate_vec,
            "los_rate": float(np.linalg.norm(los_rate_vec)),
            "a_pn": a_pn,
            "a_gamma_cmd": a_gamma_cmd,
            "a_psi_cmd": a_psi_cmd,
            "a_lift_req": a_lift_req,
            "rho": float(rho),
            "alpha_cmd": alpha_cmd,
            "alpha_unsat": float(alpha_unsat),
            "sigma_cmd": sigma_cmd,
            "sigma_unsat": float(sigma_unsat),
            "lift": float(lift),
            "drag": float(drag),
            "a_lift_avail": float(a_lift_avail),
            "e_v": e_v,
            "e_gamma": e_gamma,
            "e_psi": e_psi,
        }

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        x_m, y_m, z_m, v, gamma, psi = y[0:6]
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)
        cos_p = np.cos(psi)
        sin_p = np.sin(psi)

        cmd = self.command_and_aero(y)
        lift = float(cmd["lift"])
        drag = float(cmd["drag"])
        sigma_cmd = float(cmd["sigma_cmd"])
        cos_sigma = np.cos(sigma_cmd)
        sin_sigma = np.sin(sigma_cmd)

        xdot = v * cos_g * cos_p
        ydot = v * cos_g * sin_p
        zdot = v * sin_g

        vdot = -(drag / self.mass) - self.gravity * sin_g

        if abs(v) < 1e-6:
            gammadot = 0.0
            psidot = 0.0
        else:
            gammadot = (lift * cos_sigma) / (self.mass * v) - (self.gravity * cos_g) / v
            # Avoid heading-rate singularity near vertical flight.
            cos_g_safe = cos_g
            if abs(cos_g_safe) < 0.05:
                cos_g_safe = 0.05 * np.sign(cos_g_safe) if cos_g_safe != 0 else 0.05
            psidot = (lift * sin_sigma) / (self.mass * v * cos_g_safe)

        xt_dot, yt_dot, zt_dot = self.v_t
        return np.array([xdot, ydot, zdot, vdot, gammadot, psidot, xt_dot, yt_dot, zt_dot])

    def intercept_event(self, t: float, y: np.ndarray) -> float:
        r_m = y[0:3]
        r_t = y[6:9]
        return np.linalg.norm(r_t - r_m) - self.intercept_radius_m

    intercept_event.terminal = True
    intercept_event.direction = -1

    def ground_event(self, t: float, y: np.ndarray) -> float:
        return y[2]

    ground_event.terminal = True
    ground_event.direction = -1

    def compute_diagnostics(self, sol) -> dict[str, np.ndarray]:
        t_arr = sol.t
        n = len(t_arr)

        rho_hist = np.zeros(n)
        range_hist = np.zeros(n)
        alpha_hist = np.zeros(n)
        sigma_hist = np.zeros(n)
        lift_hist = np.zeros(n)
        drag_hist = np.zeros(n)
        a_gamma_cmd_hist = np.zeros(n)
        a_psi_cmd_hist = np.zeros(n)
        a_lift_req_hist = np.zeros(n)
        a_lift_avail_hist = np.zeros(n)
        Vc_hist = np.zeros(n)
        los_rate_hist = np.zeros(n)
        alpha_sat_hist = np.zeros(n, dtype=bool)
        sigma_sat_hist = np.zeros(n, dtype=bool)

        for i in range(n):
            y = sol.y[:, i]
            cmd = self.command_and_aero(y)
            range_hist[i] = cmd["r_norm"]
            rho_hist[i] = cmd["rho"]
            alpha_hist[i] = cmd["alpha_cmd"]
            sigma_hist[i] = cmd["sigma_cmd"]
            lift_hist[i] = cmd["lift"]
            drag_hist[i] = cmd["drag"]
            a_gamma_cmd_hist[i] = cmd["a_gamma_cmd"]
            a_psi_cmd_hist[i] = cmd["a_psi_cmd"]
            a_lift_req_hist[i] = cmd["a_lift_req"]
            a_lift_avail_hist[i] = cmd["a_lift_avail"]
            Vc_hist[i] = cmd["Vc"]
            los_rate_hist[i] = cmd["los_rate"]
            alpha_sat_hist[i] = abs(cmd["alpha_cmd"] - cmd["alpha_unsat"]) > 1e-12
            sigma_sat_hist[i] = abs(cmd["sigma_cmd"] - cmd["sigma_unsat"]) > 1e-12

        return {
            "time": t_arr,
            "range": range_hist,
            "rho": rho_hist,
            "alpha_cmd": alpha_hist,
            "sigma_cmd": sigma_hist,
            "lift": lift_hist,
            "drag": drag_hist,
            "a_gamma_cmd": a_gamma_cmd_hist,
            "a_psi_cmd": a_psi_cmd_hist,
            "a_lift_req": a_lift_req_hist,
            "a_lift_avail": a_lift_avail_hist,
            "Vc": Vc_hist,
            "los_rate": los_rate_hist,
            "gamma": sol.y[4],
            "psi": sol.y[5],
            "z": sol.y[2],
            "v": sol.y[3],
            "alpha_sat": alpha_sat_hist,
            "sigma_sat": sigma_sat_hist,
        }


def make_initial_vector(vehicle: ProNavInitialState, target: ProNavTargetState) -> np.ndarray:
    return np.array([vehicle.x, vehicle.y, vehicle.z, vehicle.v, vehicle.gamma, vehicle.psi,
                     target.x, target.y, target.z], dtype=float)


def dymos_state_to_local_cartesian(
    state: dict[str, float],
    target_theta: float,
    target_phi: float,
    reference_theta: float = 0.0,
    reference_phi: float = 0.0,
    earth_radius: float = EARTH_RADIUS_M,
) -> tuple[ProNavInitialState, ProNavTargetState]:
    """
    Convert a Dymos-style spherical state dictionary to the local Cartesian PN convention.

    Expected state keys:
        theta [rad], phi [rad], h [m], v [m/s], gamma [rad], psi [rad]

    The local flat-earth coordinates are approximate:
        x = Re*cos(reference_phi)*(theta - reference_theta)
        y = Re*(phi - reference_phi)
        z = h
    """
    x_vehicle = earth_radius * np.cos(reference_phi) * (state["theta"] - reference_theta)
    y_vehicle = earth_radius * (state["phi"] - reference_phi)
    z_vehicle = state["h"]

    x_target = earth_radius * np.cos(reference_phi) * (target_theta - reference_theta)
    y_target = earth_radius * (target_phi - reference_phi)

    vehicle = ProNavInitialState(
        x=float(x_vehicle),
        y=float(y_vehicle),
        z=float(z_vehicle),
        v=float(state["v"]),
        gamma=float(state["gamma"]),
        psi=float(state["psi"]),
    )
    target = ProNavTargetState(x=float(x_target), y=float(y_target), z=0.0)
    return vehicle, target


def _event_state(sol, event_index: int) -> Optional[np.ndarray]:
    if sol.t_events[event_index].size == 0:
        return None
    if sol.sol is None:
        return None
    return sol.sol(sol.t_events[event_index][0])


def run_pronav_terminal(
    vehicle: ProNavInitialState,
    target: ProNavTargetState,
    cfg: Optional[ProNavConfig] = None,
    label: str = "PN terminal",
) -> dict[str, Any]:
    """
    Run one terminal ProNav simulation and return a mission-stage dictionary.

    This is the main function intended to be called from a larger mission script.
    """
    cfg = cfg or ProNavConfig()
    scenario = ProNav3DAeroScenario(cfg)
    y0 = make_initial_vector(vehicle, target)

    t_eval = np.linspace(0.0, cfg.t_final, cfg.num_time_points)
    kwargs = dict(
        fun=scenario.dynamics,
        t_span=(0.0, cfg.t_final),
        y0=y0,
        t_eval=t_eval,
        dense_output=cfg.dense_output,
        events=[scenario.intercept_event, scenario.ground_event],
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if cfg.max_step is not None:
        kwargs["max_step"] = cfg.max_step

    sol = solve_ivp(**kwargs)
    diag = scenario.compute_diagnostics(sol)

    # Determine terminal/event status and exact final state if available.
    status = "time_limit"
    event_time = None
    y_event = None
    if sol.t_events[0].size > 0:
        status = "intercept"
        event_time = float(sol.t_events[0][0])
        y_event = _event_state(sol, 0)
    elif sol.t_events[1].size > 0:
        status = "ground_impact"
        event_time = float(sol.t_events[1][0])
        y_event = _event_state(sol, 1)

    if y_event is not None:
        final_state = y_event
    else:
        final_state = sol.y[:, -1]

    final_sep = float(np.linalg.norm(final_state[6:9] - final_state[0:3]))

    # Store sampled arrays.  The exact event endpoint is provided separately so
    # plotting code can append it if desired.
    return {
        "label": label,
        "status": status,
        "success": status == "intercept",
        "event_time": event_time,
        "final_separation": final_sep,
        "time": sol.t,
        "x": sol.y[0],
        "y": sol.y[1],
        "z": sol.y[2],
        "v": sol.y[3],
        "gamma": sol.y[4],
        "psi": sol.y[5],
        "target_x": sol.y[6],
        "target_y": sol.y[7],
        "target_z": sol.y[8],
        "t_final": float(sol.t[-1]),
        "sol": sol,
        "scenario": scenario,
        "diagnostics": diag,
        "event_state": y_event,
    }


def append_event_endpoint(result: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Return plotting arrays with the exact event endpoint appended when available.
    """
    x = np.asarray(result["x"]).copy()
    y = np.asarray(result["y"]).copy()
    z = np.asarray(result["z"]).copy()
    xt = np.asarray(result["target_x"]).copy()
    yt = np.asarray(result["target_y"]).copy()
    zt = np.asarray(result["target_z"]).copy()

    ev = result.get("event_state")
    if ev is not None:
        x = np.append(x, ev[0])
        y = np.append(y, ev[1])
        z = np.append(z, ev[2])
        xt = np.append(xt, ev[6])
        yt = np.append(yt, ev[7])
        zt = np.append(zt, ev[8])

    return {"x": x, "y": y, "z": z, "target_x": xt, "target_y": yt, "target_z": zt}
