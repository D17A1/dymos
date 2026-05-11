"""
Post-processing utilities for the three-stage Dymos -> ProNav mission workflow.

This module contains no top-level mission execution.  It provides helpers to:
  * read Dymos recorder folders into the same dictionary format returned by dymos_runner
  * interpolate Dymos states at arbitrary times
  * convert ProNav local Cartesian output back to Dymos-style theta/phi/h arrays
  * stitch stage 1 + stage 2 + PN terminal into one continuous result
  * plot trajectories, controls, states, and PN diagnostics
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio


def _require_openmdao():
    import openmdao.api as om  # local import so plotting utilities do not require OM unless loading dbs
    return om


def find_dymos_db(folder: str | Path, db_kind: str = "simulation") -> Path:
    """Find a Dymos recorder database in a run folder.

    Parameters
    ----------
    folder : str | Path
        OpenMDAO/Dymos output directory.
    db_kind : {'simulation', 'solution'}
        'simulation' finds dymos_simulation.db recursively; 'solution' finds dymos_solution.db.
    """
    folder = Path(folder)
    if db_kind == "simulation":
        candidates = [folder / "dymos_simulation.db"] + list(folder.glob("**/dymos_simulation.db"))
    elif db_kind == "solution":
        candidates = [folder / "dymos_solution.db"] + list(folder.glob("**/dymos_solution.db"))
    else:
        raise ValueError("db_kind must be 'simulation' or 'solution'")

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {db_kind} Dymos database under {folder}")


def load_dymos_case_from_folder(folder: str | Path, label: Optional[str] = None,
                                db_kind: str = "simulation") -> Dict[str, Any]:
    """Load one Dymos result folder into the standard stage dictionary format."""
    om = _require_openmdao()
    db = find_dymos_db(folder, db_kind=db_kind)
    case = om.CaseReader(str(db)).get_case("final")
    prefix = "traj.phase0.timeseries."

    def get(var: str) -> np.ndarray:
        return case.get_val(prefix + var).ravel()

    time = get("time")
    theta_rad = get("theta")
    phi_rad = get("phi")
    gamma_rad = get("gamma")
    psi_rad = get("psi")

    out = {
        "label": label or Path(folder).name,
        "source_folder": str(folder),
        "db_path": str(db),
        "time": time,
        "theta_rad": theta_rad,
        "phi_rad": phi_rad,
        "theta": np.degrees(theta_rad),
        "phi": np.degrees(phi_rad),
        "h": get("h"),
        "v": get("v"),
        "gamma": gamma_rad,
        "psi": psi_rad,
        "gamma_deg": np.degrees(gamma_rad),
        "psi_deg": np.degrees(psi_rad),
        "t_final": float(time[-1]),
        "sim": case,
    }
    return out


def load_dymos_stages_from_folders(folder_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Load named Dymos stages from a YAML-style mapping.

    Expected keys are usually A1, B1, A2, B2, but this function accepts any mapping.
    """
    return {key: load_dymos_case_from_folder(path, label=key) for key, path in folder_map.items()}


def interpolate_stage_state(stage: Dict[str, Any], t_query: float) -> Dict[str, float]:
    """Interpolate a stage dictionary at local stage time t_query.

    Returns angles in radians, matching dymos_runner.get_state_at_time.
    """
    t = np.asarray(stage["time"]).ravel()
    if t_query < t[0] or t_query > t[-1]:
        raise ValueError(f"Requested t={t_query:.3f} outside stage range [{t[0]:.3f}, {t[-1]:.3f}]")

    def interp(key: str) -> float:
        return float(np.interp(t_query, t, np.asarray(stage[key]).ravel()))

    theta_key = "theta_rad" if "theta_rad" in stage else "theta"
    phi_key = "phi_rad" if "phi_rad" in stage else "phi"
    theta = interp(theta_key)
    phi = interp(phi_key)

    # If only degree arrays are available, convert them.
    if theta_key == "theta":
        theta = np.radians(theta)
    if phi_key == "phi":
        phi = np.radians(phi)

    return {
        "h": interp("h"),
        "theta": theta,
        "phi": phi,
        "v": interp("v"),
        "gamma": interp("gamma"),
        "psi": interp("psi"),
    }


def trim_stage(stage: Dict[str, Any], t_end: float) -> Dict[str, Any]:
    """Return a copy of a Dymos stage truncated to t <= t_end."""
    t = np.asarray(stage["time"])
    mask = t <= t_end
    if not np.any(mask):
        raise ValueError("trim_stage removed all samples; check t_end")
    out = dict(stage)
    for key in ["time", "theta", "phi", "theta_rad", "phi_rad", "h", "v", "gamma", "psi", "gamma_deg", "psi_deg"]:
        if key in stage:
            out[key] = np.asarray(stage[key])[mask]
    out["t_final"] = float(out["time"][-1])
    return out


def stitch_two_dymos_stages(stage1: Dict[str, Any], stage2: Dict[str, Any],
                            t_reveal: float, label: str) -> Dict[str, Any]:
    """Stitch stage 1 up to t_reveal with all of stage 2 shifted by t_reveal."""
    t1 = np.asarray(stage1["time"])
    m1 = t1 <= t_reveal

    out = {"label": label}
    out["time"] = np.concatenate([stage1["time"][m1], stage2["time"] + t_reveal])

    for key in ["theta", "phi", "theta_rad", "phi_rad", "h", "v", "gamma", "psi", "gamma_deg", "psi_deg"]:
        if key in stage1 and key in stage2:
            out[key] = np.concatenate([np.asarray(stage1[key])[m1], np.asarray(stage2[key])])

    out["t_final"] = float(t_reveal + stage2["t_final"])
    out["stage1"] = stage1
    out["stage2"] = stage2
    return out


def pronav_to_dymos_like(pn_result: Dict[str, Any], reference_theta: float, reference_phi: float,
                         earth_radius: float = 6_371_000.0) -> Dict[str, Any]:
    """Convert PN local Cartesian arrays back to theta/phi/h-style arrays for plotting."""
    x = np.asarray(pn_result["x"])
    y = np.asarray(pn_result["y"])
    z = np.asarray(pn_result["z"])
    tx = np.asarray(pn_result["target_x"])
    ty = np.asarray(pn_result["target_y"])
    tz = np.asarray(pn_result["target_z"])

    cos_ref = np.cos(reference_phi)
    denom = earth_radius * cos_ref if abs(cos_ref) > 1e-12 else earth_radius

    theta_rad = reference_theta + x / denom
    phi_rad = reference_phi + y / earth_radius
    target_theta_rad = reference_theta + tx / denom
    target_phi_rad = reference_phi + ty / earth_radius

    out = {
        "label": pn_result.get("label", "PN terminal"),
        "status": pn_result.get("status"),
        "success": pn_result.get("success"),
        "time": np.asarray(pn_result["time"]),
        "theta_rad": theta_rad,
        "phi_rad": phi_rad,
        "theta": np.degrees(theta_rad),
        "phi": np.degrees(phi_rad),
        "h": z,
        "v": np.asarray(pn_result["v"]),
        "gamma": np.asarray(pn_result["gamma"]),
        "psi": np.asarray(pn_result["psi"]),
        "gamma_deg": np.degrees(np.asarray(pn_result["gamma"])),
        "psi_deg": np.degrees(np.asarray(pn_result["psi"])),
        "target_theta_rad": target_theta_rad,
        "target_phi_rad": target_phi_rad,
        "target_theta": np.degrees(target_theta_rad),
        "target_phi": np.degrees(target_phi_rad),
        "target_h": tz,
        "x": x,
        "y": y,
        "z": z,
        "target_x": tx,
        "target_y": ty,
        "target_z": tz,
        "t_final": float(np.asarray(pn_result["time"])[-1]),
        "raw": pn_result,
    }
    return out


def stitch_three_stage(stage1: Dict[str, Any], stage2: Dict[str, Any], pn_stage: Dict[str, Any],
                       t_reveal: float, t_pn_start_mission: float, label: str) -> Dict[str, Any]:
    """Stitch Dymos stage 1, Dymos stage 2 up to PN handoff, and PN terminal."""
    t1 = np.asarray(stage1["time"])
    m1 = t1 <= t_reveal

    stage2_handoff_local = t_pn_start_mission - t_reveal
    stage2_trim = trim_stage(stage2, stage2_handoff_local)

    out = {"label": label}
    out["time"] = np.concatenate([
        np.asarray(stage1["time"])[m1],
        np.asarray(stage2_trim["time"]) + t_reveal,
        np.asarray(pn_stage["time"]) + t_pn_start_mission,
    ])

    for key in ["theta", "phi", "theta_rad", "phi_rad", "h", "v", "gamma", "psi", "gamma_deg", "psi_deg"]:
        pieces = []
        if key in stage1:
            pieces.append(np.asarray(stage1[key])[m1])
        if key in stage2_trim:
            pieces.append(np.asarray(stage2_trim[key]))
        if key in pn_stage:
            pieces.append(np.asarray(pn_stage[key]))
        if len(pieces) == 3:
            out[key] = np.concatenate(pieces)

    out["t_final"] = float(out["time"][-1])
    out["stage1"] = stage1
    out["stage2"] = stage2_trim
    out["pn"] = pn_stage
    out["t_reveal"] = t_reveal
    out["t_pn_start"] = t_pn_start_mission
    return out


def _get_control_timeseries(stage: Dict[str, Any], var: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Read a control timeseries from a Dymos stage dictionary, if available."""
    sim = stage.get("sim")
    if sim is None:
        return None
    try:
        t = sim.get_val("traj.phase0.timeseries.time").ravel()
        y = sim.get_val(f"traj.phase0.timeseries.{var}").ravel()
        return t, y
    except Exception:
        return None


def stitched_dymos_control(stage1: Dict[str, Any], stage2: Dict[str, Any], var: str,
                           t_reveal: float, t_pn_start: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Stitch a Dymos control for stage1 and stage2 up to PN handoff."""
    ts1 = _get_control_timeseries(stage1, var)
    ts2 = _get_control_timeseries(stage2, var)
    if ts1 is None or ts2 is None:
        return None
    t1, y1 = ts1
    t2, y2 = ts2
    m1 = t1 <= t_reveal
    stage2_handoff_local = t_pn_start - t_reveal
    m2 = t2 <= stage2_handoff_local
    return np.concatenate([t1[m1], t2[m2] + t_reveal]), np.concatenate([y1[m1], y2[m2]])


def plot_three_stage_results(vehicle_results: Dict[str, Dict[str, Any]],
                             target_guess: tuple[float, float],
                             target_true: tuple[float, float],
                             output_prefix: Optional[str] = None,
                             show: bool = True,
                             browser_3d: bool = True) -> None:
    """Generate trajectory, state, control, and PN diagnostic plots.

    vehicle_results maps names like 'A' and 'B' to stitched three-stage dictionaries.
    """
    # Top-down ground track
    plt.figure(figsize=(8, 6))
    for name, res in vehicle_results.items():
        plt.plot(res["theta"], res["phi"], label=res.get("label", name))
    plt.scatter([np.degrees(target_guess[0])], [np.degrees(target_guess[1])], marker="o", s=100, label="Target guess")
    plt.scatter([np.degrees(target_true[0])], [np.degrees(target_true[1])], marker="*", s=180, label="Updated target")
    plt.xlabel("Downrange theta [deg]")
    plt.ylabel("Crossrange phi [deg]")
    plt.title("Ground Track")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_ground_track.png", dpi=200)

    # Altitude and velocity vs time
    plt.figure(figsize=(9, 5))
    for name, res in vehicle_results.items():
        plt.plot(res["time"], res["h"], label=res.get("label", name))
    plt.xlabel("Mission time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Altitude vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_altitude.png", dpi=200)

    plt.figure(figsize=(9, 5))
    for name, res in vehicle_results.items():
        plt.plot(res["time"], res["v"], label=res.get("label", name))
    plt.xlabel("Mission time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Velocity vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_velocity.png", dpi=200)

    # Angles vs time
    plt.figure(figsize=(9, 5))
    for name, res in vehicle_results.items():
        plt.plot(res["time"], np.degrees(res["gamma"]), label=f"{name} gamma")
        plt.plot(res["time"], np.degrees(res["psi"]), "--", label=f"{name} psi")
    plt.xlabel("Mission time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Flight Path and Heading")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_angles.png", dpi=200)

    # Dymos controls up to PN start; PN controls from diagnostics if present.
    plt.figure(figsize=(9, 5))
    for name, res in vehicle_results.items():
        ctrl = stitched_dymos_control(res["stage1"], res["stage2"], "alpha", res["t_reveal"], res["t_pn_start"])
        if ctrl:
            t, y = ctrl
            plt.plot(t, np.degrees(y), label=f"{name} Dymos alpha")
        pn_diag = res.get("pn", {}).get("raw", {}).get("diagnostics")
        if pn_diag is not None:
            plt.plot(pn_diag["time"] + res["t_pn_start"], np.degrees(pn_diag["alpha_cmd"]), "--", label=f"{name} PN alpha")
    plt.xlabel("Mission time [s]")
    plt.ylabel("Alpha [deg]")
    plt.title("Angle of Attack")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_alpha.png", dpi=200)

    plt.figure(figsize=(9, 5))
    for name, res in vehicle_results.items():
        ctrl = stitched_dymos_control(res["stage1"], res["stage2"], "sigma", res["t_reveal"], res["t_pn_start"])
        if ctrl:
            t, y = ctrl
            plt.plot(t, np.degrees(y), label=f"{name} Dymos sigma")
        pn_diag = res.get("pn", {}).get("raw", {}).get("diagnostics")
        if pn_diag is not None:
            plt.plot(pn_diag["time"] + res["t_pn_start"], np.degrees(np.unwrap(pn_diag["sigma_cmd"])), "--", label=f"{name} PN sigma")
    plt.xlabel("Mission time [s]")
    plt.ylabel("Sigma [deg]")
    plt.title("Bank Angle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_sigma.png", dpi=200)

    # PN diagnostics: range and lift budget.
    plt.figure(figsize=(9, 5))
    for name, res in vehicle_results.items():
        pn_diag = res.get("pn", {}).get("raw", {}).get("diagnostics")
        if pn_diag is not None:
            plt.plot(pn_diag["time"] + res["t_pn_start"], pn_diag["range"], label=f"{name} PN range")
    plt.xlabel("Mission time [s]")
    plt.ylabel("Range to target [m]")
    plt.title("PN Terminal Closing Range")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_pn_range.png", dpi=200)

    # 3D plotly
    fig = go.Figure()
    for name, res in vehicle_results.items():
        fig.add_trace(go.Scatter3d(x=res["theta"], y=res["phi"], z=res["h"], mode="lines", name=res.get("label", name)))
    fig.add_trace(go.Scatter3d(x=[np.degrees(target_guess[0])], y=[np.degrees(target_guess[1])], z=[0.0], mode="markers", marker=dict(size=7), name="Target guess"))
    fig.add_trace(go.Scatter3d(x=[np.degrees(target_true[0])], y=[np.degrees(target_true[1])], z=[0.0], mode="markers", marker=dict(size=9, symbol="diamond"), name="Updated target"))
    fig.update_layout(scene=dict(xaxis_title="Downrange theta [deg]", yaxis_title="Crossrange phi [deg]", zaxis_title="Altitude [m]"),
                      title="Three-stage trajectory overlay", margin=dict(l=0, r=0, b=0, t=40))
    if output_prefix:
        fig.write_html(f"{output_prefix}_3d.html")
    if browser_3d:
        pio.renderers.default = "browser"
        fig.show()

    if show:
        plt.show()
    else:
        plt.close("all")
