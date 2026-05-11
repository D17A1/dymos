"""
Read and plot previously generated Dymos/OpenMDAO recorder folders.

Usage examples
--------------
Read all output folders matching the default pattern in the current directory:
    python read_dymos_results.py

Read specific folders:
    python read_dymos_results.py --folders main_dymos_mission_example_out main_dymos_mission_example4_out

Read folders matching a custom glob:
    python read_dymos_results.py --glob "*_out"

Use solution database instead of simulation database:
    python read_dymos_results.py --db solution

Save figures instead of only showing them:
    python read_dymos_results.py --save-prefix saved_mission

Notes
-----
For plotting and PN handoff, the simulation database is usually preferred:
    dymos_simulation.db
because it is the forward-integrated trajectory from the optimized controls.
In many OpenMDAO/Dymos runs this lives under:
    <run>_out/traj_simulation_0_out/dymos_simulation.db

The solution database:
    dymos_solution.db
contains collocation/transcription nodes from the optimizer.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

try:
    import openmdao.api as om
except ImportError as exc:
    raise SystemExit(
        "This script requires OpenMDAO. Run it inside the same environment used for Dymos/OpenMDAO."
    ) from exc

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


TIMESERIES_PREFIX = "traj.phase0.timeseries."


def _get(case, name: str, prefix: str = TIMESERIES_PREFIX) -> np.ndarray:
    """Safely fetch and flatten a Dymos timeseries variable."""
    return np.asarray(case.get_val(prefix + name)).ravel()


def _try_get(case, name: str, prefix: str = TIMESERIES_PREFIX) -> np.ndarray | None:
    """Fetch a timeseries variable, returning None if it does not exist."""
    try:
        return _get(case, name, prefix=prefix)
    except Exception:
        return None


def read_dymos_folder(folder: str | Path, db_kind: str = "simulation") -> dict:
    """
    Read one Dymos/OpenMDAO output folder.

    Parameters
    ----------
    folder
        Folder containing dymos_simulation.db or dymos_solution.db.
    db_kind
        "simulation" or "solution".

    Returns
    -------
    dict
        Standard trajectory dictionary with arrays in SI/radians and degrees fields.
    """
    folder = Path(folder)
    if db_kind not in {"simulation", "solution"}:
        raise ValueError("db_kind must be 'simulation' or 'solution'")

    db_name = "dymos_simulation.db" if db_kind == "simulation" else "dymos_solution.db"

    # Solution DB is usually at <run>_out/dymos_solution.db, while simulation DB
    # is often nested at <run>_out/traj_simulation_0_out/dymos_simulation.db.
    # Try direct path first, then search recursively.
    db_path = folder / db_name
    if not db_path.exists():
        matches = sorted(folder.rglob(db_name))
        if not matches:
            raise FileNotFoundError(
                f"Could not find {db_name} directly in {folder} or recursively below it."
            )
        db_path = matches[0]

    cr = om.CaseReader(str(db_path))
    case = cr.get_case("final")

    time = _get(case, "time")
    theta = _get(case, "theta")
    phi = _get(case, "phi")
    h = _get(case, "h")
    v = _try_get(case, "v")
    gamma = _try_get(case, "gamma")
    psi = _try_get(case, "psi")
    alpha = _try_get(case, "alpha")
    sigma = _try_get(case, "sigma")

    out = {
        "label": folder.name,
        "folder": str(folder),
        "db_path": str(db_path),
        "db_kind": db_kind,
        "time": time,
        "theta": theta,
        "phi": phi,
        "h": h,
        "theta_deg": np.degrees(theta),
        "phi_deg": np.degrees(phi),
        "t_final": float(time[-1]),
        "case": case,
    }

    optional = {
        "v": v,
        "gamma": gamma,
        "psi": psi,
        "alpha": alpha,
        "sigma": sigma,
    }
    for key, val in optional.items():
        if val is not None:
            out[key] = val
            if key in {"gamma", "psi", "alpha", "sigma"}:
                out[key + "_deg"] = np.degrees(val)

    return out


def read_many(folders: Iterable[str | Path], db_kind: str = "simulation") -> list[dict]:
    """Read multiple Dymos output folders."""
    runs = []
    for folder in folders:
        try:
            runs.append(read_dymos_folder(folder, db_kind=db_kind))
        except Exception as exc:
            print(f"[skip] {folder}: {exc}")
    return runs


def print_summary(runs: list[dict]) -> None:
    """Print a compact summary of the loaded runs."""
    if not runs:
        print("No runs loaded.")
        return

    print("\nLoaded Dymos runs")
    print("=" * 80)
    for i, run in enumerate(runs):
        print(f"[{i}] {run['label']}")
        print(f"    db          : {run['db_path']}")
        print(f"    final time  : {run['t_final']:.6f} s")
        print(f"    final theta : {run['theta_deg'][-1]:.6f} deg")
        print(f"    final phi   : {run['phi_deg'][-1]:.6f} deg")
        print(f"    final h     : {run['h'][-1]:.6f} m")
        if "v" in run:
            print(f"    final v     : {run['v'][-1]:.6f} m/s")
        if "gamma_deg" in run:
            print(f"    final gamma : {run['gamma_deg'][-1]:.6f} deg")
        if "psi_deg" in run:
            print(f"    final psi   : {run['psi_deg'][-1]:.6f} deg")
    print("=" * 80)


def plot_runs_matplotlib(runs: list[dict], save_prefix: str | None = None) -> None:
    """Make standard 2D diagnostic plots."""
    if not runs:
        return

    # Ground track
    plt.figure(figsize=(8, 6))
    for run in runs:
        plt.plot(run["theta_deg"], run["phi_deg"], label=run["label"])
    plt.xlabel("Downrange theta [deg]")
    plt.ylabel("Crossrange phi [deg]")
    plt.title("Ground Track")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_ground_track.png", dpi=200)

    # Altitude vs downrange
    plt.figure(figsize=(8, 5))
    for run in runs:
        plt.plot(run["theta_deg"], run["h"], label=run["label"])
    plt.xlabel("Downrange theta [deg]")
    plt.ylabel("Altitude h [m]")
    plt.title("Altitude vs Downrange")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_altitude_downrange.png", dpi=200)

    # Velocity vs time, if available
    if any("v" in run for run in runs):
        plt.figure(figsize=(8, 5))
        for run in runs:
            if "v" in run:
                plt.plot(run["time"], run["v"], label=run["label"])
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title("Velocity vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_velocity.png", dpi=200)

    # Controls, if available
    if any("alpha_deg" in run for run in runs):
        plt.figure(figsize=(8, 5))
        for run in runs:
            if "alpha_deg" in run:
                plt.plot(run["time"], run["alpha_deg"], label=run["label"])
        plt.xlabel("Time [s]")
        plt.ylabel("Angle of attack alpha [deg]")
        plt.title("Control History: Alpha")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_alpha.png", dpi=200)

    if any("sigma_deg" in run for run in runs):
        plt.figure(figsize=(8, 5))
        for run in runs:
            if "sigma_deg" in run:
                plt.plot(run["time"], run["sigma_deg"], label=run["label"])
        plt.xlabel("Time [s]")
        plt.ylabel("Bank angle sigma [deg]")
        plt.title("Control History: Sigma")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_sigma.png", dpi=200)


def plot_runs_plotly(runs: list[dict], save_html: str | None = None) -> None:
    """Make an interactive 3D plot if Plotly is installed."""
    if go is None or not runs:
        return

    fig = go.Figure()
    for run in runs:
        fig.add_trace(go.Scatter3d(
            x=run["theta_deg"],
            y=run["phi_deg"],
            z=run["h"],
            mode="lines",
            name=run["label"],
            line=dict(width=5),
        ))

        fig.add_trace(go.Scatter3d(
            x=[run["theta_deg"][-1]],
            y=[run["phi_deg"][-1]],
            z=[run["h"][-1]],
            mode="markers",
            name=f"{run['label']} final",
            marker=dict(size=5),
            showlegend=False,
        ))

    theta_all = np.concatenate([r["theta_deg"] for r in runs])
    phi_all = np.concatenate([r["phi_deg"] for r in runs])

    fig.update_layout(
        title="Dymos Trajectory Results from Recorder Folders",
        scene=dict(
            xaxis_title="Downrange theta [deg]",
            yaxis_title="Crossrange phi [deg]",
            zaxis_title="Altitude h [m]",
            xaxis=dict(range=[float(theta_all.min()), float(theta_all.max())]),
            yaxis=dict(range=[float(phi_all.min()), float(phi_all.max())]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_html:
        fig.write_html(save_html)
        print(f"Saved interactive plot to {save_html}")
    else:
        fig.show()


def discover_folders(pattern: str) -> list[str]:
    """Discover and sort recorder folders matching a glob pattern."""
    folders = [p for p in glob.glob(pattern) if Path(p).is_dir()]
    return sorted(folders, key=lambda p: (len(Path(p).name), Path(p).name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and plot Dymos recorder folders.")
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Specific output folders to read. If omitted, --glob is used.",
    )
    parser.add_argument(
        "--glob",
        default="main_dymos_mission_example*_out",
        help="Glob pattern for output folders when --folders is not supplied.",
    )
    parser.add_argument(
        "--db",
        choices=["simulation", "solution"],
        default="simulation",
        help="Which Dymos database to read.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show(); useful when saving figures on a remote machine.",
    )
    parser.add_argument(
        "--save-prefix",
        default=None,
        help="If given, save matplotlib figures with this prefix.",
    )
    parser.add_argument(
        "--save-html",
        default=None,
        help="If given, save the interactive Plotly 3D plot to this HTML file.",
    )
    parser.add_argument(
        "--no-plotly",
        action="store_true",
        help="Skip the Plotly 3D plot.",
    )
    args = parser.parse_args()

    folders = args.folders if args.folders else discover_folders(args.glob)
    if not folders:
        raise SystemExit(f"No folders found. Tried pattern: {args.glob}")

    runs = read_many(folders, db_kind=args.db)
    print_summary(runs)

    plot_runs_matplotlib(runs, save_prefix=args.save_prefix)
    if not args.no_plotly:
        plot_runs_plotly(runs, save_html=args.save_html)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
