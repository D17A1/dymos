"""
Standalone example runner for pronav_runner.py.

This mirrors the old pronav_aero.py example parameters, but uses the importable
API from pronav_runner so it can serve as a quick smoke test and a template for
main mission integration.

Run:
    python main_pronav_example.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pronav_runner import (
    ProNavConfig,
    ProNavInitialState,
    ProNavTargetState,
    run_pronav_terminal,
    append_event_endpoint,
)


def print_summary(result: dict) -> None:
    diag = result["diagnostics"]

    print("\n=== ProNav terminal summary ===")
    print(f"Label: {result['label']}")
    print(f"Status: {result['status']}")
    print(f"Success/intercept: {result['success']}")
    print(f"Event time: {result['event_time']}")
    print(f"Final sampled time: {result['t_final']:.3f} s")
    print(f"Final separation: {result['final_separation']:.3f} m")
    print(f"Speed range: {np.min(result['v']):.3f} to {np.max(result['v']):.3f} m/s")
    print(f"Minimum range sampled: {np.min(diag['range']):.3f} m")
    print(f"Max alpha_cmd: {np.max(np.degrees(diag['alpha_cmd'])):.3f} deg")
    print(f"Max |sigma_cmd|: {np.max(np.abs(np.degrees(diag['sigma_cmd']))):.3f} deg")
    print(f"Alpha saturation active: {np.any(diag['alpha_sat'])}")
    print(f"Sigma saturation active: {np.any(diag['sigma_sat'])}")
    print(f"Max required lift accel: {np.max(diag['a_lift_req']):.3f} m/s^2")
    print(f"Max available lift accel: {np.max(diag['a_lift_avail']):.3f} m/s^2")


def plot_result(result: dict) -> None:
    diag = result["diagnostics"]
    arr = append_event_endpoint(result)

    # Ground track
    plt.figure(figsize=(8, 6))
    plt.plot(arr["x"], arr["y"], label="Vehicle")
    plt.plot(arr["target_x"], arr["target_y"], label="Target")
    plt.scatter([result["x"][0]], [result["y"][0]], marker="o", s=80, label="Vehicle start")
    plt.scatter([result["target_x"][0]], [result["target_y"][0]], marker="s", s=80, label="Target start")
    plt.scatter([arr["x"][-1]], [arr["y"][-1]], marker="x", s=100, label="Vehicle end")
    plt.scatter([arr["target_x"][-1]], [arr["target_y"][-1]], marker="x", s=100, label="Target end")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Ground Track")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Altitude
    plt.figure(figsize=(8, 5))
    plt.plot(result["time"], result["z"], label="Vehicle altitude")
    plt.plot(result["time"], result["target_z"], label="Target altitude")
    plt.xlabel("Time [s]")
    plt.ylabel("z [m]")
    plt.title("Altitude vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Range
    plt.figure(figsize=(8, 5))
    plt.plot(diag["time"], diag["range"])
    plt.xlabel("Time [s]")
    plt.ylabel("Range [m]")
    plt.title("Closing Range")
    plt.grid(True)
    plt.tight_layout()

    # Speed
    plt.figure(figsize=(8, 5))
    plt.plot(result["time"], result["v"])
    plt.xlabel("Time [s]")
    plt.ylabel("Vehicle speed [m/s]")
    plt.title("Vehicle Speed")
    plt.grid(True)
    plt.tight_layout()

    # Flight angles
    plt.figure(figsize=(8, 5))
    plt.plot(result["time"], np.degrees(result["gamma"]), label="gamma")
    plt.plot(result["time"], np.degrees(result["psi"]), label="psi")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.title("Flight Path and Heading Angles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Diagnostics grid
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    axs[0, 0].plot(diag["time"], np.degrees(diag["alpha_cmd"]), label="alpha_cmd")
    axs[0, 0].plot(diag["time"], np.degrees(np.unwrap(diag["sigma_cmd"])), label="sigma_cmd unwrapped")
    axs[0, 0].set_title("Control Inputs")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Angle [deg]")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(diag["time"], diag["a_gamma_cmd"], label="a_gamma_cmd")
    axs[0, 1].plot(diag["time"], diag["a_psi_cmd"], label="a_psi_cmd")
    axs[0, 1].set_title("PN Command Components")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Acceleration [m/s²]")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(diag["time"], diag["a_lift_req"], label="required lift accel")
    axs[1, 0].plot(diag["time"], diag["a_lift_avail"], label="available lift accel")
    axs[1, 0].set_title("Lift Acceleration Budget")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Acceleration [m/s²]")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(diag["time"], diag["lift"], label="Lift")
    axs[1, 1].plot(diag["time"], diag["drag"], label="Drag")
    axs[1, 1].set_title("Aerodynamic Forces")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Force [N]")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    axs[2, 0].plot(diag["time"], diag["los_rate"], label="LOS rate")
    axs[2, 0].plot(diag["time"], diag["Vc"], label="Closing speed")
    axs[2, 0].set_title("LOS Rate / Closing Speed")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].plot(diag["time"], diag["alpha_sat"].astype(float), label="alpha sat")
    axs[2, 1].plot(diag["time"], diag["sigma_sat"].astype(float), label="sigma sat")
    axs[2, 1].set_title("Control Saturation Flags")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("0/1")
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    # Parameters copied from the old pronav_aero run_example() style case.
    cfg = ProNavConfig(
        N=3.0,
        target_speed=50.0,
        target_heading_deg=315.0,
        gravity=9.81,
        alpha_max_deg=30.0,
        sigma_max_deg=165.0,
        intercept_radius_m=5.0,
        t_final=100.0,
        num_time_points=2000,
        rtol=1e-6,
        atol=1e-8,
        dense_output=True,
    )

    vehicle = ProNavInitialState(
        x=0.0,
        y=0.0,
        z=7600.0,
        v=900.0,
        gamma=np.radians(-50.0),
        psi=np.radians(38.0),
    )

    target = ProNavTargetState(
        x=4000.0,
        y=4000.0,
        z=0.0,
    )

    result = run_pronav_terminal(
        vehicle=vehicle,
        target=target,
        cfg=cfg,
        label="Old pronav_aero parameter example",
    )

    print_summary(result)
    plot_result(result)


if __name__ == "__main__":
    main()
