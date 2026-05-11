"""
YAML-driven three-stage mission runner.

Workflow:
  1. Run or load Dymos stage 1 and stage 2 for vehicles A and B.
  2. Extract each vehicle state a configured number of seconds before Dymos impact.
  3. Run terminal ProNav from that handoff state.
  4. Stitch stage 1 + stage 2 + PN terminal into one continuous trajectory.
  5. Plot trajectory, controls, states, and PN diagnostics.

Usage:
    python main_three_stage_mission.py --config mission_config.yaml
    python main_three_stage_mission.py --config mission_config.yaml --reuse-dymos
    python main_three_stage_mission.py --config mission_config.yaml --run-dymos
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from dymos_runner import (
    DymosSolverConfig,
    TargetCondition,
    TerminalGuess,
    TwoStageSalvoConfig,
    VehicleInitialCondition,
    get_state_at_time,
    run_two_stage_salvo,
    stitch_two_stage,
)
from pronav_runner import (
    ProNavConfig,
    dymos_state_to_local_cartesian,
    run_pronav_terminal,
)
from mission_postprocess import (
    interpolate_stage_state,
    load_dymos_stages_from_folders,
    plot_three_stage_results,
    pronav_to_dymos_like,
    stitch_three_stage,
    stitch_two_dymos_stages,
)


def deg(value: float) -> float:
    return float(np.radians(value))


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def vehicle_initial_from_yaml(data: Dict[str, Any]) -> VehicleInitialCondition:
    return VehicleInitialCondition(
        h=float(data.get("h_m", 40000.0)),
        theta=deg(data.get("theta_deg", 0.0)),
        phi=deg(data.get("phi_deg", 0.0)),
        v=float(data.get("v_mps", 2000.0)),
        gamma=deg(data.get("gamma_deg", 0.0)),
        psi=deg(data.get("psi_deg", 0.0)),
    )


def target_condition_from_yaml(data: Dict[str, Any]) -> TargetCondition:
    return TargetCondition(
        theta=deg(data["theta_deg"]),
        phi=deg(data["phi_deg"]),
        h=float(data.get("h_m", 0.0)),
    )


def build_dymos_solver_config(cfg: Dict[str, Any]) -> DymosSolverConfig:
    data = cfg.get("dymos_solver", {}) or {}
    state_bounds = data.get("state_bounds_deg", {}) or {}
    control_bounds = data.get("control_bounds_deg", {}) or {}

    def bounds_deg(name: str, default: tuple[float, float]) -> tuple[float, float]:
        vals = state_bounds.get(name)
        return tuple(np.radians(vals)) if vals is not None else default

    def cbounds_deg(name: str, default: tuple[float, float]) -> tuple[float, float]:
        vals = control_bounds.get(name)
        return tuple(np.radians(vals)) if vals is not None else default

    duration_bounds = tuple(data.get("duration_bounds_s", [182.0, 500.0]))

    return DymosSolverConfig(
        optimizer=data.get("optimizer", "IPOPT"),
        duration_ref=float(data.get("duration_ref_s", 250.0)),
        duration_bounds=(float(duration_bounds[0]), float(duration_bounds[1])),
        objective=data.get("objective", "maximize_final_velocity"),
        num_segments_ipopt=int(data.get("num_segments_ipopt", 10)),
        check_setup=bool(data.get("check_setup", True)),
        simulate=bool(data.get("simulate", True)),
        theta_bounds=bounds_deg("theta", (0.0, np.radians(6.0))),
        phi_bounds=bounds_deg("phi", (-np.radians(5.0), np.radians(5.0))),
        gamma_bounds=bounds_deg("gamma", (-np.radians(89.0), np.radians(89.0))),
        psi_bounds=bounds_deg("psi", (-np.radians(90.0), np.radians(90.0))),
        sigma_bounds=cbounds_deg("sigma", (-np.radians(165.0), np.radians(165.0))),
        alpha_bounds=cbounds_deg("alpha", (0.0, np.radians(40.0))),
    )


def build_two_stage_mission_config(cfg: Dict[str, Any]) -> TwoStageSalvoConfig:
    target_guess = target_condition_from_yaml(cfg["target"]["initial_guess"])
    target_true = target_condition_from_yaml(cfg["target"]["updated"])
    terminal = cfg.get("terminal_guess", {}) or {}
    vehicles = cfg["vehicles"]

    return TwoStageSalvoConfig(
        t_reveal=float(cfg["stage_times"]["reveal_time_s"]),
        target_theta_guess=target_guess.theta,
        target_phi_guess=target_guess.phi,
        target_theta_true=target_true.theta,
        target_phi_true=target_true.phi,
        terminal_gamma_guess=deg(terminal.get("gamma_deg", -45.0)),
        terminal_psi_guess=deg(terminal.get("psi_deg", 0.0)),
        vehicle_A_initial=vehicle_initial_from_yaml(vehicles["A"]["initial"]),
        vehicle_B_initial=vehicle_initial_from_yaml(vehicles["B"]["initial"]),
    )


def build_pronav_config(cfg: Dict[str, Any]) -> ProNavConfig:
    data = cfg.get("pronav", {}) or {}
    return ProNavConfig(
        N=float(data.get("N", 3.0)),
        target_speed=float(data.get("target_speed_mps", 0.0)),
        target_heading_deg=float(data.get("target_heading_deg", 0.0)),
        alpha_max_deg=float(data.get("alpha_max_deg", 30.0)),
        sigma_max_deg=float(data.get("sigma_max_deg", 180.0)),
        intercept_radius_m=float(data.get("intercept_radius_m", 5.0)),
        t_final=float(data.get("t_final_s", cfg["stage_times"].get("pn_handoff_before_impact_s", 10.0))),
        num_time_points=int(data.get("num_time_points", 1000)),
        rtol=float(data.get("rtol", 1e-6)),
        atol=float(data.get("atol", 1e-8)),
        max_step=data.get("max_step_s", None),
    )


def run_or_load_dymos(cfg: Dict[str, Any], force_run: bool | None = None) -> Dict[str, Any]:
    run_cfg = cfg.get("run", {}) or {}
    run_dymos = bool(run_cfg.get("run_dymos", True)) if force_run is None else force_run
    mission = build_two_stage_mission_config(cfg)
    solver = build_dymos_solver_config(cfg)

    if run_dymos:
        return run_two_stage_salvo(
            mission=mission,
            solver_config=solver,
            synchronize_salvo=bool(run_cfg.get("synchronize_salvo", True)),
        )

    folders = cfg.get("dymos_folders", {}) or {}
    required = ["A1", "B1", "A2", "B2"]
    missing = [key for key in required if key not in folders]
    if missing:
        raise ValueError(f"run_dymos is false, but dymos_folders is missing keys: {missing}")

    stages = load_dymos_stages_from_folders({key: folders[key] for key in required})
    A = stitch_two_dymos_stages(stages["A1"], stages["A2"], mission.t_reveal, "Vehicle A (two-stage loaded)")
    B = stitch_two_dymos_stages(stages["B1"], stages["B2"], mission.t_reveal, "Vehicle B (two-stage loaded)")
    return {
        "mission": mission,
        "solver_config": solver,
        "A1": stages["A1"],
        "B1": stages["B1"],
        "A2": stages["A2"],
        "B2": stages["B2"],
        "A": A,
        "B": B,
        "salvo_time": max(stages["A1"]["t_final"], stages["B1"]["t_final"]),
        "loaded_from_folders": True,
    }


def run_terminal_pn_for_vehicle(vehicle_name: str, stage2: Dict[str, Any], cfg: Dict[str, Any],
                                target_true: TargetCondition) -> tuple[Dict[str, Any], float]:
    """Run PN for one vehicle from the configured handoff point."""
    pn_before = float(cfg["stage_times"]["pn_handoff_before_impact_s"])
    t_handoff_local = max(0.0, stage2["t_final"] - pn_before)
    handoff_state = interpolate_stage_state(stage2, t_handoff_local)

    pn_cfg = build_pronav_config(cfg)
    pn_cfg.t_final = pn_before

    ref_mode = ((cfg.get("pronav", {}) or {}).get("reference", {}) or {}).get("mode", "updated_target")
    if ref_mode == "updated_target":
        reference_theta = target_true.theta
        reference_phi = target_true.phi
    else:
        reference_theta = 0.0
        reference_phi = 0.0

    vehicle_pn, target_pn = dymos_state_to_local_cartesian(
        handoff_state,
        target_theta=target_true.theta,
        target_phi=target_true.phi,
        reference_theta=reference_theta,
        reference_phi=reference_phi,
    )

    pn_raw = run_pronav_terminal(vehicle_pn, target_pn, pn_cfg, label=f"Vehicle {vehicle_name} PN terminal")
    pn_stage = pronav_to_dymos_like(
        pn_raw,
        reference_theta=reference_theta,
        reference_phi=reference_phi,
    )
    return pn_stage, t_handoff_local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mission_config.yaml", help="YAML mission config path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-dymos", action="store_true", help="Force fresh Dymos optimization")
    group.add_argument("--reuse-dymos", action="store_true", help="Force loading Dymos recorder folders")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    force_run = True if args.run_dymos else False if args.reuse_dymos else None

    dymos_results = run_or_load_dymos(cfg, force_run=force_run)
    mission = dymos_results["mission"]
    target_true = TargetCondition(theta=mission.target_theta_true, phi=mission.target_phi_true, h=0.0)
    target_guess = (mission.target_theta_guess, mission.target_phi_guess)
    target_updated = (mission.target_theta_true, mission.target_phi_true)

    pn_enabled = bool((cfg.get("pronav", {}) or {}).get("enabled", True))
    if not pn_enabled:
        vehicle_results = {"A": dymos_results["A"], "B": dymos_results["B"]}
    else:
        pn_A, A_handoff_local = run_terminal_pn_for_vehicle("A", dymos_results["A2"], cfg, target_true)
        pn_B, B_handoff_local = run_terminal_pn_for_vehicle("B", dymos_results["B2"], cfg, target_true)

        A_pn_start_mission = mission.t_reveal + A_handoff_local
        B_pn_start_mission = mission.t_reveal + B_handoff_local

        A_full = stitch_three_stage(
            dymos_results["A1"], dymos_results["A2"], pn_A,
            t_reveal=mission.t_reveal,
            t_pn_start_mission=A_pn_start_mission,
            label="Vehicle A (Dymos + PN)",
        )
        B_full = stitch_three_stage(
            dymos_results["B1"], dymos_results["B2"], pn_B,
            t_reveal=mission.t_reveal,
            t_pn_start_mission=B_pn_start_mission,
            label="Vehicle B (Dymos + PN)",
        )
        vehicle_results = {"A": A_full, "B": B_full}

        print("\n=== ProNav Terminal Summary ===")
        for name, pn in [("A", pn_A), ("B", pn_B)]:
            raw = pn["raw"]
            print(f"Vehicle {name}: status={raw['status']}, success={raw['success']}, "
                  f"final separation={raw['final_separation']:.3f} m")

    print("\n=== Final Mission Times ===")
    for name, res in vehicle_results.items():
        print(f"Vehicle {name}: t_final = {res['t_final']:.3f} s")

    run_cfg = cfg.get("run", {}) or {}
    plot_three_stage_results(
        vehicle_results,
        target_guess=target_guess,
        target_true=target_updated,
        output_prefix=run_cfg.get("output_prefix"),
        show=bool(run_cfg.get("show_plots", True)),
        browser_3d=bool(run_cfg.get("browser_3d", True)),
    )


if __name__ == "__main__":
    main()
