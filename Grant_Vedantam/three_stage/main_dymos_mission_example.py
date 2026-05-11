"""
Example mission-level runner for the importable Dymos module.

This file is intentionally small: it owns the mission parameters and workflow
calls, while dymos_runner.py owns the Dymos model and solver setup.
"""
import numpy as np

from dymos_runner import (
    DymosSolverConfig,
    TwoStageSalvoConfig,
    VehicleInitialCondition,
    run_two_stage_salvo,
    concat_timeseries,
)


def main():
    mission = TwoStageSalvoConfig(
        t_reveal=50.0,
        target_theta_guess=np.radians(2.01),
        target_phi_guess=np.radians(1.05),
        target_theta_true=np.radians(2.0),
        target_phi_true=np.radians(1.0),
        terminal_gamma_guess=-np.radians(45.0),
        terminal_psi_guess=0.0,
        vehicle_A_initial=VehicleInitialCondition(phi=np.radians(1.7), psi=0.0),
        vehicle_B_initial=VehicleInitialCondition(phi=0.0 / 6_371_000.0, psi=0.0),
    )

    solver = DymosSolverConfig(
        optimizer="IPOPT",
        objective="maximize_final_velocity",
        alpha_bounds=(0.0, np.radians(40.0)),
        sigma_bounds=(-np.radians(165.0), np.radians(165.0)),
    )

    results = run_two_stage_salvo(mission, solver, synchronize_salvo=True)

    print("\n=== Impact Times ===")
    print(f"Vehicle A impact time: {results['A']['t_final']:.3f} s")
    print(f"Vehicle B impact time: {results['B']['t_final']:.3f} s")
    print(f"Stage-1 salvo time:    {results['salvo_time']:.3f} s")

    # Example: stitched controls for later plotting.
    tA_alpha, alphaA = concat_timeseries(results['A1'], results['A2'], 'alpha',
                                         mission.t_reveal, mission.t_reveal)
    tB_alpha, alphaB = concat_timeseries(results['B1'], results['B2'], 'alpha',
                                         mission.t_reveal, mission.t_reveal)
    print(f"A alpha samples: {len(alphaA)}, B alpha samples: {len(alphaB)}")


if __name__ == "__main__":
    main()
