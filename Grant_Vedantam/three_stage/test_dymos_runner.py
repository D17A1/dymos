"""
Slow smoke test for dymos_runner.py.

Run with:
    pytest -q test_dymos_runner.py -m slow

This test uses the current Vehicle A mission parameters from the original script:
initial phi = 1.7 deg, guessed target = (2.01 deg, 1.05 deg), terminal gamma
initial guess = -45 deg, terminal psi initial guess = 0 deg.
"""
import importlib.util
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _has_module(name):
    return importlib.util.find_spec(name) is not None


@pytest.mark.skipif(not _has_module("dymos") or not _has_module("openmdao"),
                    reason="Dymos/OpenMDAO not installed in this environment")
def test_vehicle_A_stage1_smoke():
    from dymos_runner import (
        DymosSolverConfig,
        VehicleInitialCondition,
        make_stage_run,
        solve_vehicle,
    )

    cfg = DymosSolverConfig(
        optimizer="IPOPT",
        check_setup=False,
        simulate=True,
    )

    run = make_stage_run(
        label="Vehicle A stage-1 smoke test",
        initial=VehicleInitialCondition(phi=np.radians(1.7), psi=0.0),
        target_theta=np.radians(2.01),
        target_phi=np.radians(1.05),
        terminal_gamma=-np.radians(45.0),
        terminal_psi=0.0,
        duration_guess=250.0,
    )

    case = solve_vehicle(run, cfg)

    assert case['time'].size > 5
    assert np.isfinite(case['t_final'])
    assert case['t_final'] > 0.0
    assert np.all(np.isfinite(case['h']))
    assert np.all(np.isfinite(case['v']))

    # theta/phi are returned in degrees for backward compatibility.
    assert np.isclose(case['theta'][-1], 2.01, atol=0.05)
    assert np.isclose(case['phi'][-1], 1.05, atol=0.05)
    assert case['h'][-1] < 100.0
