import numpy as np

from pronav_runner import (
    ProNavConfig,
    ProNavInitialState,
    ProNavTargetState,
    dymos_state_to_local_cartesian,
    run_pronav_terminal,
)


def test_pronav_terminal_returns_expected_keys():
    cfg = ProNavConfig(
        N=3.0,
        target_speed=0.0,
        t_final=3.0,
        num_time_points=60,
        rtol=1e-5,
        atol=1e-7,
    )
    vehicle = ProNavInitialState(
        x=0.0,
        y=0.0,
        z=1000.0,
        v=500.0,
        gamma=np.radians(-20.0),
        psi=np.radians(45.0),
    )
    target = ProNavTargetState(x=3000.0, y=3000.0, z=0.0)

    result = run_pronav_terminal(vehicle, target, cfg, label="test")

    required = [
        "label", "status", "success", "time", "x", "y", "z", "v",
        "gamma", "psi", "target_x", "target_y", "target_z",
        "diagnostics", "final_separation",
    ]
    for key in required:
        assert key in result

    assert result["label"] == "test"
    assert len(result["time"]) > 1
    assert result["x"].shape == result["time"].shape
    assert np.isfinite(result["final_separation"])
    assert result["status"] in {"intercept", "ground_impact", "time_limit"}

    diag = result["diagnostics"]
    for key in ["alpha_cmd", "sigma_cmd", "lift", "drag", "range"]:
        assert key in diag
        assert len(diag[key]) == len(result["time"])
        assert np.all(np.isfinite(diag[key]))


def test_dymos_state_to_local_cartesian_conversion():
    state = {
        "theta": np.radians(1.0),
        "phi": np.radians(0.5),
        "h": 7500.0,
        "v": 900.0,
        "gamma": np.radians(-45.0),
        "psi": np.radians(38.0),
    }
    vehicle, target = dymos_state_to_local_cartesian(
        state,
        target_theta=np.radians(1.1),
        target_phi=np.radians(0.6),
        reference_theta=0.0,
        reference_phi=0.0,
    )

    assert vehicle.z == state["h"]
    assert vehicle.v == state["v"]
    assert target.z == 0.0
    assert target.x > vehicle.x
    assert target.y > vehicle.y
