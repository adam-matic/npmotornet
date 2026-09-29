"""Tests for the optional muscle short-range stiffness (CompliantTendonHillMuscle srs_*).

Runs under pytest, or standalone: `python tests/test_short_range_stiffness.py`.
"""
import numpy as np

import npmotornet.effector as effector
import npmotornet.muscle as muscle

POSTURE = np.array([[0.70, 1.20]], dtype=np.float32)


def _arm(**options):
    arm = effector.CompliantTendonArm26(timestep=0.001, n_ministeps=1, tau_activation=0.075, **options)
    arm.reset(options={"joint_state": POSTURE})
    return arm


def _torque_free_pattern(arm):
    """Non-negative activation pattern producing no net joint torque (mean 1)."""
    weighted = arm.states["geometry"][0, 2:4, :].astype(np.float64) * arm.muscle.max_iso_force.reshape(-1)
    ones = np.ones(6)
    pattern = ones - weighted.T @ np.linalg.solve(weighted @ weighted.T, weighted @ ones)
    return pattern / pattern.mean()


def _load_displacement(cocontraction, **options):
    """Extra hand displacement after 0.5 s of a 2 N outward load, activations held constant."""
    arm = _arm(**options)
    home = arm.states["fingertip"][0].astype(np.float64)
    outward = home / np.linalg.norm(home)
    activation = np.clip(0.03 + cocontraction * _torque_free_pattern(arm), 0.0, 1.0).astype(np.float32)[None, :]
    endpoints = []
    for load in (np.zeros(2), 2.0 * outward):
        arm = _arm(**options)
        for _ in range(300):
            arm.step(activation, endpoint_load=np.zeros((1, 2), np.float32))
        for _ in range(500):
            arm.step(activation, endpoint_load=load[None, :].astype(np.float32))
        endpoints.append(arm.states["fingertip"][0].astype(np.float64))
    return float((endpoints[1] - endpoints[0]) @ outward)


def test_disabled_by_default_and_state_row_only_when_enabled():
    default = muscle.CompliantTendonHillMuscle()
    assert not default.short_range_stiffness
    assert default.state_dim == 7
    enabled = muscle.CompliantTendonHillMuscle(srs_gamma=100.0)
    assert enabled.state_dim == 8
    assert enabled.state_name[-1] == "short-range anchor length"
    assert _arm(srs_gamma=100.0).states["muscle"].shape == (1, 8, 6)


def test_zero_gamma_matches_the_default_model_exactly():
    rng = np.random.default_rng(0)
    default, explicit = _arm(), _arm(srs_gamma=0.0)
    for step in range(300):
        if step % 50 == 0:
            action = rng.uniform(0.02, 0.3, (1, 6)).astype(np.float32)
        default.step(action)
        explicit.step(action)
    for key in ("joint", "muscle"):
        np.testing.assert_array_equal(default.states[key], explicit.states[key])


def test_cocontraction_stiffens_only_with_short_range_stiffness():
    without = [_load_displacement(c) for c in (0.0, 0.3)]
    with_srs = [_load_displacement(c, srs_gamma=100.0) for c in (0.0, 0.3)]
    assert with_srs[1] < 0.5 * with_srs[0], with_srs
    assert with_srs[0] < without[0], (with_srs, without)
    # without the element, co-contraction does not reduce the displacement
    assert without[1] > 0.8 * without[0], without


def test_anchor_stays_within_the_short_range_of_the_fibre():
    arm = _arm(srs_gamma=100.0, srs_range=0.01)
    for _ in range(400):
        arm.step(np.full((1, 6), 0.4, np.float32), endpoint_load=np.array([[20.0, 0.0]], np.float32))
    fibre = arm.states["muscle"][0, 1, :]
    anchor = arm.states["muscle"][0, 7, :]
    limit = 0.01 * arm.muscle.l0_ce.reshape(-1)
    assert np.all(np.abs(fibre - anchor) <= limit + 1e-6)


def test_invalid_parameters_are_rejected():
    for options in ({"srs_gamma": -1.0}, {"srs_range": 0.0}, {"srs_tau": 0.0}):
        try:
            muscle.CompliantTendonHillMuscle(**options)
        except ValueError:
            continue
        raise AssertionError(f"accepted {options}")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
