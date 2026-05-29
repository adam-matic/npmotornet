"""Regression tests for the Effector numerical integration methods.

These tests pin down three bugs that were fixed:

  1. Position was integrated with first-order Euler in every method (the Runge-Kutta
     weighting never reached the position channel).
  2. RK4 evaluated its stages from the previously advanced state instead of the step's
     initial state, degrading it to first order.
  3. The embedded adaptive methods (RKF45/DOPRI5) scaled their intermediate stage states
     by the node c_i twice, and a step() call advanced by the internal sub-step size
     instead of the fixed effector timestep.

The checks use a point mass (mass = 1) with a single force-free muscle, driven by an
endpoint load, so the exact solution is known analytically. The state is stored in
float32, so the practical accuracy floor for O(1) quantities is ~1e-6.

Runs under pytest, or standalone: `python tests/test_integration.py`.
"""
import numpy as np

import npmotornet.skeleton as skeleton
import npmotornet.muscle as muscle
import npmotornet.effector as effector

BIG = 1e9  # huge joint bounds so nothing is clipped during the test


def _make_effector(method, dt, damping=0.0, joint_state=None, tol=1e-6):
    sk = skeleton.PointMass(space_dim=1, mass=1.0)
    eff = effector.Effector(
        skeleton=sk, muscle=muscle.ReluMuscle(), timestep=dt,
        integration_method=method, damping=damping,
        pos_lower_bound=-BIG, pos_upper_bound=BIG,
        vel_lower_bound=-BIG, vel_upper_bound=BIG,
        adaptive_tolerance=tol, adaptive_min_dt=1e-9, adaptive_max_dt=dt,
    )
    # a single force-free muscle (max isometric force 0) just so geometry can be built
    eff.add_muscle(path_fixation_body=[0, 1], path_coordinates=[[0.0], [0.0]],
                   max_isometric_force=0.0)
    if joint_state is None:
        joint_state = np.zeros((1, 1))
    eff.reset(options={"joint_state": np.array(joint_state, dtype=np.float32)})
    return eff


def _run(method, dt, T, force=0.0, damping=0.0, joint_state=None):
    """Run the effector for T seconds and return (position, velocity)."""
    eff = _make_effector(method, dt, damping=damping, joint_state=joint_state)
    load = np.array([[force]], dtype=np.float32)
    action = np.zeros((1, 1), dtype=np.float32)
    for _ in range(int(round(T / dt))):
        eff.step(action, endpoint_load=load)
    return float(eff.states['joint'][0, 0]), float(eff.states['joint'][0, 1])


# Exact references --------------------------------------------------------------------
# A) constant force F on mass 1: a = F, v(t) = F t, x(t) = 0.5 F t^2
# B) linear damping a = -c v with v0: v(t) = v0 * exp(-c t)
DT = 0.01
T = 1.0
FORCE = 2.0
EXACT_POS_A = 0.5 * FORCE * T ** 2   # = 1.0
EXACT_VEL_A = FORCE * T              # = 2.0
DAMP_C, DAMP_V0 = 3.0, 5.0
EXACT_VEL_B = DAMP_V0 * np.exp(-DAMP_C * T)

ALL_METHODS = ['euler', 'rk4', 'rkf45', 'dopri5']
HIGH_ORDER = ['rk4', 'rkf45', 'dopri5']


def test_euler_is_first_order():
    """Euler should converge, and its error should roughly halve when dt halves."""
    e1 = abs(_run('euler', DT, T, damping=DAMP_C, joint_state=[[0.0, DAMP_V0]])[1] - EXACT_VEL_B)
    e2 = abs(_run('euler', DT / 2, T, damping=DAMP_C, joint_state=[[0.0, DAMP_V0]])[1] - EXACT_VEL_B)
    assert e1 > 1e-3, "euler error unexpectedly small; scenario may be degenerate"
    ratio = e1 / e2
    assert 1.7 < ratio < 2.4, f"euler should be ~order 1 (ratio ~2), got {ratio:.2f}"


def test_position_uses_high_order_weighting():
    """Bug 1 regression: on a quadratic trajectory a >=2nd order method is near-exact.

    The old code integrated position with Euler regardless of method, so RK4/RKF45/DOPRI5
    had the *same* ~2e-2 position error as Euler. They must now be far more accurate.
    """
    euler_err = abs(_run('euler', DT, T, force=FORCE)[0] - EXACT_POS_A)
    assert euler_err > 1e-2, "euler position error unexpectedly small"
    for m in HIGH_ORDER:
        err = abs(_run(m, DT, T, force=FORCE)[0] - EXACT_POS_A)
        assert err < 1e-4, f"{m} position error too large ({err:.2e}); position not high-order"
        assert err < euler_err / 100, f"{m} position no better than euler ({err:.2e} vs {euler_err:.2e})"


def test_rk4_is_high_order_on_velocity():
    """Bug 2 regression: RK4 must be high order on the damped (nonlinear) velocity.

    With the stage-chaining bug RK4 was first order, matching Euler to within ~2x.
    """
    euler_err = abs(_run('euler', DT, T, damping=DAMP_C, joint_state=[[0.0, DAMP_V0]])[1] - EXACT_VEL_B)
    rk4_err = abs(_run('rk4', DT, T, damping=DAMP_C, joint_state=[[0.0, DAMP_V0]])[1] - EXACT_VEL_B)
    assert rk4_err < 1e-4, f"rk4 velocity error too large ({rk4_err:.2e})"
    assert rk4_err < euler_err / 100, f"rk4 no better than euler ({rk4_err:.2e} vs {euler_err:.2e})"


def test_adaptive_methods_are_accurate():
    """Bug 3 regression: embedded adaptive methods must be accurate, not first-order/broken."""
    for m in ('rkf45', 'dopri5'):
        err = abs(_run(m, DT, T, damping=DAMP_C, joint_state=[[0.0, DAMP_V0]])[1] - EXACT_VEL_B)
        assert err < 1e-3, f"{m} damped-velocity error too large ({err:.2e})"


def test_adaptive_step_advances_fixed_timestep():
    """Bug 3 regression: each step() must advance exactly `dt` (no clock drift).

    On the constant-force quadratic, the final position equals 0.5*F*(n*dt)^2 only if the
    total simulated time is exactly n*dt. The old adaptive code advanced by an internal
    sub-step and drifted off the grid.
    """
    for m in ('rkf45', 'dopri5'):
        pos = _run(m, DT, T, force=FORCE)[0]
        assert abs(pos - EXACT_POS_A) < 1e-2, (
            f"{m} final position {pos:.5f} != {EXACT_POS_A} -> step() did not advance exactly dt")


def test_all_methods_agree_on_a_trajectory():
    """All methods should produce close trajectories on a smooth problem."""
    refs = {m: _run(m, DT, T, damping=DAMP_C, joint_state=[[0.0, DAMP_V0]])[1] for m in ALL_METHODS}
    hi = [refs[m] for m in HIGH_ORDER]
    assert max(hi) - min(hi) < 1e-3, f"high-order methods disagree: {refs}"
    assert abs(refs['euler'] - EXACT_VEL_B) < 5e-2, "euler far from exact on a smooth problem"


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
