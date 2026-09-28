"""Tests for the TwoDofArm gravity term and its routing through the arm26 effectors.

Runs under pytest, or standalone: `python tests/test_gravity.py`.
"""
import numpy as np

import npmotornet.skeleton as skeleton
import npmotornet.muscle as muscle
import npmotornet.effector as effector

G = 9.81


def _potential(arm, q):
    """U = g*(m1*y1g + m2*y2g) with angles measured from +x."""
    t1, t2 = q
    y1g = arm.L1g * np.sin(t1)
    y2g = arm.L1 * np.sin(t1) + arm.L2g * np.sin(t1 + t2)
    return arm.g * (arm.m1 * y1g + arm.m2 * y2g)


def test_gravity_torque_is_potential_gradient():
    """The coefficients folded in __init__ must reproduce G = dU/dq."""
    arm = skeleton.TwoDofArm(m1=1.82, m2=1.43, l1g=.135, l2g=.165, i1=.051, i2=.057,
                             l1=.309, l2=.333, g=G)
    eps = 1e-6
    for q in ([0.3, 1.1], [1.2, 0.2], [0.0, 2.5]):
        q = np.array(q, dtype=np.float64)
        grad = [(_potential(arm, q + dq) - _potential(arm, q - dq)) / (2 * eps)
                for dq in (np.array([eps, 0.]), np.array([0., eps]))]
        c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
        g1 = arm.gravity_coef_1 * c1 + arm.gravity_coef_12 * c12
        g2 = arm.gravity_coef_12 * c12
        np.testing.assert_allclose([g1, g2], grad, rtol=1e-5)


def test_zero_gravity_by_default():
    arm = skeleton.TwoDofArm()
    assert arm.g == 0
    assert arm.gravity_coef_1 == 0 and arm.gravity_coef_12 == 0


def test_arm26_effectors_route_g_to_skeleton():
    rigid = effector.RigidTendonArm26(muscle=muscle.RigidTendonHillMuscle(), g=G)
    compliant = effector.CompliantTendonArm26(g=G)
    assert rigid.skeleton.g == G
    assert compliant.skeleton.g == G
    assert effector.RigidTendonArm26(muscle=muscle.RigidTendonHillMuscle()).skeleton.g == 0


def test_unactuated_arm_falls_under_gravity():
    """Horizontal upper arm, no activation: the shoulder must accelerate downward."""
    eff = effector.CompliantTendonArm26(g=G)
    eff.reset(options={"joint_state": np.array([[0.8, 0.8, 0., 0.]], dtype=np.float32)})
    for _ in range(100):
        eff.step(np.zeros((1, 6), dtype=np.float32))
    assert eff.states['joint'][0, 2] < 0, "shoulder velocity should be negative (falling)"


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
