"""
One Degree-of-Freedom Arm Example
==================================
This example demonstrates the OneDofArm skeleton: a single rigid link
rotating about a fixed pivot. The link extends along +x at theta=0.

The script doubles as a regression test: it asserts joint-to-cartesian
transforms, path2cartesian Jacobians (vs finite differences),
constant-torque dynamics (vs a closed-form solution), and the
energy-conservation prediction for a gravity-driven swing. It then runs
demonstration simulations of damping, gravity, and antagonistic muscle
control.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import npmotornet.skeleton as skeleton
import npmotornet.muscle as muscle
import npmotornet.effector as effector


def _check_kinematics(arm):
  """joint2cartesian should give (l*cos, l*sin) and the corresponding velocity."""
  l = arm.L
  cases = [
    (0.0, 0.0, (l, 0.0, 0.0, 0.0)),
    (np.pi / 2, 0.0, (0.0, l, 0.0, 0.0)),
    (np.pi / 4, 1.0, (l / np.sqrt(2), l / np.sqrt(2), -l / np.sqrt(2), l / np.sqrt(2))),
  ]
  for theta, omega, expected in cases:
    js = np.array([[theta, omega]], dtype=np.float32)
    got = arm.joint2cartesian(js)[0]
    assert np.allclose(got, expected, atol=1e-5), f"joint2cartesian({theta}, {omega}) = {got}, expected {expected}"


def _check_path2cartesian(arm):
  """Jacobian dxy/dtheta from path2cartesian should match a finite-difference estimate."""
  path_fixation_body = np.array([[[0., 1., 1.]]], dtype=np.float32)
  path_coordinates = np.array([[[0.05, 0.10, 0.20], [0.02, -0.03, 0.01]]], dtype=np.float32)

  # worldspace fixation: position == path coords, velocity and dof-derivative both zero
  js = np.array([[0.4, 0.7]], dtype=np.float32)
  xy, dxy_dt, dxy_ddof = arm.path2cartesian(path_coordinates, path_fixation_body, js)
  assert np.allclose(xy[0, :, 0], path_coordinates[0, :, 0], atol=1e-6), "worldspace point moved"
  assert np.allclose(dxy_dt[0, :, 0], 0., atol=1e-6), "worldspace point has nonzero velocity"
  assert np.allclose(dxy_ddof[0, :, :, 0], 0., atol=1e-6), "worldspace point has nonzero jacobian"

  # bone-attached point: position matches an explicit rotation
  theta = float(js[0, 0])
  c, s = np.cos(theta), np.sin(theta)
  for pt_idx in (1, 2):
    px, py = path_coordinates[0, 0, pt_idx], path_coordinates[0, 1, pt_idx]
    expected_xy = (px * c - py * s, px * s + py * c)
    assert np.allclose(xy[0, :, pt_idx], expected_xy, atol=1e-6)

  # finite-difference check on dxy/dtheta
  eps = 1e-3
  js_plus = js.copy(); js_plus[0, 0] += eps
  js_minus = js.copy(); js_minus[0, 0] -= eps
  xy_plus, _, _ = arm.path2cartesian(path_coordinates, path_fixation_body, js_plus)
  xy_minus, _, _ = arm.path2cartesian(path_coordinates, path_fixation_body, js_minus)
  fd = (xy_plus - xy_minus) / (2 * eps)
  assert np.allclose(fd, dxy_ddof[:, :, 0, :], atol=1e-4), \
    f"analytical jacobian disagrees with finite difference: max err = {np.max(np.abs(fd - dxy_ddof[:, :, 0, :]))}"

  # chain-rule check on dxy/dt
  expected_dxy_dt = dxy_ddof[:, :, 0, :] * js[0, 1]
  assert np.allclose(dxy_dt, expected_dxy_dt, atol=1e-6), "dxy/dt does not match jacobian * omega"


def _check_constant_torque(arm):
  """With no gravity and no damping, a constant torque should integrate to theta_0 + 0.5*(tau/I)*t^2 (pure
  Euler accumulates a small bias, so the tolerance is set accordingly)."""
  dt = 1e-3
  n_steps = 500
  tau = 0.4
  arm.build(timestep=dt, pos_upper_bound=arm.pos_upper_bound, pos_lower_bound=arm.pos_lower_bound,
            vel_upper_bound=arm.vel_upper_bound, vel_lower_bound=arm.vel_lower_bound)

  state = np.array([[0.3, 0.0]], dtype=np.float32)
  torque = np.array([[tau]], dtype=np.float32)
  load = np.zeros((1, 2), dtype=np.float32)
  for _ in range(n_steps):
    d = arm.ode(torque, state, load)
    state = arm.integrate(dt, d, state)

  t = n_steps * dt
  acc = tau / arm.inertia_total
  expected_omega = acc * t
  expected_theta = 0.3 + 0.5 * acc * t * t  # Euler lags by ~0.5*acc*dt*t -> bake that in
  euler_pos_bias = 0.5 * acc * dt * t
  assert abs(float(state[0, 1]) - expected_omega) < 1e-3, \
    f"velocity {state[0, 1]} != {expected_omega}"
  assert abs(float(state[0, 0]) - (expected_theta - euler_pos_bias)) < 1e-3, \
    f"position {state[0, 0]} != {expected_theta - euler_pos_bias}"


def _check_gravity_energy():
  """Released from horizontal (theta=0) with no damping, the swing-through speed at the bottom of the arc
  (theta=-pi/2) should match the value predicted by conservation of energy: omega^2 = 2*m*g*lg / I_total
  (the center of mass drops by lg)."""
  arm = skeleton.OneDofArm(g=9.81, viscosity=0.0)
  dt = 1e-4
  arm.build(timestep=dt, pos_upper_bound=arm.pos_upper_bound, pos_lower_bound=arm.pos_lower_bound,
            vel_upper_bound=arm.vel_upper_bound, vel_lower_bound=arm.vel_lower_bound)

  state = np.array([[0.0, 0.0]], dtype=np.float32)  # link along +x, horizontal
  torque = np.zeros((1, 1), dtype=np.float32)
  load = np.zeros((1, 2), dtype=np.float32)

  best_omega = 0.0
  for _ in range(20000):
    d = arm.ode(torque, state, load)
    state = arm.integrate(dt, d, state)
    best_omega = max(best_omega, abs(float(state[0, 1])))

  expected_omega = np.sqrt(2 * arm.m * arm.g * arm.Lg / arm.inertia_total)
  rel_err = abs(best_omega - expected_omega) / expected_omega
  assert rel_err < 0.01, f"peak |omega| = {best_omega}, expected {expected_omega} (rel err {rel_err:.4f})"


def main():
  print("=" * 60)
  print("One Degree-of-Freedom Arm Example")
  print("=" * 60)

  print("\n[Assertions]")
  arm = skeleton.OneDofArm()
  _check_kinematics(arm)
  print("  joint2cartesian: OK")
  arm.build(timestep=0.01, pos_upper_bound=arm.pos_upper_bound, pos_lower_bound=arm.pos_lower_bound,
            vel_upper_bound=arm.vel_upper_bound, vel_lower_bound=arm.vel_lower_bound)
  _check_path2cartesian(arm)
  print("  path2cartesian + finite-difference Jacobian: OK")
  _check_constant_torque(skeleton.OneDofArm())
  print("  constant-torque dynamics match closed-form: OK")
  _check_gravity_energy()
  print("  gravity swing-through energy conservation: OK")

  print("\n1. Creating a OneDofArm skeleton...")
  arm = skeleton.OneDofArm(m=1.43, l=0.333, lg=0.165, i=0.057)
  print(f"   dof: {arm.dof}   space_dim: {arm.space_dim}")
  print(f"   length: {arm.l:.3f} m   mass: {arm.m:.3f} kg   inertia (total about pivot): {arm.inertia_total:.5f} kg.m^2")

  print("\n2. joint2cartesian spot checks:")
  for theta_deg in (0, 45, 90, 135):
    theta = np.deg2rad(theta_deg)
    js = np.array([[theta, 0.0]], dtype=np.float32)
    xy = arm.joint2cartesian(js)[0]
    print(f"   theta = {theta_deg:3d} deg -> endpoint = ({xy[0]:+.4f}, {xy[1]:+.4f}) m")

  print("\n3. Antagonistic muscles via the standard Effector API...")
  arm_effector = effector.Effector(skeleton=arm, muscle=muscle.ReluMuscle(), timestep=0.01)
  arm_effector.add_muscle(
    path_fixation_body=[0, 1],
    path_coordinates=[[0.0, 0.05], [0.1, 0.0]],
    name='Flexor',
    max_isometric_force=80.0,
  )
  arm_effector.add_muscle(
    path_fixation_body=[0, 1],
    path_coordinates=[[0.0, -0.05], [0.1, 0.0]],
    name='Extensor',
    max_isometric_force=80.0,
  )
  print(f"   muscles: {arm_effector.muscle_name}")

  for label, action in (('flexor only', [1.0, 0.0]), ('extensor only', [0.0, 1.0])):
    arm_effector.reset(options={'joint_state': np.array([[0.0]], dtype=np.float32)})
    a = np.array([action], dtype=np.float32)
    for _ in range(50):
      arm_effector.step(a)
    final_theta = float(arm_effector.states['joint'][0, 0])
    print(f"   {label:14s} -> theta after 0.5 s = {np.rad2deg(final_theta):+7.2f} deg")

  print("\n4. Damped relaxation (no input, viscosity > 0):")
  damped = skeleton.OneDofArm(viscosity=0.5)
  dt = 0.005
  damped.build(timestep=dt, pos_upper_bound=damped.pos_upper_bound, pos_lower_bound=damped.pos_lower_bound,
               vel_upper_bound=damped.vel_upper_bound, vel_lower_bound=damped.vel_lower_bound)
  state = np.array([[0.0, 3.0]], dtype=np.float32)
  zero_torque = np.zeros((1, 1), dtype=np.float32)
  zero_load = np.zeros((1, 2), dtype=np.float32)
  speeds = []
  for _ in range(200):
    d = damped.ode(zero_torque, state, zero_load)
    state = damped.integrate(dt, d, state)
    speeds.append(abs(float(state[0, 1])))
  assert all(speeds[i + 1] <= speeds[i] + 1e-6 for i in range(len(speeds) - 1)), "damping did not produce monotonic decay"
  print(f"   |omega| over 1.0 s: {speeds[0]:.3f} -> {speeds[-1]:.4f} rad/s (monotonic decay confirmed)")

  print("\n5. Gravity swing from horizontal (g=9.81, no damping):")
  swinger = skeleton.OneDofArm(g=9.81, viscosity=0.0)
  dt = 0.001
  swinger.build(timestep=dt, pos_upper_bound=swinger.pos_upper_bound, pos_lower_bound=swinger.pos_lower_bound,
                vel_upper_bound=swinger.vel_upper_bound, vel_lower_bound=swinger.vel_lower_bound)
  state = np.array([[0.0, 0.0]], dtype=np.float32)
  for _ in range(500):
    d = swinger.ode(zero_torque, state, zero_load)
    state = swinger.integrate(dt, d, state)
  theta = float(state[0, 0])
  omega = float(state[0, 1])
  print(f"   after 0.5 s: theta = {np.rad2deg(theta):+7.2f} deg, omega = {omega:+.3f} rad/s")

  print("\n" + "=" * 60)
  print("Example completed successfully!")
  print("=" * 60)


if __name__ == "__main__":
  main()
