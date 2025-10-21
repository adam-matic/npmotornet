"""
Numba-optimized implementations of performance-critical functions.

This module provides JIT-compiled versions of computationally intensive
operations in npmotornet, offering significant speed improvements for
muscle force calculations, skeleton kinematics, and geometric transformations.

To use these optimizations, simply import the optimized muscle or skeleton
classes from this module instead of the standard ones.
"""

import numpy as np
from numba import jit, float32, int32
import numba


# ==============================================================================
# Optimized Muscle Force Calculations
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def mujoco_hill_bump(L, lmin, mid, lmax):
    """Optimized bump function for MujocoHillMuscle.

    This is a JIT-compiled version of the _bump method that computes
    a skewed quadratic spline function.

    Args:
        L: Muscle length (normalized)
        lmin: Minimum length
        mid: Midpoint length
        lmax: Maximum length

    Returns:
        Bump function value
    """
    batch_size = L.shape[0]
    n_muscles = L.shape[2]
    out = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)

    for b in range(batch_size):
        for m in range(n_muscles):
            l = L[b, 0, m]
            lmin_val = lmin[0, 0, m]
            mid_val = mid[0, 0, m]
            lmax_val = lmax[0, 0, m]

            # Compute boundaries
            left = 0.5 * (lmin_val + mid_val)
            right = 0.5 * (mid_val + lmax_val)

            # Check if out of range
            if l <= lmin_val or l >= lmax_val:
                out[b, 0, m] = 0.0
                continue

            # Compute x based on position
            if l < left:
                x = (l - lmin_val) / (left - lmin_val)
            elif l < mid_val:
                x = (mid_val - l) / (mid_val - left)
            elif l < right:
                x = (l - mid_val) / (right - mid_val)
            else:
                x = (lmax_val - l) / (lmax_val - right)

            # Compute y based on position
            pfivexx = 0.5 * x * x
            if l < left:
                out[b, 0, m] = pfivexx
            elif l < mid_val:
                out[b, 0, m] = 1.0 - pfivexx
            elif l < right:
                out[b, 0, m] = 1.0 - pfivexx
            else:
                out[b, 0, m] = pfivexx

    return out


@jit(nopython=True, cache=True, fastmath=True)
def mujoco_hill_integrate_core(
    activation, muscle_len, muscle_vel,
    l0_ce, l0_se, vmax, lmin, lmax, b, c, p1, p2, mid, fvmax, max_iso_force, passive_forces
):
    """Core integration logic for MujocoHillMuscle (Numba-optimized).

    This function implements the muscle force calculation logic from
    MujocoHillMuscle._integrate, optimized with Numba JIT compilation.

    Args:
        All muscle parameters and state variables

    Returns:
        Tuple of (flpe, flce, fvce, force) arrays
    """
    batch_size = muscle_len.shape[0]
    n_muscles = muscle_len.shape[2]

    # Pre-allocate outputs
    flpe = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)
    flce = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)
    fvce = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)
    force = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)

    # Flatten parameter arrays for easier indexing
    l0_ce_flat = l0_ce.ravel()
    b_flat = b.ravel()
    p1_flat = p1.ravel()
    p2_flat = p2.ravel()
    mid_flat = mid.ravel()
    lmax_flat = lmax.ravel()
    lmin_flat = lmin.ravel()
    c_flat = c.ravel()
    fvmax_flat = fvmax.ravel()
    max_iso_force_flat = max_iso_force.ravel()

    for batch_idx in range(batch_size):
        for muscle_idx in range(n_muscles):
            ml = muscle_len[batch_idx, 0, muscle_idx]
            mv = muscle_vel[batch_idx, 0, muscle_idx]
            act = activation[batch_idx, 0, muscle_idx]

            # Get muscle parameters using flattened arrays
            l_ce = l0_ce_flat[muscle_idx]
            b_val = b_flat[muscle_idx]
            p1_val = p1_flat[muscle_idx]
            p2_val = p2_flat[muscle_idx]
            mid_val = mid_flat[muscle_idx]
            lmax_val = lmax_flat[muscle_idx]
            lmin_val = lmin_flat[muscle_idx]
            c_val = c_flat[muscle_idx]
            fvmax_val = fvmax_flat[muscle_idx]
            max_force = max_iso_force_flat[muscle_idx]

            # Passive element force-length
            if ml <= 1.0:
                x_pe = 0.0
                flpe[batch_idx, 0, muscle_idx] = 0.0
            elif ml <= b_val:
                x_pe = (ml - 1.0) / p1_val
                flpe[batch_idx, 0, muscle_idx] = p2_val * x_pe * x_pe * x_pe
            else:
                x_pe = (ml - b_val) / p1_val
                flpe[batch_idx, 0, muscle_idx] = p2_val * (1.0 + 3.0 * x_pe)

            # Active element force-length (using bump functions)
            # Main bump
            left1 = 0.5 * (lmin_val + 1.0)
            right1 = 0.5 * (1.0 + lmax_val)

            if ml <= lmin_val or ml >= lmax_val:
                bump1 = 0.0
            elif ml < left1:
                x1 = (ml - lmin_val) / (left1 - lmin_val)
                bump1 = 0.5 * x1 * x1
            elif ml < 1.0:
                x1 = (1.0 - ml) / (1.0 - left1)
                bump1 = 1.0 - 0.5 * x1 * x1
            elif ml < right1:
                x1 = (ml - 1.0) / (right1 - 1.0)
                bump1 = 1.0 - 0.5 * x1 * x1
            else:
                x1 = (lmax_val - ml) / (lmax_val - right1)
                bump1 = 0.5 * x1 * x1

            # Secondary bump
            left2 = 0.5 * (lmin_val + mid_val)
            right2 = 0.5 * (mid_val + 0.95)

            if ml <= lmin_val or ml >= 0.95:
                bump2 = 0.0
            elif ml < left2:
                x2 = (ml - lmin_val) / (left2 - lmin_val)
                bump2 = 0.5 * x2 * x2
            elif ml < mid_val:
                x2 = (mid_val - ml) / (mid_val - left2)
                bump2 = 1.0 - 0.5 * x2 * x2
            elif ml < right2:
                x2 = (ml - mid_val) / (right2 - mid_val)
                bump2 = 1.0 - 0.5 * x2 * x2
            else:
                x2 = (0.95 - ml) / (0.95 - right2)
                bump2 = 0.5 * x2 * x2

            flce[batch_idx, 0, muscle_idx] = bump1 + 0.15 * bump2

            # Velocity-active force
            if mv <= -1.0:
                fvce[batch_idx, 0, muscle_idx] = 0.0
            elif mv <= 0.0:
                fvce[batch_idx, 0, muscle_idx] = (mv + 1.0) * (mv + 1.0)
            elif mv <= c_val:
                fvce[batch_idx, 0, muscle_idx] = fvmax_val - (c_val - mv) * (c_val - mv) / c_val
            else:
                fvce[batch_idx, 0, muscle_idx] = fvmax_val

            # Total force
            force[batch_idx, 0, muscle_idx] = (act * flce[batch_idx, 0, muscle_idx] * fvce[batch_idx, 0, muscle_idx] + passive_forces * flpe[batch_idx, 0, muscle_idx]) * max_force

    return flpe, flce, fvce, force


@jit(nopython=True, cache=True, fastmath=True)
def rigid_tendon_hill_integrate_core(
    activation, muscle_len, muscle_vel, muscle_len_n, muscle_vel_n, muscle_strain,
    l0_ce, l0_pe, vmax, max_iso_force, k_pe, min_flce, f_iso_n_den, q_crit, s_as
):
    """Core integration logic for RigidTendonHillMuscle (Numba-optimized).

    Optimized muscle force calculations for the Kistemaker rigid tendon model.

    Args:
        All muscle parameters and state variables

    Returns:
        Tuple of (flpe, flce, active_force, force) arrays
    """
    batch_size = muscle_len_n.shape[0]
    n_muscles = muscle_len_n.shape[2]

    # Pre-allocate outputs
    flpe = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)
    flce = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)
    active_force = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)
    force = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)

    # Flatten parameter arrays for easier indexing
    k_pe_flat = k_pe.ravel()
    max_iso_force_flat = max_iso_force.ravel()

    for batch_idx in range(batch_size):
        for muscle_idx in range(n_muscles):
            # Get values for this batch and muscle
            act = activation[batch_idx, 0, muscle_idx]
            ml_n = muscle_len_n[batch_idx, 0, muscle_idx]
            mv_n = muscle_vel_n[batch_idx, 0, muscle_idx]
            strain = muscle_strain[batch_idx, 0, muscle_idx]

            # Passive force-length
            flpe[batch_idx, 0, muscle_idx] = k_pe_flat[muscle_idx] * strain * strain

            # Active force-length
            val = 1.0 + (-ml_n * ml_n + 2.0 * ml_n - 1.0) / f_iso_n_den
            flce[batch_idx, 0, muscle_idx] = max(val, min_flce)

            # Compute velocity-dependent terms
            if ml_n > 1.0:
                a_rel_st = 0.41 * flce[batch_idx, 0, muscle_idx]
            else:
                a_rel_st = 0.41

            if act < q_crit:
                b_rel_st = 5.2 * (1.0 - 0.9 * ((act - q_crit) / (5e-3 - q_crit))) ** 2
            else:
                b_rel_st = 5.2

            # Inverse of slope at isometric point
            dfdvcon0 = act * (flce[batch_idx, 0, muscle_idx] + a_rel_st) / b_rel_st

            # Speed up computation
            f_x_a = flce[batch_idx, 0, muscle_idx] * act

            tmp_p_nom = f_x_a * 0.5
            tmp_p_den = s_as - dfdvcon0 * 2.0

            p1 = -tmp_p_nom / tmp_p_den
            p2 = (tmp_p_nom * tmp_p_nom) / tmp_p_den
            p3 = -1.5 * f_x_a

            # Compute active force based on velocity
            if mv_n < 0.0:
                nom = mv_n * act * a_rel_st + f_x_a * b_rel_st
                den = b_rel_st - mv_n
            else:
                nom = -p1 * p3 + p1 * s_as * mv_n + p2 - p3 * mv_n + s_as * mv_n * mv_n
                den = p1 + mv_n

            active_force[batch_idx, 0, muscle_idx] = max(nom / den, 0.0)
            force[batch_idx, 0, muscle_idx] = (active_force[batch_idx, 0, muscle_idx] + flpe[batch_idx, 0, muscle_idx]) * max_iso_force_flat[muscle_idx]

    return flpe, flce, active_force, force


@jit(nopython=True, cache=True, fastmath=True)
def compliant_tendon_normalized_muscle_vel_core(
    muscle_len_n, activation, active_force,
    f_iso_n_den, min_flce, q_crit, s_as
):
    """Core computation for CompliantTendonHillMuscle normalized muscle velocity.

    Computes the normalized muscle velocity for compliant tendon model.
    This is the most complex part of the compliant tendon model.

    Args:
        muscle_len_n: Normalized muscle length
        activation: Muscle activation
        active_force: Active force
        f_iso_n_den, min_flce, q_crit, s_as: Muscle parameters

    Returns:
        Normalized muscle velocity array
    """
    batch_size = muscle_len_n.shape[0]
    n_muscles = muscle_len_n.shape[2]

    muscle_vel_n = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)

    for batch_idx in range(batch_size):
        for muscle_idx in range(n_muscles):
            ml_n = muscle_len_n[batch_idx, 0, muscle_idx]
            act = activation[batch_idx, 0, muscle_idx]
            af = active_force[batch_idx, 0, muscle_idx]

            # Force-length of contractile element
            flce = max(1.0 + (-ml_n * ml_n + 2.0 * ml_n - 1.0) / f_iso_n_den, min_flce)

            # Compute velocity-dependent terms
            if ml_n < 1.0:
                a_rel_st = 0.41 * flce
            else:
                a_rel_st = 0.41

            if act < q_crit:
                b_rel_st = 5.2 * (1.0 - 0.9 * ((act - q_crit) / (5e-3 - q_crit))) ** 2
            else:
                b_rel_st = 5.2

            # Speed up computation
            f_x_a = flce * act
            dfdvcon0 = (f_x_a + act * a_rel_st) / b_rel_st

            p1 = -f_x_a * 0.5 / (s_as - dfdvcon0 * 2.0)
            p3 = -1.5 * f_x_a
            p2_containing_term = (4.0 * ((f_x_a * 0.5) ** 2) * (-s_as)) / (s_as - dfdvcon0 * 2.0)

            # Defensive code to avoid negative square root
            sqrt_term = (af ** 2 + 2.0 * af * p1 * s_as +
                        2.0 * af * p3 + p1 ** 2 * s_as ** 2 +
                        2.0 * p1 * p3 * s_as + p2_containing_term + p3 ** 2)
            sqrt_term = max(sqrt_term, 0.0)

            # Compute muscle velocity
            if af < f_x_a:
                nom = b_rel_st * (af - f_x_a)
                den = af + act * a_rel_st
            else:
                nom = -af + p1 * s_as - p3 - np.sqrt(sqrt_term)
                den = -2.0 * s_as

            muscle_vel_n[batch_idx, 0, muscle_idx] = nom / den

    return muscle_vel_n


@jit(nopython=True, cache=True, fastmath=True)
def compliant_tendon_hill_integrate_core(
    activation, muscle_len, muscle_len_n, musculotendon_len,
    d_activation, muscle_vel_n, dt,
    l0_ce, l0_se, l0_pe, vmax, max_iso_force, k_pe, k_se, min_activation
):
    """Core integration logic for CompliantTendonHillMuscle (Numba-optimized).

    Optimized muscle integration for compliant tendon model.

    Args:
        All muscle parameters and state variables

    Returns:
        New muscle state array
    """
    batch_size = muscle_len.shape[0]
    n_muscles = muscle_len.shape[2]

    # Pre-allocate output (7 states per muscle)
    new_state = np.zeros((batch_size, 7, n_muscles), dtype=np.float32)

    # Flatten parameter arrays
    l0_ce_flat = l0_ce.ravel()
    l0_se_flat = l0_se.ravel()
    l0_pe_flat = l0_pe.ravel()
    vmax_flat = vmax.ravel()
    max_iso_force_flat = max_iso_force.ravel()
    k_pe_flat = k_pe.ravel()

    # k_se is converted to array in wrapper, so just flatten
    k_se_flat = k_se.ravel()

    for batch_idx in range(batch_size):
        for muscle_idx in range(n_muscles):
            ml = muscle_len[batch_idx, 0, muscle_idx]
            ml_n = muscle_len_n[batch_idx, 0, muscle_idx]
            mt_len = musculotendon_len[batch_idx, 0, muscle_idx]
            d_act = d_activation[batch_idx, 0, muscle_idx]
            mv_n = muscle_vel_n[batch_idx, 0, muscle_idx]

            # Get parameters
            l_ce = l0_ce_flat[muscle_idx]
            l_se = l0_se_flat[muscle_idx]
            l_pe = l0_pe_flat[muscle_idx]
            v_max = vmax_flat[muscle_idx]
            max_force = max_iso_force_flat[muscle_idx]
            k_pe_val = k_pe_flat[muscle_idx]
            k_se_val = k_se_flat[min(muscle_idx, len(k_se_flat) - 1)]  # Handle broadcast case

            # Compute geometry
            tendon_len = mt_len - ml
            tendon_strain = max((tendon_len - l_se) / l_se, np.float32(0.0))
            muscle_strain = max((ml - l_pe) / l_ce, np.float32(0.0))

            # Compute forces
            flse = min(k_se_val * (tendon_strain ** 2), np.float32(1.0))
            flpe = k_pe_val * (muscle_strain ** 2)
            active_force = max(flse - flpe, np.float32(0.0))

            # Integrate activation
            new_activation = max(np.float32(min_activation), min(activation[batch_idx, 0, muscle_idx] + d_act * dt, np.float32(1.0)))

            # Integrate muscle length
            new_muscle_len = (ml_n + dt * mv_n) * l_ce

            # Compute muscle velocity
            muscle_vel = mv_n * v_max

            # Total force
            force = flse * max_force

            # Store state: [activation, muscle_len, muscle_vel, flpe, flse, active_force, force]
            new_state[batch_idx, 0, muscle_idx] = new_activation
            new_state[batch_idx, 1, muscle_idx] = new_muscle_len
            new_state[batch_idx, 2, muscle_idx] = muscle_vel
            new_state[batch_idx, 3, muscle_idx] = flpe
            new_state[batch_idx, 4, muscle_idx] = flse
            new_state[batch_idx, 5, muscle_idx] = active_force
            new_state[batch_idx, 6, muscle_idx] = force

    return new_state


# ==============================================================================
# Optimized Skeleton Kinematics
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def two_dof_arm_ode_core(pos0, pos1, vel0, vel1, inputs, endpoint_load,
                          m1, m2, L1, L2, L1g, L2g, I1, I2, coriolis_1, coriolis_2, c_viscosity):
    """Core ODE computation for TwoDofArm (Numba-optimized).

    Computes joint accelerations from torques and current state using
    inverse dynamics.

    Args:
        pos0, pos1: Joint positions (shoulder, elbow)
        vel0, vel1: Joint velocities
        inputs: Joint torques
        endpoint_load: External forces at endpoint
        m1, m2: Link masses
        L1, L2: Link lengths
        L1g, L2g: Centers of mass
        I1, I2: Moments of inertia
        coriolis_1, coriolis_2: Coriolis coefficients
        c_viscosity: Viscosity damping

    Returns:
        Joint accelerations array
    """
    batch_size = pos0.shape[0]
    acc = np.zeros((batch_size, 2), dtype=np.float32)

    for b in range(batch_size):
        p0 = pos0[b]
        p1 = pos1[b]
        v0 = vel0[b]
        v1 = vel1[b]
        inp0 = inputs[b, 0]
        inp1 = inputs[b, 1]
        eload0 = endpoint_load[b, 0]
        eload1 = endpoint_load[b, 1]

        pos_sum = p0 + p1
        c1 = np.cos(p0)
        c2 = np.cos(p1)
        c12 = np.cos(pos_sum)
        s1 = np.sin(p0)
        s2 = np.sin(p1)
        s12 = np.sin(pos_sum)

        # Inertia matrix components
        inertia_11_c = m1 * L1g * L1g + I1 + m2 * (L2g * L2g + L1 * L1) + I2
        inertia_12_c = m2 * L2g * L2g + I2
        inertia_22_c = m2 * L2g * L2g + I2
        inertia_11_m = 2.0 * m2 * L1 * L2g
        inertia_12_m = m2 * L1 * L2g

        inertia_11 = inertia_11_c + c2 * inertia_11_m
        inertia_12 = inertia_12_c + c2 * inertia_12_m
        inertia_21 = inertia_12
        inertia_22 = inertia_22_c

        # Coriolis torques
        cor1 = (coriolis_1 * s2 * (2.0 * v0 + v1)) * v1 + c_viscosity * v0
        cor2 = (coriolis_2 * s2 * v0) * v0 + c_viscosity * v1

        # Jacobian for endpoint loads
        jacobian_11 = -L1 * s1 - L2 * s12
        jacobian_12 = -L2 * s12
        jacobian_21 = L1 * c1 + L2 * c12
        jacobian_22 = L2 * c12

        # Apply external loads
        r_col = jacobian_11 * eload0 + jacobian_21 * eload1
        l_col = jacobian_12 * eload0 + jacobian_22 * eload1
        torque0 = inp0 + r_col
        torque1 = inp1 + l_col

        # Right-hand side
        rhs0 = -cor1 + torque0
        rhs1 = -cor2 + torque1

        # Inertia matrix inversion (2x2)
        det = inertia_11 * inertia_22 - inertia_12 * inertia_21
        inv_det = 1.0 / det

        inv_11 = inertia_22 * inv_det
        inv_12 = -inertia_12 * inv_det
        inv_21 = -inertia_21 * inv_det
        inv_22 = inertia_11 * inv_det

        # Compute acceleration
        acc[b, 0] = inv_11 * rhs0 + inv_12 * rhs1
        acc[b, 1] = inv_21 * rhs0 + inv_22 * rhs1

    return acc


@jit(nopython=True, cache=True, fastmath=True)
def two_dof_arm_path2cartesian_core(
    path_coordinates, path_fixation_body, sho, elb_wrt_sho, sho_vel, elb_vel, L1
):
    """Core path2cartesian computation for TwoDofArm (Numba-optimized).

    Converts muscle path coordinates from bone-relative to global Cartesian.

    Args:
        path_coordinates: Fixation point coordinates relative to bones
        path_fixation_body: Which bone each point is attached to
        sho: Shoulder angle
        elb_wrt_sho: Elbow angle relative to shoulder
        sho_vel: Shoulder angular velocity
        elb_vel: Elbow angular velocity
        L1: Upper arm length

    Returns:
        Tuple of (xy, dxy_dt, dxy_da) - position, velocity, and moment arms
    """
    batch_size = sho.shape[0]
    n_points = path_fixation_body.shape[2]

    xy = np.zeros((batch_size, 2, n_points), dtype=np.float32)
    dxy_dt = np.zeros((batch_size, 2, n_points), dtype=np.float32)
    dxy_da = np.zeros((batch_size, 2, 2, n_points), dtype=np.float32)

    for b in range(batch_size):
        sho_ang = sho[b, 0]
        elb_ang = elb_wrt_sho[b, 0] + sho_ang
        sv = sho_vel[b, 0]
        ev = elb_vel[b, 0] + sv

        sin_sho = np.sin(sho_ang)
        cos_sho = np.cos(sho_ang)
        sin_elb = np.sin(elb_ang)
        cos_elb = np.cos(elb_ang)

        elb_x = L1 * cos_sho
        elb_y = L1 * sin_sho

        for p in range(n_points):
            fixation = int(path_fixation_body[0, 0, p])
            coord_x = path_coordinates[0, 0, p]
            coord_y = path_coordinates[0, 1, p]

            # Select angle based on fixation body
            if fixation == 0:
                ang = 0.0
                ca = 1.0
                sa = 0.0
            elif fixation == 1:
                ang = -sho_ang
                ca = cos_sho
                sa = -sin_sho
            else:  # fixation == 2
                ang = -elb_ang
                ca = cos_elb
                sa = -sin_elb

            # Rotation matrix application
            rot1_x = ca
            rot1_y = sa
            rot2_x = -sa
            rot2_y = ca

            # Derivative of position wrt angle of fixation bone
            dx_da = -(coord_x * rot2_x + coord_y * rot2_y)
            dy_da = coord_x * rot1_x + coord_y * rot1_y

            # Position
            if fixation == 0:
                dx_da1 = 0.0
                dy_da1 = 0.0
            else:
                dx_da1 = dx_da
                dy_da1 = dy_da

            if fixation == 2:
                dx_da1 += -elb_y
                dy_da1 += elb_x

            dx_da2 = dx_da if fixation == 2 else 0.0
            dy_da2 = dy_da if fixation == 2 else 0.0

            # Store derivatives
            dxy_da[b, 0, 0, p] = dx_da1
            dxy_da[b, 1, 0, p] = dy_da1
            dxy_da[b, 0, 1, p] = dx_da2
            dxy_da[b, 1, 1, p] = dy_da2

            # Velocity
            dxy_dt[b, 0, p] = dx_da1 * sv + dx_da2 * ev
            dxy_dt[b, 1, p] = dy_da1 * sv + dy_da2 * ev

            # Position
            bone_origin_x = elb_x if fixation == 2 else 0.0
            bone_origin_y = elb_y if fixation == 2 else 0.0
            xy[b, 0, p] = dy_da + bone_origin_x
            xy[b, 1, p] = -dx_da + bone_origin_y

    return xy, dxy_dt, dxy_da


# ==============================================================================
# Enhanced Optimizations for CompliantTendonHillMuscle
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def compliant_tendon_force_velocity_residual(
    muscle_vel_n, muscle_len_n, activation, active_force,
    f_iso_n_den, min_flce, q_crit, s_as
):
    """Compute force-velocity residual for Newton-Raphson solver.

    This function computes F(v) - F_target, where F(v) is the force produced
    at velocity v. The Newton-Raphson solver finds v where this equals zero.

    Args:
        muscle_vel_n: Normalized muscle velocity (trial value)
        muscle_len_n: Normalized muscle length
        activation: Muscle activation
        active_force: Target active force
        f_iso_n_den, min_flce, q_crit, s_as: Muscle parameters

    Returns:
        Residual value (should be zero at solution)
    """
    batch_size = muscle_len_n.shape[0]
    n_muscles = muscle_len_n.shape[2]

    residual = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)

    for b in range(batch_size):
        for m in range(n_muscles):
            ml_n = muscle_len_n[b, 0, m]
            act = activation[b, 0, m]
            af = active_force[b, 0, m]
            mv_n = muscle_vel_n[b, 0, m]

            # Force-length
            flce = max(1.0 + (-ml_n * ml_n + 2.0 * ml_n - 1.0) / f_iso_n_den, min_flce)

            # Velocity-dependent terms
            if ml_n < 1.0:
                a_rel_st = 0.41 * flce
            else:
                a_rel_st = 0.41

            if act < q_crit:
                b_rel_st = 5.2 * (1.0 - 0.9 * ((act - q_crit) / (5e-3 - q_crit))) ** 2
            else:
                b_rel_st = 5.2

            f_x_a = flce * act

            # Compute force at this velocity
            if mv_n < 0.0:
                # Concentric contraction
                nom = mv_n * act * a_rel_st + f_x_a * b_rel_st
                den = b_rel_st - mv_n
            else:
                # Eccentric contraction
                dfdvcon0 = (f_x_a + act * a_rel_st) / b_rel_st
                p1 = -f_x_a * 0.5 / (s_as - dfdvcon0 * 2.0)
                p3 = -1.5 * f_x_a
                nom = -p1 * p3 + p1 * s_as * mv_n + p1 ** 2 * s_as ** 2 - p3 * mv_n + s_as * mv_n ** 2
                den = p1 + mv_n

            computed_force = max(nom / den, 0.0)
            residual[b, 0, m] = computed_force - af

    return residual


@jit(nopython=True, cache=True, fastmath=True)
def compliant_tendon_normalized_muscle_vel_newton(
    muscle_len_n, activation, active_force,
    f_iso_n_den, min_flce, q_crit, s_as,
    max_iterations=5, tolerance=1e-6
):
    """Newton-Raphson solver for normalized muscle velocity.

    This is an alternative to the analytical sqrt-based solver that uses
    iterative Newton-Raphson. Often faster because:
    1. No expensive sqrt operation
    2. Converges in 2-4 iterations typically
    3. Good initial guess from previous state

    Args:
        muscle_len_n, activation, active_force: Current muscle state
        f_iso_n_den, min_flce, q_crit, s_as: Muscle parameters
        max_iterations: Maximum Newton-Raphson iterations
        tolerance: Convergence tolerance

    Returns:
        Normalized muscle velocity
    """
    batch_size = muscle_len_n.shape[0]
    n_muscles = muscle_len_n.shape[2]

    muscle_vel_n = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)

    for b in range(batch_size):
        for m in range(n_muscles):
            ml_n = muscle_len_n[b, 0, m]
            act = activation[b, 0, m]
            af = active_force[b, 0, m]

            # Force-length
            flce = max(1.0 + (-ml_n * ml_n + 2.0 * ml_n - 1.0) / f_iso_n_den, min_flce)

            # Velocity-dependent terms
            if ml_n < 1.0:
                a_rel_st = 0.41 * flce
            else:
                a_rel_st = 0.41

            if act < q_crit:
                b_rel_st = 5.2 * (1.0 - 0.9 * ((act - q_crit) / (5e-3 - q_crit))) ** 2
            else:
                b_rel_st = 5.2

            f_x_a = flce * act

            # Initial guess: isometric (v=0) or based on whether shortening/lengthening
            if af < f_x_a:
                # Concentric - initial guess negative
                v_guess = -0.1
            else:
                # Eccentric - initial guess positive
                v_guess = 0.1

            # Newton-Raphson iterations
            for iteration in range(max_iterations):
                mv_n = v_guess

                # Compute force and derivative at current guess
                if mv_n < 0.0:
                    # Concentric
                    nom = mv_n * act * a_rel_st + f_x_a * b_rel_st
                    den = b_rel_st - mv_n
                    force = nom / den

                    # Derivative dF/dv (quotient rule)
                    dnom_dv = act * a_rel_st
                    dden_dv = -1.0
                    dforce_dv = (dnom_dv * den - nom * dden_dv) / (den ** 2)
                else:
                    # Eccentric
                    dfdvcon0 = (f_x_a + act * a_rel_st) / b_rel_st
                    p1 = -f_x_a * 0.5 / (s_as - dfdvcon0 * 2.0)
                    p3 = -1.5 * f_x_a

                    nom = -p1 * p3 + p1 * s_as * mv_n + p1 ** 2 * s_as ** 2 - p3 * mv_n + s_as * mv_n ** 2
                    den = p1 + mv_n
                    force = nom / den

                    # Derivative
                    dnom_dv = p1 * s_as - p3 + 2.0 * s_as * mv_n
                    dden_dv = 1.0
                    dforce_dv = (dnom_dv * den - nom * dden_dv) / (den ** 2)

                force = max(force, 0.0)

                # Newton-Raphson update
                residual = force - af

                if abs(residual) < tolerance:
                    break

                if abs(dforce_dv) > 1e-10:
                    v_guess = v_guess - residual / dforce_dv
                    # Bound the guess to reasonable range
                    v_guess = max(-2.0, min(2.0, v_guess))
                else:
                    break

            muscle_vel_n[b, 0, m] = v_guess

    return muscle_vel_n


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def compliant_tendon_normalized_muscle_vel_core_parallel(
    muscle_len_n, activation, active_force,
    f_iso_n_den, min_flce, q_crit, s_as
):
    """Parallel version of compliant tendon velocity calculation.

    Uses Numba's parallel features to process batches in parallel.
    Most beneficial when batch_size > 4.
    """
    batch_size = muscle_len_n.shape[0]
    n_muscles = muscle_len_n.shape[2]

    muscle_vel_n = np.zeros((batch_size, 1, n_muscles), dtype=np.float32)

    for batch_idx in numba.prange(batch_size):  # Parallel loop
        for muscle_idx in range(n_muscles):
            ml_n = muscle_len_n[batch_idx, 0, muscle_idx]
            act = activation[batch_idx, 0, muscle_idx]
            af = active_force[batch_idx, 0, muscle_idx]

            # Force-length of contractile element
            flce = max(1.0 + (-ml_n * ml_n + 2.0 * ml_n - 1.0) / f_iso_n_den, min_flce)

            # Compute velocity-dependent terms
            if ml_n < 1.0:
                a_rel_st = 0.41 * flce
            else:
                a_rel_st = 0.41

            if act < q_crit:
                b_rel_st = 5.2 * (1.0 - 0.9 * ((act - q_crit) / (5e-3 - q_crit))) ** 2
            else:
                b_rel_st = 5.2

            # Speed up computation
            f_x_a = flce * act
            dfdvcon0 = (f_x_a + act * a_rel_st) / b_rel_st

            p1 = -f_x_a * 0.5 / (s_as - dfdvcon0 * 2.0)
            p3 = -1.5 * f_x_a
            p2_containing_term = (4.0 * ((f_x_a * 0.5) ** 2) * (-s_as)) / (s_as - dfdvcon0 * 2.0)

            # Defensive code to avoid negative square root
            sqrt_term = (af ** 2 + 2.0 * af * p1 * s_as +
                        2.0 * af * p3 + p1 ** 2 * s_as ** 2 +
                        2.0 * p1 * p3 * s_as + p2_containing_term + p3 ** 2)
            sqrt_term = max(sqrt_term, 0.0)

            # Compute muscle velocity
            if af < f_x_a:
                nom = b_rel_st * (af - f_x_a)
                den = af + act * a_rel_st
            else:
                nom = -af + p1 * s_as - p3 - np.sqrt(sqrt_term)
                den = -2.0 * s_as

            muscle_vel_n[batch_idx, 0, muscle_idx] = nom / den

    return muscle_vel_n


# ==============================================================================
# Module Info
# ==============================================================================

__all__ = [
    'mujoco_hill_bump',
    'mujoco_hill_integrate_core',
    'rigid_tendon_hill_integrate_core',
    'compliant_tendon_normalized_muscle_vel_core',
    'compliant_tendon_hill_integrate_core',
    'compliant_tendon_normalized_muscle_vel_newton',
    'compliant_tendon_normalized_muscle_vel_core_parallel',
    'two_dof_arm_ode_core',
    'two_dof_arm_path2cartesian_core',
]


def get_numba_info():
    """Print information about Numba configuration."""
    print("=" * 70)
    print("Numba Optimization Module")
    print("=" * 70)
    print(f"Numba version: {numba.__version__}")
    print(f"NumPy version: {np.__version__}")
    print()
    print("Available optimized functions:")
    for func_name in __all__:
        print(f"  • {func_name}")
    print()
    print("To use these optimizations, import optimized muscle/skeleton")
    print("classes that utilize these JIT-compiled functions.")
    print("=" * 70)
