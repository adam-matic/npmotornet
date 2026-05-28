"""
Two Degrees-of-Freedom Arm Example
===================================
This example demonstrates how to use a 2DOF planar arm skeleton with
Hill-type muscles. It shows joint-to-cartesian transformations and
basic arm movements.

Converted from MotorNet PyTorch to npmotornet (NumPy).
"""

import numpy as np
import sys
import os

# Add parent directory to path to import npmotornet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import npmotornet.skeleton as skeleton
import npmotornet.muscle as muscle
import npmotornet.effector as effector


def simulate_arm_motion(effector_obj, action, movement_duration):
    """Simulate arm motion with constant muscle activation."""
    # Start with shoulder and elbow at 45 degrees
    initial_angles = np.array([[np.pi/4, np.pi/4]])
    effector_obj.reset(options={"joint_state": initial_angles})

    j_state = effector_obj.states["joint"]
    cart_state = effector_obj.states["cartesian"]

    n_steps = int(movement_duration / effector_obj.dt)
    action = np.ones((1, effector_obj.n_muscles)) * np.array(action)

    for _ in range(n_steps):
        effector_obj.step(action)
        j_state = np.concatenate([j_state, effector_obj.states["joint"]], axis=0)
        cart_state = np.concatenate([cart_state, effector_obj.states["cartesian"]], axis=0)

    return j_state, cart_state


def main():
    print("=" * 60)
    print("Two Degrees-of-Freedom Arm Example")
    print("=" * 60)

    # Create a 2DOF arm skeleton
    print("\n1. Creating 2DOF arm skeleton...")
    arm_skeleton = skeleton.TwoDofArm(
        m1=1.82,  # mass of upper arm (kg)
        m2=1.43,  # mass of forearm (kg)
        l1=0.309,  # length of upper arm (m)
        l2=0.333,  # length of forearm (m)
        l1g=0.135,  # center of mass position, upper arm (m)
        l2g=0.165,  # center of mass position, forearm (m)
        i1=0.051,  # moment of inertia, upper arm (kg.m^2)
        i2=0.057,  # moment of inertia, forearm (kg.m^2)
    )

    print(f"   Skeleton degrees of freedom: {arm_skeleton.dof}")
    print(f"   Skeleton space dimensionality: {arm_skeleton.space_dim}")
    print(f"   Upper arm length: {arm_skeleton.l1:.3f} m")
    print(f"   Forearm length: {arm_skeleton.l2:.3f} m")

    # Test joint-to-cartesian transformation
    print("\n2. Testing joint-to-cartesian transformation...")

    # Test with shoulder=0, elbow=0
    joint_config = np.array([[0.0, 0.0, 0.0, 0.0]])  # [shoulder_pos, elbow_pos, shoulder_vel, elbow_vel]
    cart_state = arm_skeleton.joint2cartesian(joint_config)
    endpoint_pos = cart_state[0, :2]
    print(f"   Joint angles: [0°, 0°] -> Endpoint: [{endpoint_pos[0]:.4f}, {endpoint_pos[1]:.4f}] m")

    # Test with shoulder=90°, elbow=0°
    joint_config = np.array([[np.pi/2, 0.0, 0.0, 0.0]])
    cart_state = arm_skeleton.joint2cartesian(joint_config)
    endpoint_pos = cart_state[0, :2]
    print(f"   Joint angles: [90°, 0°] -> Endpoint: [{endpoint_pos[0]:.4f}, {endpoint_pos[1]:.4f}] m")

    # Test with shoulder=45°, elbow=45°
    joint_config = np.array([[np.pi/4, np.pi/4, 0.0, 0.0]])
    cart_state = arm_skeleton.joint2cartesian(joint_config)
    endpoint_pos = cart_state[0, :2]
    print(f"   Joint angles: [45°, 45°] -> Endpoint: [{endpoint_pos[0]:.4f}, {endpoint_pos[1]:.4f}] m")

    # Create simple arm effector with muscles
    print("\n3. Creating arm effector with 2 simple muscles...")

    # Use simpler ReLu muscles for this demo
    simple_muscle = muscle.ReluMuscle()
    arm_effector = effector.Effector(
        skeleton=arm_skeleton,
        muscle=simple_muscle,
        timestep=0.01
    )

    # Add a shoulder flexor (pulls upper arm forward)
    arm_effector.add_muscle(
        path_fixation_body=[0, 1],  # from worldspace to upper arm
        path_coordinates=[[0.05, 0.05], [0.1, 0.0]],
        name='ShoulderFlexor',
        max_isometric_force=100.0
    )

    # Add an elbow flexor (pulls forearm toward upper arm)
    arm_effector.add_muscle(
        path_fixation_body=[1, 2],  # from upper arm to forearm
        path_coordinates=[[0.2, 0.02], [0.1, 0.0]],
        name='ElbowFlexor',
        max_isometric_force=80.0
    )

    print(f"   Added {arm_effector.n_muscles} muscles: {arm_effector.muscle_name}")

    # Simulate shoulder flexion
    print("\n4. Simulating shoulder flexion...")
    initial_angles = np.array([[np.pi/6, np.pi/6]])  # 30°, 30°
    arm_effector.reset(options={"joint_state": initial_angles})

    initial_cart = arm_effector.states["cartesian"]
    print(f"   Initial endpoint: [{initial_cart[0,0]:.4f}, {initial_cart[0,1]:.4f}] m")

    joint_state, cart_state = simulate_arm_motion(
        arm_effector,
        [1.0, 0.0],  # Activate shoulder flexor only
        0.5  # 500ms movement
    )

    final_angles = joint_state[-1, :2] * 180 / np.pi
    final_endpoint = cart_state[-1, :2]
    print(f"   Final joint angles: [{final_angles[0]:.1f}°, {final_angles[1]:.1f}°]")
    print(f"   Final endpoint: [{final_endpoint[0]:.4f}, {final_endpoint[1]:.4f}] m")

    # Simulate elbow flexion
    print("\n5. Simulating elbow flexion...")
    arm_effector.reset(options={"joint_state": initial_angles})

    joint_state, cart_state = simulate_arm_motion(
        arm_effector,
        [0.0, 1.0],  # Activate elbow flexor only
        0.5
    )

    final_angles = joint_state[-1, :2] * 180 / np.pi
    final_endpoint = cart_state[-1, :2]
    print(f"   Final joint angles: [{final_angles[0]:.1f}°, {final_angles[1]:.1f}°]")
    print(f"   Final endpoint: [{final_endpoint[0]:.4f}, {final_endpoint[1]:.4f}] m")

    # Use pre-built arm with realistic muscle wrapping
    print("\n6. Using pre-built RigidTendonArm26...")
    realistic_arm = effector.RigidTendonArm26(
        muscle=muscle.RigidTendonHillMuscle(),
        timestep=0.01
    )

    print(f"   Number of muscles: {realistic_arm.n_muscles}")
    print(f"   Muscle names: {realistic_arm.muscle_name}")

    # Initialize and check muscle configuration
    realistic_arm.reset(options={"joint_state": np.array([[np.pi/4, np.pi/4]])})
    initial_muscle_state = realistic_arm.states["muscle"]

    print("\n   Initial muscle lengths:")
    for i, name in enumerate(realistic_arm.muscle_name):
        length = initial_muscle_state[0, 1, i]
        print(f"     {name:20s}: {length:.4f} m")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
