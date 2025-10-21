"""
Utility functions for npmotornet.

This module provides helper functions for efficient simulation,
including pre-allocation utilities and performance optimization tools.
"""

import numpy as np


class TrajectoryBuffer:
    """Pre-allocated buffer for storing simulation trajectories efficiently.

    Instead of concatenating arrays at each timestep (which creates new arrays),
    this class pre-allocates memory and fills it in-place for better performance.

    Example:
        >>> buffer = TrajectoryBuffer(n_steps=1000, batch_size=1, state_dim=4)
        >>> for i in range(n_steps):
        ...     buffer.append(effector.states["joint"], step=i)
        >>> trajectory = buffer.get_data()

    Args:
        n_steps: Integer, number of timesteps to allocate space for.
        batch_size: Integer, batch size dimension.
        state_dim: Integer, dimensionality of the state vector.
        dtype: NumPy dtype for the buffer. Default: np.float32.
    """

    def __init__(self, n_steps: int, batch_size: int, state_dim: int, dtype=np.float32):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.dtype = dtype
        self.buffer = np.zeros((n_steps, batch_size, state_dim), dtype=dtype)
        self.current_step = 0

    def append(self, state, step=None):
        """Append a state to the buffer.

        Args:
            state: Array of shape (batch_size, state_dim) to append.
            step: Integer, optional step index. If None, uses internal counter.
        """
        if step is None:
            step = self.current_step
            self.current_step += 1

        if step >= self.n_steps:
            raise IndexError(f"Step {step} exceeds buffer size {self.n_steps}")

        self.buffer[step] = state

    def get_data(self, squeeze_batch=True):
        """Get the buffered trajectory data.

        Args:
            squeeze_batch: If True and batch_size==1, squeeze the batch dimension.

        Returns:
            Array of shape (n_steps, batch_size, state_dim) or (n_steps, state_dim)
            if squeeze_batch=True and batch_size==1.
        """
        if squeeze_batch and self.batch_size == 1:
            return self.buffer[:, 0, :]
        return self.buffer

    def reset(self):
        """Reset the buffer to zeros and reset the step counter."""
        self.buffer.fill(0)
        self.current_step = 0


class MultiStateBuffer:
    """Pre-allocated buffers for multiple state types.

    Efficiently stores joint, muscle, geometry, and cartesian states
    during simulation.

    Example:
        >>> buffer = MultiStateBuffer.from_effector(effector, n_steps=1000)
        >>> for i in range(n_steps):
        ...     effector.step(action)
        ...     buffer.append_all(effector.states, step=i)
        >>> trajectories = buffer.get_all()

    Args:
        n_steps: Integer, number of timesteps to allocate space for.
        batch_size: Integer, batch size dimension.
        joint_dim: Integer, joint state dimensionality.
        muscle_dim: Integer, muscle state dimensionality.
        geometry_dim: Integer, geometry state dimensionality.
        n_muscles: Integer, number of muscles.
        dtype: NumPy dtype for the buffers. Default: np.float32.
    """

    def __init__(
        self,
        n_steps: int,
        batch_size: int,
        joint_dim: int,
        muscle_dim: int,
        geometry_dim: int,
        n_muscles: int,
        dtype=np.float32
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.dtype = dtype
        self.current_step = 0

        # Pre-allocate buffers for each state type
        self.buffers = {
            "joint": np.zeros((n_steps, batch_size, joint_dim), dtype=dtype),
            "muscle": np.zeros((n_steps, batch_size, muscle_dim, n_muscles), dtype=dtype),
            "geometry": np.zeros((n_steps, batch_size, geometry_dim, n_muscles), dtype=dtype),
            "cartesian": np.zeros((n_steps, batch_size, joint_dim), dtype=dtype),
        }

    @classmethod
    def from_effector(cls, effector, n_steps: int, batch_size: int = 1):
        """Create a MultiStateBuffer from an effector object.

        Args:
            effector: An Effector instance to infer dimensions from.
            n_steps: Integer, number of timesteps to allocate.
            batch_size: Integer, batch size. Default: 1.

        Returns:
            MultiStateBuffer instance with appropriate dimensions.
        """
        return cls(
            n_steps=n_steps,
            batch_size=batch_size,
            joint_dim=effector.state_dim,
            muscle_dim=effector.muscle_state_dim,
            geometry_dim=effector.geometry_state_dim,
            n_muscles=effector.n_muscles,
        )

    def append_all(self, states: dict, step=None):
        """Append all states from a state dictionary.

        Args:
            states: Dictionary with keys "joint", "muscle", "geometry", "cartesian".
            step: Integer, optional step index. If None, uses internal counter.
        """
        if step is None:
            step = self.current_step
            self.current_step += 1

        if step >= self.n_steps:
            raise IndexError(f"Step {step} exceeds buffer size {self.n_steps}")

        for key in ["joint", "muscle", "geometry", "cartesian"]:
            if key in states and states[key] is not None:
                self.buffers[key][step] = states[key]

    def get_all(self, squeeze_batch=True):
        """Get all buffered trajectories.

        Args:
            squeeze_batch: If True and batch_size==1, squeeze the batch dimension.

        Returns:
            Dictionary mapping state names to their trajectory arrays.
        """
        if squeeze_batch and self.batch_size == 1:
            return {key: buf[:, 0] for key, buf in self.buffers.items()}
        return self.buffers.copy()

    def reset(self):
        """Reset all buffers to zeros and reset the step counter."""
        for buf in self.buffers.values():
            buf.fill(0)
        self.current_step = 0


def simulate_with_buffer(effector, actions, n_steps=None):
    """Efficiently simulate an effector with pre-allocated buffers.

    This is a convenience function that handles buffer creation and
    simulation in one call, using efficient pre-allocation.

    Args:
        effector: An Effector instance to simulate.
        actions: Array of shape (n_steps, n_muscles) or (n_muscles,) for constant action.
        n_steps: Integer, number of steps to simulate. If None, inferred from actions shape.

    Returns:
        Dictionary with keys "joint", "muscle", "geometry", "cartesian" mapping to
        trajectory arrays of shape (n_steps, state_dim) or (n_steps, state_dim, n_muscles).

    Example:
        >>> # Constant action
        >>> trajectories = simulate_with_buffer(effector, actions=[0.5, 0.5, 0.5, 0.5], n_steps=1000)
        >>> # Time-varying actions
        >>> actions = np.random.uniform(0, 1, (1000, 4))
        >>> trajectories = simulate_with_buffer(effector, actions)
    """
    # Handle constant vs time-varying actions
    actions = np.array(actions, dtype=np.float32)

    # Ensure actions is at least 1D
    if actions.ndim == 0:
        actions = actions.reshape(1)

    # Check if this is a constant action (1D or 2D with batch_size=1)
    if actions.ndim == 1:
        # 1D constant action - replicate for all timesteps
        if n_steps is None:
            raise ValueError("n_steps must be provided for constant actions")
        actions = np.tile(actions, (n_steps, 1))
    elif actions.ndim == 2:
        # Could be: (n_steps, n_muscles) time-varying OR (1, n_muscles) constant
        if actions.shape[0] == 1:
            # Constant action with batch dimension
            if n_steps is None:
                raise ValueError("n_steps must be provided for constant actions")
            actions = np.tile(actions, (n_steps, 1))
        else:
            # Time-varying actions
            if n_steps is None:
                n_steps = actions.shape[0]
            elif n_steps != actions.shape[0]:
                raise ValueError(f"n_steps ({n_steps}) doesn't match actions shape ({actions.shape[0]})")
    else:
        raise ValueError(f"actions must be 1D or 2D, got shape {actions.shape}")

    # Get batch size from current state (assume effector has been reset)
    batch_size = effector.states["joint"].shape[0] if effector.states["joint"] is not None else 1

    # Create buffer
    buffer = MultiStateBuffer.from_effector(effector, n_steps=n_steps + 1, batch_size=batch_size)

    # Store initial state
    buffer.append_all(effector.states, step=0)

    # Run simulation
    for i in range(n_steps):
        action = actions[i:i+1] if batch_size == 1 else actions[i]
        effector.step(action)
        buffer.append_all(effector.states, step=i + 1)

    return buffer.get_all(squeeze_batch=(batch_size == 1))


def print_performance_tips():
    """Print performance optimization tips for npmotornet users."""
    tips = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           npmotornet Performance Optimization Tips           ║
    ╚══════════════════════════════════════════════════════════════╝

    1. Use Pre-allocated Buffers
       ✗ Slow:  trajectories = []; trajectories.append(state)
       ✓ Fast:  buffer = MultiStateBuffer.from_effector(effector, n_steps)

    2. Avoid Array Concatenation in Loops
       ✗ Slow:  states = np.concatenate([states, new_state])
       ✓ Fast:  Use TrajectoryBuffer or pre-allocate full array

    3. Use Larger Timesteps When Possible
       • Rigid tendon: dt=0.01s is typically sufficient
       • Compliant tendon: dt=0.0002s may be needed for stability
       • Larger dt = fewer iterations = faster simulation

    4. Batch Simulations When Possible
       • Run multiple parameter sets in parallel with batch_size > 1
       • Better utilizes vectorization

    5. Use Euler Integration for Speed (when applicable)
       • Runge-Kutta 4 is 4x slower but more accurate
       • For many applications, Euler with small dt is sufficient

    6. Consider n_ministeps Carefully
       • Higher n_ministeps = more accurate but slower
       • n_ministeps=1 is often sufficient for dt=0.01s

    For more tips, see the documentation or examples/performance_demo.py
    """
    print(tips)
