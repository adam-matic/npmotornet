"""
NumPy-based MotorNet
A conversion of MotorNet's core components from PyTorch to NumPy.
Includes skeleton, muscle, and effector modules for biomechanical simulation.

Performance optimizations:
  - Numba JIT-compiled functions for Hill muscle models and arm kinematics
  - Pre-allocation utilities for efficient simulation
  - Optimized vectorized operations
"""

from . import skeleton
from . import muscle
from . import effector
from . import utils

# Optional Numba optimizations (requires numba to be installed)
try:
    from . import numba_optimized
    from . import numba_classes
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

__version__ = "0.1.0"
