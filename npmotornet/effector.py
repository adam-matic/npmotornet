import numpy as np
from typing import Union, Any

# Try to import gymnasium, but provide fallback if not available
try:
    from gymnasium.utils import seeding
except ImportError:
    # Simple fallback for seeding if gymnasium is not installed
    class seeding:
        @staticmethod
        def np_random(seed=None):
            """Fallback random number generator initialization."""
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
            return rng, seed

from npmotornet.skeleton import TwoDofArm, PointMass
from npmotornet.muscle import CompliantTendonHillMuscle, ReluMuscle


class Effector:
  """Base class for `Effector` objects.

  Args:
    skeleton: A :class:`npmotornet.skeleton.Skeleton` object class or subclass. This defines the type of
      skeleton that the muscles will wrap around.
    muscle: A :class:`npmotornet.muscle.Muscle` object class or subclass. This defines the type of
      muscle that will be added each time the :meth:`add_muscle` method is called.
    name: `String`, the name of the object instance.
    timestep: `Float`, size of a single timestep (in sec).
    n_ministeps" `Integer`, number of integration ministeps per timestep. This assumes the action input is constant
      across ministeps.
    integration_method: `String`, the integration method to use. Options:
      - "euler": First-order Euler method (fixed timestep)
      - "rk4", "rungekutta4", "runge-kutta4", "runge-kutta-4": Fourth-order Runge-Kutta (fixed timestep)
      - "rkf45", "runge-kutta-fehlberg", "adaptive": Runge-Kutta-Fehlberg 4/5 (adaptive timestep)
      - "dopri5", "dormand-prince": Dormand-Prince 5/4 (adaptive timestep)
      This argument is case-insensitive.
    damping: `Float`, the damping coefficient applied at each joint, proportional to joint velocity. This value
      should be positive to reduce joint torques proportionally to joint velocity.
    pos_upper_bound: `Float`, `list` or `tuple`, indicating the upper boundary of the skeleton's joint position.
      This should be a `n`-elements vector or `list`, with `n` the number of joints of the skeleton. For instance,
      for a two degrees-of-freedom arm, we would have `n=2`.
    pos_lower_bound: `Float`, `list` or `tuple`, indicating the lower boundary of the skeleton's joint position.
      This should be a `n`-elements vector or `list`, with `n` the number of joints of the skeleton. For instance,
      for a two degrees-of-freedom arm, we would have `n=2`.
    vel_upper_bound: `Float`, `list` or `tuple`, indicating the upper boundary of the skeleton's joint velocity.
      This should be a `n`-elements vector or `list`, with `n` the number of joints of the skeleton. For instance,
      for a two degrees-of-freedom arm, we would have `n=2`.
    vel_lower_bound: `Float`, `list` or `tuple`, indicating the lower boundary of the skeleton's joint velocity.
      This should be a `n`-elements vector or `list`, with `n` the number of joints of the skeleton. For instance,
      for a two degrees-of-freedom arm, we would have `n=2`.
    adaptive_tolerance: `Float`, error tolerance for adaptive timestep methods (default: 1e-6). Lower values
      increase accuracy but may slow down simulation. Only used with adaptive integration methods.
    adaptive_min_dt: `Float`, minimum allowed timestep for adaptive methods (default: 1e-6). This prevents
      the timestep from becoming too small in stiff regions.
    adaptive_max_dt: `Float`, maximum allowed timestep for adaptive methods (default: 0.1). This prevents
      the timestep from growing too large in smooth regions.
  """

  def __init__(
    self,
    skeleton,
    muscle,
    name: str = 'Effector',
    n_ministeps: int = 1,
    timestep: float = 0.01,
    integration_method: str = 'euler',
    damping: float = 0.,
    pos_lower_bound: Union[float, list, tuple] = None,
    pos_upper_bound: Union[float, list, tuple] = None,
    vel_lower_bound: Union[float, list, tuple] = None,
    vel_upper_bound: Union[float, list, tuple] = None,
    adaptive_tolerance: float = 1e-6,
    adaptive_min_dt: float = 1e-6,
    adaptive_max_dt: float = 0.1,
  ):

    self.__name__ = name
    self.damping = np.array(damping, dtype=np.float32)
    self.skeleton = skeleton
    self.dof = self.skeleton.dof
    self.space_dim = self.skeleton.space_dim
    self.state_dim = self.skeleton.state_dim
    self.output_dim = self.skeleton.output_dim
    self.n_ministeps = n_ministeps
    self.dt = timestep
    self.minidt = self.dt / self.n_ministeps
    self.half_minidt = self.minidt / 2  # to reduce online calculations for RK4 integration
    self.integration_method = integration_method.casefold()  # make string fully in lower case
    self._np_random = None
    self.seed = None

    # Adaptive timestep control parameters
    self.adaptive_tolerance = adaptive_tolerance
    self.adaptive_min_dt = adaptive_min_dt
    self.adaptive_max_dt = adaptive_max_dt
    self.current_adaptive_dt = self.minidt  # Initialize to minidt
    self.adaptive_step_count = 0
    self.adaptive_rejected_count = 0

    # handle position & velocity ranges
    pos_lower_bound = self.skeleton.pos_lower_bound if pos_lower_bound is None else pos_lower_bound
    pos_upper_bound = self.skeleton.pos_upper_bound if pos_upper_bound is None else pos_upper_bound
    vel_lower_bound = self.skeleton.vel_lower_bound if vel_lower_bound is None else vel_lower_bound
    vel_upper_bound = self.skeleton.vel_upper_bound if vel_upper_bound is None else vel_upper_bound
    pos_bounds = self._set_state_limit_bounds(lb=pos_lower_bound, ub=pos_upper_bound)
    vel_bounds = self._set_state_limit_bounds(lb=vel_lower_bound, ub=vel_upper_bound)
    pos_range = np.array(pos_bounds[:, 0] - pos_bounds[:, 1], dtype=np.float32)
    vel_range = np.array(vel_bounds[:, 0] - vel_bounds[:, 1], dtype=np.float32)

    self.pos_upper_bound = pos_bounds[:, 1]
    self.pos_lower_bound = pos_bounds[:, 0]
    self.vel_upper_bound = vel_bounds[:, 1]
    self.vel_lower_bound = vel_bounds[:, 0]
    self.pos_range_bound = pos_range
    self.vel_range_bound = vel_range

    self.skeleton.build(
      timestep=self.dt,
      pos_upper_bound=self.pos_upper_bound,
      pos_lower_bound=self.pos_lower_bound,
      vel_upper_bound=self.vel_upper_bound,
      vel_lower_bound=self.vel_lower_bound,
    )

    # initialize muscle system
    self.muscle = muscle
    self.force_index = self.muscle.state_name.index('force')  # column index of muscle state containing output force
    self.MusclePaths = []  # a list of all the muscle paths
    self.n_muscles = 0
    self.input_dim = 0
    self.muscle_name = []
    self.muscle_state_dim = self.muscle.state_dim
    self.geometry_state_dim = 2 + self.dof  # musculotendon length & velocity + as many moments as dofs
    self.geometry_state_name = [
      'musculotendon length',
      'musculotendon velocity'
      ] + [
      'moment for joint ' + str(d) for d in range(self.dof)
      ]
    self.tobuild__muscle = self.muscle.to_build_dict
    self.tobuild__default = self.muscle.to_build_dict_default

    # these attributes hold numpy versions of the variables, which are easier to manipulate in the `build()` method
    self._path_fixation_body = np.empty((1, 1, 0)).astype('float32')
    self._path_coordinates = np.empty((1, self.skeleton.space_dim, 0)).astype('float32')
    self._muscle_index = np.empty(0).astype('float32')
    self._muscle_transitions = None
    self._row_splits = None
    # these attributes will hold array versions of the above
    self.path_fixation_body = None
    self.path_coordinates = None
    self.muscle_index = None
    self.muscle_transitions = None
    self.row_splits = None
    self.section_splits = None
    self._muscle_config_is_empty = True

    self.default_endpoint_load = np.zeros((1, self.skeleton.space_dim), dtype=np.float32)
    self.default_joint_load = np.zeros((1, self.skeleton.dof), dtype=np.float32)

    if self.integration_method == 'euler':
      self._integrate = self._euler
    elif self.integration_method in ('rk4', 'rungekutta4', 'runge-kutta4', 'runge-kutta-4'):  # tuple faster than set
      self._integrate = self._rungekutta4
    elif self.integration_method in ('rkf45', 'runge-kutta-fehlberg', 'adaptive'):
      self._integrate = self._rkf45_adaptive
    elif self.integration_method in ('dopri5', 'dormand-prince'):
      self._integrate = self._dopri5_adaptive
    else:
      raise ValueError("Provided integration method not recognized : {}".format(self.integration_method))

    self.states = {key: None for key in ["joint", "cartesian", "muscle", "geometry", "fingertip"]}

  def step(self, action, **kwargs):
    endpoint_load = kwargs.get('endpoint_load', self.default_endpoint_load)
    joint_load = kwargs.get('joint_load', self.default_joint_load)

    action = action if isinstance(action, np.ndarray) else np.array(action, dtype=np.float32)
    a = self.muscle.clip_activation(action)

    for _ in range(self.n_ministeps):
      self.integrate(a, endpoint_load, joint_load)

  def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
    """Sets initial states (joint, cartesian, muscle, geometry) that are biomechanically compatible with each other.

    Args:
      seed: `Integer`, the seed that is used to initialize the environment's PRNG (`np_random`).
        If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
        a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
        However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
        If you pass an integer, the PRNG will be reset even if it already exists.
        Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
      options: `Dictionary`, optional kwargs. This is mainly useful to pass `batch_size` and `joint_state` kwargs if
        desired, as described below.

    Options:
      - **batch_size**: `Integer`, the desired batch size. Default: `1`.
      - **joint_state**: The joint state from which the other state values are inferred. If `None`, random initial
        joint states are drawn, from which the other state values are inferred. Default: `None`.
    """
    # Initialize the RNG if the seed is manually passed
    if seed is not None:
      self._np_random, self.seed = seeding.np_random(seed)

    options = {} if options is None else options
    batch_size: int = options.get('batch_size', 1)
    joint_state: np.ndarray | None = options.get('joint_state', None)

    if joint_state is not None:
      joint_state_shape = np.shape(joint_state)
      if joint_state_shape[0] > 1:
        batch_size = joint_state_shape[0]

    joint0 = self._parse_initial_joint_state(joint_state=joint_state, batch_size=batch_size)
    geometry0 = self.get_geometry(joint0)
    muscle0 = self.muscle.get_initial_muscle_state(batch_size=batch_size, geometry_state=geometry0)
    states = {"joint": joint0, "muscle": muscle0, "geometry": geometry0}

    self._set_state(states)

  @property
  def np_random(self) -> np.random.Generator:
    """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

    Returns:
      Instances of `np.random.Generator`
    """
    if self._np_random is None:
      self._np_random, _ = seeding.np_random()
    return self._np_random

  @np_random.setter
  def np_random(self, rng: np.random.Generator):
    self._np_random = rng

  def add_muscle(self, path_fixation_body: list, path_coordinates: list, name: str = None, **kwargs):
    """Adds a muscle to the effector.

    Args:
      path_fixation_body: `List`, containing the index of the fixation body (or fixation bone) for each fixation
        point in the muscle. The index `0` always stands for the worldspace, *i.e.* a fixation point outside of
        the skeleton.
      path_coordinates:  A `List` of `lists`. There should be as many lists in the main list as there are fixation
        points for that muscle. Each nested `list` within the main `list` should contain a series of `n`
        coordinate `float` values, with `n` being the dimensionality of the worldspace. For instance, in a 2D
        environment, we would need two coordinate values. The coordinate system of each bone is centered on that
        bone's origin point. Its first dimension is always alongside the length of the bone, and the next
        dimension(s) proceed(s) from there on orthogonally to that first dimension.
      name: `String`, the name to give to the muscle being added. If ``None`` is given, the name defaults to
        `"muscle_m"`, with `m` being a counter for the number of muscles.
      **kwargs: This is used to pass the set of properties required to build the type of muscle specified at
        initialization. What it should contain varies depending on the muscle type being used. A `TypeError`
        will be raised by this method if a muscle property pertaining to the muscle type specified is missing.

    Raises:
      TypeError: If an argument is missing to build the type of muscle specified at initialization.
    """

    path_fixation_body = np.array(path_fixation_body).astype('float32').reshape((1, 1, -1))
    n_points = path_fixation_body.size
    path_coordinates = np.array(path_coordinates).astype('float32').T[np.newaxis, :, :]
    assert path_coordinates.shape[1] == self.skeleton.space_dim
    assert path_coordinates.shape[2] == n_points
    self.n_muscles += 1
    self.input_dim += self.muscle.input_dim

    # path segments & coordinates should be a (batch_size * n_coordinates  * n_segments * (n_muscles * n_points)
    self._path_fixation_body = np.concatenate([self._path_fixation_body, path_fixation_body], axis=-1)
    self._path_coordinates = np.concatenate([self._path_coordinates, path_coordinates], axis=-1)
    self._muscle_index = np.hstack([self._muscle_index, np.tile(np.max(self.n_muscles), [n_points])])
    # indexes where the next item is from a different muscle, to indicate when their difference is meaningless
    self._muscle_transitions = np.diff(self._muscle_index.reshape((1, 1, -1))) == 1.
    # to create the ragged tensors when collapsing each muscle's segment values
    n_total_points = np.array([len(self._muscle_index)])
    self._row_splits = np.concatenate([np.zeros(1), np.diff(self._muscle_index).nonzero()[0] + 1, n_total_points-1])

    self.path_fixation_body = np.array(self._path_fixation_body, dtype=np.float32)
    self.path_coordinates = np.array(self._path_coordinates, dtype=np.float32)
    self.muscle_index = np.array(self._muscle_index, dtype=np.float32)
    self.muscle_transitions = np.array(self._muscle_transitions, dtype=bool)
    self.row_splits = np.array(self._row_splits, dtype=np.float32)
    self.section_splits = np.diff(self._row_splits).astype(int).tolist()

    # kwargs loop
    for key, val in kwargs.items():
      if key in self.tobuild__muscle:
        self.tobuild__muscle[key].append(val)
    for key, val in self.tobuild__muscle.items():
      # if not added in the kwargs loop
      if len(val) < self.n_muscles:
        # if the muscle object contains a default, use it, else raise error
        if key in self.tobuild__default:
          self.tobuild__muscle[key].append(self.tobuild__default[key])
        else:
          raise TypeError('Missing keyword argument ' + key + '.')
    self.muscle.build(timestep=self.minidt, **self.tobuild__muscle)

    name = name if name is not None else 'muscle_' + str(self.n_muscles)
    self.muscle_name.append(name)
    self._muscle_config_is_empty = False

  def get_muscle_cfg(self):
    """Gets the wrapping configuration of muscles added through the :meth:`add_muscle` method.

    Returns:
      A `dictionary` containing a key for each muscle name, associated to a nested dictionary containing
      information fo that muscle.
    """

    cfg = {}
    for m in range(self.n_muscles):
      ix = np.where(self._muscle_index == (m + 1))[0]

      d = {
        "n_fixation_points": len(ix),
        "fixation body": [int(k) for k in self._path_fixation_body.squeeze()[ix].tolist()],
        "coordinates": [self._path_coordinates.squeeze()[:, k].tolist() for k in ix],
      }

      if not self._muscle_config_is_empty:
        for param, value in self.tobuild__muscle.items():
          d[param] = value[m]

      cfg[self.muscle_name[m]] = d
    if not cfg:
      cfg = {"Placeholder Message": "No muscles were added using the `add_muscle` method."}
    return cfg

  def print_muscle_wrappings(self):
    """Prints the wrapping configuration of the muscles added using the :meth:`add_muscle` method in a readable
    format."""

    cfg = self.get_muscle_cfg()
    if self._muscle_config_is_empty:
      print(cfg)
      return

    for muscle, params in cfg.items():
      print("MUSCLE NAME: " + muscle)
      print("-" * (13 + len(muscle)))
      for key, param in params.items():
          print(key + ": ", param)
      print("\n")

  def get_geometry(self, joint_state):
    """Computes the geometry state from the joint state.
    Geometry state dimensionality is `[n_batch, n_timesteps, n_states, n_muscles]`. By default, there are as many
    states as there are moments (that is, one per degree of freedom in the effector) plus two for musculotendon length
    and musculotendon velocity. However, note that how many states and what they represent may vary depending on
    the :class:`Effector` subclass. This should be available via the
    :attr:`geometry_state_names` attribute.

    Args:
      joint_state: `Array`, the joint state from which the geometry state is computed.

    Returns:
      The geometry state corresponding to the joint state provided.
    """
    return self._get_geometry(joint_state)

  def _get_geometry(self, joint_state):
    # dxy_ddof --> (n_batches, n_dof, n_dof, n_points)
    xy, dxy_dt, dxy_ddof = self.skeleton.path2cartesian(self.path_coordinates, self.path_fixation_body, joint_state)
    diff_pos = xy[:, :, 1:] - xy[:, :, :-1]
    diff_vel = dxy_dt[:, :, 1:] - dxy_dt[:, :, :-1]
    diff_ddof = dxy_ddof[:, :, :, 1:] - dxy_ddof[:, :, :, :-1]

    # length, velocity and moment of each path segment
    # -----------------------
    # segment length is just the euclidian distance between the two points
    segment_len = np.sqrt(np.sum(diff_pos ** 2, axis=1, keepdims=True))
    # segment velocity is trickier: we are not after radial velocity but relative velocity.
    # https://math.stackexchange.com/questions/1481701/time-derivative-of-the-distance-between-2-points-moving-over-time
    # Formally, if segment_len=0 then segment_vel is not defined. We could substitute with 0 here because a
    # muscle segment will never flip backward, so the velocity can only be positive afterwards anyway.
    segment_vel = np.sum(diff_pos * diff_vel / segment_len, axis=1, keepdims=True)
    # for moment arm calculation, see Sherman, Seth, Delp (2013) -- DOI:10.1115/DETC2013-13633
    segment_moments = np.sum(diff_ddof * diff_pos[:, :, None], axis=1) / segment_len

    # remove differences between points that don't belong to the same muscle
    segment_len_cleaned = np.where(self.muscle_transitions, 0., segment_len)
    segment_vel_cleaned = np.where(self.muscle_transitions, 0., segment_vel)
    segment_mom_cleaned = np.where(self.muscle_transitions, 0., segment_moments)

    # sum up the contribution of all the segments belonging to a given muscle
    # OPTIMIZED: Use np.add.reduceat for vectorized segmented summation instead of Python loop
    # Compute cumulative indices for reduceat (start indices for each muscle)
    if not hasattr(self, '_reduceat_indices'):
      # Cache the indices since section_splits doesn't change after muscles are added
      cumsum = np.cumsum([0] + self.section_splits[:-1])
      self._reduceat_indices = cumsum.astype(int)

    # For batch dimension, we need to sum along axis=-1 for each segment
    # reduceat works on 1D, so we need to handle batch dimension separately
    batch_size = segment_len_cleaned.shape[0]
    n_features_len = segment_len_cleaned.shape[1]
    n_features_mom = segment_mom_cleaned.shape[1]

    # Reshape to (batch * features, n_segments), apply reduceat, reshape back
    musculotendon_len = np.add.reduceat(
      segment_len_cleaned.reshape(batch_size * n_features_len, -1),
      self._reduceat_indices,
      axis=1
    ).reshape(batch_size, n_features_len, self.n_muscles)

    musculotendon_vel = np.add.reduceat(
      segment_vel_cleaned.reshape(batch_size * n_features_len, -1),
      self._reduceat_indices,
      axis=1
    ).reshape(batch_size, n_features_len, self.n_muscles)

    moment_arms = np.add.reduceat(
      segment_mom_cleaned.reshape(batch_size * n_features_mom, -1),
      self._reduceat_indices,
      axis=1
    ).reshape(batch_size, n_features_mom, self.n_muscles)

    # pack all this into one state array and flip the dimensions back (batch_size * n_features * n_muscles)
    geometry_state = np.concatenate([musculotendon_len, musculotendon_vel, moment_arms], axis=1)
    return geometry_state

  def _set_state(self, states):
    for key, val in states.items():
      self.states[key] = val
    self.states["cartesian"] = self.joint2cartesian(joint_state=states["joint"])
    self.states["fingertip"] = np.split(self.states["cartesian"], 2, axis=-1)[0]

  def integrate(self, action, endpoint_load, joint_load):
    """Integrates the effector over one timestep. To do so, it first calls the :meth:`update_ode` method to obtain
    state derivatives from evaluation of the Ordinary Differential Equations. Then it performs the numerical
    integration over one timestep using the :meth:`integration_step` method, and updates the states to the
    resulting values.

    Args:
      action: `Array`, the input to the muscles (motor command). Typically, this should be the output of
        the controller or policy network's forward pass.
      endpoint_load: `Array`, the load(s) to apply at the skeleton's endpoint.
      joint_load: `Array`, the load(s) to apply at the joints.
    """
    self._integrate(action, endpoint_load, joint_load)

  def _euler(self, action, endpoint_load, joint_load):
    states0 = self.states
    state_derivative = self.ode(action, states0, endpoint_load, joint_load)
    states = self.integration_step(self.minidt, state_derivative=state_derivative, states=states0)
    self._set_state(states)

  def _rungekutta4(self, action, endpoint_load, joint_load):
    states0 = self.states
    k1 = self.ode(action, states=states0, endpoint_load=endpoint_load, joint_load=joint_load)
    states = self.integration_step(self.half_minidt, state_derivative=k1, states=states0)
    k2 = self.ode(action, states=states, endpoint_load=endpoint_load, joint_load=joint_load)
    states = self.integration_step(self.half_minidt, state_derivative=k2, states=states)
    k3 = self.ode(action, states=states, endpoint_load=endpoint_load, joint_load=joint_load)
    states = self.integration_step(self.minidt, state_derivative=k3, states=states)
    k4 = self.ode(action, states=states, endpoint_load=endpoint_load, joint_load=joint_load)
    k = {key: (k1[key] + 2 * (k2[key] + k3[key]) + k4[key]) / 6 for key in k1.keys()}
    states = self.integration_step(self.minidt, state_derivative=k, states=states0)
    self._set_state(states)

  def _rkf45_adaptive(self, action, endpoint_load, joint_load):
    """Runge-Kutta-Fehlberg (RKF45) with adaptive timestep control.

    This method uses an embedded 5th and 4th order Runge-Kutta method to estimate
    the error and adaptively adjust the timestep for optimal accuracy and speed.
    """
    states0 = self.states
    dt = self.current_adaptive_dt
    max_attempts = 100

    for attempt in range(max_attempts):
      # RKF45 Butcher tableau coefficients
      # k1
      k1 = self.ode(action, states=states0, endpoint_load=endpoint_load, joint_load=joint_load)

      # k2
      states_temp = self.integration_step(dt * 1/4, state_derivative=k1, states=states0)
      k2 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k3
      k_combined = {key: (3*k1[key] + 9*k2[key]) / 32 for key in k1.keys()}
      states_temp = self.integration_step(dt * 3/8, state_derivative=k_combined, states=states0)
      k3 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k4
      k_combined = {key: (1932*k1[key] - 7200*k2[key] + 7296*k3[key]) / 2197 for key in k1.keys()}
      states_temp = self.integration_step(dt * 12/13, state_derivative=k_combined, states=states0)
      k4 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k5
      k_combined = {key: (439*k1[key]/216 - 8*k2[key] + 3680*k3[key]/513 - 845*k4[key]/4104) for key in k1.keys()}
      states_temp = self.integration_step(dt, state_derivative=k_combined, states=states0)
      k5 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k6
      k_combined = {key: (-8*k1[key]/27 + 2*k2[key] - 3544*k3[key]/2565 + 1859*k4[key]/4104 - 11*k5[key]/40) for key in k1.keys()}
      states_temp = self.integration_step(dt * 1/2, state_derivative=k_combined, states=states0)
      k6 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # 4th order solution
      k_4th = {key: (25*k1[key]/216 + 1408*k3[key]/2565 + 2197*k4[key]/4104 - k5[key]/5) for key in k1.keys()}
      states_4th = self.integration_step(dt, state_derivative=k_4th, states=states0)

      # 5th order solution
      k_5th = {key: (16*k1[key]/135 + 6656*k3[key]/12825 + 28561*k4[key]/56430 - 9*k5[key]/50 + 2*k6[key]/55) for key in k1.keys()}
      states_5th = self.integration_step(dt, state_derivative=k_5th, states=states0)

      # Error estimation
      error = self._compute_state_error(states_4th, states_5th)

      # Compute optimal timestep
      if error == 0:
        # Perfect accuracy, maximize timestep
        dt_new = self.adaptive_max_dt
        accept = True
      else:
        # Standard adaptive step formula with safety factor
        safety = 0.9
        dt_new = safety * dt * (self.adaptive_tolerance / error) ** 0.2
        dt_new = np.clip(dt_new, self.adaptive_min_dt, self.adaptive_max_dt)
        accept = error <= self.adaptive_tolerance

      if accept:
        # Accept the step
        self._set_state(states_5th)
        self.current_adaptive_dt = dt_new
        self.adaptive_step_count += 1
        break
      else:
        # Reject the step and retry with smaller dt
        dt = dt_new
        self.adaptive_rejected_count += 1

        if attempt == max_attempts - 1:
          # If we've exhausted attempts, accept with minimum timestep
          self._set_state(states_5th)
          self.current_adaptive_dt = self.adaptive_min_dt
          self.adaptive_step_count += 1
          break

  def _dopri5_adaptive(self, action, endpoint_load, joint_load):
    """Dormand-Prince (DOPRI5) with adaptive timestep control.

    This is a more efficient embedded 5th/4th order method than RKF45.
    """
    states0 = self.states
    dt = self.current_adaptive_dt
    max_attempts = 100

    for attempt in range(max_attempts):
      # DOPRI5 Butcher tableau coefficients
      # k1
      k1 = self.ode(action, states=states0, endpoint_load=endpoint_load, joint_load=joint_load)

      # k2
      states_temp = self.integration_step(dt * 1/5, state_derivative=k1, states=states0)
      k2 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k3
      k_combined = {key: (3*k1[key] + 9*k2[key]) / 40 for key in k1.keys()}
      states_temp = self.integration_step(dt * 3/10, state_derivative=k_combined, states=states0)
      k3 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k4
      k_combined = {key: (44*k1[key]/45 - 56*k2[key]/15 + 32*k3[key]/9) for key in k1.keys()}
      states_temp = self.integration_step(dt * 4/5, state_derivative=k_combined, states=states0)
      k4 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k5
      k_combined = {key: (19372*k1[key]/6561 - 25360*k2[key]/2187 + 64448*k3[key]/6561 - 212*k4[key]/729) for key in k1.keys()}
      states_temp = self.integration_step(dt * 8/9, state_derivative=k_combined, states=states0)
      k5 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k6
      k_combined = {key: (9017*k1[key]/3168 - 355*k2[key]/33 + 46732*k3[key]/5247 + 49*k4[key]/176 - 5103*k5[key]/18656) for key in k1.keys()}
      states_temp = self.integration_step(dt, state_derivative=k_combined, states=states0)
      k6 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # k7
      k_combined = {key: (35*k1[key]/384 + 500*k3[key]/1113 + 125*k4[key]/192 - 2187*k5[key]/6784 + 11*k6[key]/84) for key in k1.keys()}
      states_temp = self.integration_step(dt, state_derivative=k_combined, states=states0)
      k7 = self.ode(action, states=states_temp, endpoint_load=endpoint_load, joint_load=joint_load)

      # 4th order solution (embedded)
      k_4th = {key: (5179*k1[key]/57600 + 7571*k3[key]/16695 + 393*k4[key]/640 - 92097*k5[key]/339200 + 187*k6[key]/2100 + k7[key]/40) for key in k1.keys()}
      states_4th = self.integration_step(dt, state_derivative=k_4th, states=states0)

      # 5th order solution
      k_5th = {key: (35*k1[key]/384 + 500*k3[key]/1113 + 125*k4[key]/192 - 2187*k5[key]/6784 + 11*k6[key]/84) for key in k1.keys()}
      states_5th = self.integration_step(dt, state_derivative=k_5th, states=states0)

      # Error estimation
      error = self._compute_state_error(states_4th, states_5th)

      # Compute optimal timestep
      if error == 0:
        dt_new = self.adaptive_max_dt
        accept = True
      else:
        safety = 0.9
        dt_new = safety * dt * (self.adaptive_tolerance / error) ** 0.2
        dt_new = np.clip(dt_new, self.adaptive_min_dt, self.adaptive_max_dt)
        accept = error <= self.adaptive_tolerance

      if accept:
        self._set_state(states_5th)
        self.current_adaptive_dt = dt_new
        self.adaptive_step_count += 1
        break
      else:
        dt = dt_new
        self.adaptive_rejected_count += 1

        if attempt == max_attempts - 1:
          self._set_state(states_5th)
          self.current_adaptive_dt = self.adaptive_min_dt
          self.adaptive_step_count += 1
          break

  def _compute_state_error(self, states1, states2):
    """Compute the normalized error between two state dictionaries.

    Args:
      states1: First state dictionary
      states2: Second state dictionary

    Returns:
      Normalized error as a scalar
    """
    error = 0.0
    n_elements = 0

    for key in ['muscle', 'joint']:
      if key in states1 and key in states2:
        diff = np.abs(states1[key] - states2[key])
        # Normalize by state magnitude + tolerance to avoid division by zero
        scale = np.abs(states1[key]) + self.adaptive_tolerance
        normalized_diff = diff / scale
        error += np.sum(normalized_diff ** 2)
        n_elements += normalized_diff.size

    if n_elements > 0:
      # Root mean square error
      error = np.sqrt(error / n_elements)

    return error

  def integration_step(self, dt, state_derivative, states):
    """Performs one numerical integration step for the :class:`npmotornet.muscle.Muscle` object class or
    subclass, and then for the :class:`npmotornet.skeleton.Skeleton` object class or subclass.

    Args:
      dt: `Float`, size of a single timestep (in sec).
      state_derivative: `Dictionary`, contains the derivatives of the joint, muscle, and geometry states as
        `array`, mapped to a "joint", "muscle", and "geometry" key, respectively. This is usually
        obtained using the :meth:`update_ode` method.
      states: A `Dictionary` containing the joint, muscle, and geometry states as `array`, mapped to a "joint",
        "muscle", and "geometry" key, respectively.

    Returns:
      A `dictionary` containing the updated joint, muscle, and geometry states following integration.
    """

    new_states = {
      "muscle": self.muscle.integrate(dt, state_derivative["muscle"], states["muscle"], states["geometry"]),
      "joint": self.skeleton.integrate(dt, state_derivative["joint"], states["joint"])}
    new_states["geometry"] = self.get_geometry(new_states["joint"])
    return new_states

  def ode(self, action, states, endpoint_load, joint_load):
    """Computes state derivatives by evaluating the Ordinary Differential Equations of the
    ``npmotornet.muscle.Muscle`` object class or subclass, and then of the
    :class:`npmotornet.skeleton.Skeleton` object class or subclass.

    Args:
      action: `Array`, the input to the muscles (motor command). Typically, this should be the output of
        the controller or policy network's forward pass.
      states: `Dictionary` contains the joint, muscle, and geometry states as `array`, mapped to a "joint",
        "muscle", and "geometry" key, respectively.
      endpoint_load: `Array`, the load(s) to apply at the skeleton's endpoint.
      joint_load: `Array`, the load(s) to apply at the joints.

    Returns:
      A `dictionary` containing the derivatives of the the joint, muscle, and geometry states as `array`,
      mapped to a "joint", "muscle", and "geometry" key, respectively.
    """

    moments = states["geometry"][:, 2:, :]
    forces = states["muscle"][:, self.force_index:self.force_index+1, :]
    joint_vel = np.split(states["joint"], 2, axis=-1)[-1]

    generalized_forces = - np.sum(forces * moments, axis=-1) + joint_load - self.damping * joint_vel

    state_derivative = {
      "muscle": self.muscle.ode(action, states["muscle"]),
      "joint": self.skeleton.ode(generalized_forces, states["joint"], endpoint_load=endpoint_load)
      }
    return state_derivative

  def draw_random_uniform_states(self, batch_size):
    """Draws joint states according to a random uniform distribution, bounded by the position and velocity boundary
    attributes defined at initialization.

    Args:
      batch_size: `Integer`, the desired batch size.

    Returns:
      An `array` containing `batch_size` joint states.
    """
    sz = (batch_size, self.dof)
    rnd = np.array(self.np_random.uniform(size=sz), dtype=np.float32)
    pos = self.pos_range_bound * rnd + self.skeleton.pos_upper_bound
    vel = np.zeros(sz, dtype=np.float32)
    return np.concatenate([pos, vel], axis=1)

  def _parse_initial_joint_state(self, joint_state, batch_size):
    if joint_state is None:
      joint0 = self.draw_random_uniform_states(batch_size=batch_size)
    else:
      joint_state_shape = np.shape(joint_state)
      if joint_state_shape[0] > 1:
        batch_size = 1
      n_state = joint_state.shape[1]
      if n_state == self.state_dim:
        position, velocity = np.split(joint_state, 2, axis=-1)
        joint0 = self.draw_fixed_states(position=position, velocity=velocity, batch_size=batch_size)
      elif n_state == int(self.state_dim / 2):
        joint0 = self.draw_fixed_states(position=joint_state, batch_size=batch_size)
      else:
        raise ValueError

    return joint0

  def draw_fixed_states(self, batch_size, position, velocity=None):
    """Creates a joint state `array` corresponding to the specified position, tiled `batch_size` times.

    Args:
      position: The position to tile in the state `array`.
      velocity: The velocity to tile in the state `array`. If `None`, this will default to `0` (null velocity).
      batch_size: `Integer`, the desired batch size.

    Returns:
      An `array` containing `batch_size` joint states.
    """
    if velocity is None:
      velocity = np.zeros_like(position)
    # in case input is a list or a numpy array
    pos = np.array(position) if not isinstance(position, np.ndarray) else position
    vel = np.array(velocity) if not isinstance(velocity, np.ndarray) else velocity
    if len(pos.shape) == 1:
        pos = pos.reshape((1, -1))
    if len(vel.shape) == 1:
        vel = vel.reshape((1, -1))

    assert pos.shape == vel.shape
    assert pos.shape[1] == self.dof
    assert len(pos.shape) == 2
    assert np.all(pos >= self.pos_lower_bound)
    assert np.all(pos <= self.pos_upper_bound)
    assert np.all(vel >= self.vel_lower_bound)
    assert np.all(vel <= self.vel_upper_bound)

    vel = np.array(vel, dtype=np.float32)
    pos = np.array(pos, dtype=np.float32)
    states = np.concatenate([pos, vel], axis=1)
    return np.tile(states, (batch_size, 1))

  def _set_state_limit_bounds(self, lb, ub):
    lb = np.array(lb).reshape((-1, 1)).astype(np.float32)  # ensure this is a 2D array
    ub = np.array(ub).reshape((-1, 1)).astype(np.float32)
    bounds = np.hstack((lb, ub))
    bounds = bounds * np.ones((self.dof, 2)).astype(np.float32)  # if one bound pair, broadcast to dof rows
    return bounds

  def get_save_config(self):
    """Gets the effector object's configuration as a `dictionary`.

    Returns:
      A `dictionary` containing the skeleton and muscle configurations as nested `dictionary` objects, and
      parameters of the effector's configuration. Specifically, the size of the timestep (sec), the name
      of each muscle added via the :meth:`add_muscle` method, the number of muscles, the visual and
      proprioceptive delay, the standard deviation of the excitation noise, and the muscle wrapping configuration
      as returned by :meth:`get_muscle_cfg`.
    """
    muscle_cfg = self.muscle.get_save_config()
    skeleton_cfg = self.skeleton.get_save_config()
    cfg = {'muscle': muscle_cfg,
            'skeleton': skeleton_cfg,
            'dt': self.dt, 'n_ministeps': self.n_ministeps,
            'minidt': self.minidt, 'half_minidt': self.half_minidt,
            'muscle_names': self.muscle_name,
            'n_muscles': self.n_muscles,
            'muscle_wrapping_cfg': self.get_muscle_cfg()}
    return cfg

  def joint2cartesian(self, joint_state: np.ndarray) -> np.ndarray:
    """Computes the cartesian state given the joint state.

    Args:
      joint_state: `Array`, the current joint configuration.

    Returns:
      The current cartesian configuration (position, velocity) as an `array`.
    """
    return self.skeleton.joint2cartesian(joint_state=joint_state)

  def setattr(self, name: str, value):
    """Changes the value of an attribute held by this object.

    Args:
      name: `String`, attribute to set to a new value.
      value: Value that the attribute should take.
    """
    self.__setattr__(name, value)

  def _merge_muscle_kwargs(self, muscle_kwargs: dict):
    """
    Merges the muscle_kwargs argument with the default muscle_kwargs argument, and stores the result in the
    tobuild__muscle attribute.

    Args:
      muscle_kwargs: `Dictionary`, contains the muscle_kwargs argument passed to the
        :meth:`npmotornet.muscle.Muscle.build()` method.
    """
    # kwargs loop
    for key, val in muscle_kwargs.items():
      if key in self.tobuild__muscle.keys():
        self.tobuild__muscle[key].append(val)
      else:
        raise KeyError('Unexpected key "' + key + '" in muscle_kwargs argument.')

    for key, val in self.tobuild__muscle.items():
      # if not added in the kwargs loop
      if len(val) == 0 and key in self.tobuild__default.keys():
        self.tobuild__muscle[key].append(self.tobuild__default[key])


class ReluPointMass24(Effector):
  """This object implements a 2D point-mass skeleton attached to 4 ``npmotornet.muscle.ReluMuscle`` muscles
  in a "X" configuration. The outside attachement points are the corners of a
  `(2, 2) -> (2, -2) -> (-2, -2) -> (-2, 2)` frame, and the point-mass is constrained to a
  `(1, 1) -> (1, -1) -> (-1, -1) -> (-1, 1)` space.

  Args:
    timestep: `Float`, size of a single timestep (in sec).
    max_isometic_force: `Float`, the maximum force (N) that each muscle can produce.
    mass: `Float`, the mass (kg) of the point-mass.
    **kwargs: The `kwargs` inputs are passed as-is to the parent :class:`npmotornet.Effector` class.
  """

  def __init__(self, timestep: float = 0.01, max_isometric_force: float = 500, mass: float = 1, **kwargs):
    skeleton = PointMass(space_dim=2, mass=mass)
    super().__init__(skeleton=skeleton, muscle=ReluMuscle(), timestep=timestep, **kwargs)

    # path coordinates for each muscle
    ur = [[2, 2], [0, 0]]
    ul = [[-2, 2], [0, 0]]
    lr = [[2, -2], [0, 0]]
    ll = [[-2, -2], [0, 0]]

    f = max_isometric_force
    self.add_muscle(path_fixation_body=[0, 1], path_coordinates=ur, name='UpperRight', max_isometric_force=f)
    self.add_muscle(path_fixation_body=[0, 1], path_coordinates=ul, name='UpperLeft', max_isometric_force=f)
    self.add_muscle(path_fixation_body=[0, 1], path_coordinates=lr, name='LowerRight', max_isometric_force=f)
    self.add_muscle(path_fixation_body=[0, 1], path_coordinates=ll, name='LowerLeft', max_isometric_force=f)


class RigidTendonArm26(Effector):
  """This pre-built effector class is an implementation of a 6-muscles, "lumped-muscle" model from `[1]`. Because
  lumped-muscle models are functional approximations of biological reality, this class' geometry does not rely on the
  default geometry methods, but on its own, custom-made geometry. The moment arm approximation is based on a set of
  polynomial functions. The default integration method is Euler.

  If no `skeleton` input is provided, this object will use a :class:`npmotornet.skeleton.TwoDofArm`
  skeleton, with the following parameters (from `[1]`):

  - `m1 = 1.82`
  - `m2 = 1.43`
  - `l1g = 0.135`
  - `l2g = 0.165`
  - `i1 = 0.051`
  - `i2 = 0.057`
  - `l1 = 0.309`
  - `l2 = 0.333`

  The default shoulder and elbow lower limits are defined as `0`, and their default upper limits as `135` and `155`
  degrees, respectively.

  The `kwargs` inputs are passed as-is to the parent :class:`Effector` class.

  References:
    [1] `Nijhof, E.-J., & Kouwenhoven, E. Simulation of Multijoint Arm Movements (2000). In J. M. Winters & P. E.
    Crago, Biomechanics and Neural Control of Posture and Movement (pp. 363–372). Springer New York.
    doi: 10.1007/978-1-4612-2104-3_29`

  Args:
    muscle: A :class:`npmotornet.muscle.Muscle` object class or subclass. This defines the type of muscle
      that will be added each time the :meth:`add_muscle` method is called.
    skeleton: A :class:`npmotornet.skeleton.Skeleton` object class or subclass. This defines the type of
      skeleton that the muscles will wrap around. See above for details on what this argument defaults to if no
      argument is passed.
    timestep: `Float`, size of a single timestep (in sec).
    muscle_kwargs: `Dictionary`, contains the muscle parameters to be passed to the
      :meth:`npmotornet.muscle.Muscle.build() method.`
    **kwargs: All contents are passed to the parent :class:`Effector` class. Also allows for some backward
      compatibility.
  """

  def __init__(self, muscle, skeleton=None, timestep=0.01, muscle_kwargs: dict = {}, **kwargs):
    sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
    elb_limit = np.deg2rad([0, 155])
    pos_lower_bound = kwargs.pop('pos_lower_bound', [sho_limit[0], elb_limit[0]])
    pos_upper_bound = kwargs.pop('pos_upper_bound', [sho_limit[1], elb_limit[1]])

    if skeleton is None:
      skeleton = TwoDofArm(m1=1.82, m2=1.43, l1g=.135, l2g=.165, i1=.051, i2=.057, l1=.309, l2=.333)

    super().__init__(
      skeleton=skeleton,
      muscle=muscle,
      timestep=timestep,
      pos_lower_bound=pos_lower_bound,
      pos_upper_bound=pos_upper_bound,
      **kwargs)

    # build muscle system
    self.muscle_state_dim = self.muscle.state_dim
    self.geometry_state_dim = 2 + self.skeleton.dof  # musculotendon length & velocity + as many moments as dofs
    self.n_muscles = 6
    self.input_dim = self.n_muscles

    self.muscle_name = ['pectoralis', 'deltoid', 'brachioradialis', 'tricepslat', 'biceps', 'tricepslong']

    self._merge_muscle_kwargs(muscle_kwargs)

    self.tobuild__muscle['max_isometric_force'] = [838, 1207, 1422, 1549, 414, 603]
    self.tobuild__muscle['tendon_length'] = [0.039, 0.066, 0.172, 0.187, 0.204, 0.217]
    self.tobuild__muscle['optimal_muscle_length'] = [0.134, 0.140, 0.092, 0.093, 0.137, 0.127]

    self.muscle.build(timestep=self.dt, **self.tobuild__muscle)

    a0 = [0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989]
    a1 = [-.03, .03, 0, 0, -.03, .03, 0, 0, -.014, .025, -.016, .03]
    a2 = [0, 0, 0, 0, 0, 0, 0, 0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]
    a3 = [np.pi / 2, 0.]
    self.a0 = np.array(np.array(a0).reshape((1, 1, 6)), dtype=np.float32)
    self.a1 = np.array(np.array(a1).reshape((1, 2, 6)), dtype=np.float32)
    self.a2 = np.array(np.array(a2).reshape((1, 2, 6)), dtype=np.float32)
    self.a3 = np.array(np.array(a3).reshape((1, 2, 1)), dtype=np.float32)

  def _get_geometry(self, joint_state):
    old_pos, old_vel = np.split(joint_state[:, :, np.newaxis], 2, axis=1)
    old_pos = old_pos - self.a3
    moment_arm = old_pos * self.a2 * 2 + self.a1
    musculotendon_len = np.sum((self.a1 + old_pos * self.a2) * old_pos, axis=1, keepdims=True) + self.a0
    musculotendon_vel = np.sum(old_vel * moment_arm, axis=1, keepdims=True)
    return np.concatenate([musculotendon_len, musculotendon_vel, moment_arm], axis=1)


class CompliantTendonArm26(RigidTendonArm26):
  """This is the compliant-tendon version of the :class:`RigidTendonArm26` class. Note that the default integration
  method is Runge-Kutta 4, instead of Euler.

  Args:
    timestep: `Float`, size of a single timestep (in sec).
    skeleton: A :class:`npmotornet.skeleton.Skeleton` object class or subclass. This defines the type of
      skeleton that the muscles will wrap around. If no skeleton is passed, this will default to the skeleton
      used in the parent :class:`RigidTendonArm26` class.

    **kwargs: All contents are passed to the parent :class:`RigidTendonArm26` class. This also
      allows for some backward compatibility.
  """

  def __init__(self, timestep=0.0002, skeleton=None, muscle_kwargs: dict = {}, **kwargs):
    integration_method = kwargs.pop('integration_method', 'rk4')
    if skeleton is None:
      skeleton = TwoDofArm(m1=1.82, m2=1.43, l1g=.135, l2g=.165, i1=.051, i2=.057, l1=.309, l2=.333)

    super().__init__(
      muscle=CompliantTendonHillMuscle(),
      skeleton=skeleton,
      timestep=timestep,
      integration_method=integration_method,
      **kwargs)

    # build muscle system
    self._merge_muscle_kwargs(muscle_kwargs)

    self.tobuild__muscle['max_isometric_force'] = [838, 1207, 1422, 1549, 414, 603]
    self.tobuild__muscle['tendon_length'] = [0.070, 0.070, 0.172, 0.187, 0.204, 0.217]
    self.tobuild__muscle['optimal_muscle_length'] = [0.134, 0.140, 0.092, 0.093, 0.137, 0.127]

    self.muscle.build(timestep=timestep, **self.tobuild__muscle)

    # Adjust some parameters to relax overly stiff tendon values.
    # This should greatly help with stability during numerical integration.
    a0 = [0.182, 0.2362, 0.2859, 0.2355, 0.3329, 0.2989]
    self.a0 = np.array(np.array(a0).reshape((1, 1, 6)), dtype=np.float32)
