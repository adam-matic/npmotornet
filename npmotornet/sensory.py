"""
Proprioceptive sensory receptors for muscle simulation.

This module implements sensory feedback mechanisms including Golgi tendon organs
and muscle spindles for realistic neuromuscular modeling.

References:
    [1] Houk JC, Henneman E. (1967). Responses of Golgi tendon organs to active
        contractions of the soleus muscle of the cat. J Neurophysiol, 30(3), 466-481.
    [2] Jami L. (1992). Golgi tendon organs in mammalian skeletal muscle: functional
        properties and central actions. Physiol Rev, 72(3), 623-666.
    [3] Crago PE, Houk JC, Rymer WZ. (1982). Sampling of total muscle force by
        tendon organs. J Neurophysiol, 47(6), 1069-1083.
"""

import numpy as np


class GolgiTendonOrgan:
    """Golgi tendon organ (GTO) sensory receptor model.

    GTOs are mechanoreceptors located at the musculotendinous junction that
    respond to muscle tension/force. They provide sensory feedback about force
    production during both active contraction and passive stretch.

    This implementation models GTO firing rate as a function of tendon force
    with both static (proportional to force) and dynamic (proportional to rate
    of force change) components.

    The model equation is:
        firing_rate = baseline + k_static × F + k_dynamic × dF/dt

    where F is tendon force and dF/dt is the rate of force change.

    Args:
        n_receptors: Integer, number of GTO receptors (typically one per muscle).
        k_static: Float, static gain converting force (N) to firing rate (Hz/N).
            Typical range: 0.1-0.3 Hz/N. Default: 0.2 Hz/N.
        k_dynamic: Float, dynamic gain converting force rate (N/s) to firing rate (Hz/(N/s)).
            Typically 10-20% of static contribution. Default: 0.03 Hz/(N/s).
        baseline_firing: Float, spontaneous firing rate (Hz) at zero force.
            Typical range: 0-10 Hz. Default: 5 Hz.
        saturation_rate: Float, maximum firing rate (Hz) beyond which response saturates.
            Typical range: 80-120 Hz. Default: 100 Hz.
        tau_adaptation: Float or None, time constant (s) for firing rate adaptation.
            If None, no adaptation is applied. Default: None.

    References:
        Static and dynamic gains based on Houk & Henneman (1967) and Jami (1992).
        Typical mammalian GTOs show firing rates of 5-100 Hz across physiological
        force ranges of 0-500 N.
    """

    def __init__(
        self,
        n_receptors: int,
        k_static: float = 0.2,
        k_dynamic: float = 0.03,
        baseline_firing: float = 5.0,
        saturation_rate: float = 100.0,
        tau_adaptation: float = None,
    ):
        self.n_receptors = n_receptors
        self.k_static = np.array(k_static, dtype=np.float32)
        self.k_dynamic = np.array(k_dynamic, dtype=np.float32)
        self.baseline_firing = np.array(baseline_firing, dtype=np.float32)
        self.saturation_rate = np.array(saturation_rate, dtype=np.float32)
        self.tau_adaptation = tau_adaptation

        # State tracking
        self.previous_force = None
        self.adaptation_state = None
        self.dt = None

    def reset(self, batch_size: int = 1):
        """Reset the internal state of the GTO receptors.

        Args:
            batch_size: Integer, size of the batch dimension.
        """
        self.previous_force = np.zeros((batch_size, self.n_receptors), dtype=np.float32)
        if self.tau_adaptation is not None:
            self.adaptation_state = np.zeros((batch_size, self.n_receptors), dtype=np.float32)

    def get_firing_rate(
        self,
        tendon_force: np.ndarray,
        dt: float,
        include_dynamic: bool = True,
    ) -> np.ndarray:
        """Compute GTO firing rate from tendon force.

        Args:
            tendon_force: Array of shape (batch_size, n_muscles), the current
                tendon force for each muscle (N).
            dt: Float, timestep duration (s) for computing force derivative.
            include_dynamic: Boolean, whether to include dynamic (rate-sensitive)
                component. Default: True.

        Returns:
            firing_rate: Array of shape (batch_size, n_receptors), the firing
                rate (Hz) for each GTO receptor.
        """
        self.dt = dt

        # Initialize state on first call
        if self.previous_force is None:
            self.reset(batch_size=tendon_force.shape[0])

        # Ensure force is non-negative (GTOs only respond to tension)
        tendon_force = np.clip(tendon_force, a_min=0., a_max=None)

        # Static component: proportional to force
        static_response = self.k_static * tendon_force

        # Dynamic component: proportional to rate of force change
        if include_dynamic:
            force_derivative = (tendon_force - self.previous_force) / dt
            dynamic_response = self.k_dynamic * force_derivative
        else:
            dynamic_response = 0.0

        # Total firing rate
        firing_rate = self.baseline_firing + static_response + dynamic_response

        # Apply adaptation if enabled
        if self.tau_adaptation is not None:
            # Simple exponential adaptation: firing rate gradually decreases
            # toward adapted level during sustained activation
            adaptation_increment = dt * (firing_rate - self.adaptation_state) / self.tau_adaptation
            self.adaptation_state += adaptation_increment
            firing_rate = firing_rate - 0.2 * self.adaptation_state  # 20% adaptation strength

        # Clip to physiological range (no negative firing, saturation at max rate)
        firing_rate = np.clip(firing_rate, a_min=0., a_max=self.saturation_rate)

        # Update state for next timestep
        self.previous_force = tendon_force.copy()

        return firing_rate

    def get_static_response(self, tendon_force: np.ndarray) -> np.ndarray:
        """Get steady-state firing rate for a given force (no dynamics).

        Useful for analyzing static force-firing rate relationships.

        Args:
            tendon_force: Array of shape (batch_size, n_muscles), tendon force (N).

        Returns:
            firing_rate: Array of shape (batch_size, n_receptors), steady-state
                firing rate (Hz).
        """
        tendon_force = np.clip(tendon_force, a_min=0., a_max=None)
        firing_rate = self.baseline_firing + self.k_static * tendon_force
        firing_rate = np.clip(firing_rate, a_min=0., a_max=self.saturation_rate)
        return firing_rate


class MuscleSpindle:
    """Muscle spindle sensory receptor model.

    Muscle spindles are mechanoreceptors located within skeletal muscles that
    respond to muscle fiber length and velocity. They provide critical sensory
    feedback for proprioception and motor control.

    This implementation models both:
    - Ia (primary) afferents: Encode both muscle length (static) and velocity (dynamic)
    - II (secondary) afferents: Encode muscle length only (static)

    The model also includes gamma motor neuron (fusimotor) drive:
    - Gamma dynamic: Modulates Ia dynamic sensitivity to velocity
    - Gamma static: Modulates both Ia and II static sensitivity to length

    The phenomenological model equations are:
        Ia_firing = baseline_Ia + k_Ia_static × ΔL × (1 + γ_static × g_s)
                    + k_Ia_dynamic × V × (1 + γ_dynamic × g_d)
        II_firing = baseline_II + k_II_static × ΔL × (1 + γ_static × g_s)

    where ΔL = (muscle_length - optimal_length), V = muscle_velocity,
    γ_static and γ_dynamic are gamma motor neuron inputs (0-1 range),
    and g_s, g_d are gain factors.

    Args:
        n_receptors: Integer, number of muscle spindle receptors (typically one per muscle).
        k_Ia_static: Float, Ia static gain converting length deviation (m) to firing rate (Hz/m).
            Typical range: 100-400 Hz/m. Default: 250 Hz/m.
        k_Ia_dynamic: Float, Ia dynamic gain converting velocity (m/s) to firing rate (Hz/(m/s)).
            Typical range: 200-600 Hz/(m/s). Default: 400 Hz/(m/s).
        k_II_static: Float, II static gain converting length deviation (m) to firing rate (Hz/m).
            Typically lower than Ia. Typical range: 50-200 Hz/m. Default: 100 Hz/m.
        baseline_Ia: Float, spontaneous Ia firing rate (Hz) at optimal length, zero velocity.
            Typical range: 10-40 Hz. Default: 20 Hz.
        baseline_II: Float, spontaneous II firing rate (Hz) at optimal length.
            Typical range: 5-20 Hz. Default: 10 Hz.
        saturation_Ia: Float, maximum Ia firing rate (Hz).
            Typical range: 100-200 Hz. Default: 150 Hz.
        saturation_II: Float, maximum II firing rate (Hz).
            Typical range: 60-120 Hz. Default: 80 Hz.
        optimal_length: Float, reference muscle length (m) at which baseline firing occurs.
            Typically set to the muscle's optimal length (l0_ce). Default: 0.08 m.
        gamma_dynamic_gain: Float, scaling factor for gamma dynamic modulation.
            Determines how much gamma_dynamic input affects Ia velocity sensitivity.
            Typical range: 0.5-2.0. Default: 1.0.
        gamma_static_gain: Float, scaling factor for gamma static modulation.
            Determines how much gamma_static input affects length sensitivity.
            Typical range: 0.5-2.0. Default: 1.0.

    References:
        [1] Mileusnic MP, Brown IE, Lan N, Loeb GE. (2006). Mathematical models
            of proprioceptors. I. Control and transduction in the muscle spindle.
            J Neurophysiol, 96(4), 1789-1802.
        [2] Prochazka A, Gorassini M. (1998). Ensemble firing of muscle afferents
            recorded during normal locomotion in cats. J Physiol, 507(1), 293-304.
        [3] Matthews PB. (1972). Mammalian muscle receptors and their central actions.
            London: Edward Arnold.
    """

    def __init__(
        self,
        n_receptors: int,
        k_Ia_static: float = 250.0,
        k_Ia_dynamic: float = 400.0,
        k_II_static: float = 100.0,
        baseline_Ia: float = 20.0,
        baseline_II: float = 10.0,
        saturation_Ia: float = 150.0,
        saturation_II: float = 80.0,
        optimal_length: float = 0.08,
        gamma_dynamic_gain: float = 1.0,
        gamma_static_gain: float = 1.0,
    ):
        self.n_receptors = n_receptors
        self.k_Ia_static = np.array(k_Ia_static, dtype=np.float32)
        self.k_Ia_dynamic = np.array(k_Ia_dynamic, dtype=np.float32)
        self.k_II_static = np.array(k_II_static, dtype=np.float32)
        self.baseline_Ia = np.array(baseline_Ia, dtype=np.float32)
        self.baseline_II = np.array(baseline_II, dtype=np.float32)
        self.saturation_Ia = np.array(saturation_Ia, dtype=np.float32)
        self.saturation_II = np.array(saturation_II, dtype=np.float32)
        self.optimal_length = np.array(optimal_length, dtype=np.float32)
        self.gamma_dynamic_gain = np.array(gamma_dynamic_gain, dtype=np.float32)
        self.gamma_static_gain = np.array(gamma_static_gain, dtype=np.float32)

        # State tracking
        self.previous_length = None
        self.dt = None

    def reset(self, batch_size: int = 1):
        """Reset the internal state of the muscle spindle receptors.

        Args:
            batch_size: Integer, size of the batch dimension.
        """
        self.previous_length = None

    def get_firing_rate(
        self,
        muscle_length: np.ndarray,
        muscle_velocity: np.ndarray,
        dt: float,
        gamma_dynamic: np.ndarray = None,
        gamma_static: np.ndarray = None,
    ) -> dict:
        """Compute muscle spindle firing rates from muscle length and velocity.

        Args:
            muscle_length: Array of shape (batch_size, n_muscles), the current
                muscle fiber length for each muscle (m). NOT musculotendon length.
            muscle_velocity: Array of shape (batch_size, n_muscles), the current
                muscle fiber velocity for each muscle (m/s). Positive = lengthening.
            dt: Float, timestep duration (s).
            gamma_dynamic: Array of shape (batch_size, n_muscles) or None, the
                gamma dynamic motor neuron drive (0-1 range). If None, defaults to 0.
            gamma_static: Array of shape (batch_size, n_muscles) or None, the
                gamma static motor neuron drive (0-1 range). If None, defaults to 0.

        Returns:
            Dictionary with keys:
                'Ia': Array of shape (batch_size, n_receptors), Ia afferent firing rate (Hz)
                'II': Array of shape (batch_size, n_receptors), II afferent firing rate (Hz)
        """
        self.dt = dt

        # Initialize state on first call
        if self.previous_length is None:
            self.reset(batch_size=muscle_length.shape[0])
            self.previous_length = muscle_length.copy()

        # Default gamma inputs to zero if not provided
        if gamma_dynamic is None:
            gamma_dynamic = np.zeros_like(muscle_length)
        if gamma_static is None:
            gamma_static = np.zeros_like(muscle_length)

        # Clip gamma inputs to valid range (0-1)
        gamma_dynamic = np.clip(gamma_dynamic, 0.0, 1.0)
        gamma_static = np.clip(gamma_static, 0.0, 1.0)

        # Compute length deviation from optimal
        length_deviation = muscle_length - self.optimal_length

        # Compute gamma modulation factors
        # Gamma drive increases sensitivity multiplicatively
        static_modulation = 1.0 + gamma_static * self.gamma_static_gain
        dynamic_modulation = 1.0 + gamma_dynamic * self.gamma_dynamic_gain

        # Ia (primary) afferent response: sensitive to both length and velocity
        Ia_static_component = self.k_Ia_static * length_deviation * static_modulation
        Ia_dynamic_component = self.k_Ia_dynamic * muscle_velocity * dynamic_modulation
        Ia_firing = self.baseline_Ia + Ia_static_component + Ia_dynamic_component
        Ia_firing = np.clip(Ia_firing, a_min=0.0, a_max=self.saturation_Ia)

        # II (secondary) afferent response: sensitive to length only
        II_static_component = self.k_II_static * length_deviation * static_modulation
        II_firing = self.baseline_II + II_static_component
        II_firing = np.clip(II_firing, a_min=0.0, a_max=self.saturation_II)

        # Update state for next timestep
        self.previous_length = muscle_length.copy()

        return {'Ia': Ia_firing, 'II': II_firing}

    def get_Ia_response(
        self,
        muscle_length: np.ndarray,
        muscle_velocity: np.ndarray,
        dt: float,
        gamma_dynamic: np.ndarray = None,
        gamma_static: np.ndarray = None,
    ) -> np.ndarray:
        """Compute only Ia (primary) afferent firing rate.

        Args:
            muscle_length: Array of shape (batch_size, n_muscles), muscle fiber length (m).
            muscle_velocity: Array of shape (batch_size, n_muscles), muscle fiber velocity (m/s).
            dt: Float, timestep duration (s).
            gamma_dynamic: Array or None, gamma dynamic drive (0-1).
            gamma_static: Array or None, gamma static drive (0-1).

        Returns:
            Ia_firing: Array of shape (batch_size, n_receptors), Ia firing rate (Hz).
        """
        return self.get_firing_rate(muscle_length, muscle_velocity, dt,
                                     gamma_dynamic, gamma_static)['Ia']

    def get_II_response(
        self,
        muscle_length: np.ndarray,
        muscle_velocity: np.ndarray,
        dt: float,
        gamma_static: np.ndarray = None,
    ) -> np.ndarray:
        """Compute only II (secondary) afferent firing rate.

        Args:
            muscle_length: Array of shape (batch_size, n_muscles), muscle fiber length (m).
            muscle_velocity: Array of shape (batch_size, n_muscles), muscle fiber velocity (m/s).
            dt: Float, timestep duration (s).
            gamma_static: Array or None, gamma static drive (0-1).

        Returns:
            II_firing: Array of shape (batch_size, n_receptors), II firing rate (Hz).
        """
        return self.get_firing_rate(muscle_length, muscle_velocity, dt,
                                     gamma_dynamic=None, gamma_static=gamma_static)['II']

    def get_static_response(
        self,
        muscle_length: np.ndarray,
        gamma_static: np.ndarray = None,
    ) -> dict:
        """Get steady-state firing rates for given muscle lengths (no velocity).

        Useful for analyzing static length-firing rate relationships.

        Args:
            muscle_length: Array of shape (batch_size, n_muscles), muscle fiber length (m).
            gamma_static: Array or None, gamma static drive (0-1).

        Returns:
            Dictionary with keys:
                'Ia': Steady-state Ia firing rate (Hz)
                'II': Steady-state II firing rate (Hz)
        """
        # Default gamma to zero if not provided
        if gamma_static is None:
            gamma_static = np.zeros_like(muscle_length)

        gamma_static = np.clip(gamma_static, 0.0, 1.0)

        # Compute length deviation
        length_deviation = muscle_length - self.optimal_length

        # Static modulation from gamma
        static_modulation = 1.0 + gamma_static * self.gamma_static_gain

        # Ia static response (no velocity component)
        Ia_firing = self.baseline_Ia + self.k_Ia_static * length_deviation * static_modulation
        Ia_firing = np.clip(Ia_firing, a_min=0.0, a_max=self.saturation_Ia)

        # II static response
        II_firing = self.baseline_II + self.k_II_static * length_deviation * static_modulation
        II_firing = np.clip(II_firing, a_min=0.0, a_max=self.saturation_II)

        return {'Ia': Ia_firing, 'II': II_firing}
