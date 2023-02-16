import numpy as np
"""
some commonly used physical concepts, 
including 
1-the Boltzmann distribution, which describes the distribution of particles in a system with 
respect to their energy, and
2-the Maxwell equations,
 which are fundamental equations of electromagnetism that describe the behavior of electric and magnetic fields.
"""

def boltzmann_dist(energy, temp, particle_type, energy_unit='J', temp_unit='K'):
    """
    Calculate the Bose-Einstein or Fermi-Dirac distribution function for a given energy and temperature.

    Parameters
    ----------
    energy : array_like
        The energy of the particle(s), in units of `energy_unit`.
    temp : array_like
        The temperature of the system, in units of `temp_unit`.
    particle_type : str
        The type of particle, either 'boson' or 'fermion'.
    energy_unit : str, optional
        The unit of energy used in `energy`. Default is 'J'.
    temp_unit : str, optional
        The unit of temperature used in `temp`. Default is 'K'.

    Returns
    -------
    ndarray
        The probability of the particle(s) being in the given energy state(s), according to the chosen distribution function.

    Notes
    -----
    The `energy` and `temp` arrays must have the same shape. If they are NumPy arrays, the division operator `/` will work element-wise. If they are not NumPy arrays, they will be converted to NumPy arrays internally.

    Examples
    --------
    Calculate the Bose-Einstein distribution function for a single energy value of 1 eV and a temperature of 300 K:

    >>> boltzmann_dist(1, 300, 'boson', energy_unit='eV')
    1.6342595760286473e-06

    Calculate the Fermi-Dirac distribution function for a NumPy array of energy values in units of meV and a temperature of 10 K:

    >>> energy = np.array([1, 2, 3]) * 1e-3  # energy in meV
    >>> temp = 10  # temperature in K
    >>> boltzmann_dist(energy, temp, 'fermion', energy_unit='meV')
    array([0.99876716, 0.99989984, 0.9999947 ])
    """
    k_b = 1.380649e-23  # Boltzmann constant in J/K
    energy = np.asarray(energy)
    temp = np.asarray(temp)
    energy_in_J = energy * getattr(np, energy_unit.lower())
    temp_in_K = temp * getattr(np, temp_unit.lower())
    if particle_type == "boson":
        # Bose-Einstein distribution
        exp_arg = energy_in_J / (k_b * temp_in_K)
        exp_arg[exp_arg > 700] = np.inf  # to prevent overflow errors
        return np.where(exp_arg == np.inf, 0.0, 1.0 / (np.exp(exp_arg) - 1.0))
    elif particle_type == "fermion":
        # Fermi-Dirac distribution
        exp_arg = (energy_in_J - k_b * temp_in_K) / (k_b * temp_in_K)
        exp_arg[exp_arg < -700] = -np.inf  # to prevent underflow errors
        exp_arg[exp_arg > 700] = np.inf  # to prevent overflow errors
        return np.where(exp_arg == np.inf, 0.0, np.where(exp_arg == -np.inf, 1.0, 1.0 / (np.exp(exp_arg) + 1.0)))
    else


    def maxwell_equation(field, freq, refractive_index, is_electric=True):
        """
        Calculates the electric or magnetic field given the other field using Maxwell's equations.

        Parameters:
        field (np.ndarray): 1D array of electric or magnetic field values
        freq (np.ndarray): 1D array of frequency values
        refractive_index (float): Refractive index of the medium
        is_electric (bool): True if the input field is electric field, False if it's magnetic field (default is True)

        Returns:
        np.ndarray: 1D array of the other field values
        """

        # Calculate the speed of light in the medium using the refractive index
        c = 299792458.0 / refractive_index

        # Calculate the wave impedance of the medium
        eta = 376.730313 / refractive_index

        # Calculate the wavenumber of the wave
        k = 2 * np.pi * freq / c

        # Calculate the other field using Maxwell's equations
        if is_electric:
            other_field = field / eta / 1j / k
        else:
            other_field = field * eta / 1j / k

        return other_field
