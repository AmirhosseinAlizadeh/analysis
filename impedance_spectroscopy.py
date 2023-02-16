import numpy as np
from scipy.optimize import curve_fit
"""
Impedance spectroscopy is a technique that measures the impedance of a material or system as a function of frequency 
to extract information about its electrical and physical properties. 
An equivalent circuit model is commonly used to analyze the impedance data. 
The parallel_rc_series function models the equivalent impedance of parallel units of resistors and capacitors connected
in series, and the fit_parallel_rc_series function fits experimental impedance data to this model. 
These functions enable the analysis of impedance spectroscopy data to extract important information about the
 electrical and physical properties of a material or system.
"""
def parallel_rc_series(n, R, C):
    """
    Calculates the equivalent impedance of n parallel units of resistors and capacitors connected in series.

    Parameters:
    n (int): Number of parallel units
    R (np.ndarray): 1D array of resistance values for each unit
    C (np.ndarray): 1D array of capacitance values for each unit

    Returns:
    complex: Equivalent impedance
    """
    # Calculate the impedance of each parallel unit
    s = 1j * 2 * np.pi * 1.0
    Yr = 1 / R
    Yc = s * C
    Y = Yr + Yc
    Zp = 1 / Y

    # Calculate the equivalent impedance of the series connection of the parallel units
    Zeq = Zp
    for i in range(n-1):
        Zeq += Zp

    return Zeq

def fit_parallel_rc_series(freq, Z_exp, n0, R0, C0):
    """
    Fits experimental data to the equivalent impedance of n parallel units of resistors and capacitors connected in series.

    Parameters:
    freq (np.ndarray): 1D array of frequencies for the experimental data
    Z_exp (np.ndarray): 1D array of complex impedance values for the experimental data
    n0 (int): Initial guess for the number of parallel units
    R0 (float): Initial guess for the resistance value of each unit
    C0 (float): Initial guess for the capacitance value of each unit

    Returns:
    tuple: Tuple containing the optimal values for the number of parallel units and the resistance and capacitance values for each unit
    """
    # Define the function to fit to the experimental data
    def Z_model(freq, n, R, C):
        return parallel_rc_series(n, np.full(n, R), np.full(n, C)) # Assumes all units have the same R and C values

    # Perform the curve fit using the scipy.optimize.curve_fit function
    p0 = [n0, R0, C0]
    popt, pcov = curve_fit(Z_model, freq, Z_exp, p0=p0)

    # Extract the optimal parameter values
    n_opt = int(round(popt[0]))
    R_opt = popt[1]
    C_opt = popt[2]

    return (n_opt, R_opt, C_opt)
