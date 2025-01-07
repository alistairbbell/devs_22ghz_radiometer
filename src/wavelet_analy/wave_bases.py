#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:26:05 2024

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import numpy as np
from scipy.special import gamma

def wave_bases(mother, k, scale, param):
    
    """
    Calculates the wavelet function for specified Fourier frequencies, scales, and 
    parameters, essential for performing a wavelet transform in Fourier space.
    
    This function, typically called by the 'wavelet' function, computes the daughter 
    wavelet (the wavelet function itself) and related parameters based on the chosen 
    mother wavelet type. It facilitates the transformation of data into the wavelet 
    domain, particularly useful for analyzing time series in terms of frequency and 
    scale.
    
    Parameters
    ----------
    MOTHER: str
        The type of mother wavelet to use. Accepted values are 'MORLET', 'PAUL', 
        or 'DOG'.
    K: numpy.ndarray
        A vector of Fourier frequencies at which the wavelet function will be 
        evaluated.
    SCALE: float
        The wavelet scale, which determines the resolution of the wavelet analysis.
    PARAM: float
        The nondimensional parameter specific to the chosen wavelet function, 
        which affects the shape and characteristics of the wavelet.
    
    Returns
    -------
    DAUGHTER: numpy.ndarray
        The wavelet function evaluated at the specified Fourier frequencies and 
        scale. This vector represents the wavelet used in the transform.
    FOURIER_FACTOR: float
        The ratio of the Fourier period to the wavelet scale, providing a link 
        between the wavelet and Fourier analyses.
    COI: float
        The size of the cone-of-influence at the specified scale, indicating the 
        region in the wavelet spectrum where edge effects become significant.
    DOFMIN: float
        The minimum number of degrees of freedom for each point in the wavelet 
        power spectrum. This value is typically 2 for 'Morlet' and 'Paul' wavelets, 
        and 1 for the 'DOG' wavelet.
    
    Note
    ----
    This script is based on the matlab package from Torrence and Compo (1995-1998)
    """
    mother = mother.upper()
    n = len(k)

    if mother == 'MORLET':  # Morlet wavelet
        if param == -1:
            param = 6.
        k0 = param
        expnt = -((scale * k - k0) ** 2) / 2.0 * (k > 0)
        norm = np.sqrt(scale * k[1]) * (np.pi ** -0.25) * np.sqrt(n)
        daughter = norm * np.exp(expnt)
        daughter *= (k > 0)  # Heaviside step function
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 2

    elif mother == 'PAUL':  # Paul wavelet
        if param == -1:
            param = 4.
        m = param
        expnt = -(scale * k) * (k > 0)
        norm = np.sqrt(scale * k[1]) * (2 ** m / np.sqrt(m * np.prod(np.arange(2, 2 * m)))) * np.sqrt(n)
        daughter = norm * ((scale * k) ** m) * np.exp(expnt)
        daughter *= (k > 0)  # Heaviside step function
        fourier_factor = 4 * np.pi / (2 * m + 1)
        coi = fourier_factor * np.sqrt(2)
        dofmin = 2

    elif mother == 'DOG':  # DOG wavelet
        if param == -1:
            param = 2.
        m = param
        expnt = -((scale * k) ** 2) / 2.0
        norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
        daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
        fourier_factor = 2 * np.pi * np.sqrt(2.0 / (2 * m + 1))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 1

    else:
        raise ValueError("Mother must be one of 'MORLET', 'PAUL', 'DOG'")

    return daughter, fourier_factor, coi, dofmin
