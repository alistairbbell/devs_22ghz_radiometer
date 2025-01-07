#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:13:37 2023

Author: Alistair Bell

Contact: alistair.bell@unibe.ch

Description: Significance testing for the 1D Wavelet transform wavelet
codes for running wavelet analysis translated from maltab from 
scripts written by C. Torrence. 
"""

import numpy as np
from scipy.stats import chi2

def wave_signif(Y, dt, scale1, sigtest=-1, lag1=-1, siglvl=-1, dof=-1, mother=-1, param=-1):
    """
    Computes the significance levels for a wavelet transform, considering a red-noise 
    background spectrum.
    
    This function takes a time series (or its variance) and, using a given sampling 
    time and a set of scale indices from a previous wavelet transform, calculates the 
    significance levels for each scale in the context of a red-noise background 
    spectrum. It is particularly useful for assessing the statistical significance of 
    wavelet power spectra.
    
    Parameters
    ----------
    Y: numpy.ndarray or float
        The time series data, or if a single value is provided, it is treated as 
        the variance of the time series.
    dt: float
        The sampling time, i.e., the time interval between consecutive data points 
        in the time series Y.
    scale1: numpy.ndarray
        An array of scale indices obtained from a previous call to a wavelet 
        transform function.
    
    Returns
    -------
    SIGNIF: numpy.ndarray
        An array of significance levels corresponding to each scale in the SCALE 
        vector. These levels can be used to interpret the wavelet power spectrum 
        in the context of a red-noise background.
    FFT_THEOR: numpy.ndarray
        The theoretical red-noise spectrum as a function of period. This output 
        provides a baseline spectrum for comparison with the actual wavelet power 
        spectrum, aiding in the identification of significant power peaks against 
        the background noise.
        
    Note
    ----
    This script is based on the matlab package from Torrence and Compo (1995-1998)
    """

    if mother == -1:
        mother = "MORLET"
    mother = mother.upper()

    if isinstance(Y, (list, np.ndarray)):
        n1 = len(Y)
    elif isinstance(Y, float):
        n1 = 1
    else:
        raise TypeError("Y must be either a list, numpy array, or float")
    J1 = len(scale1) - 1
    scale = scale1.copy()

    s0 = min(scale)
    dj = np.log(scale[1] / scale[0]) / np.log(2)

    variance = Y if n1 == 1 else np.var(Y)

    if lag1 == -1:
        lag1 = 0.0
    if siglvl == -1:
        siglvl = 0.95

    # Define parameters based on mother wavelet
    if mother == 'MORLET':
        param = 6. if param == -1 else param
        k0 = param
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        empir = [2., -1, -1, -1]
        if k0 == 6:
            empir[1:] = [0.776, 2.32, 0.60]
    elif mother == 'PAUL':
        param = 4. if param == -1 else param
        m = param
        fourier_factor = 4 * np.pi / (2 * m + 1)
        empir = [2., -1, -1, -1]
        if m == 4:
            empir[1:] = [1.132, 1.17, 1.5]

    elif mother == 'DOG':
        param = 2. if param == -1 else param
        m = param
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
        empir = [1., -1, -1, -1]
        if m == 2:
            empir[1:] = [3.541, 1.43, 1.4]
        elif m == 6:
            empir[1:] = [1.966, 1.37, 0.97]

    else:
        raise ValueError("Mother must be one of 'MORLET', 'PAUL', 'DOG'")


    period = scale * fourier_factor
    dofmin = empir[0]
    Cdelta = empir[1]
    gamma_fac = empir[2]
    dj0 = empir[3]

    freq = dt / period
    fft_theor = (1 - lag1 ** 2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1 ** 2)
    fft_theor *= variance
    signif = fft_theor.copy()

    dof = np.where(dof == -1, dofmin, dof)

    # Perform significance tests
    if sigtest == 0:
        dof = dofmin
        chisquare = chi2.ppf(siglvl, dof) / dof
        signif = fft_theor * chisquare
    elif sigtest == 1:
        if len(dof) == 1:
            dof = np.zeros(J1 + 1) + dof
        truncate = np.where(dof < 1)
        dof[truncate] = 1
        dof = dofmin * np.sqrt(1 + (dof * dt / gamma_fac / scale) ** 2)
        truncate = np.where(dof < dofmin)
        dof[truncate] = dofmin
        signif = np.zeros(J1 + 1)
        for a1 in range(J1 + 1):
            chisquare = chi2.ppf(siglvl, dof[a1]) / dof[a1]
            signif[a1] = fft_theor[a1] * chisquare

    elif sigtest == 2:
        if len(dof) != 2:
            raise ValueError('DOF must be set to [S1, S2], the range of scale-averages')
        if Cdelta == -1:
            raise ValueError(f'Cdelta and dj0 not defined for {mother} with param = {param}')
        s1, s2 = dof
        avg = np.where((scale >= s1) & (scale <= s2))[0]
        navg = len(avg)
        if navg == 0:
            raise ValueError(f'No valid scales between {s1} and {s2}')
        Savg = 1. / np.sum(1. / scale[avg])
        Smid = np.exp((np.log(s1) + np.log(s2)) / 2.)
        dof = (dofmin * navg * Savg / Smid) * np.sqrt(1 + (navg * dj / dj0) ** 2)
        fft_theor_avg = np.sum(fft_theor[avg] / scale[avg])
        fft_theor = Savg * fft_theor_avg
        chisquare = chi2.ppf(siglvl, dof) / dof
        signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare

    else:
        raise ValueError('sigtest must be either 0, 1, or 2')

    return signif, fft_theor


