#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:27:37 2024

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import numpy as np
from scipy.fft import fft, ifft
from wave_bases import wave_bases



def wavelet(Y, dt, pad=0, dj=0.25, s0=-1, J1=-1, mother='MORLET', param=-1):
    """ 
    Computes the wavelet transform of the vector Y (length N), with sampling 
    rate dt.

    By default, the Morlet wavelet (k0=6) is used.
    The wavelet basis is normalized to have total energy=1 at all scales.
    
    Parameters
    ----------
    Y: numpy.ndarray
        The time series of length N.
    dt: float
        Amount of time between each Y value, i.e. the sampling time.
    pad: int, optional
        If set to 1 (default is 0), pad the time series with zeroes to get
        N up to the next higher power of 2. This prevents wraparound from 
        the end of the time series to the beginning, and also speeds up the FFT's 
        used to do the wavelet transform. This will not eliminate all edge effects 
        (see COI below).
    dj: float, optional
        The spacing between discrete scales. Default is 0.25. A smaller number 
        will give better scale resolution, but be slower to plot.
    s0: float, optional
        The smallest scale of the wavelet. Default is 2*DT.
    J1: int, optional
        The number of scales minus one. Scales range from S0 up to S0*2^(J1*DJ),
        to give a total of (J1+1) scales. Default is J1 = (LOG2(N DT/S0))/DJ.
    mother: str, optional
        The mother wavelet function. Choices are 'MORLET', 'PAUL', or 'DOG'.
        Default is 'MORLET'.
    param: int or float, optional
        The mother wavelet parameter. For 'MORLET' this is k0 (wavenumber), 
        default is 6. For 'PAUL' this is m (order), default is 4. For 'DOG' 
        this is m (m-th derivative), default is 2.
            
    Returns
    -------
    wave: numpy.ndarray
        The wavelet transform of Y. This is a complex array of dimensions (J1+1, N),
        where J1 is the number of scales and N is the length of the time series.
        FLOAT(WAVE) gives the wavelet amplitude, ATAN(IMAGINARY(WAVE),FLOAT(WAVE)) 
        gives the wavelet phase. The wavelet power spectrum is ABS(WAVE)^2.

    period: numpy.ndarray
        The vector of "Fourier" periods (in time units) that corresponds to the scales.
        This array has the same length as the number of scales (J1+1).

    scale: numpy.ndarray
        The vector of scale indices, given by S0*2^(j*DJ), j=0...J1 where J1+1 
        is the total number of scales.

    coi: numpy.ndarray
        The cone-of-influence, which is a vector of N points that contains the 
        maximum period of useful information at that particular time. Periods 
        greater than this are subject to edge effects. This array has the same 
        length as the time series (N).
        
    Note
    ----
    This script is based on the matlab package from Torrence and Compo (1995-1998)
    """

    n1 = len(Y)
    if s0 == -1:
        s0 = 2 * dt
    if J1 == -1:
        J1 = int(np.log2(n1 * dt / s0) / np.log2(2) / dj)
    if mother == -1:
        mother = 'MORLET'

    # Construct time series to analyze, pad if necessary
    x = Y - np.mean(Y)
    if pad == 1:
        base2 = int(np.log(n1) / np.log(2) + 0.4999)  # power of 2 nearest to N
        x = np.concatenate([x, np.zeros(2 ** (base2 + 1) - n1)])

    n = len(x)

    # Construct wavenumber array used in transform
    k = np.arange(1, n // 2 + 1)
    k = k * (2 * np.pi / (n * dt))
    k = np.concatenate([np.array([0.]), k, -k[int((n - 1) / 2 - 1)::-1]])

    # Compute FFT of the (padded) time series
    f = fft(x)

    # Construct scale array & empty period & wave arrays
    scale = s0 * 2 ** (np.arange(J1 + 1) * dj)
    period = scale
    wave = np.zeros((J1 + 1, n), dtype=complex)  # define the wavelet array

    # Loop through all scales and compute transform
    for a1 in range(J1 + 1):
        daughter, fourier_factor, coi, dofmin = wave_bases(mother, k, scale[a1], param)
        wave[a1, :] = ifft(f * daughter)  # wavelet transform

    period = fourier_factor * scale
    coi = coi * dt * np.concatenate([np.array([1E-5]), 1 + np.arange(1, (n1 + 1) // 2 - 1), np.flipud(np.arange(1, n1 // 2 - 1)), np.array([1E-5])])
    wave = wave[:, :n1]  # get rid of padding before returning

    return wave, period, scale, coi
