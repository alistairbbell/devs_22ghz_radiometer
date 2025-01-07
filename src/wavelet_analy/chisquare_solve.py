#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:24:40 2024

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import scipy.special as sp

def chisquare_solve(XGUESS, P, V):
    """
    Calculates the difference between a target percentile and the estimated percentile 
    in a chi-square distribution.
    
    This function is an internal component used by the 'chisquare_inv' function. It 
    computes the discrepancy between a given percentile (P) and the percentile 
    calculated using a guessed chi-square statistic value (XGUESS) under a specified 
    number of degrees of freedom (V).
    
    Parameters
    ----------
    XGUESS: float
        The estimated chi-square statistic value.
    P: float
        The target percentile in the chi-square distribution, expressed as a 
        decimal (e.g., 0.95 for the 95th percentile).
    V: int
        The degrees of freedom in the chi-square distribution.
    
    Returns
    -------
    PDIFF: float
        The absolute difference between the estimated percentile (calculated 
        using XGUESS and V) and the target percentile P. This value represents 
        the accuracy of XGUESS in estimating the percentile P.
    Note
    ----
    This script is based on the matlab package from Torrence and Compo (1995-1998)
    """

    # Calculate the guessed probability using the incomplete gamma function
    PGUESS = sp.gammainc(V / 2, V * XGUESS / 2)

    # Calculate the error in the guessed probability
    PDIFF = abs(PGUESS - P)

    # Tolerance for near-1 probability
    TOL = 1E-4
    if PGUESS >= 1 - TOL:  # if P is very close to 1 (i.e., a bad guess)
        PDIFF = XGUESS  # then just assign some large number like XGUESS

    return PDIFF
