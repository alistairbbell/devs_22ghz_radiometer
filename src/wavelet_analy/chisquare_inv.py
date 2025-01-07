#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:23:58 2024

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import scipy.stats as stats

def chisquare_inv(P, V):
    """
    Inverse of chi-square cumulative distribution function (cdf).

    Parameters:
    P: Probability for which the inverse cdf is computed
    V: Degrees of freedom

    Returns:
    X: The value such that P*100 percent of the chi-square distribution
       with V degrees of freedom lies between 0 and X.
       
    Note
    ----
    This script is based on the matlab package from Torrence and Compo (1995-1998)
    """

    if P < 0 or P > 0.9999:
        raise ValueError("P must be in the range (0, 0.9999)")
    if V <= 0:
        raise ValueError("Degrees of freedom V must be positive")

    # Special case
    if P == 0.95 and V == 2:
        return 5.9915

    # Use the SciPy function to compute the inverse cdf
    X = stats.chi2.ppf(P, V)

    return X
