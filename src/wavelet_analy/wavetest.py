#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:29:38 2024


Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from wavelet import wavelet
from wave_signif import wave_signif
from wave_bases import wave_bases

#%%
# Load the data
sst_nino3 =  np.genfromtxt('../../data/test/sst_nino3.dat', dtype=None,
                     delimiter=',')
#sst_nino3 = data['sst_nino3']
sst = sst_nino3.flatten()

# Computation
variance = np.std(sst)**2
sst = (sst - np.mean(sst)) / np.sqrt(variance)

n = len(sst)
dt = 0.25
time = np.arange(0, n) * dt + 1871.0
xlim = [1870, 2000]
pad = 1
dj = 0.25
s0 = 2 * dt
j1 = int(7 / dj)
lag1 = 0.72
mother = 'Morlet'


#%%
# Wavelet transform
wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
power = np.abs(wave) ** 2

# Significance levels
signif, fft_theor = wave_signif([1.0], dt, scale, 0, lag1, -1, -1, mother)
sig95 = np.outer(signif, np.ones(n))
sig95 = power / sig95

# Global wavelet spectrum & significance levels
global_ws = variance * (np.sum(power, axis=1) / n)
dof = n - scale
global_signif = wave_signif(variance, dt, scale, 1, lag1, -1, dof, mother)

# Scale-average between El Nino periods of 2--8 years
avg = np.logical_and(scale >= 2, scale < 8)
Cdelta = 0.776
scale_avg = np.outer(scale, np.ones(n))
scale_avg = power / scale_avg
scale_avg = variance * dj * dt / Cdelta * np.sum(scale_avg[avg, :], axis=0)
scaleavg_signif = wave_signif(variance, dt, scale, 2, lag1, -1, [2, 7.9], mother)

#%%

# Plotting
plt.figure(figsize=(20, 20))
#%%
# Plot time series
plt.subplot(411)
plt.plot(time, sst)
plt.xlim(xlim)
plt.xlabel('Time (year)')
plt.ylabel('NINO3 SST (degC)')
plt.title('a) NINO3 Sea Surface Temperature (seasonal)')
#%%
# Contour plot wavelet power spectrum
plt.subplot()
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
Yticks = 2 ** np.arange(np.floor(np.log2(min(period))), np.ceil(np.log2(max(period))) + 1)
plt.contourf(time, np.log2(period), np.log2(power), np.log2(levels))
plt.xlabel('Time (year)')
plt.ylabel('Period (years)')
plt.title('b) NINO3 SST Wavelet Power Spectrum')
plt.xlim(xlim)
plt.ylim(np.log2([min(period), max(period)]))
plt.gca().invert_yaxis()
plt.yticks(np.log2(Yticks), Yticks)
plt.contour(time, np.log2(period), sig95, [-99, 1], colors='k')
plt.plot(time[1:-1], np.log2(coi), 'k')

#%%
# Plot global wavelet spectrum
plt.subplot(424)
plt.plot(global_ws, np.log2(period))
plt.plot(global_signif[0], np.log2(period), '--')
plt.xlabel('Power (degC^2)')
plt.title('c) Global Wavelet Spectrum')
plt.ylim(np.log2([min(period), max(period)]))
plt.gca().invert_yaxis()
plt.yticks(np.log2(Yticks), Yticks)
plt.xlim([0, 1.25 * max(global_ws)])

#%%

# Plot 2--8 yr scale-average time series
plt.subplot(413)
plt.plot(time, scale_avg)
plt.xlim(xlim)
plt.xlabel('Time (year)')
plt.ylabel('Avg variance (degC^2)')
plt.title('d) 2-8 yr Scale-average Time Series')
plt.plot(xlim, scaleavg_signif + [0, 0], '--')

plt.tight_layout()
plt.show()
