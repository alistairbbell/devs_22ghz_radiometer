#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:43:04 2024

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""


#imports
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import sys
import matplotlib.dates as mdates
import uuid
from smb.SMBConnection import SMBConnection
import logging
import traceback
from netCDF4 import Dataset
from logging.handlers import RotatingFileHandler
from matplotlib import animation
from matplotlib.animation import PillowWriter
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from scipy import interpolate
from matplotlib.animation import FuncAnimation
import pandas as pd
sys.path.append('../../src/wavelet_analy')
from wavelet import *
from chisquare_inv import *
from chisquare_solve import *
from wave_signif import *

#%% paths
data_folder = "../../data"
interim_dir = os.path.join(data_folder, "interim")
fig_file = "../../output/figs"
h2o_string = 'retrieval'
log_folder = "../../log"

miawara_Filename_old = 'MIAWARA_concat_H2O_2010_2023_inc_A.nc'
miawara_Filename = 'MIAWARA_concat_H2O_2010_2023_rescaled_inc_A.nc'

miawara_old_fullpath = os.path.join(interim_dir, miawara_Filename_old)
miawara_fullpath = os.path.join(interim_dir, miawara_Filename)

#%%load dataset
miawara_xr = xr.load_dataset(miawara_fullpath, decode_times = False)
miawara_xr = miawara_xr.sortby('time')
miawara_xr = miawara_xr.drop('A')

datetimes_mia = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in miawara_xr.time.values]

doy_mia = [i.timetuple().tm_yday for i in datetimes_mia]

dt_64 = [np.datetime64(i).astype('datetime64[D]') for i in datetimes_mia]
miawara_xr['time'] = dt_64

z = miawara_xr.z.values
p =  miawara_xr.pressure.values
h2o_55k = miawara_xr.q.values[30,:]*1e6
h2o_55k_a = miawara_xr.q_a.values[30,:]*1e6


#%%anomaly calculations
df = pd.DataFrame({'Date': datetimes_mia, 'q': h2o_55k, 'q_a': h2o_55k_a})
df['DayOfYear'] = df['Date'].dt.dayofyear
yearly_median = df.groupby('DayOfYear')['q'].median()

df = df.merge(yearly_median_df, on='DayOfYear', suffixes=('', '_median'))
df = df.sort_values('Date')
df['q_median_moving_avg'] = df['q_median'].rolling(window=30, min_periods=1).mean()
df['SeasonalAnomaly'] = df['q'] - df['q_median_moving_avg']
df = df.sort_values('Date')
df['SeasonalAnomaly_running_mean'] = df['SeasonalAnomaly'].rolling(window=20, min_periods=1).mean()

mia_55k_season_anom = df['q'].to_numpy()

#%% interpolate for missing days
date_range = pd.date_range(start=dt_64[0], end=dt_64[-1], freq='D')
series = pd.Series(mia_55k_season_anom, index=dt_64).reindex(date_range)

# Perform spline interpolation
series_interpolated = series.interpolate(method='spline', order=2)

# Extract interpolated values and complete date range
complete_h2o = series_interpolated.values
complete_datetime = series_interpolated.index.values

#%% wavelet analysis
n = len(complete_h2o)
delta_t = 1
time = np.arange(0, n) + (dt.datetime(2010,10,29) - dt.datetime(1970,1,1) ).days
xlim = [time[0], time[-1]]
pad = 1
dj = 0.5 #spacing between discrete scales
s0 = 2 * delta_t
j1 = -1 #int(7 / dj)
mother = 'MORLET'
#lag1=0.72
#%% find the lag-1 correllation

series = pd.Series(complete_h2o)
lag1 = series.autocorr(lag=1)

#%%
wave, period, scale, coi = wavelet(complete_h2o, delta_t, pad, dj, s0, j1, mother)
power = np.abs(wave) ** 2

# Significance levels
signif, fft_theor = wave_signif([1.0], delta_t, scale, 0, lag1, -1, -1, mother)
sig95 = np.outer(signif, np.ones(n))
sig95 = power / sig95
#%%plots 
#Anomaly Plot

plt.plot(complete_datetime ,complete_h2o )
plt.ylim([-2,2])
#%%
plt.plot(np.arange(1,367), running_mean  )

#%%
plt.plot(df.Date,df.SeasonalAnomaly)
plt.ylim([-4,4])
plt.xlim([dt.datetime(2010,10,1), dt.datetime(2023,11,1)])
plt.grid()
plt.ylabel('Water Vapour Mixing Ratio Anomaly (PPMV)')
plt.xlabel('Time')
#plt.savefig('/home/alistair/miawara_reprocessing/output/figs/h2o_series_anomaly.png', format = 'png')


#%%
# Contour plot wavelet power spectrum
fig, ax = plt.subplots(figsize=(12, 8))
levels = np.arange(-4,4,1)
Yticks = 2 ** np.arange(np.floor(np.log2(min(period))), np.ceil(np.log2(max(period))) + 1)

lowest_viridis_color = plt.cm.viridis(0)
highest_viridis_color = plt.cm.viridis(1.0)

cmap = plt.cm.viridis 
cmap.set_under(lowest_viridis_color)  
cmap.set_over(highest_viridis_color)  

contour = plt.contourf(complete_datetime, np.log2(period), np.log2(power), levels=levels, cmap=cmap, extend='both')
cbar = plt.colorbar(contour)
cbar.set_label('Power Spectrum Intensity (PPMV\u00B2)', fontsize=15)  # Adding color bar label with squared units
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Time (year)', fontsize = 15)
plt.ylabel('Period (days)', fontsize = 15)
plt.title('H2O Wavelet Power Spectrum', fontsize = 15)
plt.xlim( [dt.datetime(2022,1,1), dt.datetime(2023,9,20)])
plt.gca().invert_yaxis()
plt.yticks(np.log2(Yticks)[:-1], Yticks[:-1])
plt.ylim([6,1])

#cbar.set_label('Water Vapour Anomaly (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')


plt.contour(time[:], np.log2(period), sig95[:,:], colors='k')
plt.plot(time[1:-1], np.log2(coi), 'k')


for t, c in zip(time[1-1::10], coi[::10]):
    y_crosses = np.linspace(np.log2(c), np.log2(min(period)), num=10)
    x_crosses = np.full_like(y_crosses, t)
    plt.plot([t, t], [np.log2(c), np.log2(max(period))], color='black', linestyle='--')


#plt.savefig('/home/alistair/miawara_reprocessing/output/figs/h2o_power_spectrum_45km_2010_2023.png', format = 'png')



#%%
fig, ax = plt.subplots(figsize=(12, 8))
contour = plt.contourf(complete_datetime, np.log2(period), np.log2(power), levels=levels)
cbar = plt.colorbar(contour)
cbar.ax.tick_params(labelsize=20)
plt.xlabel('Time (year)')
plt.ylabel('Period (days)')
plt.title('H2O Wavelet Power Spectrum')
plt.xlim(xlim)
plt.ylim(np.log2([min(period), max(period)]))
plt.gca().invert_yaxis()
plt.yticks(np.log2(Yticks)[:-1], Yticks[:-1])
plt.contour(time[:], np.log2(period), sig95[:,:], colors='k')
plt.plot(time[:], coi_adjusted, 'k')


