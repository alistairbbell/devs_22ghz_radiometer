#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:40:16 2023

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import sys
sys.path.append('../../')
import matplotlib.dates as mdates
import uuid
from smb.SMBConnection import SMBConnection
import logging
import traceback
from netCDF4 import Dataset
from logging.handlers import RotatingFileHandler
from scipy import interpolate
import matplotlib.colors as mcolors

#%% paths
#set working directory
script_location = os.path.abspath(__file__)

#directory containing the script
script_directory = os.path.dirname(script_location)

#set working dir
os.chdir(script_directory)
#%%
data_folder = "../../data"
interim_dir = os.path.join(data_folder, "interim")
fig_file = "../../output/figs"
h2o_string = 'retrieval'
log_folder = "../../log"
MLS_Bern_Filename = 'MLS_concat_H2O.nc'
#miawara_Filename = 'MIAWARA_C_concat_H2O_2022_2024_rescaled.nc'
miawara_Filename = 'MIAWARA_C_concat_H2O_2018_2024_rescaled_ndacc.nc'
miawara_fullpath = os.path.join(interim_dir, miawara_Filename)
MLS_fullpath = os.path.join(interim_dir, MLS_Bern_Filename)

#%%
miawara_xr = xr.load_dataset(miawara_fullpath, decode_times = False)
datetimes_mw = [dt.datetime(1970,1,1) + dt.timedelta(seconds = float(i)) for i in miawara_xr.time.values]
dt_64 = [np.datetime64(i) for i in datetimes_mw]
miawara_xr['time'] =  dt_64
miawara_xr = miawara_xr.sortby('time')
miawara_xr = miawara_xr.sel(time=slice('2020-01-01', '2024-12-31'))

#%%measurement response analysis

mr = np.array(miawara_xr['measurement_response']).T
maxDex = []
minDex = []

for i in range(len(mr[0,:])):
    mr_valid = np.where(mr[:,i]>0.6)[0]
    if len(mr_valid>0):
        maxDex.append(mr_valid.max()+1)
        minDex.append(mr_valid.min()-1)
    else:
        maxDex.append(np.nan)
        minDex.append(np.nan)

#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(4e3, .10)
ax.set_xlim(dt.datetime(2023,10,1), dt.datetime(2024,10,1))

ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

min, max, diff = 2, 9, 0.01

# Initial contourf plot
cmap = plt.get_cmap('viridis')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('yellow')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = miawara_xr.q.values * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure = miawara_xr['pressure'].values
time = miawara_xr['time'].values

contour = ax.contourf(time,pressure/100,plotting_data.T, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Mixing Ratio (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min, max+.01, (max-min)/7)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

nan_indices =  np.where(np.isnan(maxDex))
filtered_minDex = np.delete(minDex, nan_indices)
filtered_maxDex = np.delete(maxDex, nan_indices)
filtered_time = np.delete(time, nan_indices)

filtered_minDex = [int(i) for i in filtered_minDex]
filtered_maxDex = [int(i) for i in filtered_maxDex]


#plt.plot(filtered_time, pressure[filtered_maxDex]/100, color = 'orange')
#plt.plot(filtered_time, pressure[filtered_minDex]/100,color = 'orange')

#Add grey to no data potts
for i in range(1, len(time)):
     gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
     if gap > 3:
         print(time[i])
         ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)
         
    
#plt.savefig(os.path.join(fig_file, 'MIA_C_time_series.png'), format = 'png', dpi = 400)

#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(4e3, .10)
ax.set_xlim(dt.datetime(2023,10,1), dt.datetime(2024,10,1))

ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

min, max, diff = -4, 4, 0.01

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = (miawara_xr.q.values - miawara_xr.q_a.values) * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure = miawara_xr['pressure'].values
time = miawara_xr['time'].values

contour = ax.contourf(time,pressure/100,plotting_data.T, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Mixing Ratio (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min, max+.01, (max-min)/7)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

nan_indices =  np.where(np.isnan(maxDex))
filtered_minDex = np.delete(minDex, nan_indices)
filtered_maxDex = np.delete(maxDex, nan_indices)
filtered_time = np.delete(time, nan_indices)

filtered_minDex = [int(i) for i in filtered_minDex]
filtered_maxDex = [int(i) for i in filtered_maxDex]


#plt.plot(filtered_time, pressure[filtered_maxDex]/100, color = 'orange')
#plt.plot(filtered_time, pressure[filtered_minDex]/100,color = 'orange')

#Add grey to no data potts
for i in range(1, len(time)):
     gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
     if gap > 3:
         print(time[i])
         ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)
         


#%% plot the anomaly from MIAWARA
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(1e3, .10)
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

min, max, diff = -2,2,0.01

plotting_data = miawara_xr.q.values - miawara_xr.q_a.values
plotting_data = plotting_data*1e6
plotting_data = np.where(plotting_data<min, min, plotting_data)
plotting_data = np.where(plotting_data>max, max, plotting_data)

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

contour = ax.contourf(time,pressure/100,plotting_data.T, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour)
cbar.set_label(r'$\Delta$ q (PPMV)', fontsize = 20)
ticks_loc = np.arange(min, max+.01, (max-min)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data points
for i in range(1, len(time)):
    gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
    if gap > 10:
        print(time[i])
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)
        
plt.savefig(os.path.join(fig_file, 'MIA_C_anomaly_time_series.png'), format = 'png', dpi = 400)


#%%



