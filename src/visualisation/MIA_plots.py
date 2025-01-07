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
#miawara_Filename = 'MIAWARA_concat_H2O_2010_2023_inc_A.nc'
miawara_Filename = 'MIAWARA_concat_H2O_l2_scaled_plus_routine.nc'


miawara_fullpath = os.path.join(interim_dir, miawara_Filename)
MLS_fullpath = os.path.join(interim_dir, MLS_Bern_Filename)

#%%
miawara_xr = xr.load_dataset(miawara_fullpath, decode_times = False)
datetimes_mw = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in miawara_xr.time.values]
dt_64 = [np.datetime64(i) for i in datetimes_mw]
miawara_xr['time'] =  dt_64
miawara_xr = miawara_xr.sortby('time')
z =  np.array(miawara_xr['z'])


#%%
xArrayMia =  xr.load_dataset(miawara_fullpath, decode_times = False)
xArrayMia = xArrayMia.sortby('time')
datetimes_mw = [dt.datetime(1970,1,1)+dt.timedelta(seconds = int(i)) for i in xArrayMia.time.values]
dt_64 = np.array(datetimes_mw, dtype = 'datetime64[ns]')
xArrayMia['time'] = dt_64
dt_64_lst = list(dt_64)
xArrayMia = xArrayMia.sortby('time')

#water vapour
q = np.array(xArrayMia['q'])
q_a = np.array(xArrayMia['q_a'])

ArithmeticErrorq_a =  np.array(xArrayMia['q_a'])
q_err = np.array(xArrayMia['q_err'])

#pressure
p = np.array(xArrayMia['pressure'])

#time
datetime64_series = pd.Series(xArrayMia['time'].values)
datetime_list = datetime64_series.dt.to_pydatetime().tolist()
datetime_list = [dt.replace(hour=12, minute=0, second=0, microsecond=0) for dt in datetime_list]

years_MIA = np.array([date.year for date in datetime_list]) #corresponding year
doy = np.array([date.timetuple().tm_yday for date in datetime_list]) #day of year

conv = np.array(xArrayMia['convergence_flag'])


nan_height, nan_inds = np.where(q[10:40,:] <0.01e-6)
nan_inds = np.append(nan_inds, np.where(q[:,:] >15e-6)[1])
nan_inds =np.unique(nan_inds)

for i in nan_inds:
    q[:, i] = np.ones(len(q[:,0])) * np.nan
#%%

days_in_year = 366

years_MIA = np.array([date.year for date in datetimes_mw])

# Initialise arrays to store daily climatology and last year means
q_climatology_doy = np.full((q.shape[0], days_in_year), np.nan)
q_last_year_doy = np.full((q.shape[0], days_in_year), np.nan)

start_year = years_MIA.min()  # First year in dataset
end_year_clim = start_year + 8  # End of 8-year period

# Indices corresponding to the first 8 years
climatology_inds = np.where((years_MIA >= start_year) & (years_MIA <= end_year_clim))[0]

# Loop over each day of the year
for day in range(1, days_in_year + 1):
    # Indices for the first 8 years corresponding to the current DOY
    doy_clim_inds = doy[climatology_inds]
    
    if len(doy_clim_inds) > 0:
        # Calculate the daily mean for climatology (first 8 years)
        q_climatology_doy[:, day - 1] = np.nanmean(q[:, doy_clim_inds], axis=1)
    
    last_year = years_MIA.max()
    last_year_inds = np.where(years_MIA >= last_year-1)[0]
    # Indices for the last year corresponding to the current DOY
    doy_last_year_inds = doy[last_year_inds]
    date_last_year_inds = [datetimes_mw[int(i)] for i in last_year_inds]
    
    q_last_year_doy = q[:, last_year_inds]
    q_anom = np.full((q.shape[0], len(doy_last_year_inds)), np.nan)

    
    if len(doy_last_year_inds) > 0:
        for i, my_doy in enumerate(doy_last_year_inds):
            clim_inds = np.where((doy[climatology_inds] >= my_doy - 2) & (doy[climatology_inds] <= my_doy + 2))[0]
            if len(clim_inds) > 0:
                q_anom[:, i] = q_last_year_doy[:, i] - np.nanmean(q[:, climatology_inds[clim_inds]], axis=1)


# Difference between climatology and last year's daily mean
#q_diff_doy = q_last_year_doy - q_climatology_doy
#%%measurement response analysis

mr = miawara_xr['measurement_response']
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
ax.set_ylim(10, .010)
ax.set_xlim(dt.datetime(2015,10,1), dt.datetime(2024,10,1))

ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

min, max, diff = 2, 10, 0.01

# Initial contourf plot
cmap = plt.get_cmap('viridis')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('yellow')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = q * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure =p# miawara_xr['pressure'].values
time = datetime_list#  miawara_xr['time'].values
time_dt64 = np.array([np.datetime64(t) for t in time])


contour = ax.contourf(time,pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
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


plt.plot(filtered_time, pressure[filtered_maxDex]/100, color = 'orange')
plt.plot(filtered_time, pressure[filtered_minDex]/100,color = 'orange')

#Add grey to no data potts
for i in range(1, len(time)):
     gap = ((time_dt64[i] - time_dt64[i-1]) / np.timedelta64(1, 'D')).astype(int)
     if gap > 1:
         print(time_dt64[i])
         ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)
         
#plt.savefig(os.path.join(fig_file, 'MIA_time_series_2023_2024.png'), format = 'png', dpi = 400)

#%%

fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(3e1, .010)
ax.set_xlim(dt.datetime(2023,10,1), dt.datetime(2024,10,1))

ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

min, max, diff = -4, 4.0, 0.01

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('yellow')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = q_anom * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure = p
time = date_last_year_inds
time_dt64 = np.array([np.datetime64(t) for t in time])

contour = ax.contourf(time,pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Mixing Ratio (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min, max+.01, (max-min)/8)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

plt.plot(filtered_time, pressure[filtered_maxDex]/100, color = 'orange')
plt.plot(filtered_time, pressure[filtered_minDex]/100,color = 'orange')

#Add grey to no data potts
for i in range(1, len(time)):
     gap = ((time_dt64[i] - time_dt64[i-1]) / np.timedelta64(1, 'D')).astype(int)
     if gap > 1:
         print(time_dt64[i])
         ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)

#plt.savefig(os.path.join(fig_file, 'MIA_anomaly_time_series_2023_2024.png'), format = 'png', dpi = 400)


#%%
fig, ax = plt.subplots(figsize=(20, 8))

ax.set_ylim(20, .010)
ax.set_xlim(dt.datetime(2023,10,1), dt.datetime(2024,10,1))
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

min, max, diff = 2, 10, 0.01

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

contour = ax.contourf(time,pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Mixing Ratio (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min, max+.01, (max-min)/10)
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
        
plt.savefig(os.path.join(fig_file, 'MIA_ndacc_2022_2023.png'), format = 'png', dpi = 400)

#%%
fig, ax = plt.subplots(figsize=(20, 8))

ax.set_ylim(20, .010)
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

contour = ax.contourf(time,pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Mixing Ratio (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min, max+.01, (max-min)/10)
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


#plt.savefig(os.path.join(fig_file, 'MIA_anom_ndacc_2022_2023.png'), format = 'png', dpi = 400)

#%% plot the anomaly from MIAWARA
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(100, .0010)

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

contour = ax.contourf(time,pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
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
        
plt.savefig(os.path.join(fig_file, 'MIA_anomaly_time_series.png'), format = 'png', dpi = 400)


