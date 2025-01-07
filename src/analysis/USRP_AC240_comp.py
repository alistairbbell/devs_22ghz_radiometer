#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:37:17 2023

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

#%%
#imports
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
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
<<<<<<< Updated upstream
from matplotlib.ticker import MultipleLocator

=======
>>>>>>> Stashed changes

#%%Paths
#set file directory to working dir
absolute_script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(absolute_script_path)
os.chdir(script_directory)

#%%
data_folder = "../../data"
interim_dir = os.path.join(data_folder, "interim")
fig_file = "../../output/figs"
h2o_string = 'retrieval'
log_folder = "../../log"
miawara_Filename = 'MIAWARA_concat_H2O_2010_2023_inc_A.nc'
miawara_ac240_scaled_Filename = 'MIAWARA_concat_H2O_202_2023_rescaled_inc_A.nc'
miawara_usrp_Filename = 'MIAWARA_USRP_incNoise_concat_H2O_2022_2023.nc'

#%%output of plot
plotFigAvK = 'av_kernel_sensitivity_usrp.pdf'
plotFigAvK_ac240 = 'av_kernel_sensitivity_ac240_rescaled.pdf'

#%% functions
def compute_hpbw(kernel, pressure):
    """
    Computes the pressre at the maxima fo the averaging kernels, and upper
    and lower pressures at which the averaging kernel is equal to half of the 
    maxima.

    Args:
    - kernel (array-like): The values of the averaging kernel.
    - pressure (array-like): Corresponding pressure levels for the kernel values.

    Returns:
    - hpbw (float): The HPBW value in terms of pressure difference.
    """
    
    pressure_dense = np.logspace(np.log10(pressure.min()), np.log10(pressure.max()), 1000)
    f = interp1d(pressure, kernel, kind='cubic')
    kernel_smooth = f(pressure_dense)
    idx_max = np.argmax(kernel_smooth)

    # Retrieve the corresponding pressure
    pressure_at_max_kernel = pressure_dense[idx_max]
    kernelMax = kernel_smooth[idx_max]
    
    # Find the maximum of the kernel
    max_val_ind = np.argmax(kernel)

    # Find the half power value
    half_power_val =kernelMax/ 2

    p_dense_left = pressure_dense[:idx_max]
    kernel_smooth_left = f(pressure_dense[:idx_max])
    idx_half_left = np.argmin(np.abs(kernel_smooth_left - half_power_val))
    P_half_left = p_dense_left[idx_half_left]
        
    p_dense_right = pressure_dense[idx_max:]
    kernel_smooth_right = f(p_dense_right)
    idx_half_right = np.argmin(np.abs(kernel_smooth_right - half_power_val))
    P_half_right = p_dense_right[idx_half_right]
    
    return P_half_left, P_half_right, pressure_at_max_kernel


#%%load arrays

miawara_fullpath = os.path.join(interim_dir, miawara_Filename)
miawara_ac240_xr = xr.open_dataset(miawara_fullpath, decode_times = False)

miawara_fullpath_ac240_scaled = os.path.join(interim_dir, miawara_ac240_scaled_Filename)
miawara_ac240_scaled_xr = xr.open_dataset(miawara_fullpath_ac240_scaled, decode_times = False)

miawara_usrp_fullpath =  os.path.join(interim_dir, miawara_usrp_Filename)
miawara_usrp_xr = xr.open_dataset(miawara_usrp_fullpath, decode_times = False)


#%% analyse the differences in retrieved profile
ac240_time = miawara_ac240_xr.time.values
ac240_scaled_time = miawara_ac240_scaled_xr.time.values
usrp_time = miawara_usrp_xr.time.values 

indices_ac240 = np.where(np.isin(ac240_time, usrp_time) & np.isin(ac240_time, ac240_scaled_time))[0]
indices_ac240_scaled = np.where(np.isin(ac240_scaled_time, usrp_time) & np.isin(ac240_scaled_time, ac240_time))[0]
indices_usrp = np.where(np.isin(usrp_time, ac240_time) & np.isin(usrp_time, ac240_scaled_time))[0]

ac240_q = miawara_ac240_xr.q.values[:,indices_ac240]
ac240_scaled_q = miawara_ac240_scaled_xr.q.values[:,indices_ac240_scaled]
usrp_q = miawara_usrp_xr.q.values[:,indices_usrp]
prior_q = miawara_usrp_xr.q_a.values[:,indices_usrp]

ac240_time = miawara_ac240_xr.time.values[indices_ac240]
ac240_scaled_time = miawara_ac240_scaled_xr.time.values[indices_ac240_scaled]
usrp_time = miawara_usrp_xr.time.values[indices_usrp]

q_diff = usrp_q - ac240_q
q_scaled_diff = usrp_q - ac240_scaled_q

#%%
A_ac240 = np.nanmean(miawara_ac240_xr.A.values, axis = 2)
A_usrp =  np.nanmean(miawara_usrp_xr.A.values, axis = 2)
pressure = miawara_ac240_xr.pressure.values
pressureusrp = miawara_ac240_xr.pressure.values


#%%find hpbw
left_ac240, right_ac240, max_ac240 = (np.zeros(80),np.zeros(80),np.zeros(80))
left_usrp, right_usrp, max_usrp = (np.zeros(80),np.zeros(80),np.zeros(80))

for i in range(80):
    left_ac240[i], right_ac240[i], max_ac240[i] = compute_hpbw(A_ac240[:,i], pressure)
    
#%%
for i in range(80):
    left_usrp[i], right_usrp[i], max_usrp[i] = compute_hpbw(A_usrp[:,i], pressure)







###############################################################################
############################ PLOTS ############################################
###############################################################################
#%% plotting of averaging kernel maximum

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_ylim(100, .0010)
ax.set_xlim(100, .0010)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel('Av. kernel central Pressure (hPa)', fontsize = 20)
ax.set_ylabel('Level Pressure (hPa)', fontsize = 20)

plt.plot(np.logspace(-3,2,60),np.logspace(-3,2,60), color = 'black', linestyle = 'dashed')
plt.plot(max_ac240/100, pressure/100, label = 'AC-240', linewidth = 4)
plt.plot(max_usrp/100, pressure/100, color = 'orange', label = 'USRP', linewidth = 4)
plt.legend( fontsize = 18)
#%%
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_ylim(100, .0010)
ax.set_xlim(100, .001)

ax.grid()
ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel('Pressure of Maximum Sensitivity (hPa)', fontsize = 20)
ax.set_ylabel('Level Pressure (hPa)', fontsize = 20)

plt.legend( fontsize = 18)
plt.plot(np.logspace(-3,2,60),np.logspace(-3,2,60), color = 'black', alpha = 0.3)

plt.errorbar(max_ac240/100,  pressure/100, xerr=[(max_ac240 - left_ac240)/100, (right_ac240 - max_ac240)/100], fmt='o', capsize=5, label = 'AC-240')
plt.errorbar(max_usrp/100,  pressure/100, xerr=[(max_usrp - left_usrp)/100, (right_usrp - max_usrp)/100], fmt='o', capsize=5, label = 'USRP')
plt.legend( fontsize = 18)

plt.savefig(os.path.join(fig_file,plotFigAvK), format = 'pdf')

#%% Plotting of miawara averaging kernel
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('A (PPMV/PPMV)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

plt.grid(which = 'major', alpha = 0.5, color = 'black')
plt.grid(which = 'minor', alpha = 0.2)

colors = ["darkblue", "blue", "green", "darkgoldenrod", "orange", "red"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=50)

for i in range(80):
    ax.plot( A_usrp[:,i],pressure/100, alpha=0.3, color = 'black')

for i in np.arange(4,60, 5):
    ax.plot(A_usrp[:,i],pressure/100, alpha=1, label = '{:.2g}hPa'.format(pressure[i]/100), color=cmap(i/55))
plt.legend()

<<<<<<< Updated upstream
ax.xaxis.set_major_locator(MultipleLocator(0.01))  

plt.savefig(os.path.join(fig_file, 'averaging_kernel_usrp_multicolor_finegrid.pdf'), format = 'pdf')
=======
plt.savefig(os.path.join(fig_file, 'averaging_kernel_usrp_multicolor.pdf'), format = 'pdf')
>>>>>>> Stashed changes


#%% for AC-240
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('A (PPMV/PPMV)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

plt.grid(which = 'major', alpha = 0.5, color = 'black')
plt.grid(which = 'minor', alpha = 0.2)

colors = ["darkblue", "blue", "green", "darkgoldenrod", "orange", "red"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=50)

for i in range(80):
    ax.plot( A_ac240[:,i],pressure/100, alpha=0.3, color = 'black')

for i in np.arange(4,60, 5):
    ax.plot(A_ac240[:,i],pressure/100, alpha=1, label = '{:.2g}hPa'.format(pressure[i]/100), color=cmap(i/55))
plt.legend()

plt.savefig(os.path.join(fig_file, 'averaging_kernel_multicolor_ac240.pdf'), format = 'pdf')


#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)
ax.tick_params(axis='x', rotation=45)

min, max, diff = -2,2, 0.01

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = q_diff[:,:-1] * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure = miawara_usrp_xr.pressure.values
datetimes = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in ac240_time]

contour = ax.contourf(datetimes[:-1],pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label(r'$\delta$ q (PPMV)', fontsize = 20)
ticks_loc = np.arange(min, max+.01, (max-min)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(datetimes)):
    gap = int((datetimes[i] - datetimes[i-1]) / dt.timedelta(days = 1))
    if gap > 10:
        print(datetimes[i])
        ax.axvspan(datetimes[i-1], datetimes[i], facecolor='grey', alpha=1)
        
plt.savefig(os.path.join(fig_file, 'AC240_USRP_diffseries.png'), format = 'png', dpi = 400)

#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)
ax.tick_params(axis='x', rotation=45)

min, max, diff = -2,2, 0.01

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = q_scaled_diff[:,:-1] * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure = miawara_usrp_xr.pressure.values
datetimes = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in ac240_time]

contour = ax.contourf(datetimes[:-1],pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label(r'$\delta$ q (PPMV)', fontsize = 20)
ticks_loc = np.arange(min, max+.01, (max-min)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(datetimes)):
    gap = int((datetimes[i] - datetimes[i-1]) / dt.timedelta(days = 1))
    if gap > 10:
        print(datetimes[i])
        ax.axvspan(datetimes[i-1], datetimes[i], facecolor='grey', alpha=1)
        
plt.savefig(os.path.join(fig_file, 'AC240_scaled_USRP_diffseries.png'), format = 'png', dpi = 400)


#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)
ax.tick_params(axis='x', rotation=45)

min, max, diff = 2,10, 0.01

# Initial contourf plot
cmap = plt.get_cmap('viridis')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = usrp_q * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure = miawara_usrp_xr.pressure.values
datetimes = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in ac240_time]

contour = ax.contourf(datetimes,pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Mixing Ratio (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min, max+.01, (max-min)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(datetimes)):
    gap = int((datetimes[i] - datetimes[i-1]) / dt.timedelta(days = 1))
    if gap > 10:
        print(datetimes[i])
        ax.axvspan(datetimes[i-1], datetimes[i], facecolor='grey', alpha=1)
        
plt.savefig(os.path.join(fig_file, 'USRP_series.png'), format = 'png', dpi = 400, bbox_inches = 'tight')


#%%
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_ylim(100, .010)
ax.set_yscale('log')
ax.set_xlabel('Latitude (degrees)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

plt.plot(ac240_q[:, -10], pressure/100, color = 'purple')
plt.plot(usrp_q[:, -10], pressure/100, color = 'orange')

# plt.plot(miawara_usrp_xr.q.values[:, 50], pressure/100, color = 'red')
# plt.plot(miawara_usrp_xr.q_a.values[:, 50], pressure/100, color = 'black')

#%%Profile plot
fig, ax = plt.subplots(figsize=(8, 12))
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('q (PPMV)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plt.grid(which = 'major', alpha = 0.9)
ax.grid(which='minor', axis = 'both', alpha=0.2)

usrpq_median = np.nanpercentile(usrp_q, 50, axis = 1)*1e6
usrpq_25 = np.nanpercentile(usrp_q, 25, axis = 1)*1e6
usrpq_75 = np.nanpercentile(usrp_q, 75, axis = 1)*1e6

ac240q_median = np.nanpercentile(ac240_q, 50, axis = 1)*1e6
ac240q_25 = np.nanpercentile(ac240_q, 25, axis = 1)*1e6
ac240q_75 = np.nanpercentile(ac240_q, 75, axis = 1)*1e6

ac240q_scaled_median = np.nanpercentile(ac240_scaled_q, 50, axis = 1)*1e6
ac240q_scaled_25 = np.nanpercentile(ac240_scaled_q, 25, axis = 1)*1e6
ac240q_scaled_75 = np.nanpercentile(ac240_scaled_q, 75, axis = 1)*1e6

priorq_median = np.nanpercentile(prior_q, 50, axis = 1)*1e6
priorq_25 = np.nanpercentile(prior_q, 25, axis = 1)*1e6
priorq_75 = np.nanpercentile(prior_q, 75, axis = 1)*1e6

ax.plot(priorq_median, pressure/100, color = 'green', label='A Priori')
ax.fill_betweenx(pressure/100, priorq_25, priorq_75, color='green', alpha=0.2)

ax.plot(usrpq_median, pressure/100, color = 'blue', label='USRP')
ax.fill_betweenx(pressure/100, usrpq_25, usrpq_75, color='blue', alpha=0.15)


ax.plot(ac240q_median, pressure/100, color = (1, 0.3, 0.35), label='AC-240')
ax.fill_betweenx(pressure/100, ac240q_25, ac240q_75, color=(1, 0.3, 0.35), alpha=0.15)

ax.plot(ac240q_scaled_median, pressure/100, color = (1, 0.001, 0.8), label='AC-240 Scaled')
ax.fill_betweenx(pressure/100, ac240q_scaled_25, ac240q_scaled_75, color=(1, 0.001, 0.8), alpha=0.15)


plt.legend(fontsize = 20)
#plt.savefig(os.path.join(fig_file, 'mean_prof_comp_rescaled.png'), format = 'png', dpi = 400, bbox_inches = 'tight')


#%% both anomaly plots together
fig = plt.figure(figsize=(28, 12))
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 8])

#plot1
ax1 = plt.subplot(gs[0])
ax1.set_ylim(100, .0010)
ax1.set_yscale('log')
ax1.set_xlabel('Time', fontsize=20)
ax1.set_ylabel('Pressure (hPa)', fontsize=20)
ax1.tick_params(axis='x', rotation=45)
ax1.grid()


min, max, diff = -2,2, 0.01

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = q_diff[:,:-1] * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure = miawara_usrp_xr.pressure.values
datetimes = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in ac240_time]

contour = ax1.contourf(datetimes[:-1],pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label(r'$\delta$ q (PPMV)', fontsize = 20)
ticks_loc = np.arange(min, max+.01, (max-min)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(datetimes)):
    gap = int((datetimes[i] - datetimes[i-1]) / dt.timedelta(days = 1))
    if gap > 10:
        print(datetimes[i])
        ax1.axvspan(datetimes[i-1], datetimes[i], facecolor='grey', alpha=1)

# Plot 2
ax2 = plt.subplot(gs[1])
ax2.set_ylim(100, .0010)
ax2.set_yscale('log')
ax2.set_xlabel('q (PPMV)', fontsize=20)
ax2.set_ylabel('Pressure (hPa)', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='minor', labelsize=12)

ax2.plot(priorq_median, pressure/100, color = 'green', label='A Priori')
ax2.fill_betweenx(pressure/100, priorq_25, priorq_75, color='green', alpha=0.2)

ax2.plot(ac240q_median, pressure/100, color = 'red', label='AC-240')
ax2.fill_betweenx(pressure/100, ac240q_25, ac240q_75, color='red', alpha=0.15)

ax2.plot(usrpq_median, pressure/100, color = 'blue', label='USRP')
ax2.fill_betweenx(pressure/100, usrpq_25, usrpq_75, color='blue', alpha=0.15)
ax2.legend(fontsize=20, loc='best')
ax2.grid()


plt.tight_layout()
#plt.savefig(os.path.join(fig_file, 'time_series_w_med_prof_2_spectrometers.png'), format = 'png', dpi = 500, bbox_inches = 'tight')

#%% both anomaly plots together
fig = plt.figure(figsize=(28, 12))
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 8])

#plot1
ax1 = plt.subplot(gs[0])
ax1.set_ylim(100, .0010)
ax1.set_yscale('log')
ax1.set_xlabel('Time', fontsize=20)
ax1.set_ylabel('Pressure (hPa)', fontsize=20)
ax1.tick_params(axis='x', rotation=45)
ax1.grid()


min, max, diff = -2,2, 0.01

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = q_scaled_diff[:,:-1] * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
pressure = miawara_usrp_xr.pressure.values
datetimes = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in ac240_time]

contour = ax1.contourf(datetimes[:-1],pressure/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label(r'$\delta$ q (PPMV)', fontsize = 20)
ticks_loc = np.arange(min, max+.01, (max-min)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(datetimes)):
    gap = int((datetimes[i] - datetimes[i-1]) / dt.timedelta(days = 1))
    if gap > 10:
        print(datetimes[i])
        ax1.axvspan(datetimes[i-1], datetimes[i], facecolor='grey', alpha=1)

# Plot 2
ax2 = plt.subplot(gs[1])
ax2.set_ylim(100, .0010)
ax2.set_yscale('log')
ax2.set_xlabel('q (PPMV)', fontsize=20)
ax2.set_ylabel('Pressure (hPa)', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='minor', labelsize=12)

ax2.plot(priorq_median, pressure/100, color = 'green', label='A Priori')
ax2.fill_betweenx(pressure/100, priorq_25, priorq_75, color='green', alpha=0.2)

ax2.plot(ac240q_scaled_median, pressure/100, color = 'red', label='AC-240 Scaled')
ax2.fill_betweenx(pressure/100, ac240q_scaled_25, ac240q_scaled_75, color='red', alpha=0.15)

ax2.plot(usrpq_median, pressure/100, color = 'blue', label='USRP')
ax2.fill_betweenx(pressure/100, usrpq_25, usrpq_75, color='blue', alpha=0.15)
ax2.legend(fontsize=20, loc='best')
ax2.grid()

plt.tight_layout()
#plt.savefig(os.path.join(fig_file, 'time_series_w_med_prof_2_spectrometers_scaled.png'), format = 'png', dpi = 500, bbox_inches = 'tight')



#%%%histogram plots 
lev_0_01 = 46
lev_0_1 = 41
lev_1 = 36
lev_4 = 31


for i in [lev_0_01, lev_0_1, lev_1,lev_4]:
    oneD_nonScaled = q_diff[i,:]*1e6
    oneD_Scaled =q_scaled_diff[i,:]*1e6
    
    mean_ab, std_ab = np.median(oneD_nonScaled), np.std(oneD_nonScaled)
    mean_ac, std_ac = np.median(oneD_Scaled), np.std(oneD_Scaled)
    
    text_str = (f'Non Scaled:\nMedian = {mean_ab:.2f}\nStd = {std_ab:.2f}\n\n'
            f'Scaled\nMedian = {mean_ac:.2f}\nStd = {std_ac:.2f}')
    
    plt.hist(oneD_nonScaled, bins=np.arange(-1, 1, .025), label='Non-scaled', alpha=0.5)
    plt.hist(oneD_Scaled , bins=np.arange(-1, 1, .025), label='Scaled', alpha=0.5)
    #plt.legend(loc='upper right')
    plt.xlabel(r'$\delta$ q (PPMV)')
    plt.ylabel('Frequency')
    plt.title('Pressure Level {:.2g}hPa'.format(pressure[i]/100))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc =  'upper right')
    # Add text box with mean and std
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=props)
    plt.savefig(os.path.join(fig_file,'hist_usrp_ac240_ret_diff{:.2g}.pdf'.format(pressure[i]/100)), 
                bbox_inches = 'tight', dpi = 300, format = 'pdf')
    plt.close()






