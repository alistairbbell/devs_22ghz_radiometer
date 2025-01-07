#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:27:05 2023

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
data_folder = "../../data"
interim_dir = os.path.join(data_folder, "interim")
fig_file = "../../output/figs"
h2o_string = 'retrieval'
log_folder = "../../log"
MLS_Bern_Filename = 'MLS_concat_H2O.nc'
miawara_Filename_old = 'MIAWARA_concat_H2O_2010_2023_inc_A.nc'
miawara_Filename = 'MIAWARA_concat_H2O_2010_2023_rescaled_inc_A.nc'

MLS_fullpath = os.path.join(interim_dir, MLS_Bern_Filename)
miawara_old_fullpath = os.path.join(interim_dir, miawara_Filename_old)
miawara_fullpath = os.path.join(interim_dir, miawara_Filename)

#%% function definitions
def conv( x_t, x_a , A ):
    dx = x_t - x_a
    x_smooth = x_a + np.matmul(A, dx)
    return x_smooth

#%% load data
MLS_xr = xr.load_dataset(MLS_fullpath)
miawara_xr = xr.load_dataset(miawara_fullpath, decode_times = False)
miawara_old_xr = xr.load_dataset(miawara_old_fullpath, decode_times = False)

temp = miawara_xr.time.values
other  = miawara_xr.q.values

datetimes_mw = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in miawara_xr.time.values]
dt_64 = [np.datetime64(i) for i in datetimes_mw]
miawara_xr['time'] =  dt_64


datetimes_mw_old = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in miawara_old_xr.time.values]
dt_64_old = [np.datetime64(i) for i in datetimes_mw_old]
miawara_old_xr['time'] =  dt_64_old

#%%
#initialise datetime range 

start_date = np.datetime64('2015-01-01')
end_date = np.datetime64('2023-09-20')

sub_mw = miawara_xr.where(miawara_xr['time']> np.datetime64('2014-12-31'), drop = True)

time = np.empty(0, dtype = 'datetime64')
MLS_conv = np.zeros([80,0])
MIA_ref = np.zeros([80,0])
h2o_diff =  np.zeros([80,0])
MIA_clim = np.zeros([80,0])


#iterate by day
for date in np.arange(start_date, end_date):
    day_after = date + np.timedelta64(1, 'D')
    
    #extract sub xr dataset for valid date range
    sub_mia =  miawara_xr.where((miawara_xr['time'] > date) & (miawara_xr['time'] < day_after), drop = True)
    sub_mls = MLS_xr.where((MLS_xr['time'] > date) & (MLS_xr['time'] < day_after), drop = True)
    
    #if data exists for both instruments
    if (len(sub_mia.time)>0) &  (len(sub_mls.time)>0):
        
        #extract mls pressure and h2o values
        pres_mls = sub_mls.pressure.values
        h2o_mls = np.nanmean(sub_mls.value, axis = 0)
        
        #extract miawara pressure and h2o values
        pres_mia = sub_mia.pressure.values
        h2o_mia = sub_mia.q.values
        h2o_prior = sub_mia.q_a.values
        
        #create interpolation function to use mls at miawara resolution
        f = interpolate.interp1d(pres_mls, h2o_mls, kind='cubic', fill_value="extrapolate")
        h2o_mls_intp1 = f(pres_mia)
        
        #convolve interpolated mls values with miawara averaging kernel and a priori
        mls_conv = conv( h2o_mls_intp1, sub_mia.q_a.values[:,0] , sub_mia.A.values[:,:,0] )
        mls_conv = np.reshape(mls_conv, [80,1] )
        
        #calculate difference between retrieval methods
        diff =  mls_conv - h2o_mia
        
        #add values to initial structures
        time = np.append(time,date)
        MLS_conv = np.append(MLS_conv, mls_conv, axis =1 )
        MIA_ref = np.append(MIA_ref, h2o_mia, axis =1 )
        h2o_diff = np.append(h2o_diff, diff, axis =1 )
        MIA_clim = np.append(MIA_clim, h2o_prior, axis =1 )

#%%
#initialise datetime range 

#time = np.empty(0, dtype = 'datetime64')
#MLS_conv = np.zeros([80,0])
MIA_qold = np.zeros([80,0])

#iterate by day
for date in np.arange(start_date, end_date):
    day_after = date + np.timedelta64(1, 'D')
    
    #extract sub xr dataset for valid date range
    sub_mia =  miawara_old_xr.where((miawara_xr['time'] > date) & (miawara_xr['time'] < day_after), drop = True)
    sub_mls = MLS_xr.where((MLS_xr['time'] > date) & (MLS_xr['time'] < day_after), drop = True)
    
    #if data exists for both instruments
    if (len(sub_mia.time)>0) &  (len(sub_mls.time)>0):
        
        #extract mls pressure and h2o values
       
        #extract miawara pressure and h2o values
        pres_mia = sub_mia.pressure.values
        h2o_mia = sub_mia.q.values
        h2o_prior = sub_mia.q_a.values
        
        #add values to initial structures
        time = np.append(time,date)
        MIA_qold = np.append(MIA_qold, h2o_mia, axis =1 )





#%%
fig, ax = plt.subplots(figsize=(12, 8))
#ax.set_xlim(-80, 80)
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

min, max, diff = -2, 2, 0.01

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = h2o_diff * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data)
contour = ax.contourf(time,pres_mia/100,plotting_data, levels=np.arange(min, max+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label(r'$\Delta$ q (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min, max+.01, (max-min)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(time)):
    gap = (time[i] - time[i-1]).astype(int)
    if gap > 2:
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)
        
#plt.savefig(os.path.join(fig_file, 'MLS_MIA_scaled_diff_conv.png'), format = 'png', dpi = 400)

#%% Plotting of miawara averaging kernel

a_kern = miawara_xr.A.values
pressure = miawara_xr.pressure.values

a_kern_av = np.nanmean(a_kern, axis = 2)
#a_kern_av = a_kern[:,:,:100]

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('Averaging Kernel', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

plt.grid(which = 'major', alpha = 0.5, color = 'black')
plt.grid(which = 'minor', alpha = 0.2)

colors = ["darkblue", "blue", "green", "darkgoldenrod", "orange", "red"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=50)

for i in range(80):
    ax.plot( a_kern_av[:,i],pressure/100, alpha=0.3, color = 'black')

for i in np.arange(4,60, 5):
    ax.plot(a_kern_av[:,i],pressure/100, alpha=1, label = '{:.2g}hPa'.format(pressure[i]/100), color=cmap(i/55))
plt.legend()


#plt.savefig(os.path.join(fig_file, 'averaging_kernel_multicolor.pdf'), format = 'pdf')

#%%Profile plot
fig, ax = plt.subplots(figsize=(8, 14))
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlim(1, 10)
ax.set_xlabel('q (PPMV)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plt.grid(which = 'major', alpha = 0.9)
ax.grid(which='minor', axis = 'both', alpha=0.2)

mia_orig_median = np.nanpercentile(MIA_qold, 50, axis = 1)*1e6
mia_orig_25 = np.nanpercentile(MIA_qold, 25, axis = 1)*1e6
mia_orig_75 = np.nanpercentile(MIA_qold, 75, axis = 1)*1e6

MLS_median = np.nanpercentile(MLS_conv, 50, axis = 1)*1e6
MLS_25 = np.nanpercentile(MLS_conv, 25, axis = 1)*1e6
MLS_75 = np.nanpercentile(MLS_conv, 75, axis = 1)*1e6

MIA_clim_med = np.nanpercentile(MIA_clim, 50, axis = 1)*1e6
MIA_clim_25 = np.nanpercentile(MIA_clim, 25, axis = 1)*1e6
MIA_clim_75 = np.nanpercentile(MIA_clim, 75, axis = 1)*1e6

Mia_scaled_median = np.nanpercentile(MIA_ref, 50, axis = 1)*1e6
Mia_scaled_25 = np.nanpercentile(MIA_ref, 25, axis = 1)*1e6
Mia_scaled_75 = np.nanpercentile(MIA_ref, 75, axis = 1)*1e6

ax.plot(MIA_clim_med, pres_mia/100, color = 'green', label='Climatology')
ax.fill_betweenx(pres_mia/100, MIA_clim_25, MIA_clim_75, color='green', alpha=0.2)

ax.plot(MLS_median, pres_mia/100, color = 'blue', label='MLS')
ax.fill_betweenx(pres_mia/100, MLS_25, MLS_75, color='blue', alpha=0.15)

ax.plot(mia_orig_median, pres_mia/100, color = (1, 0.3, 0.35), label='MIAWARA Unscaled')
ax.fill_betweenx(pres_mia/100, mia_orig_25, mia_orig_75, color=(1, 0.3, 0.35), alpha=0.15)

ax.plot(Mia_scaled_median, pres_mia/100, color = (1, 0.001, 0.8), label='MIAWARA Scaled')
ax.fill_betweenx(pres_mia/100, Mia_scaled_25, Mia_scaled_75, color=(1, 0.001, 0.8), alpha=0.15)

plt.legend(fontsize = 20)

#plt.savefig(os.path.join(fig_file, 'mean_prof_mls_MIA_w_scaled.png'), format = 'png', dpi = 400, bbox_inches = 'tight')

#%%Profile plot
fig, ax = plt.subplots(figsize=(8, 14))
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlim(-1, 1)
ax.set_xlabel('$\delta$q (PPMV)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plt.grid(which = 'major', alpha = 0.9)
ax.grid(which='minor', axis = 'both', alpha=0.2)

bias_prior = np.nanmean(MLS_conv  -  MIA_clim, axis = 1)*1e6
bias_orig = np.nanmean( MLS_conv - MIA_qold  ,  axis = 1)*1e6
bias_new = np.nanmean( MLS_conv -MIA_ref ,  axis = 1)*1e6

rmse_prior = np.sqrt(np.nanmean((( MLS_conv - MIA_clim)*1e6)**2, axis = 1))
rmse_orig = np.sqrt(np.nanmean((( MLS_conv - MIA_qold)*1e6)**2 , axis = 1))
rmse_new = np.sqrt(np.nanmean((( MLS_conv - MIA_ref)*1e6)**2 , axis = 1))

rmse_prior = np.nanstd( MLS_conv - MIA_clim , axis = 1)*1e6
rmse_orig = np.nanstd( MLS_conv - MIA_qold ,  axis = 1)*1e6
rmse_new = np.nanstd( MLS_conv - MIA_ref ,  axis = 1)*1e6

ax.plot(np.zeros(50), np.logspace(-2,3), color = 'black')
ax.plot(bias_orig, pres_mia/100, color = 'red', label='Unscaled Bias')
ax.plot(rmse_orig, pres_mia/100, color = 'red', label='Unscaled STD', linestyle = 'dashed')

ax.plot(bias_new, pres_mia/100, color = 'purple', label='Scaled Bias')
ax.plot(rmse_new, pres_mia/100, color = 'purple', label='Scaled STD', linestyle = 'dashed')

plt.legend(fontsize = 20, loc  = 'upper left')

<<<<<<< Updated upstream
#plt.savefig(os.path.join(fig_file, 'mls_MIA_scaled_rmse_bias.png'), format = 'png', dpi = 400, bbox_inches = 'tight')
=======
plt.savefig(os.path.join(fig_file, 'mls_MIA_scaled_rmse_bias.png'), format = 'png', dpi = 400, bbox_inches = 'tight')
>>>>>>> Stashed changes


