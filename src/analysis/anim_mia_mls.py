#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:36:36 2023
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
sys.path.append('../../')
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
#%%load dataset
MLS_xr = xr.load_dataset(MLS_fullpath)
miawara_xr = xr.load_dataset(miawara_fullpath, decode_times = False)
miawara_xr = miawara_xr.sortby('time')
miawara_xr = miawara_xr.drop('A')

datetimes_mia = [dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in miawara_xr.time.values]
dt_64 = [np.datetime64(i).astype('datetime64[D]') for i in datetimes_mia]
miawara_xr['time'] = dt_64
#%%
MLS_running = MLS_xr.rolling(time=20, center=True).mean()
miawara_running = miawara_xr.rolling(time=20, center=True).mean()
pressure  = miawara_xr.pressure.values

q  = miawara_running.q.values
q_a = miawara_running.q_a.values

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

start_date = np.datetime64('2022-01-01')
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
    if (len(sub_mia.time)>0) &  (len(sub_mls.time)>0)  & (sub_mia.cost< 0.1):
        
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

#%%additional processing

# Create a window for the running average
window_size = 5
window = np.ones(int(window_size)) / float(window_size)

# Prepare empty arrays for the running averages with the correct shapes
# The length along the convolving axis will be reduced by window_size - 1
running_avg_MLS_conv = np.empty((MLS_conv.shape[0], MLS_conv.shape[1] - window_size + 1))
running_avg_MIA_ref = np.empty((MIA_ref.shape[0], MIA_ref.shape[1] - window_size + 1))
running_avg_MIA_clim = np.empty((MIA_clim.shape[0], MIA_clim.shape[1] - window_size + 1))

# Convolve along the second axis (x-axis) for each row (along the first axis)
for i in range(MLS_conv.shape[0]):
    running_avg_MLS_conv[i, :] = np.convolve(MLS_conv[i, :], window, 'valid')
    running_avg_MIA_ref[i, :] = np.convolve(MIA_ref[i, :], window, 'valid')
    running_avg_MIA_clim[i, :] = np.convolve(MIA_clim[i, :], window, 'valid')

#%% time to days
timeDays = np.array(time, dtype='datetime64[D]')

# Now you can subtract start_date from datetimes_mw_np and get the difference in days
day_mls_mia = timeDays - start_date

#%%set up plot
fig, ax = plt.subplots(figsize=(8, 12))
ax.set_xlim(0, 10)
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('H2O (PPMV)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)
plt.grid()


line_q_a, = ax.plot(MIA_clim[:,0]*1e6, pressure/100, label='Prior Zimmerwald')  # Initial line for q_a
line_q, = ax.plot( MIA_ref[:,0 ]*1e6,pressure/100, label='Measurement Zimmerwald')  # Initial line for q
line_q_MLS, = ax.plot( MLS_conv[:,0 ]*1e6,pressure/100, label='MLS')  # Initial line for q

ax.legend( fontsize = 20, loc = 1)

# Define the initialization function
def init():
    line_q.set_xdata([np.nan] * len(pressure))
    line_q_a.set_xdata([np.nan] * len(pressure))
    line_q_MLS.set_xdata([np.nan] * len(pressure))

    return line_q, line_q_a

# Define the update function for the animation
def animate(i):
    line_q_a.set_xdata(running_avg_MIA_clim[:, i]*1e6)  # Update the data for line_q_a
    line_q.set_xdata(running_avg_MIA_ref[:, i]*1e6)  # Update the data for line_q
    line_q_MLS.set_xdata(running_avg_MLS_conv[:, i]*1e6)  # Update the data for line_q_a
    ax.set_title("Jan 1 2022 + {} days".format(day_mls_mia[i]), fontsize=20)
    return line_q, line_q_a

#%%
# Create the animation
anim = FuncAnimation(fig, animate, frames=len(running_avg_MIA_clim[0,:])-6, init_func=init, blit=True)
savefile= os.path.join(fig_file, 'anim_hunga_tonga_miawara_downscale.gif')
anim.save(savefile, writer=PillowWriter(fps=5), dpi = 40)

