#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:52:21 2023
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
<<<<<<< Updated upstream
sys.path.append('../')

=======
>>>>>>> Stashed changes
import matplotlib.dates as mdates
from utils import temps
import uuid
from smb.SMBConnection import SMBConnection
import logging
import traceback
from netCDF4 import Dataset
from logging.handlers import RotatingFileHandler


#%% path definitions

log_folder = "../../log"
data_folder = "../../data"
plot_folder = "../../output/figs"

output_file_name = 'MIAWARA_concat_H2O_2010_2023.nc'
interim_dir = os.path.join(data_folder, "interim")
input_fullfile = os.path.join(interim_dir, output_file_name)

#%% function to handle missing values in list
def get_values(data, indices):
    empty = np.zeros(indices.shape)
    for i in np.arange(len(indices)):
        if  ~np.isnan(indices[i]):
            empty[i] = data[int(indices[i])]
        else:
            empty[i] = np.nan
    return empty

#%% open data
my_xr = xr.open_dataset(input_fullfile, decode_times=False)
my_xr['time'] = pd.to_datetime(my_xr['time'], unit='s')

my_xr['pressure'] = my_xr['pressure'] / 100
my_xr['pressure'].attrs['units'] = 'hPa'
#%%
measurement_response_0_8 = np.where(my_xr['measurement_response']>0.8, 1, np.nan)
grid = np.tile(np.arange(measurement_response_0_8.shape[0]), (measurement_response_0_8.shape[1], 1)).T
index_grid = measurement_response_0_8*grid
max_ind = np.nanmax(index_grid, axis = 0)
min_ind = np.nanmin(index_grid, axis = 0)
#%%
pressure_max = get_values(np.array(my_xr['pressure']), max_ind)
pressure_min = get_values(np.array(my_xr['pressure']), min_ind)

#%%
upper_bound_nan =np.array(  my_xr['measurement_response'].where(my_xr['measurement_response'] > 0.8))
# non_nan_indices_upper = np.argmax((~np.isnan(upper_bound_nan)).astype(int) * np.arange(1, upper_bound_nan.shape[0] + 1)[:, np.newaxis], axis=0)
# non_nan_indices_upper[(~np.isnan(upper_bound_nan)).sum(axis=0) == 0] = np.nan

#%%
non_nan_indices_lower = np.argmin((~np.isnan(upper_bound_nan)).astype(int) * np.arange(1, upper_bound_nan.shape[0] + 1)[:, np.newaxis], axis=0)                        
lower_bounds_mr = my_xr['measurement_response'].where(my_xr['measurement_response'] > 0.8).idxmax(dim='pressure')


#%% analysis
mean_mr = my_xr['measurement_response'].mean(dim='time')
#%% plotting
plt.plot(mean_mr, my_xr['pressure'])
plt.yscale('log')
plt.ylim([1e1,1e-2])
plt.ylabel('Pressure (hPa)')
plt.xlabel('Measurement Response')
plt.grid()
<<<<<<< Updated upstream
#plt.savefig(os.path.join(plot_folder, 'average_measurement_response.pdf'), format = 'pdf')
=======
plt.savefig(os.path.join(plot_folder, 'average_measurement_response.pdf'), format = 'pdf')
>>>>>>> Stashed changes
#%% plot water vapour time series

plot = my_xr.q.plot.contourf(x = 'time', y = 'pressure', levels = 20, vmin = 1e-6, vmax = 12e-6)
# Adjust y-axis labels to be in hPa instead of Pa
y_ticks = plot.axes.get_yticks()
plot.axes.set_ylim([10,0.01])
plt.gca().set_yscale('log')
plt.plot(my_xr.time, pressure_max, color='white', linestyle='--', label='Lower Bound')
plt.plot(my_xr.time, pressure_min, color='white', linestyle='--', label='Lower Bound')
plt.show()

#%%

