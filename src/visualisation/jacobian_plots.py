#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:45:35 2024

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
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcb
import matplotlib.cm as cm
import matplotlib.dates as mdates
#from utils import temps
import uuid
from smb.SMBConnection import SMBConnection
import logging
import traceback
from logging.handlers import RotatingFileHandler


#%% paths
log_folder = "../../log"
data_folder = "../../data/"
s_dir = os.path.join(data_folder, "interim")
miawara_filename = 'MIAWARA_concat_H2O_2024_with_J.nc'

plot_outdir =  "../../output/figs"

#%% colour functions
def generate_gray_colors(n_shades):
    colormap = plt.colormaps.get_cmap('gray')
    gray_colors = [colormap(i) for i in np.linspace(0.2, .8, n_shades)]
    return gray_colors

def generate_blue_colors(n_shades):
    colormap = plt.colormaps.get_cmap('Blues')
    gray_colors = [colormap(i) for i in np.linspace(0.2, .8, n_shades)]
    return gray_colors


#%%
xArrayMia = xr.open_dataset(os.path.join(s_dir, miawara_filename), decode_times = False)

#water vapour
q = np.array(xArrayMia['q'])
q_a =  np.array(xArrayMia['q_a'])
q_err = np.array(xArrayMia['q_err'])

#pressure
p = np.array(xArrayMia['pressure'])

#Jacobians
J = np.array(xArrayMia['J'])

#Frequency
freq = np.array(xArrayMia['frequency'])

#time
time = xArrayMia['time']
time_datetime = [ dt.datetime(1970,1,1) + dt.timedelta(seconds = i) for i in np.array(time) ]

years_MIA = np.array([date.year for date in time_datetime]) #corresponding year
doy = np.array([date.timetuple().tm_yday for date in time_datetime]) #day of year


conv = np.array(xArrayMia['convergence_flag'])
#%% plotting at certain height






#plt.savefig(os.path.join(plot_outdir, 'miawara_anomaly_1hPa.png'), format = 'png', dpi = 500)

#%% plotting at certain height
grey_colors = generate_gray_colors(7)
grey_colors.append('red')
grey_colors.append('green')

plt.close()
plt.figure(figsize = (8,4))
for index, year in enumerate(range(2015,2024)):
    time_inds = np.where(years_MIA == year)[0]
    plot_data = (q[47,time_inds] - q_a[47,time_inds])*1e6
    plot_data = np.where(plot_data**2 >10, np.nan, plot_data)
    df = pd.DataFrame({'DOY': doy[time_inds], 'WV': plot_data[:]})
    df_running = df.rolling(window = 6, min_periods = 3, center = True, closed = 'both').mean()
    plt.plot(df_running['DOY'], df_running['WV'],  label = year, c = grey_colors[index], linewidth  =3 )
plt.legend()
plt.grid()
plt.xlabel('Day of Year', fontsize = 14)
plt.ylabel('Water Vapor Mixing Ratio (PPMV)', fontsize = 14)
plt.tick_params(axis = 'both', labelsize = 12)
plt.savefig(os.path.join(plot_outdir, 'miawara_anomaly_0_1hPa.png'), format = 'png', dpi = 500)

