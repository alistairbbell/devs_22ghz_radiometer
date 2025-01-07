#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:14:54 2023

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
import os

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
tub_experiment_dir = '/storage/tub/instruments/miawara/l2/experiment2/2023/'

#%%
miawara_xr = xr.load_dataset(miawara_fullpath, decode_times = False)
pressure = miawara_xr.pressure.values/100
bl = np.loadtxt('/storage/tub/instruments/miawara/l2/experiment2/2023/baseline.txt')

# Define the base file name pattern and the range of file indices
file_pattern = os.path.join(tub_experiment_dir, 'retrieval_{}.nc') # Replace with your actual file pattern
file_indices = range(1, 20)  # Adjust the range according to your file numbering

# Generate a list of file names to open based on the pattern and indices
file_names = [file_pattern.format(i) for i in file_indices]


datasets = [xr.open_dataset(fn, decode_times = False) for fn in file_names]
temp = xr.open_dataset(file_names[0], decode_times = False)

listvars = list(datasets[0].variables)
#datasets = [fn.drop(listvars[:6])  for fn in datasets]
#datasets = [fn.drop(listvars[10:])  for fn in datasets]


#%%



#%%set up plot
fig, ax = plt.subplots(figsize=(8, 12))
ax.set_xlim(0, 10)
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('H2O (PPMV)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)
plt.grid()

line_q_a, = ax.plot(datasets[0]['q_a'].values*1e6, pressure, label='Prior')  # Initial line for q_a
line_q, = ax.plot(datasets[0]['q_a'].values*1e6,pressure, label='Current iteration')  # Initial line for q
ax.set_title("i = 0", fontsize=20)
ax.legend( fontsize = 20, loc = 1)

# Define the initialization function
def init():
    line_q.set_xdata([np.nan] * len(pressure))
    line_q_a.set_xdata([np.nan] * len(pressure))
    return line_q, line_q_a

# Define the update function for the animation
def animate(i):   
    line_q_a.set_xdata(datasets[0]['q_a'].values*1e6)  # Update the data for line_q_a
    line_q.set_xdata(datasets[i]['q'].values*1e6)  # Update the data for line_q
    ax.set_title("i = {}".format(i+1), fontsize=20)
    return line_q, line_q_a

#%%
# Create the animation
anim = FuncAnimation(fig, animate, frames=5, init_func=init, blit=True)
savefile= os.path.join(fig_file, 'iteration_steps_x.gif')
anim.save(savefile, writer=PillowWriter(fps=1), dpi = 60)

#%%
freq = np.linspace(22.23508-.5, 22.23508-.5, 16384)
freq_inds = np.array(datasets[0]['frequency'].values)/1e9

#%%set up plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(22.235-.05,22.235+.05)
ax.set_ylim(-.4, .4)
ax.set_xlabel('Frequency (GHz)', fontsize = 20)
ax.set_ylabel('Balanced BT (K)', fontsize = 20)
plt.grid()
my_bl = (bl[::-1]-np.mean(bl))*3


line_q_a, = ax.plot( freq_inds, np.array(datasets[0]['y'].values[:,0])+my_bl,label='Observation', linewidth = 3, color = 'purple')  # Initial line for q_a
line_q, = ax.plot(freq_inds, datasets[0]['yf'].values[:,0], label='Simulated', linewidth = 4, color = 'orange')  # Initial line for q
#ax.plot(freq_inds, my_bl, label='baseline')  # Initial line for q
ax.set_title("i = 0", fontsize=20)
ax.legend( fontsize = 20, loc = 1)
plt.show()

# Define the initialization function
def init():
    line_q.set_ydata([np.nan] * len(freq_inds))
    line_q_a.set_ydata([np.nan] * len(freq_inds))
    return line_q, line_q_a

# Define the update function for the animation
def animate(i):   
    my_bl = (bl[::-1]-np.mean(bl))*3
    line_q_a.set_ydata((datasets[0]['y'].values.T + my_bl)[0,:])  # Update the data for line_q_a
    line_q.set_ydata(datasets[i]['yf'] ) # Update the data for line_q
    ax.set_title("i = {}".format(i+1), fontsize=20)
    return line_q, line_q_a


#%%# Create the animation
anim = FuncAnimation(fig, animate, frames=5, init_func=init, blit=True)
savefile= os.path.join(fig_file, 'iteration_steps_y.gif')
anim.save(savefile, writer=PillowWriter(fps=1), dpi = 60)

