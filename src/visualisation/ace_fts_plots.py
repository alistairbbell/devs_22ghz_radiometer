#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:04:53 2024

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
filename = 'ACE_FTS_zimmerwald.nc'

ace_fullpath = os.path.join(interim_dir, filename)

#%%

ace_xr = xr.load_dataset(ace_fullpath, decode_times = False)

#%%
datetimes_mw = for i in miawara_xr.time.values]
dt_64 = [np.datetime64(i) for i in datetimes_mw]
miawara_xr['time'] =  dt_64
miawara_xr = miawara_xr.sortby('time')
















