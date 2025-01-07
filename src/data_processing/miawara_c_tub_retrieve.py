#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:08:03 2024

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
import configparser

#%%
absolute_script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(absolute_script_path)
<<<<<<< Updated upstream
#absolute path of the current script
=======
# This is the absolute path of the current script
>>>>>>> Stashed changes
os.chdir(script_directory)
#%%
credentials_file_path = '/home/alistair/.smbcredentials'
#%%
server_name = 'tank'  # Simplified name for readability
server_ip = 'tank.mw.iap.unibe.ch'  # Extracted from the URL
client_name = 'user'  # Assuming this remains the same
domain = 'campus'
share_name = 'atmosphere'  # Extracted from the URL
server_basepath = '/instruments/miawarac/l2/'  # Assuming root share access, adjust as needed

data_folder = "../../data"
download_folder = os.path.join(data_folder, "tmp")
temp_file_path = os.path.join(download_folder, "temp_file.nc")
save_output_dir = os.path.join(data_folder, "interim")
<<<<<<< Updated upstream
#output_file_name = 'MIAWARA_C_concat_H2O_2018_2024_rescaled_v4.nc'
output_file_name = 'MIAWARA_C_concat_H2O_2018_2024_rescaled_ndacc_v2.nc'

=======
output_file_name = 'MIAWARA_C_concat_H2O_2018_2024_rescaled.nc'
>>>>>>> Stashed changes
outFullPath = os.path.join(save_output_dir, output_file_name)
h2o_string = 'retrieval'
log_folder = "../../log"

#%% Read credentials
credentials = {}
try:
    with open(credentials_file_path, 'r') as creds_file:
        for line in creds_file:
            key, value = line.strip().split('=')
            credentials[key] = value

    # Now you can access your credentials as needed
    username = credentials.get('user')
    password = credentials.get('password')  # Add this line to your .smbcredentials file with your actual password
    domain = credentials.get('domain')  # Add this if your file includes a domain

except FileNotFoundError:
    print(f"Credentials file not found at {credentials_file_path}")
    # Handle the error appropriately - exit or raise exception
except ValueError:
    print("Error processing credentials file. Each line must be in the format key=value.")
    # Handle the error appropriately - exit or raise exception

#%% set up error logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler(os.path.join(log_folder,"miawara_data_retrieve.txt"), maxBytes=1*1024*1024, backupCount=1)
logger.addHandler(handler)

#%%#%% Retrieve the MIAWARA data from the tub server
<<<<<<< Updated upstream
years = np.arange(2015,2025)
=======
years = np.arange(2018,2025)
>>>>>>> Stashed changes
years_s = [str(i) for i in years]

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, domain = domain,use_ntlm_v2=True)
conn.connect(server_ip)

# List all files in the root of the share
file_paths_to_merge = []

#variables to drop from the xarray dataset (dataset too big if we keep them)
<<<<<<< Updated upstream
vars_to_drop = ['lon', 'lat','y', 'yf', 'tau', 'J', 'frequency', 'A', 'q_err_smooth', 'z']
=======
vars_to_drop = ['y', 'yf', 'tau', 'J', 'frequency', 'A', 'q_err_smooth']
>>>>>>> Stashed changes

#initialise file iteration number
iterno = 0

for year in years_s:
    svr_data_path = os.path.join(server_basepath, year) 
    files = conn.listPath(share_name, svr_data_path)
    
    for file in files:
        print(file.filename)
        if h2o_string in file.filename and file.filename.endswith("00.nc"):
            print(os.path.join(svr_data_path, file.filename))  # for debug
            
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"{temp_file_path} has been deleted!")
                
            with open(temp_file_path, 'wb') as local_file:
                conn.retrieveFile(share_name, os.path.join(svr_data_path, file.filename), local_file)
    
            # Load the file into xarray
            ds = xr.open_dataset(temp_file_path, decode_times=False)
            ds_red = ds.drop_vars(vars_to_drop, errors='ignore')  # Drop variables in one go, safely ignore errors for non-existent variables

            # Load the reduced dataset into memory to avoid indexing issues
            ds_red = ds_red.load()  # This line is crucial to avoid the IndexError

            print(f"Current ds_red time shape: {ds_red.dims.get('time', 'No time dim')}")
            if ('time' in ds_red.dims):
                print("Attempting concatenation")
                if iterno == 0:  # if first file, initialise xarray
                    concatenated_ds = ds_red
                    iterno += 1
                else:  # else, try to append to existing xarray
                    concatenated_ds = xr.concat([concatenated_ds, ds_red], dim='time')
                    iterno += 1
            ds.close()  # Close temporary datasets

concatenated_ds.to_netcdf(outFullPath)
conn.close()

#%%#%% Handling of Averaging Kernel the MIAWARA data from the tub server
years = np.arange(2022,2025)
years_s = [str(i) for i in years]

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True)
conn.connect(server_ip)

# List all files in the root of the share
file_paths_to_merge = []

#variables to drop from the xarray dataset (dataset too big if we keep them)
vars_to_drop = ['y', 'yf', 'tau', 'J']

A_all = np.zeros([85,85,0])
time_all = np.zeros([0])
#initialise file iteration number
iterno = 0

for year in years_s:
    svr_data_path = os.path.join(server_basepath, year) 
    files = conn.listPath(share_name, svr_data_path)
    
    for file in files:
        if h2o_string in file.filename and file.filename.endswith(".nc"):
            print(os.path.join(svr_data_path, file.filename)) #for debug
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"{temp_file_path} has been deleted!")
            with open(temp_file_path, 'wb') as local_file:
                conn.retrieveFile(share_name, os.path.join(svr_data_path, file.filename), local_file)
    
            # Load the file into xarray
            ds = xr.open_dataset(temp_file_path, decode_times=False)
            A = np.expand_dims( np.array(ds.A), axis=-1)
            A_all = np.concatenate((A_all, A), axis=2)
            
            time_all = np.append(time_all, ds.time.values)
conn.close()
#%%
#concatenated_ds = concatenated_ds.assign_coords(new_pressure=ds['pressure'])
#concatenated_ds['A'] = (('pressure', 'new_pressure', 'time'), A_all)
concatenated_ds = concatenated_ds.sortby('time')
concatenated_ds.to_netcdf(outFullPath)

#%%
if __name__ == "__main__":
    a_kern = concatenated_ds.A.values
    pressure = concatenated_ds.pressure.values
    
    a_kern_av = np.nanmean(A_all, axis = 2)
    #a_kern_av = a_kern[:,:,100]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_ylim(1e4, .10)
    ax.set_yscale('log')
    ax.set_xlabel('Latitude (degrees)', fontsize = 20)
    ax.set_ylabel('Pressure (hPa)', fontsize = 20)
    
    for i in range(85):
        ax.plot( a_kern_av[:,i],pressure/100, alpha=0.3)
    
    for i in np.arange(0,85, 5):
        ax.plot(a_kern_av[:,i],pressure/100, alpha=1)
        
    plt.grid()



