#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:54:30 2024

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
import h5netcdf as h5
import h5py
#%%
absolute_script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(absolute_script_path)
#absolute path of the current script
os.chdir(script_directory)
#%%
credentials_file_path = '/home/alistair/.smbcredentials'
#%%
server_name = 'tank'  # Simplified name for readability
server_ip = 'tank.mw.iap.unibe.ch'  # Extracted from the URL
client_name = 'user'  # Assuming this remains the same
domain = 'campus'
share_name = 'atmosphere'  # Extracted from the URL
server_basepath = '/instruments/tempera/level2/'  # Assuming root share access, adjust as needed

data_folder = "../../data"
download_folder = os.path.join(data_folder, "tmp")
temp_file_path = os.path.join(download_folder, "temp_file.nc")
save_output_dir = os.path.join(data_folder, "interim")
output_file_name = 'TEMPERA_concat_T_2021_2024.nc'
outFullPath = os.path.join(save_output_dir, output_file_name)
h2o_string = 'TEMPERA_level2'
log_folder = "../../log"

fig_outdir = '../../output/figs/'

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

#%%
# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, domain = domain,use_ntlm_v2=True)
conn.connect(server_ip)

#variables to drop from the xarray dataset (dataset too big if we keep them)
vars_to_drop = ['tmp_avk', 'G', 'alt']



files = conn.listPath(share_name, server_basepath)
sorted_files = sorted(files, key=lambda file: file.filename)[:]
file = sorted_files[200]

print(os.path.join(server_basepath, file.filename))  # for debug

if os.path.exists(temp_file_path):
    os.remove(temp_file_path)
    print(f"{temp_file_path} has been deleted!")
    
with open(temp_file_path, 'wb') as local_file:
    conn.retrieveFile(share_name, os.path.join(server_basepath, file.filename), local_file)

# Load the file into xarray
ds =  xr.open_dataset(temp_file_path, engine='netcdf4',  
    decode_times=True, backend_kwargs={'group': 'temperature'})

pressure = ds['p']
p_vals = pressure.values[0,:]
#%%



#%%#%% Retrieve the MIAWARA data from the tub server
years = np.arange(2021,2025)
years_s = [str(i) for i in years]

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, domain = domain,use_ntlm_v2=True)
conn.connect(server_ip)

# List all files in the root of the share
file_paths_to_merge = []

#variables to drop from the xarray dataset (dataset too big if we keep them)
vars_to_drop = ['tmp_avk', 'G', 'alt']

#initialise file iteration number
iterno = 0

files = conn.listPath(share_name, server_basepath)
sorted_files = sorted(files, key=lambda file: file.filename)[500:]
for file in sorted_files:
    if iterno<600:
        print(file.filename)
        if ('2023' in file.filename or '2024' in file.filename) and \
        h2o_string in file.filename and file.filename.endswith(".h5"):
            print(os.path.join(server_basepath, file.filename))  # for debug
            
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"{temp_file_path} has been deleted!")
                
            with open(temp_file_path, 'wb') as local_file:
                conn.retrieveFile(share_name, os.path.join(server_basepath, file.filename), local_file)
    
            # Load the file into xarray
            ds =  xr.open_dataset(temp_file_path, engine='netcdf4',  
                decode_times=True, backend_kwargs={'group': 'temperature'})
            # Drop variables in one go, safely ignore errors for non-existent variables
            ds_red = ds.drop_vars(vars_to_drop, errors='ignore') 
    
            # Load the reduced dataset into memory to avoid indexing issues
            ds_red = ds_red.load()  # This line is crucial to avoid the IndexError
            
            date_str = file.filename[:-3].split('_')[2:5]  # Assuming format: TEMPERA_level2_yyyy_mm_dd.h5
            date_formatted = '-'.join(date_str)  # Format as 2024-06-14
            times = pd.date_range(start=date_formatted, periods=48, freq='30T')  # 48 periods for 0 to 47 (inclusive) at 30 min intervals
            ds_red = ds_red.assign_coords(time=times)  # Replace time coordinate in dataset
    
            print(f"Current ds_red time shape: {ds_red.dims.get('time', 'No time dim')}")
            if ('time' in ds_red.dims):
                print("Attempting concatenation")
                if iterno == 0:  #if first file, initialise xarray
                    concatenated_ds = ds_red
                    iterno += 1
                else:  # else, try to append to existing xarray
                    concatenated_ds = xr.concat([concatenated_ds, ds_red], dim='time')
                    iterno += 1
            ds.close()  # Close temporary datasets

concatenated_ds = concatenated_ds.sortby('time')
concatenated_ds.to_netcdf(outFullPath)
conn.close()

new_da = concatenated_ds.assign_coords(pressure=("altitude", np.array(p_vals)))
new_da = new_da.swap_dims({"altitude": "pressure"})
new_da = new_da.drop_vars("altitude")

#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(1e3, 1e-1)
ax.set_xlim(dt.datetime(2023,1,1), dt.datetime(2024,3,1))
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (Pa)', fontsize = 20)

min_val, max_val, diff = -20, 20, .1

# Initial contourf plot
cmap = plt.get_cmap('coolwarm').copy()  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('yellow')    # Color for values > vmax
cmap.set_bad(color = 'grey') 
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

ax.set_title('Temperature Difference', fontsize = 20)

plotting_data = new_da.tmp.values - new_da.apr_tmp.values
plotting_data = np.where(np.isnan(plotting_data), np.nan, np.clip(plotting_data, min_val, max_val))

plotting_data = np.where(plotting_data < min_val, min_val, plotting_data)
plotting_data = np.where(plotting_data > max_val, max_val, plotting_data)
plotting_data = np.ma.masked_invalid(plotting_data)  # Mask NaN values for handling in contourf
p = new_da['pressure'].values
time = new_da['time'].values

ax.set_facecolor("gray")
contour = ax.contourf(time,p/1e2,plotting_data.T, levels=np.arange(min_val, max_val+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Temperature (K)', fontsize = 20)
ticks_loc = np.arange(min_val, max_val+.01, (max_val-min_val)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(time)):
    gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
    if gap > 2:
        print(time[i])
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)
        
#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(1e3, 1e-1)
ax.set_xlim(dt.datetime(2023,5,10), dt.datetime(2023,5,24))
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (Pa)', fontsize = 20)

min_val, max_val, diff = 200, 280, .1

# Initial contourf plot
cmap = plt.get_cmap('viridis').copy()  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('yellow')    # Color for values > vmax
cmap.set_bad(color = 'grey') 
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

ax.set_title('A Priori Temperature', fontsize = 20)

plotting_data = new_da.apr_tmp.values
plotting_data = np.where(np.isnan(plotting_data), np.nan, np.clip(plotting_data, min_val, max_val))

plotting_data = np.where(plotting_data < min_val, min_val, plotting_data)
plotting_data = np.where(plotting_data > max_val, max_val, plotting_data)
plotting_data = np.ma.masked_invalid(plotting_data)  # Mask NaN values for handling in contourf
p = new_da['pressure'].values
time = new_da['time'].values

ax.set_facecolor("gray")
contour = ax.contourf(time,p/1e2,plotting_data.T, levels=np.arange(min_val, max_val+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Temperature (K)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min_val, max_val+.01, (max_val-min_val)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(time)):
    gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
    if gap > 2:
        print(time[i])
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)
        
#plt.savefig(os.path.join(fig_file, 'MIA_time_series.png'), format = 'png', dpi = 400)
#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(1e3, 1e-1)
ax.set_xlim(dt.datetime(2023,5,10), dt.datetime(2023,5,24))
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize = 20)
ax.set_ylabel('Pressure (Pa)', fontsize = 20)

min_val, max_val, diff = 200, 280, .1

# Initial contourf plot
cmap = plt.get_cmap('viridis').copy()  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('yellow')    # Color for values > vmax
cmap.set_bad(color = 'grey') 
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

ax.set_title('Retrieved Temperature', fontsize = 20)
plotting_data = new_da.tmp.values
plotting_data = np.where(np.isnan(plotting_data), np.nan, np.clip(plotting_data, min_val, max_val))

plotting_data = np.where(plotting_data < min_val, min_val, plotting_data)
plotting_data = np.where(plotting_data > max_val, max_val, plotting_data)
plotting_data = np.ma.masked_invalid(plotting_data)  # Mask NaN values for handling in contourf
p = new_da['pressure'].values
time = new_da['time'].values

ax.set_facecolor("gray")
contour = ax.contourf(time,p/1e2,plotting_data.T, levels=np.arange(min_val, max_val+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Temperature (K)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
ticks_loc = np.arange(min_val, max_val+.01, (max_val-min_val)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(time)):
    gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
    if gap > 2:
        print(time[i])
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)

#%%
fig, ax = plt.subplots(figsize=(20, 8))
ax.set_ylim(4e2, 1e-1)
ax.set_xlim(dt.datetime(2023, 1, 1), dt.datetime(2024, 5, 24))
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize=20)
ax.set_ylabel('Pressure (Pa)', fontsize=20)

min_val, max_val, diff = 190, 290, .1

# Copy and modify the colormap
cmap = plt.get_cmap('viridis').copy()
cmap.set_under('blue')
cmap.set_over('yellow')
#cmap.set_bad('gray')  # Set NaNs to grey

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = new_da.tmp.values
#plotting_data = np.where(np.isnan(plotting_data), np.nan, np.clip(plotting_data, min_val, max_val))
plotting_data = np.ma.masked_invalid(plotting_data)  # Mask NaN values for handling in contourf
p = new_da.pressure.values
time = new_da['time'].values

ax.set_facecolor("gray")

contour = ax.contourf(time, p / 1e2, plotting_data.T, levels=np.arange(min_val, max_val + (diff / 2), diff), cmap=cmap)
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (K)', fontsize=20)
ticks_loc = np.arange(min_val, max_val + .01, (max_val - min_val) / 10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(time)):
    gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
    if gap > 2:
        print(time[i])
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)


plt.show()

#%%
# First plot: A Priori Temperature
fig, axs = plt.subplots(3, 1, figsize=(20, 24), sharex=True)

# Plot 1: A Priori Temperature
ax = axs[0]
ax.set_ylim(1e3, 1e-1)
ax.set_yscale('log')
ax.set_ylabel('Pressure (Pa)', fontsize=20)
min_val, max_val, diff = 200, 280, .1
cmap = plt.get_cmap('viridis').copy()
cmap.set_under('blue')
cmap.set_over('yellow')
cmap.set_bad(color='grey')
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)
ax.set_title('A Priori Temperature', fontsize=20)

plotting_data = new_da.apr_tmp.values
plotting_data = np.where(np.isnan(plotting_data), np.nan, np.clip(plotting_data, min_val, max_val))
plotting_data = np.where(plotting_data < min_val, min_val, plotting_data)
plotting_data = np.where(plotting_data > max_val, max_val, plotting_data)
plotting_data = np.ma.masked_invalid(plotting_data)
p = new_da['pressure'].values
time = new_da['time'].values
ax.set_facecolor("gray")
contour = ax.contourf(time, p/1e2, plotting_data.T, levels=np.arange(min_val, max_val+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (K)', fontsize=20)
ticks_loc = np.arange(min_val, max_val+.01, (max_val-min_val)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)
for i in range(1, len(time)):
    gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
    if gap > 2:
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)

# Second plot: Retrieved Temperature
ax = axs[1]
ax.set_ylim(1e3, 1e-1)
ax.set_yscale('log')
ax.set_ylabel('Pressure (Pa)', fontsize=20)
min_val, max_val, diff = 200, 280, .1
cmap = plt.get_cmap('viridis').copy()
cmap.set_under('blue')
cmap.set_over('yellow')
cmap.set_bad(color='grey')
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)
ax.set_title('Retrieved Temperature', fontsize=20)
plotting_data = new_da.tmp.values
plotting_data = np.where(np.isnan(plotting_data), np.nan, np.clip(plotting_data, min_val, max_val))
plotting_data = np.where(plotting_data < min_val, min_val, plotting_data)
plotting_data = np.where(plotting_data > max_val, max_val, plotting_data)
plotting_data = np.ma.masked_invalid(plotting_data)
p = new_da['pressure'].values
time = new_da['time'].values
ax.set_facecolor("gray")
contour = ax.contourf(time, p/1e2, plotting_data.T, levels=np.arange(min_val, max_val+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (K)', fontsize=20)
ticks_loc = np.arange(min_val, max_val+.01, (max_val-min_val)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)
for i in range(1, len(time)):
    gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
    if gap > 2:
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)

# Third plot: General Temperature
ax = axs[2]
ax.set_ylim(4e2, 1e-1)
ax.set_yscale('log')
ax.set_xlabel('Time', fontsize=20)
ax.set_ylabel('Pressure (Pa)', fontsize=20)
min_val, max_val, diff = -20, 20, .1

# Initial contourf plot
cmap = plt.get_cmap('coolwarm').copy()  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('yellow')    # Color for values > vmax
cmap.set_bad(color = 'grey') 
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

ax.set_title('Temperature Difference', fontsize = 20)

plotting_data = new_da.tmp.values - new_da.apr_tmp.values
plotting_data = np.where(np.isnan(plotting_data), np.nan, np.clip(plotting_data, min_val, max_val))

plotting_data = np.where(plotting_data < min_val, min_val, plotting_data)
plotting_data = np.where(plotting_data > max_val, max_val, plotting_data)
plotting_data = np.ma.masked_invalid(plotting_data)  # Mask NaN values for handling in contourf
p = new_da['pressure'].values
time = new_da['time'].values

ax.set_facecolor("gray")
contour = ax.contourf(time,p/1e2,plotting_data.T, levels=np.arange(min_val, max_val+(diff/2), diff), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Temperature (K)', fontsize = 20)
ticks_loc = np.arange(min_val, max_val+.01, (max_val-min_val)/10)
cbar.set_ticks(ticks_loc)
cbar.ax.tick_params(labelsize=20)

#Add grey to no data potts
for i in range(1, len(time)):
    gap = ((time[i] - time[i-1]) / np.timedelta64(1, 'D')).astype(int)
    if gap > 2:
        print(time[i])
        ax.axvspan(time[i-1], time[i], facecolor='grey', alpha=1)
        

plt.tight_layout()  # Ensures labels do not overlap

plt.savefig(os.path.join(fig_outdir, 'Temperature_ret_prior_diff.png'), format = 'png')

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
    



