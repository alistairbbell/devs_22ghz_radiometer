"""
Author: Alistair Bell
Contact: alistair.bell@unibe.ch
Created: 11.09.23
About:
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


#%%
absolute_script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(absolute_script_path)
# This is the absolute path of the current script
os.chdir(script_directory)
#%%
# Connection details
server_name = 'datatub'  #server descriptor
server_ip = 'datatub.mw.iap.unibe.ch'
username = 'tub_r'
password = ''
client_name = 'user'  # local machine
share_name = 'srv'
server_basepath = 'instruments/miawara/l2/l2_scaled/'
data_folder = "../../data"
download_folder = os.path.join(data_folder, "tmp")
temp_file_path = os.path.join(download_folder, "temp_file.nc")
save_output_dir = os.path.join(data_folder, "interim")
output_file_name = 'MIAWARA_concat_H2O_2010_2023_rescaled_inc_A.nc'
outFullPath = os.path.join(save_output_dir, output_file_name)
h2o_string = 'retrieval'
log_folder = "../../log"

#%% set up error logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler(os.path.join(log_folder,"miawara_data_retrieve.txt"), maxBytes=1*1024*1024, backupCount=1)
logger.addHandler(handler)

#%%#%% Retrieve the MIAWARA data from the tub server
years = np.arange(2010,2024)
years_s = [str(i) for i in years]

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True)
conn.connect(server_ip)

# List all files in the root of the share
file_paths_to_merge = []

#variables to drop from the xarray dataset (dataset too big if we keep them)
vars_to_drop = ['y', 'yf', 'tau', 'J', 'frequency'] 

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
            ds_red = ds.drop_vars(vars_to_drop) #drop heavy vars
            ds_red = ds_red.drop_vars('A') #drop heavy vars
            
            #if not (ds.time.dtype == np.float64 or ds.time.dtype == np.dtype('O')):
            if iterno == 0: #if first file initialise xarray
                concatenated_ds = ds_red
                iterno += 1
            else: #else try to append to existing xarray
                try:
                    concatenated_ds = xr.concat([concatenated_ds, ds_red], dim = 'time')
                    iterno += 1
                except Exception as e:
                    print('error encountered')
                    logger.error(traceback.format_exc())
            ds.close() #close temporary datasets
            ds_red.close()

#concatenated_ds.to_netcdf(outFullPath)
conn.close()

#%%#%% Handling of Averaging Kernel the MIAWARA data from the tub server
years = np.arange(2010,2024)
years_s = [str(i) for i in years]

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True)
conn.connect(server_ip)

# List all files in the root of the share
file_paths_to_merge = []

#variables to drop from the xarray dataset (dataset too big if we keep them)
vars_to_drop = ['y', 'yf', 'tau', 'J']

A_all = np.zeros([80,80,0])
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
concatenated_ds = concatenated_ds.assign_coords(new_pressure=ds['pressure'])
concatenated_ds['A'] = (('pressure', 'new_pressure', 'time'), A_all)
concatenated_ds = concatenated_ds.sortby('time')
concatenated_ds.to_netcdf(outFullPath)

#%%
if __name__ == "__main__":
    a_kern = concatenated_ds.A.values
    pressure = concatenated_ds.pressure.values
    
    a_kern_av = np.nanmean(A_all, axis = 2)
    #a_kern_av = a_kern[:,:,100]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_ylim(100, .0010)
    ax.set_yscale('log')
    ax.set_xlabel('Latitude (degrees)', fontsize = 20)
    ax.set_ylabel('Pressure (hPa)', fontsize = 20)
    
    for i in range(80):
        ax.plot( a_kern_av[:,i],pressure/100, alpha=0.3)
    
    for i in np.arange(0,80, 5):
        ax.plot(a_kern_av[:,i],pressure/100, alpha=1)
        
    plt.grid()



