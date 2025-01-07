#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:24:24 2023

Author: Alistair Bell

Contact: alistair.bell@unibe.ch

about: file to download, concatenate and save MLS water vapour files as the 
zonal average
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
from utils import temps
import uuid
from smb.SMBConnection import SMBConnection
import logging
import traceback
from netCDF4 import Dataset
from logging.handlers import RotatingFileHandler
import h5netcdf
import h5py

#%%read and concat functions
def read_and_concatenate_groups(filepath, groups, my_engine):
    with xr.open_dataset(filepath, engine='h5netcdf') as ds:

        # Create a list to store individual datasets from each group
        datasets = []

        # Iterate over each group, reading the data and appending to the list
        for group in groups:
            group_ds = xr.open_dataset(filepath, group=group, engine=my_engine)
            datasets.append(group_ds)
        datasets = [ds.drop_vars('ChunkNumber', errors='ignore') for ds in datasets]
        # Concatenate datasets along a new dimension (for example 'group')
        # If the datasets have matching dimensions and coordinates, you might want to use merge instead
        combined_ds = xr.merge(datasets, compat='override')

        return combined_ds

#%%

def read_average(files, h2o_string, groups, temp_file_loc, outpath, server_path,
                 file_ext = '.he5', identifier = '1', my_engine = 'h5netcdf'):
    
    iterno = 0
    temp_path =  os.path.join(temp_file_loc, 'temp' + file_ext)
    bin_edges = list(range(-90, 91, 2)) #for averaging zonally
    
    ref_date = dt.datetime(1993, 1, 1) 
    
    outend = 'MLS_Zonal_{}.nc'.format(identifier)
    outfullPath = os.path.join(outpath, outend)
    
    for file in files:
    
        if h2o_string in file.filename and file.filename.endswith(file_ext):
            
            print(os.path.join(server_path, file.filename)) #for debug
            with open(temp_path, 'wb') as local_file:
                conn.retrieveFile(share_name, os.path.join(server_path, file.filename), local_file)
            
            year = int(file.filename[-12:-8])
            days = int(file.filename[-7:-4])
            ## Load the file into xarray
            try:
                ds = read_and_concatenate_groups(temp_path, groups, my_engine)
                
                #Average by laitude
                averaged_by_latitude = ds.groupby_bins('Latitude', bins=bin_edges).mean()
                
                av_datetime = dt.datetime(year,1,1 ) + dt.timedelta(days = days -1)
                print(av_datetime)
                
                averaged_by_latitude = averaged_by_latitude.drop('Time')
                averaged_by_latitude = averaged_by_latitude.expand_dims({"time": [av_datetime]})
                averaged_by_latitude['time'].attrs = {'long_name': 'time',
                                    'standard_name': 'time'}
                            
                ## Average zonally
                if iterno < 1: #if first file initialise xarray add date!!!!
                    concatenated_ds = averaged_by_latitude
                    iterno += 1
                else: #else try to append to existing xarray
                    try:
                        concatenated_ds = xr.concat([concatenated_ds, averaged_by_latitude], dim = 'time')
                        iterno += 1
                    except Exception as e:
                        print('error encountered')
                        logger.error(traceback.format_exc())
                        ds.close() #close temporary dataset
            except Exception as e:
                print('error encountered for file:{}'.format(file.filename))
                logger.error(traceback.format_exc())
            if os.path.exists(temp_path):
                print('removing tmp file')
                os.remove(temp_path)
    bin_indices = range(len(concatenated_ds['Latitude_bins']))
    concatenated_ds = concatenated_ds.assign_coords(bin_index=('Latitude_bins', bin_indices))
    for var in concatenated_ds.data_vars:
        if 'Latitude_bins' in ds[var].dims:
            concatenated_ds[var] = concatenated_ds[var].swap_dims({'Latitude_bins': 'bin_index'})
    concatenated_ds = concatenated_ds.drop('Latitude_bins')
    concatenated_ds.to_netcdf(outfullPath)


#%%
if __name__ == "__main__":
#%%
    # Connection details
    server_name = 'datatub'  # This is just a descriptor
    server_ip = 'datatub.mw.iap.unibe.ch'
    username = 'tub_r'
    password = ''
    client_name = 'user'  # No account needed to read files
    share_name = 'srv'
    svr_data_path = 'atmosphere/AuraMLS/Level2_v5/H2O/'
    data_folder = "../../data"
    log_folder = "../../log"
    download_folder = os.path.join(data_folder, "tmp")
    temp_file_path = os.path.join(download_folder, "temp_file.he5")
    save_output_dir = os.path.join(data_folder, "interim")
    output_file_name = 'MLS_zonal_H2O'
    h2o_string = 'MLS-Aura'
    group_extract = ['HDFEOS/SWATHS/H2O-APriori/Geolocation Fields/',
     'HDFEOS/SWATHS/H2O-APriori/Data Fields/',
     'HDFEOS/SWATHS/H2O/Data Fields/']


#%% set up error logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler(os.path.join(log_folder,"mls_data_retrieve.txt"), maxBytes=1*1024*1024, backupCount=1)
logger.addHandler(handler)

#%% Retrieve the MLS data from the tub server

    
#%%

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True)
conn.connect(server_ip)
for year in np.arange(2010,2024):
    print(year)
    fileDir = os.path.join(svr_data_path, str(year))
    files = conn.listPath(share_name, fileDir)
            
    read_average(files, h2o_string, group_extract, download_folder, save_output_dir,
                  fileDir, identifier = str(year))
    
#%%
for file in files:
    if h2o_string in file.filename and file.filename.endswith("he5"):
        print(os.path.join(fileDir, file.filename)) #for debug
#%%

for file in files:
    if h2o_string in file.filename and file.filename.endswith(".nc"):
        print(os.path.join(svr_data_path, file.filename)) #for debug
        with open(temp_file_path, 'wb') as local_file:
            conn.retrieveFile(share_name, os.path.join(svr_data_path, file.filename), local_file)

        # Load the file into xarray
        ds = xr.open_dataset(temp_file_path)
        if not (ds.time.dtype == np.float64 or ds.time.dtype == np.dtype('O')):
            if iterno == 0: #if first file initialise xarray
                concatenated_ds = ds
                iterno += 1
            else: #else try to append to existing xarray
                try:
                    concatenated_ds = xr.merge([concatenated_ds, ds])
                    iterno += 1
                except Exception as e:
                    print('error encountered')
                    logger.error(traceback.format_exc())
        ds.close() #close temporary dataset
               
conn.close()



