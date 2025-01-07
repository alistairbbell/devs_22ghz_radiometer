#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:09:55 2023

@author: ab22l909
"""

import netCDF4 as nc
import os
import h5py
from datetime import datetime
import datetime as dt
import numpy as np
import datetime as dt
from pyhdf.SD import SD, SDC

#%% Change directory to the script's directory
absolute_script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(absolute_script_path)
os.chdir(script_directory)

#%%
server_basepath = '//storage/tub/instruments/miawara/l2/l2_scaled/'
metadata_file_path = '/home/ab22l909/MIAWARA_reprocessing_analysis/additional_files/groundbased_mwr.h2o_ubern112_final_bern___002.meta'

#%%
def parse_metadata_file(file_path):
    global_attributes = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains a global attribute
            if line.startswith('!') or not line.strip():
                continue  # Skip comments and empty lines
            key, value = line.strip().split('=', 1)
            global_attributes[key] = value
    return global_attributes

def datetime_to_mjd2k(mydate):
    """
    Convert a datetime object to Modified Julian Date 2000 (MJD2K).
    """
    # MJD starts from November 17, 1858
    mjd_start = dt.datetime(1858, 11, 17)
    # Number of days since MJD start
    days_since_mjd_start = (mydate - mjd_start).total_seconds() / 86400
    # MJD of January 1, 2000
    mjd2k_start = 51544    
    # Calculate MJD2K
    mjd2k = days_since_mjd_start - mjd2k_start
    return mjd2k

def extract_attributes(dataset):
    my_datetime = dt.datetime.fromtimestamp(int(dataset['time'][:])-3600*14)
    MJK200 = datetime_to_mjd2k(my_datetime)
    
    data_dict = {
        'LATITUDE.INSTRUMENT': 46.87699,
        'LONGITUDE.INSTRUMENT': 7.46521,
        'ALTITUDE.INSTRUMENT': 840,
        'DATETIME': MJK200,
        'ANGLE.VIEW_AZIMUTH': 0,
        'DATETIME_START': MJK200,
        'OPACITY.ATMOSPHERIC_EMISSION': np.nanmedian(dataset['tau'][:]),
        'DATETIME_END': MJK200+1,
        'PRESSURE_INDEPENDENT': dataset['pressure'][:]/ 100,
        'H2O_MIXING_RATIO_VOLUME_EMISSION': dataset['q'][:],
        'H2O_MIXING_RATIO_VOLUME_EMISSION_APRIORI': dataset['q_a'][:],
        'H2O.MIXING.RATIO.VOLUME_EMISSION_UNCERTAINTY.COMBINED.STANDARD': dataset['q_err'][:],
        'H2O.MIXING.RATIO.VOLUME_EMISSION_ALTITUDE': dataset['z'][:],
        'H2O.MIXING.RATIO.VOLUME_EMISSION_APRIORI_CONTRIBUTION': (1 - dataset['measurement_response'][:]) * 100,
        'H2O.MIXING.RATIO.VOLUME_EMISSION_AVK': dataset['A'][:],
        
        # Additional variables
        'DATA_FILE_VERSION': '002',
        'DATA_QUALITY': 'final;',
        'DATA_TEMPLATE': 'GEOMS-TE-MWR-003.csv',
        'FILE_META_VERSION': '04R077;IDLCR8HDF',
        'VAR_NAME': 'H2O.MIXING.RATIO.VOLUME_EMISSION_AVK',

    }
    return data_dict

# Parse the metadata file
global_attributes = parse_metadata_file(metadata_file_path)


# Write the global attributes to an HDF4 file
hdf4_file_path = 'output.hdf4'  # Replace with your desired output file path
with SD(hdf4_file_path, SDC.WRITE | SDC.CREATE) as hdf4_file:
    for attr_name, attr_value in global_attributes.items():
        hdf4_file.attr(attr_name).set(attr_value)
