# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#imports

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

#%% paths
server_basepath = '//storage/tub/instruments/miawara/l2/l2_scaled/'
meta_file_path = '/home/ab22l909/MIAWARA_reprocessing_analysis/additional_files/groundbased_mwr.h2o_ubern112_final_bern___002.meta'


# Define custom global attributes
global_attributes = {
    'PI_NAME' : 'Murk; Axel',
    'PI_AFFILIATION' : 'Institute of Applied Physics, University of Bern; IAP UniBe',
    'PI_ADDRESS': 'Sidlerstrasse 5, 3012 Bern; Switzerland',
    'PI_EMAIL': 'axel.murk@unibe.ch',
    
    'DO_NAME': 'Bell;Alistair',
    'DO_AFFILIATION': 'Institute of Applied Physics, University of Bern; IAP UniBe',
    'DO_ADDRESS': 'Sidlerstrasse 5, 3012 Bern; Switzerland',
    'DO_EMAIL': 'alistair.bell@unibe.ch',
    
    'DS_NAME': 'Bell;Alistair',
    'DS_AFFILIATION': 'Institute of Applied Physics, University of Bern; IAP UniBe',
    'DS_ADDRESS': 'Sidlerstrasse 5, 3012 Bern; Switzerland',
    'DS_EMAIL': 'alistair.bell@unibe.ch'
    }


# Function to parse the .meta file
def parse_meta_file(meta_file_path):
    attributes = {}
    with open(meta_file_path, 'r') as file:
        for line in file:
            # Skip empty lines and comments
            if not line.strip() or line.startswith('!'):
                continue
            
            # Split line into key and value
            if '=' in line:
                key, value = line.split('=', 1)
                attributes[key.strip()] = value.strip()

    return attributes

# Function to add attributes to HDF5 file
def add_attributes_to_hdf5(hdf5_file, attributes):
    for key, value in attributes.items():
        hdf5_file.attrs[key] = value

# paths
server_basepath = '//storage/tub/instruments/miawara/l2/l2_scaled/'
meta_file_path = '/home/ab22l909/MIAWARA_reprocessing_analysis/additional_files/groundbased_mwr.h2o_ubern112_final_bern___002.meta'

# Parse the .meta file
attributes = parse_meta_file(meta_file_path)


nc_fields = {'year', 'month', 'day', 'hour', 'minute', 'second', 'cost',
    'cost_x', 'cost_y', 'species1_p', 'species1_x', 'species1_e',
    'species1_es', 'species1_mr','species1_z', 'y', 'yf' ,
    'species1_xa', 'converged', 'J', 'f', 'species1_A'}


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


        

def convert_type_if_needed(value):
    """
    Determines the HDF4 data type for the given value.
    Converts non-string values to strings and encodes them.
    """
    if isinstance(value, str):
        return SDC.CHAR8, value.encode('utf-8')
    else:
        # Convert non-string values to string and then encode
        str_value = str(value)
        return SDC.CHAR8, str_value.encode('utf-8')
    
    
def add_global_attributes_to_hdf4(hdf4_file, attributes):
    for key, value in attributes.items():
        # Check if value is a string
        if isinstance(value, str):
            # Encode the string as bytes
            encoded_value = value.encode('utf-8')
            # Set the attribute with CHAR8 data type
            hdf4_file.attr(key).set(SDC.CHAR8, encoded_value)
        else:
            # For non-string values, convert them to string first
            str_value = str(value)
            # Set the attribute with CHAR data type
            hdf4_file.attr(key).set(SDC.CHAR, str_value)


def add_variable_attributes_to_dataset(sds, attributes):
    for key, value in attributes.items():
        if isinstance(value, np.ndarray):
            sds.setattr(key, value)
        else:
            sds.setattr(key, np.array([value]))

def save_to_hdf4(hdf4_file_path, data_dict, global_attributes, variable_attributes):
    hdf4_file = SD(hdf4_file_path, SDC.WRITE | SDC.CREATE)

    # Add global attributes
    add_global_attributes_to_hdf4(hdf4_file, global_attributes)

    # Add data fields and their attributes
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            sds = hdf4_file.create(key, SDC.FLOAT32, value.shape)
            sds[:] = value
            add_variable_attributes_to_dataset(sds, variable_attributes.get(key, {}))
        else:
            sds = hdf4_file.create(key, SDC.FLOAT32, (1,))
            sds[0] = value
            add_variable_attributes_to_dataset(sds, variable_attributes.get(key, {}))
        sds.endaccess()

    hdf4_file.end()


#%%

for year in range(2010, 2011):
    yearpath = os.path.join(server_basepath, str(year))
    for filename in os.listdir(yearpath):
        if filename.endswith('.nc'):
            print(filename)
            nc_file_path = os.path.join(yearpath, filename)
            with nc.Dataset(nc_file_path, 'r') as dataset:
                # Extract data and attributes
                variable_attributes = extract_attributes(dataset)
                hdf4_file_path = os.path.join(server_basepath, filename[:-3] + '.hdf')
                save_to_hdf4(hdf4_file_path, variable_attributes, global_attributes, variable_attributes)




