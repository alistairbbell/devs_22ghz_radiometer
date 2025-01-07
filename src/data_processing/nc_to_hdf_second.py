#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:51:53 2023
Author: Alistair Bell
Contact: alistair.bell@unibe.ch
"""

import numpy as np
import h5py
import os
import pyhdf
from pyhdf.SD import SD, SDC, HDF4Error
from pyhdf.HDF import HDF
from pyhdf.V import V
import datetime as dt
from netCDF4 import Dataset
from pyhdf.SD import *
import struct
#%% paths
server_basepath = '//storage/tub/instruments/miawara/l2/l2_scaled_hdf_fields/'
outdir = '/home/alistair/miawara_reprocessing/data/tmp/'
metadata_filepath = '/home/alistair/miawara_reprocessing/additional_files/groundbased_mwr.h2o_ubern112_final_bern___002.meta'

proxy_datetime = dt.datetime(2023,3,21,13)
startdate = proxy_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
enddate = proxy_datetime.replace(hour=23, minute=59, second=59, microsecond=0)

#%%General functions
def parse_metadata_file(file_path):
    global_attributes = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains a global attribute
            if line.startswith('!') or not line.strip():
                continue  # Skip comments and empty lines
            key, value = line.strip().split('=', 1)
            # Check if value is empty and assign 'na' if it is
            global_attributes[key] = value if value.strip() else 'na'
    return global_attributes

def parse_meta_file(file_path):
    """
    Parses a .meta file and returns a dictionary where keys are 'VAR_NAME' and 
    values are dictionaries of each field.
    Args:
    file_path (str): The path to the .meta file.
    Returns:
    dict: A dictionary with the parsed data.
    """
    data = {}
    current_var = None

    with open(file_path, 'r') as file:
        for line in file:
            # Check for a new variable block
            if line.strip() == "! Variable Attribute":
                current_var = None
            elif line.startswith("VAR_NAME="):
                current_var = line.split("=")[1].strip()
                data[current_var] = {}
            elif current_var and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Assign a space if value is empty
                data[current_var][key] = value if value else 'na'

    return data

def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return value
    
def are_all_elements_floats(m_array):
    return np.issubdtype(m_array.dtype, np.floating)

def are_all_elements_integers(m_array):

    return np.issubdtype(m_array.dtype, np.integer)

def is_indexable(obj):
    """
    Test if the given object is indexable.

    Args:
    obj (any): The object to be tested.

    Returns:
    bool: True if the object is indexable, False otherwise.
    """
    try:
        _ = obj[0]  # Attempt to access the first element
        return True
    except (TypeError, IndexError):
        return False

def get_object_size(obj):
    if isinstance(obj, np.ndarray):
        # Handle NumPy arrays
        if obj.ndim == 2 and obj.shape[0] > 1 and obj.shape[1] > 1:
            # It's a 2D NumPy array with both dimensions larger than 1
            return [value.shape[i] for i in range(len(value.shape))]
        else:
            # For other NumPy arrays, return the shape
            return [obj.shape[0]]
    elif isinstance(obj, str):
        # Return 1 if the object is a single string
        return [1]
    elif isinstance(obj, (int, float)):
        # Handle single numbers
        return [1]
    elif isinstance(obj, (list, tuple)):
        # Check if it's a two-dimensional array
        if all(isinstance(subobj, (list, tuple)) and len(subobj) > 1 for subobj in obj):
            # It's a 2D array with both dimensions larger than 1, get dimensions
            dim1 = len(obj)
            dim2 = len(obj[0])  # Assumes all sub-arrays have equal length

            return [dim1, dim2]
        elif all(isinstance(subobj, (list, tuple)) for subobj in obj):
            # It's a 2D array but one of the dimensions might be 1
            dim1 = len(obj)
            dim2 = len(obj[0]) if dim1 > 0 else 0

            if dim1 == 1 or dim2 == 1:
                # Return the larger dimension if one of them is 1
                return [max(dim1, dim2)]
            else:
                # Both dimensions are greater than 1
                return [dim1, dim2]
        else:
            # It's a 1D list/tuple
            return [len(obj)]
    else:
        # Handle other iterable data types
        try:
            return [len(obj)]
        except TypeError:
            return ['']

def list_to_semicolon_separated_string(lst):
    # Convert each number in the list to a string
    string_list = [str(item) for item in lst]

    # Join the string representations with a semicolon
    return ';'.join(string_list)
        
def convert_value(value, type):
    if type == 'REAL':  # Convert to 32-bit float
        original_value = float(value)  # Convert to Python float (64-bit)
        packed_value = struct.pack('f', original_value)
        #unpacked_value = struct.unpack('f', packed_value)[0]
        #print(f"Value: {unpacked_value}, Type: 32-bit float")
        return packed_value
    elif type == 'DOUBLE':  # Keep as 64-bit float
        original_value = float(value)
        print(f"Value: {original_value}, Type: 64-bit float")
        return original_value
    elif type == 'INTEGER':  # Convert to integer
        return int(value)
    else:
        return str(value)
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
    return np.float64(mjd2k)

def datetime_to_stringDate(myDatetime):
    """
    Convert a datetime object to a string date time in a format desired by GEOMS.
    """
    formatted_date = myDatetime.strftime('%Y%m%dT%H%M%S') + 'Z'
    return formatted_date

def float_precision_test(value):
    # Convert the value to 32-bit float and back to Python float (64-bit)
    value_32bit = struct.unpack('f', struct.pack('f', value))[0]

    # Compare the original value with the 32-bit converted value
    if value == value_32bit:
        return "32-bit"
    else:
        return "64-bit"

#%%Function to extract data from Netcdf

def extract_attributes(dataset):
    my_datetime = dt.datetime.fromtimestamp(int(dataset['time'][:])).replace(hour =12, minute=0, second = 0, microsecond = 0)
    datetime_start = dt.datetime.fromtimestamp(int(dataset['time'][:])).replace(hour =0, minute=0, second = 0, microsecond = 0)
    datetime_end = dt.datetime.fromtimestamp(int(dataset['time'][:])).replace(hour =23, minute=59, second = 59, microsecond = 0)
    
    valid_stringtime = datetime_to_stringDate(my_datetime)
    start_stringtime = datetime_to_stringDate(my_datetime.replace(hour =0, minute=0, second = 0))
    end_stringtime = datetime_to_stringDate(my_datetime.replace(hour =23, minute=59, second = 59) )

    data_dict = {
        'LATITUDE.INSTRUMENT': 46.87699,
        'LONGITUDE.INSTRUMENT': 7.46521,
        'ALTITUDE.INSTRUMENT': int(840),
        'DATETIME': datetime_to_mjd2k(my_datetime),
        'ANGLE.VIEW_AZIMUTH': 0. ,
        'DATETIME.START':  datetime_to_mjd2k(datetime_start),
        'DATETIME.STOP' : datetime_to_mjd2k(datetime_end),
        'OPACITY.ATMOSPHERIC_EMISSION' : dataset['tau'][:],
        'PRESSURE_INDEPENDENT': dataset['pressure'][:]/ 100,
        'ALTITUDE': np.array([int(i) for i in dataset['z'][:]]),
        'H2O.MIXING.RATIO.VOLUME_EMISSION': dataset['q'][:],
        'H2O.MIXING.RATIO.VOLUME_EMISSION_APRIORI': dataset['q_a'][:],
        'H2O.MIXING.RATIO.VOLUME_EMISSION_UNCERTAINTY.COMBINED.STANDARD': dataset['q_err'][:],
        'H2O.MIXING.RATIO.VOLUME_EMISSION_RESOLUTION.ALTITUDE': np.array([int(i) for i in dataset['z'][:]]),
        'H2O.MIXING.RATIO.VOLUME_EMISSION_APRIORI.CONTRIBUTION': (1 - dataset['measurement_response'][:]) * 100,
        'H2O.MIXING.RATIO.VOLUME_EMISSION_AVK': dataset['A'][:],
        'H2O.MIXING.RATIO.VOLUME_EMISSION_UNCERTAINTY.RANDOM.STANDARD': dataset['q_err'][:],
        'H2O.MIXING.RATIO.VOLUME_EMISSION_UNCERTAINTY.SYSTEMATIC.STANDARD': np.zeros(len(dataset['q_err'][:])),
        'ANGLE.VIEW_ZENITH_MEAN' : dataset['ZA'][:],
        'ANGLE.SOLAR_ZENITH_MEAN': np.zeros(len(dataset['tau'][:])),
        'INTEGRATION.TIME': np.float64(dataset['tint'][:]),
        'TEMPERATURE_INDEPENDENT': dataset['species1_T'][:],  
        
    }
    return data_dict





#%%

def write_hdf_from_nc(outpath, global_atts,data_dict):
    # Open the HDF4 file in write mode
    hdf = SD(outpath, SDC.WRITE | SDC.CREATE)
    # Loop through the dictionary and set each attribute
    for key, value in global_atts.items():
        #print(key, value)
        # Create or get the attribute
        attr = hdf.attr(key)
        value = value if value else 'na'
            # Determine the type of the value and set attribute
        if isinstance(value, str):
            # Set string attribute
            attr.set(SDC.CHAR, value)
        elif isinstance(value, int):
            # Set integer attribute
            attr.set(SDC.INT32, [value])
        elif isinstance(value, float):
            # Set float attribute
            attr.set(SDC.FLOAT64, [value])
    for key, value in data_dict.items():
        #print(key)
        
        tempdict = variable_atts[key]
        datatype = tempdict['VAR_DATA_TYPE']
        # Create a dataset
        if isinstance(value, str):
            # Handle single string data
            sds = hdf.create(key, SDC.CHAR, (len(value),))
            sds[:] = np.array(list(value), 'c')
    
        elif is_indexable(value):
            shape = value.shape
            if len(value.shape)<2:
                shape = (value.shape[0], len(value.shape))
            
            if are_all_elements_floats(value):
                if datatype == 'REAL':
                    sds = hdf.create(key, SDC.FLOAT32, shape)
                    sds[:] = value.astype(np.float32).tolist()
                elif datatype == 'DOUBLE':
                    sds = hdf.create(key, SDC.FLOAT64, shape)
                    sds[:] = value.astype(np.float64).tolist()
    
                elif datatype == 'INTEGER':
                    sds = hdf.create(key, SDC.INT32, shape)
                    sds[:] = value.astype(np.int32).tolist()
                else:
                    print(datatype+'data type not recognised')
    
                
            elif are_all_elements_integers(value):
                sds = hdf.create(key, SDC.INT32, value.shape)
                sds[:] = value.astype(np.int32).tolist()
                
            elif isinstance(value, list) and all(isinstance(item, float) for item in value):
                sds = hdf.create(key, SDC.FLOAT32, (len(value), len(value[0])))
                sds[:] = value
                
        elif isinstance(value, float) or  isinstance(value, int):
    
            if datatype == 'REAL':
                sds = hdf.create(key, SDC.FLOAT32, 1)
                sds[:] = value
    
            elif datatype == 'DOUBLE':
                sds = hdf.create(key, SDC.FLOAT64, 1)
                sds[:] = value
    
            elif datatype == 'INTEGER':
                sds = hdf.create(key, SDC.INT32, 1)
                sds[:] = np.int32(value)
    
            else:
                print(datatype+'data type not recognised')
                sds[:] = value
           
        else:
            # Handle float data
            sds = hdf.create(key, SDC.CHAR, (len('nan'),))
            sds[:] = 'nan'
        #load a temporary dicitionary containing the specific variable attributes
        sds.VAR_NAME = key
        sds.VAR_SIZE =  list_to_semicolon_separated_string(get_object_size(value))
        #print(list_to_semicolon_separated_string(get_object_size(value)))
        #write these to the variable options
        sds.VAR_DESCRIPTION = tempdict['VAR_DESCRIPTION']
        sds.VAR_NOTES = tempdict['VAR_NOTES']
        sds.VAR_UNITS  = tempdict['VAR_UNITS']
        sds.VAR_DEPEND =  tempdict['VAR_DEPEND']
        sds.VAR_DATA_TYPE=tempdict['VAR_DATA_TYPE']
        sds.VAR_UNITS=tempdict['VAR_UNITS']
        sds.VAR_SI_CONVERSION=tempdict['VAR_SI_CONVERSION']
        
        if tempdict['VAR_DATA_TYPE'] == 'REAL':
            att1 = sds.attr('VAR_VALID_MIN')
            att1.set(SDC.FLOAT32, float(tempdict['VAR_VALID_MIN']))
            
            att2 = sds.attr('VAR_VALID_MAX')
            att2.set(SDC.FLOAT32, float(tempdict['VAR_VALID_MAX']))
            
            att3 = sds.attr('VAR_FILL_VALUE')
            att3.set(SDC.FLOAT32, float(tempdict['VAR_FILL_VALUE']))
    
        else: 
            sds.VAR_VALID_MIN= convert_value(tempdict['VAR_VALID_MIN'],  tempdict['VAR_DATA_TYPE'])
            sds.VAR_VALID_MAX=convert_value(tempdict['VAR_VALID_MAX'],  tempdict['VAR_DATA_TYPE'])
            sds.VAR_FILL_VALUE=convert_value(tempdict['VAR_FILL_VALUE'],  tempdict['VAR_DATA_TYPE'])
        
        del tempdict
        sds.endaccess()
    hdf.end()

#%% define global attributes and variable attributes common to all observations
global_atts = parse_metadata_file(metadata_filepath)
global_atts = {key: value for key, value in global_atts.items() if not key.startswith('VAR')}
variable_atts = parse_meta_file(metadata_filepath)
#%% load data from nc file
filefullpath = os.path.join(server_basepath, '2023', 'retrieval_202303011200.nc')
dataset = Dataset(filefullpath)
data_dict = extract_attributes(dataset)

#%%input daily global attributes
global_atts['DATA_START_DATE'] = datetime_to_stringDate(startdate)
global_atts['DATA_STOP_DATE'] = datetime_to_stringDate(enddate)
global_atts['FILE_GENERATION_DATE'] = datetime_to_stringDate(dt.datetime.now())

#%%
filename_lst = [
    str(global_atts['DATA_DISCIPLINE']).split(';')[2],
    str(global_atts['DATA_SOURCE']),
    str(global_atts['DATA_LOCATION']),
    str(global_atts['DATA_START_DATE']),
    str(global_atts['DATA_STOP_DATE']),
    str(global_atts['DATA_FILE_VERSION']) + '.hdf'  # Concatenating the version and file extension
]

filename = '_'.join(filename_lst).lower()
outpath = os.path.join(outdir, filename)
#%%
global_atts['FILE_NAME'] = os.path.basename(outpath)
varlist = [key for key, item in data_dict.items()]
global_atts['DATA_VARIABLES'] = ';'.join(str(item) for item in varlist)
    
#%%
hdf4_file_path = outpath
# Open the HDF4 file
hdf4_file = SD(hdf4_file_path, SDC.READ)

file_attributes = hdf4_file.attributes()
print("File Attributes:")
for attr_name, attr_value in file_attributes.items():
    print(attr_name)
    print(attr_value)

# List all datasets in the HDF4 file
datasets = hdf4_file.datasets()
for ds_name in datasets.keys():
    print('Dataset name:', ds_name)

# Access a specific dataset (replace 'dataset_name' with the actual name)
ds = hdf4_file.select('TEMPERATURE_INDEPENDENT')
data = ds[:]
hdf4_file.end()

##%bla


# Reading the file and printing the contents
hdf = SD(outpath, SDC.READ)
print("Global Attributes:")
for key in global_atts.keys():
    try:
        attr = hdf.attr(key)
        print(f"{key}: {attr.get()}")
    except HDF4Error:
        print(f"Attribute {key} not found")

print("\nData in the File:")
sds = hdf.select('DataSet')
print(f"  Data: {sds.get()}")
hdf.end
write_hdf_from_nc(outpath, global_atts,data_dict)
