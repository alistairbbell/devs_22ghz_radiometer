#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:34:50 2023
Author: Alistair Bell
Contact: alistair.bell@unibe.ch
"""
import numpy as np
import os
from netCDF4 import Dataset
import glob
import datetime as dt
from  nc_to_hdf import * 

#%% paths
server_basepath = '//storage/atmosphere/instruments/miawarac/l2/l2_reproc'
outdir = '/storage/atmosphere/instruments/miawarac/l2/NDACC_v2/'
metadata_filepath = '../../additional_files/groundbased_mwr.h2o_ubern112_final_ny.alesund___002.meta'
#%% define global attributes and variable attributes common to all observations
global_atts = parse_metadata_file(metadata_filepath)
global_atts = {key: value for key, value in global_atts.items() if not key.startswith('VAR')}
variable_atts = parse_meta_file(metadata_filepath)
#%%load all nc files
nc_files = [file for year in np.arange(2018, 2024)
                for file in glob.glob(os.path.join(server_basepath, str(year), '*.nc'))]
nc_files.sort()
#verbosity 0 - 9
verbosity = 9
#%% iterate through files, generate hdf
i = 0

for filefullpath in nc_files:
    print(filefullpath)
    in_filename = os.path.basename(filefullpath)
    date_str = in_filename.split('_')[1].split('.')[0]
    date_time_obj = dt.datetime.strptime(date_str, '%Y%m%d%H%M')
    if verbosity>3: print(date_time_obj)        
    startdate = date_time_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    enddate = date_time_obj.replace(hour=23, minute=59, second=59, microsecond=0)

    try:
        #load the netcdf file
        dataset = Dataset(filefullpath)
        data_dict = extract_attributes_miawara_c(dataset, latitude =  78.9235, longitude = 11.9099,
                                       altitude = 32, p_unit = 'hPa')
            
        #input daily global attributes
        global_atts['DATA_START_DATE'] = datetime_to_stringDate(startdate)
        global_atts['DATA_STOP_DATE'] = datetime_to_stringDate(enddate)
        global_atts['FILE_GENERATION_DATE'] = datetime_to_stringDate(dt.datetime.now())
        
        #generate filename from metadata and date
        filename_lst = [
            str(global_atts['DATA_DISCIPLINE']).split(';')[2],
            str(global_atts['DATA_SOURCE']),
            str(global_atts['DATA_LOCATION']),
            str(global_atts['DATA_START_DATE']),
            str(global_atts['DATA_STOP_DATE']),
            str(global_atts['DATA_FILE_VERSION']) + '.hdf' 
        ]

        # Concatenating the version and file extension
        filename = '_'.join(filename_lst).lower()
        subpath = os.path.join(outdir,str(date_time_obj.year))

        if not os.path.exists(subpath):
            os.makedirs(subpath)  

        outpath = os.path.join(subpath, filename)

        #add attributes depending on filename and metadata
        if verbosity > 7: print('writing file attributes')
        global_atts['FILE_NAME'] = os.path.basename(outpath)

        varlist = [key for key, item in data_dict.items()]
        global_atts['DATA_VARIABLES'] = ';'.join(str(item) for item in varlist)

        #write the hdf
        if verbosity>5: print('writing hdf file')
        write_hdf_from_nc(outpath, global_atts, variable_atts, data_dict)
       

    except:
        print('An error for the hdf file generation on: {}'.format(date_time_obj.strftime("%d/%m/%Y")))

#%% code to check the file
# hdf4_file_path = outpath
# # Open the HDF4 file
# hdf4_file = SD(hdf4_file_path, SDC.READ)

# file_attributes = hdf4_file.attributes()
# print("File Attributes:")
# for attr_name, attr_value in file_attributes.items():
#     print(attr_name)
#     print(attr_value)

# # List all datasets in the HDF4 file
# datasets = hdf4_file.datasets()
# for ds_name in datasets.keys():
#     print('Dataset name:', ds_name)

# # Access a specific dataset (replace 'dataset_name' with the actual name)
# ds = hdf4_file.select('TEMPERATURE_INDEPENDENT')
# data = ds[:]
# hdf4_file.end()

# # Reading the file and printing the contents
# hdf = SD(outpath, SDC.READ)
# print("Global Attributes:")
# for key in global_atts.keys():
#     try:
#         attr = hdf.attr(key)
#         print(f"{key}: {attr.get()}")
#     except HDF4Error:
#         print(f"Attribute {key} not found")

# print("\nData in the File:")
# sds = hdf.select('DataSet')
# print(f"  Data: {sds.get()}")
# hdf.end
