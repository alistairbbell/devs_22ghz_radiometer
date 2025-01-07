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
<<<<<<< Updated upstream
server_basepath = '//storage/atmosphere/instruments/miawara/l2/l2_scaled_hdf_fields/'
outdir = '/storage/atmosphere/instruments/miawara/l2/NDACC_v2/'
=======
server_basepath = '//storage/tub/instruments/miawara/l2/l2_scaled_hdf_fields/'
outdir = '/storage/tub/instruments/miawara/l2/NDACC_v2/'
>>>>>>> Stashed changes
metadata_filepath = '../../additional_files/groundbased_mwr.h2o_ubern112_final_bern___002.meta'

#%% define global attributes and variable attributes common to all observations
global_atts = parse_metadata_file(metadata_filepath)
global_atts = {key: value for key, value in global_atts.items() if not key.startswith('VAR')}
variable_atts = parse_meta_file(metadata_filepath)
#%%load all nc files
<<<<<<< Updated upstream
nc_files = [file for year in np.arange(2024, 2025) 
=======
nc_files = [file for year in np.arange(2010, 2024) 
>>>>>>> Stashed changes
                for file in glob.glob(os.path.join(server_basepath, str(year), '*.nc'))]
#%% iterate through files, generate hdf

for filefullpath in nc_files:

	in_filename = os.path.basename(filefullpath)
	date_str = in_filename.split('_')[1].split('.')[0]
	date_time_obj = dt.datetime.strptime(date_str, '%Y%m%d%H%M')
<<<<<<< Updated upstream
	print(date_time_obj)
=======
	print(date_time_obj)	    
>>>>>>> Stashed changes
	startdate = date_time_obj.replace(hour=0, minute=0, second=0, microsecond=0)
	enddate = date_time_obj.replace(hour=23, minute=59, second=59, microsecond=0)

	try:
		#load the netcdf file
		dataset = Dataset(filefullpath)
		data_dict = extract_attributes(dataset)
<<<<<<< Updated upstream

=======
			
>>>>>>> Stashed changes
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
		global_atts['FILE_NAME'] = os.path.basename(outpath)

		varlist = [key for key, item in data_dict.items()]
		global_atts['DATA_VARIABLES'] = ';'.join(str(item) for item in varlist)

		#write the 
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
