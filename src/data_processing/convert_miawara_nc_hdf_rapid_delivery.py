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
import datetime as dt

#%% paths
<<<<<<< Updated upstream
server_basepath = '//storage/atmosphere/instruments/miawara/l2/'
outdir = '/storage/atmosphere/instruments/miawara/l2/NDACC/rapid_delivery'
=======
server_basepath = '//storage/tub/instruments/miawara/l2/'
outdir = '/storage/tub/instruments/miawara/l2/NDACC/rapid_delivery'
>>>>>>> Stashed changes
metadata_filepath = '../../additional_files/groundbased_mwr.h2o_ubern112_final_bern___001_rd.meta'

#%% datetime define
dt_now = dt.datetime.now() - dt.timedelta(days = 3)
nc_filepath = os.path.join(server_basepath, str(dt_now.year))
nc_filename = 'retrieval_{}{}{}1200.nc'.format(str(dt_now.year), str(dt_now.month), str(dt_now.day))
nc_fullpath = os.path.join(nc_filepath, nc_filename)

#%% define global attributes and variable attributes common to all observations
global_atts = parse_metadata_file(metadata_filepath)
global_atts = {key: value for key, value in global_atts.items() if not key.startswith('VAR')}
variable_atts = parse_meta_file(metadata_filepath)

#%% iterate through files, generate hdf

date_time_obj = dt_now 

startdate = date_time_obj.replace(hour=0, minute=0, second=0, microsecond=0)
enddate = date_time_obj.replace(hour=23, minute=59, second=59, microsecond=0)

#load the netcdf file
dataset = Dataset(nc_fullpath)
data_dict = extract_attributes(dataset)
    
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
subpath = os.path.join(outdir)

if not os.path.exists(subpath):
    os.makedirs(subpath)  

outpath = os.path.join(subpath, filename)

#add attributes depending on filename and metadata
global_atts['FILE_NAME'] = os.path.basename(outpath)

varlist = [key for key, item in data_dict.items()]
global_atts['DATA_VARIABLES'] = ';'.join(str(item) for item in varlist)

#write the 
write_hdf_from_nc(outpath, global_atts, variable_atts, data_dict)