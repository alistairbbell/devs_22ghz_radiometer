"""
Author: Alistair Bell
Contact: alistair.bell@unibe.ch
Created: 12.09.23
About: Routine to download and concat all MLS files over Bern, saving nc file 
locally.
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
#from utils import temps
#import smbprotocol
import uuid
from smb.SMBConnection import SMBConnection
import logging
import traceback
from netCDF4 import Dataset
from logging.handlers import RotatingFileHandler


#%%
# Connection details
server_name = 'datatub'  # This is just a descriptor
server_ip = 'datatub.mw.iap.unibe.ch'
username = 'tub_r'
password = ''
client_name = 'user'  # No account needed to read files
share_name = 'srv'
svr_data_path = 'atmosphere/AuraMLS/Level2_v5/locations/BERN'
data_folder = "../../data"
log_folder = "../../log"
download_folder = os.path.join(data_folder, "tmp")
temp_file_path = os.path.join(download_folder, "temp_file.nc")
save_output_dir = os.path.join(data_folder, "interim")
output_file_name = 'MLS_concat_H2O_2023.nc'
outFullPath = os.path.join(save_output_dir, output_file_name)
h2o_string = 'AuraMLS_L2GP-H2O_v5'
h2o_string_2 = '2023'
#%% set up error logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler(os.path.join(log_folder,"mls_data_retrieve.txt"), maxBytes=1*1024*1024, backupCount=1)
logger.addHandler(handler)


tempfilename = 'AuraMLS_L2GP-H2O_v5_2023-12-18_800_BERN.nc'
#%% Retrieve the MLS data from the tub server

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True)
conn.connect(server_ip)

# List all files in the root of the share
files = conn.listPath(share_name, svr_data_path)
filenamelist = [file.filename for file in files]
iterno = 0

for file in files:
    #print(file.filename)
    if (h2o_string in file.filename) and (h2o_string_2 in file.filename) and file.filename.endswith(".nc"):
        print(os.path.join(svr_data_path, file.filename)) #for debug
        with open(temp_file_path, 'wb') as local_file:
            conn.retrieveFile(share_name, os.path.join(svr_data_path, file), local_file)

        # Load the file into xarray
        ds = xr.open_dataset(temp_file_path, decode_times = False)
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
#%%
concatenated_ds.to_netcdf(outFullPath)
print("Merged file saved to {}".format(outFullPath))




