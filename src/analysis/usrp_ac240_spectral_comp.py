#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:43:38 2023
Author: Alistair Bell
Contact: alistair.bell@unibe.ch
"""

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
from scipy import interpolate
import matplotlib.colors as mcolors
from scipy.stats import linregress
#%% paths

l1b_basepath = '/storage/tub/instruments/miawara/l1b/'
fig_file = "../../output/figs"
#%%

iterno = 0

time_ac240 = np.zeros(0)
spect_ac240 = np.zeros((0,16384))

time_usrp = np.zeros(0)
spect_usrp = np.zeros((0,16384))

#%%


for year in range(2022,2024):
    str_year = str(year)
    year_dir = os.path.join(l1b_basepath, str_year)
    for file in os.listdir(year_dir):
        if file.endswith('.nc'):
            try:
                temp_ac240 = xr.open_dataset(os.path.join(year_dir,file), group = 'spectrometer1', decode_times = False)
                temp_usrp = xr.open_dataset(os.path.join(year_dir,file), group = 'spectrometer2', decode_times = False)
                
                print('opened file: {}'.format(os.path.join(year_dir,file)))
            except Exception as e:
                print("error encountered: can't read both files")
                
            
            time_ac240 = np.append(time_ac240, temp_ac240.time)
            spect_ac240 = np.append(spect_ac240, temp_ac240.Tb, axis = 0 )
                
            time_usrp = np.append(time_usrp, temp_usrp.time )
            spect_usrp = np.append(spect_usrp, temp_usrp.Tb, axis = 0 )
                        
            temp_ac240.close() #close temporary datasets
            temp_usrp.close()
            
#%%

freq_ac240 = xr.open_dataset(os.path.join(year_dir,file), group = 'spectrometer1', decode_times = False).frequency.values[0]
freq_usrp = xr.open_dataset(os.path.join(year_dir,file), group = 'spectrometer2', decode_times = False).frequency.values[0]

#%%
C1 = ~np.isnan(spect_usrp)&(spect_usrp!=0)
C2 = ~np.isnan(spect_ac240)&(spect_ac240!=0)
              
valid_x = np.unique(np.where(C1&C2)[0])

usrp_spect_valid = spect_usrp[valid_x,:]
ac240_spect_valid = spect_ac240[valid_x,:]
valid_time = time_ac240[valid_x]

sort_time = np.argsort(valid_time)

#%%interpolate usrp to AC240

min_freq = max([freq_ac240.min(), freq_usrp.min()])
max_freq = min(freq_ac240.max(), freq_usrp.max())

bin_size = freq_ac240[0]- freq_ac240[1]  
bin_edges = np.arange(min_freq, max_freq + bin_size, bin_size)

def average_data_in_bins(data, freqs, bin_edges):
    bin_avg = []
    for i in range(len(bin_edges) - 1):
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i+1])
        bin_avg.append(np.mean(data[:,mask], axis = 1))
    return np.array(bin_avg)

ac240_binned = average_data_in_bins(ac240_spect_valid[sort_time], freq_ac240, bin_edges)
usrp_binned = average_data_in_bins(usrp_spect_valid[sort_time], freq_usrp, bin_edges)

#%%detect outliers

def replace_outliers_and_fill_columns(data, outlier_threshold=3, nan_threshold=0.05):
    """
    Replaces outliers in each row of a 2D array with NaN and then fills whole columns with NaN
    if more than a specified percentage of their values are NaNs.

    Parameters:
    - data: 2D numpy array.
    - outlier_threshold: Number of standard deviations a value must be away from the mean to be considered an outlier.
    - nan_threshold: Percentage of NaNs in a column to fill the whole column with NaNs.

    Returns:
    - A 2D numpy array of the same shape as data, where outliers are replaced by NaN, and columns with more than
      the specified percentage of NaNs are entirely filled with NaNs.
    """
    # Replace outliers with NaN
    row_mean = np.mean(data, axis=1, keepdims=True)
    row_std = np.std(data, axis=1, keepdims=True)
    outliers = np.abs(data - row_mean) > outlier_threshold * row_std
    data_with_nans = np.where(outliers, np.nan, data)

    # Check each column for NaN threshold
    nan_count = np.count_nonzero(np.isnan(data_with_nans), axis=0)
    columns_to_fill = nan_count > nan_threshold * data_with_nans.shape[0]

    data_with_nans[:, columns_to_fill] = np.nan

    return data_with_nans


def scaling_factor_corr(array, alpha):
    mean_tb = np.nanmean(array, axis = 0)
    mean_tb_reshaped = mean_tb.reshape(-1, 1).T
    term1 = 1/(1-alpha)
    term2 = term1*(array - alpha*mean_tb_reshaped )
    return term2
    
ac240_with_nans = replace_outliers_and_fill_columns(ac240_binned)
usrp_with_nans = replace_outliers_and_fill_columns(usrp_binned)

diff = usrp_with_nans[:, -200:] - ac240_with_nans[:, -200:]
mean_diff = np.nanmedian(diff, axis = 1)

diff_correction1 = usrp_with_nans[:, -150:] - ac240_with_nans[:, -150:]*1.046
ac240_correction = scaling_factor_corr(ac240_with_nans[:, -200:], .048)
diff_correction2 = np.nanmedian(usrp_with_nans[:, -200:] - ac240_correction, axis = 1)

ac240_100_corrected = np.mean(ac240_with_nans[:, 200]) + (ac240_with_nans[:, 200] - np.mean(ac240_with_nans[:, 200]))*1.046

#%%

#plt.plot(range(3277), np.nanmean(ac240_binned[:,100:200]), axis =1 ))

#plt.plot(bin_edges[:-1]/1e9, np.nanmean(usrp_with_nans[:, 100:200], axis =1))
plt.plot(bin_edges[:-1]/1e9 -22.235, mean_diff, label = 'Median Difference')
plt.plot(bin_edges[:-1]/1e9-22.235, diff_correction2, label = 'Median difference with 4.6% scaling factor')
plt.ylim([-.03,.1])
plt.grid()
plt.xlabel('Frequency (GHz)')
plt.ylabel(r'$\Delta$ Tb')
plt.legend()
plt.savefig(os.path.join(fig_file,'usrp_ac240_long_term_diff_with_scaling.pdf'), format = 'pdf', bbox_inches = 'tight',dpi = 400 )

#%%
plt.plot(bin_edges[:-1]/1e9 -22.235, usrp_with_nans[:,200] - ac240_with_nans[:,200] , label = 'Original Difference')

plt.plot(bin_edges[:-1]/1e9 -22.235, ac240_100_corrected - usrp_with_nans[:,200] , label = 'Corrected Difference')
plt.ylim([-.15,.1])
plt.grid()
plt.xlabel('Frequency (GHz)')
plt.ylabel(r'$\Delta$ Tb')
plt.legend()

#%%
plt.plot(bin_edges[:-1]/1e9 -22.235, np.nanmedian(usrp_with_nans[:,-200:], axis = 1) , label = 'USRP', color = 'blue')

plt.plot(bin_edges[:-1]/1e9 -22.235,np.nanmedian(ac240_with_nans[:,-200:], axis = 1) , label = 'AC-240', color = 'red')
plt.ylim([-1,1])
plt.grid()
plt.xlabel('Frequency (GHz)')
plt.ylabel(r'$\Delta$ Tb')
plt.legend()
plt.savefig(os.path.join(fig_file,'usrp_ac240_long_term_bal_Tb.pdf'), format = 'pdf', bbox_inches = 'tight',dpi = 400 )


#%%
diff_binned = np.nanmean(( usrp_binned - ac240_binned), axis = 1)


slope, intercept, _, _, _ = linregress(np.nanmean(ac240_with_nans[500:-500, -150:],axis = 1), np.nanmean(usrp_with_nans[500:-500, -150:],axis = 1))





