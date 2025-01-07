#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:53:06 2023

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

#%%imports
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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

#%% paths
data_folder = "../../data"
log_folder = "../../log"
MLS_filename = 'MLS_concat_H2O_v2.nc'
fig_file = "../../output/figs"
data_file = "../../output/data"

interim_data= os.path.join(data_folder, 'interim', MLS_filename)
#%%
ds = xr.open_dataset(interim_data, decode_times = False)
current_a_prior = '/home/alistair/MIAWARA_ret/operational/MIAWARA/extra_files/ecmwf_2010_2015_3_9_15_21h.nc'
ds_ecmwf = xr.open_dataset(current_a_prior, decode_times = False)


#%%first entry each year
first_entry_each_year = ds.groupby('time.year').first()
q = np.array(ds['q'][:,:])

qMean = np.mean(q, axis = 0)
rmse = np.sqrt( np.mean((q-qMean)**2, axis = 0))

q_std = np.array(first_entry_each_year['value'].std('year'))
P_mean = np.array(first_entry_each_year['pressure'])

#%%
q_std_list = []

# Create an empty list to store the pressure values for the first day of each month
P_mean_list = []

# Iterate through each month
for month in range(1, 13):  # 1 to 12 for January to December
    # Select data for the first day of the current month across all years
    monthly_first_entries = ds.sel(time=ds['time.month'] == month).groupby('time.year').first()

    # Extract the 'value' data
    q = np.array(monthly_first_entries['value'])

    # Compute the standard deviation for this month's first day across all years
    q_std = q.std(axis = 0)
    q_std_list.append(q_std)

    # Extract 'pressure' data for this month's first day
    P_mean_list.append(P_mean)

# Convert the lists to numpy arrays
q_std_array = np.array(q_std_list)
P_mean = np.array(monthly_first_entries['pressure'])

q_std_all = np.stack(q_std_array)
q_std_yearly = np.mean(q_std_all, axis = 0)
#%% plotting
plt.figure(figsize=(6, 8))

plt.plot(q_std_list[1]*1e6, P_mean/100, label = 'January')
plt.plot(q_std_list[4]*1e6, P_mean/100, label = 'April')
plt.plot(q_std_list[7]*1e6, P_mean/100, label = 'July')
plt.plot(q_std_list[11]*1e6, P_mean/100, label = 'November')

plt.plot(q_std_yearly*1e6, P_mean/100, label = 'Annual', color = 'black')


plt.legend()

plt.ylim([1000,0.0001])
plt.yscale('log')

plt.ylabel('Pressure (hPa)')
plt.xlabel('Standard Deviation q (PPMv)')

plt.grid()

plt.savefig(os.path.join(fig_file, 'MLS_std_month.pdf'), format  = 'pdf', 
            dpi = 300, bbox_inches = 'tight')

#%%add dummy values to extend range

a = [P_mean[:],np.array(P_mean[-3:-1])]
P_combined = np.insert(P_mean, len(P_mean),  P_mean[-1]/10 )
P_combined = np.insert(P_combined, 0,  P_mean[0]*10 )

q_combined = np.insert(q_std_yearly, len(q_std_yearly),  q_std_yearly[-1] )
q_combined = np.insert(q_combined, 0,  q_std_yearly[0]*10 )


#%%
f = interp1d(P_combined, q_combined, kind='cubic',fill_value="extrapolate")

pressure_dense = np.logspace(np.log10(P_combined.min()), np.log10(P_combined.max()), 120)  # Add more points
data_interpolated = f(pressure_dense)

# Smoothing using Gaussian filter
data_smoothed = gaussian_filter1d(data_interpolated, sigma=2)

# Plotting
plt.figure(figsize=(6, 8))
plt.plot(q_std_yearly*1e6, P_mean/100,  'o-', label='MLS Data')
plt.plot(data_interpolated*1e6,pressure_dense/100, '--', label='Interpolated')
plt.plot(data_smoothed*1e6, pressure_dense/100,  '-', label='Smoothed')
plt.gca().invert_yaxis()  # Typically pressure decreases with height
plt.legend(fontsize = 12)
plt.yscale('log')
plt.xlim([0,15e-1])
plt.ylim([1e3,1e-4])

plt.ylabel("Pressure (hPa)", fontsize = 12)
plt.xlabel("q Err (PPMv)",fontsize = 12)
plt.grid()
plt.savefig(os.path.join(fig_file, 'a_priori.pdf'), format  = 'pdf', \
            dpi = 300, bbox_inches = 'tight')
plt.show()


#%%save file to csv

df = pd.DataFrame({
    str(len(pressure_dense)): pressure_dense[::-1],
    str(2): data_smoothed[::-1]
})

save_filename = os.path.join(data_file, 'prior_errors.aa')
with open(save_filename, 'w') as f:
    f.write("#Data extracted from yearly differences \
            \n#in MLS data \n1 \n")

#df.to_csv(save_filename, mode='a', index=False, sep=' ')


#%%save inflated file to csv

df = pd.DataFrame({
    str(len(pressure_dense)): pressure_dense[::-1],
    str(2): data_smoothed[::-1]*2
})

save_filename = os.path.join(data_file, 'prior_errors_double.aa')
with open(save_filename, 'w') as f:
    f.write("#Data extracted from yearly differences \
            \n#in MLS data \n1 \n")

#df.to_csv(save_filename, mode='a', index=False, sep=' ')
#%%
mean_smoothed  = np.mean(df['2'][df['120']<3000])
scaled = mean_smoothed+  .5 * (df['2'] -  mean_smoothed)
df['2'] = scaled

save_filename = os.path.join(data_file, 'prior_errors_double_scaled.aa')
with open(save_filename, 'w') as f:
    f.write("#Data extracted from yearly differences \
            \n#in MLS data \n1 \n")
#df.to_csv(save_filename, mode='a', index=False, sep=' ')
#%%

# Plotting
plt.figure(figsize=(6, 8))
plt.plot(q_std_yearly*1e6, P_mean/100,  'o-', label='MLS Data')
plt.plot(data_interpolated*1e6,pressure_dense/100, '--', label='Interpolated')
plt.plot(data_smoothed*1e6, pressure_dense/100,  '-', label='Smoothed')
plt.plot(scaled[::-1]*1e6, pressure_dense/100,  '-', label='Scaled')

plt.gca().invert_yaxis()  # Typically pressure decreases with height
plt.legend(fontsize = 12)
plt.yscale('log')
plt.xlim([0,15e-1])
plt.ylim([1e3,1e-4])

plt.ylabel("Pressure (hPa)", fontsize = 12)
plt.xlabel("q Err (PPMv)",fontsize = 12)
plt.grid()
plt.savefig(os.path.join(fig_file, 'a_priori_with_scaled.pdf'), format  = 'pdf', \
            dpi = 300, bbox_inches = 'tight')
plt.show()
