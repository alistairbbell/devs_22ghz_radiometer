#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:01:29 2024

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import num2date
from datetime import datetime
import os
from matplotlib.colors import ListedColormap
#%%

#inpath = '../../data/interim/gromos_concat_H2O_2022_2024_rescaled_ndacc_v2.nc'
inpath = '//storage/atmosphere/instruments/gromos/level2/GROMORA/v3'

infilenames = ['GROMOS_2023_06_30_AC240_v21.nc',
               'GROMOS_2023_12_31_AC240_v21.nc']

#%%

# Load the NetCDF file
dataset = nc.Dataset(os.path.join(inpath, infilenames[0]), mode='r')

qc = dataset.variables['retrieval_quality'][:]
# Read variables
o3_x = dataset.variables['o3_x'][:]  # Ozone profile (time, o3_p)
oem_diagnostics = dataset.variables['oem_diagnostics'][:]  # OEM diagnostics (time, oem_diagnostics_idx)
diagnostic_value_0 = oem_diagnostics[:, 0]  # First diagnostic value
o3_p = dataset.variables['o3_p'][:]  # Pressure levels for ozone retrieval (o3_p)
time = dataset.variables['time'][:]  # Time variable

# Convert time to datetime using num2date
time_units = dataset.variables['time'].units
time_calendar = dataset.variables['time'].calendar
time_cftime = num2date(time, units=time_units, calendar=time_calendar)

# Convert cftime to Python datetime
time_dt = [datetime.fromisoformat(str(t)) for t in time_cftime]

# Filter profiles where diagnosticValue0 = 0
valid_profiles = diagnostic_value_0 == 0
o3_x_filtered = np.where(valid_profiles[:, None], o3_x, np.nan)

# Define colormap limits
min_val = 0  # Minimum value for colormap
max_val = 12e-6  # Maximum value for colormap
diff = (max_val - min_val) / 20  # Interval for levels

#%%
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.get_cmap('viridis')
cmap.set_under('midnightblue')
cmap.set_over('gold')

# Plot the colormap
plotting_data = o3_x_filtered# * 1e6  # Convert to ppmv for better readability
plotting_data = np.where(plotting_data < min_val, min_val, plotting_data)
plotting_data = np.where(plotting_data > max_val, max_val, plotting_data)

contour = ax.contourf(
    time_dt,
    o3_p / 100,  # Convert pressure to hPa
    plotting_data.T,  # Transpose for proper orientation
    levels=np.arange(min_val, max_val + diff, diff),
    cmap=cmap,
    extend='both'
)

for i, date in enumerate(time_dt):
    date_np64 = np.datetime64(date)  # Convert datetime to numpy.datetime64
    if qc[i] == 0:
        ax.axvspan(date_np64, date_np64 + np.timedelta64(1, 'h'), color='grey', alpha=1.0)



# Add colorbar
cbar = plt.colorbar(contour, ticks=np.arange(0, 14e-6, 2e-6))
cbar.set_label('Ozone Mixing Ratio (PPMV)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Configure axes
ax.invert_yaxis()  # Pressure decreases with altitude
ax.set_yscale('log')
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Pressure (hPa)", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# Show the plot
plt.title("Ozone Mixing Ratio Profiles", fontsize=16)
plt.ylim([100,0.5])
plt.xlim([datetime(2023,7,1), datetime(2024,1,1)])

plt.savefig('/home/alistair/other_radiometers/output/plots/gromos_qc_consolodated_2023_01_to_2023_07.png')

plt.tight_layout()
plt.show()


#%%

# Close the dataset
dataset.close()