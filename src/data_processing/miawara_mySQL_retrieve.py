#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:11:27 2023

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

import numpy as np
import mysql.connector as mysql
import paramiko
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


host = "juno.mw.iap.unibe.ch"
port = 22
username = "miawara"
password = "miawara-pw"

mydb = mysql.connect(
  host=host,
  user=username,
  password=password,
  database = 'MIAWARA'
)

#%%
host = "juno.mw.iap.unibe.ch"
port = 22
username = "miawara"
password = "miawara-pw"

#%%

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

msql_cmd = 'SELECT timestamp FROM level2_header WHERE profile_ID <=2000;'
base_command = 'mysql MIAWARA -e "{}"'.format(msql_cmd)

stdin, stdout, stderr = ssh.exec_command(base_command)
dates = stdout.readlines()
print(dates[:5])

#%%
datetimes = []
for a in dates[1:]:
    b = dt.datetime(int(a[:4]), int(a[5:7]), int(a[8:10]), int(a[11:13]), int(a[14:16]), int(a[17:18]) )
    datetimes.append(b)
    
#%%

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

msql_cmd = 'SELECT profile_ID FROM level2_header WHERE profile_ID >= 50000 AND profile_ID <=440000;'
base_command = 'mysql MIAWARA -e "{}"'.format(msql_cmd)

stdin, stdout, stderr = ssh.exec_command(base_command)
lines = stdout.readlines()
print(dates[:5])
prof_id = []


for i in lines[:]:
    if i == 'profile_ID\n':
        print( 'string')
    else:
        #print(i)
        prof_id.append(float(i[:]))
#%%      

time_indx_dataframe = pd.DataFrame({'Datetime': datetimes, 'profile_ID': prof_id })

#%%
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

msql_cmd = 'SELECT pressure FROM level2_profile WHERE profile_ID >= 140000 AND profile_ID <=225000;'
base_command = 'mysql MIAWARA -e "{}"'.format(msql_cmd)
stdin, stdout, stderr = ssh.exec_command(base_command)
lines = stdout.readlines()
errlines =  stderr.readlines()
pressures  = []

for i in lines[:]:
    if i == 'pressure\n':
        print( 'string')
    else:
        #print(i)
        pressures.append(float(i[:]))
        
pressures = np.array(pressures)
#%%        
        
pressures_1 = np.array(pressures[:635000])
length = int((len(pressures_1))/50)
pressures_1 = pressures_1.reshape(length,50)


#%%
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

msql_cmd = 'SELECT profile_ID FROM level2_profile WHERE profile_ID >= 140000 AND profile_ID <=225000;'
base_command = 'mysql MIAWARA -e "{}"'.format(msql_cmd)
stdin, stdout, stderr = ssh.exec_command(base_command)
lines = stdout.readlines()
errlines =  stderr.readlines()
profiles  = []

for i in lines[:]:
    if i == 'profile_ID\n':
        print( 'string')
    else:
        #print(i)
        profiles.append(float(i[:]))

profiles = np.array(profiles)
#%%
profiles_1 = np.array(profiles[:635000])
length = int((len(profiles_1))/50)
profiles_1 = profiles_1.reshape(length,50)

#%%
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

msql_cmd = 'SELECT vmr FROM level2_profile WHERE profile_ID>= 140000 AND profile_ID <=225000;'
base_command = 'mysql MIAWARA -e "{}"'.format(msql_cmd)
stdin, stdout, stderr = ssh.exec_command(base_command)
lines = stdout.readlines()
errlines =  stderr.readlines()
vmr  = []

for i in lines[:]:
    if i == 'vmr\n':
        print( 'string')
    else:
        #print(i)
        vmr.append(float(i[:]))
        
vmr = np.array(vmr)

#%%
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

msql_cmd = 'SELECT temperature FROM level2_profile WHERE profile_ID>= 140000 AND profile_ID <=225000;'
base_command = 'mysql MIAWARA -e "{}"'.format(msql_cmd)
stdin, stdout, stderr = ssh.exec_command(base_command)
lines = stdout.readlines()
errlines =  stderr.readlines()
temperatures  = []


for i in lines[:]:
    if i == 'temperature\n':
        print( 'string')
    else:
        #print(i)
        temperatures.append(float(i[:]))
        
temperatures = np.array(temperatures)


#%%        
idx_100hPa = np.where((pressures > 1048) & (pressures < 1049))
vmr_prof_id = profiles[list(idx_100hPa)]
vm_110hPa = vmr[list(idx_100hPa)]
tmps_110hpa = temperatures[list(idx_100hPa)]
pres_110hPa = pressures[list(idx_100hPa)]

#%%        
vmr_1 = np.array(vmr[:635000])
length = int((len(vmr_1))/50)
vmr_1 = vmr_1.reshape(length,50)

#%%
df2 = pd.DataFrame({'profile_ID':vmr_prof_id[:], 'VMR_110_hPa': vm_110hPa[:], 'temperatures_110_hpa':tmps_110hpa, 'pressure':  pres_110hPa })

main_df = pd.merge(time_indx_dataframe, df2, on = 'profile_ID', how = 'outer')
main_df2 = main_df[(main_df.Datetime > dt.datetime(2014,1,1)) & ( main_df.Datetime < dt.datetime(2015,1,1))]
main_df2 = main_df2.sort_values('Datetime')
tmp = main_df2.Datetime
mylist = [i.date() for i in main_df2.Datetime]
main_df2['day'] = np.array(mylist)
main_df2_sortby = main_df2.groupby('day').mean().reset_index()

prof_max = main_df2.profile_ID.max()
prof_min = main_df2.profile_ID.min()

#%%
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(main_df2_sortby.day, main_df2_sortby.VMR_110_hPa*1000000, linewidth = 1, color = 'red', label = 'Water Vapour')
#ax2 = ax1.twinx()
#ax2.plot(main_df2_sortby.day, main_df2_sortby.temperatures_110_hpa, linewidth = 1, label = 'Temperature')
ax1.set_ylabel('Water Vapour Mixing Ratio (PPMV)')
#ax2.set_ylabel('Temperature (K)')
#fig.legend(loc = 'upper right', bbox_to_anchor=(0.9, 0.85))
plt.title('Water Vapour Retrieval at 10hPa from MIAWARA')
plt.savefig('/home/alistair/plots/Miawara/old_retrievals/water_vapour_2014.png', dpi= 300)


