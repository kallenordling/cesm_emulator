import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

def calcmean(ds):
	weights = np.cos(np.deg2rad(ds.lat))
	ds_weighted = ds.weighted(weights)
	return ds_weighted.mean(("lon", "lat"))	


temp_p = calcmean(xr.open_mfdataset("predicted/pred_TREFHT_*.nc"))['TREFHT_pred'].load()
temp_p = temp_p - temp_p.sel(year=slice(1850,1900)).mean('year')
print("#### PREDICTED #####")
print(temp_p.isel(member_id=0))


emiss= xr.open_dataset("../CESM2-LESN_emulator/co2_final.nc")['CO2_em_anthro'].mean('member_id').sum(['lat','lon'])
temp = xr.open_dataset('../CESM2-LESN_emulator/splits/fold_1/climate_data_train_fold1.nc')["TREFHT"].mean('member_id')
temp = calcmean(temp)
temp = temp - temp.sel(year=slice(1850,1900)).mean('year')
print("#### emiss #####")
print(emiss)
print("###### TEMP #######")
print(temp)
plt.plot(emiss,temp,label="training set")
plt.plot(emiss[0:len(temp_p)],temp_p,label="predicted")
plt.savefig('cumulative_emiss.png')
