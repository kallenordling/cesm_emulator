import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from inference import predict_temperature_from_emissions
from utils import unstandardize

def calcmean(ds):
	weights = np.cos(np.deg2rad(ds.lat))
	ds_weighted = ds.weighted(weights)
	return ds_weighted.mean(("lon", "lat"))	


import xarray as xr
from inference import predict_temperature_from_emissions

CKPT = "runs/exp3/checkpoints/ckpt_epoch_1890.pt"
COND = "../CESM2-LESN_emulator/co2_final.nc"                 # your emissions file
COND_VAR = "CO2_em_anthro"
YEAR = 2010                           # <- pick the year you want
TMP = "cond_year.nc"        # temp file for the one-year subset
OUT = "predicted/pred_TREFHT_2010.nc" # optional output
TARGET_FILE = "../CESM2-LESN_emulator/splits/fold_1/climate_data_train_fold1.nc"  # use the file used for that checkpoint
TARGET_VAR  = "TREFHT"

with xr.open_dataset(TARGET_FILE) as ds:
    da = ds[TARGET_VAR].load()  # dims like (year, member_id, lat, lon)
    t_mean = float(da.mean().values)
    t_std  = float(da.std().values)
for YEAR in range(1850,2101):
	OUT = "predicted/pred_TREFHT_"+str(YEAR)+".nc" # optional output
	print("mean:", t_mean, "std:", t_std)
	# 1) Make a one-year subset (adjust dim names if needed)
	with xr.open_dataset(COND) as ds:
	    ds_year = ds.sel(year=[YEAR]).isel(member_id=[0])           # or: ds.isel(year=index)
	    # (optional) pick one member too: ds_year = ds_year.sel(member_id=0)
	    ds_year.to_netcdf(TMP)

	# 2) Predict temperature for that single year (and all members kept in TMP)
	da = predict_temperature_from_emissions(
	    ckpt_path=CKPT,
	    cond_file=TMP,
	    cond_var=COND_VAR,
	    out_path=OUT,            # or None to skip writing
	    device="auto",
	    batch_size=8,
	    stack_dim="year",
	    member_dim="member_id",
	    lat_name="lat",
	    lon_name="lon",
	    normalize_cond=True,
	    # set these if you want physical units (K/°C) instead of standardized:
	    target_mean=None,
	    target_std=None,
	)
	da=unstandardize(da,t_mean,t_std)-273.1
	print(da,t_mean,t_std) 
	da.to_netcdf(OUT)   
'''

emiss= xr.open_dataset("co2_final.nc")['CO2_em_anthro'].mean('member_id').sum(['lat','lon'])
temp = xr.open_dataset('splits/fold_1/climate_data_train_fold1.nc')["TREFHT"].mean('member_id')
temp = calcmean(temp)
temp = temp - temp.sel(year=slice(1850,1900)).mean('year')
print("#### emiss #####")
print(emiss)
print("###### TEMP #######")
print(temp)
plt.plot(emiss,temp)
plt.savefig('cumulative_emiss.png')
'''
