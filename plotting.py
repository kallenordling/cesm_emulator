import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from inference import predict_temperature_from_emissions
from utils import unstandardize


def calcmean(ds: xr.Dataset) -> xr.DataArray:
    """Area-weighted mean over latitude and longitude."""
    weights = np.cos(np.deg2rad(ds.lat))
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(("lon", "lat"))


def plot_emiss_vs_temp(
    predicted_pattern: str = "predicted/pred_TREFHT_*.nc",
    emission_path: str = "../CESM2-LESN_emulator/co2_final.nc",
    emission_var: str = "CO2_em_anthro",
    climate_path: str = "../CESM2-LESN_emulator/splits/fold_1/climate_data_train_fold1.nc",
    climate_var: str = "TREFHT",
    output: str = "cumulative_emiss.png",
) -> None:
    """Plot cumulative emissions against temperature."""
    temp_p = calcmean(xr.open_mfdataset(predicted_pattern)["TREFHT_pred"].load())
    temp_p = temp_p - temp_p.sel(year=slice(1850, 1900)).mean("year")

    emiss = (
        xr.open_dataset(emission_path)[emission_var]
        .isel(member_id=0)
        .sum(["lat", "lon"])
    )
    temp = xr.open_dataset(climate_path)[climate_var].isel(member_id=0)
    temp = calcmean(temp)
    temp = temp - temp.sel(year=slice(1850, 1900)).mean("year")

    plt.plot(emiss, temp, label="training set")
    plt.plot(emiss[0 : len(temp_p)], temp_p, label="predicted")
    plt.legend()
    plt.savefig(output)


def predict_cumulative_temperature(
    ckpt_path: str,
    cond_file: str,
    cond_var: str,
    target_file: str,
    target_var: str,
    out_dir: str = "predicted",
    year_range=range(1850, 2101),
) -> None:
    """Predict temperature for a range of years given emissions."""
    with xr.open_dataset(target_file) as ds:
        da = ds[target_var].load()
        t_mean = float(da.mean().values)
        t_std = float(da.std().values)

    for year in year_range:
        tmp = "cond_year.nc"
        out_path = f"{out_dir}/pred_TREFHT_{year}.nc"
        with xr.open_dataset(cond_file) as ds:
            ds_year = ds.sel(year=[year]).isel(member_id=[0])
            ds_year.to_netcdf(tmp)

        da = predict_temperature_from_emissions(
            ckpt_path=ckpt_path,
            cond_file=tmp,
            cond_var=cond_var,
            out_path=out_path,
            device="auto",
            batch_size=8,
            stack_dim="year",
            member_dim="member_id",
            lat_name="lat",
            lon_name="lon",
            normalize_cond=True,
            target_mean=None,
            target_std=None,
        )
        da = unstandardize(da, t_mean, t_std) - 273.1
        da.to_netcdf(out_path)
