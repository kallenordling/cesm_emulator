
"""
Inference helper: load model from checkpoint and predict temperature from emissions.
Assumes your training saved a checkpoint with "config" containing UNet/Diffusion settings.
"""

from typing import Optional, Tuple, Dict, Any
import os
import numpy as np
import torch
import xarray as xr

# Your project modules (must exist in PYTHONPATH / same folder)
from model import UNet, Diffusion

# -----------------------
# Loading helpers
# -----------------------

def _device_from_str(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _build_model_from_ckpt_config(cfg: Dict[str, Any], device: torch.device) -> Tuple[UNet, Diffusion]:
    unet_cfg = cfg.get("unet", {})
    train_cfg = cfg.get("train", {})
    unet = UNet(
        in_channels=unet_cfg.get("in_channels", 2),
        out_channels=unet_cfg.get("out_channels", 1),
        base_ch=unet_cfg.get("base_ch", 64),
        ch_mults=tuple(unet_cfg.get("ch_mults", (1, 2, 4))),
        num_res_blocks=unet_cfg.get("num_res_blocks", 2),
        time_dim=unet_cfg.get("time_dim", 256),
        groups=unet_cfg.get("groups", 8),
        dropout=unet_cfg.get("dropout", 0.0),
    ).to(device)

    diffusion = Diffusion(
        unet,
        img_channels=1,
        timesteps=train_cfg.get("timesteps", 1000),
        beta_schedule=train_cfg.get("beta_schedule", "linear"),
    ).to(device)
    return unet, diffusion

def load_diffusion_from_checkpoint(ckpt_path: str, device: str = "auto") -> Tuple[Diffusion, Dict[str, Any]]:
    """Load Diffusion+UNet from a training checkpoint.
    Returns (diffusion_model_in_eval_mode, config_dict).
    """
    dev = _device_from_str(device)
    ckpt = torch.load(ckpt_path, map_location=dev)
    cfg = ckpt.get("config", {})
    unet, diffusion = _build_model_from_ckpt_config(cfg, dev)
    print("target_stats:", ckpt.get("target_stats"))
    print("target_mean:", ckpt.get("target_mean"))
    print("target_std:",  ckpt.get("target_std"))
    # Model weights
    missing, unexpected = unet.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print("[UNet] missing keys:", missing)
        print("[UNet] unexpected keys:", unexpected)

    # Restore diffusion buffers (e.g., precomputed alphas) if present
    diff_state = diffusion.state_dict()
    diff_state.update(ckpt.get("diffusion_buffers", {}))
    diffusion.load_state_dict(diff_state, strict=False)

    diffusion.eval()
    for p in diffusion.parameters():
        p.requires_grad_(False)

    return diffusion, cfg

# -----------------------
# Data helpers
# -----------------------

def _order_hw_dims(da: xr.DataArray, lat_name: Optional[str], lon_name: Optional[str]) -> Tuple[str, str]:
    dims = list(da.dims)
    # prefer explicit names
    for cand in ((lat_name, lon_name), ("lat","lon"), ("y","x"), ("nlat","nlon")):
        if cand[0] in dims and cand[1] in dims:
            return cand[0], cand[1]
    # fallback: last two
    return dims[-2], dims[-1]

def _load_condition(
    cond_file: str,
    cond_var: str,
    stack_dim: str = "year",
    member_dim: str = "member_id",
    lat_name: Optional[str] = "lat",
    lon_name: Optional[str] = "lon",
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load emissions/condition field and return (cond_np, meta).
    cond_np shape: (T, M, 1, H, W). meta contains coords and dim names.
    """
    with xr.open_dataset(cond_file) as ds:
        da = ds[cond_var].load()
        Hname, Wname = _order_hw_dims(da, lat_name, lon_name)
        if stack_dim not in da.dims:
            raise ValueError(f"stack_dim '{stack_dim}' not found in {da.dims}")
        if member_dim not in da.dims:
            raise ValueError(f"member_dim '{member_dim}' not found in {da.dims}")
        da = da.transpose(stack_dim, member_dim, Hname, Wname)  # (T,M,H,W)

        cond_np = da.values.astype(np.float32)[:, :, None, :, :]  # (T,M,1,H,W)

        # Prefer coordinates from the parent dataset to preserve exact labels/attrs
        stack_coord = None
        member_coord = None
        stack_attrs = {}
        member_attrs = {}
        lat_attrs = {}
        lon_attrs = {}
        if stack_dim in ds:
            stack_coord = ds[stack_dim].values
            stack_attrs = dict(getattr(ds[stack_dim], "attrs", {}))
        elif stack_dim in da.coords:
            stack_coord = da.coords[stack_dim].values
        else:
            stack_coord = np.arange(da.sizes[stack_dim])

        if member_dim in ds:
            member_coord = ds[member_dim].values
            member_attrs = dict(getattr(ds[member_dim], "attrs", {}))
        elif member_dim in da.coords:
            member_coord = da.coords[member_dim].values
        else:
            member_coord = np.arange(da.sizes[member_dim])

        lat_vals, lon_vals = None, None
        if lat_name in ds:
            lat_vals = ds[lat_name].values
            lat_attrs = dict(getattr(ds[lat_name], "attrs", {}))
        if lon_name in ds:
            lon_vals = ds[lon_name].values
            lon_attrs = dict(getattr(ds[lon_name], "attrs", {}))

        meta = {
            "stack_dim": stack_dim,
            "member_dim": member_dim,
            "lat_name": lat_name,
            "lon_name": lon_name,
            "Hname": Hname,
            "Wname": Wname,
            "stack_coord": stack_coord,
            "member_coord": member_coord,
            "stack_attrs": stack_attrs,
            "member_attrs": member_attrs,
            "lat": lat_vals,
            "lon": lon_vals,
            "lat_attrs": lat_attrs,
            "lon_attrs": lon_attrs,
        }

    if normalize:
        c_mean = float(cond_np.mean())
        c_std  = float(cond_np.std() + 1e-8)
        print(f"[Cond] mean={c_mean:.4e} std={c_std:.4e}")
        cond_np = (cond_np - c_mean) / c_std
        meta["cond_mean"] = c_mean
        meta["cond_std"]  = c_std

    return cond_np, meta

# -----------------------
# Prediction
# -----------------------

@torch.no_grad()
def predict_temperature_from_emissions(
    ckpt_path: str,
    cond_file: str,
    cond_var: str,
    out_path: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 16,
    stack_dim: str = "year",
    member_dim: str = "member_id",
    lat_name: str = "lat",
    lon_name: str = "lon",
    normalize_cond: bool = True,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
) -> xr.DataArray:
    """Load a trained model and predict temperature from emission maps.

    Notes on units:
      - If the model was trained on *standardized* temperature, predictions here
        will be in standardized units as well. To get physical units (e.g., K or °C),
        pass the training target mean/std via `target_mean` and `target_std`.
      - If you don't have those, you can still use the standardized outputs for
        relative comparisons and mapping.

    Returns:
      xarray.DataArray with dims (stack_dim, member_dim, lat, lon)
    """
    dev = _device_from_str(device)
    torch.backends.cudnn.benchmark = True

    diffusion, cfg = load_diffusion_from_checkpoint(ckpt_path, device=device)

    # Load condition/emissions
    cond_np, meta = _load_condition(
        cond_file=cond_file,
        cond_var=cond_var,
        stack_dim=stack_dim,
        member_dim=member_dim,
        lat_name=lat_name,
        lon_name=lon_name,
        normalize=normalize_cond,
    )

    T, M, C, H, W = cond_np.shape
    N = T * M

    # Flatten over (T,M)
    cond_flat = cond_np.reshape(N, C, H, W)
    cond_tensor = torch.from_numpy(cond_flat).to(dev)

    preds = []
    for i in range(0, N, batch_size):
        c = cond_tensor[i:i+batch_size]
        # Diffusion sampling
        # The sampler may be stochastic; for deterministic output, set the RNG seed outside.
        y = diffusion.sample(c, shape=(c.size(0), 1, H, W), device=dev)
        preds.append(y.cpu())
    pred_tensor = torch.cat(preds, dim=0)  # (N,1,H,W)
    pred_np = pred_tensor.numpy().astype(np.float32).reshape(T, M, 1, H, W)  # (T,M,1,H,W)

    # Optional un-standardization to physical units
    if (target_mean is not None) and (target_std is not None):
        pred_np = pred_np * float(target_std) + float(target_mean)

    # Build proper coordinate DataArrays to preserve labels and attrs
    stack_coord_da = xr.DataArray(
        meta["stack_coord"],
        dims=(stack_dim,),
        name=stack_dim,
        attrs=meta.get("stack_attrs", {}),
    )
    member_coord_da = xr.DataArray(
        meta["member_coord"],
        dims=(member_dim,),
        name=member_dim,
        attrs=meta.get("member_attrs", {}),
    )
    if meta.get("lat") is not None:
        lat_coord_da = xr.DataArray(meta["lat"], dims=(lat_name,), name=lat_name, attrs=meta.get("lat_attrs", {}))
    else:
        lat_coord_da = xr.DataArray(np.arange(H), dims=(lat_name,), name=lat_name)
    if meta.get("lon") is not None:
        lon_coord_da = xr.DataArray(meta["lon"], dims=(lon_name,), name=lon_name, attrs=meta.get("lon_attrs", {}))
    else:
        lon_coord_da = xr.DataArray(np.arange(W), dims=(lon_name,), name=lon_name)

    da_pred = xr.DataArray(
        pred_np[:, :, 0, :, :],
        dims=(stack_dim, member_dim, lat_name, lon_name),
        coords={
            stack_dim: stack_coord_da,
            member_dim: member_coord_da,
            lat_name: lat_coord_da,
            lon_name: lon_coord_da,
        },
        name="TREFHT_pred",
        attrs={
            "description": "Predicted near-surface air temperature from emissions via diffusion model",
            "units": "standardized" if (target_mean is None or target_std is None) else "K (or training target units)",
            "checkpoint": os.path.abspath(ckpt_path),
            "cond_file": os.path.abspath(cond_file),
            "cond_var": cond_var,
        },
    )

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        da_pred.to_netcdf(out_path)
        print(f"[Saved] {out_path}")

    return da_pred

# -----------------------
# CLI
# -----------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Predict temperature from emissions using a trained diffusion model.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    p.add_argument("--cond_file", required=True, help="NetCDF with emission maps")
    p.add_argument("--cond_var", required=True, help="Variable name for emissions in cond_file")
    p.add_argument("--out", default=None, help="Output NetCDF path (optional)")
    p.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--stack_dim", default="year")
    p.add_argument("--member_dim", default="member_id")
    p.add_argument("--lat_name", default="lat")
    p.add_argument("--lon_name", default="lon")
    p.add_argument("--no_norm", action="store_true", help="Disable condition normalization")
    p.add_argument("--target_mean", type=float, default=None)
    p.add_argument("--target_std", type=float, default=None)
    args = p.parse_args()

    da = predict_temperature_from_emissions(
        ckpt_path=args.ckpt,
        cond_file=args.cond_file,
        cond_var=args.cond_var,
        out_path=args.out,
        device=args.device,
        batch_size=args.batch_size,
        stack_dim=args.stack_dim,
        member_dim=args.member_dim,
        lat_name=args.lat_name,
        lon_name=args.lon_name,
        normalize_cond=(not args.no_norm),
        target_mean=args.target_mean,
        target_std=args.target_std,
    )
    print(da)

if __name__ == "__main__":
    _cli()
