
import os
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import xarray as xr
from scipy.ndimage import gaussian_filter
import csv
from collections import deque

# ---------------- I/O and logging ----------------

class LossLogger:
    def __init__(self, path: str, smooth: int = 100):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", newline="")
        self.writer = csv.writer(self.fh)
        if os.stat(path).st_size == 0:
            self.writer.writerow(["epoch", "step", "loss", "loss_smooth"])
        self.buf = deque(maxlen=smooth)

    def log(self, epoch: int, step: int, loss: float):
        self.buf.append(float(loss))
        sm = sum(self.buf) / len(self.buf)
        self.writer.writerow([epoch, step, float(loss), sm])
        self.fh.flush()

    def close(self):
        self.fh.close()

# ---------------- Distributed helpers ----------------

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    else:
        # single-process fallback
        pass

def get_diff_mod(model):
    from torch.nn.parallel import DistributedDataParallel as DDP
    return model.module if isinstance(model, DDP) else model

# ---------------- Array / tensor utilities ----------------

def _minmax01(img: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-image min-max to [0,1].
    img: (B,1,H,W) or (1,H,W)
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B = img.size(0)
    flat_min = img.view(B, -1).min(dim=1).values.view(B, 1, 1, 1)
    flat_max = img.view(B, -1).max(dim=1).values.view(B, 1, 1, 1)
    return (img - flat_min) / (flat_max - flat_min + eps)

def gaussian_smooth_latlon(da: xr.DataArray, sigma_lat: float = 1.5, sigma_lon: float = 1.5) -> xr.DataArray:
    # Work on a copy; assume last two dims are lat, lon
    assert da.dims[-2:] == ("lat","lon")
    pad = 3 * int(np.ceil(max(sigma_lat, sigma_lon)))
    da_wrap = xr.concat([da.isel(lon=slice(-pad,None)), da, da.isel(lon=slice(0,pad))], dim="lon")
    arr = gaussian_filter(da_wrap.values, sigma=[0]*(da_wrap.ndim-2)+[sigma_lat, sigma_lon], mode="nearest")
    out = da_wrap.copy(data=arr)
    out = out.isel(lon=slice(pad, pad+da.sizes["lon"]))
    return out

def _order_hw_dims(da: xr.DataArray,
                   y_name: Optional[str], x_name: Optional[str],
                   lat_name: Optional[str], lon_name: Optional[str]) -> List[str]:
    dims = list(da.dims)
    if y_name in dims and x_name in dims:
        return [y_name, x_name]  # type: ignore[list-item]
    if lat_name in dims and lon_name in dims:
        return [lat_name, lon_name]  # type: ignore[list-item]
    for cand in (("y", "x"), ("lat", "lon"), ("nlat", "nlon")):
        if all(c in dims for c in cand):
            return list(cand)
    return dims[-2:]

def _ensure_hw_like(da: xr.DataArray, expect_leading: int,
                    y_name: Optional[str], x_name: Optional[str],
                    lat_name: Optional[str], lon_name: Optional[str]) -> xr.DataArray:
    hw = _order_hw_dims(da, y_name, x_name, lat_name, lon_name)
    leading = [d for d in da.dims if d not in hw]
    if len(leading) > expect_leading:
        for d in leading[expect_leading:]:
            if da.sizes[d] == 1:
                da = da.squeeze(d, drop=True)
        hw = _order_hw_dims(da, y_name, x_name, lat_name, lon_name)
        leading = [d for d in da.dims if d not in hw]
        if len(leading) != expect_leading:
            leading = leading[:expect_leading]
    return da.transpose(*leading, *hw)

def _move_to_stack_hw(da: xr.DataArray,
                      stack_dim: str,
                      y_name: Optional[str], x_name: Optional[str],
                      lat_name: Optional[str], lon_name: Optional[str],
                      expect_leading: int) -> xr.DataArray:
    if stack_dim not in da.dims:
        raise ValueError(f"Expected stack_dim '{stack_dim}' in {da.dims}")
    da = da.transpose(stack_dim, ...)
    return _ensure_hw_like(da, expect_leading=expect_leading,
                           y_name=y_name, x_name=x_name,
                           lat_name=lat_name, lon_name=lon_name)

def _find_member_dim(da: xr.DataArray, hint: Optional[str]) -> Optional[str]:
    if hint and hint in da.dims:
        return hint
    for cand in ("member", "member_id", "members", "ens", "ensemble", "realization", "realisation"):
        if cand in da.dims:
            return cand
    for d in da.dims:
        if da.sizes[d] == 34:
            return d
    spatial_names = {"y", "x", "lat", "lon", "latitude", "longitude", "nlat", "nlon"}
    for d in da.dims:
        if d not in spatial_names and d.lower() not in ("time", "year"):
            if da.sizes[d] <= 128:
                return d
    return None
    
def unstandardize(x, mean, std, *, units=None, name=None, attrs_update=None):
    """
    Reverse standardization: y = x * std + mean.
    Supports torch.Tensor, numpy.ndarray, and xarray.DataArray.
    """
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None
    try:
        import xarray as _xr
    except Exception:
        _xr = None

    if _torch is not None and isinstance(x, _torch.Tensor):
        mean_t = _torch.as_tensor(mean, dtype=x.dtype, device=x.device)
        std_t  = _torch.as_tensor(std, dtype=x.dtype, device=x.device)
        y = x * std_t + mean_t
        return y

    if isinstance(x, _np.ndarray):
        return _np.asarray(x) * _np.asarray(std) + _np.asarray(mean)

    if _xr is not None and isinstance(x, _xr.DataArray):
        y = x * std + mean
        if units is not None:
            y.attrs = dict(getattr(y, "attrs", {}))
            y.attrs["units"] = units
        if attrs_update:
            y.attrs.update(attrs_update)
        if name is not None:
            y.name = name
        return y

    return x * std + mean
def load_cond_and_target(
    cond_file: str,
    cond_var: str,
    target_file: str,
    target_var: str,
    stack_dim: str = "year",
    member_dim: Optional[str] = "member_id",
    y_name: Optional[str] = None,
    x_name: Optional[str] = None,
    lat_name: Optional[str] = None,
    lon_name: Optional[str] = None,
    normalize: bool = True,
):
    """
    Returns:
      cond_np: (T, M, 1, H, W)
      tgt_np:  (T, M, 1, H, W)  # (we add channel dim)
      time_ids: (T,) int64 indices 0..T-1
    """
    # --- load condition ---
    with xr.open_dataset(cond_file) as ds_c:
        da_c = ds_c[cond_var].load()
        Hname, Wname = _order_hw_dims(da_c, y_name, x_name, lat_name, lon_name)
        if member_dim not in da_c.dims:
            raise ValueError(f"Condition var lacks member dim '{member_dim}'. Found dims={da_c.dims}")
        da_c = da_c.transpose(stack_dim, member_dim, Hname, Wname)
        cond_np = da_c.values.astype(np.float32)                # (T, M, H, W)
        cond_np = cond_np[:, :, None, :, :]                     # (T, M, 1, H, W)
        time_ids = np.arange(cond_np.shape[0], dtype=np.int64)

    # --- load target ---
    with xr.open_dataset(target_file) as ds_t:
        da_t = ds_t[target_var].load()
        Hname, Wname = _order_hw_dims(da_t, y_name, x_name, lat_name, lon_name)
        if member_dim not in da_t.dims:
            raise ValueError(f"Target var lacks member dim '{member_dim}'. Found dims={da_t.dims}")
        da_t = da_t.transpose(stack_dim, member_dim, Hname, Wname)  # (T,M,H,W)
        tgt_np = da_t.values.astype(np.float32)
        tgt_np = tgt_np[:, :, None, :, :]                            # (T, M, 1, H, W)

    if normalize:
        c_mean, c_std = float(cond_np.mean()), float(cond_np.std() + 1e-8)
        t_mean, t_std = float(tgt_np.mean()), float(tgt_np.std() + 1e-8)
        print(f"[Cond raw] mean={c_mean:.4e} std={c_std:.4e}")
        print(f"[Tgt  raw] mean={t_mean:.4e} std={t_std:.4e}")
        cond_np = (cond_np - c_mean) / c_std
        tgt_np = (tgt_np - t_mean) / t_std
        print(f"[Cond norm] mean={cond_np.mean():.3e} std={cond_np.std():.3e}")
        print(f"[Tgt  norm] mean={tgt_np.mean():.3e} std={tgt_np.std():.3e}")

    return cond_np, tgt_np, time_ids

def pick_mid_t(T: int) -> int:
    return T // 2
