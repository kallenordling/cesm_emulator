import os
import json
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import xarray as xr
from torch.utils.data.distributed import DistributedSampler
from scipy.ndimage import gaussian_filter
import csv
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from dataset_single_member import WindowedAllMembersDataset
from dataset_single_member import WindowedAllMembersDataset_random
from utils_conf import load_config, apply_overrides
import json, os, pathlib, argparse
from functools import partial
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl

# --- NEW: optional backends (FSDP / DeepSpeed) ---
import contextlib
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        CPUOffload,
        StateDictType,
        FullStateDictConfig,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    _HAS_FSDP = True
except Exception:
    _HAS_FSDP = False

try:
    import deepspeed
    _HAS_DEEPSPEED = True
except Exception:
    _HAS_DEEPSPEED = False


class LossLogger:
    def __init__(self, path, smooth=100):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", newline="")
        self.writer = csv.writer(self.fh)
        if os.stat(path).st_size == 0:
            self.writer.writerow(["epoch", "step", "loss", "loss_smooth"])
        self.buf = deque(maxlen=smooth)

    def log(self, epoch, step, loss):
        self.buf.append(float(loss))
        sm = sum(self.buf) / len(self.buf)
        self.writer.writerow([epoch, step, float(loss), sm])
        self.fh.flush()

    def close(self):
        self.fh.close()


class MetricLogger:
    def __init__(self, path, smooth=100):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", newline="")
        self.writer = csv.writer(self.fh)
        self.buf_total = deque(maxlen=smooth)
        if not os.path.exists(path) or os.stat(path).st_size == 0:
            self.writer.writerow(["epoch","step","mse_raw","mse_lat","cond_loss","total","total_smooth"])
    def log(self, epoch, step, mse_raw, mse_lat, cond_loss, total):
        self.buf_total.append(float(total))
        sm = sum(self.buf_total) / len(self.buf_total)
        self.writer.writerow([epoch, step, float(mse_raw), float(mse_lat), float(cond_loss), float(total), sm])
        self.fh.flush()


def _latitude_weights(height: int, device: torch.device):
    lat = torch.linspace(-90, 90, steps=height, device=device)
    w = torch.cos(torch.deg2rad(lat)).clamp_min(0)
    return w / (w.mean() + 1e-8)


def _area_weighted_mean(tensor: torch.Tensor) -> torch.Tensor:
    B, C, H, W = tensor.shape
    w = _latitude_weights(H, tensor.device).view(1, 1, H, 1)
    return (tensor * w).mean(dim=(-2, -1))


def apply_act_ckpt_to_unet(unet):
    """
    Recursively wraps heavy compute blocks in activation checkpointing.
    Edit the `TARGETS` list to match your block class names.
    """
    # Put your block class names here (strings are fine)
    TARGETS = {
        "ResBlock", "ResidualBlock", "ConvBlock",
        "AttentionBlock", "SelfAttention2d", "CrossAttention",
        "DownBlock", "UpBlock", "MidBlock"
    }

    for name, m in list(unet.named_children()):
        cls_name = m.__class__.__name__
        # Heuristic: checkpoint modules that (a) are in TARGETS or (b) have params and are not containers
        is_target = (cls_name in TARGETS)
        has_params = any(p.requires_grad or True for p in m.parameters(recurse=False))
        has_children = any(True for _ in m.children())

        if is_target or (has_params and not has_children):
            wrapped = checkpoint_wrapper(
                m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
            setattr(unet, name, wrapped)
        else:
            apply_act_ckpt_to_unet(m)  # recurse

# flag for torchvision availability (for fast grids/saving)
try:
    import torchvision  # noqa: F401
    _HAS_TV = True
except Exception:
    _HAS_TV = False

from model import UNet, Diffusion
from dataset_single_member import SingleMemberDataset, AllMembersDataset

@torch.no_grad()
def pick_mid_t(T):
    return T // 2

@torch.no_grad()
def counterfactual_delta(diff_mod, cond,scale_mask=None, scale=1.1, steps=None):
    device = cond.device
    B, _, H, W = cond.shape
    if steps is None:
        steps = diff_mod.T
    base = diff_mod.sample(cond, shape=(B,1,H,W), device=device)
    cond2 = cond.clone()
    if scale_mask is not None:
        cond2 = cond2 * (1 + (scale-1) * scale_mask)
    else:
        cond2 = cond2 * scale
    cf = diff_mod.sample(cond2, shape=(B,1,H,W), device=device,)
    return cf - base

def saliency_wrt_cond(diff_mod, x0, cond, years=None, t=None):
    device = x0.device
    B = x0.size(0)
    if t is None:
        t = torch.full((B,), diff_mod.T // 2, device=device, dtype=torch.long)
    cond_req = cond.detach().clone().requires_grad_(True)
    x_t, noise = diff_mod.q_sample(x0, t)
    eps_pred = diff_mod.model(x_t, cond_req, t, years=years)
    loss = F.mse_loss(eps_pred, noise)
    g = torch.autograd.grad(loss, cond_req, retain_graph=False, create_graph=False)[0].detach().abs()
    g = g / (g.amax(dim=(2,3), keepdim=True) + 1e-8)
    return g

def gaussian_smooth_latlon(da: xr.DataArray, sigma_lat=1.5, sigma_lon=1.5):
    assert da.dims[-2:] == ("lat","lon")
    pad = 3 * int(np.ceil(max(sigma_lat, sigma_lon)))
    da_wrap = xr.concat([da.isel(lon=slice(-pad,None)),
                         da,
                         da.isel(lon=slice(0,pad))], dim="lon")
    arr = gaussian_filter(da_wrap.values, sigma=[0]*(da_wrap.ndim-2)+[sigma_lat, sigma_lon], mode="nearest")
    out = da_wrap.copy(data=arr)
    out = out.isel(lon=slice(pad, pad+da.sizes["lon"]))
    return out

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def barrier():
    if is_dist():
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        try:
            dist.barrier(device_ids=[local_rank])
        except TypeError:
            dist.barrier()


def setup_distributed():
    import os
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not dist.is_available() or world == 1:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=30),
    )


def _minmax01(img: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B = img.size(0)
    flat_min = img.view(B, -1).min(dim=1).values.view(B, 1, 1, 1)
    flat_max = img.view(B, -1).max(dim=1).values.view(B, 1, 1, 1)
    return (img - flat_min) / (flat_max - flat_min + eps)

def _order_hw_dims(da: xr.DataArray,
                   y_name: str | None, x_name: str | None,
                   lat_name: str | None, lon_name: str | None) -> list[str]:
    dims = list(da.dims)
    if y_name in dims and x_name in dims:
        return [y_name, x_name]
    if lat_name in dims and lon_name in dims:
        return [lat_name, lon_name]
    for cand in (("y", "x"), ("lat", "lon"), ("nlat", "nlon")):
        if all(c in dims for c in cand):
            return list(cand)
    return dims[-2:]

import torch
from torch.nn import functional as F

def saliency_wrt_cond(diff_mod, x0, cond, t=None):
    device = x0.device
    B = x0.size(0)
    if t is None:
        t = torch.full((B,), diff_mod.T // 2, device=device, dtype=torch.long)
    cond_req = cond.detach().clone().requires_grad_(True)
    x_t, noise = diff_mod.q_sample(x0, t)
    eps_pred = diff_mod.model(x_t, cond_req, t)
    loss = F.mse_loss(eps_pred, noise)
    loss.backward()
    g = cond_req.grad.detach().abs()
    g = g / (g.amax(dim=(2,3), keepdim=True) + 1e-8)
    return g

@torch.no_grad()
def quad_with_saliency(
    diffusion,
    cond_batch: torch.Tensor,
    truth_batch: torch.Tensor,
    path: str,
    device: torch.device,
    return_tensor: bool = False,
):
    diff_mod = get_diff_mod(diffusion)
    cond = cond_batch.to(device)
    truth = truth_batch.to(device)
    B, _, H, W = cond.shape
    pred = diff_mod.sample(cond, shape=(B, 1, H, W), device=device)
    with torch.enable_grad():
        sal = saliency_wrt_cond(diff_mod, x0=truth.detach(), cond=cond.detach())
    cond_v  = _minmax01(cond).cpu()
    truth_v = _minmax01(truth).cpu()
    pred_v  = _minmax01(pred).cpu()
    sal_v   = _minmax01(sal).cpu()
    titles = ["Condition", "Truth", "Prediction", "Saliency"]
    title_bar = 22
    panels = []
    for i in range(B):
        imgs = [TF.to_pil_image(cond_v[i]), TF.to_pil_image(truth_v[i]),
                TF.to_pil_image(pred_v[i]), TF.to_pil_image(sal_v[i])]
        titled = []
        for img, title in zip(imgs, titles):
            canvas = Image.new("RGB", (img.width, img.height + title_bar), color=(255, 255, 255))
            canvas.paste(img, (0, title_bar))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 2), title, fill=(0, 0, 0))
            titled.append(canvas)
        total_w = sum(im.width for im in titled)
        concat_img = Image.new("RGB", (total_w, titled[0].height), color=(255, 255, 255))
        x = 0
        for im in titled:
            concat_img.paste(im, (x, 0))
            x += im.width
        panels.append(concat_img)
    grid_w = max(p.width for p in panels)
    grid_h = sum(p.height for p in panels)
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    y = 0
    for pimg in panels:
        grid.paste(pimg, (0, y))
        y += pimg.height
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid.save(path)

# --------------- Counterfactual helpers ----------------

def _read_lat_lon_from_file(path, lat_name="lat", lon_name="lon"):
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            lat = ds[lat_name].values
            lon = ds[lon_name].values
        return lat, lon
    except Exception as e:
        print(f"[CF] Warning: could not read lat/lon from {path}: {e}")
        return None, None

def _box_mask_from_coords(lat_vec, lon_vec, lat_min, lat_max, lon_min, lon_max, device, H=None, W=None):
    if lat_vec is not None and lon_vec is not None:
        lat = np.asarray(lat_vec)
        lon = np.asarray(lon_vec)
        lat_sel = (lat >= lat_min) & (lat <= lat_max)
        Lmin, Lmax = lon_min % 360, lon_max % 360
        lon360 = np.mod(lon, 360)
        if Lmin <= Lmax:
            lon_sel = (lon360 >= Lmin) & (lon360 <= Lmax)
        else:
            lon_sel = (lon360 >= Lmin) | (lon360 <= Lmax)
        H2, W2 = len(lat), len(lon)
        m = np.zeros((H2, W2), dtype=np.float32)
        m[np.ix_(lat_sel, lon_sel)] = 1.0
    else:
        if H is None or W is None:
            return None
        m = np.ones((H, W), dtype=np.float32)
    t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device)
    return t

def _sample_once(c):
    print('sample once')
    return diff_mod.sample(c, shape=(B,1,H,W), device=device)

@torch.no_grad()
def counterfactual_delta(diffusion, cond, scale=1.10, mask=None, n_samples=1, seed=None):
    print('get diff mod')
    diff_mod = get_diff_mod(diffusion)
    print('cond dievice')
    device = cond.device
    B, _, H, W = cond.shape
    g = None
    print('init g')
    if seed is not None:
        print('init g in if')
        g = torch.Generator(device=device).manual_seed(seed)
    def _mean_of_samples(c, n):
        outs = []
        for i in range(n):
            print('mean sample',i)
            if g is not None:
                torch.manual_seed(seed + i)
            outs.append(_sample_once(c))
        return torch.stack(outs, dim=0).mean(dim=0)
    print("get baseline")
    baseline = _mean_of_samples(cond, n_samples)
    if mask is None:
        cond2 = cond * scale
    else:
        cond2 = cond * (1 + (scale - 1) * mask)
    if g is not None:
        torch.manual_seed(seed)
    print('mean sample')
    cf = _mean_of_samples(cond2, n_samples)
    delta = cf - baseline
    return {"baseline": baseline, "cf": cf, "delta": delta}


def _minmax01_np(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mn = t.amin(dim=(2,3), keepdim=True)
    mx = t.amax(dim=(2,3), keepdim=True)
    return (t - mn) / (mx - mn + eps)

@torch.no_grad()
def counterfactual_panels(
    diffusion,
    cond_batch: torch.Tensor,
    truth_batch: torch.Tensor | None,
    path: str,
    device: torch.device,
    cf_cfg: dict | None = None,
    data_cfg: dict | None = None
):
    print("counterfactual_panels SAVE")
    cond = cond_batch.to(device)
    truth = truth_batch.to(device) if truth_batch is not None else None
    B, _, H, W = cond.shape
    mask = None
    print('build mask')
    if cf_cfg is not None and isinstance(cf_cfg.get("region", {}), dict):
        region = cf_cfg["region"]
        if region.get("type", "none") == "box":
            lat, lon = None, None
            if data_cfg is not None:
                lat_name = data_cfg.get("lat_name", "lat")
                lon_name = data_cfg.get("lon_name", "lon")
                lat, lon = _read_lat_lon_from_file(data_cfg.get("cond_file"), lat_name=lat_name, lon_name=lon_name)
            mask = _box_mask_from_coords(lat, lon, region["lat_min"], region["lat_max"], region["lon_min"], region["lon_max"], device, H=H, W=W)
    print('generate output')
    out = counterfactual_delta(
        diffusion, cond,
        scale=cf_cfg.get("scale", 1.10),
        mask=mask,
        n_samples=int(cf_cfg.get("n_samples", 1)),
        seed=cf_cfg.get("seed", None),
    )
    print('prepare panels')
    cond_v = _minmax01_np(cond).cpu()
    base_v = _minmax01_np(out["baseline"]).cpu()
    cf_v   = _minmax01_np(out["cf"]).cpu()
    dlt    = out["delta"].cpu()
    mean = dlt.mean(dim=(2,3), keepdim=True)
    std  = dlt.std(dim=(2,3), keepdim=True) + 1e-8
    dlt_v = _minmax01_np((dlt - mean) / std)

    titles = ["Condition", "Baseline", "Counterfactual", "Δ (z-scored)"]
    if truth is not None:
        truth_v = _minmax01_np(truth).cpu()
        titles = ["Condition", "Truth", "Baseline", "Counterfactual", "Δ (z)"]

    panels = []
    title_bar = 22
    for i in range(B):
        imgs = [TF.to_pil_image(cond_v[i])]
        if truth is not None:
            imgs.append(TF.to_pil_image(truth_v[i]))
        imgs.extend([TF.to_pil_image(base_v[i]), TF.to_pil_image(cf_v[i]), TF.to_pil_image(dlt_v[i])])

        titled = []
        for img, title in zip(imgs, titles):
            canvas = Image.new("RGB", (img.width, img.height + title_bar), color=(255, 255, 255))
            canvas.paste(img, (0, title_bar))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 2), title, fill=(0, 0, 0))
            titled.append(canvas)

        total_w = sum(im.width for im in titled)
        concat_img = Image.new("RGB", (total_w, titled[0].height), color=(255, 255, 255))
        x = 0
        for im in titled:
            concat_img.paste(im, (x, 0))
            x += im.width
        panels.append(concat_img)

    grid_w = max(p.width for p in panels)
    grid_h = sum(p.height for p in panels)
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    y = 0
    for pimg in panels:
        grid.paste(pimg, (0, y))
        y += pimg.height
    print('sace cf iamge')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid.save(path)

@torch.no_grad()
def _sample_compat(diff_mod, cond, shape, device, steps=None, ddim_eta=0.0):
    try:
        return diff_mod.sample(cond, shape=shape, device=device, steps=steps, ddim_eta=ddim_eta)
    except TypeError:
        return diff_mod.sample(cond, shape=shape, device=device)

@torch.no_grad()
def save_triptych_samples(
    diffusion,
    cond_batch: torch.Tensor,
    truth_batch: torch.Tensor,
    save_path: str,
    device: torch.device,
    ema=None,
    steps: int | None = None,
    ddim_eta: float = 0.0,
    return_tensor: bool = False,
):
    diffusion.eval()
    cond  = cond_batch.to(device, non_blocking=True)
    truth = truth_batch.to(device, non_blocking=True)
    diff_mod = get_diff_mod(diffusion)
    B, _, H, W = cond.shape

    def _sample_with_optional_ema():
        if ema is None:
            return _sample_compat(diff_mod, cond, (B,1,H,W), device, steps=steps, ddim_eta=ddim_eta)
        original = diff_mod.model
        try:
            diff_mod.model = ema.ema_model
            return _sample_compat(diff_mod, cond, (B,1,H,W), device, steps=steps, ddim_eta=ddim_eta)
        finally:
            diff_mod.model = original

    with torch.inference_mode():
        pred = _sample_with_optional_ema()

    try:
        target_mean = _area_weighted_mean(truth).detach()
        pred_mean   = _area_weighted_mean(pred)
        scale = (target_mean / (pred_mean + 1e-8)).view(-1, 1, 1, 1)
        pred = pred * scale
    except Exception as e:
        print(f"[warn] triptych scale-adjust failed: {e}")

    cond_v  = _minmax01_np(cond).cpu()
    truth_v = _minmax01_np(truth).cpu()
    pred_v  = _minmax01_np(pred).cpu()

    imgs = torch.empty((3 * B, 1, H, W), dtype=cond_v.dtype)
    imgs[0::3] = cond_v
    imgs[1::3] = truth_v
    imgs[2::3] = pred_v

    tb_grid = None
    if '_HAS_TV' in globals() and _HAS_TV:
        try:
            from torchvision.utils import make_grid, save_image
            grid = make_grid(imgs, nrow=3)
            save_image(grid, save_path)
            tb_grid = grid
            return (save_path, tb_grid) if return_tensor else save_path
        except Exception as e:
            print(f"[warn] torchvision grid/save failed: {e}")

    try:
        import numpy as np
        from PIL import Image
        rows = []
        for i in range(B):
            row = torch.cat([imgs[3*i + j] for j in range(3)], dim=-1)[0]
            rows.append(row)
        big = torch.cat(rows, dim=-2)
        big_np = (big.clamp(0,1).numpy() * 255.0).astype(np.uint8)
        Image.fromarray(big_np).save(save_path)
        tb_grid = big.unsqueeze(0)
    except Exception as e:
        print(f"[error] PIL fallback failed: {e}")

    return (save_path, tb_grid) if return_tensor else save_path


def _ensure_hw_like(da: xr.DataArray, expect_leading: int,
                    y_name: str | None, x_name: str | None,
                    lat_name: str | None, lon_name: str | None) -> xr.DataArray:
    hw = _order_hw_dims(da, y_name, x_name, lat_name, lon_name)
    leading = [d for d in da.dims if d not in hw]
    print(leading,expect_leading)
    if len(leading) > expect_leading:
        for d in leading[expect_leading:]:
            if da.sizes[d] == 1:
                da = da.squeeze(d, drop=True)
        hw = _order_hw_dims(da, y_name, x_name, lat_name, lon_name)
        leading = [d for d in da.dims if d not in hw]
        if len(leading) != expect_leading:
            leading = leading[:expect_leading]
        print(leading)
    return da.transpose(*leading, *hw)

def _move_to_stack_hw(da: xr.DataArray,
                      stack_dim: str,
                      y_name: str | None, x_name: str | None,
                      lat_name: str | None, lon_name: str | None,
                      expect_leading: int) -> xr.DataArray:
    if stack_dim not in da.dims:
        raise ValueError(f"Expected stack_dim '{stack_dim}' in {da.dims}")
    print(da)
    da = da.transpose(stack_dim, ...)
    return _ensure_hw_like(da, expect_leading=expect_leading,
                           y_name=y_name, x_name=x_name,
                           lat_name=lat_name, lon_name=lon_name)

def _find_member_dim(da: xr.DataArray, hint: str | None) -> str | None:
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

def load_cond_and_target(
    cond_file: str,
    cond_var: str,
    target_file: str,
    target_var: str,
    stack_dim: str = "year",
    member_dim: str | None = "member_id",
    y_name: str | None = None,
    x_name: str | None = None,
    lat_name: str | None = None,
    lon_name: str | None = None,
    normalize: bool = True,
):
    import xarray as xr
    import numpy as np

    with xr.open_dataset(cond_file) as ds_c:
        da_c = ds_c[cond_var].load()
        Hname, Wname = _order_hw_dims(da_c, y_name, x_name, lat_name, lon_name)
        dims_c = [d for d in (stack_dim, member_dim, Hname, Wname) if d in da_c.dims]
        if member_dim not in da_c.dims:
            raise ValueError(f"Condition var lacks member dim '{member_dim}'. Found dims={da_c.dims}")
        da_c = da_c.transpose(*dims_c)
        cond_np = da_c.values.astype(np.float32)
        cond_np = cond_np[:, :, None, :, :]

        if stack_dim in da_c.coords:
            time_ids = np.arange(da_c.sizes[stack_dim], dtype=np.int64)
        else:
            time_ids = np.arange(cond_np.shape[0], dtype=np.int64)

    with xr.open_dataset(target_file) as ds_t:
        da_t = ds_t[target_var].load()
        Hname, Wname = _order_hw_dims(da_t, y_name, x_name, lat_name, lon_name)
        if member_dim not in da_t.dims:
            raise ValueError(f"Target var lacks member dim '{member_dim}'. Found dims={da_t.dims}")
        da_t = da_t.transpose(stack_dim, member_dim, Hname, Wname)
        tgt_np = da_t.values.astype(np.float32)
        tgt_np = tgt_np[:, :, None, :, :]

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

def sample_preview(diffusion, cond_batch, save_path):
    diffusion.eval()
    device = cond_batch.device
    B, _, H, W = cond_batch.shape
    samples = diffusion.sample(cond_batch, shape=(B, 1, H, W), device=device)
    imgs = []
    for i in range(B):
        s = samples[i]
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        imgs.append(s)
    out = torch.stack(imgs, dim=0)
    if _HAS_TV:
        from torchvision.utils import save_image
        save_image(out, save_path, nrow=min(4, out.shape[0]), padding=2)
    else:
        torch.save(out.cpu(), save_path.replace(".png", ".pt"))

def build_model_from_config(cfg_unet: Dict[str, Any]) -> UNet:
    return UNet(
        in_channels=cfg_unet.get("in_channels", 2),
        out_channels=cfg_unet.get("out_channels", 1),
        base_ch=cfg_unet.get("base_ch", 64),
        ch_mults=tuple(cfg_unet.get("ch_mults", (1, 2, 4))),
        num_res_blocks=cfg_unet.get("num_res_blocks", 2),
        time_dim=cfg_unet.get("time_dim", 256),
        groups=cfg_unet.get("groups", 8),
        use_checkpoint=cfg_unet.get("use_checkpoint",True),
        dropout=cfg_unet.get("dropout", 0.0),
    )

def get_diff_mod(model):
    base = model
    while hasattr(base, "module"):
        base = base.module
    return base

def _sample_compat_old(diff_mod, cond, B, H, W, device, steps=None, ddim_eta=0.0):
    try:
        return diff_mod.sample(cond, shape=(B,1,H,W), device=device, steps=steps, ddim_eta=ddim_eta)
    except TypeError:
        return diff_mod.sample(cond, shape=(B,1,H,W), device=device)
    from torch.nn.parallel import DistributedDataParallel as DDP
    return model.module if isinstance(model, DDP) else model

# ----------------- NEW: FSDP helpers -----------------

def _mp_policy(kind: str | None):
    if kind is None or kind.lower() == "none":
        return None
    if kind.lower() == "bf16":
        return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    if kind.lower() == "fp16":
        return MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
    return None

def _make_mixed_precision(mp_cfg):
    """Accepts None | str | dict | MixedPrecision -> MixedPrecision|None"""
    if mp_cfg is None or mp_cfg is False:
        return None
    if isinstance(mp_cfg, MixedPrecision):
        return mp_cfg
    if isinstance(mp_cfg, str):
        s = mp_cfg.lower()
        if s in ("bf16", "bfloat16"):
            dt = torch.bfloat16
        elif s in ("fp16", "float16", "amp"):
            dt = torch.float16
        elif s in ("fp32", "float32", "none"):
            return None
        else:
            raise ValueError(f"Unknown mixed_precision string: {mp_cfg}")
        return MixedPrecision(param_dtype=dt, reduce_dtype=dt, buffer_dtype=dt)
    if isinstance(mp_cfg, dict):
        map_dt = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32,
        }
        p = map_dt.get(str(mp_cfg.get("param", "fp32")).lower(), torch.float32)
        r = map_dt.get(str(mp_cfg.get("reduce", "fp32")).lower(), torch.float32)
        b = map_dt.get(str(mp_cfg.get("buffer", "fp32")).lower(), torch.float32)
        return MixedPrecision(param_dtype=p, reduce_dtype=r, buffer_dtype=b)
    return None

def _make_sharding_strategy(s):
    """Accepts None | str | ShardingStrategy -> ShardingStrategy|None"""
    if not s:
        return None
    if isinstance(s, ShardingStrategy):
        return s
    m = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    key = str(s).lower()
    if key not in m:
        raise ValueError(f"Unknown sharding_strategy: {s}")
    return m[key]

def wrap_fsdp(model, fsdp_cfg):
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16) \
         if fsdp_cfg.get("mp", "bf16")=="bf16" else \
         MixedPrecision(param_dtype=torch.float16,  reduce_dtype=torch.float16,  buffer_dtype=torch.float16)

    auto_wrap_policy = partial(size_based_auto_wrap_policy,
                               min_num_params=fsdp_cfg.get("min_params", 200_000))  # smaller => more wrapping

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=fsdp_cfg.get("cpu_offload", False)),
        device_id=(torch.cuda.current_device() if torch.cuda.is_available() else None),
    )

# ----------------- NEW: DeepSpeed cfg -----------------

def build_deepspeed_cfg(ds_cfg: dict):
    zero_stage = int(ds_cfg.get("zero_stage", 3))
    bf16 = bool(ds_cfg.get("bf16", True))
    off_p = bool(ds_cfg.get("offload_params_to_cpu", True))
    off_o = bool(ds_cfg.get("offload_optimizer_to_cpu", True))
    use_nvme = bool(ds_cfg.get("nvme_offload", False))

    cfg = {
        "train_batch_size": 0,
        "gradient_accumulation_steps": int(ds_cfg.get("gradient_accumulation_steps", ds_cfg.get("accum_steps", 1))),
        "zero_optimization": {
            "stage": zero_stage,
            "offload_param": {"device": "cpu", "pin_memory": True} if off_p else None,
            "offload_optimizer": {"device": "cpu", "pin_memory": True} if off_o else None,
        },
        "bf16": {"enabled": bf16},
        "optimizer": ds_cfg.get("optimizer", {"type": "AdamW", "params": {"lr": 2e-4}})
    }
    if use_nvme:
        cfg["zero_optimization"]["offload_param"] = {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "buffer_count": 4,
            "buffer_size": 1e8
        }
        cfg["zero_optimization"]["offload_optimizer"] = {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "buffer_count": 4,
            "buffer_size": 1e8
        }
    return cfg

# ----------------- TRAIN ONE EPOCH (backend-aware) -----------------

def train_one_epoch(
    diffusion,
    dl,
    optimizer,
    device,
    scaler,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    epoch: int = 1,
    metric_logger=None,
    tb_writer=None,
    ema=None,
    backend: str = "ddp",
    ds_engine=None,
):
    diffusion.train()
    diff_mod = get_diff_mod(diffusion)

    total = 0.0
    steps = 0

    for step, (cond, x0) in enumerate(dl,start=1):
        cond = cond.to(device, non_blocking=True)
        x0   = x0.to(device, non_blocking=True)

        if backend == "deepspeed":
            assert ds_engine is not None
            optimizer.zero_grad(set_to_none=True)
            if hasattr(diff_mod, 'loss_components'):
                comps = diff_mod.loss_components(x0, cond)
                loss = comps['total']
            else:
                loss = diff_mod.loss(x0, cond)
                comps = {'mse_raw': loss.detach(), 'mse_lat': loss.detach(), 'cond_loss': torch.tensor(0.0, device=device)}
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}: {loss.item()}")
            ds_engine.backward(loss)
            if max_grad_norm is not None and max_grad_norm > 0:
                ds_engine.clip_grad_norm(max_grad_norm)
            ds_engine.step()

        else:
            optimizer.zero_grad(set_to_none=True)
            try:
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        if hasattr(diff_mod, 'loss_components'):
                            comps = diff_mod.loss_components(x0, cond)
                            loss = comps['total']
                        else:
                            loss = diff_mod.loss(x0, cond)
                            comps = {'mse_raw': loss.detach(), 'mse_lat': loss.detach(), 'cond_loss': torch.tensor(0.0, device=device)}
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}: {loss.item()}")
                    scaler.scale(loss).backward()
                    if max_grad_norm is not None and max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if hasattr(diff_mod, 'loss_components'):
                        comps = diff_mod.loss_components(x0, cond)
                        loss = comps['total']
                    else:
                        loss = diff_mod.loss(x0, cond)
                        comps = {'mse_raw': loss.detach(), 'mse_lat': loss.detach(), 'cond_loss': torch.tensor(0.0, device=device)}
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}: {loss.item()}")
                    loss.backward()
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_grad_norm)
                    optimizer.step()
            except Exception as e:
                if get_rank() == 0:
                    import traceback
                    print(f"[FATAL] Exception during training (epoch {epoch} step {step}): {e}")
                    traceback.print_exc()
                if is_dist():
                    try:
                        dist.destroy_process_group()
                    finally:
                        pass
                raise

        if ema is not None and backend != "deepspeed":
            ema.update()

        if metric_logger is not None:
            try:
                metric_logger.log(epoch, steps, comps.get('mse_raw', loss).item(), comps.get('mse_lat', loss).item(), comps.get('cond_loss', 0.0).item(), loss.item())
                if tb_writer is not None:
                    gs = (epoch-1) * len(dl) + steps
                    tb_writer.add_scalar('loss/mse_raw', comps.get('mse_raw', loss).item(), gs)
                    tb_writer.add_scalar('loss/mse_lat', comps.get('mse_lat', loss).item(), gs)
                    tb_writer.add_scalar('loss/cond_loss', comps.get('cond_loss', 0.0).item(), gs)
                    tb_writer.add_scalar('loss/total', loss.item(), gs)
            except Exception:
                pass

        total += float(loss.detach().item())
        steps += 1

    return total / max(1, steps)

# ----------------- CHECKPOINT LOADER (portable) -----------------

def load_checkpoint(
    ckpt_path: str,
    unet,
    diffusion,
    optimizer=None,
    scaler=None,
    device="cuda"
):
    ckpt = torch.load(ckpt_path, map_location=device)

    missing, unexpected = unet.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print("[UNet] missing keys:", missing)
        print("[UNet] unexpected keys:", unexpected)

    diff_state = diffusion.state_dict()
    diff_state.update(ckpt.get("diffusion_buffers", {}))
    diffusion.load_state_dict(diff_state, strict=False)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    print(f"[Resume] Loaded {ckpt_path}. Resuming at epoch {start_epoch}.")
    return start_epoch


def main(config: Dict[str, Any]):
    setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    train_cfg = config["train"]
    unet_cfg = config["unet"]

    # Allow fsdp/deepspeed config either under train.* or top-level
    fsdp_cfg = {**config.get("fsdp", {}), **train_cfg.get("fsdp", {})}
    ds_cfg   = {**config.get("deepspeed", {}), **train_cfg.get("deepspeed", {})}
    backend = train_cfg.get("backend", "ddp").lower()
    use_amp = bool(train_cfg.get("use_amp", True))

    save_dir = train_cfg.get("save_dir", "runs/exp1")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb")) if get_rank()==0 else None
    metric_logger = MetricLogger(os.path.join(save_dir, "metrics.csv"), smooth=train_cfg.get("smooth", 100)) if get_rank()==0 else None

    cond_np, tgt_np, times_ids = load_cond_and_target(
        cond_file=data_cfg["cond_file"],
        cond_var=data_cfg["cond_var"],
        target_file=data_cfg["target_file"],
        target_var=data_cfg["target_var"],
        stack_dim=data_cfg.get("stack_dim", "year"),
        member_dim=data_cfg.get("member_dim"),
        y_name=data_cfg.get("y_name"),
        x_name=data_cfg.get("x_name"),
        lat_name=data_cfg.get("lat_name"),
        lon_name=data_cfg.get("lon_name"),
    )

    print('init logger')
    log_csv = os.path.join(save_dir, "loss.csv")
    loss_logger = LossLogger(log_csv, smooth=train_cfg.get("smooth_steps", 100)) if get_rank()==0 else None

    print('init ds')
    ds = WindowedAllMembersDataset_random(
        cond_np, tgt_np,
        K=config["dataset"]["K"],
        center=config["dataset"]["center"],
        crop_hw=tuple(config["dataset"]["crop_hw"]) if config["dataset"]["crop_hw"] else None,
        crop_mode=config["dataset"]["crop_mode"],
        time_reverse_p=config["dataset"]["time_reverse_p"],
        sample_mode=config["dataset"]["sample_mode"],
        window_radius=config["dataset"]["window_radius"],
        keep_chronology=config["dataset"]["keep_chronology"],
        causal=config["dataset"]["causal"],
    )
    sampler = DistributedSampler(ds, shuffle=True, drop_last=False) if is_dist() else None

    print('init dataloader')
    dl = DataLoader(
        ds,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=True
    )

    if get_rank() == 0:
        print(f"[Data] cond shape: {cond_np.shape} (T,M,C,H,W)")
        print(f"[Data] tgt  shape: {tgt_np.shape}  (T,M,C,H,W)")
        print(f"[Data] dataset length: {len(ds)}  (should be T*M)")
        print(f"[Data] batches/epoch on this rank: {len(dl)}")

    print('build model')
    unet = build_model_from_config(unet_cfg).to(device)
    if train_cfg.get("activation_checkpointing", True):
        apply_act_ckpt_to_unet(unet)
    print('diffusion')
    diffusion = Diffusion(
        unet,
        img_channels=1,
        timesteps=train_cfg.get("timesteps", config.get("diffusion", {}).get("timesteps", 1000)),
        beta_schedule=train_cfg.get("beta_schedule", config.get("diffusion", {}).get("beta_schedule", "linear"))
    ).to(device)

    # ----- resume BEFORE wrapping for FSDP/DeepSpeed -----
    start_epoch = 1
    resume_path = train_cfg.get("resume")
    if resume_path and os.path.isfile(resume_path) and backend in ("fsdp", "deepspeed"):
        start_epoch = load_checkpoint(
            resume_path,
            unet=unet,
            diffusion=diffusion,
            optimizer=None,
            scaler=None,
            device="cpu"
        )

    # ----- wrap backend -----
    ds_engine = None
    optimizer = None
    scaler = None

    if backend == "fsdp":
        assert _HAS_FSDP, "Requested FSDP but not available"
        diffusion = wrap_fsdp(diffusion, fsdp_cfg)
        opt_cfg = train_cfg.get("optimizer", {})
        optimizer = torch.optim.AdamW(
            diffusion.parameters(),
            lr=opt_cfg.get("lr", 2e-4),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
            weight_decay=opt_cfg.get("weight_decay", 1e-4)
        )
        scaler = GradScaler(enabled=use_amp)

    elif backend == "deepspeed":
        assert _HAS_DEEPSPEED, "Requested DeepSpeed but not available"
        ds_json = build_deepspeed_cfg(ds_cfg)
        ds_engine, optimizer, _, _ = deepspeed.initialize(
            model=diffusion,
            model_parameters=diffusion.parameters(),
            config=ds_json
        )
        diffusion = ds_engine
        device = ds_engine.device
        scaler = None

    else:  # DDP / single GPU
        if is_dist():
            diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        opt_cfg = train_cfg.get("optimizer", {})
        optimizer = torch.optim.AdamW(
            diffusion.parameters(),
            lr=opt_cfg.get("lr", 2e-4),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
            weight_decay=opt_cfg.get("weight_decay", 1e-4)
        )
        scaler = GradScaler(enabled=use_amp)

    # ----- resume AFTER wrapping for DDP -----
    if resume_path and os.path.isfile(resume_path) and backend == "ddp":
        start_epoch = load_checkpoint(
            resume_path,
            unet=unet,
            diffusion=diffusion,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )

    num_epochs   = train_cfg.get("num_epochs", 10)
    save_every   = train_cfg.get("save_every", 1)
    sample_every = train_cfg.get("sample_every", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    fixed_preview = None

    for epoch in range(start_epoch, num_epochs + 1):
        print("EPOCH: ",epoch)
        if is_dist() and isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch)

        loss_avg = train_one_epoch(
            diffusion, dl, optimizer, device, scaler,
            max_grad_norm=max_grad_norm, use_amp=use_amp,
            epoch=epoch,
            metric_logger=loss_logger if get_rank()==0 else None,
            backend=backend,
            ds_engine=ds_engine
        )
        rank0 = (get_rank() == 0)
        if rank0:
            print(f"[Epoch {epoch}/{num_epochs}] loss={loss_avg:.6f}")

        if is_dist():
            barrier()

        # ----- checkpointing per backend -----
        if rank0 and epoch % save_every == 0:
            ckpt_dir = os.path.join(save_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            if backend == "deepspeed":
                ds_engine.save_checkpoint(ckpt_dir, tag=f"epoch_{epoch:04d}")
                base = get_diff_mod(ds_engine)
                full = base.state_dict()
                model_sd = {k[len("model."):]: v for k, v in full.items() if k.startswith("model.")}
                diff_buf = {k: v for k, v in full.items() if not k.startswith("model.")}
                torch.save({
                    "epoch": epoch,
                    "model": model_sd,
                    "diffusion_buffers": diff_buf,
                    "config": config
                }, os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))

            elif backend == "fsdp":
                with fsdp_rank0_fullstate_ctx(diffusion):
                    full = diffusion.state_dict()
                model_sd = {k[len("model."):]: v for k, v in full.items() if k.startswith("model.")}
                diff_buf = {k: v for k, v in full.items() if not k.startswith("model.")}
                torch.save({
                    "epoch": epoch,
                    "model": model_sd,
                    "diffusion_buffers": diff_buf,
                    "config": config
                }, os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))

            else:  # DDP / single
                base = get_diff_mod(diffusion)
                full = base.state_dict()
                model_sd = {k[len("model."):]: v for k, v in full.items() if k.startswith("model.")}
                diff_buf = {k: v for k, v in full.items() if not k.startswith("model.")}
                torch.save({
                    "epoch": epoch,
                    "model": model_sd,
                    "diffusion_buffers": diff_buf,
                    "optimizer": optimizer.state_dict(),
                    "config": config
                }, os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt"))
                print(f"Saved checkpoint -> {os.path.join(ckpt_dir, f'ckpt_epoch_{epoch:04d}.pt')}")

        if is_dist():
            barrier()

    # ----- final save -----
    if get_rank() == 0:
        ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if backend == "deepspeed":
            ds_engine.save_checkpoint(ckpt_dir, tag="final")
            base = get_diff_mod(ds_engine)
            full = base.state_dict()
            model_sd = {k[len("model."):]: v for k, v in full.items() if k.startswith("model.")}
            diff_buf = {k: v for k, v in full.items() if not k.startswith("model.")}
            torch.save({"epoch": num_epochs, "model": model_sd, "diffusion_buffers": diff_buf, "config": config},
                       os.path.join(ckpt_dir, "final.pt"))
        elif backend == "fsdp":
            with fsdp_rank0_fullstate_ctx(diffusion):
                full = diffusion.state_dict()
            model_sd = {k[len("model."):]: v for k, v in full.items() if k.startswith("model.")}
            diff_buf = {k: v for k, v in full.items() if not k.startswith("model.")}
            torch.save({"epoch": num_epochs, "model": model_sd, "diffusion_buffers": diff_buf, "config": config},
                       os.path.join(ckpt_dir, "final.pt"))
        else:
            base = get_diff_mod(diffusion)
            full = base.state_dict()
            model_sd = {k[len("model."):]: v for k, v in full.items() if k.startswith("model.")}
            diff_buf = {k: v for k, v in full.items() if not k.startswith("model.")}
            torch.save({"epoch": num_epochs, "model": model_sd, "diffusion_buffers": diff_buf,
                        "optimizer": optimizer.state_dict(), "config": config},
                       os.path.join(ckpt_dir, "final.pt"))
            print(f"Training done. Final checkpoint -> {os.path.join(ckpt_dir, 'final.pt')}")

    if get_rank()==0 and loss_logger is not None:
        loss_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config (JSON or YAML)")
    parser.add_argument("--set", nargs="*", default=[],
                        help='Dot overrides, e.g. train.batch_size=4 unet.base_ch=64')
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    print(cfg)
    main(cfg)
