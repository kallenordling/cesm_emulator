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
    """
    Compute latitude-based weights ~ cos(lat) for a grid of given height H.
    Args:
        height (int): number of latitude points
        device: torch.device
    Returns:
        torch.Tensor of shape [H], normalized to mean=1
    """
    lat = torch.linspace(-90, 90, steps=height, device=device)
    w = torch.cos(torch.deg2rad(lat)).clamp_min(0)
    return w / (w.mean() + 1e-8)


def _area_weighted_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute area-weighted mean of a field [B,C,H,W] using cos(lat).
    Args:
        tensor: [B, C, H, W]
    Returns:
        [B, C] tensor of area-weighted means
    """
    B, C, H, W = tensor.shape
    w = _latitude_weights(H, tensor.device).view(1, 1, H, 1)
    return (tensor * w).mean(dim=(-2, -1))  # average over H,W


# flag for torchvision availability (for fast grids/saving)
try:
    import torchvision  # noqa: F401
    _HAS_TV = True
except Exception:
    _HAS_TV = False

from model import UNet, Diffusion
from dataset_single_member import SingleMemberDataset, AllMembersDataset

@torch.no_grad()
def pick_mid_t(T):  # choose a representative diffusion step
    return T // 2

@torch.no_grad()
def counterfactual_delta(diff_mod, cond,scale_mask=None, scale=1.1, steps=None):
    """
    Generate two samples: baseline and perturbed cond (e.g., +10% in a region).
    Return difference map (Prediction_perturbed - Prediction_base).
    """
    device = cond.device
    B, _, H, W = cond.shape
    if steps is None:
        steps = diff_mod.T  # full sampling

    # baseline
    base = diff_mod.sample(cond, shape=(B,1,H,W), device=device)

    # perturbed cond
    cond2 = cond.clone()
    if scale_mask is not None:
        cond2 = cond2 * (1 + (scale-1) * scale_mask)  # scale only in masked region
    else:
        cond2 = cond2 * scale
    cf = diff_mod.sample(cond2, shape=(B,1,H,W), device=device,)

    return cf - base

def saliency_wrt_cond(diff_mod, x0, cond, years=None, t=None):
    """
    Gradient saliency w.r.t. the condition map.
    Returns a [0,1]-normalized saliency tensor with shape (B,1,H,W).
    """
    device = x0.device
    B = x0.size(0)

    # Pick a middle diffusion step if not given
    if t is None:
        t = torch.full((B,), diff_mod.T // 2, device=device, dtype=torch.long)

    # Enable gradient for the condition input
    cond_req = cond.detach().clone().requires_grad_(True)

    # Forward diffuse the target to x_t
    x_t, noise = diff_mod.q_sample(x0, t)

    # Predict noise with current cond
    eps_pred = diff_mod.model(x_t, cond_req, t, years=years)

    # Loss and direct gradient wrt condition (no param grad accumulation)
    loss = F.mse_loss(eps_pred, noise)
    g = torch.autograd.grad(loss, cond_req, retain_graph=False, create_graph=False)[0].detach().abs()  # (B,1,H,W)

    # Per-sample [0,1] normalization for visualization
    g = g / (g.amax(dim=(2,3), keepdim=True) + 1e-8)
    return g
    
def gaussian_smooth_latlon(da: xr.DataArray, sigma_lat=1.5, sigma_lon=1.5):
    # Work on a copy; assume last two dims are lat, lon
    assert da.dims[-2:] == ("lat","lon")
    # Periodic wrap in longitude to avoid edge artifacts:
    pad = 3 * int(np.ceil(max(sigma_lat, sigma_lon)))
    da_wrap = xr.concat([da.isel(lon=slice(-pad,None)),
                         da,
                         da.isel(lon=slice(0,pad))], dim="lon")
    # Apply Gaussian on raw values
    arr = gaussian_filter(da_wrap.values, sigma=[0]*(da_wrap.ndim-2)+[sigma_lat, sigma_lon], mode="nearest")
    out = da_wrap.copy(data=arr)
    # Unwrap back to original lon extent
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
            barrier()


def setup_distributed():
    import os
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not dist.is_available() or world == 1:
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # set current device before init

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=30),
    )


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
    """
    Option 1: Gradient saliency w.r.t. the condition map.
    Returns a [0,1]-normalized saliency tensor with shape (B,1,H,W).
    """
    device = x0.device
    B = x0.size(0)
    # Pick a middle diffusion step if not given
    if t is None:
        t = torch.full((B,), diff_mod.T // 2, device=device, dtype=torch.long)

    # Enable gradient for cond
    cond_req = cond.detach().clone().requires_grad_(True)

    # Forward diffuse the target to x_t
    x_t, noise = diff_mod.q_sample(x0, t)

    # Predict noise with current cond
    eps_pred = diff_mod.model(x_t, cond_req, t)

    # Loss and gradients
    loss = F.mse_loss(eps_pred, noise)
    loss.backward()

    # |grad| and per-sample min-max normalization
    g = cond_req.grad.detach().abs()  # (B,1,H,W)
    g = g / (g.amax(dim=(2,3), keepdim=True) + 1e-8)
    return g

@torch.no_grad()
def quad_with_saliency(
    diffusion,
    cond_batch: torch.Tensor,   # (B,1,H,W)
    truth_batch: torch.Tensor,  # (B,1,H,W)
    path: str,
    device: torch.device,
    return_tensor: bool = False,

):
    """
    Saves a 4-pane image per item: [Condition | Truth | Prediction | Saliency]
    """
    diff_mod = get_diff_mod(diffusion)
    cond = cond_batch.to(device)
    truth = truth_batch.to(device)
    B, _, H, W = cond.shape

    # Generate prediction (full sampler)
    pred = diff_mod.sample(cond, shape=(B, 1, H, W), device=device)
    print('save quad with')
    # Saliency requires grads; compute on a small subset (here same batch)
    # Detach copies to avoid autograd interaction with the sampled pred
    with torch.enable_grad():
        sal = saliency_wrt_cond(diff_mod, x0=truth.detach(), cond=cond.detach())

    # Normalize panels for visualization
    cond_v  = _minmax01(cond).cpu()
    truth_v = _minmax01(truth).cpu()
    pred_v  = _minmax01(pred).cpu()
    sal_v   = _minmax01(sal).cpu()

    titles = ["Condition", "Truth", "Prediction", "Saliency"]
    title_bar = 22  # px

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
    """
    Build a (1,1,H,W) mask from numeric lat/lon vectors if available,
    otherwise from H,W and fractional indices (fallback: no-op mask).
    """
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
        # Fallback: whole domain
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
    """
    Generate baseline and counterfactual samples and return their difference.
    """
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
    cond_batch: torch.Tensor,   # (B,1,H,W)
    truth_batch: torch.Tensor | None,  # (B,1,H,W) or None
    path: str,
    device: torch.device,
    cf_cfg: dict | None = None,
    data_cfg: dict | None = None
):
    """
    Save [Condition | Baseline | Counterfactual | Δ] panels (and Truth if provided).
    Region is optional; if provided, mask is a lat/lon box.
    """
    print("counterfactual_panels SAVE")
    cond = cond_batch.to(device)
    truth = truth_batch.to(device) if truth_batch is not None else None
    B, _, H, W = cond.shape

    # Build mask from config if requested
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
    # Prepare panes
    cond_v = _minmax01_np(cond).cpu()
    base_v = _minmax01_np(out["baseline"]).cpu()
    cf_v   = _minmax01_np(out["cf"]).cpu()
    dlt    = out["delta"].cpu()
    # z-score per-sample for contrast then minmax
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
    """
    Call diff_mod.sample(...) with (steps, ddim_eta) if supported, else fall back to legacy signature.
    """
    try:
        return diff_mod.sample(cond, shape=shape, device=device, steps=steps, ddim_eta=ddim_eta)
    except TypeError:
        return diff_mod.sample(cond, shape=shape, device=device)

@torch.no_grad()
def save_triptych_samples(
    diffusion,
    cond_batch: torch.Tensor,   # (B,1,H,W)
    truth_batch: torch.Tensor,  # (B,1,H,W)
    save_path: str,
    device: torch.device,
    ema=None,
    steps: int | None = None,
    ddim_eta: float = 0.0,
    return_tensor: bool = False,
):
    """
    Fast triptych: [cond | truth | pred] for each sample, written as a single PNG.
    If return_tensor=True, ALWAYS returns (save_path, tb_grid) where tb_grid can be None on failure.
    """
    diffusion.eval()
    cond  = cond_batch.to(device, non_blocking=True)
    truth = truth_batch.to(device, non_blocking=True)
    diff_mod = get_diff_mod(diffusion)

    B, _, H, W = cond.shape

    # ---- sample with optional EMA, without copying state_dict ----
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

    # ---- optional global-scale adjustment ----
    try:
        target_mean = _area_weighted_mean(truth).detach()
        pred_mean   = _area_weighted_mean(pred)
        scale = (target_mean / (pred_mean + 1e-8)).view(-1, 1, 1, 1)
        pred = pred * scale
    except Exception as e:
        print(f"[warn] triptych scale-adjust failed: {e}")

    # ---- build grid [cond | truth | pred] ----
    cond_v  = _minmax01_np(cond).cpu()
    truth_v = _minmax01_np(truth).cpu()
    pred_v  = _minmax01_np(pred).cpu()

    imgs = torch.empty((3 * B, 1, H, W), dtype=cond_v.dtype)
    imgs[0::3] = cond_v
    imgs[1::3] = truth_v
    imgs[2::3] = pred_v

    tb_grid = None  # make sure we always define this
    # Try torchvision fast path
    if '_HAS_TV' in globals() and _HAS_TV:
        try:
            from torchvision.utils import make_grid, save_image
            grid = make_grid(imgs, nrow=3)  # [C,H,W], in [0,1]
            save_image(grid, save_path)
            tb_grid = grid  # already [C,H,W]
            return (save_path, tb_grid) if return_tensor else save_path
        except Exception as e:
            print(f"[warn] torchvision grid/save failed: {e}")

    # Fallback to PIL/NumPy
    try:
        import numpy as np
        from PIL import Image
        rows = []
        for i in range(B):
            row = torch.cat([imgs[3*i + j] for j in range(3)], dim=-1)[0]  # (H, 3W)
            rows.append(row)
        big = torch.cat(rows, dim=-2)  # (B*H, 3W)
        big_np = (big.clamp(0,1).numpy() * 255.0).astype(np.uint8)
        Image.fromarray(big_np).save(save_path)
        tb_grid = big.unsqueeze(0)  # [1, H_total, W_total]
    except Exception as e:
        print(f"[error] PIL fallback failed: {e}")
        # leave tb_grid as None

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
    """
    Returns:
      cond_np: (T, M, 1, H, W)
      tgt_np:  (T, M, 1, H, W)  # (we add channel dim)
      time_ids: (T,) int64 indices 0..T-1 (or actual years if you prefer)
    """
    import xarray as xr
    import numpy as np

    # --- load condition ---
    with xr.open_dataset(cond_file) as ds_c:
        da_c = ds_c[cond_var].load()
        # Expect stack_dim + member_dim + H,W
        # Ensure dims order: (stack_dim, member_dim, H, W)
        Hname, Wname = _order_hw_dims(da_c, y_name, x_name, lat_name, lon_name)
        dims_c = [d for d in (stack_dim, member_dim, Hname, Wname) if d in da_c.dims]
        if member_dim not in da_c.dims:
            raise ValueError(f"Condition var lacks member dim '{member_dim}'. Found dims={da_c.dims}")
        da_c = da_c.transpose(*dims_c)
        cond_np = da_c.values.astype(np.float32)                # (T, M, H, W)
        cond_np = cond_np[:, :, None, :, :]                     # (T, M, 1, H, W)

        # time ids (0..T-1); swap with actual years if you want
        if stack_dim in da_c.coords:
            time_ids = np.arange(da_c.sizes[stack_dim], dtype=np.int64)
        else:
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
        # global z-score per tensor (channel-agnostic)
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
        dropout=cfg_unet.get("dropout", 0.0),
    )

def get_diff_mod(model):
    """
    Return the underlying Diffusion module, unwrapping common wrappers like
    DistributedDataParallel (DDP), FSDP, and nested .module attributes.

    Works even if model is wrapped multiple times.
    """
    # unwrap any `.module` chain (DDP, some Accelerate wrappers, nested wrappers)
    base = model
    while hasattr(base, "module"):
        base = base.module
    return base
def _sample_compat_old(diff_mod, cond, B, H, W, device, steps=None, ddim_eta=0.0):
    """Call Diffusion.sample with steps if supported; otherwise fall back to legacy signature."""
    try:
        return diff_mod.sample(cond, shape=(B,1,H,W), device=device, steps=steps, ddim_eta=ddim_eta)
    except TypeError:
        # older model without steps/ddim_eta
        return diff_mod.sample(cond, shape=(B,1,H,W), device=device)
    from torch.nn.parallel import DistributedDataParallel as DDP
    return model.module if isinstance(model, DDP) else model
'''
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
):
    diffusion.train()
    running = 0.0
    diff_mod = get_diff_mod(diffusion)

    for cond, x0 in dl:
        cond = cond.to(device, non_blocking=True)
        x0   = x0.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda'):
                if hasattr(diff_mod, 'loss_components'):
                    comps = diff_mod.loss_components(x0, cond)
                    loss = comps['total']
                else:
                    loss = diff_mod.loss(x0, cond)
                    comps = {'mse_raw': loss.detach(), 'mse_lat': loss.detach(), 'cond_loss': torch.tensor(0.0, device=device)}
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(diffusion.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
                if ema is not None:
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
        else:
            if hasattr(diff_mod, 'loss_components'):
                comps = diff_mod.loss_components(x0, cond)
                loss = comps['total']
            else:
                if hasattr(diff_mod, 'loss_components'):
                    comps = diff_mod.loss_components(x0, cond)
                    loss = comps['total']
                else:
                    loss = diff_mod.loss(x0, cond)
                    comps = {'mse_raw': loss.detach(), 'mse_lat': loss.detach(), 'cond_loss': torch.tensor(0.0, device=device)}
                comps = {'mse_raw': loss.detach(), 'mse_lat': loss.detach(), 'cond_loss': torch.tensor(0.0, device=device)}
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(diffusion.parameters(), max_grad_norm)
            optimizer.step()

        running += float(loss.item())
    return running / max(1, len(dl))
'''

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
):
    diffusion.train()
    diff_mod = get_diff_mod(diffusion)

    total = 0.0
    steps = 0

    #if isinstance(dl.sampler, torch.utils.data.distributed.DistributedSampler):
    #    dl.sampler.set_epoch(epoch)

    for step, batch in enumerate(dl, start=1):
        # support (cond, x0) or (cond, x0, years)
        if len(batch) == 3:
                cond, x0, years = batch
                years = years.to(device, non_blocking=True)
        else:
                cond, x0 = batch
                years = None

        cond = cond.to(device, non_blocking=True)
        x0   = x0.to(device, non_blocking=True)

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
                # catch non-finite loss early
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}: {loss.item()}")

                scaler.scale(loss).backward()

                if max_grad_norm is not None and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
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
            # Print once, then bring all ranks down to avoid NCCL hangs
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
            
        total += float(loss.detach().item())
        steps += 1

        #if loss_logger is not None:
        #    loss_logger.log(epoch, step, float(loss.detach().item()))

    return total / max(1, steps)

def load_checkpoint(
    ckpt_path: str,
    unet,                 # your UNet() instance
    diffusion,            # your Diffusion() instance
    optimizer=None,       # torch.optim.Optimizer or None
    scaler=None,          # GradScaler or None
    device="cuda"
):
    ckpt = torch.load(ckpt_path, map_location=device)

    # 1) Load UNet weights
    missing, unexpected = unet.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print("[UNet] missing keys:", missing)
        print("[UNet] unexpected keys:", unexpected)

    # 2) Load diffusion buffers (betas, alphas, etc.)
    #    We saved them without the "model." prefix, so load non-strict into diffusion
    diff_state = diffusion.state_dict()
    diff_state.update(ckpt.get("diffusion_buffers", {}))
    diffusion.load_state_dict(diff_state, strict=False)

    # 3) Load optimizer (optional)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

        # Move optimizer state tensors to current device if needed
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # 4) Load scaler (optional – only if you saved it)
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

    save_dir = train_cfg.get("save_dir", "runs/exp1")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb")) if get_rank()==0 else None
    metric_logger = MetricLogger(os.path.join(save_dir, "metrics.csv"), smooth=train_cfg.get("smooth", 100)) if get_rank()==0 else None

    cond_np, tgt_np,times_ids = load_cond_and_target(
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
    loss_logger = LossLogger(log_csv, smooth= train_cfg.get("smooth_steps", 100)) if get_rank()==0 else None
    print('init ds')
    #ds = SingleMemberDataset(
    #    cond_np, tgt_np,
    #    member_mode=data_cfg.get("member_mode", "random"),
    #    fixed_member=data_cfg.get("fixed_member", 0)
    #)
    ds = AllMembersDataset(cond_np, tgt_np)  # covers all members every epoch

    sampler = None
    if is_dist():
        sampler = DistributedSampler(ds, shuffle=True, drop_last=False)
    print('init dataloader')
    dl = DataLoader(
        ds,
        batch_size=train_cfg.get("batch_size", 2),          # this is PER-GPU batch size
        shuffle=(sampler is None),                          # shuffle in sampler if distributed
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
    print('diffiousion')
    diffusion = Diffusion(
        unet,
        img_channels=1,
        timesteps=train_cfg.get("timesteps", 1000),
        beta_schedule=train_cfg.get("beta_schedule", "linear")
    ).to(device)

    if is_dist():
        diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    print('init optimizser')
    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=opt_cfg.get("lr", 2e-4),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        weight_decay=opt_cfg.get("weight_decay", 1e-4)
    )
    print('scaler')
    scaler = GradScaler(enabled=train_cfg.get("use_amp", True))

    num_epochs   = train_cfg.get("num_epochs", 10)
    save_every   = train_cfg.get("save_every", 1)
    sample_every = train_cfg.get("sample_every", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    fixed_preview = None
    start_epoch = 1
    resume_path = train_cfg.get("resume")
    if resume_path and os.path.isfile(resume_path):
        start_epoch = load_checkpoint(
            resume_path,
            unet=unet,
            diffusion=diffusion,
            optimizer=optimizer,
            scaler=scaler,             # only matters if you saved it
            device=device
        )
    for epoch in range(start_epoch, num_epochs + 1):
        print("EPOCH: ",epoch)
        if is_dist() and isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch)
            
        
            
        loss_avg = train_one_epoch(
            diffusion, dl, optimizer, device, scaler,
            max_grad_norm=max_grad_norm,use_amp=train_cfg.get("use_amp", True),
            epoch=epoch,
            metric_logger=loss_logger if get_rank()==0 else None,
        )
        rank0 = (get_rank() == 0)
        if rank0:
            print(f"[Epoch {epoch}/{num_epochs}] loss={loss_avg:.6f}")

        xai_cfg = train_cfg.get("xai", {})

        if rank0 and epoch % sample_every == 0:
            print("sample data")
            if fixed_preview is None:
                cond_preview, truth_preview = next(iter(dl))  # truth = x0 from dataset
                Bshow = min(train_cfg.get("sample_batch", 4), cond_preview.size(0))
                cond_preview  = cond_preview[:Bshow]
                truth_preview = truth_preview[:Bshow]
                fixed_preview = (cond_preview.to(device), truth_preview.to(device))

            cond_fix, truth_fix = fixed_preview
            trip_path = os.path.join(save_dir, "samples", f"epoch_{epoch:04d}_triptych.png")
            print("path",trip_path)
            '''
            if config["train"].get("xai", {}).get("saliency", False):
                quad_path = os.path.join(save_dir, "samples", f"epoch_{epoch:04d}_quad_xai.png")
                out_path, tb_img = save_quad_with_saliency(diffusion, cond_fix, truth_fix, quad_path, device, return_tensor=True)
                if tb_writer is not None:
                    tb_writer.add_image('preview/quad_saliency', tb_img, global_step=epoch)
                print(f"Saved quad+saliency -> {quad_path}")
            
            if config["train"].get("xai", {}).get("counterfactual", False) :
                cf_path = os.path.join(save_dir, f"epoch_{epoch:04d}_counterfactual.png")
                save_counterfactual_panels(
                diffusion,
                cond_fix,
                truth_fix,
                cf_path,
                device,
                cf_cfg=xai_cfg["counterfactual"],
                data_cfg=data_cfg
                )
                print(f"Saved cf -> {cf_path}")
            '''
            #if not (config["train"].get("xai", {}).get("counterfactual", False) or config["train"].get("xai", {}).get("saliency", False)):
            out_path, tb_img = save_triptych_samples(diffusion, cond_fix, truth_fix, trip_path, device, return_tensor=True)
            if tb_writer is not None:
                    tb_writer.add_image('preview/triptych', tb_img, global_step=epoch)
            print(f"Saved triptych -> {trip_path}")
            
            
            
        
        if is_dist():
            barrier()
            
        if rank0 and epoch % save_every == 0:
            ckpt_path = os.path.join(save_dir, "checkpoints", f"ckpt_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": unet.state_dict(),
                "diffusion_buffers": {k: v for k, v in diffusion.state_dict().items() if "model." not in k},
                "optimizer": optimizer.state_dict(),
                "config": config
            }, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

        if is_dist():
            barrier()

    final_path = os.path.join(save_dir, "checkpoints", "final.pt")
    torch.save({
        "epoch": num_epochs,
        "model": unet.state_dict(),
        "diffusion_buffers": {k: v for k, v in diffusion.state_dict().items() if "model." not in k},
        "optimizer": optimizer.state_dict(),
        "config": config
    }, final_path)
    print(f"Training done. Final checkpoint -> {final_path}")

    if get_rank()==0 and loss_logger is not None:
        loss_logger.close()
    
default_config = {
    "data": {
        "cond_file":  "../CESM2-LESN_emulator/co2_final.nc",
        "cond_var":   "CO2_em_anthro",
        "target_file":"../CESM2-LESN_emulator/splits/fold_1/climate_data_train_fold1.nc",
        "target_var": "TREFHT",
        "stack_dim":  "year",
        "member_dim": "member_id",
        "lat_name":   "lat",
        "lon_name":   "lon",
        "member_mode": "random",
        "fixed_member": 0
    },
    "unet": {
        "in_channels": 2,
        "out_channels": 1,
        "base_ch": 48,
        "ch_mults": [1, 2, 4,8],
        "num_res_blocks": 8,
        "time_dim": 124,
        "groups": 8,
        "dropout": 0.0
    },
    "train": {
        "resume": "runs/exp3/checkpoints/ckpt_epoch_1040.pt",
        "xai": {
             "saliency": False,
             "counterfactual": {
                 "enabled": False,
                 "scale": 1.10,
                 "n_samples": 1,
                 "seed": 123,
                 "region": {
                     "type": "box",
                     "lat_min": 30,
                     "lat_max": 60,
                     "lon_min": -130,
                     "lon_max": -60
                 }
             }
         },
        "batch_size": 10,
        "num_workers": 0,
        "timesteps": 1000,
        "beta_schedule": "linear",
        "optimizer": {
            "lr": 2e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-4
        },
        "sample_steps": 20,
        "ddim_eta": 0.0,
        "num_epochs": 10000,
        "use_amp": True,
        "max_grad_norm": 1.0,
        "save_dir": "runs/exp3",
        "save_every": 10,
        "sample_every": 100,
        "sample_batch": 2
    }
}

if __name__ == "__main__":
    cfg = default_config
    # If you prefer a JSON file:
    # with open("config.json") as f:
    #     cfg = json.load(f)
    main(cfg)
