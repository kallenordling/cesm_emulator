
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

# Import your 3D model architecture (you uploaded video_net.py separately)
from video_net import UNetModel3D

# -----------------------------
# Helpers

def _latitude_weights(H: int, device):
    lat = torch.linspace(-90, 90, steps=H, device=device)
    w = torch.cos(torch.deg2rad(lat)).clamp_min(0)
    return w / (w.mean() + 1e-8)

def _area_weighted_mean(x):
    B, C, H, W = x.shape
    w = _latitude_weights(H, x.device).view(1, 1, H, 1)
    return (x * w).mean(dim=(-2, -1))

# -----------------------------

def timestep_embedding(t, dim):
    device = t.device
    half = dim // 2
    if half < 1:
        return t.float().unsqueeze(1)
    freqs = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1 if half > 1 else 1)))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# -----------------------------
# 2D-facing wrapper around your 3D UNet
# Keeps train.py API the same: forward(x, cond, t) -> (B,1,H,W)
# Internally: adds a singleton time dim (T=1) and calls UNetModel3D
# -----------------------------

class UNet(nn.Module):
    """Wrapper that adapts your 3D UNetModel3D to the existing 2D training code.
    We map common 2D U-Net hyperparams onto UNetModel3D's constructor.
    """
    def __init__(
        self,
        in_channels=2,      # x + cond (kept for compatibility; internally we fix n_vars=1 and concat)
        out_channels=1,
        base_ch=64,         # mapped to model_dim
        ch_mults=(1, 2, 4), # mapped to dim_mults
        num_res_blocks=2,   # (unused by UNetModel3D; kept for compatibility)
        time_dim=256,       # (unused directly; UNetModel3D handles time embedding internally)
        groups=8,           # mapped to resnet_groups
        dropout=0.0,        # (unused if your model doesn’t expose it)
    ):
        super().__init__()
        self.inner = UNetModel3D(
            n_vars=1,                        # we predict 1-channel epsilon
            model_dim=base_ch,               # map base_ch -> model_dim
            dim_mults=tuple(ch_mults),       # map ch_mults -> dim_mults
            attn_heads=8,
            attn_dim_head=32,
            use_sparse_linear_attn=True,
            use_mid_attn=False,
            init_kernel_size=7,
            resnet_groups=groups,
            use_checkpoint=False,
            use_temp_attn=True,
            day_cond=False,                  # off by default here
            year_cond=False,                 # off by default here
            cond_map=True                    # we pass the condition map
        )

    def forward(self, x2d, cond2d, t):
        # x2d, cond2d: (B,1,H,W). Add T=1 for video model.
        x3d = x2d.unsqueeze(2)     # (B,1,1,H,W)
        c3d = cond2d.unsqueeze(2)  # (B,1,1,H,W)
        # Call your 3D model. It expects named args like timesteps and cond_map.
        eps3d = self.inner(x3d, timesteps=t, days=None, years=None, cond_map=c3d)  # (B,1,1,H,W)
        return eps3d.squeeze(2)    # back to (B,1,H,W)

# -----------------------------
# Diffusion wrapper (2D API)
# Uses standard DDPM math in 2D, but calls the wrapped 3D model under the hood.
# -----------------------------

class Diffusion(nn.Module):
    def __init__(self, model, img_channels=1, timesteps=1000, beta_schedule="linear", cond_loss_scaling: float = 0.0, lat_weighted_loss: bool = True):
        super().__init__()
        self.model = model
        self.img_channels = img_channels
        self.T = timesteps
        self.cond_loss_scaling = cond_loss_scaling
        self.lat_weighted_loss = lat_weighted_loss

        # Map our config to diffusers naming
        schedule_name = "squaredcos_cap_v2" if beta_schedule == "cosine" else "linear"
        self.scheduler = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule=schedule_name)
 torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    @torch.no_grad()
    def p_sample(self, x_t, cond, t):
        # Predict noise and take one reverse-diffusion step via diffusers scheduler
        eps_pred = self.model(x_t, cond, t)
        out = self.scheduler.step(eps_pred, t, x_t)
        return out.prev_sample
        else:
            posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, cond, shape, device):
        # shape: (B,1,H,W) — same as before
        B, _, H, W = shape
        x = torch.randn(shape, device=device)
        for tt in reversed(range(self.T)):
            t_tensor = torch.full((B,), tt, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t_tensor)
        return x

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        # diffusers expects timesteps shaped [B] on CPU or GPU equally fine
        return self.scheduler.add_noise(x0, noise, t), noise

    def loss(self, x0, cond):
        B = x0.size(0)
        # sample integer t in [0, T-1]
        t = torch.randint(0, self.T, (B,), device=x0.device).long()
        noise = torch.randn_like(x0)
        x_t = self.scheduler.add_noise(x0, noise, t)
        eps_pred = self.model(x_t, cond, t)

        # latitude-weighted noise MSE
        if self.lat_weighted_loss:
            H = eps_pred.shape[-2]
            w = _latitude_weights(H, eps_pred.device).view(1, 1, H, 1)
            mse = ((eps_pred - noise) ** 2 * w).mean()
        else:
            mse = F.mse_loss(eps_pred, noise)

        # reconstruct x0_hat using alpha_bar from scheduler
        # scheduler stores alphas_cumprod as a buffer (torch tensor)
        a_bar = self.scheduler.alphas_cumprod.to(x0.device)[t].view(-1,1,1,1)
        x0_hat = (x_t - torch.sqrt(1 - a_bar) * eps_pred) / (torch.sqrt(a_bar) + 1e-8)

        mean_hat = _area_weighted_mean(x0_hat)
        mean_true = _area_weighted_mean(x0)
        cond_loss = ((mean_hat - mean_true) ** 2).mean()

        return mse + self.cond_loss_scaling * cond_loss
