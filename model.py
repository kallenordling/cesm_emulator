
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

# Import your 3D UNet backbone
from video_net import UNetModel3D

# -----------------------------
# Helpers
# -----------------------------

def _latitude_weights(H: int, device):
    lat = torch.linspace(-90, 90, steps=H, device=device)
    w = torch.cos(torch.deg2rad(lat)).clamp_min(0)
    return w / (w.mean() + 1e-8)

def _area_weighted_mean(x):
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    w = _latitude_weights(H, x.device).view(1, 1, H, 1)
    return (x * w).mean(dim=(-2, -1))  # [B, C]

# -----------------------------
# UNet wrapper (if needed)
# -----------------------------

class UNet(nn.Module):
    """Thin wrapper if your training code expects UNet() type."""
    def __init__(self, **kwargs):
        super().__init__()
        self.model = UNetModel3D(**kwargs)

    def forward(self, x_t, cond, t):
        # Adjust to your UNetModel3D forward signature if different
        return self.model(x_t, cond, t)

# -----------------------------
# Diffusion with diffusers Scheduler
# -----------------------------

class Diffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        img_channels: int = 1,
        timesteps: int = 1000,
        beta_schedule: str = "linear",   # "linear" or "cosine"
        cond_loss_scaling: float = 0.0,
        lat_weighted_loss: bool = True,
    ):
        super().__init__()
        self.model = model
        self.img_channels = img_channels
        self.T = timesteps
        self.cond_loss_scaling = cond_loss_scaling
        self.lat_weighted_loss = lat_weighted_loss

        schedule_name = "squaredcos_cap_v2" if beta_schedule == "cosine" else "linear"
        self.scheduler = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule=schedule_name)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, shape=None, device=None):
        device = device or (cond.device if isinstance(cond, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        B = cond.size(0)
        if shape is None:
            H, W = cond.shape[-2], cond.shape[-1]
            shape = (B, self.img_channels, H, W)

        x = torch.randn(shape, device=device)
        self.scheduler.set_timesteps(self.T, device=device)
        for t in self.scheduler.timesteps:
            t_tensor = torch.full((B,), int(t), device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t_tensor)
        return x

    def p_sample(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        # Predict noise and take one reverse-diffusion step via diffusers scheduler
        eps_pred = self.model(x_t, cond, t)
        out = self.scheduler.step(eps_pred, t, x_t)
        return out.prev_sample

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.scheduler.add_noise(x0, noise, t), noise

    def loss(self, x0: torch.Tensor, cond: torch.Tensor):
        B = x0.size(0)
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
        a_bar = self.scheduler.alphas_cumprod.to(x0.device)[t].view(-1, 1, 1, 1)
        x0_hat = (x_t - torch.sqrt(1 - a_bar) * eps_pred) / (torch.sqrt(a_bar) + 1e-8)

        mean_hat = _area_weighted_mean(x0_hat)
        mean_true = _area_weighted_mean(x0)
        cond_loss = ((mean_hat - mean_true) ** 2).mean()

        return mse + self.cond_loss_scaling * cond_loss
