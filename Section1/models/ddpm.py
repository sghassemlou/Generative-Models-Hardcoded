import math
import torch
from torch import nn
import torch.nn.functional as F

### WARNING: DO NOT EDIT THE CLASS NAME, INITIALIZER, AND GIVEN INPUTS AND ATTRIBUTES. OTHERWISE, YOUR TEST CASES CAN FAIL. ###


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(self.act(self.norm1(x)))
        time_term = self.time_proj(self.act(time_emb))
        h = h + time_term[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, downsample):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_channels)
        self.res2 = ResidualBlock(out_channels, out_channels, time_channels)
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if downsample else None
        )

    def forward(self, x, time_emb):
        h = self.res1(x, time_emb)
        h = self.res2(h, time_emb)
        skip = h
        if self.downsample is not None:
            h = self.downsample(h)
        return skip, h


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, upsample):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_channels)
        self.res2 = ResidualBlock(out_channels, out_channels, time_channels)
        self.upsample = (
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if upsample else None
        )

    def forward(self, x, skip, time_emb):
        h = torch.cat([x, skip], dim=1)
        h = self.res1(h, time_emb)
        h = self.res2(h, time_emb)
        if self.upsample is not None:
            h = self.upsample(h)
        return h


class DenoiseUNet(nn.Module):
    def __init__(self, image_channels=3, base_channels=64, time_channels=256, timesteps=1000):
        super().__init__()
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.time_channels = time_channels
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas) ## betas is the beta values for the DDPM model
        self.register_buffer("alphas_cumprod", alphas_cumprod) ## alphas_cumprod is the cumulative product of the alpha values for the DDPM model
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)) ## sqrt_alphas_cumprod is the square root of the cumulative product of the alpha values for the DDPM model
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)) ## sqrt_one_minus_alphas_cumprod is the square root of 1 - alphas_cumprod
        ### use self.sqrt_alphas_cumprod and self.sqrt_one_minus_alphas_cumprod to compute the mean and standard deviation of the noisy sample in the q_sample function
        self.time_embedding = self._build_time_embedding(time_channels)
        self.model = self._build_network(image_channels, base_channels, time_channels)

    def _build_time_embedding(self, time_channels):
        time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_channels),
            nn.Linear(time_channels, time_channels * 4),
            nn.SiLU(),
            nn.Linear(time_channels * 4, time_channels),
        )
        return time_embedding

    def _build_network(self, image_channels, base_channels, time_channels):
        network = nn.ModuleDict()
        network["init"] = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        network["down0"] = DownBlock(base_channels, base_channels, time_channels, downsample=True)
        network["down1"] = DownBlock(base_channels, base_channels * 2, time_channels, downsample=True)
        network["down2"] = DownBlock(base_channels * 2, base_channels * 4, time_channels, downsample=False)
        network["mid"] = ResidualBlock(base_channels * 4, base_channels * 4, time_channels)
        network["up2"] = UpBlock(base_channels * 8, base_channels * 2, time_channels, upsample=True)
        network["up1"] = UpBlock(base_channels * 4, base_channels, time_channels, upsample=True)
        network["up0"] = UpBlock(base_channels * 2, base_channels, time_channels, upsample=False)
        network["out"] = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1),
        )
        return network

    def q_sample(self, x_start, t, noise):
        shape = (t.size(0),) + (1,) * (x_start.dim() - 1)
        mean = self.sqrt_alphas_cumprod[t].view(shape) * x_start  # NOTE: compute the mean of q(x_t | x_0), hint: use self.sqrt_alphas_cumprod
        std = self.sqrt_one_minus_alphas_cumprod[t].view(shape)  # NOTE: compute the standard deviation term, hint: use self.sqrt_one_minus_alphas_cumprod
        return mean + std * noise  # NOTE: return the noisy sample x_t

    def forward(self, batch):
        x0 = batch["images"]
        t = batch.get("timesteps")
        if t is None:
            t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device).long()  # NOTE: draw random timesteps for the batch
        noise = batch.get("noise")
        if noise is None:
            noise = torch.randn_like(x0)  # NOTE: draw Gaussian noise for the forward process
        xt = self.q_sample(x0, t, noise)  # NOTE: sample from q(x_t | x_0)
        time_emb = self.time_embedding(t)
        h0 = self.model["init"](xt)  # NOTE: apply the initial convolution
        skip0, h1 = self.model["down0"](h0, time_emb)  # NOTE: apply the first down block
        skip1, h2 = self.model["down1"](h1, time_emb)  # NOTE: apply the second down block
        skip2, h3 = self.model["down2"](h2, time_emb)  # NOTE: apply the last down block
        h_mid = self.model["mid"](h3, time_emb)  # NOTE: apply the middle residual block
        h = self.model["up2"](h_mid, skip2, time_emb)  # NOTE: apply the first up block
        h = self.model["up1"](h, skip1, time_emb)  # NOTE: apply the second up block
        h = self.model["up0"](h, skip0, time_emb)  # NOTE: apply the final up block
        pred_noise = self.model["out"](h)  # NOTE: map to the predicted noise
        loss = F.mse_loss(pred_noise, noise, reduction='mean')
        return {
            "loss": loss,
            "predicted_noise": pred_noise,
            "noisy_images": xt,
            "timesteps": t,
        }