import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def extract(a, t, x_shape):
    """
    Takes a data tensor `a` and an index tensor `t`, and returns a new tensor
    whose i^th element is just a[t[i]]. This will be useful when we want to choose
    the alphas or betas corresponding to different indices `t` in a batched manner
    without for loops.

    Inputs:
        a: Tensor, generally of shape (batch_size,)
        t: Tensor, generally of shape (batch_size,)
        x_shape: Shape of the data, generally (batch_size, 3, 32, 32)
    Returns:
        out: Tensor of shape (batch_size, 1, 1, 1) generally, where the number of 1s 
             is determined by the number of dimensions in x_shape.
             out[i] contains a[t[i]]
    """

    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, coefficients, noise=None):
    """
    Forward Diffusion Sampling Process.

    Inputs:
        x_start: Tensor of original images of size (batch_size, 3, 32, 32)
        t: Tensor of timesteps, of shape (batch_size,)
        noise: Optional tensor of same shape as x_start, signifying that the noise 
               to add is already provided.
        coefficients: 2-tuple
    Returns:
        x_noisy: Tensor of noisy images of size (batch_size, 3, 32, 32)
                 x_noisy[i] is sampled from q(x_{t[i]} | x_start[i])
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = coefficients
    sqrt_alpha_bar_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha_bar_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    x_noisy = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    x_noisy = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    return x_noisy


def p_sample(model, x, t, t_index, coefficients, noise=None):
    """
    Given the denoising model, batched input x, and time-step t, returns a slightly 
    denoised sample at time-step t-1.

    Inputs:
        model: The denoising (parameterized noise) model
        x: Batched noisy input at time t; size (batch_size, 3, 32, 32)
        t: Batched time steps; size (batch_size,)
        t_index: Single time-step, whose batched version is present in t
        coefficients: 4-tuple
    Returns:
        sample: A sample from the distribution p_θ(x_{t-1} | x_t); mode if t=0
    """
    with torch.no_grad():
        betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance = (
            coefficients
        )
        predicted_noise = model(x, t)
        coeff_1 = extract(sqrt_recip_alphas, t, x.shape)
        coeff_2 = extract(betas, t, x.shape) / extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        p_mean = coeff_1 * (x - coeff_2 * predicted_noise)
        if t_index == 0:
            sample = p_mean
        else:
            if noise is None:
                noise = torch.randn_like(x)
            
            variance = extract(posterior_variance, t, x.shape)
            sample = p_mean + torch.sqrt(variance) * noise 

        return sample


def p_sample_loop(model, shape, timesteps, T, coefficients, noise=None):
    """
    Given the model and the shape of the image, returns a sample from the data 
    distribution by running through the backward diffusion process.

    Inputs:
        model: The denoising model
        shape: Shape of the samples; set as (batch_size, 3, 32, 32)
        noise: (timesteps+1, batch_size, 3, 32, 32)
    Returns:
        imgs: Samples obtained, as well as intermediate denoising steps, 
              of shape (T, batch_size, 3, 32, 32)
    """
    with torch.no_grad():
        b = shape[0]
        img = torch.randn(shape, device=model.device) if noise is None else noise[0]
        imgs = []

        for i in tqdm(
            reversed(range(0, timesteps)), desc="Sampling", total=T, leave=False
        ):
            t = torch.full((b,), i, device=model.device, dtype=torch.long)
            if noise is None:
                img = p_sample(model, img, t, i, coefficients, noise=None)
            else:
                img = p_sample(model, img, t, i, coefficients, noise=noise[i]) 
            imgs.append(img.cpu())

        return torch.stack(imgs)


def p_losses(denoise_model, x_start, t, coefficients, noise=None):
    """
    Returns the loss for training of the denoising model.

    Inputs:
        denoise_model: The parameterized model
        x_start: The original images; size (batch_size, 3, 32, 32)
        t: Timesteps (can be different at different indices); size (batch_size,)
    Returns:
        loss: Loss for training the model
    """
    noise = torch.randn_like(x_start) if noise is None else noise

    x_noisy = q_sample(x_start, t, coefficients, noise) 
    predicted_noise = denoise_model(x_noisy, t)
    
    loss = F.smooth_l1_loss(noise, predicted_noise)

    return loss


def t_sample(timesteps, batch_size, device):
    """
    Returns randomly sampled timesteps.

    Inputs:
        timesteps: The max number of timesteps; T
        batch_size: Batch size used in training
    Returns:
        ts: Tensor of size (batch_size,) containing timesteps randomly sampled 
            from 0 to timesteps-1
    """
    ts = torch.randint(0, timesteps, (batch_size,), device=device).long()
    return ts

