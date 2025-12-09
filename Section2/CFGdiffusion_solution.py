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

    x_noisy = extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    return x_noisy



@torch.no_grad()
def p_sample(model, x, t, t_index, coefficients, y=None, guidance_scale=1.0, noise=None):
    """
    One reverse step of diffusion with Classifier-Free Guidance (CFG).

    Purpose:
        Compute a single reverse step x_{t-1} from x_t using a guidance-weighted
        noise prediction. We combine unconditional and conditional noise estimates
        as:
            ε̂_theta = ε_uncond + s * (ε_cond − ε_uncond),
        where s is the guidance_scale (s=0 ⇒ unconditional; s=1 ⇒ standard
        conditional; s>1 ⇒ stronger class guidance).

    Model API:
        model(x, t, y=None) -> ε_theta (predicted noise).
        - Call once with y=None (unconditional).
        - Call once with y (e.g., MNIST class indices) for conditional prediction.

    Inputs:
        model: denoising network predicting noise ε_theta.
        x:     noisy batch at step t, shape (B, C, H, W).
        t:     per-sample timesteps, shape (B,).
        t_index: integer timestep used to decide whether to add sampling noise.
        coefficients: tuple (betas, sqrt_one_minus_alphas_cumprod,
                             sqrt_recip_alphas, posterior_variance).
        y:     class labels for conditioning (B,) or None for unconditional sampling.
        guidance_scale: float s controlling guidance strength.
        noise: optional external noise for this step (for deterministic testing).

    Returns:    
        sample: A sample from the distribution p_θ(x_{t-1} | x_t); mode if t=0
    """
    
    betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance = coefficients

    # --- predict eps with and without conditioning
    eps_uncond = model(x, t, y=None)
    if y is None or guidance_scale == 0.0:
        eps_theta = eps_uncond
    else:
        eps_cond = model(x, t, y)

        eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    coeff_1 = extract(sqrt_recip_alphas, t, x.shape)
    coeff_2 = extract(betas, t, x.shape) / extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    
    p_mean = coeff_1 * (x - coeff_2 * eps_theta)

    # no noise when t = 0
    if t_index == 0:
        return p_mean

    if noise is None:
        noise = torch.randn_like(x)
    variance = extract(posterior_variance, t, x.shape)
    sample = p_mean + torch.sqrt(variance) * noise
    return sample


@torch.no_grad()
def p_sample_loop(model, shape, timesteps, T, coefficients, y=None, guidance_scale=1.0, noise=None):
    """
    Full reverse diffusion loop with Classifier-Free Guidance (CFG).

    Purpose:
        Generate samples by iteratively applying CFG-based reverse steps from
        t = T-1 down to 0. At each step, combine unconditional and conditional
        noise predictions with guidance_scale to steer samples toward the class.

    Inputs:
        model: denoising network with signature model(x, t, y=None).
        shape: output tensor shape (B, C, H, W).
        timesteps: total number of diffusion steps (T).
        T: same as timesteps; kept for compatibility with the original template.
        coefficients: tuple needed by the reverse step (see p_sample).
        y: class labels for conditioning (B,) or None for unconditional sampling.
           Provide a per-sample label vector to condition each image in the batch.
        guidance_scale: CFG scale s (0 ⇒ unconditional, 1 ⇒ standard conditional,
                        >1 ⇒ stronger class alignment).
        noise: optional tensor of pre-fixed noises for reproducibility,
               shape (timesteps+1, B, C, H, W). If None, draws fresh noise.

    Returns:
        imgs: stacked intermediate samples across all steps,
              shape (T, B, C, H, W). imgs[-1] corresponds to x_0.
    """
    #NOTE OLD

    b = shape[0]

    img = torch.randn(shape, device=model.device) if noise is None else noise[0]
    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)), desc="Sampling", total=T, leave=False
    ):
        t = torch.full((b,), i, device=model.device, dtype=torch.long)

        if noise is None:
            img = p_sample(
                model, img, t, i, coefficients,
                y=y, guidance_scale=guidance_scale, noise=None
            )
        else:

            step_noise = noise[i]
            img = p_sample(
                model, img, t, i, coefficients,
                y=y, guidance_scale=guidance_scale, noise=step_noise
            )

        imgs.append(img.cpu())

    return torch.stack(imgs)




def p_losses(denoise_model, x_start, t, coefficients, y=None, p_uncond=0.1, noise=None, loss_type="l2"):
    """
    Training loss for Classifier-Free Guidance (CFG).

    Purpose:
        Train a single model to produce BOTH unconditional and conditional noise
        predictions by randomly dropping the label (“conditioning dropout”) with
        probability p_uncond. At each iteration:
            1) Sample noise ε ~ N(0, I) and form x_t = q(x_t | x_0, ε).
            2) Compute ε_uncond = model(x_t, t, y=None).
            3) Compute ε_cond   = model(x_t, t, y)        (if labels available).
            4) For each sample in the batch, pick ε_uncond or ε_cond according to
               a Bernoulli(p_uncond) mask to simulate dropping the condition.
            5) Minimize loss(ε̂_theta, ε), where ε̂_theta is the per-sample choice.

    Inputs:
        denoise_model: model(x, t, y=None) -> ε_theta.
        x_start: clean inputs x_0, shape (B, C, H, W).
        t:       per-sample timesteps, shape (B,). Will be cast to torch.long.
        coefficients: tuple required by q_sample.
        y:       class labels (B,) for MNIST. If None, trains unconditional only.
        p_uncond: probability of dropping the condition per sample.
        noise:   optional external ε for reproducibility; if None, sampled here.
        loss_type: "l2" (MSE, default), "l1", or "smooth_l1".

    Returns:
        loss: Loss for training the model
    """
    if t.dtype != torch.long:
        t = t.long()

    noise = torch.randn_like(x_start) if noise is None else noise
    x_noisy = q_sample(x_start, t, coefficients, noise)
    
    y_for_model = y
    mask = None
    if y is not None:
        mask = torch.rand((x_start.shape[0],), device=x_start.device) < p_uncond


    if y is None:
        eps_theta = denoise_model(x_noisy, t, y=None)
    else:
        eps_uncond =   denoise_model(x_noisy, t, y=None)
        eps_cond   = denoise_model(x_noisy, t, y=y)
        mask_reshaped = mask.reshape(-1, 1, 1, 1)
        eps_theta = torch.where(mask_reshaped, eps_uncond, eps_cond)


    if loss_type == "l1":
        loss = F.l1_loss(eps_theta, noise)
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(eps_theta, noise)
    else:  
        loss = F.mse_loss(eps_theta, noise)
        
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

