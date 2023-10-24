import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import *
from tqdm import tqdm

# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def pixel_loss(denoise_model, x_start, t, noise=None, loss_type='l1'):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t) # UNet

    # original papar use huber-loss
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    if loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    if loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss

    
# sample
@torch.no_grad()
def p_sample(denoise_model, x, t, t_idx):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t/ sqrt_one_minus_alphas_cumprod_t * denoise_model(x,t)
    )
    
    if t_idx == 0: #最后一步不再去噪
        return model_mean
    else:
        betas_hat_t = extract(betas_hat, t, x.shape)
        noise = torch.randn_like(x)
        
        return model_mean + torch.sqrt(betas_hat_t) * noise

@torch.no_grad()
def p_sample_loop(denoise_model, shape):
    device = next(denoise_model.parameters()).device

    b = shape[0]

    # start from random noise
    imgs = []
    img = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(0, timesteps))):
        img = p_sample(denoise_model, img, torch.full((b,), i, device=device), i)
        imgs.append(img.cpu().numpy()) #转换成numpy，方便之后转换成Image格式
    
    return imgs

@torch.no_grad()
def sample(denoise_model, img_size, batch_size=16, channels=3):
    return p_sample_loop(denoise_model, shape=(batch_size, channels, img_size, img_size))