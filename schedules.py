import torch
import torch.nn as nn

def cosine_beta_schedule(timesteps, s=0.008):
    # as imporved ddpm
    steps = timesteps + 1
    x = torch.linspace(0,timesteps, steps)
    f_t = torch.cos((x/timesteps + s)/(1+s)*torch.pi/2)**2
    alpha_cumprod = f_t/f_t[0]
    betas = 1- (alpha_cumprod[1:]/alpha_cumprod[:-1])

    return torch.clip(betas, max=0.999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps)**2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6,6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

