import schedules
import torch
import torch.nn as nn
import torch.nn.functional as F

timesteps = 200

# beta schedule
betas = schedules.cosine_beta_schedule(timesteps)

# alpha schedule
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas,dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), mode='constant', value=1)
sqrt_recip_alphas = torch.sqrt(1./alphas)

# for q(x_t|x_{t-1})
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# for q(x_{t-1}|x_t, x_0)
betas_hat = (1. - alphas_cumprod_prev)/(1. - alphas_cumprod) * betas

# 获得每一个变量在t时刻的值
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1,t.cpu())
    return out.reshape(batch_size, *((1,)* (len(x_shape)-1) )).to(t.device)

