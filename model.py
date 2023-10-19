import math
from inspect import isfunction
from functools import partial

%matplotlib inline
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

def default(val,d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

# a residual network
class Residual(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
    def forward(self,x,*args,**kwargs):
        return self.fn(x,*args,**kwargs) + x

# up sampling
def Upsample(dim):
    # h_out = 2 * h_in
    return nn.ConvTranspose2d(dim,dim,4,2,1)

#down sampling
def Downsample(dim):
    # h_out = 1/2 * h_in
    return nn.Conv2d(dim,dim,4,2,1)

# position embeadding
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim//2
        embeddings = math.log(10000)/(half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:,None] * embeddings[None,:]
        embeddings = torch.cat((embeddings.sin(),embeddings.cos()),dim=-1)
        return embeddings

# basic network, containing: convolution, group normalization, activation
class Block(nn.Module):
    def __init__(self, dim, dim_out, num_groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim,dim_out, kernel_size=3,padding=1)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim_out)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x*(scale + 1) + shift
        
        x = self.act(x)

# ResNet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *,time_embed_dim=None, num_groups=8) -> None:
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, dim_out)
            )
            if time_embed_dim is not None else None            
        )

        self.block1 = Block(dim, dim_out, num_groups=num_groups)
        self.block2 = Block(dim_out, dim_out, num_groups=num_groups)
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim!=dim_out else nn.Identity()

    def forward(self, x, time_embed=None):
        h = self.block1(x)

        if (self.mlp is not None) and (time_embed is not None):
            time_embed = self.mlp(time_embed)
            h = time_embed[:,:,None,None] + h
        
        h = self.block2(h)

        return h + self.res_conv(x)

"""
    class ConvNeXtBlock(nn.Module):
        pass
"""

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, dim_per_head=32):
        super().__init__()
        self.scale = dim_per_head ** -0.5
        self.num_heads = num_heads
        hidden_dim = num_heads * dim_per_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self,x):
        b,c,h,w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q,k,v = map(
            lambda t: rearrange(t, 'b (h d) x y -> b h d (x y)', h=self.num_heads), qkv
        )
        q = q * self.scale
        
        sim = einsum("b h d i, b h d j -> b h i j",q,k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        atten = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", atten, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h,y=w)
        
        return self.to_out(out)
    
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dim_per_head=32):
        super().__init__()
        self.scale = dim_per_head ** -0.5
        self.num_heads = num_heads
        hidden_dim = num_heads*dim_per_head
        
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.GroupNorm(1, dim)
        )
    
    def forward(self, x):
        b,c,h,w = x.shape
        # (b,hidden_dim,h,w) -> (b,hidden_dim*3, h, w)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange("b (h d) x y -> b h d (x y)",h=self.num_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        # 这里采取不同的处理复杂度会不同,主要的复杂度在最后一维
        context = einsum(" b h d n, b h e n -> b h d e",k,v)
        
        out = einsum("b h d n, b h d e -> b h e n", q,context)
        out = rearrange(out, "b h d (x y) -> b (h d) x y ",x=h,y=w)
        return self.to_out

class PreNorm(nn.Module):
    def __init__(self, fn ,dim):
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    