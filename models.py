import math
from inspect import isfunction
from functools import partial

# %matplotlib inline
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
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3,padding=1)
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

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dim_out, *,time_embed_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp=(
            nn.Sequential(nn.GELU(), nn.Linear(time_embed_dim, dim))
            if time_embed_dim is not None
            else None
        )
        # depth-wise conv
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1,dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out,kernel_size=3,padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if not dim==dim_out else nn.Identity()
    
    def forward(self, x, time_embed=None):
        h = self.ds_conv(x)

        if self.mlp is not None and time_embed is not None:
            condition = self.mlp(time_embed)
            h = h + condition[:,:,None,None]
        
        h = self.net(h)
        return h + self.res_conv(x)


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
            lambda t: rearrange(t, "b (h d) x y -> b h d (x y)",h=self.num_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        # 这里采取不同的处理复杂度会不同,主要的复杂度在最后一个维度
        context = einsum(" b h d n, b h e n -> b h d e",k,v)
        
        out = einsum("b h d n, b h d e -> b h e n", q,context)
        out = rearrange(out, "b h d (x y) -> b (h d) x y ",x=h,y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim ,fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    
    def forward(self, x):
        return self.fn(self.norm(x))

"""
    首先，对噪声图像进行卷积处理，对噪声水平进行进行位置编码（embedding）
    然后，进入一个序列的下采样阶段，每个下采样阶段由两个ResNet/ConvNeXT模块+分组归一化+注意力模块+残差链接+下采样完成。
    在网络的中间层，再一次用ResNet/ConvNeXT模块，中间穿插着注意力模块(Attention)。
    下一个阶段，则是序列构成的上采样阶段，每个上采样阶段由两个ResNet/ConvNeXT模块+分组归一化+注意力模块+残差链接+上采样完成。
    最后，一个ResNet/ConvNeXT模块后面跟着一个卷积层。
"""
class UNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1,2,4,8),
        channels=3,
        with_time_embed=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()
        self.channels = channels
        
        #为什么默认的init_dim是 dim//3*2
        init_dim = default(init_dim, dim//3*2)
        
        self.init_conv = nn.Conv2d(channels, init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *map(lambda t: dim*t, dim_mults)]
        
        # 巧妙地将dim序列打包成 等同于in = dim[k], out = dim[k+1]
        in_out = list(zip(dims[:-1],dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNeXtBlock,mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, num_groups=resnet_block_groups)
        
        # time embeddings
        # 这里的dimension设计是为什么
        if with_time_embed:
            time_dim = dim*4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        
        # layers
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # 定义UNet下采样部分
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out)-1

            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_out, time_embed_dim=time_dim),
                    block_klass(dim_out,dim_out, time_embed_dim=time_dim),
                    Residual(PreNorm(dim_out,LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ])
            )

        # 定义中间部分
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_embed_dim=time_dim)
        self.mid_atten = Residual(PreNorm(mid_dim, Attention(mid_dim))) # 这里为何选择这个 不用Linear Attention
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_embed_dim=time_dim)

        # 定义UNet上采样部分
        for idx,(dim_in, dim_out) in enumerate(reversed(in_out[1:])): # 最后一层做特殊处理，所以从1开始
            is_last = idx == len(in_out)-1
            
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out*2, dim_in, time_embed_dim=time_dim), #dim_out为什么乘以2---这里的dim选择有问题
                    block_klass(dim_in, dim_in,time_embed_dim=time_dim),
                    Residual(PreNorm(dim_in,LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                ])
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim, time_embed_dim=None),
            nn.Conv2d(dim, out_dim, 1)
        )
    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if self.time_mlp is not None else None

        h = []

        # downsample
        for block1, block2, atten, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = atten(x)
            h.append(x)
            x = downsample(x)
        # bottle neck
        x = self.mid_block1(x, t)
        x = self.mid_atten(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, atten, upsample in self.ups:
            x = torch.cat((x,h.pop()),dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = atten(x)
            x = upsample(x)

        return self.final_conv(x)

    

            