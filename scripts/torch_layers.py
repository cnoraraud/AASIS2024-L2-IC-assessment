import torch.nn as nn
import torch
import math
import numpy as np
import torch_wrapper as tw
import keywords_recipes as kwr

def conv(dims):
    if dims == 1:
        return nn.Conv1d
    if dims == 2:
        return nn.Conv2d
    if dims == 3:
        return nn.Conv3d
    return nn.Identity

def layernorm(skip=False):
    if not skip:
        return nn.LayerNorm
    return Identity

def batchnorm(dims, skip=False):
    if not skip:
        if dims == 1:
            return nn.BatchNorm1d
        if dims == 2:
            return nn.BatchNorm2d
        if dims == 3:
            return nn.BatchNorm3d
    return nn.Identity

def lppool(dims):
    if dims == 1:
        return nn.LPPool1d
    if dims == 2:
        return nn.LPPool2d
    if dims == 3:
        return nn.LPPool3d
    return nn.Identity

def adaptivelppooling(dims, p):
    if p == torch.inf:
        if dims == 1:
            return nn.AdaptiveMaxPool1d
        if dims == 2:
            return nn.AdaptiveMaxPool2d
        if dims == 3:
            return nn.AdaptiveMaxPool3d
    else:
        if dims == 1:
            return nn.AdaptiveAvgPool1d
        if dims == 2:
            return nn.AdaptiveAvgPool2d
        if dims == 3:
            return nn.AdaptiveMaxPool3d
    return nn.Identity

def positional_encoding(skip=False):
    if not skip:
        return RotaryPositionalEmbeddings
    return nn.Identity

def se(skip=False):
    if not skip:
        return SELayer
    return nn.Identity

class FlattenLayer(nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
        self.flat = nn.Flatten(start, end)
    
    def f_shape(self, x_shape):
        x_shape = list(x_shape)
        end = self.end
        if end == -1:
            end = len(x_shape)
        flatten_shape = x_shape[self.start : end + 1]
        x_shape = x_shape[:self.start] + [math.prod(flatten_shape)] + x_shape[end + 1:]
        return torch.Size(x_shape)

    def forward(self, x):
        return self.flat(x)

class UnflattenLayer(nn.Module):
    def __init__(self, dim, sizes):
        super().__init__()
        self.dim = dim
        self.sizes = sizes
        self.unflat = nn.Unflatten(dim, sizes)
    
    def f_shape(self, x_shape):
        x_shape = list(x_shape)
        flatten_dim_input_size = x_shape[self.dim]
        flatten_dim_sizes = list(self.sizes)
        total_size = math.prod(flatten_dim_sizes)
        if total_size < 0:
            flatten_dim_sizes[flatten_dim_sizes.index(-1)] = flatten_dim_input_size // abs(total_size) 
        x_shape = x_shape[:self.dim] + flatten_dim_sizes + x_shape[self.dim + 1:]
        return torch.Size(x_shape)

    def forward(self, x):
        return self.unflat(x)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.id = nn.Identity()
    
    def f_shape(self, x_shape):
        return x_shape
    
    def forward(self, x):
        return self.id(x)

# https://github.com/aju22/RoPE-PyTorch/blob/main/RoPE.ipynb
class RotaryPositionalEmbeddings(nn.Module):
  def __init__(self, embed_dim: int, base: int = 10_000):
    super().__init__()
    self.base = base
    self.embed_dim = embed_dim
    self.cos_cached = None
    self.sin_cached = None

  def _build_cache(self, x: torch.Tensor):
    if self.cos_cached is not None and x.shape[2] <= self.cos_cached.shape[2]:
      return

    seq_len = x.shape[2]
    theta = 1. / (self.base ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)).to(x.device)
    seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
    idx_theta = torch.einsum('n,d->dn', seq_idx, theta)
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=0) 

    self.cos_cached = idx_theta2.cos()[None, :self.embed_dim, :]
    self.sin_cached = idx_theta2.sin()[None, :self.embed_dim, :]

  def _neg_half(self, x: torch.Tensor):
    d_2 = self.embed_dim // 2
    return torch.cat([-x[:,d_2:,:], x[:,:d_2,:]], dim=1)

  def forward(self, x: torch.Tensor):
    self._build_cache(x)
    neg_half_x = self._neg_half(x)
    x_rope = (x * self.cos_cached[:, :x.shape[1], :x.shape[2]]) + (neg_half_x * self.sin_cached[:, :x.shape[1], :x.shape[2]])
    return x_rope

#https://arxiv.org/pdf/1709.01507
class SELayer(nn.Module):
    def __init__(self, recipe, channel_dim, input_dims, dim, squeeze_r=None, pool_p=None, activation=None, dims=None):
        super().__init__()
        self.channel_dim = channel_dim
        self.input_dims = input_dims
        self.dim = dim
        
        self.squeeze_r = tw.get(squeeze_r, recipe, kwr.squeeze_r)
        self.pool_p = tw.get(pool_p, recipe, kwr.pool_p)
        self.activation = tw.get(activation, recipe, kwr.activation)
        
        self.hidden_dim = tw.size_mult(self.channel_dim, 1 / self.squeeze_r)
        self.pool_shape = tw.dim_fit((1, self.channel_dim), self.input_dims, repeat=0)
        
        self.se = nn.Sequential(
            adaptivelppooling(self.input_dims, self.pool_p)(self.pool_shape),
            nn.Linear(self.channel_dim, self.hidden_dim),
            self.activation(),
            nn.Linear(self.hidden_dim, self.channel_dim),
            nn.Sigmoid(),
        )
    
    def f_shape(self, x_shape):
        return x_shape
    
    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        s = self.se(x)
        x = x*s
        x = torch.transpose(x, -1, self.dim)
        return x

class FeedForwardLayer(nn.Module):
    def __init__(self, recipe, in_channel, out_channel, dims=None, dropout=None, activation=None, skip_res=None, skip_norm=None, skip_se=None):
        super().__init__()
        self.x = None
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.dims = tw.get(dims, recipe, kwr.dims)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.skip_res = tw.get(skip_res, recipe, kwr.skip_res)
        self.skip_norm = tw.get(skip_norm, recipe, kwr.skip_norm)
        self.skip_se = tw.get(skip_se, recipe, kwr.skip_se)

        if self.in_channel != self.out_channel:
            self.skip_res = True
        if not skip_res:
            self.skip_norm = False

        self.stack = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.Dropout(self.dropout),
            self.activation(),
            se(skip = self.skip_se)(recipe, out_channel, max(2, self.dims), -1),
        )
        self.norm = layernorm(skip_norm)(self.out_channel)
    
    def f_shape(self, x_shape):
        x_shape = list(x_shape)
        x_shape[-1] = self.out_channel
        return torch.Size(x_shape)
    
    def forward(self, x):
        res = x
        x = self.stack(x)
        if not self.skip_res:
            x = x + res
        x = self.norm(x)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, recipe, embed_dim, attention_heads=None, dropout=None, hidden_dim=None, activation=None):
        super().__init__()
        self.embed_dim = embed_dim

        self.attention_heads = tw.get(attention_heads, recipe, kwr.attention_heads)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.hidden_dim = tw.get(hidden_dim, recipe, kwr.hidden_dim)
        self.activation = tw.get(activation, recipe, kwr.activation)

        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.attention_heads, batch_first=True)
        self.drop1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            self.activation(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )
        self.drop2 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def f_shape(self, x_shape):
        return x_shape

    def forward(self, x):
        x_res1 = x
        x, x_w = self.mha(x,x,x)
        x = self.drop1(x)
        x = x + x_res1
        x = self.norm1(x)
        x_res2 = x
        x = self.ff(x)
        x = self.drop2(x)
        x = x + x_res2
        x = self.norm2(x)
        return x

class StandardClassificationLayer(nn.Module):
    def __init__(self, recipe, y_n=None, skip_logsoftmax=None):
        super().__init__()
        self.y_n = tw.get(y_n, recipe, kwr.y_n)
        self.skip_logsoftmax = tw.get(skip_logsoftmax, recipe, kwr.skip_logsoftmax)

        self.transform = Identity()
        if not self.skip_logsoftmax:
            self.transform = nn.LogSoftmax(-1)

    def f_shape(self, x_shape):
        x_shape = list(x_shape)
        x_shape = x_shape[:-1] + [self.y_n, x_shape[-1]//self.y_n]
        return torch.Size(x_shape)

    def forward(self, x):
        x_shape = list(x.shape)
        last_dim = x_shape[-1]
        x_shape[-1] = self.y_n
        x_shape.append(last_dim//self.y_n)
        x_shape = torch.Size(x_shape)
        x = x.reshape(x_shape)
        x = self.transform(x)
        return x

class EmbeddingCNNnD(nn.Module):
    def __init__(self, recipe, input_channels, output_channels, K=None, P=None, activation=None, dropout=None, skip_norm=None, pool_p=None, dims=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.K = tw.get(K, recipe, kwr.kernel)
        self.P = tw.get(P, recipe, kwr.pooling_kernel)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.skip_norm = tw.get(skip_norm, recipe, kwr.skip_norm)
        self.pool_p = tw.get(pool_p, recipe, kwr.pool_p)
        self.dims = tw.get(dims, recipe, kwr.dims)

        self.K = tw.dim_fit(self.K, self.dims, torch.Size)
        self.K_c = tw.dim_fit(1, self.dims, torch.Size)
        self.P = tw.dim_fit(self.P, self.dims, torch.Size)
       
        self.primary_conv = nn.Sequential(
            nn.Dropout(self.dropout),
            conv(self.dims)(self.input_channels, self.output_channels, kernel_size=self.K, stride=self.K),
            nn.Dropout(self.dropout),
            batchnorm(self.dims, skip=self.skip_norm)(self.output_channels),
            self.activation(),
            lppool(self.dims)(self.pool_p, kernel_size=self.P, stride=self.P)
        )

    def f_shape(self, x_shape):
        #in
        x_shape = list(x_shape)
        batch = x_shape[0]
        while 2 + self.dims > len(x_shape):
            x_shape.insert(1, 1)
        x_shape = x_shape[2:]
        x_shape = np.atleast_1d(x_shape)
        K_c = np.atleast_1d(self.K_c)
        K = np.atleast_1d(self.K)
        P = np.atleast_1d(self.P)
        #primary_conv
        x_shape = x_shape - (K - K_c)
        x_shape = np.ceil(x_shape / K)
        x_shape = x_shape - (P - K_c)
        x_shape = np.ceil(x_shape / P)
        #out
        x_shape = x_shape.astype(int).tolist()
        return torch.Size([batch, self.output_channels] + x_shape)
        
    def forward(self, x):
        while 2 + self.dims > x.dim():
            x = x.unsqueeze(1)
        x = self.primary_conv(x)
        return x
    
class CNNnD(nn.Module):
    def __init__(self, recipe, channel_in, channel_out, first_pad=False, K=None, S=None, dropout=None, skip_norm=None, skip_se=None, repeats=None, activation=None, dims=None):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out

        self.K = tw.get(K, recipe, kwr.kernel)
        self.S = tw.get(S, recipe, kwr.stride)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.skip_norm = tw.get(skip_norm, recipe, kwr.skip_norm)
        self.skip_se = tw.get(skip_se, recipe, kwr.skip_se)
        self.repeats = tw.get(repeats, recipe, kwr.cnn_repeats)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.dims = tw.get(dims, recipe, kwr.dims)

        
        self.K = tw.dim_fit(self.K, self.dims, torch.Size)
        self.S = tw.dim_fit(self.S, self.dims, torch.Size)
        self.K_c = tw.dim_fit(1, self.dims, torch.Size)
        self.padding_0 = 0
        self.padding_r = torch.Size((np.atleast_1d(self.K)-np.atleast_1d(self.K_c))//2)
        if first_pad:
            self.padding_0 = self.padding_r

        self.primary_conv = nn.Sequential(
            nn.Dropout(self.dropout),
            conv(self.dims)(self.channel_in, self.channel_out, kernel_size=self.K, stride=self.S, padding=self.padding_0),
            se(skip = self.skip_se)(recipe, self.channel_out, max(2, self.dims), 1),
            batchnorm(self.dims, skip=self.skip_norm)(self.channel_out),
            self.activation()
        )
        
        self.repeater_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(self.dropout),
                    conv(self.dims)(self.channel_out, self.channel_out, kernel_size=self.K, stride=self.S, padding=self.padding_r),
                    se(skip = self.skip_se)(recipe, self.channel_out, max(2, self.dims), 1),
                    batchnorm(self.dims, skip=self.skip_norm)(self.channel_out),
                    self.activation(),
                ) for i in range(self.repeats)
            ]
        )

    def f_shape(self, x_shape):
        #in
        batch = x_shape[0]
        x_shape = x_shape[2:]
        x_shape = np.atleast_1d(x_shape)
        K_c = np.atleast_1d(self.K_c)
        K = np.atleast_1d(self.K)
        S = np.atleast_1d(self.S)
        #conv
        x_shape = x_shape - (K - K_c)
        x_shape = np.ceil(x_shape / S)
        #out
        x_shape = x_shape.astype(int).tolist()
        return torch.Size([batch, self.channel_out] + x_shape)
    
    def forward(self, x):
        x = self.primary_conv(x)
        for repeater_conv in self.repeater_convs:
            x = repeater_conv(x)
        return x

class ResCNNnD(nn.Module):
    def __init__(self, recipe, channel_in, channel_out, K=None, S=None, dropout=None, skip_norm=None, skip_se=None, repeats=None, activation=None, dims=None):
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        
        self.K = tw.get(K, recipe, kwr.kernel)
        self.S = tw.get(S, recipe, kwr.stride)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.skip_norm = tw.get(skip_norm, recipe, kwr.skip_norm)
        self.skip_se = tw.get(skip_se, recipe, kwr.skip_se)
        self.repeats = tw.get(repeats, recipe, kwr.cnn_repeats)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.dims = tw.get(dims, recipe, kwr.dims)
        
        self.K = tw.dim_fit(self.K, dims, torch.Size)
        self.S = tw.dim_fit(self.S, self.dims, torch.Size)
        self.K_c = tw.dim_fit(1, self.dims, torch.Size)
        self.padding = torch.Size((np.atleast_1d(self.K)-np.atleast_1d(self.K_c))//2)

        self.primary_conv = conv(self.dims)(self.channel_in, self.channel_out, kernel_size=self.K, stride=self.S)
        
        self.repeater_convs = nn.ModuleList(
            [
                nn.Sequential(
                    conv(self.dims)(self.channel_out, self.channel_out, kernel_size=self.K, stride=tw.dim_fit(1, self.dims, torch.Size), padding=self.padding),
                    self.activation(),
                    nn.Dropout(self.dropout),
                ) for i in range(self.repeats)
            ]
        )
        self.norms = nn.ModuleList([batchnorm(self.dims, skip=self.skip_norm)(self.channel_out) for i in range(self.repeats)])

    def f_shape(self, x_shape):
        #in
        batch = x_shape[0]
        x_shape = x_shape[2:]
        x_shape = np.atleast_1d(x_shape)
        K_c = np.atleast_1d(self.K_c)
        K = np.atleast_1d(self.K)
        S = np.atleast_1d(self.S)
        #conv1
        x_shape = x_shape - (K - K_c)
        x_shape = np.ceil(x_shape / S)
        #out
        x_shape = x_shape.astype(int).tolist()
        return torch.Size([batch, self.channel_out] + x_shape)
    
    def forward(self, x):
        x = self.primary_conv(x)
        for repeater_conv, norm in zip(self.repeater_convs, self.norms):
            res_x = x
            x = repeater_conv(x)
            x = x + res_x
            x = norm(x)
        return x        
    
class AdaptivePoolingCnD(nn.Module):
    def __init__(self, recipe, channels=True, embed_shape=None, dropout=None, activation=None, pool_p=None, dims=None, skip_norm=None):
        super().__init__()
        self.channels = channels

        self.embed_shape = tw.get(embed_shape, recipe, kwr.embed_shape)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.pool_p = tw.get(pool_p, recipe, kwr.pool_p)
        self.dims = tw.get(dims, recipe, kwr.dims)
        self.skip_norm = tw.get(skip_norm, recipe, kwr.skip_norm)
        
        self.embed_shape = tw.dim_fit(self.embed_shape, self.dims, torch.Size, repeat=0)
        
        self.pool = nn.Sequential(
            adaptivelppooling(self.dims, self.pool_p)(self.embed_shape),
            nn.Dropout(self.dropout),
            self.activation(),
            layernorm(skip_norm)(self.embed_shape),
        )

    def f_shape(self, x_shape):
        x_shape = list(x_shape)
        embed_shape = list(np.atleast_1d(self.embed_shape))
        if self.channels:
            x_shape = x_shape[:2]
        else:
            x_shape = x_shape[:1]
        return torch.Size(x_shape + embed_shape)
    
    def forward(self, x):
        if not self.channels:
            x = x.unsqueeze(1)
        x = self.pool(x)
        if not self.channels:
            x = x.squeeze(1)
        return x