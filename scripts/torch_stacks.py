import torch
import torch.nn as nn
import torch_layers as tl
import torch_wrapper as tw
import keywords_recipes as kwr
import math

class TransformerStack(nn.Module):
    def __init__(self, recipe, embed_dim, skip_posenc=None, attention_heads=None, hidden_dim=None, transformer_layers=None, dropout=None, activation=None):
        super().__init__()

        self.embed_dim = embed_dim

        self.attention_heads = tw.get(attention_heads, recipe, kwr.attention_heads)
        self.hidden_dim = tw.get(hidden_dim, recipe, kwr.hidden_dim)
        self.transformer_layers = tw.get(transformer_layers, recipe, kwr.transformer_layers)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.skip_posenc = tw.get(skip_posenc, recipe, kwr.skip_posenc)

        while self.embed_dim % self.attention_heads != 0:
            self.attention_heads -= 1
        
        self.pos_enc = tl.positional_encoding(skip=self.skip_posenc)(self.embed_dim)
        self.transformers = nn.ModuleList([tl.SelfAttentionLayer(recipe, embed_dim=self.embed_dim, attention_heads=self.attention_heads, dropout=self.dropout, hidden_dim=self.hidden_dim, activation=self.activation) for i in range(self.transformer_layers)])

    def f_shape(self, x_shape):
        for transformer in self.transformers:
            x_shape = transformer.f_shape(x_shape)
        return x_shape

    def forward(self, x):
        x = self.pos_enc(x)
        x = x.transpose(1,-1)
        for i, transformer in enumerate(self.transformers):
            x = transformer(x)
        x = x.transpose(-1,1)
        return x

class CNNStack(nn.Module):
    def __init__(self, recipe, input_channels, output_channels, in_n, out_n, peak_channels=None, K=None, S=None, dropout=None, cnn_repeats=None, pad_depth=None, activation=None, cnn_layer=None, dims=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.in_n = in_n
        self.out_n = out_n
        
        self.peak_channels = tw.get(peak_channels, recipe, kwr.channel_dim)
        self.K = tw.get(K, recipe, kwr.kernel)
        self.S = tw.get(S, recipe, kwr.stride)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.pad_depth = tw.get(pad_depth, recipe, kwr.pad_depth)
        self.cnn_repeats = tw.get(cnn_repeats, recipe, kwr.cnn_repeats)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.cnn_layer = tw.get(cnn_layer, recipe, kwr.cnn_layer)
        self.dims = tw.get(dims, recipe, kwr.dims)

        channel_list = tw.get_size_list(self.input_channels, self.peak_channels, self.output_channels, self.in_n, self.out_n)
        self.cnn_stack = nn.ModuleList()
        for in_channel, out_channel, i in zip(channel_list[:-1], channel_list[1:], range(len(channel_list) - 1)):
            do_pad = False if self.pad_depth is None else i >= self.pad_depth
            self.cnn_stack.append(self.cnn_layer(recipe,in_channel, out_channel, first_pad=do_pad, K=self.K, S=self.S, dropout=self.dropout, repeats=self.cnn_repeats, activation=self.activation, dims=self.dims))

    def f_shape(self, x_shape):
        for cnn in self.cnn_stack:
            x_shape = cnn.f_shape(x_shape)
        return x_shape
    
    def forward(self, x):
        for cnn in self.cnn_stack:
            x = cnn(x)
        return x

class FeedForwardStack(nn.Module):
    def __init__(self, recipe, dim, embed_length, output_length=None, hidden_layers=None, squeeze_r=None, dropout=None, activation=None, skip_res=None, skip_norm=None, skip_se=None):
        super().__init__()
        self.dim = dim
        self.embed_length = embed_length
        
        self.hidden_layers = tw.get(hidden_layers, recipe, kwr.hidden_layers)
        self.squeeze_r = tw.get(squeeze_r, recipe, kwr.squeeze_r)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.skip_res = tw.get(skip_res, recipe, kwr.skip_res)
        self.skip_norm = tw.get(skip_norm, recipe, kwr.skip_norm)
        self.skip_se = tw.get(skip_se, recipe, kwr.skip_se)

        self.in_n = self.hidden_layers // 2
        self.out_n = self.hidden_layers - self.in_n

        self.output_length = output_length or self.embed_length
        self.squeeze_length = tw.size_mult(tw.size_norm(math.sqrt(self.embed_length * self.output_length)), 1 / self.squeeze_r)
        self.length_list = tw.get_size_list(self.embed_length, self.squeeze_length, self.output_length, self.in_n, self.out_n)
        
        self.ff_stack = nn.ModuleList()
        for in_channel, out_channel in zip(self.length_list[:-1], self.length_list[1:]):
            self.ff_stack.append(
                tl.FeedForwardLayer(recipe, in_channel, out_channel, dropout=self.dropout, activation=self.activation, skip_res=self.skip_res, skip_norm=self.skip_norm, skip_se=self.skip_se)
            )
    
    def f_shape(self, x_shape):
        x_shape = list(x_shape)
        x_shape[self.dim] = self.output_length
        return torch.Size(x_shape)
    
    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        for ff in self.ff_stack:
            x = ff(x)
        x = torch.transpose(x, -1, self.dim)
        return x

class FeedForwardHead(nn.Module):
    def __init__(self, recipe, input_dim, output_dim, hidden_dim=None, activation=None, dropout=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim = tw.get(hidden_dim, recipe, kwr.hidden_dim) or tw.find_hidden_dim(input_dim)
        self.activation = tw.get(activation, recipe, kwr.activation)
        self.dropout = tw.get(dropout, recipe, kwr.dropout_rate)

        self.stack = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            self.activation(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
    
    def f_shape(self, x_shape):
        x_shape = list(x_shape)
        x_shape = [x_shape[0], self.output_dim]
        return torch.Size(x_shape)

    def forward(self, x):
        return self.stack(x)