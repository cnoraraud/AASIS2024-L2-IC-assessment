import torch
import torch.nn as nn
import math
import numpy as np
import torch_layers as tl
import torch_stacks as ts
import torch_wrapper as tw
import keywords_recipes as kwr

class TranscoderEmbedder(nn.Module):
    def __init__(self, input_shape, recipe):
        super().__init__()
        self.dims = recipe.get("dims") or 2
        self.embed_shape = recipe.get(kwr.embed_shape)
        self.attention_heads = recipe.get(kwr.attention_heads)
        self.pool_shape = (recipe.get(kwr.hidden_dim), 1)

        input_channels = 1 if self.dims > 1 else input_shape[1]

        self.input_embedding = tl.EmbeddingCNNnD(recipe, input_channels=input_channels, output_channels=self.embed_shape[0], dims=self.dims)
        self.encoder_stack = ts.CNNStack(
            recipe,
            input_channels=self.embed_shape[0],
            output_channels=self.embed_shape[0],
            in_n=2, 
            out_n=2
        )
        self.flatten_layer1 = tl.FlattenLayer(1, min(1 + self.dims - 1, 2))
        self.pool1 = tl.AdaptivePoolingCnD(recipe, channels=False, dims=2)
        self.transformer_stack = ts.TransformerStack(recipe, embed_dim=self.embed_shape[0], attention_heads=self.attention_heads)
        self.decoder_stack = ts.CNNStack(
            recipe,
            input_channels=self.embed_shape[0],
            output_channels=self.embed_shape[0],
            in_n=2, 
            out_n=2,
            K=3,
            S=1,
            dims=1
        )
        self.pool2 = tl.AdaptivePoolingCnD(recipe, embed_shape=self.pool_shape, dims=1)
        self.flatten_layer2 = tl.FlattenLayer(1, -1)

    def f_shape(self, x_shape):
        x_shape = self.input_embedding.f_shape(x_shape)
        x_shape = self.encoder_stack.f_shape(x_shape)
        x_shape = self.flatten_layer1.f_shape(x_shape)
        x_shape = self.pool1.f_shape(x_shape)
        x_shape = self.transformer_stack.f_shape(x_shape)
        x_shape = self.decoder_stack.f_shape(x_shape)
        x_shape = self.pool2.f_shape(x_shape)
        x_shape = self.flatten_layer2.f_shape(x_shape)
        return x_shape
    
    def forward(self, x):
        x = self.input_embedding(x)
        x = self.encoder_stack(x)
        x = self.flatten_layer1(x)
        x = self.pool1(x)
        x = self.transformer_stack(x)
        x = self.decoder_stack(x)
        x = self.pool2(x)
        x = self.flatten_layer2(x)
        return x

class BasicTDNNEmbedder(nn.Module):
    def __init__(self, input_shape, recipe):
        super().__init__()
        self.dims = recipe.get(kwr.dims) or 2
        self.hidden_dim = recipe.get(kwr.hidden_dim)
        self.pool_shape = (self.hidden_dim, 1)
        self.depth = recipe.get(kwr.depth)

        self.input_channels = 1 if self.dims > 1 else input_shape[1]
        self.ch_post_embed = max(1, recipe.get(kwr.channel_dim)//4)
        self.in_n = self.depth // 2
        self.out_n = self.depth - self.in_n


        self.input_embedding = tl.EmbeddingCNNnD(recipe, input_channels=self.input_channels, output_channels=self.ch_post_embed, dims=self.dims)
        self.cnn_stack = ts.CNNStack(
            recipe,
            input_channels=self.ch_post_embed,
            output_channels=self.hidden_dim,
            in_n=self.in_n,
            out_n=self.out_n,
            dims=self.dims
        )
        self.pool = tl.AdaptivePoolingCnD(recipe, channels=False, embed_shape=self.pool_shape, dims=2)
        self.flatten = tl.FlattenLayer(1, -1)
    
    def f_shape(self, x_shape):
        x_shape = self.input_embedding.f_shape(x_shape)
        x_shape = self.cnn_stack.f_shape(x_shape)
        x_shape = self.pool.f_shape(x_shape)
        x_shape = self.flatten.f_shape(x_shape)
        return x_shape
    
    def forward(self, x):
        x = self.input_embedding(x)
        x = self.cnn_stack(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

class TransformerTDNNEmbedder(nn.Module):
    def __init__(self, input_shape, recipe):
        super().__init__()
        self.dims = recipe.get(kwr.dims)
        assert self.dims == 1
        
        self.pool_shape = (recipe.get(kwr.hidden_dim), 1)

        input_channels = 1 if self.dims > 1 else input_shape[1]
        output_channels = tw.find_hidden_dim(input_channels, low=False)

        self.input_embedding = tl.EmbeddingCNNnD(recipe, input_channels=input_channels, output_channels=output_channels, dims=self.dims)
        embed_shape = self.input_embedding.f_shape(input_shape)
        self.transformer_stack = ts.TransformerStack(recipe, embed_dim=embed_shape[-2])
        self.pool = tl.AdaptivePoolingCnD(recipe, channels=False, embed_shape=self.pool_shape, dims=2)
        self.flatten = tl.FlattenLayer(1, -1)
    def f_shape(self, x_shape):
        x_shape = self.input_embedding.f_shape(x_shape)
        x_shape = self.transformer_stack.f_shape(x_shape)
        x_shape = self.pool.f_shape(x_shape)
        x_shape = self.flatten.f_shape(x_shape)
        return x_shape
    
    def forward(self, x):
        x = self.input_embedding(x)
        x = self.transformer_stack(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x   


class ATTDNNEmbedder(nn.Module):
    def __init__(self, input_shape, recipe):
        super().__init__()
        self.dims = recipe.get("dims") or 2
        self.embed_shape = recipe.get(kwr.embed_shape)
        self.pool_shape = (recipe.get(kwr.hidden_dim), 1)

        input_channels = 1 if self.dims > 1 else input_shape[1]

        self.input_embedding = tl.EmbeddingCNNnD(recipe, input_channels=input_channels, output_channels=self.embed_shape[0], dims=self.dims)
        self.flatten1 = tl.FlattenLayer(1, min(1 + self.dims - 1, 2))
        self.pool1 = tl.AdaptivePoolingCnD(recipe, channels=False, embed_shape=self.embed_shape, dims=2)
        self.transformer_stack = ts.TransformerStack(recipe, embed_dim=self.embed_shape[0])
        self.pool2 = tl.AdaptivePoolingCnD(recipe, channels=False, embed_shape=self.pool_shape, dims=2)
        self.flatten2 = tl.FlattenLayer(1, -1)
    
    def f_shape(self, x_shape):
        x_shape = self.input_embedding.f_shape(x_shape)
        x_shape = self.flatten1.f_shape(x_shape)
        x_shape = self.pool1.f_shape(x_shape)
        x_shape = self.transformer_stack.f_shape(x_shape)
        x_shape = self.pool2.f_shape(x_shape)
        x_shape = self.flatten2.f_shape(x_shape)
        return x_shape

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.flatten1(x)
        x = self.pool1(x)
        x = self.transformer_stack(x)
        x = self.pool2(x)
        x = self.flatten2(x)
        return x
    
class SimpleArchitecture(nn.Module):
    def __init__(self, input_shape, recipe, embedder):
        super().__init__()
        self.pre_embedder = tl.Identity()
        if not recipe.get(kwr.skip_preembed):
            self.pre_embedder = ts.FeedForwardStack(recipe, 1, input_shape[1], skip_res=False)
        self.embedder = embedder(input_shape, recipe)
        mlp_input_dim = self.embedder.f_shape(input_shape)[-1]
        mlp_output_dim = (recipe.get(kwr.output_dim) or 1) * (recipe.get(kwr.y_n) or 1)
        self.mlp_head = ts.FeedForwardHead(recipe, input_dim=mlp_input_dim, output_dim=mlp_output_dim, hidden_dim=False, dropout=0)
        self.output = tl.Identity()
        if recipe.get(kwr.classes):
            self.output = tl.StandardClassificationLayer(recipe, recipe.get(kwr.y_n))
    
    def f_shape(self, x_shape):
        x_shape = self.pre_embedder.f_shape(x_shape)
        x_shape = self.embedder.f_shape(x_shape)
        x_shape = self.mlp_head.f_shape(x_shape)
        x_shape = self.output.f_shape(x_shape)
        return x_shape
    
    def forward(self, x):
        x = self.pre_embedder(x)
        x = self.embedder(x)
        x = self.mlp_head(x)
        x = self.output(x)
        return x