# mostly copy and pasted from https://github.com/brianfitzgerald/jax-mmdit/blob/main

import jax
from jax import Array
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class TimestepEmbedding(nn.Module):
    hidden_size: int
    frequency_embedding_size: int
    
    @nn.compact
    def __call__(self, t: Array) -> Array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb: Array = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.silu,
                nn.Dense(self.hidden_size)
            ]
        )(t_freq)
        return t_emb
    
    @staticmethod
    def timestep_embedding(t: Array, freq_embed_size: int, max_period=10000):
        half = freq_embed_size // 2
        frequencies = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half) / half)
        args = t[:, None] * frequencies[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if freq_embed_size % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding
    

class Attention(nn.Module):
    dim: int
    n_heads: int
    dtype: jnp.dtype
    
    def setup(self):
        self.head_dim = self.dim // self.n_heads
        
        self.wq = nn.Dense(
            self.n_heads * self.head_dim, use_bias=False, dtype=self.dtype
        )
        self.wk = nn.Dense(
            self.n_heads * self.head_dim, use_bias=False, dtype=self.dtype
        )
        self.wv = nn.Dense(
            self.n_heads * self.head_dim, use_bias=False, dtype=self.dtype
        )
        self.wo = nn.Dense(self.dim, use_bias=False)
        self.q_norm = nn.LayerNorm(dtype=self.dtype)
        self.k_norm = nn.LayerNorm(dtype=self.dtype)
        
    @staticmethod
    def reshape_for_broadcast()
        