# a mish-mash of
# https://github.com/brianfitzgerald/jax-mmdit/blob/main
# https://github.com/kvfrans/jax-diffusion-transformer/blob/main
import jax
from jax import Array
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
from einops import repeat, rearrange


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
    

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    mlp_ratio: int
    dtype: jnp.dtype
    
    @nn.compact
    def __call__(self, x: Array):
        x1 = nn.Dense(
            features=self.dim * self.mlp_ratio,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.xavier_uniform()
        )(x)
        x1 = nn.gelu(x1)
        x2 = nn.Dense(
            features=self.dim * self.mlp_ratio,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.xavier_uniform()
        )(x)
        x3 = x1 * x2
        output = nn.Dense(
            features=self.dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.xavier_uniform()
            )
        return output
    

    
        