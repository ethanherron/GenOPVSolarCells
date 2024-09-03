# a mish-mash of
# https://github.com/brianfitzgerald/jax-mmdit/blob/main
# https://github.com/kvfrans/jax-diffusion-transformer/blob/main
import jax
from jax import Array
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


class TimestepEmbedding(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256
    
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
    mlp_dim: int
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x: Array):
        out_dim = x.shape[-1]
        x = nn.Dense(
            features=self.mlp_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.xavier_uniform()
        )(x)
        x = nn.gelu(x)
        output = nn.Dense(
            features=out_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.xavier_uniform()
        )(x)
        return output
    

class PatchEmbed(nn.Module):
    patch_size: int
    embed_dim: int
    
    @nn.compact
    def __call__(self, x: Array):
        B, H, W, C = x.shape
        patch_tuple = (self.patch_size, self.patch_size)
        num_patches = (H // self.patch_size)
        x = nn.Conv(self.embed_dim, patch_tuple, patch_tuple, use_bias=False, padding='VALID', kernel_init=jax.nn.initializers.xavier_uniform())(x)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches, w=num_patches)
        return x
    

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.
    
    @nn.compact
    def __call__(self, x: Array, t: Array):
        c = nn.silu(t)
        c = nn.Dense(
            features = 6 * self.hidden_size,
            kernel_init = jax.nn.initializers.constant(0.)
        )(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)
        x_norm = nn.LayerNorm(
            use_bias = False,
            use_scale = False,
        )(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(
            kernel_init = jax.nn.initializers.xavier_uniform(),
            num_heads = self.num_heads,
        )(x_mod, x_mod)
        x = x + (gate_msa[:,None] * attn_x)
        x_norm2 = nn.LayerNorm(
            use_bias=False,
            use_scale=False
        )(x)
        x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = FeedForward(
            mlp_dim = int(self.hidden_size * self.mlp_ratio)
            )(x_mod2)
        x = x + (gate_mlp[:,None] * mlp_x)
        return x
    
    
class FinalLayer(nn.Module):
    patch_size: int
    out_channels: int
    hidden_size: int
    
    @nn.compact
    def __call__(self, x: Array, t: Array):
        t = nn.silu(t)
        t = nn.Dense(
            features = self.hidden_size * 2,
            kernel_init = jax.nn.initializers.constant(0)
        )(t)
        shift, scale = jnp.split(t, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias = False, use_scale = False)(x), shift, scale)
        x = nn.Dense(
            features = self.patch_size * self.patch_size * self.out_channels,
            kernel_init = jax.nn.initializers.constant(0)
            )(x)
        return x
    
    
class DiT(nn.Module):
    in_channels: int
    out_channels: int
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    
    @nn.compact
    def __call__(self, x: Array, t: Array):
        batch_size = x.shape[0]
        input_size = x.shape[1]
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_per_side = input_size // self.patch_size
        pos_embed = nn.Embed(
            num_embeddings = num_patches,
            features = self.hidden_size,
            embedding_init = jax.nn.initializers.xavier_uniform()
        )(jnp.arange(num_patches))
        x = PatchEmbed(self.patch_size, self.hidden_size)(x)
        x = x + pos_embed
        t = TimestepEmbedding(self.hidden_size)(t)
        for _ in range(self.depth):
            x = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(x, t)
        x = FinalLayer(self.patch_size, self.out_channels, self.hidden_size)(x, t)
        x = jnp.reshape(x, (batch_size, num_patches_per_side, num_patches_per_side, self.patch_size, self.patch_size, self.out_channels))
        x = jnp.einsum('bhwpqc-> bhpwqc', x)
        x = rearrange(x, 'b h p w q c -> b (h p) (w q) c', h = int(num_patches_per_side), w = int(num_patches_per_side))
        assert x.shape == (batch_size, input_size, input_size, self.out_channels)
        return x
    
    
if __name__ == '__main__':
    model = DiT(
        in_channels=3,
        out_channels=4,
        patch_size=2,
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.
        )
    rng = jax.random.PRNGKey(1)
    x_rng, t_rng, init_rng = jax.random.split(rng, 3)
    
    x = jax.random.normal(x_rng, (32, 256, 256, 3))
    t = jax.random.normal(t_rng, (32,))
    t = jax.nn.sigmoid(t)
    
    params = model.init(init_rng, x, t)
    
    pred = model.apply(params, x, t)
    print(pred.shape)
    

        
        
    

    




    
        