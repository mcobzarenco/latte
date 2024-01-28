from __future__ import annotations

from typing import Any, Literal, assert_never

import jax
from flax import linen as nn
from flax.linen.attention import SelfAttention
from jax import Array
from jax import numpy as jnp
from optax import softmax_cross_entropy_with_integer_labels


class Transformer(nn.Module):
    attention_type: Literal["standard"] | Literal["latte"] | Literal["latte-wrap"]

    num_layers: int
    num_heads: int
    num_embeddings: int
    embedding_size: int
    context_size: int
    dropout: float = 0.0

    @nn.compact
    def __call__(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        indices: Array,
        deterministic: bool,
        targets: Array | None = None,
    ) -> tuple[Array, Array | None]:
        """
        :param indices: (B, T)
        """
        token_embeddings = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embedding_size,
        )
        positional_embeddings = nn.Embed(
            num_embeddings=self.context_size,
            features=self.embedding_size,
        )
        x = token_embeddings(indices)  # (B, T, D)
        x = x + jnp.expand_dims(positional_embeddings.embedding[: indices.shape[1]], 0)  # (B, T, D)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                attention_type=self.attention_type,
                num_heads=self.num_heads,
                embedding_size=self.embedding_size,
                context_size=self.context_size,
                dropout=self.dropout,
            )(x, deterministic)

        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)  # (B, T, D)
        logits = x @ token_embeddings.embedding.T  # (B, T, V)

        loss = None
        if targets is not None:
            loss = softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        return (logits, loss)


class TransformerBlock(nn.Module):
    attention_type: Literal["standard"] | Literal["latte"] | Literal["latte-wrap"]
    num_heads: int
    embedding_size: int
    context_size: int = 2000
    dropout: float = 0.0

    def attention_module(self) -> CausalLatentAttention | SelfAttention | CausalSelfAttention:
        match self.attention_type:
            case "latte":
                return CausalLatentAttention(
                    num_heads=self.num_heads,
                    hidden_dim=self.embedding_size,
                    max_seq_len=self.context_size,
                )
            case "latte-wrap":
                return SelfAttention(
                    attention_fn=_latte,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout,
                )
            case "standard":
                return CausalSelfAttention(
                    num_heads=self.num_heads,
                    embedding_size=self.embedding_size,
                    context_size=self.context_size,
                    dropout=self.dropout,
                )
            case _ as unreachable:
                assert_never(unreachable)

    @nn.compact
    def __call__(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        x: Array,
        deterministic: bool,
    ) -> jax.Array:
        attention_ln = nn.LayerNorm(use_bias=False, use_scale=False)
        attention = self.attention_module()

        feedforward_ln = nn.LayerNorm(use_bias=False, use_scale=False)
        feedforward = MLP(
            embedding_size=self.embedding_size,
            hidden_size=4 * self.embedding_size,
            dropout=self.dropout,
            use_bias=True,
        )

        x = x + attention(attention_ln(x), deterministic=deterministic)
        x = x + feedforward(feedforward_ln(x), deterministic=deterministic)
        return x


class MLP(nn.Module):
    embedding_size: int
    hidden_size: int
    dropout: float = 0.0
    use_bias: bool = True

    @nn.compact
    def __call__(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        x: Array,
        deterministic: bool,
    ) -> Array:
        """
        :param X: (B, T, D)
        :returns: (B, T, D)
        """
        hidden = nn.Dense(
            features=self.hidden_size,
            use_bias=self.use_bias,
        )(
            x
        )  # (B, T, 4 * D)

        output = nn.Dense(
            features=self.embedding_size,
            use_bias=self.use_bias,
        )(
            nn.gelu(hidden)
        )  # (B, T, 4 * D)

        return nn.Dropout(self.dropout, deterministic=deterministic)(output)


class CausalLatentAttention(nn.Module):
    """
    Causal multihead latent attention (latte)
    """

    num_heads: int = 4
    hidden_dim: int = 128
    max_seq_len: int = 2000

    dropout: float = 0.0

    def setup(self) -> None:
        assert self.hidden_dim % self.num_heads == 0, (self.hidden_dim, self.num_heads)

        # Key, query, value projections for all heads (Wq, Wk, Wv matrices) packed together
        self.w = self.param(
            "w",
            jax.nn.initializers.lecun_normal(),
            (3 * self.hidden_dim, self.hidden_dim),
        )  # (D, 3 * D)
        self.output_projection = nn.Dense(self.hidden_dim, use_bias=False)

        # Regularization
        self.attention_dropout = nn.Dropout(self.dropout)
        self.residual_dropout = nn.Dropout(self.dropout)

    def __call__(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        x: Array,
        deterministic: bool,
    ) -> Array:
        """
         B: batch size
        Hn: number of heads
         T: sequence length
         D: hidden_dim
         L: latent dimension

        Args:
            x: jnp.array(B, T, D)
        """

        # Sizes
        B, T, D = x.shape
        num_heads, L = self.num_heads, self.hidden_dim // self.num_heads

        # Query, keys, values
        q, k, v = jnp.split(jnp.matmul(x, self.w.T), 3, axis=2)  # 3x (B, T, D)
        q = q.reshape(B, T, num_heads, L).transpose(1, 0, 2, 3)  # (T, B, Hn, L)
        k = k.reshape(B, T, num_heads, L).transpose(1, 0, 2, 3)  # (T, B, Hn, L)
        v = v.reshape(B, T, num_heads, L).transpose(1, 0, 2, 3)  # (T, B, Hn, L)

        scale = jax.lax.rsqrt(jnp.float32(L))
        # scale = 1.0

        k_exp = jnp.exp(k * scale) + 1e-6  # (T, B, Hn, L)
        k_norm = k_exp.cumsum(axis=0)  # (T, B, Hn, L)

        qs = jax.nn.softmax(q * scale, axis=-1) / k_norm  # (T, B, Hn, L)
        qs = self.attention_dropout(qs, deterministic=deterministic)
        _, y = jax.lax.scan(
            self.accumulate,
            init=jnp.zeros((B, num_heads, L, L)),
            xs=(qs, k_exp, v),
            unroll=512,
        )
        y = y.reshape(T, B, D).transpose((1, 0, 2))  # (T, B, D)

        return self.residual_dropout(self.output_projection(y), deterministic=deterministic)

    @staticmethod
    def accumulate(
        carry: jax.Array,
        args: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        qs_t, k_exp_t, v_t = args  # (B, Hn, L), (B, Hn, L), (B, Hn, L)

        k_exp_t = jnp.expand_dims(k_exp_t, axis=-1)  # (B, Hn, L) -> (B, Hn, L, 1)
        v_t = jnp.expand_dims(v_t, axis=-2)  # (B, Hn, L) -> (B, Hn, 1, L)

        # cummulative outer product between k_exp[t] and v[t]
        carry = carry + jax.lax.batch_matmul(k_exp_t, v_t)  # (B, Hn, L, L)

        qs_t = jnp.expand_dims(qs_t, -2)  # (B, Hn, L) -> (B, Hn, 1, L)
        y = jax.lax.batch_matmul(qs_t, carry).squeeze()  # (B, Hn, L)

        return (carry, y)  # (B, Hn, L, L), (B, Hn, L)


class CausalSelfAttention(nn.Module):
    num_heads: int
    embedding_size: int
    context_size: int
    dropout: float

    @nn.compact
    def __call__(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        x: Array,
        deterministic: bool,
    ):
        # key, query, value projections for all heads, but in a batch
        c_attn = nn.Dense(3 * self.embedding_size, use_bias=False)
        # output projection
        c_proj = nn.Dense(self.embedding_size, use_bias=False)

        # regularization
        attn_dropout = nn.Dropout(rate=self.dropout, deterministic=deterministic)
        resid_dropout = nn.Dropout(rate=self.dropout, deterministic=deterministic)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = jnp.tril(jnp.ones(shape=(self.context_size, self.context_size))).reshape(
            1, 1, self.context_size, self.context_size
        )

        B, T, L = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move
        # head forward to be the batch dim
        q, k, v = jnp.split(c_attn(x), 3, axis=2)
        k = k.reshape(B, T, self.num_heads, L // self.num_heads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        q = q.reshape(B, T, self.num_heads, L // self.num_heads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        v = v.reshape(B, T, self.num_heads, L // self.num_heads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ jnp.swapaxes(k, -2, -1)) * jax.lax.rsqrt(jnp.float32(L))

        # att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
        att = jnp.where(bias[:, :, :T, :T] == 0, float("-inf"), att)
        att = jax.nn.softmax(att, axis=-1)

        att = attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(0, 2, 1, 3).reshape(B, T, L)  # re-assemble all head outputs side by side

        # output projection
        y = resid_dropout(c_proj(y))
        return y


def _latte(
    q: Array,
    k: Array,
    v: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Any = None,
    precision: nn.linear.PrecisionLike = None,
    module: nn.Module | None = None,
):
    def accumulate(
        carry: jax.Array,
        args: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        qs_t, k_exp_t, v_t = args  # (B, Hn, L), (B, Hn, L), (B, Hn, L)

        k_exp_t = jnp.expand_dims(k_exp_t, axis=-1)  # (B, Hn, L, 1)
        v_t = jnp.expand_dims(v_t, axis=-2)  # (B, Hn, 1, L)

        # cummulative outer product between k_exp[t] and v[t]
        carry = carry + jax.lax.batch_matmul(k_exp_t, v_t)  # (B, Hn, L, L)

        qs_t = jnp.expand_dims(qs_t, -2)  # (B, Hn, 1, L)
        y = jax.lax.batch_matmul(qs_t, carry).squeeze()  # (B, Hn, L)

        return (carry, y)

    # Shapes
    B, T, num_heads, L = q.shape
    scale = jax.lax.rsqrt(jnp.float32(L))

    # Reshape qkv (B, T, Hn, L) -> (T, B, Hn, L)
    q = q.reshape(B, T, num_heads, L).transpose(1, 0, 2, 3)
    k = k.reshape(B, T, num_heads, L).transpose(1, 0, 2, 3)
    v = v.reshape(B, T, num_heads, L).transpose(1, 0, 2, 3)

    k_exp = jnp.exp(k * scale) + 1e-6  # (T, B, Hn, L)
    k_norm = k_exp.cumsum(axis=0)  # (T, B, Hn, L)

    qs = jax.nn.softmax(q * scale, axis=-1) / k_norm  # (T, B, Hn, L)
    qs = nn.Dropout(dropout_rate)(qs, deterministic=deterministic)

    _, y = jax.lax.scan(
        accumulate,
        init=jnp.zeros((B, num_heads, L, L)),
        xs=(qs, k_exp, v),
        unroll=512,
    )  # (T, B, Hn, L)
    y = y.transpose((1, 0, 2, 3))  # (B, T, Nh, L)

    return y
