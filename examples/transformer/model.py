import math

import numpy as np

import chainermin
import chainermin.functions as F
import chainermin.links as L


class RoPE(chainermin.function.FunctionNode):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, cos, sin):
        self.cos = cos
        self.sin = sin

    def forward(self, inputs):
        (x,) = inputs
        xp = chainermin.backend.get_array_module(x)
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * self.cos - x2 * self.sin
        y2 = x1 * self.sin + x2 * self.cos
        return (xp.concatenate([y1, y2], axis=-1),)

    def backward(self, inputs, grad_outputs):
        (gy,) = grad_outputs
        xp = chainermin.backend.get_array_module(gy)
        d = gy.shape[-1] // 2
        gy1, gy2 = gy[..., :d], gy[..., d:]
        gx1 = gy1 * self.cos + gy2 * self.sin
        gx2 = -gy1 * self.sin + gy2 * self.cos
        return (xp.concatenate([gx1, gx2], axis=-1),)


def rope(x, cos, sin):
    return RoPE(cos, sin)(x)


class MultiHeadAttention(chainermin.Chain):
    """Multi-head causal self-attention (single matmul formulation)."""

    def __init__(self, num_heads, embd_size, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # 各ヘッド分の重みを1つの Linear に統合
        self.key = L.Linear(embd_size, embd_size, nobias=True)
        self.query = L.Linear(embd_size, embd_size, nobias=True)
        self.value = L.Linear(embd_size, embd_size, nobias=True)
        self.proj = L.Linear(embd_size, embd_size, nobias=True)
        self._cos = None
        self._sin = None
        self._causal_mask = None

    def _get_cache(self, T, dtype):
        if (
            self._cos is None
            or self._cos.shape[2] < T
            or self._cos.dtype != dtype
            or chainermin.backend.get_array_module(self._cos) is not self.xp
        ):
            max_len = max(T, 1024)
            half_d = self.head_size // 2
            inv_freq = self.xp.exp(
                -math.log(10000.0) * self.xp.arange(0, half_d, dtype=np.float32) / half_d
            )
            t = self.xp.arange(max_len, dtype=np.float32)
            freqs = self.xp.outer(t, inv_freq)[None, None, :, :]
            self._cos = self.xp.cos(freqs).astype(dtype, copy=False)
            self._sin = self.xp.sin(freqs).astype(dtype, copy=False)
            self._causal_mask = self.xp.triu(
                self.xp.ones((1, 1, max_len, max_len), dtype=bool), k=1
            )
        return (
            self._cos[:, :, :T, :],
            self._sin[:, :, :T, :],
            self._causal_mask[:, :, :T, :T],
        )

    def __call__(self, x):
        dtype = x.data.dtype
        B, T, _ = x.shape
        H = self.num_heads
        d = self.head_size

        # (B, T, H*d) = (B, T, embd_size)
        q = self.query(x, n_batch_axes=2)
        k = self.key(x, n_batch_axes=2)
        v = self.value(x, n_batch_axes=2)

        # (B, T, H, d) -> (B, H, T, d)
        q = q.reshape(B, T, H, d).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, d).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, d).transpose(0, 2, 1, 3)

        # RoPE の適用
        cos, sin, causal_mask = self._get_cache(T, dtype)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)

        # (B, H, T, d) @ (B, H, d, T) -> (B, H, T, T)
        attn = F.batch_matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d, dtype=dtype)
        attn = F.where(causal_mask, -np.inf, attn)
        attn = F.softmax(attn, axis=-1)

        # (B, H, T, T) @ (B, H, T, d) -> (B, H, T, d)
        out = F.batch_matmul(attn, v)

        # (B, H, T, d) -> (B, T, H*d) = (B, T, embd_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        out = self.proj(out, n_batch_axes=2)
        out = F.dropout(out, ratio=0.1)
        return out


class FeedForward(chainermin.Chain):
    """SwiGLU feed-forward network."""

    def __init__(self, embd_size):
        super().__init__()
        hidden_size = int(4 * embd_size * 2 / 3)
        self.w_gate = L.Linear(embd_size, hidden_size, nobias=True)
        self.w_up = L.Linear(embd_size, hidden_size, nobias=True)
        self.w_down = L.Linear(hidden_size, embd_size, nobias=True)

    def __call__(self, x):
        gate = F.silu(self.w_gate(x, n_batch_axes=2))
        up = self.w_up(x, n_batch_axes=2)
        x = self.w_down(gate * up, n_batch_axes=2)
        x = F.dropout(x, ratio=0.1)
        return x


class DecoderBlock(chainermin.Chain):
    """Decoder block: causal self-attention followed by feed-forward."""

    def __init__(self, num_heads, embd_size):
        super().__init__()
        head_size = embd_size // num_heads
        assert head_size * num_heads == embd_size, "embd_size must be divisible by num_heads"
        self.sa = MultiHeadAttention(num_heads, embd_size, head_size)
        self.ffwd = FeedForward(embd_size)
        self.ln1 = L.LayerNormalization(embd_size)
        self.ln2 = L.LayerNormalization(embd_size)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x, n_batch_axes=2))
        x = x + self.ffwd(self.ln2(x, n_batch_axes=2))
        return x


class SmallLanguageModel(chainermin.Chain):
    """GPT-style small language model."""

    def __init__(self, vocab_size, context_length, num_layers, num_heads, embd_size):
        super().__init__()
        self.tok_embd = L.EmbedID(vocab_size, embd_size, wscale=0.02)
        for idx in range(num_layers):
            layer_name = f"block_{idx}"
            layer = DecoderBlock(num_heads, embd_size)
            setattr(self, layer_name, layer)
        self.ln = L.LayerNormalization(embd_size)

        self.num_layers = num_layers
        self.context_length = context_length

    def __call__(self, x):
        # Token embeddings (positional information is injected via RoPE in each block)
        x = self.tok_embd(x)

        # Dropout after embeddings
        x = F.dropout(x, ratio=0.1)

        # Stacked decoder blocks
        for idx in range(self.num_layers):
            block = getattr(self, f"block_{idx}")
            x = block(x)

        x = self.ln(x, n_batch_axes=2)
        return F.linear(x, self.tok_embd.W, n_batch_axes=2)

    def generate(self, start_tokens, temperature=1.0, top_k=10, repetition_penalty=1.0):
        # (1, T) の形状に変換
        curr_x = self.xp.array(start_tokens, dtype=np.int32)[None, :]

        # context_length を超えないように末尾をクロップ
        x_crop = curr_x[:, -self.context_length :]
        logits = self(x_crop)

        # 最後のステップの Logits を取得 (確率計算の精度維持のため float32 にキャスト)
        logits = logits.data
        next_token_logits = logits[:, -1, :].astype(np.float32, copy=False)

        # Repetition Penalty の適用 (既出トークンのLogitを減衰)
        if repetition_penalty != 1.0:
            seen_tokens = self.xp.unique(x_crop)
            seen_logits = next_token_logits[:, seen_tokens]
            next_token_logits[:, seen_tokens] = self.xp.where(
                seen_logits < 0,
                seen_logits * repetition_penalty,
                seen_logits / repetition_penalty,
            )

        # Temperature スケーリング
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Top-K サンプリング
        if top_k is not None:
            indices_to_remove = (
                next_token_logits < self.xp.sort(next_token_logits, axis=-1)[:, -top_k:][:, 0:1]
            )
            next_token_logits[indices_to_remove] = -np.inf

        probs = F.softmax(next_token_logits, axis=-1).data
        next_token_id = self.xp.random.choice(len(probs[0]), size=1, p=probs[0])

        # 生成されたトークンを結合
        next_token = next_token_id.reshape(1, 1).astype(np.int32)
        curr_x = self.xp.concatenate([curr_x, next_token], axis=1)

        # (T_new,) の 1D 配列を返す
        return curr_x[0]
