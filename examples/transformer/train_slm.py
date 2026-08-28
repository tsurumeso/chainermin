import numpy as np
import time

import chainermin
import chainermin.functions as F
import chainermin.links as L


# class AttentionHead(chainermin.Chain):
#     """Single causal self-attention head."""

#     def __init__(self, embd_size, head_size):
#         super(AttentionHead, self).__init__()
#         self.key = L.Linear(embd_size, head_size)
#         self.query = L.Linear(embd_size, head_size)
#         self.value = L.Linear(embd_size, head_size)

#     def __call__(self, x, causal_mask):
#         k = self.key(x, n_batch_axes=2)
#         q = self.query(x, n_batch_axes=2)
#         attn = F.batch_matmul(q, k.transpose(0, 2, 1)) / np.sqrt(k.shape[-1])
#         attn = attn * causal_mask - (1.0 - causal_mask) * 1e+9
#         attn = F.softmax(attn, axis=-1)
#         v = self.value(x, n_batch_axes=2)
#         return F.batch_matmul(attn, v)


# class MultiHeadAttention(chainermin.Chain):
#     """Multi-head causal self-attention."""

#     def __init__(self, num_heads, embd_size, head_size):
#         super(MultiHeadAttention, self).__init__()
#         self.heads = [AttentionHead(embd_size, head_size) for _ in range(num_heads)]
#         self.proj = L.Linear(head_size * num_heads, embd_size)

#     def __call__(self, x, causal_mask):
#         out = F.concat([h(x, causal_mask) for h in self.heads], axis=-1)
#         out = self.proj(out, n_batch_axes=2)
#         out = F.dropout(out, ratio=0.1)
#         return out


class MultiHeadAttention(chainermin.Chain):
    """Multi-head causal self-attention (single matmul formulation)."""

    def __init__(self, num_heads, embd_size, head_size):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # 各ヘッド分の重みを1つの Linear に統合
        self.key = L.Linear(embd_size, embd_size, nobias=True)
        self.query = L.Linear(embd_size, embd_size, nobias=True)
        self.value = L.Linear(embd_size, embd_size, nobias=True)
        self.proj = L.Linear(embd_size, embd_size, nobias=True)

    def __call__(self, x, causal_mask):
        B, T, _ = x.shape
        H = self.num_heads
        d = self.head_size

        # (B, T, H*d)
        q = self.query(x, n_batch_axes=2)  
        k = self.key(x, n_batch_axes=2)
        v = self.value(x, n_batch_axes=2)

        # (B, T, H, d) → (B, H, T, d)
        q = q.reshape(B, T, H, d).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, d).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, d).transpose(0, 2, 1, 3)

        # (B, H, T, d) @ (B, H, d, T) → (B, H, T, T)
        attn = F.batch_matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d)
        attn = attn * causal_mask - (1.0 - causal_mask) * 1e+9
        attn = F.softmax(attn, axis=-1)

        # (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
        out = F.batch_matmul(attn, v)

        # (B, H, T, d) → (B, T, H*d) = (B, T, embd_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        out = self.proj(out, n_batch_axes=2)
        out = F.dropout(out, ratio=0.1)
        return out


class FeedForward(chainermin.Chain):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, embd_size):
        super(FeedForward, self).__init__()
        self.fc1 = L.Linear(embd_size, 4 * embd_size, nobias=True)
        self.fc2 = L.Linear(4 * embd_size, embd_size, nobias=True)

    def __call__(self, x):
        x = F.relu(self.fc1(x, n_batch_axes=2))
        x = self.fc2(x, n_batch_axes=2)
        x = F.dropout(x, ratio=0.1)
        return x


class DecoderBlock(chainermin.Chain):
    """Decoder block: causal self-attention followed by feed-forward."""

    def __init__(self, num_heads, embd_size):
        super(DecoderBlock, self).__init__()
        head_size = embd_size // num_heads
        assert head_size * num_heads == embd_size, "embd_size must be divisible by num_heads"
        self.sa = MultiHeadAttention(num_heads, embd_size, head_size)
        self.ffwd = FeedForward(embd_size)
        self.ln1 = L.LayerNormalization(embd_size)
        self.ln2 = L.LayerNormalization(embd_size)

    def __call__(self, x, causal_mask):
        x = self.ln1(x, n_batch_axes=2)
        x = x + self.sa(x, causal_mask)
        x = self.ln2(x, n_batch_axes=2)
        x = x + self.ffwd(x)
        return x


class SmallLanguageModel(chainermin.Chain):
    """Decoder-only autoregressive Transformer (GPT-style)."""

    def __init__(self, vocab_size, context_length, num_layers, num_heads, embd_size):
        super(SmallLanguageModel, self).__init__()
        self.tok_embd = L.EmbedID(vocab_size, embd_size)
        self.pos_embd = L.EmbedID(context_length, embd_size)
        for idx in range(num_layers):
            layer_name = f"block_{idx}"
            layer = DecoderBlock(num_heads, embd_size)
            setattr(self, layer_name, layer)
        self.ln = L.LayerNormalization(embd_size)
        self.lm_head = L.Linear(embd_size, vocab_size)

        self.num_layers = num_layers

    def __call__(self, x, causal_mask):
        # Token + position embeddings
        pos_ids = self.xp.arange(x.shape[1], dtype=np.int32)
        x = self.tok_embd(x) + self.pos_embd(pos_ids)

        # Dropout after embeddings
        x = F.dropout(x, ratio=0.1)

        # Stacked decoder blocks
        for idx in range(self.num_layers):
            block = getattr(self, f"block_{idx}")
            x = block(x, causal_mask)

        x = self.ln(x, n_batch_axes=2)
        return self.lm_head(x, n_batch_axes=2)


if __name__ == '__main__':
    # Hyperparameters
    num_layers = 12
    num_heads = 12
    context_length = 1024
    vocab_size = 1024
    embd_size = 768

    model = SmallLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        embd_size=embd_size
    )

    # Move model parameters to GPU
    # model.to_gpu()

    # Input: single token sequence (autoregressive)
    x = model.xp.random.randint(0, vocab_size, size=(1, context_length), dtype=np.int32)
    # Target: dummy target for loss computation
    t = model.xp.random.rand(1, context_length, vocab_size).astype(np.float32)

    # Causal mask: lower-triangular
    causal_mask = model.xp.tril(model.xp.ones((context_length, context_length), dtype=np.int32))

    for _ in range(10):
        # Training mode: dropout active, graph built
        start = time.perf_counter()
        out = model(x, causal_mask)
        end = time.perf_counter()
        print("Training forward: {:.4f}s".format(end - start))

        start = time.perf_counter()
        loss = F.mean_squared_error(out, t)
        loss.backward()
        model.zerograds()
        end = time.perf_counter()
        print("Training backward: {:.4f}s".format(end - start))

        # Inference: dropout disabled, no graph
        start = time.perf_counter()
        with chainermin.inference_mode():
            out = model(x, causal_mask)
        end = time.perf_counter()
        print("Inference forward: {:.4f}s".format(end - start))
