import numpy as np
import time

import chainermin
import chainermin.functions as F
import chainermin.links as L


class Head(chainermin.Chain):

    def __init__(self, embd_size, head_size):
        super(Head, self).__init__()
        self.key = L.Linear(embd_size, head_size)
        self.query = L.Linear(embd_size, head_size)
        self.value = L.Linear(embd_size, head_size)

    def __call__(self, x):
        k = self.key(x, n_batch_axes=2)
        q = self.query(x, n_batch_axes=2)
        attn = F.softmax(F.batch_matmul(q, k.transpose(0, 2, 1)) / np.sqrt(k.shape[-1]), axis=-1)
        attn = F.dropout(attn, ratio=0.2)
        v = self.value(x, n_batch_axes=2)
        out = F.batch_matmul(attn, v)
        return out


class MultiHeadAttention(chainermin.Chain):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, embd_size, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = [Head(embd_size, head_size) for _ in range(num_heads)]
        self.proj = L.Linear(head_size * num_heads, embd_size)

    def __call__(self, x):
        out = F.concat([h(x) for h in self.heads], axis=-1)
        out = self.proj(out, n_batch_axes=2)
        out = F.dropout(out, ratio=0.2)
        return out


class FeedForward(chainermin.Chain):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embd_size):
        super(FeedForward, self).__init__()
        self.fc1 = L.Linear(embd_size, 4 * embd_size)
        self.fc2 = L.Linear(4 * embd_size, embd_size)

    def __call__(self, x):
        x = F.relu(self.fc1(x, n_batch_axes=2))
        x = self.fc2(x, n_batch_axes=2)
        x = F.dropout(x, ratio=0.2)
        return x


class Block(chainermin.Chain):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_heads, embd_size):
        super(Block, self).__init__()
        head_size = embd_size // num_heads
        assert head_size * num_heads == embd_size, "embd_size must be divisible by num_heads"
        self.sa = MultiHeadAttention(num_heads, embd_size, head_size)
        self.ffwd = FeedForward(embd_size)
        self.ln1 = L.LayerNormalization(embd_size)
        self.ln2 = L.LayerNormalization(embd_size)

    def __call__(self, x):
        x = self.ln1(x, n_batch_axes=2)
        x = x + self.sa(x)
        x = self.ln2(x, n_batch_axes=2)
        x = x + self.ffwd(x)
        return x


class SmallLanguageModel(chainermin.Chain):

    def __init__(self, vocab_size, context_length, num_layers, num_heads, embd_size):
        super().__init__()
        self.tok_embd = L.EmbedID(vocab_size, embd_size)
        self.pos_embd = L.EmbedID(context_length, embd_size)
        self.blocks = [Block(num_heads=num_heads, embd_size=embd_size) for _ in range(num_layers)]
        self.ln = L.LayerNormalization(embd_size)
        self.lm_head = L.Linear(embd_size, vocab_size)

    def __call__(self, x):
        # Add token and position embeddings
        tok_embd = self.tok_embd(x)
        pos_ids = np.arange(x.shape[1], dtype=np.int32)
        pos_embd = self.pos_embd(pos_ids)
        x = tok_embd + pos_embd

        for block in self.blocks:
            x = block(x)
        x = self.ln(x, n_batch_axes=2)
        logits = self.lm_head(x, n_batch_axes=2)

        return logits


if __name__ == '__main__':
    # Hyperparameters for the small language model
    num_layers = 12
    num_heads = 12
    context_length = 1024
    vocab_size = 1024
    embd_size = 768

    model = SmallLanguageModel(vocab_size=vocab_size, context_length=context_length, num_layers=num_layers, num_heads=num_heads, embd_size=embd_size)

    # Create a random input tensor
    x = np.random.randint(0, vocab_size, size=(1, context_length), dtype=np.int32)
    t = np.random.rand(1, context_length, vocab_size).astype(np.float32)

    # Training mode (default): dropout is active, graph is built
    start = time.perf_counter()
    out = model(x)
    end = time.perf_counter()
    print("Training forward: {:.4f}s".format(end - start))
    print("Output shape:", out.shape)

    start = time.perf_counter()
    loss = F.mean_squared_error(out, t)
    loss.backward()
    end = time.perf_counter()
    print("Training backward: {:.4f}s".format(end - start))

    # Inference: dropout is disabled, no graph
    start = time.perf_counter()
    with chainermin.inference_mode():
        out = model(x)
    end = time.perf_counter()
    print("Inference forward: {:.4f}s".format(end - start))
    print("Output shape:", out.shape)
