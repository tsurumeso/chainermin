import numpy as np

import chainermin
import chainermin.functions as F
import chainermin.links as L
from chainermin import optimizers


class Head(chainermin.Chain):

    def __init__(self, embd_size, head_size):
        super(Head, self).__init__()
        self.key = L.Linear(embd_size, head_size)
        self.query = L.Linear(embd_size, head_size)
        self.value = L.Linear(embd_size, head_size)

    def __call__(self, x, train=True):
        k = self.key(x, n_batch_axes=2)
        q = self.query(x, n_batch_axes=2)
        attn = F.softmax(F.batch_matmul(q, k.transpose(0, 2, 1)) / np.sqrt(k.shape[-1]), axis=-1)
        attn = F.dropout(attn, ratio=0.2, train=train)
        v = self.value(x, n_batch_axes=2)
        out = F.batch_matmul(attn, v)
        return out


class MultiHeadAttention(chainermin.Chain):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, embd_size, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = [Head(embd_size, head_size) for _ in range(num_heads)]
        self.proj = L.Linear(head_size * num_heads, embd_size)

    def __call__(self, x, train=True):
        out = F.concat([h(x) for h in self.heads], axis=-1)
        out = self.proj(out, n_batch_axes=2)
        out = F.dropout(out, ratio=0.2, train=train)
        return out


class FeedForward(chainermin.Chain):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embd_size):
        super(FeedForward, self).__init__()
        self.fc1 = L.Linear(embd_size, 4 * embd_size)
        self.fc2 = L.Linear(4 * embd_size, embd_size)

    def __call__(self, x, train=True):
        x = F.relu(self.fc1(x, n_batch_axes=2))
        x = self.fc2(x, n_batch_axes=2)
        x = F.dropout(x, ratio=0.2, train=train)
        return x


class Block(chainermin.Chain):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_heads, embd_size, head_size):
        super(Block, self).__init__()
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

    def __init__(self, vocab_size, num_layers, num_heads, embd_size, head_size):
        super().__init__()
        self.blocks = [Block(num_heads=num_heads, embd_size=embd_size, head_size=head_size) for _ in range(num_layers)]
        self.ln = L.LayerNormalization(embd_size)
        self.lm_head = L.Linear(embd_size, vocab_size)

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.ln(x, n_batch_axes=2)
        logits = self.lm_head(x, n_batch_axes=2)

        return logits


if __name__ == '__main__':
    # Example usage
    num_layers = 12
    num_heads = 6
    context_length = 256
    vocab_size = 1024
    embd_size = 384
    head_size = embd_size // num_heads
    model = SmallLanguageModel(vocab_size=vocab_size, num_layers=num_layers, num_heads=num_heads, embd_size=embd_size, head_size=head_size)

    # Create a random input tensor
    x = np.random.rand(2, 10, embd_size).astype(np.float32)
    out = model(x)
    print("Output shape:", out.shape)
    print(out.data)

    # Test backward
    t = np.random.rand(*out.shape).astype(np.float32)
    loss = F.mean_squared_error(out, t)
    loss.backward()
