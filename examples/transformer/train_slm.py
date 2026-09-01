import glob
import math
import sys

import numpy as np
import tokenizer

import chainermin
import chainermin.functions as F
import chainermin.links as L
from chainermin import optimizers

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
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # 各ヘッド分の重みを1つの Linear に統合
        self.key = L.Linear(embd_size, embd_size, nobias=True)
        self.query = L.Linear(embd_size, embd_size, nobias=True)
        self.value = L.Linear(embd_size, embd_size, nobias=True)
        self.proj = L.Linear(embd_size, embd_size, nobias=True)

    def __call__(self, x, causal_mask):
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

        # (B, H, T, d) @ (B, H, d, T) -> (B, H, T, T)
        attn = F.batch_matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d, dtype=dtype)
        attn = F.where(causal_mask[None, None, :, :] == 0, -np.inf, attn)
        attn = F.softmax(attn, axis=-1)

        # (B, H, T, T) @ (B, H, T, d) -> (B, H, T, d)
        out = F.batch_matmul(attn, v)

        # (B, H, T, d) -> (B, T, H*d) = (B, T, embd_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        out = self.proj(out, n_batch_axes=2)
        out = F.dropout(out, ratio=0.1)
        return out


class FeedForward(chainermin.Chain):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, embd_size):
        super().__init__()
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
        super().__init__()
        head_size = embd_size // num_heads
        assert head_size * num_heads == embd_size, "embd_size must be divisible by num_heads"
        self.sa = MultiHeadAttention(num_heads, embd_size, head_size)
        self.ffwd = FeedForward(embd_size)
        self.ln1 = L.LayerNormalization(embd_size)
        self.ln2 = L.LayerNormalization(embd_size)

    def __call__(self, x, causal_mask):
        x = x + self.sa(self.ln1(x, n_batch_axes=2), causal_mask)
        x = x + self.ffwd(self.ln2(x, n_batch_axes=2))
        return x


class SmallLanguageModel(chainermin.Chain):
    """GPT-style small language model."""

    def __init__(self, vocab_size, context_length, num_layers, num_heads, embd_size):
        super().__init__()
        self.tok_embd = L.EmbedID(vocab_size, embd_size)
        self.pos_embd = L.EmbedID(context_length, embd_size)
        for idx in range(num_layers):
            layer_name = f"block_{idx}"
            layer = DecoderBlock(num_heads, embd_size)
            setattr(self, layer_name, layer)
        self.ln = L.LayerNormalization(embd_size)
        self.lm_head = L.Linear(embd_size, vocab_size)

        self.num_layers = num_layers
        self.context_length = context_length

    def __call__(self, x, causal_mask):
        # Token + position embeddings
        pos_ids = self.xp.arange(x.shape[1], dtype=np.int32)
        batched_pos_ids = self.xp.broadcast_to(pos_ids, x.shape)
        x = self.tok_embd(x) + self.pos_embd(batched_pos_ids)

        # Dropout after embeddings
        x = F.dropout(x, ratio=0.1)

        # Stacked decoder blocks
        for idx in range(self.num_layers):
            block = getattr(self, f"block_{idx}")
            x = block(x, causal_mask)

        x = self.ln(x, n_batch_axes=2)
        return self.lm_head(x, n_batch_axes=2)

    def generate(self, start_tokens, temperature=1.0, top_k=10):
        # (1, T) の形状に変換
        curr_x = self.xp.array(start_tokens, dtype=np.int32)[None, :]

        # context_length を超えないように末尾をクロップ
        x_crop = curr_x[:, -self.context_length :]

        # 現在の系列長に応じた Causal Mask を作成
        T = x_crop.shape[1]
        causal_mask = self.xp.tril(self.xp.ones((T, T), dtype=np.int32))
        logits = self(x_crop, causal_mask)

        # 最後のステップの Logits を取得
        logits = logits.data
        next_token_logits = logits[:, -1, :]

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


def get_batch(dataset, batch_size, context_length):
    idx = np.random.randint(0, len(dataset) - context_length - 1, size=batch_size)
    x = np.stack([dataset[i : i + context_length] for i in idx])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in idx])
    return x, y


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, max_lr, min_lr, warmup_iters, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    # coeff ranges 0..1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


if __name__ == "__main__":
    # Hyperparameters
    num_layers = 12
    num_heads = 12
    context_length = 512
    embd_size = 768

    batch_size = 8
    accumulation_steps = 8
    start_iter = 0
    max_iters = 10000
    eval_iters = 10
    eval_interval = 10
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_iters = 100
    lr_decay_iters = 10000

    trn_dataset = []
    val_dataset = []
    files = glob.glob("dataset/training/**/*.txt", recursive=True)
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            trn_dataset.append(f.read())

    files = glob.glob("dataset/validation/**/*.txt", recursive=True)
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            val_dataset.append(f.read())

    tokenizer = tokenizer.MeCabTokenizer("vocab.json")
    trn_dataset = tokenizer.encode("\n".join(trn_dataset))
    val_dataset = tokenizer.encode("\n".join(val_dataset))

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    model = SmallLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        embd_size=embd_size,
    )

    # Move model parameters to GPU
    model.to_gpu()
    # model.load_npz("best_model_params.npz")

    optimizer = optimizers.AdamW(alpha=max_lr)
    optimizer.setup(model)

    # Causal Mask: lower-triangular
    causal_mask = model.xp.tril(model.xp.ones((context_length, context_length), dtype=np.int32))

    best_loss = np.inf
    for it in range(start_iter, max_iters):
        sum_loss = 0
        lr = get_lr(it, max_lr, min_lr, warmup_iters, lr_decay_iters)
        optimizer.alpha = lr
        model.zerograds()
        for j in range(accumulation_steps):
            x, t = get_batch(trn_dataset, batch_size, context_length)
            x = model.xp.array(x)
            t = model.xp.array(t)

            y = model(x, causal_mask)
            y_flat = y.reshape(-1, tokenizer.vocab_size)
            t_flat = t.reshape(-1)

            loss = F.softmax_cross_entropy(y_flat, t_flat)
            loss /= accumulation_steps
            loss.backward()

            sum_loss += loss.data

        optimizer.update()
        print(f"Iteration {it + 1}, Loss: {sum_loss:.4f}, Learning Rate: {lr:.6f}")

        if it == 0 or (it + 1) % eval_interval != 0:
            continue

        # Inference: dropout disabled, no graph
        with chainermin.inference_mode():
            eval_loss = 0
            for j in range(eval_iters):
                x, t = get_batch(val_dataset, batch_size, context_length)
                x = model.xp.array(x)
                t = model.xp.array(t)

                y = model(x, causal_mask)
                y_flat = y.reshape(-1, tokenizer.vocab_size)
                t_flat = t.reshape(-1)

                loss = F.softmax_cross_entropy(y_flat, t_flat)

                eval_loss += loss.data

            eval_loss /= eval_iters
            if eval_loss < best_loss:
                best_loss = eval_loss
                model.save_npz("best_model_params.npz")
                print(f"New best model saved with loss: {best_loss:.4f}")

            # プロンプトとして評価データの先頭Nトークンを使用して逐次生成を試す
            prompt_length = 128
            prompt = x[0, :prompt_length]

            prompt_text = tokenizer.decode(chainermin.backend.to_cpu(prompt))
            sys.stdout.write(f"[Prompt]     : {prompt_text}\n")
            sys.stdout.write("[Generated]  : ")
            sys.stdout.flush()

            for _ in range(context_length - prompt_length):
                prompt = model.generate(start_tokens=prompt, temperature=1.0, top_k=10)

                generated_text = tokenizer.decode([prompt[-1].item()])
                sys.stdout.write(generated_text)
                sys.stdout.flush()

            sys.stdout.write("\n")
            sys.stdout.flush()
