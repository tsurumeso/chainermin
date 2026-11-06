import glob
import math
import os
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

        # RoPE の適用 (角度計算は精度維持のため float32 で行い dtype にキャスト)
        half_d = d // 2
        inv_freq = self.xp.exp(
            -math.log(10000.0) * self.xp.arange(0, half_d, dtype=np.float32) / half_d
        )
        t = self.xp.arange(T, dtype=np.float32)
        freqs = self.xp.outer(t, inv_freq)[None, None, :, :]
        cos = self.xp.cos(freqs).astype(dtype, copy=False)
        sin = self.xp.sin(freqs).astype(dtype, copy=False)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)

        # (B, H, T, d) @ (B, H, d, T) -> (B, H, T, T)
        attn = F.batch_matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d, dtype=dtype)
        causal_mask = self.xp.tril(self.xp.ones((T, T), dtype=np.int32))
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


def load_dataset(pattern, tokenizer, cache_path, chunk_size=10_000_000):
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        return np.load(cache_path)

    files = sorted(glob.glob(pattern, recursive=True))
    chunks = []
    buf = []
    buf_len = 0
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            while True:
                lines = f.readlines(chunk_size)
                if not lines:
                    break
                text = "".join(lines)
                buf.append(text)
                buf_len += len(text)
                if buf_len >= chunk_size:
                    chunks.append(np.array(tokenizer.encode("\n".join(buf)), dtype=np.int32))
                    buf.clear()
                    buf_len = 0
    if buf:
        chunks.append(np.array(tokenizer.encode("\n".join(buf)), dtype=np.int32))
        buf.clear()

    dataset = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
    np.save(cache_path, dataset)
    print(f"Saved cached dataset to {cache_path} ({len(dataset):,} tokens)")
    return dataset


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
    num_layers = 24
    num_heads = 16
    context_length = 1024
    embd_size = 1024

    batch_size = 2
    accumulation_steps = 64
    start_iter = 0
    max_iters = start_iter + 10000
    eval_iters = 32
    eval_interval = 100
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_iters = start_iter + 100
    lr_decay_iters = start_iter + 10000
    loss_scale = 1024.0

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    # tokenizer = tokenizer.MeCabTokenizer("vocab.json")
    tokenizer = tokenizer.SentencePieceTokenizer("sp_model_32768.model")
    trn_dataset = load_dataset(
        os.path.join(dataset_dir, "training", "**", "*.txt"),
        tokenizer,
        os.path.join(dataset_dir, "trn_dataset.npy"),
    )
    val_dataset = load_dataset(
        os.path.join(dataset_dir, "validation", "**", "*.txt"),
        tokenizer,
        os.path.join(dataset_dir, "val_dataset.npy"),
    )

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    model = SmallLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        embd_size=embd_size,
    )

    # Move model parameters to GPU and convert to FP16
    model.to_dtype(np.float16)
    model.to_gpu()
    # model.load_npz("best_model_params.npz")

    optimizer = optimizers.AdamW(alpha=max_lr, loss_scale=loss_scale)
    optimizer.setup(model)
    for name, param in model.namedparams():
        if param.ndim < 2:
            optimizer.states[name]["weight_decay"] = 0.0

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

            y = model(x)
            y_flat = y.reshape(-1, tokenizer.vocab_size)
            t_flat = t.reshape(-1)

            loss = F.softmax_cross_entropy(y_flat, t_flat)
            del y, y_flat

            loss.backward(loss_scale=loss_scale / accumulation_steps)
            sum_loss += loss.data

        optimizer.update()
        avg_loss = sum_loss / accumulation_steps
        print(f"Iteration {it + 1}, Loss: {avg_loss:.4f}, Learning Rate: {lr:.6f}")

        if it == 0 or (it + 1) % eval_interval != 0:
            continue

        # Inference: dropout disabled, no graph
        with chainermin.inference_mode():
            eval_loss = 0
            for j in range(eval_iters):
                x, t = get_batch(val_dataset, batch_size, context_length)
                x = model.xp.array(x)
                t = model.xp.array(t)

                y = model(x)
                y_flat = y.reshape(-1, tokenizer.vocab_size)
                t_flat = t.reshape(-1)

                loss = F.softmax_cross_entropy(y_flat, t_flat)
                del y, y_flat

                eval_loss += loss.data

            eval_loss /= eval_iters
            print(f"Evaluation Loss: {eval_loss:.4f}")

            if eval_loss < best_loss:
                best_loss = eval_loss
                model.save_npz("best_model_params.npz")
                print("New best model saved")

            # プロンプトとして評価データの先頭Nトークンを使用して逐次生成を試す
            prompt_length = 256
            prompt = x[0, :prompt_length]

            prompt_text = tokenizer.decode(chainermin.backend.to_cpu(prompt))
            sys.stdout.write(f"[Prompt]     : {prompt_text}\n")
            sys.stdout.write("[Generated]  : ")
            sys.stdout.flush()

            prev_text = ""
            for _ in range(context_length - prompt_length):
                prompt = model.generate(
                    start_tokens=prompt, temperature=0.8, top_k=20, repetition_penalty=1.2
                )

                generated_ids = chainermin.backend.to_cpu(prompt[prompt_length:])
                decoded_text = tokenizer.decode(generated_ids)
                if not decoded_text.endswith("\ufffd"):
                    sys.stdout.write(decoded_text[len(prev_text) :])
                    sys.stdout.flush()
                    prev_text = decoded_text

            sys.stdout.write("\n")
            sys.stdout.flush()
