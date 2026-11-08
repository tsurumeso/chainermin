import glob
import math
import os
import sys

import numpy as np
import tokenizer
from model import SmallLanguageModel

import chainermin
import chainermin.functions as F
from chainermin import optimizers


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
    tokenizer = tokenizer.SentencePieceTokenizer("ja_unigram_32768.model")
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
