import sys

import numpy as np
import tokenizer
from model import SmallLanguageModel

import chainermin

if __name__ == "__main__":
    # モデル設定 (学習時と同じハイパーパラメータ)
    num_layers = 24
    num_heads = 16
    context_length = 1024
    embd_size = 1024

    # 生成設定
    max_new_tokens = 256
    temperature = 0.8
    top_k = 20
    repetition_penalty = 1.2

    tokenizer = tokenizer.SentencePieceTokenizer("ja_unigram_32768.model")
    model = SmallLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        embd_size=embd_size,
    )
    model.to_dtype(np.float16)
    model.to_gpu()
    model.load_npz("best_model_params.npz")

    prompt_text = sys.argv[1]
    prompt = model.xp.array(tokenizer.encode(prompt_text), dtype=np.int32)
    prompt_length = len(prompt)

    sys.stdout.write(f"[Prompt]     : {prompt_text}\n")
    sys.stdout.write("[Generated]  : ")
    sys.stdout.flush()

    with chainermin.inference_mode():
        prev_text = ""
        for _ in range(max_new_tokens):
            prompt = model.generate(
                start_tokens=prompt,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            generated_ids = chainermin.backend.to_cpu(prompt[prompt_length:])
            decoded_text = tokenizer.decode(generated_ids)
            if not decoded_text.endswith("\ufffd"):
                sys.stdout.write(decoded_text[len(prev_text) :])
                sys.stdout.flush()
                prev_text = decoded_text

        sys.stdout.write("\n")
        sys.stdout.flush()
