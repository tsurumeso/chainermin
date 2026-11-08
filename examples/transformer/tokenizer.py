import sentencepiece


class SentencePieceTokenizer:
    def __init__(self, model_path=None):
        self.sp = sentencepiece.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    def tokenize(self, text):
        return self.sp.encode_as_pieces(text)

    def fit(self, **kwargs):
        sentencepiece.SentencePieceTrainer.train(
            **kwargs,
        )

    def save(self):
        raise NotImplementedError()

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def encode(self, text):
        ids = self.sp.encode_as_ids(text)
        return ids

    def decode(self, ids):
        return self.sp.decode_ids(ids)


if __name__ == "__main__":
    model_type = "unigram"
    vocab_size = 32768

    tokenizer = SentencePieceTokenizer()
    tokenizer.fit(
        input="corpus_pretokenized.txt",
        model_prefix=f"ja_{model_type}_{vocab_size}",
        model_type=model_type,
        vocab_size=32768,
        num_threads=8,
        normalization_rule_name="nmt_nfkc",
        user_defined_symbols=["\n"],
        pretokenization_delimiter="\x9e",
        train_extremely_large_corpus=True,
        split_digits=True,
        byte_fallback=True,
    )
