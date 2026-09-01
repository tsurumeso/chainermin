import collections
import json

import ipadic
import MeCab
import sentencepiece


class MeCabTokenizer:
    def __init__(self, model_path=None):
        mecab_args = f"-Owakati {ipadic.MECAB_ARGS}"
        # -Owakati オプションで分かち書きモードにする
        self.tagger = MeCab.Tagger(mecab_args)
        self.vocab = {"<unk>": 0, "<pad>": 1}
        self.inv_vocab = {0: "<unk>", 1: "<pad>"}
        if model_path:
            with open(model_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
                self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        # 文章を単語リストに分解 ("吾輩は猫である\n" -> ["吾輩", "は", "猫", "で", "ある"])
        return self.tagger.parse(text).strip().split()

    def fit(self, corpus_text, min_count=5):
        # 1. コーパス全体の単語の出現頻度をカウント
        tokens = self.tokenize(corpus_text)
        token_counts = collections.Counter(tokens)
        token_counts = collections.Counter(
            {k: v for k, v in token_counts.items() if v >= min_count}
        )

        # 2. コーパスに含まれる「すべての単文字」を抽出 (未知語対策)
        unique_chars = sorted(set(corpus_text))
        print(len(corpus_text), "characters in corpus.")
        print(len(unique_chars), "unique characters found in corpus.")
        print(len(token_counts), "unique vocabularies found in corpus.")

        # 3. 「すべての単文字」を優先的に登録
        for char in unique_chars:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inv_vocab[idx] = char

        # 4. 出現頻度の高い語彙を登録
        for token, _ in token_counts.most_common():
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.inv_vocab[idx] = token

    def save(self, model_path):
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        tokens = self.tokenize(text)
        encoded_ids = []
        for t in tokens:
            if t in self.vocab:
                # 登録されている単語ならそのままID化
                encoded_ids.append(self.vocab[t])
            else:
                # 辞書にないレア単語は、1文字ずつに分解してID化
                for char in t:
                    # fit時に存在しなかった完全初見の文字が来た場合のみ <unk>
                    encoded_ids.append(self.vocab.get(char, self.vocab["<unk>"]))
        return encoded_ids

    def decode(self, ids):
        return "".join([self.inv_vocab.get(i, "<unk>") for i in ids])


class SentencePieceTokenizer:
    def __init__(self, model_path=None):
        self.sp = sentencepiece.SentencePieceProcessor()
        if model_path:
            self.sp.load(model_path)

    def tokenize(self, text):
        return self.sp.encode_as_pieces(text)

    def fit(self, corpus_file, vocab_size=32768, model_prefix="sp_model"):
        sentencepiece.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=f"{model_prefix}_{vocab_size}",
            vocab_size=vocab_size,
            model_type="bpe",
            normalization_rule_name="nmt_nfkc",
        )

        self.sp.load(f"{model_prefix}.model")

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
    import glob

    dataset = []
    files = glob.glob("dataset/training/**/*.txt", recursive=True)
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            dataset.append(f.read())

    tokenizer = MeCabTokenizer()
    tokenizer.fit("\n".join(dataset))
    tokenizer.save("vocab.json")

    print(f"Vocabulary size: {tokenizer.vocab_size}")
