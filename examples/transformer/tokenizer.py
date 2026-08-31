import MeCab
import ipadic
import numpy as np
import collections


class MeCabTokenizer:
    def __init__(self):
        mecab_args = f"-Owakati {ipadic.MECAB_ARGS}"
        # -Owakati オプションで分かち書きモードにする
        self.tagger = MeCab.Tagger(mecab_args)
        self.vocab = {"<unk>": 0, "<pad>": 1}
        self.inv_vocab = {0: "<unk>", 1: "<pad>"}

    def tokenize(self, text):
        # 文章を単語リストに分解 ("吾輩は猫である\n" -> ["吾輩", "は", "猫", "で", "ある"])
        return self.tagger.parse(text).strip().split()

    def fit(self, corpus_text, min_count=5):
        # 1. コーパス全体の単語の出現頻度をカウント
        tokens = self.tokenize(corpus_text)
        token_counts = collections.Counter(tokens)
        token_counts = collections.Counter({k: v for k, v in token_counts.items() if v >= min_count})

        # 2. コーパスに含まれる「すべての単文字」を抽出 (未知語対策)
        unique_chars = sorted(list(set(corpus_text)))
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


if __name__ == "__main__":
    import glob

    dataset = []
    files = glob.glob("examples/transformer/dataset/**/*.txt", recursive=True)
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            dataset.append(f.read())

    dataset = "\n".join(dataset)
    tokenizer = MeCabTokenizer()
    tokenizer.fit(dataset)
    dataset = tokenizer.encode(dataset)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
