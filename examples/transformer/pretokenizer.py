import glob
import os

import ipadic
import MeCab

DELIMITER = "\x9e"
INPUT_DIR = os.path.join("dataset", "training", "**", "*.txt")
OUTPUT_FILE = "corpus_pretokenized.txt"


def build_tagger(delimiter: str) -> MeCab.Tagger:
    mecab_args = f'{ipadic.MECAB_ARGS} -F"%M{delimiter}" -U"%M{delimiter}" -E"\n" -b 5242880'
    return MeCab.Tagger(mecab_args)


def main():
    tagger = build_tagger(DELIMITER)
    files = sorted(glob.glob(INPUT_DIR, recursive=True))
    suffix = f"{DELIMITER}\n"

    with open(OUTPUT_FILE, "w", encoding="utf-8", buffering=1024 * 1024) as out_f:
        for file in files:
            with open(file, "r", encoding="utf-8", buffering=1024 * 1024) as in_f:
                for line in in_f:
                    line = line.strip().replace(DELIMITER, "")
                    if not line:
                        continue

                    parsed = tagger.parse(line).removesuffix(suffix)
                    out_f.write(parsed + "\n")


if __name__ == "__main__":
    main()
