from functools import reduce
from collections import Counter
from operator import iconcat


class CountVectorizer:
    def __init__(self, ngram_size):
        self.ngram_size = ngram_size
        self.vocab = None

    def fit(self, corpus):
        result = []
        reduce(iconcat, map(self.tokenize_word, corpus), result)
        self.vocab = dict(zip(sorted(set(result)), range(len(result))))

    def transform(self, corpus):
        return list(map(self.encode_token, corpus))

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)

    def tokenize_word(self, word: str) -> [str]:
        return [word[index:index + self.ngram_size]
                for index in range(len(word) - self.ngram_size + 1)]

    def encode_token(self, word: str) -> [int]:
        counter = Counter(self.tokenize_word(word))
        return [counter[key] if key in counter else 0 for key in self.vocab]


