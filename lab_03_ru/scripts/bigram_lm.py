from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Привести к нижнему регистру и выделить слова с апострофами Unicode."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class BigramLM:
    k: float
    vocabulary: set[str]
    bigram_counts: Counter[tuple[str, str]]
    context_counts: Counter[str]

    @classmethod
    def train(cls, sentences: Iterable[str], *, k: float = 0.5) -> "BigramLM":
        if k <= 0:
            raise ValueError("для сглаживания add-k требуется k > 0")
        tokenized = [tokenize(sentence) for sentence in sentences]
        if not tokenized:
            raise ValueError("обучающий корпус должен содержать хотя бы одно предложение")
        lexical_vocabulary = {word for sentence in tokenized for word in sentence}
        vocabulary = lexical_vocabulary | {UNK, EOS}
        bigram_counts: Counter[tuple[str, str]] = Counter()
        context_counts: Counter[str] = Counter()

        for sentence in tokenized:
            previous = BOS
            for word in sentence + [EOS]:
                if word not in lexical_vocabulary and word != EOS:
                    word = UNK
                bigram_counts[(previous, word)] += 1
                context_counts[previous] += 1
                previous = word

        return cls(k, vocabulary, bigram_counts, context_counts)

    def _context(self, word: str) -> str:
        if word == BOS:
            return BOS
        return word if word in self.vocabulary else UNK

    def _target(self, word: str) -> str:
        return word if word == EOS or word in self.vocabulary else UNK

    def bigram_probability(self, previous: str, word: str) -> float:
        previous = self._context(previous)
        word = self._target(word)
        vocab_size = len(self.vocabulary)  # предсказываются слова, <unk> и </s>, но не <s>
        numerator = # напишите ваш код здесь
        denominator = # напишите ваш код здесь
        return numerator / denominator

    def sentence_logprob(self, sentence: str) -> float:
        previous = BOS
        score = 0.0
        for word in tokenize(sentence) + [EOS]:
            word = self._target(word)
            score += math.log10(self.bigram_probability(previous, word))
            previous = word
        return score
