"""Versioned lexical analyzers for BM25 corpus and query processing."""

import hashlib
import logging
import os
import re
from pathlib import Path


_LATIN_PATTERN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]+")
DEFAULT_DICTIONARY = (
    Path(__file__).resolve().parent / "resources" / "paperstorm_domain_terms.txt"
)


def _latin_tokens(text):
    return _LATIN_PATTERN.findall(str(text or "").lower())


def _cjk_bigrams(sequence):
    return [sequence[index : index + 2] for index in range(len(sequence) - 1)]


class CjkBigramAnalyzer:
    """Compatibility analyzer: Latin terms plus CJK unigrams and bigrams."""

    name = "cjk-bigram"
    revision = "cjk-bigram-v1"

    def tokenize(self, text):
        value = str(text or "").lower()
        tokens = _latin_tokens(value)
        for sequence in _CJK_PATTERN.findall(value):
            tokens.extend(sequence)
            tokens.extend(_cjk_bigrams(sequence))
        return tokens


class JiebaDomainAnalyzer:
    """Private Jieba tokenizer augmented by a versioned technical dictionary."""

    name = "jieba-domain"

    def __init__(self, dictionary_path=None):
        self.dictionary_path = Path(dictionary_path or DEFAULT_DICTIONARY)
        if not self.dictionary_path.is_file():
            raise FileNotFoundError(
                "domain dictionary is missing: {0}".format(self.dictionary_path)
            )
        self.terms = tuple(
            line.strip()
            for line in self.dictionary_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        digest = hashlib.sha256()
        digest.update(b"paperstorm-jieba-domain-v1\0")
        digest.update("\n".join(self.terms).encode("utf-8"))
        self.revision = "jieba-domain-v1:" + digest.hexdigest()[:16]
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                import jieba
            except ImportError as exc:
                raise RuntimeError(
                    "domain-aware BM25 requires dependency jieba"
                ) from exc
            jieba.setLogLevel(logging.ERROR)
            tokenizer = jieba.Tokenizer()
            for term in self.terms:
                if _CJK_PATTERN.search(term):
                    tokenizer.add_word(term, freq=10**9)
            self._tokenizer = tokenizer
        return self._tokenizer

    def tokenize(self, text):
        value = str(text or "").lower()
        tokens = _latin_tokens(value)
        tokenizer = self._get_tokenizer()
        for sequence in _CJK_PATTERN.findall(value):
            tokens.extend(
                token.strip()
                for token in tokenizer.cut(sequence, HMM=False)
                if token.strip()
            )
            tokens.extend(
                term for term in self.terms if term in sequence and _CJK_PATTERN.fullmatch(term)
            )
            tokens.extend(_cjk_bigrams(sequence))
        return list(dict.fromkeys(tokens))


def build_text_analyzer(name=None, dictionary_path=None):
    selected = str(
        name or os.getenv("PAPERSTORM_TEXT_ANALYZER") or "jieba-domain"
    ).strip().lower()
    if selected in {"jieba", "jieba-domain", "domain"}:
        return JiebaDomainAnalyzer(dictionary_path=dictionary_path)
    if selected in {"legacy", "cjk-bigram", "bigram"}:
        return CjkBigramAnalyzer()
    raise ValueError("unsupported text analyzer: {0}".format(selected))
