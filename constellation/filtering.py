"""Basic sample filters for pilot data."""

from __future__ import annotations

from collections import Counter

from constellation.schema import CanonicalSample


def is_malformed(sample: CanonicalSample) -> bool:
    has_user = any(turn.role == "user" for turn in sample.messages)
    has_trainable = any(turn.trainable for turn in sample.messages)
    has_empty = any(not turn.content.strip() for turn in sample.messages)
    return has_empty or not has_user or not has_trainable


def token_count_estimate(sample: CanonicalSample) -> int:
    return max(1, len(sample.joined_text()) // 4)


def repetition_ratio(text: str, *, ngram_size: int = 5) -> float:
    words = text.lower().split()
    if len(words) < ngram_size * 2:
        return 0.0
    ngrams = tuple(tuple(words[index : index + ngram_size]) for index in range(len(words) - ngram_size))
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(ngrams))


def passes_basic_filters(
    sample: CanonicalSample,
    *,
    min_tokens: int = 64,
    max_tokens: int = 32768,
    max_repetition_ratio: float = 0.20,
    require_success: bool = False,
) -> bool:
    if is_malformed(sample):
        return False
    if require_success and sample.success is not True:
        return False
    tokens = token_count_estimate(sample)
    if tokens < min_tokens or tokens > max_tokens:
        return False
    if repetition_ratio(sample.joined_text()) > max_repetition_ratio:
        return False
    return True
