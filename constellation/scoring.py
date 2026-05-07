"""Lightweight quality scoring for data triage."""

from __future__ import annotations

from constellation.filtering import repetition_ratio, token_count_estimate
from constellation.schema import CanonicalSample


def score_sample(sample: CanonicalSample) -> float:
    score = 0.35

    if sample.success is True:
        score += 0.25
    elif sample.success is False:
        score -= 0.25

    if sample.trainable_turn_count() >= 2:
        score += 0.10
    if any(turn.type == "tool_call" for turn in sample.messages):
        score += 0.10
    if any(turn.type == "observation" for turn in sample.messages):
        score += 0.10
    if sample.capabilities:
        score += 0.05

    tokens = token_count_estimate(sample)
    if 256 <= tokens <= 12000:
        score += 0.10
    elif tokens > 24000:
        score -= 0.10

    score -= min(0.25, repetition_ratio(sample.joined_text()))

    return round(min(1.0, max(0.0, score)), 4)


def with_quality_score(sample: CanonicalSample) -> CanonicalSample:
    sample.quality_score = score_sample(sample)
    return sample
