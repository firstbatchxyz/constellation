"""Handpicked labeling probe set for quick taxonomy calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample, CanonicalTurn

PROBE_SUITE_VERSION = "domain-probes-v1"


PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "probe_science_reasoning",
        "sample_type": "reasoning",
        "content": (
            "Task: Explain why a candle flame goes out after a glass jar is placed over it. "
            "Use evidence about oxygen, combustion products, heat, and pressure. Compare two "
            "alternative hypotheses and decide which explanation is best supported."
        ),
        "expected_capabilities": ["STRUCTURED_REASONING"],
        "expected_domains": ["SCIENCE"],
    },
    {
        "id": "probe_science_retrieval",
        "sample_type": "reasoning",
        "content": (
            "Task: Search the scientific literature for recent evidence on ocean acidification "
            "effects on coral reef calcification. Summarize the strongest findings, note "
            "uncertainties, and cite the kinds of sources that should be checked."
        ),
        "expected_capabilities": ["RETRIEVAL_SEARCH", "STRUCTURED_REASONING"],
        "expected_domains": ["SCIENCE"],
    },
    {
        "id": "probe_math_proof",
        "sample_type": "reasoning",
        "content": (
            "Task: Prove by induction that the sum of the first n odd positive integers is n^2. "
            "Show the base case, induction hypothesis, induction step, and final conclusion."
        ),
        "expected_capabilities": ["STRUCTURED_REASONING"],
        "expected_domains": ["MATHEMATICS"],
    },
    {
        "id": "probe_humanities_analysis",
        "sample_type": "reasoning",
        "content": (
            "Task: Analyze how Sophocles uses burial imagery and civic duty in Antigone to "
            "frame the conflict between divine law and state authority. Support the argument "
            "with close reading rather than plot summary."
        ),
        "expected_capabilities": ["STRUCTURED_REASONING"],
        "expected_domains": ["HUMANITIES"],
    },
    {
        "id": "probe_writing_composition",
        "sample_type": "reasoning",
        "content": (
            "Task: Draft a personal essay about learning patience while caring for a community "
            "garden. Use a reflective voice, vivid concrete scenes, and a clear emotional arc."
        ),
        "expected_capabilities": ["COMPOSITION"],
        "expected_domains": ["WRITING"],
    },
    {
        "id": "probe_writing_revision",
        "sample_type": "reasoning",
        "content": (
            "Task: Revise the following paragraph to improve clarity, flow, and tone while "
            "preserving the meaning. Make the prose concise, professional, and easier to scan."
        ),
        "expected_capabilities": ["REVISION"],
        "expected_domains": ["WRITING"],
    },
    {
        "id": "probe_medicine_differential",
        "sample_type": "reasoning",
        "content": (
            "Task: Given a patient with fever, productive cough, pleuritic chest pain, and low "
            "oxygen saturation, build a differential diagnosis. Explain which symptoms support "
            "pneumonia versus pulmonary embolism and what tests would help distinguish them."
        ),
        "expected_capabilities": ["STRUCTURED_REASONING"],
        "expected_domains": ["MEDICINE_HEALTH"],
    },
    {
        "id": "probe_data_analysis",
        "sample_type": "reasoning",
        "content": (
            "Task: Analyze a CSV of subscription cohorts. Compute monthly retention, compare "
            "conversion rates by acquisition channel, identify anomalous cohorts, and describe "
            "which chart would best communicate the trend."
        ),
        "expected_capabilities": ["STRUCTURED_REASONING"],
        "expected_domains": ["DATA_ANALYSIS", "BUSINESS_OPERATIONS"],
    },
    {
        "id": "probe_social_science",
        "sample_type": "reasoning",
        "content": (
            "Task: Design a survey study to estimate whether a housing voucher policy changes "
            "school attendance. Define treatment and comparison groups, confounders, survey "
            "questions, and limitations of causal interpretation."
        ),
        "expected_capabilities": ["PLANNING", "STRUCTURED_REASONING"],
        "expected_domains": ["SOCIAL_SCIENCE"],
    },
    {
        "id": "probe_business_planning",
        "sample_type": "reasoning",
        "content": (
            "Task: Create a rollout plan for a customer support triage process. Include phases, "
            "owners, success metrics, risks, and a weekly operating cadence for the team."
        ),
        "expected_capabilities": ["PLANNING"],
        "expected_domains": ["BUSINESS_OPERATIONS"],
    },
    {
        "id": "probe_general_knowledge",
        "sample_type": "reasoning",
        "content": (
            "Task: Answer a mixed general-knowledge question for a curious reader: explain what "
            "time zones are, why countries use them, and give a simple example of converting "
            "between two cities."
        ),
        "expected_capabilities": ["STRUCTURED_REASONING"],
        "expected_domains": ["GENERAL_KNOWLEDGE"],
    },
    {
        "id": "probe_coding_debugging",
        "sample_type": "coding",
        "content": (
            "Task: A Python repository has a failing unit test after a refactor. Inspect the "
            "traceback, search the codebase for the renamed function, patch the implementation, "
            "and rerun the tests to confirm the fix."
        ),
        "expected_capabilities": ["DEBUGGING", "CODEBASE_NAVIGATION", "CODE_EDITING", "TEST_WRITING"],
        "expected_domains": ["CODING_SOFTWARE"],
    },
    {
        "id": "probe_terminal_recovery",
        "sample_type": "agent",
        "content": (
            "Task: In a shell session, `uv run pytest` fails with ModuleNotFoundError after a "
            "dependency sync. Diagnose the environment, inspect pyproject.toml, retry with the "
            "right command, and recover without discarding local edits."
        ),
        "expected_capabilities": ["TERMINAL_WORKFLOW", "ERROR_RECOVERY", "DEBUGGING"],
        "expected_domains": ["CODING_SOFTWARE"],
    },
)


def probe_samples() -> list[CanonicalSample]:
    samples: list[CanonicalSample] = []
    for probe in PROBES:
        samples.append(
            CanonicalSample(
                id=str(probe["id"]),
                source_dataset=f"constellation/probes:{PROBE_SUITE_VERSION}",
                sample_type=str(probe["sample_type"]),
                messages=[
                    CanonicalTurn(
                        role="user",
                        type="message",
                        content=str(probe["content"]),
                    )
                ],
                quality_score=1.0,
                metadata={
                    "probe_suite": PROBE_SUITE_VERSION,
                    "expected_capabilities": list(probe["expected_capabilities"]),
                    "expected_domains": list(probe["expected_domains"]),
                },
            )
        )
    return samples


def write_labeling_probes(output_path: str | Path) -> dict[str, Any]:
    samples = probe_samples()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = write_jsonl(output_path, (sample.to_dict() for sample in samples))
    return {
        "output": str(output_path),
        "suite": PROBE_SUITE_VERSION,
        "written": written,
    }


def _compare(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    expected_set = set(expected)
    predicted_set = set(predicted)
    covered = sorted(expected_set & predicted_set)
    missing = sorted(expected_set - predicted_set)
    extra = sorted(predicted_set - expected_set)
    return {
        "expected": sorted(expected_set),
        "predicted": sorted(predicted_set),
        "covered": covered,
        "missing": missing,
        "extra": extra,
        "coverage": round(len(covered) / len(expected_set), 4) if expected_set else 1.0,
        "exact_match": not missing and not extra,
    }


def probe_report(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    rows = 0
    capability_covered = 0
    capability_expected = 0
    domain_covered = 0
    domain_expected = 0
    strict_matches = 0
    probes: list[dict[str, Any]] = []

    for row in iter_jsonl(input_path):
        rows += 1
        metadata = dict(row.get("metadata") or {})
        expected_capabilities = list(metadata.get("expected_capabilities") or [])
        expected_domains = list(metadata.get("expected_domains") or [])
        capability_cmp = _compare(expected_capabilities, list(row.get("capabilities") or []))
        domain_cmp = _compare(expected_domains, list(row.get("domains") or []))
        capability_covered += len(capability_cmp["covered"])
        capability_expected += len(capability_cmp["expected"])
        domain_covered += len(domain_cmp["covered"])
        domain_expected += len(domain_cmp["expected"])
        if capability_cmp["exact_match"] and domain_cmp["exact_match"]:
            strict_matches += 1
        probes.append(
            {
                "id": row.get("id"),
                "capabilities": capability_cmp,
                "domains": domain_cmp,
            }
        )

    report = {
        "input": str(input_path),
        "rows": rows,
        "strict_matches": strict_matches,
        "strict_match_rate": round(strict_matches / rows, 4) if rows else 0.0,
        "capability_expected_coverage": round(capability_covered / capability_expected, 4)
        if capability_expected
        else 1.0,
        "domain_expected_coverage": round(domain_covered / domain_expected, 4)
        if domain_expected
        else 1.0,
        "probes": probes,
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
