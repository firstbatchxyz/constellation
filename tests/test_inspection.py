import json
import tempfile
import unittest
from pathlib import Path

from constellation.inspection import inspect_samples
from constellation.schema import CanonicalSample, CanonicalTurn


def sample_row(sample_id: str, domains: list[str]) -> dict:
    return CanonicalSample(
        id=sample_id,
        source_dataset="fixture",
        sample_type="reasoning",
        messages=[
            CanonicalTurn(
                role="user",
                type="message",
                content="Task: inspect this labeled medical differential diagnosis example.",
            )
        ],
        capabilities=["STRUCTURED_REASONING"],
        domains=domains,
        quality_score=0.9,
        metadata={
            "domain_labeling": {"method": "test"},
            "label_guardrails": {"applied": True},
        },
    ).to_dict()


class InspectionTests(unittest.TestCase):
    def test_inspect_samples_by_domain_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "rows.jsonl"
            input_path.write_text(
                json.dumps(sample_row("a", ["MEDICINE_HEALTH"])) + "\n"
                + json.dumps(sample_row("b", ["SCIENCE"])) + "\n",
                encoding="utf-8",
            )

            report = inspect_samples(
                input_path,
                label="MEDICINE_HEALTH",
                axis="domains",
                limit=2,
            )

            self.assertEqual(report["count"], 1)
            self.assertEqual(report["samples"][0]["id"], "a")
            self.assertEqual(report["samples"][0]["domains"], ["MEDICINE_HEALTH"])
            self.assertIn("medical differential", report["samples"][0]["text"])
            self.assertEqual(report["samples"][0]["metadata"]["domain_labeling"], {"method": "test"})

    def test_inspect_samples_requires_filter(self):
        with self.assertRaises(ValueError):
            inspect_samples("unused.jsonl")


if __name__ == "__main__":
    unittest.main()
