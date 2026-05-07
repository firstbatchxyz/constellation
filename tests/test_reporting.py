import json
import tempfile
import unittest
from pathlib import Path

from constellation.reporting import label_report, specialist_target_report, write_label_report
from constellation.schema import CanonicalSample, CanonicalTurn


def canonical_row(sample_id, capabilities, domains):
    return CanonicalSample(
        id=sample_id,
        source_dataset="fixture",
        sample_type="agent",
        messages=[
            CanonicalTurn(
                role="user",
                type="message",
                content=f"Task {sample_id}: analyze this trajectory and produce a robust answer.",
            ),
            CanonicalTurn(
                role="assistant",
                type="final",
                content=f"Completed labeled response for {sample_id}.",
                trainable=True,
            ),
        ],
        capabilities=capabilities,
        domains=domains,
        quality_score=0.9,
    ).to_dict()


class ReportingTests(unittest.TestCase):
    def test_label_report_counts_labels_and_empty_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "labels.jsonl"
            output_path = Path(tmpdir) / "report.json"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "a",
                                "capabilities": ["DEBUGGING"],
                                "domains": ["CODING_SOFTWARE"],
                            }
                        ),
                        json.dumps({"id": "b", "capabilities": [], "domains": []}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = label_report(input_path, top_examples=1)
            written = write_label_report(input_path, output_path, top_examples=1)

            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["empty"], 1)
            self.assertEqual(report["sources"], {"unknown": 2})
            self.assertEqual(report["sample_types"], {"unknown": 2})
            self.assertEqual(report["capabilities"]["DEBUGGING"], 1)
            self.assertEqual(report["domains"]["CODING_SOFTWARE"], 1)
            self.assertEqual(
                report["breakdowns"]["capabilities_by_source"]["unknown"]["DEBUGGING"],
                1,
            )
            self.assertEqual(
                report["breakdowns"]["domains_by_sample_type"]["unknown"]["CODING_SOFTWARE"],
                1,
            )
            self.assertEqual(report["examples"]["capabilities"]["DEBUGGING"], ["a"])
            self.assertEqual(written, json.loads(output_path.read_text(encoding="utf-8")))

    def test_specialist_target_report_counts_target_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "labels.jsonl"
            targets_path = root / "targets.json"
            input_path.write_text(
                json.dumps(canonical_row("science", ["STRUCTURED_REASONING"], ["SCIENCE"])) + "\n"
                + json.dumps(canonical_row("debug", ["DEBUGGING"], ["CODING_SOFTWARE"])) + "\n",
                encoding="utf-8",
            )
            targets_path.write_text(
                json.dumps(
                    {
                        "version": "test",
                        "max_distillations": 3,
                        "targets": [
                            {
                                "id": "science_reasoner",
                                "model_name": "Science",
                                "target_capabilities": ["STRUCTURED_REASONING"],
                                "target_domains": ["SCIENCE"],
                            },
                            {
                                "id": "writer",
                                "model_name": "Writer",
                                "target_capabilities": ["COMPOSITION"],
                                "target_domains": ["WRITING"],
                            },
                            {
                                "id": "general_agent",
                                "model_name": "General",
                                "target_capabilities": [],
                                "target_domains": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = specialist_target_report(
                input_path,
                specialist_targets_path=targets_path,
                min_tokens=1,
                min_target_samples=1,
                top_examples=1,
            )

            targets = {target["id"]: target for target in report["targets"]}
            self.assertEqual(report["eligible_rows"], 2)
            self.assertEqual(targets["science_reasoner"]["count"], 1)
            self.assertEqual(targets["science_reasoner"]["status"], "ready")
            self.assertEqual(targets["science_reasoner"]["examples"], ["science"])
            self.assertEqual(targets["writer"]["status"], "empty")
            self.assertEqual(targets["general_agent"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
