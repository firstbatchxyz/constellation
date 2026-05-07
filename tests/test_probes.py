import json
import tempfile
import unittest
from pathlib import Path

from constellation.probes import PROBES, probe_report, probe_samples, write_labeling_probes


class ProbeTests(unittest.TestCase):
    def test_probe_samples_are_canonical_and_have_expected_labels(self):
        samples = probe_samples()

        self.assertEqual(len(samples), len(PROBES))
        self.assertTrue(all(sample.metadata.get("expected_domains") for sample in samples))
        self.assertTrue(all(sample.metadata.get("expected_capabilities") for sample in samples))

    def test_write_labeling_probes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "probes.jsonl"
            summary = write_labeling_probes(output_path)
            rows = output_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(summary["written"], len(PROBES))
            self.assertEqual(len(rows), len(PROBES))
            self.assertEqual(json.loads(rows[0])["source_dataset"], "constellation/probes:domain-probes-v1")

    def test_probe_report_compares_expected_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "labeled.jsonl"
            rows = [
                {
                    "id": "ok",
                    "capabilities": ["STRUCTURED_REASONING"],
                    "domains": ["SCIENCE"],
                    "metadata": {
                        "expected_capabilities": ["STRUCTURED_REASONING"],
                        "expected_domains": ["SCIENCE"],
                    },
                },
                {
                    "id": "miss",
                    "capabilities": ["COMPOSITION"],
                    "domains": ["WRITING"],
                    "metadata": {
                        "expected_capabilities": ["REVISION"],
                        "expected_domains": ["WRITING"],
                    },
                },
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = probe_report(input_path)

            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["strict_matches"], 1)
            self.assertEqual(report["capability_expected_coverage"], 0.5)
            self.assertEqual(report["domain_expected_coverage"], 1.0)
            self.assertEqual(report["probes"][1]["capabilities"]["missing"], ["REVISION"])


if __name__ == "__main__":
    unittest.main()
