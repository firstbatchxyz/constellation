import json
import tempfile
import unittest
from pathlib import Path

from constellation.reporting import label_report, write_label_report


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
            self.assertEqual(report["examples"]["capabilities"]["DEBUGGING"], ["a"])
            self.assertEqual(written, json.loads(output_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
