import json
import tempfile
import unittest
from pathlib import Path

from constellation.sampling import sample_jsonl


class SamplingTests(unittest.TestCase):
    def test_sample_jsonl_limits_each_source_deterministically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.jsonl"
            output_path = Path(tmpdir) / "out.jsonl"
            rows = [
                {"id": f"a-{index}", "source_dataset": "a", "sample_type": "agent"}
                for index in range(5)
            ] + [
                {"id": f"b-{index}", "source_dataset": "b", "sample_type": "agent"}
                for index in range(5)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = sample_jsonl(
                input_path=input_path,
                output_path=output_path,
                group_by="source_dataset",
                max_per_group=2,
                seed="fixed",
            )
            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["seen"], 10)
            self.assertEqual(summary["written"], 4)
            self.assertEqual(summary["selected_by_group"], {"a": 2, "b": 2})
            self.assertEqual(
                sorted(row["source_dataset"] for row in written),
                ["a", "a", "b", "b"],
            )


if __name__ == "__main__":
    unittest.main()
