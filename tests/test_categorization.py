import json
import tempfile
import unittest
from pathlib import Path

from constellation.categorization import (
    export_classifier_jsonl,
    export_labeling_prompts_jsonl,
    relabel_jsonl,
)
from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample, CanonicalTurn
from constellation.taxonomy import CapabilityTaxonomy


def canonical_sample(sample_id="sample-1"):
    return CanonicalSample(
        id=sample_id,
        source_dataset="fixture",
        sample_type="agent",
        messages=[
            CanonicalTurn(
                role="user",
                type="message",
                content="Debug the failing pytest run and patch the bug.",
            ),
            CanonicalTurn(
                role="assistant",
                type="tool_call",
                content="{\"name\":\"run\",\"arguments\":{\"cmd\":\"pytest\"}}",
                trainable=True,
            ),
            CanonicalTurn(role="tool", type="observation", content="Traceback: AssertionError"),
            CanonicalTurn(
                role="assistant",
                type="final",
                content="The traceback shows a parser bug; patch the implementation.",
                trainable=True,
            ),
        ],
        capabilities=[],
        success=True,
        quality_score=0.9,
        metadata={"category": "bugfix", "subcategory": "command_line"},
    )


def science_sample(sample_id="science-1"):
    return CanonicalSample(
        id=sample_id,
        source_dataset="fixture",
        sample_type="reasoning",
        messages=[
            CanonicalTurn(
                role="user",
                type="message",
                content="Explain the chemistry experiment and reason about the evidence.",
            ),
            CanonicalTurn(
                role="assistant",
                type="reasoning",
                content="Analyze the hypothesis, molecule behavior, and experimental evidence.",
                trainable=True,
            ),
        ],
        capabilities=[],
        domains=[],
        success=True,
        quality_score=0.9,
        metadata={"category": "science"},
    )


class CategorizationTests(unittest.TestCase):
    def test_taxonomy_normalizes_source_aliases(self):
        taxonomy = CapabilityTaxonomy.load()

        self.assertEqual(taxonomy.normalize_label("bugfix"), "DEBUGGING")
        self.assertEqual(taxonomy.normalize_label("command line"), "TERMINAL_WORKFLOW")
        self.assertIsNone(taxonomy.normalize_label("not-a-real-label"))

    def test_relabel_jsonl_writes_evidence_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            write_jsonl(input_path, [canonical_sample().to_dict()])

            summary = relabel_jsonl(
                input_path=input_path,
                output_path=output_path,
                taxonomy_path=Path("configs/capability_taxonomy.json"),
                domain_taxonomy_path=Path("configs/domain_taxonomy.json"),
                min_score=0.45,
            )
            rows = list(iter_jsonl(output_path))

            self.assertEqual(summary["written"], 1)
            self.assertIn("DEBUGGING", rows[0]["capabilities"])
            self.assertIn("TERMINAL_WORKFLOW", rows[0]["capabilities"])
            self.assertIn("CODING_SOFTWARE", rows[0]["domains"])
            self.assertIn("capability_labeling", rows[0]["metadata"])
            self.assertIn("domain_labeling", rows[0]["metadata"])
            self.assertEqual(
                rows[0]["metadata"]["capability_labeling"]["taxonomy_version"],
                "capability-taxonomy-v1",
            )

    def test_export_classifier_jsonl_writes_label_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "classifier.jsonl"
            row = canonical_sample().to_dict()
            row["capabilities"] = ["DEBUGGING", "TERMINAL_WORKFLOW"]
            write_jsonl(input_path, [row])

            summary = export_classifier_jsonl(
                input_path=input_path,
                output_path=output_path,
                taxonomy_path=Path("configs/capability_taxonomy.json"),
                domain_taxonomy_path=Path("configs/domain_taxonomy.json"),
            )
            exported = next(iter_jsonl(output_path))

            self.assertEqual(summary["written"], 1)
            self.assertEqual(exported["labels"], ["DEBUGGING", "TERMINAL_WORKFLOW"])
            self.assertEqual(len(exported["label_vector"]), len(summary["labels"]))
            self.assertEqual(len(exported["domain_vector"]), len(summary["domains"]))
            self.assertGreater(sum(exported["label_vector"]), 0)
            json.dumps(exported)

    def test_export_labeling_prompts_includes_taxonomy_and_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            examples_path = root / "examples.jsonl"
            output_path = root / "prompts.jsonl"
            row = canonical_sample().to_dict()
            row["capabilities"] = ["DEBUGGING", "TERMINAL_WORKFLOW"]
            write_jsonl(input_path, [row])
            write_jsonl(examples_path, [row])

            summary = export_labeling_prompts_jsonl(
                input_path=input_path,
                output_path=output_path,
                taxonomy_path=Path("configs/capability_taxonomy.json"),
                domain_taxonomy_path=Path("configs/domain_taxonomy.json"),
                examples_path=examples_path,
                max_examples_per_label=1,
                max_chars=4000,
                allow_weak_examples=True,
            )
            exported = next(iter_jsonl(output_path))

            self.assertEqual(summary["written"], 1)
            self.assertEqual(summary["example_count"], 1)
            self.assertIn("Return strict JSON only", exported["prompt"])
            self.assertIn("DEBUGGING", exported["prompt"])
            self.assertIn("SCIENCE", exported["prompt"])
            self.assertIn("domains", exported["prompt"])
            self.assertIn("Example 1 trajectory", exported["prompt"])
            self.assertIn("Trajectory to label", exported["prompt"])

    def test_export_labeling_prompts_skips_weak_examples_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            examples_path = root / "examples.jsonl"
            output_path = root / "prompts.jsonl"
            row = canonical_sample().to_dict()
            row["capabilities"] = ["DEBUGGING"]
            row["metadata"]["capability_labeling"] = {"method": "weak_heuristic_v1"}
            write_jsonl(input_path, [row])
            write_jsonl(examples_path, [row])

            summary = export_labeling_prompts_jsonl(
                input_path=input_path,
                output_path=output_path,
                taxonomy_path=Path("configs/capability_taxonomy.json"),
                domain_taxonomy_path=Path("configs/domain_taxonomy.json"),
                examples_path=examples_path,
            )

            self.assertEqual(summary["example_count"], 0)

    def test_relabel_jsonl_tracks_science_domain_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "science.jsonl"
            output_path = root / "science.labeled.jsonl"
            write_jsonl(input_path, [science_sample().to_dict()])

            summary = relabel_jsonl(
                input_path=input_path,
                output_path=output_path,
                taxonomy_path=Path("configs/capability_taxonomy.json"),
                domain_taxonomy_path=Path("configs/domain_taxonomy.json"),
                min_score=0.45,
            )
            row = next(iter_jsonl(output_path))

            self.assertIn("SCIENCE", row["domains"])
            self.assertIn("STRUCTURED_REASONING", row["capabilities"])
            self.assertEqual(summary["domain_counts"]["SCIENCE"], 1)


if __name__ == "__main__":
    unittest.main()
