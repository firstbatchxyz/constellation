import json
import tempfile
import unittest
from pathlib import Path

from constellation.llm_labeling import (
    build_llm_label_prompt,
    extract_json_object,
    label_sample_with_llm_response,
    llm_label_jsonl,
    normalize_llm_payload,
)
from constellation.schema import CanonicalSample, CanonicalTurn
from constellation.taxonomy import CapabilityTaxonomy, DomainTaxonomy


def sample() -> CanonicalSample:
    return CanonicalSample(
        id="science-debug-1",
        source_dataset="fixture",
        sample_type="agent",
        messages=[
            CanonicalTurn(
                role="user",
                type="message",
                content="Task Description: Debug a physics simulation that violates energy conservation.",
            ),
            CanonicalTurn(
                role="assistant",
                type="final",
                content="I found the integration step bug and explained the physics evidence.",
                trainable=True,
            ),
        ],
        quality_score=1.0,
    )


class LLMLabelingTests(unittest.TestCase):
    def test_extract_json_object_from_chatty_response(self):
        parsed = extract_json_object(
            'Sure.\n{"capabilities":["DEBUGGING"],"domains":["SCIENCE"],"confidence":0.8}\nDone.'
        )

        self.assertEqual(parsed["capabilities"], ["DEBUGGING"])
        self.assertEqual(parsed["domains"], ["SCIENCE"])

    def test_normalize_llm_payload_validates_labels_and_limits(self):
        normalized = normalize_llm_payload(
            {
                "capabilities": ["debugging", "STRUCTURED_REASONING", "NOPE"],
                "domains": ["natural science", "WRITING"],
                "confidence": "0.91",
                "rationale": "evidence",
            },
            capability_taxonomy=CapabilityTaxonomy.load(),
            domain_taxonomy=DomainTaxonomy.load(),
            max_capabilities=1,
            max_domains=2,
        )

        self.assertEqual(normalized["capabilities"], ["DEBUGGING"])
        self.assertEqual(normalized["domains"], ["SCIENCE", "WRITING"])
        self.assertEqual(normalized["confidence"], 0.91)

    def test_prompt_contains_taxonomies_and_json_contract(self):
        prompt = build_llm_label_prompt(
            sample(),
            capability_taxonomy=CapabilityTaxonomy.load(),
            domain_taxonomy=DomainTaxonomy.load(),
            max_chars=1000,
        )

        self.assertIn("STRUCTURED_REASONING", prompt)
        self.assertIn("SCIENCE", prompt)
        self.assertIn("strict JSON object", prompt)
        self.assertIn("Debug a physics simulation", prompt)

    def test_label_sample_with_llm_response_updates_metadata(self):
        labeled, parsed_ok = label_sample_with_llm_response(
            sample(),
            response_text=json.dumps(
                {
                    "capabilities": ["DEBUGGING", "STRUCTURED_REASONING"],
                    "domains": ["SCIENCE"],
                    "confidence": 0.87,
                    "rationale": "debugging a physics simulation",
                }
            ),
            capability_taxonomy=CapabilityTaxonomy.load(),
            domain_taxonomy=DomainTaxonomy.load(),
            model_name="fake-qwen",
            max_capabilities=4,
            max_domains=2,
        )

        self.assertTrue(parsed_ok)
        self.assertEqual(labeled.capabilities, ["DEBUGGING", "STRUCTURED_REASONING"])
        self.assertEqual(labeled.domains, ["SCIENCE"])
        self.assertEqual(labeled.metadata["capability_labeling"]["method"], "llm_json_v1")
        self.assertEqual(labeled.metadata["domain_labeling"]["model"], "fake-qwen")

    def test_llm_label_jsonl_accepts_fake_generator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.jsonl"
            output_path = Path(tmpdir) / "out.jsonl"
            input_path.write_text(json.dumps(sample().to_dict()) + "\n", encoding="utf-8")

            summary = llm_label_jsonl(
                input_path=input_path,
                output_path=output_path,
                taxonomy_path=Path("configs/capability_taxonomy.json"),
                domain_taxonomy_path=Path("configs/domain_taxonomy.json"),
                model_name="fake-qwen",
                generator=lambda prompt: json.dumps(
                    {
                        "capabilities": ["DEBUGGING"],
                        "domains": ["SCIENCE"],
                        "confidence": 0.8,
                        "rationale": "physics debugging",
                    }
                ),
            )

            self.assertEqual(summary["written"], 1)
            self.assertEqual(summary["parse_errors"], 0)
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(row["capabilities"], ["DEBUGGING"])
            self.assertEqual(row["domains"], ["SCIENCE"])


if __name__ == "__main__":
    unittest.main()
