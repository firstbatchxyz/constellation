import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from constellation.llm_labeling import (
    DEFAULT_LLM_LABEL_MODEL,
    LLM_LABEL_METHOD,
    apply_label_guardrails,
    build_llm_label_prompt,
    extract_json_object,
    label_sample_with_llm_response,
    label_response_schema,
    llm_label_jsonl,
    make_openai_chat_generator,
    normalize_llm_payload,
    normalize_openai_base_url,
    openai_chat_completion_url,
    openai_message_content,
    parse_openai_chat_response,
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
    def test_default_llm_label_model_is_qwen35_small(self):
        self.assertEqual(DEFAULT_LLM_LABEL_MODEL, "Qwen/Qwen3.5-0.8B")

    def test_extract_json_object_from_chatty_response(self):
        parsed = extract_json_object(
            'Sure.\n{"capabilities":["DEBUGGING"],"domains":["SCIENCE"],"confidence":0.8}\nDone.'
        )

        self.assertEqual(parsed["capabilities"], ["DEBUGGING"])
        self.assertEqual(parsed["domains"], ["SCIENCE"])

    def test_openai_compatible_helpers(self):
        self.assertEqual(
            normalize_openai_base_url("http://127.0.0.1:30000"),
            "http://127.0.0.1:30000/v1",
        )
        self.assertEqual(
            openai_chat_completion_url("http://127.0.0.1:30000/v1"),
            "http://127.0.0.1:30000/v1/chat/completions",
        )
        content = openai_message_content("label me", model_name="Qwen/Qwen3.5-0.8B", content_format="auto")
        self.assertEqual(content, [{"type": "text", "text": "label me"}])
        self.assertEqual(
            parse_openai_chat_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"capabilities":[],"domains":[],"confidence":0.2}'
                            }
                        }
                    ]
                }
            ),
            '{"capabilities":[],"domains":[],"confidence":0.2}',
        )

    def test_label_response_schema_constrains_labels(self):
        capability_taxonomy = CapabilityTaxonomy.load()
        domain_taxonomy = DomainTaxonomy.load()

        schema = label_response_schema(
            capability_taxonomy=capability_taxonomy,
            domain_taxonomy=domain_taxonomy,
            max_capabilities=4,
            max_domains=2,
        )

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["capabilities"]["maxItems"], 4)
        self.assertIn("DEBUGGING", schema["properties"]["capabilities"]["items"]["enum"])
        self.assertEqual(schema["properties"]["domains"]["maxItems"], 2)
        self.assertIn("SCIENCE", schema["properties"]["domains"]["items"]["enum"])
        self.assertFalse(schema["additionalProperties"])

    def test_openai_generator_sends_structured_response_format(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"capabilities":[],"domains":[],"confidence":0.2,"rationale":"none"}'
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            del timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        generator = make_openai_chat_generator(
            api_base="http://127.0.0.1:30000/v1",
            api_key=None,
            model_name="Qwen/Qwen3.5-0.8B",
            max_new_tokens=64,
            request_timeout=3,
            content_format="auto",
            response_schema={"type": "object", "properties": {}, "required": []},
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            response = generator("label this")

        self.assertIn("response_format", captured["payload"])
        self.assertEqual(captured["payload"]["response_format"]["type"], "json_schema")
        self.assertEqual(captured["payload"]["messages"][0]["content"], [{"type": "text", "text": "label this"}])
        self.assertIn('"capabilities"', response)

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
        self.assertIn("Do not default to CODING_SOFTWARE", prompt)
        self.assertIn("Domain guardrails", prompt)
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
        self.assertEqual(labeled.metadata["capability_labeling"]["method"], LLM_LABEL_METHOD)
        self.assertEqual(labeled.metadata["domain_labeling"]["model"], "fake-qwen")

    def test_label_guardrails_remove_coding_false_positives_and_add_reasoning(self):
        noncoding = CanonicalSample(
            id="medicine",
            source_dataset="fixture",
            sample_type="reasoning",
            messages=[
                CanonicalTurn(
                    role="user",
                    type="message",
                    content=(
                        "Task: Given a patient with fever, productive cough, pleuritic chest pain, "
                        "and low oxygen saturation, build a differential diagnosis."
                    ),
                )
            ],
            quality_score=1.0,
        )

        guarded = apply_label_guardrails(
            noncoding,
            capabilities=["CODEBASE_NAVIGATION"],
            domains=["CODING_SOFTWARE"],
            max_capabilities=4,
            max_domains=2,
        )

        self.assertEqual(guarded["capabilities"], ["STRUCTURED_REASONING"])
        self.assertEqual(guarded["domains"], ["MEDICINE_HEALTH"])
        self.assertEqual(guarded["metadata"]["dropped_capabilities"], ["CODEBASE_NAVIGATION"])
        self.assertEqual(guarded["metadata"]["added_capabilities"], ["STRUCTURED_REASONING"])

    def test_label_guardrails_keep_real_coding_capabilities(self):
        coding = CanonicalSample(
            id="coding",
            source_dataset="fixture",
            sample_type="coding",
            messages=[
                CanonicalTurn(
                    role="user",
                    type="message",
                    content=(
                        "Task: A Python repository has a failing unit test. Search the codebase "
                        "for the renamed function, patch the implementation, and rerun tests."
                    ),
                )
            ],
            quality_score=1.0,
        )
        guarded = apply_label_guardrails(
            coding,
            capabilities=["DEBUGGING", "CODEBASE_NAVIGATION", "CODE_EDITING", "TEST_WRITING"],
            domains=["CODING_SOFTWARE"],
            max_capabilities=4,
            max_domains=2,
        )

        self.assertEqual(
            guarded["capabilities"],
            ["DEBUGGING", "CODEBASE_NAVIGATION", "CODE_EDITING", "TEST_WRITING"],
        )
        self.assertEqual(guarded["domains"], ["CODING_SOFTWARE"])

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
