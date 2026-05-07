import tempfile
import unittest
from pathlib import Path

from constellation.eval import score_generation, valid_tool_call
from constellation.formatting import IGNORE_INDEX, tokenize_with_loss_mask
from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample, CanonicalTurn
from constellation.streaming import DatasetSource, stream_convert
from constellation.subsets import build_debugging_pilot_subsets


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


def sample(
    sample_id,
    capabilities,
    domains=None,
    quality=0.9,
    source="fixture",
    original_id=None,
):
    return CanonicalSample(
        id=sample_id,
        source_dataset=source,
        sample_type="agent",
        messages=[
            CanonicalTurn(role="user", type="message", content=f"Debug failing test {sample_id}."),
            CanonicalTurn(
                role="assistant",
                type="reasoning",
                content=f"Inspect failure {sample_id}.",
                trainable=True,
            ),
            CanonicalTurn(
                role="assistant",
                type="tool_call",
                content="{\"name\":\"run\",\"arguments\":{\"cmd\":\"pytest\"}}",
                trainable=True,
            ),
            CanonicalTurn(
                role="tool",
                type="observation",
                content=f"AssertionError in parser_{sample_id}.py",
            ),
            CanonicalTurn(
                role="assistant",
                type="final",
                content=f"Patch parser_{sample_id}.py and rerun tests.",
                trainable=True,
            ),
        ],
        capabilities=capabilities,
        domains=domains or [],
        success=True,
        quality_score=quality,
        metadata={"original_id": original_id or sample_id},
    )


class PilotPipelineTests(unittest.TestCase):
    def test_tokenize_with_loss_mask_masks_observations(self):
        canonical = sample("a", ["DEBUGGING", "TOOL_USE"])
        encoded = tokenize_with_loss_mask(FakeTokenizer(), canonical, max_length=4096)
        text = "".join(chr(token_id) for token_id in encoded["input_ids"] if token_id != 0)
        observation_index = text.index("AssertionError")
        reasoning_index = text.index("Inspect failure")

        self.assertEqual(encoded["labels"][observation_index], IGNORE_INDEX)
        self.assertNotEqual(encoded["labels"][reasoning_index], IGNORE_INDEX)

    def test_build_debugging_pilot_subsets_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "canonical.jsonl"
            rows = [
                sample("debug-1", ["DEBUGGING", "TOOL_USE"]).to_dict(),
                sample("debug-2", ["DEBUGGING", "TOOL_USE"]).to_dict(),
                sample("terminal-1", ["TERMINAL_WORKFLOW"]).to_dict(),
                sample("planner-1", ["PLANNING"]).to_dict(),
            ]
            write_jsonl(input_path, rows)

            first = build_debugging_pilot_subsets(
                inputs=[input_path],
                output_dir=root / "first",
                max_train_tokens=10000,
                eval_fraction=0.5,
                eval_max_samples=1,
                min_tokens=1,
                seed="fixture",
            )
            second = build_debugging_pilot_subsets(
                inputs=[input_path],
                output_dir=root / "second",
                max_train_tokens=10000,
                eval_fraction=0.5,
                eval_max_samples=1,
                min_tokens=1,
                seed="fixture",
            )

            self.assertEqual(first["counts"], second["counts"])
            first_train = list(iter_jsonl(root / "first" / "debugging_specialist.train.jsonl"))
            second_train = list(iter_jsonl(root / "second" / "debugging_specialist.train.jsonl"))
            self.assertEqual(
                [row["id"] for row in first_train],
                [row["id"] for row in second_train],
            )

    def test_build_domain_capability_target_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "canonical.jsonl"
            rows = [
                sample(
                    "science-1",
                    ["STRUCTURED_REASONING"],
                    domains=["SCIENCE"],
                ).to_dict(),
                sample(
                    "science-2",
                    ["STRUCTURED_REASONING"],
                    domains=["SCIENCE"],
                ).to_dict(),
                sample(
                    "writing-1",
                    ["COMPOSITION"],
                    domains=["WRITING"],
                ).to_dict(),
            ]
            write_jsonl(input_path, rows)

            manifest = build_debugging_pilot_subsets(
                inputs=[input_path],
                output_dir=root / "science",
                target_capability="",
                target_capabilities=["STRUCTURED_REASONING"],
                target_domains=["SCIENCE"],
                output_prefix="science_reasoner",
                max_train_tokens=10000,
                eval_fraction=0.5,
                eval_max_samples=1,
                min_tokens=1,
                seed="fixture",
            )
            train_rows = list(iter_jsonl(root / "science" / "science_reasoner.train.jsonl"))

            self.assertEqual(manifest["target_capabilities"], ["STRUCTURED_REASONING"])
            self.assertEqual(manifest["target_domains"], ["SCIENCE"])
            self.assertTrue(all("SCIENCE" in row["domains"] for row in train_rows[:1]))

    def test_eval_metrics_detect_tool_call_and_observation_overlap(self):
        canonical = sample("b", ["DEBUGGING", "TOOL_USE"])
        generation = (
            "<tool_call>{\"name\":\"run\",\"arguments\":{\"cmd\":\"pytest\"}}</tool_call>\n"
            "The AssertionError points at parser_b.py, so I will patch the parser."
        )
        metrics = score_generation(canonical, generation)

        self.assertTrue(valid_tool_call(generation))
        self.assertTrue(metrics["valid_tool_call"])
        self.assertTrue(metrics["observation_grounded"])
        self.assertTrue(metrics["patch_intent"])

    def test_stream_convert_does_not_prefetch_past_max_rows(self):
        import constellation.streaming as streaming

        original_iter_hf_rows = streaming.iter_hf_rows

        def fake_iter_hf_rows(source):
            del source
            for index in range(10):
                yield {
                    "path": f"task-{index}",
                    "result": "success",
                    "conversations": [
                        {"role": "user", "content": "Debug this failing test."},
                        {
                            "role": "assistant",
                            "content": "I inspected the traceback and found the parser bug.",
                        },
                    ],
                }
            raise AssertionError("stream_convert prefetched past max_rows")

        try:
            streaming.iter_hf_rows = fake_iter_hf_rows
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "out.jsonl"
                stats = stream_convert(
                    source=DatasetSource(dataset_path="fixture", parser="agenttrove"),
                    output=output,
                    max_rows=10,
                    min_tokens=1,
                    max_tokens=32768,
                    min_quality=0.0,
                    require_success=False,
                    skip_errors=False,
                )
                self.assertEqual(stats["seen"], 10)
                self.assertEqual(stats["errors"], 0)
                self.assertEqual(sum(1 for _ in iter_jsonl(output)), 10)
        finally:
            streaming.iter_hf_rows = original_iter_hf_rows


if __name__ == "__main__":
    unittest.main()
