import tempfile
import unittest
from pathlib import Path

from constellation.eval import score_generation, valid_tool_call
from constellation.formatting import IGNORE_INDEX, tokenize_with_loss_mask
from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample, CanonicalTurn
from constellation.subsets import build_debugging_pilot_subsets


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


def sample(sample_id, capabilities, quality=0.9, source="fixture", original_id=None):
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


if __name__ == "__main__":
    unittest.main()
