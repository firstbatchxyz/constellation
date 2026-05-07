import unittest

from constellation.filtering import passes_basic_filters
from constellation.parsers import parse_agenttrove_row, parse_hermes_row
from constellation.schema import CanonicalSample


class ParserTests(unittest.TestCase):
    def test_hermes_parser_splits_reasoning_and_tool_call(self):
        sample = parse_hermes_row(
            {
                "id": "abc",
                "category": "Terminal & Coding",
                "subcategory": "Terminal Tasks",
                "task": "Debug a failing Python test",
                "tools": "[{\"name\":\"run\"}]",
                "conversations": [
                    {"from": "system", "value": "You are an agent."},
                    {"from": "human", "value": "Fix the failing test."},
                    {
                        "from": "gpt",
                        "value": (
                            "<think>Inspect the failure first.</think>"
                            "<tool_call>{\"name\":\"run\",\"arguments\":{\"cmd\":\"pytest\"}}</tool_call>"
                        ),
                    },
                    {"from": "tool", "value": "AssertionError: expected 2"},
                    {"from": "gpt", "value": "The test reveals an off-by-one bug."},
                ],
            }
        )

        self.assertEqual(sample.id, "abc")
        self.assertEqual(sample.sample_type, "agent")
        self.assertIn("DEBUGGING", sample.capabilities)
        self.assertIn("TERMINAL_WORKFLOW", sample.capabilities)
        self.assertEqual(
            [turn.type for turn in sample.messages],
            [
                "message",
                "message",
                "reasoning",
                "tool_call",
                "observation",
                "final",
            ],
        )
        self.assertEqual(
            [turn.trainable for turn in sample.messages],
            [
                False,
                False,
                True,
                True,
                False,
                True,
            ],
        )

    def test_agenttrove_parser_uses_reward_success(self):
        sample = parse_agenttrove_row(
            {
                "task_id": "task-1",
                "original_source": "swesmith",
                "original_teacher": "teacher",
                "reward": 1.0,
                "messages": [
                    {"role": "user", "content": "Debug this repo and fix the tests."},
                    {"role": "assistant", "content": "I will inspect the failure."},
                    {"role": "tool", "content": "1 failed, 2 passed"},
                    {"role": "assistant", "content": "Patch applied and tests pass."},
                ],
            }
        )

        self.assertIs(sample.success, True)
        self.assertEqual(sample.metadata["teacher_model"], "teacher")
        self.assertIn("DEBUGGING", sample.capabilities)
        self.assertEqual(CanonicalSample.from_dict(sample.to_dict()).id, "task-1")

    def test_basic_filters_reject_failed_when_required(self):
        sample = parse_agenttrove_row(
            {
                "task_id": "task-2",
                "reward": 0.0,
                "messages": [
                    {"role": "user", "content": "Run tests and debug the failure."},
                    {
                        "role": "assistant",
                        "content": "I inspected the traceback and found the failing assertion in the test.",
                    },
                ],
            }
        )

        self.assertFalse(passes_basic_filters(sample, require_success=True))


if __name__ == "__main__":
    unittest.main()
