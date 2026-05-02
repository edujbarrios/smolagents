# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for VLMCodeAgent, ImageAnalysisTool, and UsageTracker."""

import unittest

from smolagents import CodeAgent, VLMCodeAgent
from smolagents.default_tools import ImageAnalysisTool
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage, UsageTracker, get_usage_tracker


# ---------------------------------------------------------------------------
# Minimal fake model for tests that do not require a real LLM.
# ---------------------------------------------------------------------------


class _FakeModel(Model):
    """A deterministic fake model that always returns a final-answer code block."""

    def __init__(self, model_id: str = "fake-model", **kwargs):
        super().__init__(model_id=model_id, **kwargs)

    def generate(self, messages, **kwargs):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content='<code>final_answer("ok")</code>',
            token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        )


class _FakeVLMModel(Model):
    """Fake VLM model that echoes the text part of the last user message."""

    def __init__(self, model_id: str = "fake-vlm", **kwargs):
        super().__init__(model_id=model_id, **kwargs)

    def generate(self, messages, **kwargs):
        # For ImageAnalysisTool tests the last message has image+text content.
        last_message = messages[-1]
        if isinstance(last_message.content, list):
            question = next(
                (p["text"] for p in last_message.content if p.get("type") == "text"),
                "?",
            )
        else:
            question = str(last_message.content)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"Vision answer: {question}",
            token_usage=TokenUsage(input_tokens=8, output_tokens=4),
        )


# ---------------------------------------------------------------------------
# UsageTracker tests
# ---------------------------------------------------------------------------


class TestUsageTracker(unittest.TestCase):
    def setUp(self):
        # Always start each test from a clean state.
        get_usage_tracker().reset()

    def test_singleton(self):
        t1 = get_usage_tracker()
        t2 = get_usage_tracker()
        self.assertIs(t1, t2)

    def test_initial_state(self):
        tracker = get_usage_tracker()
        summary = tracker.get_summary()
        self.assertEqual(summary["run_count"], 0)
        self.assertEqual(summary["vlm_run_count"], 0)
        self.assertEqual(summary["model_invocations"], {})
        self.assertEqual(summary["tool_invocations"], {})
        self.assertEqual(summary["total_token_usage"]["total_tokens"], 0)

    def test_record_run_increments_count(self):
        tracker = get_usage_tracker()
        tracker.record_run(model_id="m1")
        tracker.record_run(model_id="m1")
        tracker.record_run(model_id="m2")
        summary = tracker.get_summary()
        self.assertEqual(summary["run_count"], 3)
        self.assertEqual(summary["model_invocations"]["m1"], 2)
        self.assertEqual(summary["model_invocations"]["m2"], 1)

    def test_record_vlm_run(self):
        tracker = get_usage_tracker()
        tracker.record_run(is_vlm=True)
        tracker.record_run(is_vlm=False)
        summary = tracker.get_summary()
        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["vlm_run_count"], 1)

    def test_record_tool_call(self):
        tracker = get_usage_tracker()
        tracker.record_tool_call("web_search")
        tracker.record_tool_call("web_search")
        tracker.record_tool_call("python_interpreter")
        summary = tracker.get_summary()
        self.assertEqual(summary["tool_invocations"]["web_search"], 2)
        self.assertEqual(summary["tool_invocations"]["python_interpreter"], 1)

    def test_token_usage_accumulation(self):
        tracker = get_usage_tracker()
        tracker.record_run(token_usage=TokenUsage(input_tokens=10, output_tokens=5))
        tracker.record_run(token_usage=TokenUsage(input_tokens=20, output_tokens=8))
        tu = tracker.total_token_usage
        self.assertEqual(tu.input_tokens, 30)
        self.assertEqual(tu.output_tokens, 13)
        self.assertEqual(tu.total_tokens, 43)

    def test_reset(self):
        tracker = get_usage_tracker()
        tracker.record_run(model_id="m", is_vlm=True, token_usage=TokenUsage(10, 5))
        tracker.record_tool_call("tool")
        tracker.reset()
        summary = tracker.get_summary()
        self.assertEqual(summary["run_count"], 0)
        self.assertEqual(summary["vlm_run_count"], 0)
        self.assertEqual(summary["model_invocations"], {})
        self.assertEqual(summary["tool_invocations"], {})
        self.assertEqual(summary["total_token_usage"]["total_tokens"], 0)

    def test_agent_run_updates_tracker(self):
        """Running CodeAgent updates run_count and model_invocations."""
        tracker = get_usage_tracker()
        agent = CodeAgent(tools=[], model=_FakeModel(model_id="test-model"))
        agent.run("dummy task")
        summary = tracker.get_summary()
        self.assertEqual(summary["run_count"], 1)
        self.assertEqual(summary["model_invocations"].get("test-model", 0), 1)

    def test_vlm_agent_run_increments_vlm_count(self):
        tracker = get_usage_tracker()
        agent = VLMCodeAgent(tools=[], model=_FakeModel())
        agent.run("dummy task")
        summary = tracker.get_summary()
        self.assertEqual(summary["run_count"], 1)
        self.assertEqual(summary["vlm_run_count"], 1)

    def test_code_agent_run_does_not_increment_vlm_count(self):
        tracker = get_usage_tracker()
        agent = CodeAgent(tools=[], model=_FakeModel())
        agent.run("dummy task")
        summary = tracker.get_summary()
        self.assertEqual(summary["run_count"], 1)
        self.assertEqual(summary["vlm_run_count"], 0)


# ---------------------------------------------------------------------------
# VLMCodeAgent tests
# ---------------------------------------------------------------------------


class TestVLMCodeAgent(unittest.TestCase):
    def test_is_vlm_agent_flag(self):
        agent = VLMCodeAgent(tools=[], model=_FakeModel())
        self.assertTrue(agent._is_vlm_agent)

    def test_code_agent_is_not_vlm(self):
        agent = CodeAgent(tools=[], model=_FakeModel())
        self.assertFalse(agent._is_vlm_agent)

    def test_default_vlm_instructions_applied(self):
        agent = VLMCodeAgent(tools=[], model=_FakeModel())
        instructions = agent.instructions.lower()
        self.assertTrue(
            "vision" in instructions or "image" in instructions,
            "Default VLM instructions should mention vision or images.",
        )

    def test_custom_instructions_override_default(self):
        custom = "My specialized instructions for testing."
        agent = VLMCodeAgent(tools=[], model=_FakeModel(), instructions=custom)
        self.assertEqual(agent.instructions, custom)

    def test_vlm_agent_run_returns_result(self):
        agent = VLMCodeAgent(tools=[], model=_FakeModel(), verbosity_level=-1)
        result = agent.run("Analyze this image.")
        self.assertEqual(result, "ok")

    def test_in_agent_registry(self):
        from smolagents.agents import AGENT_REGISTRY

        self.assertIn("VLMCodeAgent", AGENT_REGISTRY)
        self.assertIs(AGENT_REGISTRY["VLMCodeAgent"], VLMCodeAgent)


# ---------------------------------------------------------------------------
# ImageAnalysisTool tests
# ---------------------------------------------------------------------------


class TestImageAnalysisTool(unittest.TestCase):
    def _make_dummy_image(self):
        try:
            import PIL.Image

            return PIL.Image.new("RGB", (10, 10), color=(128, 64, 32))
        except ImportError:
            self.skipTest("PIL not available")

    def test_tool_metadata(self):
        tool = ImageAnalysisTool(model=_FakeVLMModel())
        self.assertEqual(tool.name, "image_analysis")
        self.assertIn("image", tool.inputs)
        self.assertIn("question", tool.inputs)
        self.assertEqual(tool.output_type, "string")

    def test_forward_returns_string(self):
        tool = ImageAnalysisTool(model=_FakeVLMModel())
        image = self._make_dummy_image()
        result = tool.forward(image=image, question="What color is the dominant hue?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_forward_echoes_question(self):
        """The fake model echoes the question; ensure it reaches forward correctly."""
        tool = ImageAnalysisTool(model=_FakeVLMModel())
        image = self._make_dummy_image()
        question = "How many objects are visible?"
        result = tool.forward(image=image, question=question)
        self.assertIn(question, result)

    def test_tool_usable_in_agent(self):
        """ImageAnalysisTool can be added to an agent without errors."""
        vlm_model = _FakeVLMModel()
        tool = ImageAnalysisTool(model=vlm_model)
        # Just verify it can be registered as an agent tool.
        agent = VLMCodeAgent(tools=[tool], model=_FakeModel(), verbosity_level=-1)
        self.assertIn("image_analysis", agent.tools)


if __name__ == "__main__":
    unittest.main()
