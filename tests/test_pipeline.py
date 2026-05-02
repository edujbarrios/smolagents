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

"""Tests for AgentConfig, AgentFactory, PipelineStep, and AgentPipeline."""

import unittest

from smolagents import CodeAgent, ToolCallingAgent, VLMCodeAgent
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage, get_usage_tracker
from smolagents.pipeline import AgentConfig, AgentFactory, AgentPipeline, PipelineStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModel(Model):
    """Returns a deterministic final-answer code block."""

    def __init__(self, answer: str = "ok", **kwargs):
        super().__init__(model_id="fake-model", **kwargs)
        self._answer = answer

    def generate(self, messages, **kwargs):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f'<code>final_answer("{self._answer}")</code>',
            token_usage=TokenUsage(input_tokens=4, output_tokens=2),
        )


def _make_config(answer: str = "ok", **kwargs) -> AgentConfig:
    return AgentConfig(
        agent_class=CodeAgent,
        model=_FakeModel(answer=answer),
        max_steps=3,
        verbosity_level=-1,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# AgentConfig tests
# ---------------------------------------------------------------------------


class TestAgentConfig(unittest.TestCase):
    def test_validate_ok(self):
        cfg = _make_config()
        cfg.validate()  # should not raise

    def test_validate_bad_class(self):
        cfg = _make_config()
        cfg.agent_class = str  # not a MultiStepAgent subclass
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_bad_max_steps(self):
        cfg = _make_config()
        cfg.max_steps = 0
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_bad_planning_interval(self):
        cfg = _make_config()
        cfg.planning_interval = -1
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_planning_interval_none_ok(self):
        cfg = _make_config(planning_interval=None)
        cfg.validate()  # None is allowed

    def test_to_dict_contains_required_keys(self):
        cfg = _make_config(instructions="do this")
        d = cfg.to_dict()
        self.assertEqual(d["agent_class"], "CodeAgent")
        self.assertEqual(d["max_steps"], 3)
        self.assertEqual(d["instructions"], "do this")

    def test_supports_vlm_agent_class(self):
        cfg = AgentConfig(agent_class=VLMCodeAgent, model=_FakeModel(), max_steps=2, verbosity_level=-1)
        cfg.validate()

    def test_additional_kwargs_stored(self):
        cfg = AgentConfig(
            agent_class=CodeAgent,
            model=_FakeModel(),
            additional_kwargs={"additional_authorized_imports": ["json"]},
        )
        self.assertIn("additional_authorized_imports", cfg.additional_kwargs)


# ---------------------------------------------------------------------------
# AgentFactory tests
# ---------------------------------------------------------------------------


class TestAgentFactory(unittest.TestCase):
    def setUp(self):
        self.factory = AgentFactory()
        self.factory.register("agent_a", _make_config(answer="A"))
        self.factory.register("agent_b", _make_config(answer="B"))

    def test_registered_names(self):
        self.assertEqual(self.factory.registered_names, ["agent_a", "agent_b"])

    def test_create_returns_agent(self):
        agent = self.factory.create("agent_a")
        self.assertIsInstance(agent, CodeAgent)

    def test_create_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.factory.create("unknown")

    def test_duplicate_register_raises(self):
        with self.assertRaises(ValueError):
            self.factory.register("agent_a", _make_config())

    def test_update_replaces_existing(self):
        self.factory.update("agent_a", _make_config(answer="A-updated"))
        agent = self.factory.create("agent_a")
        self.assertIsInstance(agent, CodeAgent)

    def test_unregister(self):
        self.factory.unregister("agent_a")
        self.assertNotIn("agent_a", self.factory.registered_names)

    def test_unregister_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.factory.unregister("nonexistent")

    def test_create_with_override(self):
        agent = self.factory.create("agent_a", max_steps=7)
        self.assertEqual(agent.max_steps, 7)

    def test_create_from_config(self):
        cfg = _make_config(answer="direct")
        agent = self.factory.create_from_config(cfg)
        self.assertIsInstance(agent, CodeAgent)

    def test_repr(self):
        r = repr(self.factory)
        self.assertIn("agent_a", r)
        self.assertIn("agent_b", r)

    def test_get_config(self):
        cfg = self.factory.get_config("agent_a")
        self.assertIsInstance(cfg, AgentConfig)

    def test_validation_off_accepts_bad_config(self):
        factory = AgentFactory(validate_on_register=False)
        bad_cfg = AgentConfig(agent_class=str, model=_FakeModel(), max_steps=0)
        factory.register("bad", bad_cfg)  # should not raise
        self.assertIn("bad", factory.registered_names)

    def test_creates_separate_instances(self):
        a1 = self.factory.create("agent_a")
        a2 = self.factory.create("agent_a")
        self.assertIsNot(a1, a2)

    def test_creates_vlm_agent(self):
        self.factory.register("vlm", AgentConfig(
            agent_class=VLMCodeAgent,
            model=_FakeModel(),
            verbosity_level=-1,
        ))
        agent = self.factory.create("vlm")
        self.assertIsInstance(agent, VLMCodeAgent)
        self.assertTrue(agent._is_vlm_agent)


# ---------------------------------------------------------------------------
# PipelineStep tests
# ---------------------------------------------------------------------------


class TestPipelineStep(unittest.TestCase):
    def setUp(self):
        self.agent = CodeAgent(tools=[], model=_FakeModel(), verbosity_level=-1)

    def test_default_name_is_class_name(self):
        step = PipelineStep(self.agent)
        self.assertEqual(step.name, "CodeAgent")

    def test_custom_name(self):
        step = PipelineStep(self.agent, name="my-step")
        self.assertEqual(step.name, "my-step")

    def test_default_transforms_are_identity_like(self):
        step = PipelineStep(self.agent)
        self.assertEqual(step.input_transform("hello"), "hello")
        self.assertEqual(step.output_transform(42), 42)

    def test_custom_input_transform(self):
        step = PipelineStep(
            self.agent,
            input_transform=lambda x: f"Task: {x}",
        )
        self.assertEqual(step.input_transform("foo"), "Task: foo")

    def test_custom_output_transform(self):
        step = PipelineStep(
            self.agent,
            output_transform=str.upper,
        )
        self.assertEqual(step.output_transform("hello"), "HELLO")

    def test_repr(self):
        step = PipelineStep(self.agent, name="s1")
        self.assertIn("s1", repr(step))
        self.assertIn("CodeAgent", repr(step))


# ---------------------------------------------------------------------------
# AgentPipeline tests
# ---------------------------------------------------------------------------


class TestAgentPipeline(unittest.TestCase):
    def _agent(self, answer="ok"):
        return CodeAgent(tools=[], model=_FakeModel(answer=answer), verbosity_level=-1)

    def test_empty_steps_raises(self):
        with self.assertRaises(ValueError):
            AgentPipeline([])

    def test_single_step_run(self):
        pipeline = AgentPipeline([self._agent("42")])
        result = pipeline.run("any task")
        self.assertEqual(result, "42")

    def test_two_step_pipeline_chains_output(self):
        """Second agent receives first agent's output as its task."""
        # Both agents always output the same string regardless of input.
        pipeline = AgentPipeline([self._agent("first-out"), self._agent("second-out")])
        result = pipeline.run("start")
        self.assertEqual(result, "second-out")

    def test_num_steps(self):
        pipeline = AgentPipeline([self._agent(), self._agent(), self._agent()])
        self.assertEqual(pipeline.num_steps, 3)

    def test_describe_contains_step_names(self):
        pipeline = AgentPipeline([
            PipelineStep(self._agent(), name="alpha"),
            PipelineStep(self._agent(), name="beta"),
        ])
        desc = pipeline.describe()
        self.assertIn("alpha", desc)
        self.assertIn("beta", desc)

    def test_repr(self):
        pipeline = AgentPipeline([
            PipelineStep(self._agent(), name="s1"),
            PipelineStep(self._agent(), name="s2"),
        ])
        self.assertIn("s1", repr(pipeline))
        self.assertIn("s2", repr(pipeline))

    def test_input_transform_applied(self):
        """input_transform is invoked for steps after the first."""
        received = []

        class _CapturingModel(Model):
            def __init__(self, **kwargs):
                super().__init__(model_id="cap", **kwargs)

            def generate(self, messages, **kwargs):
                # Capture the task from messages
                for msg in messages:
                    if msg.role == MessageRole.USER and isinstance(msg.content, list):
                        for part in msg.content:
                            if part.get("type") == "text":
                                received.append(part["text"])
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content='<code>final_answer("captured")</code>',
                    token_usage=TokenUsage(1, 1),
                )

        second = CodeAgent(tools=[], model=_CapturingModel(), verbosity_level=-1)
        step2 = PipelineStep(
            second,
            input_transform=lambda x: f"TRANSFORMED:{x}",
        )
        pipeline = AgentPipeline([self._agent("first-output"), step2])
        pipeline.run("initial")
        # Check that at least one captured message contains the transformed prefix
        self.assertTrue(any("TRANSFORMED:first-output" in t for t in received))

    def test_output_transform_applied(self):
        step = PipelineStep(self._agent("hello"), output_transform=str.upper)
        pipeline = AgentPipeline([step])
        result = pipeline.run("any")
        self.assertEqual(result, "HELLO")

    def test_pipe_operator_two_agents(self):
        a1 = self._agent("a-out")
        a2 = self._agent("b-out")
        pipeline = a1 | a2
        self.assertIsInstance(pipeline, AgentPipeline)
        self.assertEqual(pipeline.num_steps, 2)

    def test_pipe_operator_three_agents(self):
        a1 = self._agent()
        a2 = self._agent()
        a3 = self._agent("final")
        pipeline = a1 | a2 | a3
        self.assertIsInstance(pipeline, AgentPipeline)
        self.assertEqual(pipeline.num_steps, 3)

    def test_pipe_operator_agent_and_pipeline(self):
        a1 = self._agent()
        a2 = self._agent()
        a3 = self._agent()
        inner = AgentPipeline([a2, a3])
        combined = a1 | inner
        self.assertEqual(combined.num_steps, 3)

    def test_pipe_operator_pipeline_and_agent(self):
        a1 = self._agent()
        a2 = self._agent()
        a3 = self._agent()
        inner = AgentPipeline([a1, a2])
        combined = inner | a3
        self.assertEqual(combined.num_steps, 3)

    def test_pipeline_with_pipeline_step(self):
        step = PipelineStep(self._agent("ps-out"), name="my-step")
        pipeline = AgentPipeline([step])
        result = pipeline.run("go")
        self.assertEqual(result, "ps-out")

    def test_usage_tracker_records_each_step(self):
        get_usage_tracker().reset()
        pipeline = AgentPipeline([self._agent(), self._agent(), self._agent()])
        pipeline.run("task")
        summary = get_usage_tracker().get_summary()
        # Three agents ran, so run_count should be at least 3.
        self.assertGreaterEqual(summary["run_count"], 3)

    def test_vlm_agent_in_pipeline(self):
        vlm = VLMCodeAgent(tools=[], model=_FakeModel("vlm-answer"), verbosity_level=-1)
        pipeline = AgentPipeline([self._agent("first"), vlm])
        result = pipeline.run("task")
        self.assertEqual(result, "vlm-answer")

    def test_step_error_raises_runtime_error(self):
        class _BrokenModel(Model):
            def __init__(self, **kwargs):
                super().__init__(model_id="broken", **kwargs)

            def generate(self, messages, **kwargs):
                raise ValueError("model exploded")

        broken_agent = CodeAgent(tools=[], model=_BrokenModel(), verbosity_level=-1)
        pipeline = AgentPipeline([self._agent(), broken_agent])
        with self.assertRaises(RuntimeError):
            pipeline.run("task")

    def test_factory_and_pipeline_integration(self):
        factory = AgentFactory()
        factory.register("a", _make_config(answer="stage-a"))
        factory.register("b", _make_config(answer="stage-b"))
        factory.register("c", _make_config(answer="stage-c"))

        pipeline = AgentPipeline([
            factory.create("a"),
            factory.create("b"),
            factory.create("c"),
        ])
        result = pipeline.run("start")
        self.assertEqual(result, "stage-c")


if __name__ == "__main__":
    unittest.main()
