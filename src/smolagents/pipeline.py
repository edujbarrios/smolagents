#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Parametrized agent toolkit and multi-agent pipeline utilities.

This module provides three cooperating primitives:

* :class:`AgentConfig` – a fully parametrized, serialisable description of a
  single agent (class, model, tools, instructions, …).
* :class:`AgentFactory` – creates and optionally caches named agent instances
  from :class:`AgentConfig` objects.
* :class:`PipelineStep` – wraps an agent for use in a pipeline, with optional
  per-step input/output transform functions.
* :class:`AgentPipeline` – connects agents in a linear sequence so that each
  agent's output becomes the next agent's task.

Typical usage::

    from smolagents import CodeAgent, VLMCodeAgent, InferenceClientModel
    from smolagents.pipeline import AgentConfig, AgentFactory, AgentPipeline

    factory = AgentFactory()

    factory.register(
        "researcher",
        AgentConfig(
            agent_class=CodeAgent,
            model=InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking"),
            tools=[WebSearchTool()],
            instructions="You research topics thoroughly.",
            max_steps=10,
        ),
    )
    factory.register(
        "summariser",
        AgentConfig(
            agent_class=CodeAgent,
            model=InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking"),
            instructions="Summarise the provided research concisely.",
        ),
    )

    pipeline = AgentPipeline([
        factory.create("researcher"),
        factory.create("summariser"),
    ])

    result = pipeline.run("What are the latest advances in quantum computing?")

Agents can also be chained with the ``|`` operator::

    pipeline = researcher_agent | summariser_agent
    result = pipeline.run("…")
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import PIL.Image

from .monitoring import LogLevel, get_usage_tracker
from .tools import Tool

logger = getLogger(__name__)


__all__ = [
    "AgentConfig",
    "AgentFactory",
    "PipelineStep",
    "AgentPipeline",
]


# ---------------------------------------------------------------------------
# AgentConfig – fully parametrized agent descriptor
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Full parametrized configuration for a single smolagents agent.

    An :class:`AgentConfig` is a plain data object that describes every
    aspect of an agent without instantiating it.  Pass it to
    :class:`AgentFactory` to create agent instances.

    Args:
        agent_class (`type`): The agent class to instantiate, e.g.
            :class:`~smolagents.CodeAgent`, :class:`~smolagents.ToolCallingAgent`,
            or :class:`~smolagents.VLMCodeAgent`.
        model (`~smolagents.Model`): The language model the agent will use.
        tools (`list[Tool]`, default `[]`): Tools available to the agent.
        instructions (`str`, *optional*): Custom system instructions inserted in
            the agent's system prompt.
        max_steps (`int`, default `20`): Hard limit on the number of reasoning
            steps per run.
        name (`str`, *optional*): Name for the agent (required when the agent
            is used as a managed agent inside another agent).
        description (`str`, *optional*): Human-readable description of the
            agent's purpose.
        planning_interval (`int`, *optional*): Number of steps between
            re-planning passes.  ``None`` disables planning.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Log verbosity.
        add_base_tools (`bool`, default `False`): Whether to add default base
            tools to the agent's toolbox.
        additional_kwargs (`dict`, default `{}`): Extra keyword arguments
            forwarded verbatim to the agent constructor (e.g.
            ``executor_type``, ``additional_authorized_imports``, …).

    Example:
        ```python
        from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool
        from smolagents.pipeline import AgentConfig, AgentFactory

        cfg = AgentConfig(
            agent_class=CodeAgent,
            model=InferenceClientModel(),
            tools=[DuckDuckGoSearchTool()],
            max_steps=5,
            instructions="Always cite your sources.",
        )
        agent = AgentFactory().create_from_config(cfg)
        ```
    """

    agent_class: type
    model: Any  # Model instance (avoid circular import with forward ref)
    tools: list[Tool] = field(default_factory=list)
    instructions: str | None = None
    max_steps: int = 20
    name: str | None = None
    description: str | None = None
    planning_interval: int | None = None
    verbosity_level: LogLevel = LogLevel.INFO
    add_base_tools: bool = False
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is invalid.

        Checks:
        * ``agent_class`` must be a subclass of
          :class:`~smolagents.agents.MultiStepAgent`.
        * ``max_steps`` must be a positive integer.
        * ``planning_interval``, when set, must be a positive integer.
        """
        from .agents import MultiStepAgent

        if not (isinstance(self.agent_class, type) and issubclass(self.agent_class, MultiStepAgent)):
            raise ValueError(
                f"'agent_class' must be a subclass of MultiStepAgent, got {self.agent_class!r}."
            )
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError(f"'max_steps' must be a positive integer, got {self.max_steps!r}.")
        if self.planning_interval is not None and (
            not isinstance(self.planning_interval, int) or self.planning_interval < 1
        ):
            raise ValueError(
                f"'planning_interval' must be a positive integer or None, got {self.planning_interval!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation (model and tools excluded).

        The model and tool instances are not serialised here; only the
        structural/metadata fields are included so the config can be logged or
        stored as lightweight metadata.
        """
        return {
            "agent_class": self.agent_class.__name__,
            "max_steps": self.max_steps,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "planning_interval": self.planning_interval,
            "verbosity_level": int(self.verbosity_level),
            "add_base_tools": self.add_base_tools,
            "additional_kwargs": self.additional_kwargs,
        }


# ---------------------------------------------------------------------------
# AgentFactory – creates and caches agents from configs
# ---------------------------------------------------------------------------


class AgentFactory:
    """Creates and optionally caches named agent instances from :class:`AgentConfig` objects.

    The factory maintains an internal registry that maps string names to
    :class:`AgentConfig` instances.  You can register configs by name, then
    instantiate them on demand (optionally with per-call overrides).

    Args:
        validate_on_register (`bool`, default `True`): Whether to call
            :meth:`AgentConfig.validate` every time a config is registered.

    Example:
        ```python
        from smolagents import CodeAgent, InferenceClientModel
        from smolagents.pipeline import AgentConfig, AgentFactory

        factory = AgentFactory()
        factory.register(
            "writer",
            AgentConfig(
                agent_class=CodeAgent,
                model=InferenceClientModel(),
                instructions="Write clear, concise prose.",
            ),
        )
        writer = factory.create("writer")
        writer.run("Write a haiku about the ocean.")
        ```
    """

    def __init__(self, validate_on_register: bool = True) -> None:
        self._registry: dict[str, AgentConfig] = {}
        self._validate_on_register = validate_on_register

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register(self, name: str, config: AgentConfig) -> "AgentFactory":
        """Register a named :class:`AgentConfig`.

        Args:
            name (`str`): Unique name for this configuration.
            config (`AgentConfig`): The agent configuration to register.

        Returns:
            `AgentFactory`: ``self``, to allow chaining.

        Raises:
            ValueError: If ``validate_on_register`` is ``True`` and the config
                is invalid, or if ``name`` is already registered.
        """
        if name in self._registry:
            raise ValueError(
                f"An agent config named '{name}' is already registered. "
                "Use update() to replace it, or choose a different name."
            )
        if self._validate_on_register:
            config.validate()
        self._registry[name] = config
        logger.debug("AgentFactory: registered config '%s' (%s).", name, config.agent_class.__name__)
        return self

    def update(self, name: str, config: AgentConfig) -> "AgentFactory":
        """Replace (or add) a named :class:`AgentConfig`.

        Unlike :meth:`register`, this does not raise an error if ``name`` is
        already in the registry.

        Args:
            name (`str`): Name of the configuration to replace or add.
            config (`AgentConfig`): The new configuration.

        Returns:
            `AgentFactory`: ``self``, to allow chaining.
        """
        if self._validate_on_register:
            config.validate()
        self._registry[name] = config
        return self

    def unregister(self, name: str) -> None:
        """Remove a named config from the registry.

        Args:
            name (`str`): Name to remove.

        Raises:
            KeyError: If ``name`` is not found.
        """
        if name not in self._registry:
            raise KeyError(f"No agent config named '{name}' is registered.")
        del self._registry[name]

    @property
    def registered_names(self) -> list[str]:
        """Sorted list of registered config names."""
        return sorted(self._registry.keys())

    def get_config(self, name: str) -> AgentConfig:
        """Return the :class:`AgentConfig` registered under *name*.

        Args:
            name (`str`): Registered config name.

        Raises:
            KeyError: If *name* is not found.
        """
        if name not in self._registry:
            raise KeyError(
                f"No agent config named '{name}'. "
                f"Registered names: {self.registered_names}"
            )
        return self._registry[name]

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def create(self, name: str, **override_kwargs: Any) -> Any:
        """Instantiate the agent registered under *name*.

        Args:
            name (`str`): Registered config name.
            **override_kwargs: Any keyword argument accepted by the agent
                constructor that should override the registered config values.
                Common overrides: ``max_steps``, ``verbosity_level``,
                ``instructions``.

        Returns:
            A new agent instance (subclass of
            :class:`~smolagents.agents.MultiStepAgent`).
        """
        config = self.get_config(name)
        return self.create_from_config(config, **override_kwargs)

    def create_from_config(self, config: AgentConfig, **override_kwargs: Any) -> Any:
        """Instantiate an agent directly from an :class:`AgentConfig`.

        The config is never mutated; *override_kwargs* are applied on top of
        the config values for this call only.

        Args:
            config (`AgentConfig`): The configuration to instantiate.
            **override_kwargs: Keyword arguments that override config fields
                for this instantiation only.

        Returns:
            A new agent instance.
        """
        config.validate()
        build_kwargs: dict[str, Any] = {
            "tools": list(config.tools),
            "model": config.model,
            "max_steps": config.max_steps,
            "verbosity_level": config.verbosity_level,
            "add_base_tools": config.add_base_tools,
            **config.additional_kwargs,
        }
        # Only forward optional fields when they are set to avoid overriding
        # agent-class defaults.
        for attr in ("instructions", "name", "description", "planning_interval"):
            value = getattr(config, attr)
            if value is not None:
                build_kwargs[attr] = value

        build_kwargs.update(override_kwargs)
        return config.agent_class(**build_kwargs)

    def __repr__(self) -> str:
        names = ", ".join(f"'{n}'" for n in self.registered_names) or "<empty>"
        return f"AgentFactory(registered=[{names}])"


# ---------------------------------------------------------------------------
# PipelineStep – single step in an AgentPipeline
# ---------------------------------------------------------------------------


class PipelineStep:
    """Wraps an agent as a named step inside an :class:`AgentPipeline`.

    Each step may carry optional *transform* callables that pre-process the
    incoming task string (from the previous step's output) and/or
    post-process the agent's result before it is forwarded to the next step.

    Args:
        agent: The agent that executes this step.
        name (`str`, *optional*): Human-readable label for logging / repr.
            Defaults to the agent's class name.
        input_transform (`Callable[[Any], str]`, *optional*): Converts the
            previous step's output into the task string for *this* step.
            Defaults to ``str``.
        output_transform (`Callable[[Any], Any]`, *optional*): Transforms
            this step's raw output before passing it to the next step.
            Defaults to the identity function.
        images (`list[PIL.Image.Image]`, *optional*): Images to pass to the
            agent for this step.  When ``None``, the pipeline forwards images
            from the initial :meth:`AgentPipeline.run` call only on the first
            step; later steps receive no images unless explicitly set here.

    Example:
        ```python
        from smolagents.pipeline import PipelineStep, AgentPipeline

        step1 = PipelineStep(research_agent, name="research")
        step2 = PipelineStep(
            summary_agent,
            name="summarise",
            input_transform=lambda text: f"Summarise: {text}",
        )
        pipeline = AgentPipeline([step1, step2])
        ```
    """

    def __init__(
        self,
        agent: Any,
        name: str | None = None,
        input_transform: Callable[[Any], str] | None = None,
        output_transform: Callable[[Any], Any] | None = None,
        images: list["PIL.Image.Image"] | None = None,
    ) -> None:
        self.agent = agent
        self.name: str = name or agent.__class__.__name__
        self.input_transform: Callable[[Any], str] = input_transform or str
        self.output_transform: Callable[[Any], Any] = output_transform or (lambda x: x)
        self.images: list["PIL.Image.Image"] | None = images

    def __repr__(self) -> str:
        return f"PipelineStep(name={self.name!r}, agent={self.agent.__class__.__name__})"


# ---------------------------------------------------------------------------
# AgentPipeline – linear multi-agent pipeline
# ---------------------------------------------------------------------------


class AgentPipeline:
    """Connects agents in a linear sequence (a *pipeline*).

    The pipeline runs each agent in order; the output of step *i* is
    converted to a task string and fed as input to step *i + 1*.  Any agent
    or :class:`PipelineStep` can be used as a stage.

    Pipelines are composable:

    * Use ``AgentPipeline([a, b, c])`` to construct from a list.
    * Use the ``|`` operator: ``a | b | c`` to build a pipeline step-by-step.
      The operator works with plain agent instances and :class:`PipelineStep`
      objects, and even with another :class:`AgentPipeline` (which is flattened
      in).

    Args:
        steps (`list`): An ordered list of agents and/or
            :class:`PipelineStep` objects.  Plain agents are automatically
            wrapped in a :class:`PipelineStep`.
        name (`str`, *optional*): Optional human-readable name for this pipeline.

    Example:
        ```python
        from smolagents import CodeAgent, InferenceClientModel, WebSearchTool
        from smolagents.pipeline import AgentPipeline, PipelineStep

        model = InferenceClientModel()

        researcher = CodeAgent(tools=[WebSearchTool()], model=model)
        summariser = CodeAgent(tools=[], model=model)
        writer      = CodeAgent(tools=[], model=model)

        # Explicit construction
        pipeline = AgentPipeline([
            PipelineStep(researcher, name="research"),
            PipelineStep(
                summariser,
                name="summarise",
                input_transform=lambda text: f"Condense this research:\\n{text}",
            ),
            PipelineStep(
                writer,
                name="write",
                input_transform=lambda summary: f"Write a blog post based on:\\n{summary}",
            ),
        ])

        result = pipeline.run("Latest advances in fusion energy")
        print(result)

        # Or use the pipe operator
        pipeline2 = researcher | summariser | writer
        result2 = pipeline2.run("Latest advances in fusion energy")
        ```
    """

    def __init__(self, steps: list[Any], name: str | None = None) -> None:
        if not steps:
            raise ValueError("AgentPipeline requires at least one step.")
        self.steps: list[PipelineStep] = [self._to_step(s) for s in steps]
        self.name: str = name or f"AgentPipeline({len(self.steps)} steps)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_step(obj: Any) -> PipelineStep:
        """Normalise *obj* to a :class:`PipelineStep`."""
        if isinstance(obj, PipelineStep):
            return obj
        # Assume it's an agent instance
        return PipelineStep(obj)

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __or__(self, other: Any) -> "AgentPipeline":
        """Return a new pipeline with *other* appended.

        ``other`` may be:

        * A plain agent instance.
        * A :class:`PipelineStep`.
        * Another :class:`AgentPipeline` (its steps are flattened in).
        """
        if isinstance(other, AgentPipeline):
            return AgentPipeline(self.steps + other.steps, name=self.name)
        return AgentPipeline(self.steps + [self._to_step(other)], name=self.name)

    def __ror__(self, other: Any) -> "AgentPipeline":
        """Support ``agent | pipeline`` syntax."""
        if isinstance(other, AgentPipeline):
            return AgentPipeline(other.steps + self.steps, name=self.name)
        return AgentPipeline([self._to_step(other)] + self.steps, name=self.name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        images: list["PIL.Image.Image"] | None = None,
        reset: bool = True,
        additional_args: dict[str, Any] | None = None,
    ) -> Any:
        """Run the pipeline end-to-end.

        The *task* string is given to the first step.  Each subsequent step
        receives the previous step's output (converted to a string via the
        step's ``input_transform``).

        Images are passed to the **first** step only by default.  If a
        :class:`PipelineStep` has its own ``images`` attribute set, those are
        used for that step regardless.

        Args:
            task (`str`): The initial task description for the first agent.
            images (`list[PIL.Image.Image]`, *optional*): Images to pass to
                the first step (or any step that has ``images=None``).
            reset (`bool`, default `True`): Whether to reset each agent's
                memory before running it.
            additional_args (`dict`, *optional*): Extra key-value pairs
                forwarded to **every** agent's ``run()`` call via its
                ``additional_args`` parameter.

        Returns:
            The output of the final pipeline step.

        Raises:
            RuntimeError: If any step raises an unhandled exception.  The
                error message includes the step name and index.
        """
        usage_tracker = get_usage_tracker()
        current_input: Any = task
        current_images = images  # images flow only into the first step by default

        logger.info("AgentPipeline '%s': starting run with %d step(s).", self.name, len(self.steps))

        for step_idx, step in enumerate(self.steps):
            # Determine task string for this step
            if step_idx == 0:
                step_task = str(current_input)
            else:
                try:
                    step_task = step.input_transform(current_input)
                except Exception as exc:
                    raise RuntimeError(
                        f"AgentPipeline step {step_idx} ('{step.name}'): "
                        f"input_transform raised an error: {exc}"
                    ) from exc

            # Determine images for this step
            step_images = step.images if step.images is not None else (current_images if step_idx == 0 else None)

            logger.info(
                "AgentPipeline '%s': running step %d/%d — '%s'.",
                self.name,
                step_idx + 1,
                len(self.steps),
                step.name,
            )

            run_kwargs: dict[str, Any] = {"reset": reset}
            if step_images:
                run_kwargs["images"] = step_images
            if additional_args:
                run_kwargs["additional_args"] = additional_args

            try:
                raw_output = step.agent.run(step_task, **run_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"AgentPipeline step {step_idx} ('{step.name}') failed: {exc}"
                ) from exc

            # Apply output transform
            try:
                current_input = step.output_transform(raw_output)
            except Exception as exc:
                raise RuntimeError(
                    f"AgentPipeline step {step_idx} ('{step.name}'): "
                    f"output_transform raised an error: {exc}"
                ) from exc

        logger.info("AgentPipeline '%s': run complete.", self.name)
        return current_input

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        """Number of steps in the pipeline."""
        return len(self.steps)

    def describe(self) -> str:
        """Return a human-readable description of the pipeline structure.

        Example output::

            AgentPipeline 'my-pipeline' (3 steps)
              [0] research   — CodeAgent
              [1] summarise  — CodeAgent
              [2] write      — VLMCodeAgent
        """
        lines = [f"AgentPipeline '{self.name}' ({self.num_steps} steps)"]
        for i, step in enumerate(self.steps):
            lines.append(f"  [{i}] {step.name:<15} — {step.agent.__class__.__name__}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        step_repr = " | ".join(f"'{s.name}'" for s in self.steps)
        return f"AgentPipeline([{step_repr}])"


def _patch_agent_pipe_operator() -> None:
    """Add ``__or__`` to :class:`~smolagents.agents.MultiStepAgent` so that
    ``agent_a | agent_b`` returns an :class:`AgentPipeline`.

    This is called once at module import time.
    """
    from .agents import MultiStepAgent

    # Use __dict__ check so we don't mistake ABCMeta.__or__ (Python 3.10
    # type-union syntax on classes) for a pre-existing instance __or__.
    if "__or__" in MultiStepAgent.__dict__:
        return  # Already patched

    def _agent_or(self: Any, other: Any) -> "AgentPipeline":
        lhs = PipelineStep(self)
        if isinstance(other, AgentPipeline):
            return AgentPipeline([lhs] + other.steps)
        return AgentPipeline([lhs, AgentPipeline._to_step(other)])

    MultiStepAgent.__or__ = _agent_or  # type: ignore[attr-defined]


_patch_agent_pipe_operator()
