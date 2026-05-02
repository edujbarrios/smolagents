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
import json
import threading
from dataclasses import dataclass, field
from enum import IntEnum

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from smolagents.utils import sanitize_for_rich


__all__ = ["AgentLogger", "LogLevel", "Monitor", "TokenUsage", "Timing", "UsageTracker", "get_usage_tracker"]


@dataclass
class TokenUsage:
    """
    Contains the token usage information for a given step or run.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int = field(init=False)

    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens

    def dict(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Timing:
    """
    Contains the timing information for a given step or run.
    """

    start_time: float
    end_time: float | None = None

    @property
    def duration(self):
        return None if self.end_time is None else self.end_time - self.start_time

    def dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return f"Timing(start_time={self.start_time}, end_time={self.end_time}, duration={self.duration})"


class Monitor:
    def __init__(self, tracked_model, logger):
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def get_total_token_counts(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.total_input_token_count,
            output_tokens=self.total_output_token_count,
        )

    def reset(self):
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log):
        """Update the metrics of the monitor.

        Args:
            step_log ([`MemoryStep`]): Step log to update the monitor with.
        """
        step_duration = step_log.timing.duration
        self.step_durations.append(step_duration)
        console_outputs = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"

        if step_log.token_usage is not None:
            self.total_input_token_count += step_log.token_usage.input_tokens
            self.total_output_token_count += step_log.token_usage.output_tokens
            console_outputs += (
                f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        self.logger.log(Text(console_outputs, style="dim"), level=1)


class LogLevel(IntEnum):
    OFF = -1  # No output
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


class UsageTracker:
    """Thread-safe singleton that tracks smolagents project-level usage across all agent runs.

    Collects statistics about which models and tools are used, how many tokens are consumed,
    and how many agent runs (including VLM runs) have been performed. Useful for understanding
    how a project is using smolagents in production or development.

    Use :func:`get_usage_tracker` to obtain the shared singleton instance.

    Example:
        ```python
        from smolagents import CodeAgent, InferenceClientModel
        from smolagents.monitoring import get_usage_tracker

        model = InferenceClientModel()
        agent = CodeAgent(tools=[], model=model)
        agent.run("What is 2+2?")

        tracker = get_usage_tracker()
        print(tracker.get_summary())
        ```
    """

    _instance: "UsageTracker | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._data_lock = threading.Lock()
        self.run_count: int = 0
        self.vlm_run_count: int = 0
        self.model_invocations: dict[str, int] = {}
        self.tool_invocations: dict[str, int] = {}
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    @classmethod
    def get_instance(cls) -> "UsageTracker":
        """Return the global singleton :class:`UsageTracker` instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_run(
        self,
        model_id: str | None = None,
        is_vlm: bool = False,
        token_usage: "TokenUsage | None" = None,
    ) -> None:
        """Record the completion of one agent run.

        Args:
            model_id (`str`, *optional*): Identifier of the model used for the run.
            is_vlm (`bool`, default `False`): Whether the run was performed by a VLM agent.
            token_usage ([`TokenUsage`], *optional*): Token counts consumed during the run.
        """
        with self._data_lock:
            self.run_count += 1
            if is_vlm:
                self.vlm_run_count += 1
            if model_id:
                self.model_invocations[model_id] = self.model_invocations.get(model_id, 0) + 1
            if token_usage is not None:
                self._total_input_tokens += token_usage.input_tokens
                self._total_output_tokens += token_usage.output_tokens

    def record_tool_call(self, tool_name: str) -> None:
        """Record a single tool invocation.

        Args:
            tool_name (`str`): Name of the tool that was called.
        """
        with self._data_lock:
            self.tool_invocations[tool_name] = self.tool_invocations.get(tool_name, 0) + 1

    @property
    def total_token_usage(self) -> "TokenUsage":
        """Cumulative token usage across all recorded runs."""
        with self._data_lock:
            return TokenUsage(
                input_tokens=self._total_input_tokens,
                output_tokens=self._total_output_tokens,
            )

    def get_summary(self) -> dict:
        """Return a snapshot of the current usage statistics.

        Returns:
            `dict`: Dictionary containing `run_count`, `vlm_run_count`,
            `model_invocations`, `tool_invocations`, and `total_token_usage`.
        """
        with self._data_lock:
            return {
                "run_count": self.run_count,
                "vlm_run_count": self.vlm_run_count,
                "model_invocations": dict(self.model_invocations),
                "tool_invocations": dict(self.tool_invocations),
                "total_token_usage": {
                    "input_tokens": self._total_input_tokens,
                    "output_tokens": self._total_output_tokens,
                    "total_tokens": self._total_input_tokens + self._total_output_tokens,
                },
            }

    def reset(self) -> None:
        """Reset all tracked statistics to zero."""
        with self._data_lock:
            self.run_count = 0
            self.vlm_run_count = 0
            self.model_invocations = {}
            self.tool_invocations = {}
            self._total_input_tokens = 0
            self._total_output_tokens = 0


def get_usage_tracker() -> UsageTracker:
    """Return the global singleton :class:`UsageTracker` instance.

    This tracker records project-wide usage of smolagents: models used, tools called,
    token consumption, and agent run counts.  It is updated automatically when agents
    run.

    Example:
        ```python
        from smolagents.monitoring import get_usage_tracker

        tracker = get_usage_tracker()
        print(tracker.get_summary())
        tracker.reset()
        ```
    """
    return UsageTracker.get_instance()


YELLOW_HEX = "#d4b702"


class AgentLogger:
    def __init__(self, level: LogLevel = LogLevel.INFO, console: Console | None = None):
        self.level = level
        if console is None:
            self.console = Console(highlight=False)
        else:
            self.console = console

    def log(self, *args, level: int | str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log_error(self, error_message: str) -> None:
        self.log(Text(sanitize_for_rich(error_message), style="bold red"), level=LogLevel.ERROR)

    def log_markdown(self, content: str, title: str | None = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            self.log(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.log(markdown_content, level=level)

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        self.log(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        self.log(
            Rule(
                "[bold white]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=LogLevel.INFO,
        )

    def log_task(self, content: str, subtitle: str, title: str | None = None, level: LogLevel = LogLevel.INFO) -> None:
        # Important: `content` can contain arbitrary tool logs / payloads. If we embed it
        # inside Rich markup (e.g. f"[bold]{content}"), any stray "[/...]" sequences or
        # binary-ish characters can crash Rich's markup parser. Render the content as
        # `Text` instead, and apply styling via Text/style, not markup.
        safe_content = sanitize_for_rich(content)
        safe_subtitle = sanitize_for_rich(subtitle)
        content_text = Text("\n") + Text(safe_content, style="bold") + Text("\n")
        subtitle_text = Text(safe_subtitle)
        self.log(
            Panel(
                content_text,
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle_text,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )

    def log_messages(self, messages: list[dict], level: LogLevel = LogLevel.DEBUG) -> None:
        messages_as_string = "\n".join([json.dumps(message.dict(), indent=4) for message in messages])
        self.log(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            ),
            level=level,
        )

    def visualize_agent_tree(self, agent):
        def create_tools_section(tools_dict):
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")
            table.add_column("Description")
            table.add_column("Arguments")

            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))

            return Group("🛠️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: str | None = None):
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            """Recursively builds the agent tree."""
            parent_tree.add(create_tools_section(agent_obj.tools))

            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("🤖 [italic #1E90FF]Managed agents:")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_tree = agents_branch.add(get_agent_headline(managed_agent, name))
                    if managed_agent.__class__.__name__ == "CodeAgent":
                        agent_tree.add(
                            f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {managed_agent.additional_authorized_imports}"
                        )
                    agent_tree.add(f"📝 [italic #1E90FF]Description:[/italic #1E90FF] {managed_agent.description}")
                    build_agent_tree(agent_tree, managed_agent)

        main_tree = Tree(get_agent_headline(agent))
        if agent.__class__.__name__ == "CodeAgent":
            main_tree.add(
                f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {agent.additional_authorized_imports}"
            )
        build_agent_tree(main_tree, agent)
        self.console.print(main_tree)
