# smolagents — VLM Fork

This is a fork of [huggingface/smolagents](https://github.com/huggingface/smolagents) that extends it with
**vision-language (VLM) task support**, parametrizable multi-agent pipelines, and first-class
integration with any **OpenAI-compatible API** (e.g. [LLM7.io](https://llm7.io), OpenAI, Groq, Together AI, local vLLM, etc.).

Install directly from source (not published to PyPI):

```bash
git clone https://github.com/edujbarrios/smolagents.git
cd smolagents
pip install -e ".[dev]"
```

For runtime only (no dev/test tooling):

```bash
pip install -e ".[toolkit]"
```

For general documentation on `smolagents`, see the [upstream docs](https://huggingface.co/docs/smolagents/index).

## Using OpenAI-Compatible APIs

All examples in this fork use `OpenAIModel`, which connects to any OpenAI-compatible endpoint.
Set `api_base` and `api_key` to point at your preferred provider:

| Provider | `api_base` |
|---|---|
| [LLM7.io](https://llm7.io) | `https://llm7.io/v1` |
| OpenAI | `https://api.openai.com/v1` (default) |
| Groq | `https://api.groq.com/openai/v1` |
| Together AI | `https://api.together.xyz/v1` |
| Local vLLM | `http://localhost:8000/v1` |

```python
from smolagents import OpenAIModel

model = OpenAIModel(
    model_id="meta-llama/Llama-3-8b-chat-hf",  # any model served by your provider
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
```

> **Note:** `pip install 'smolagents[openai]'` (or `pip install openai`) is required for `OpenAIModel`.

## Fork Add-ons

The following features extend the upstream `smolagents` library.

---

### 🖼️ VLMCodeAgent — Vision-Language Code Agent

`VLMCodeAgent` is a `CodeAgent` subclass pre-configured for multimodal (image + text) workflows.
It automatically adds vision-oriented instructions to the system prompt and flags the run as a VLM
run in the [`UsageTracker`](#-usagetracker--usage-statistics).

```python
import PIL.Image
from smolagents import OpenAIModel, ImageAnalysisTool, VLMCodeAgent

# Works with any OpenAI-compatible endpoint: LLM7.io, Groq, Together AI, local vLLM, etc.
model = OpenAIModel(
    model_id="gpt-4o",          # replace with any vision-capable model available on your provider
    api_base="https://llm7.io/v1",  # or any OpenAI-compatible base URL
    api_key="YOUR_API_KEY",
)
agent = VLMCodeAgent(
    tools=[ImageAnalysisTool(model=model)],
    model=model,
)
image = PIL.Image.open("photo.jpg")
result = agent.run("What is shown in the image?", images=[image])
```

---

### 🔍 ImageAnalysisTool — Dedicated Image Q&A Tool

`ImageAnalysisTool` wraps a vision-capable model as a smolagents `Tool` so that any agent
(not only `VLMCodeAgent`) can ask targeted questions about images.

```python
from smolagents import OpenAIModel, ImageAnalysisTool, CodeAgent

model = OpenAIModel(
    model_id="gpt-4o",
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
analysis_tool = ImageAnalysisTool(model=model)

agent = CodeAgent(tools=[analysis_tool], model=model)
```

---

### 📊 UsageTracker — Usage Statistics

`UsageTracker` is a thread-safe singleton that records project-wide statistics across every agent
run: models used, tools called, token counts, and VLM-vs-text run breakdown.  
It is updated **automatically** — no extra code needed in your agents.

#### VLM example — HuggingFace model

```python
import PIL.Image
from smolagents import ImageAnalysisTool, InferenceClientModel, VLMCodeAgent
from smolagents.monitoring import get_usage_tracker

model = InferenceClientModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
agent = VLMCodeAgent(tools=[ImageAnalysisTool(model=model)], model=model)

image = PIL.Image.open("photo.jpg")
agent.run("Describe what you see in this image in detail.", images=[image])

tracker = get_usage_tracker()
print(tracker.get_summary())
# {'run_count': 1, 'vlm_run_count': 1, 'model_invocations': {'meta-llama/Llama-3.2-11B-Vision-Instruct': 1},
#  'tool_invocations': {'image_analysis': 1}, 'total_token_usage': {...}}

tracker.reset()  # Clear all counters
```

#### VLM example — external OpenAI-compatible API

```python
import PIL.Image
from smolagents import ImageAnalysisTool, OpenAIModel, VLMCodeAgent
from smolagents.monitoring import get_usage_tracker

model = OpenAIModel(
    model_id="gpt-4o",
    api_base="https://llm7.io/v1",  # or any OpenAI-compatible URL
    api_key="YOUR_API_KEY",
)
agent = VLMCodeAgent(tools=[ImageAnalysisTool(model=model)], model=model)

image = PIL.Image.open("photo.jpg")
agent.run("Describe what you see in this image in detail.", images=[image])

tracker = get_usage_tracker()
print(tracker.get_summary())
# {'run_count': 1, 'vlm_run_count': 1, 'model_invocations': {'gpt-4o': 1},
#  'tool_invocations': {'image_analysis': 1}, 'total_token_usage': {...}}

tracker.reset()  # Clear all counters
```

> **Parametrized example:** [`examples/vlm_usage_tracking/`](examples/vlm_usage_tracking/)
> contains a ready-to-run script that runs both backends back-to-back.
> All model IDs, API keys, and agent settings live in a single
> [`config.yaml`](examples/vlm_usage_tracking/config.yaml) — no code edits required.

---

### 🔗 AgentPipeline — Parametrizable Multi-Agent Pipelines

`AgentFactory` + `AgentConfig` give you a parametrizable, registry-based way to define agents and
specific tasks, then wire them into reusable pipelines.  Switch models, tools, or instructions by
changing the config — no code restructuring required.

Four cooperating primitives let you compose agents into reusable, linearly-chained pipelines:

| Class | Purpose |
|---|---|
| `AgentConfig` | Fully parametrized, serialisable agent descriptor |
| `AgentFactory` | Creates (and optionally caches) named agent instances from `AgentConfig` |
| `PipelineStep` | Wraps an agent for use in a pipeline with optional input/output transforms |
| `AgentPipeline` | Connects agents in sequence; each agent's output becomes the next agent's task |

```python
import PIL.Image
from smolagents import CodeAgent, OpenAIModel, ImageAnalysisTool
from smolagents.pipeline import AgentConfig, AgentFactory, AgentPipeline

# Any OpenAI-compatible provider: LLM7.io, Groq, Together AI, local vLLM, etc.
vision_model = OpenAIModel(
    model_id="gpt-4o",          # vision-capable model
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
text_model = OpenAIModel(
    model_id="gpt-4o-mini",     # lighter model for text tasks
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)

factory = AgentFactory()

factory.register(
    "image_analyser",
    AgentConfig(
        agent_class=CodeAgent,
        model=vision_model,
        tools=[ImageAnalysisTool(model=vision_model)],
        instructions=(
            "You analyse images in detail. "
            "Describe all visible objects, colours, text, spatial relationships, and any notable features."
        ),
        max_steps=5,
    ),
)
factory.register(
    "reporter",
    AgentConfig(
        agent_class=CodeAgent,
        model=text_model,
        instructions=(
            "Given a detailed image analysis, produce a concise, well-structured report "
            "suitable for a non-technical audience."
        ),
    ),
)

pipeline = AgentPipeline([
    factory.create("image_analyser"),
    factory.create("reporter"),
])

image = PIL.Image.open("photo.jpg")
result = pipeline.run("Analyse this image and produce a reader-friendly report.", images=[image])
```

Agents can also be chained with the `|` operator:

```python
pipeline = image_analyser_agent | reporter_agent
result = pipeline.run("…", images=[image])
```

---

### 🤝 Combining HuggingFace and LLM7.io Agents

Because every agent just needs a model object, you can freely mix
`InferenceClientModel` (HuggingFace Inference API) and `OpenAIModel`
(LLM7.io or any OpenAI-compatible endpoint) in the same pipeline.
This lets you, for example, run a heavy research step on a large
HuggingFace-hosted model and then summarise the result with a fast
model served by LLM7.io — or vice-versa.

#### Simple two-agent pipeline (HuggingFace → LLM7.io)

```python
from smolagents import CodeAgent, InferenceClientModel, OpenAIModel
from smolagents.default_tools import DuckDuckGoSearchTool
from smolagents.pipeline import AgentConfig, AgentFactory, AgentPipeline

# Step 1 – research agent backed by a HuggingFace-hosted model
hf_model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")

# Step 2 – summariser backed by a LLM7.io-served model
llm7_model = OpenAIModel(
    model_id="meta-llama/Llama-3-8b-chat-hf",
    api_base="https://llm7.io/v1",
    api_key="YOUR_LLM7_API_KEY",
)

factory = AgentFactory()

factory.register(
    "researcher",
    AgentConfig(
        agent_class=CodeAgent,
        model=hf_model,
        tools=[DuckDuckGoSearchTool()],
        instructions="You research topics thoroughly and return detailed findings.",
        max_steps=10,
    ),
)
factory.register(
    "summariser",
    AgentConfig(
        agent_class=CodeAgent,
        model=llm7_model,
        instructions="Summarise the provided research into three clear bullet points.",
    ),
)

pipeline = AgentPipeline([
    factory.create("researcher"),
    factory.create("summariser"),
])

result = pipeline.run("What are the latest breakthroughs in quantum computing?")
print(result)
```

Or equivalently using the `|` operator:

```python
researcher_agent = factory.create("researcher")
summariser_agent = factory.create("summariser")

pipeline = researcher_agent | summariser_agent
result = pipeline.run("What are the latest breakthroughs in quantum computing?")
```

#### Reversed direction (LLM7.io → HuggingFace)

You can just as easily reverse the order — use a LLM7.io model for
the first step and a HuggingFace model for the second:

```python
from smolagents import CodeAgent, InferenceClientModel, OpenAIModel
from smolagents.pipeline import AgentConfig, AgentFactory, AgentPipeline

llm7_model = OpenAIModel(
    model_id="gpt-4o-mini",
    api_base="https://llm7.io/v1",
    api_key="YOUR_LLM7_API_KEY",
)
hf_model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")

factory = AgentFactory()

factory.register(
    "drafter",
    AgentConfig(
        agent_class=CodeAgent,
        model=llm7_model,
        instructions="Draft a concise outline for the user's topic.",
    ),
)
factory.register(
    "expander",
    AgentConfig(
        agent_class=CodeAgent,
        model=hf_model,
        instructions="Expand the provided outline into a detailed, well-written article.",
        max_steps=15,
    ),
)

pipeline = AgentPipeline([
    factory.create("drafter"),
    factory.create("expander"),
])

result = pipeline.run("Write an article about the history of the internet.")
print(result)
```

> **Tip:** Any number of agents from any mix of providers can be chained
> in a single `AgentPipeline` — there is no limit on the number of steps
> or the combination of backends.

---

## License

Licensed under the [Apache License 2.0](LICENSE).  
Original library by the HuggingFace Team.
