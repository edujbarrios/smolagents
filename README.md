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

> **Note:** `pip install 'smolagents[openai]'` (or `pip install openai`) is required for `OpenAIModel`.

For general documentation on `smolagents`, see the [upstream docs](https://huggingface.co/docs/smolagents/index).

---

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
    model_id="gpt-4o",          # any vision-capable model served by your provider
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
```

---

## Fork Add-ons

The following features extend the upstream `smolagents` library with VLM capabilities.

---

### 🖼️ VLMCodeAgent — Vision-Language Code Agent

`VLMCodeAgent` is a `CodeAgent` subclass pre-configured for multimodal (image + text) workflows.
It automatically adds vision-oriented instructions to the system prompt and flags the run as a VLM
run in the [`UsageTracker`](#-usagetracker--usage-statistics).

#### Detect objects in an image

```python
import os
import PIL.Image
from smolagents import OpenAIModel, ImageAnalysisTool, VLMCodeAgent

model = OpenAIModel(
    model_id="gpt-4o",
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
agent = VLMCodeAgent(
    tools=[ImageAnalysisTool(model=model)],
    model=model,
)
# Load a real image or create a simple test image to try it out
image = PIL.Image.open("street_scene.jpg") if os.path.exists("street_scene.jpg") \
    else PIL.Image.new("RGB", (128, 128), color=(200, 100, 50))
result = agent.run("Detect and list all objects visible in this image.", images=[image])
print(result)
# e.g. "Detected objects: car, traffic light, pedestrian, bicycle, tree, building"
```

#### Analyse an image with a HuggingFace model

```python
import os
import PIL.Image
from smolagents import InferenceClientModel, ImageAnalysisTool, VLMCodeAgent

model = InferenceClientModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
agent = VLMCodeAgent(
    tools=[ImageAnalysisTool(model=model)],
    model=model,
)
# Load a real image or create a simple test image to try it out
image = PIL.Image.open("photo.jpg") if os.path.exists("photo.jpg") \
    else PIL.Image.new("RGB", (128, 128), color=(50, 150, 200))
result = agent.run("Describe the scene and identify the main subjects.", images=[image])
print(result)
```

---

### 🔍 ImageAnalysisTool — Dedicated Image Q&A Tool

`ImageAnalysisTool` wraps a vision-capable model as a smolagents `Tool` so that any agent
can ask targeted questions about images — object detection, scene description, OCR, color
analysis, and more.

```python
import os
import PIL.Image
from smolagents import OpenAIModel, ImageAnalysisTool, VLMCodeAgent

model = OpenAIModel(
    model_id="gpt-4o",
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
analysis_tool = ImageAnalysisTool(model=model)
agent = VLMCodeAgent(tools=[analysis_tool], model=model)

# Load a real image or create a simple test image to try it out
image = PIL.Image.open("receipt.jpg") if os.path.exists("receipt.jpg") \
    else PIL.Image.new("RGB", (128, 128), color=(240, 230, 210))
result = agent.run("Read all text visible in this image and extract the total amount.", images=[image])
print(result)
# e.g. "Total amount: $47.83"
```

---

### 🗂️ VLM Agent Template (Jinja2)

A ready-to-use **Jinja2 system-prompt template** for VLM agents lives at
[`src/smolagents/prompts/vlm_agent.yaml`](src/smolagents/prompts/vlm_agent.yaml).
It ships with five worked examples covering the most common VLM tasks:

| # | Example task |
|---|---|
| 1 | Detect all objects and list them |
| 2 | Count people and describe their activities |
| 3 | Extract and OCR visible text / headlines |
| 4 | Identify dominant color per image quadrant |
| 5 | Describe scene and classify as indoor / outdoor / urban / nature |

Pass the template path to `VLMCodeAgent` (or any `CodeAgent`) via the `prompt_templates` argument:

```python
import os
import PIL.Image
from smolagents import OpenAIModel, ImageAnalysisTool, VLMCodeAgent
from smolagents.utils import load_prompt_templates

model = OpenAIModel(
    model_id="gpt-4o",
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
agent = VLMCodeAgent(
    tools=[ImageAnalysisTool(model=model)],
    model=model,
    prompt_templates=load_prompt_templates("src/smolagents/prompts/vlm_agent.yaml"),
)

# Load a real image or create a simple test image to try it out
image = PIL.Image.open("cityscape.jpg") if os.path.exists("cityscape.jpg") \
    else PIL.Image.new("RGB", (128, 128), color=(100, 150, 200))
result = agent.run(
    "Identify every object in the foreground and background of this cityscape photo.",
    images=[image],
)
print(result)
```

---

### 📊 UsageTracker — Usage Statistics

`UsageTracker` is a thread-safe singleton that records project-wide statistics across every agent
run: models used, tools called, token counts, and VLM-vs-text run breakdown.  
It is updated **automatically** — no extra code needed in your agents.

#### Object detection with HuggingFace model + usage tracking

```python
import os
import PIL.Image
from smolagents import ImageAnalysisTool, InferenceClientModel, VLMCodeAgent
from smolagents.monitoring import get_usage_tracker

model = InferenceClientModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
agent = VLMCodeAgent(tools=[ImageAnalysisTool(model=model)], model=model)

# Load a real image or create a simple test image to try it out
image = PIL.Image.open("market_scene.jpg") if os.path.exists("market_scene.jpg") \
    else PIL.Image.new("RGB", (128, 128), color=(180, 220, 130))
agent.run("List every item you can see for sale in this market photo.", images=[image])

tracker = get_usage_tracker()
print(tracker.get_summary())
# {'run_count': 1, 'vlm_run_count': 1,
#  'model_invocations': {'meta-llama/Llama-3.2-11B-Vision-Instruct': 1},
#  'tool_invocations': {'image_analysis': 1}, 'total_token_usage': {...}}

tracker.reset()  # Clear all counters
```

#### Image analysis with external OpenAI-compatible API + usage tracking

```python
import os
import PIL.Image
from smolagents import ImageAnalysisTool, OpenAIModel, VLMCodeAgent
from smolagents.monitoring import get_usage_tracker

model = OpenAIModel(
    model_id="gpt-4o",
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
agent = VLMCodeAgent(tools=[ImageAnalysisTool(model=model)], model=model)

# Load a real image or create a simple test image to try it out
image = PIL.Image.open("dashboard.png") if os.path.exists("dashboard.png") \
    else PIL.Image.new("RGB", (128, 128), color=(60, 80, 120))
agent.run("Analyse this dashboard screenshot and summarise the key metrics shown.", images=[image])

tracker = get_usage_tracker()
print(tracker.get_summary())
# {'run_count': 1, 'vlm_run_count': 1, 'model_invocations': {'gpt-4o': 1},
#  'tool_invocations': {'image_analysis': 1}, 'total_token_usage': {...}}

tracker.reset()
```

> **Parametrized example:** [`examples/vlm_usage_tracking/`](examples/vlm_usage_tracking/)
> contains a ready-to-run script that runs both backends back-to-back.
> All model IDs, API keys, and agent settings live in a single
> [`config.yaml`](examples/vlm_usage_tracking/config.yaml) — no code edits required.

---

### 🔗 AgentPipeline — Multi-Agent VLM Pipelines

`AgentFactory` + `AgentConfig` give you a parametrizable, registry-based way to define agents and
wire them into reusable pipelines.  Switch models, tools, or instructions by changing the config —
no code restructuring required.

| Class | Purpose |
|---|---|
| `AgentConfig` | Fully parametrized, serialisable agent descriptor |
| `AgentFactory` | Creates (and optionally caches) named agent instances from `AgentConfig` |
| `PipelineStep` | Wraps an agent for use in a pipeline with optional input/output transforms |
| `AgentPipeline` | Connects agents in sequence; each agent's output becomes the next agent's task |

#### Two-step pipeline: detect objects → generate accessibility description

```python
import os
import PIL.Image
from smolagents import VLMCodeAgent, OpenAIModel, InferenceClientModel, ImageAnalysisTool
from smolagents.pipeline import AgentConfig, AgentFactory, AgentPipeline

# Vision-capable model for the detection step
vision_model = OpenAIModel(
    model_id="gpt-4o",
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)
# Lighter model for the description-writing step
text_model = InferenceClientModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")

factory = AgentFactory()

factory.register(
    "object_detector",
    AgentConfig(
        agent_class=VLMCodeAgent,
        model=vision_model,
        tools=[ImageAnalysisTool(model=vision_model)],
        instructions=(
            "You detect objects in images. "
            "Return a structured list of every distinct object you can identify, "
            "including its approximate position (top/bottom, left/right/centre)."
        ),
        max_steps=5,
    ),
)
factory.register(
    "accessibility_writer",
    AgentConfig(
        agent_class=VLMCodeAgent,
        model=text_model,
        tools=[ImageAnalysisTool(model=text_model)],
        instructions=(
            "Given a structured object list, produce a concise alt-text description "
            "of the image suitable for screen-reader users."
        ),
    ),
)

pipeline = AgentPipeline([
    factory.create("object_detector"),
    factory.create("accessibility_writer"),
])

# Load a real image or create a simple test image to try it out
image = PIL.Image.open("photo.jpg") if os.path.exists("photo.jpg") \
    else PIL.Image.new("RGB", (128, 128), color=(80, 120, 200))
result = pipeline.run(
    "Detect all objects in this image, then write an accessibility description.",
    images=[image],
)
print(result)
```

Agents can also be chained with the `|` operator:

```python
detector = factory.create("object_detector")
writer   = factory.create("accessibility_writer")

pipeline = detector | writer
result = pipeline.run("Detect objects and write an accessibility description.", images=[image])
```

#### Three-step pipeline: analyse medical image → flag anomalies → write report

```python
import os
import PIL.Image
from smolagents import VLMCodeAgent, OpenAIModel, ImageAnalysisTool
from smolagents.pipeline import AgentConfig, AgentFactory, AgentPipeline

model = OpenAIModel(
    model_id="gpt-4o",
    api_base="https://llm7.io/v1",
    api_key="YOUR_API_KEY",
)

factory = AgentFactory()

factory.register(
    "image_analyser",
    AgentConfig(
        agent_class=VLMCodeAgent,
        model=model,
        tools=[ImageAnalysisTool(model=model)],
        instructions="Describe every visual feature present in this medical scan in clinical detail.",
        max_steps=5,
    ),
)
factory.register(
    "anomaly_detector",
    AgentConfig(
        agent_class=VLMCodeAgent,
        model=model,
        tools=[ImageAnalysisTool(model=model)],
        instructions=(
            "Given a detailed image description, list any features that appear abnormal "
            "or warrant further investigation."
        ),
    ),
)
factory.register(
    "report_writer",
    AgentConfig(
        agent_class=VLMCodeAgent,
        model=model,
        tools=[ImageAnalysisTool(model=model)],
        instructions=(
            "Given an image description and a list of anomalies, produce a concise "
            "radiology-style report with Findings and Impression sections."
        ),
    ),
)

pipeline = AgentPipeline([
    factory.create("image_analyser"),
    factory.create("anomaly_detector"),
    factory.create("report_writer"),
])

# Load a real image or create a simple test image to try it out
scan = PIL.Image.open("chest_xray.jpg") if os.path.exists("chest_xray.jpg") \
    else PIL.Image.new("L", (128, 128), color=180)
report = pipeline.run("Analyse this chest X-ray and write a radiology report.", images=[scan])
print(report)
```

> **Tip:** Any number of VLM agents from any mix of providers can be chained in a single
> `AgentPipeline` — there is no limit on the number of steps or the combination of backends.

---

## License

Licensed under the [Apache License 2.0](LICENSE).  
Original library by the HuggingFace Team.
