# smolagents — Research Prototype by Eduardo J. Barrios

> **This is a personal research prototype by [Eduardo J. Barrios](https://github.com/edujbarrios), not a community fork.**
> It is based on [huggingface/smolagents](https://github.com/huggingface/smolagents) and extends it with experimental features for research and rapid prototyping.
> It is **not published to PyPI** — install directly from source:
>
> ```bash
> git clone https://github.com/edujbarrios/smolagents.git
> cd smolagents
> pip install -e ".[dev]"
> ```
>
> For runtime only (no dev/test tooling):
>
> ```bash
> pip install -e ".[toolkit]"
> ```

For general documentation on `smolagents`, see the [upstream docs](https://huggingface.co/docs/smolagents/index).

## Fork Add-ons

The following features are **exclusive to this prototype** and are not part of upstream `smolagents`.

---

### 🖼️ VLMCodeAgent — Vision-Language Code Agent

`VLMCodeAgent` is a `CodeAgent` subclass pre-configured for multimodal (image + text) workflows.
It automatically adds vision-oriented instructions to the system prompt and flags the run as a VLM
run in the [`UsageTracker`](#-usagetracker--usage-statistics).

```python
import PIL.Image
from smolagents import InferenceClientModel, ImageAnalysisTool, VLMCodeAgent

model = InferenceClientModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
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
from smolagents import InferenceClientModel, ImageAnalysisTool, CodeAgent

model = InferenceClientModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
analysis_tool = ImageAnalysisTool(model=model)

agent = CodeAgent(tools=[analysis_tool], model=model)
```

---

### 📊 UsageTracker — Usage Statistics

`UsageTracker` is a thread-safe singleton that records project-wide statistics across every agent
run: models used, tools called, token counts, and VLM-vs-text run breakdown.  
It is updated **automatically** — no extra code needed in your agents.

```python
from smolagents import CodeAgent, InferenceClientModel
from smolagents.monitoring import get_usage_tracker

model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model)
agent.run("What is 2+2?")

tracker = get_usage_tracker()
print(tracker.get_summary())
# {'run_count': 1, 'vlm_run_count': 0, 'model_invocations': {...},
#  'tool_invocations': {...}, 'total_token_usage': {...}}

tracker.reset()  # Clear all counters
```

---

### 🔗 AgentPipeline — Multi-Agent Pipelines

Four cooperating primitives let you compose agents into reusable, linearly-chained pipelines:

| Class | Purpose |
|---|---|
| `AgentConfig` | Fully parametrized, serialisable agent descriptor |
| `AgentFactory` | Creates (and optionally caches) named agent instances from `AgentConfig` |
| `PipelineStep` | Wraps an agent for use in a pipeline with optional input/output transforms |
| `AgentPipeline` | Connects agents in sequence; each agent's output becomes the next agent's task |

```python
import PIL.Image
from smolagents import CodeAgent, InferenceClientModel, ImageAnalysisTool
from smolagents.pipeline import AgentConfig, AgentFactory, AgentPipeline

# Vision-capable model for image analysis
vision_model = InferenceClientModel(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
# Text model for the final reporting step
text_model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")

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

## License

Licensed under the [Apache License 2.0](LICENSE).  
Original library by the HuggingFace Team.
