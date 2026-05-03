# VLM Usage Tracking

This example shows how to run a **Vision-Language Model (VLM)** agent with smolagents
and inspect detailed per-run statistics with the built-in
[`UsageTracker`](../../src/smolagents/monitoring.py).

Two distinct backends are demonstrated side-by-side:

| Backend | smolagents class | Typical use-case |
|---|---|---|
| **HuggingFace** | `InferenceClientModel` | HF Inference API / Hub multimodal checkpoints |
| **External API** | `OpenAIModel` | OpenAI, Groq, Together AI, LLM7.io, local vLLM, … |

---

## Files

| File | Purpose |
|---|---|
| `config.yaml` | All model IDs, API keys, and agent parameters — **edit this file, not the script** |
| `vlm_usage_tracking.py` | Example script — reads `config.yaml` and runs both VLM backends |
| `README.md` | This file |

---

## Quick Start

### 1. Install dependencies

```bash
pip install 'smolagents[openai]' pyyaml pillow
```

> `pillow` is required for image handling; `pyyaml` for reading `config.yaml`.

### 2. Edit `config.yaml`

Open [`config.yaml`](config.yaml) and fill in your credentials and preferred models:

```yaml
huggingface:
  model_id: "meta-llama/Llama-3.2-11B-Vision-Instruct"
  provider: null        # e.g. "nebius" — or null for default HF Inference API
  token: null           # null → reads HF_TOKEN env variable

external_api:
  model_id: "gpt-4o"
  api_base: "https://api.openai.com/v1"   # swap for any OpenAI-compatible URL
  api_key: null         # null → reads OPENAI_API_KEY env variable

agent:
  max_steps: 5
  image_path: null      # path to your image, or null for a synthetic test image
  task: "Describe what you see in this image in detail."
```

All keys that accept `null` fall back to the corresponding environment variable
(`HF_TOKEN` / `OPENAI_API_KEY`) so you never need to commit credentials.

#### Supported `api_base` values for `external_api`

| Provider | `api_base` |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| [LLM7.io](https://llm7.io) | `https://llm7.io/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Together AI | `https://api.together.xyz/v1` |
| Local vLLM | `http://localhost:8000/v1` |

### 3. Run the script

```bash
# Run both backends sequentially (default)
python vlm_usage_tracking.py

# Run only the HuggingFace backend
python vlm_usage_tracking.py --backend hf

# Run only the external-API backend
python vlm_usage_tracking.py --backend api

# Use a different config file
python vlm_usage_tracking.py --config /path/to/my_config.yaml
```

---

## What the script does

For **each selected backend** the script:

1. Loads the model and builds a `VLMCodeAgent` with an `ImageAnalysisTool`.
2. Loads the image specified in `config.yaml` (or creates a synthetic gradient image
   if `image_path` is `null`).
3. Runs the agent on the configured `task`.
4. Calls `get_usage_tracker().get_summary()` and prints:
   - `run_count` — total agent runs recorded so far
   - `vlm_run_count` — number of those runs that used a VLM agent
   - `model_invocations` — per-model-ID invocation counts
   - `tool_invocations` — per-tool invocation counts
   - `total_token_usage` — cumulative input / output / total tokens
5. Calls `tracker.reset()` to clear counters before the next backend runs.

### Example output

```
>>> [HuggingFace] Running VLM agent with model 'meta-llama/Llama-3.2-11B-Vision-Instruct'
>>> Result: The image shows a colourful gradient …

============================================================
  UsageTracker summary — HuggingFace InferenceClientModel
============================================================
  run_count: 1
  vlm_run_count: 1
  model_invocations: {'meta-llama/Llama-3.2-11B-Vision-Instruct': 1}
  tool_invocations: {'image_analysis': 1}
  total_token_usage: {'input_tokens': 312, 'output_tokens': 87, 'total_tokens': 399}
============================================================

>>> [External API] Running VLM agent with model 'gpt-4o' at 'https://api.openai.com/v1'
>>> Result: The image displays a smooth gradient …

============================================================
  UsageTracker summary — External OpenAI-compatible API (OpenAIModel)
============================================================
  run_count: 1
  vlm_run_count: 1
  model_invocations: {'gpt-4o': 1}
  tool_invocations: {'image_analysis': 1}
  total_token_usage: {'input_tokens': 298, 'output_tokens': 74, 'total_tokens': 372}
============================================================
```

---

## Key classes used

| Class | Module | Role |
|---|---|---|
| `VLMCodeAgent` | `smolagents` | Vision-capable `CodeAgent` subclass; sets `_is_vlm_agent = True` |
| `InferenceClientModel` | `smolagents` | HuggingFace Inference API backend |
| `OpenAIModel` | `smolagents` | OpenAI-compatible API backend |
| `ImageAnalysisTool` | `smolagents` | Wraps a vision model as a callable tool |
| `UsageTracker` | `smolagents.monitoring` | Thread-safe singleton; auto-updated after every `agent.run()` |
| `get_usage_tracker()` | `smolagents.monitoring` | Returns the global `UsageTracker` singleton |
