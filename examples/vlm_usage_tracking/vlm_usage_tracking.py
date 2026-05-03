#!/usr/bin/env python
# coding=utf-8
"""VLM usage-tracking examples.

Demonstrates two distinct ways to run a Vision-Language Model (VLM) agent
and inspect per-run statistics through :class:`~smolagents.monitoring.UsageTracker`:

1. **HuggingFace model** — via :class:`~smolagents.InferenceClientModel` pointing
   at a multimodal checkpoint on the HuggingFace Hub.
2. **External API model** — via :class:`~smolagents.OpenAIModel` pointing at any
   OpenAI-compatible endpoint (OpenAI, Groq, Together AI, LLM7.io, local vLLM, …).

All model identifiers, API keys, and agent settings are read from ``config.yaml``
in the same directory — no code changes required when switching models or providers.

Usage
-----
    python vlm_usage_tracking.py [--config path/to/config.yaml] [--backend hf|api|both]

Dependencies
------------
    pip install 'smolagents[openai]' pyyaml pillow
"""

import argparse
import os
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load and return the YAML configuration file."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def _resolve_image_path(image_path: str | None, config_dir: Path) -> Path | None:
    """Return an absolute Path for *image_path*, or *None* if not provided."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.is_absolute():
        p = config_dir / p
    return p


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_or_create_image(image_path: Path | None):
    """Return a PIL image from *image_path*, or a small synthetic image if None."""
    import PIL.Image

    if image_path and image_path.exists():
        return PIL.Image.open(image_path).convert("RGB")

    # Fallback: 64×64 gradient image so the script can run without a real photo.
    import numpy as np

    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        arr[i, :, 0] = i * 4          # red gradient (rows)
        arr[:, i, 2] = i * 4          # blue gradient (cols)
    arr[:, :, 1] = 128                 # constant green
    return PIL.Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# VLM run helpers
# ---------------------------------------------------------------------------

def _print_tracker_summary(label: str, summary: dict) -> None:
    """Pretty-print a UsageTracker summary with a section label."""
    print(f"\n{'=' * 60}")
    print(f"  UsageTracker summary — {label}")
    print(f"{'=' * 60}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 60}\n")


def run_huggingface_vlm(cfg: dict, image, task: str) -> None:
    """Run a VLM agent backed by HuggingFace InferenceClientModel."""
    from smolagents import ImageAnalysisTool, InferenceClientModel, VLMCodeAgent
    from smolagents.monitoring import get_usage_tracker

    hf_cfg = cfg["huggingface"]
    agent_cfg = cfg["agent"]

    # Build model — token falls back to HF_TOKEN env variable when None.
    model_kwargs: dict = {"model_id": hf_cfg["model_id"]}
    if hf_cfg.get("provider"):
        model_kwargs["provider"] = hf_cfg["provider"]
    if hf_cfg.get("token"):
        model_kwargs["token"] = hf_cfg["token"]

    model = InferenceClientModel(**model_kwargs)

    agent = VLMCodeAgent(
        tools=[ImageAnalysisTool(model=model)],
        model=model,
        max_steps=agent_cfg.get("max_steps", 5),
    )

    print(f"\n>>> [HuggingFace] Running VLM agent with model '{hf_cfg['model_id']}'")
    result = agent.run(task, images=[image])
    print(f">>> Result: {result}")

    tracker = get_usage_tracker()
    _print_tracker_summary("HuggingFace InferenceClientModel", tracker.get_summary())
    tracker.reset()


def run_external_api_vlm(cfg: dict, image, task: str) -> None:
    """Run a VLM agent backed by an external OpenAI-compatible API."""
    from smolagents import ImageAnalysisTool, OpenAIModel, VLMCodeAgent
    from smolagents.monitoring import get_usage_tracker

    api_cfg = cfg["external_api"]
    agent_cfg = cfg["agent"]

    # API key falls back to OPENAI_API_KEY env variable when None.
    api_key = api_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")

    model = OpenAIModel(
        model_id=api_cfg["model_id"],
        api_base=api_cfg.get("api_base"),
        api_key=api_key,
    )

    agent = VLMCodeAgent(
        tools=[ImageAnalysisTool(model=model)],
        model=model,
        max_steps=agent_cfg.get("max_steps", 5),
    )

    print(f"\n>>> [External API] Running VLM agent with model '{api_cfg['model_id']}' at '{api_cfg.get('api_base')}'")
    result = agent.run(task, images=[image])
    print(f">>> Result: {result}")

    tracker = get_usage_tracker()
    _print_tracker_summary("External OpenAI-compatible API (OpenAIModel)", tracker.get_summary())
    tracker.reset()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM usage-tracking examples (HuggingFace + external API)."
    )
    parser.add_argument(
        "--config",
        default=str(_CONFIG_PATH),
        help="Path to the YAML configuration file (default: config.yaml next to this script).",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "api", "both"],
        default="both",
        help=(
            "Which backend to run: "
            "'hf' for HuggingFace InferenceClientModel, "
            "'api' for external OpenAI-compatible API, "
            "'both' to run both sequentially (default)."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    config_dir = Path(args.config).parent

    # Resolve image
    image_path = _resolve_image_path(cfg["agent"].get("image_path"), config_dir)
    image = _load_or_create_image(image_path)
    task = cfg["agent"].get("task", "Describe what you see in this image in detail.")

    if args.backend in ("hf", "both"):
        run_huggingface_vlm(cfg, image, task)

    if args.backend in ("api", "both"):
        run_external_api_vlm(cfg, image, task)


if __name__ == "__main__":
    main()
