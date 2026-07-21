"""
Provider-agnostic text generation.

Wraps the generator LLM behind a single `generate(system_prompt, user_prompt)`
function so the rest of the pipeline does not care which provider is used. Two
backends are supported, selected by the LLM_BACKEND environment variable:

  * "gemini"  -> Google Gemini via the free AI Studio tier (default; needs
                 GEMINI_API_KEY, no billing required).
  * "openai"  -> OpenAI Chat Completions (needs OPENAI_API_KEY, pay-per-use).

Keeping this abstraction isolated means switching or adding providers is a one-file
change and does not touch the RAG or baseline logic.
"""

from __future__ import annotations

import os

from .config import config


def _generate_gemini(system_prompt: str, user_prompt: str) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com "
            "and run: export GEMINI_API_KEY=..."
        )

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=config.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=config.llm_temperature,
            max_output_tokens=config.llm_max_tokens,
        ),
    )
    return (resp.text or "").strip()


def _generate_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=config.openai_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def generate(system_prompt: str, user_prompt: str) -> str:
    """Generate a response using the configured backend."""
    backend = config.llm_backend
    if backend == "gemini":
        return _generate_gemini(system_prompt, user_prompt)
    if backend == "openai":
        return _generate_openai(system_prompt, user_prompt)
    raise ValueError(f"Unknown LLM backend: {backend}")


def backend_name() -> str:
    """Return the human-readable name of the active backend + model."""
    if config.llm_backend == "gemini":
        return f"gemini:{config.gemini_model}"
    return f"openai:{config.openai_model}"
