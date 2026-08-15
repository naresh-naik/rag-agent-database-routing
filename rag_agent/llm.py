"""
RAG Agent with Database Routing - Multi-provider LLM configuration.

Supports Groq, OpenAI, and Gemini through their OpenAI-compatible endpoints,
so the whole pipeline (router, generator, fallback, vision OCR) runs against
whichever provider the user has a key for. The active provider is set at
runtime (e.g. from the UI); when nothing is configured, every call site
falls back to its module-level MODEL default (RAG_MODEL env / Groq 70b),
keeping the eval harnesses and tests unchanged.
"""

from __future__ import annotations

import os

from openai import OpenAI

PROVIDERS = {
    "groq": {
        "label": "Groq",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "vision_models": ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"],
    },
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "vision_models": ["gpt-4o-mini"],
    },
    "gemini": {
        "label": "Gemini",
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "vision_models": ["gemini-2.0-flash"],
    },
}

# Priority order for auto-selecting a provider when several keys are present.
PROVIDER_ORDER = ["groq", "openai", "gemini"]

_active = {"provider": None, "model": None}


def model_for(provider: str) -> str:
    """The text model used for routing/generation for a provider.

    The RAG_MODEL env var overrides the Groq default (escape hatch used for
    quota management); other providers keep their defaults.
    """
    if provider == "groq":
        env_model = os.getenv("RAG_MODEL", "").strip()
        if env_model:
            return env_model
    return PROVIDERS[provider]["default_model"]


def build_client(provider: str, api_key: str) -> OpenAI:
    """Build an OpenAI-compatible client for the given provider."""
    cfg = PROVIDERS[provider]
    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def set_active(provider: str) -> str:
    """Mark a provider as active; returns the model that will be used."""
    model = model_for(provider)
    _active["provider"] = provider
    _active["model"] = model
    return model


def get_active_provider() -> str | None:
    return _active["provider"]


def get_active_model() -> str | None:
    """Active text model, or None when no provider has been configured
    (callers then fall back to their own MODEL default)."""
    return _active["model"]


def vision_models_for(provider: str | None = None) -> list[str]:
    """Vision-capable models to try for image OCR, for the given (or active)
    provider."""
    provider = provider or _active["provider"] or "groq"
    return PROVIDERS[provider]["vision_models"]
