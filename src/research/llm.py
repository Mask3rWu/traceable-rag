"""Single construction point for the OpenAI-compatible chat model.

The research agent and the eval judge both talk to the same OpenAI-compatible
upstream (deepseek), so the model must be built identically in both places.
Any parameter that depends on the upstream's quirks belongs here so it is fixed
once. Two quirks drive the extra_body handling:

- langchain-openai silently renames ``max_tokens`` to
  ``max_completion_tokens`` (``_get_request_payload`` pops ``max_tokens`` into
  ``max_completion_tokens``), but deepseek does not recognize
  ``max_completion_tokens`` and ignores it. So a token cap cannot go through
  ``ChatOpenAI.max_tokens``; it must be injected via ``extra_body``, which the
  SDK merges verbatim into the request body, where deepseek honors ``max_tokens``.
- The thinking toggle is a deepseek OpenAI-compatible extension that also has to
  be a top-level body field. ``extra_body`` is the correct vehicle here too
  (``model_kwargs`` would be treated as a named ``create()`` arg and fail SDK
  validation).

Verified empirically against ``api.deepseek.com/chat/completions`` (budget 8):
``max_tokens:8`` returned exactly 8 completion tokens; ``max_completion_tokens:8``
returned 18, silently ignoring the cap.
"""
from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI


def build_chat_model(
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0,
    disable_thinking: bool = False,
    token_budget: int | None = None,
) -> ChatOpenAI:
    """Construct the OpenAI-compatible chat model with upstream-aware defaults.

    Args:
        model: Model name as the upstream expects it (e.g. ``deepseek-v4-flash``).
        base_url: OpenAI-compatible endpoint root (e.g. ``https://api.deepseek.com``).
        api_key: Credential for ``base_url``.
        temperature: Sampling temperature; the agent runs greedy (``0``).
        disable_thinking: When true, send ``extra_body.thinking.type=disabled``
            so the model stops spending completion budget on reasoning tokens.
        token_budget: Optional hard cap on completion tokens, sent as top-level
            ``extra_body.max_tokens``. Defaults to ``None`` = no cap (current
            behavior). Do not set ``ChatOpenAI.max_tokens`` directly: langchain
            would rewrite it to ``max_completion_tokens``, which deepseek ignores.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
    }
    extra_body: dict[str, Any] = {}
    if token_budget is not None:
        extra_body["max_tokens"] = token_budget
    if disable_thinking:
        extra_body["thinking"] = {"type": "disabled"}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)