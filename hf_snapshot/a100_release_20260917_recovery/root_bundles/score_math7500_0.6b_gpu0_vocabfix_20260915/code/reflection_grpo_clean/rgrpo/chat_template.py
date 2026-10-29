"""Shared chat-template controls.

Qwen3 exposes ``enable_thinking`` as a chat-template keyword. Leaving the
keyword unset preserves older-backbone behaviour; setting
``CHAT_TEMPLATE_ENABLE_THINKING=false`` hard-disables thinking for Qwen3.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def chat_template_kwargs(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    source = os.environ if env is None else env
    raw = source.get("CHAT_TEMPLATE_ENABLE_THINKING")
    if raw is None or not raw.strip():
        return {}
    value = raw.strip().lower()
    if value in _TRUE:
        return {"enable_thinking": True}
    if value in _FALSE:
        return {"enable_thinking": False}
    raise ValueError(
        "CHAT_TEMPLATE_ENABLE_THINKING must be one of "
        f"{sorted(_TRUE | _FALSE)}, got {raw!r}."
    )


def apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> Any:
    template_kwargs = chat_template_kwargs()
    template_kwargs.update(kwargs)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
