"""Anthropic provider implementation.

Wraps `client.messages.create`. Uses cache_control=ephemeral on the system
prompt so subsequent calls hit the prompt cache.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from lab.llm.provider import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    LLMToolSpec,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-opus-4-7"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._check_availability()
        import anthropic
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def _check_availability(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. "
                "Set it via `export ANTHROPIC_API_KEY=...` or pass api_key=."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. `pip install anthropic`"
            ) from e

    # -- generation ------------------------------------------------------ #

    def generate(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[LLMToolSpec] | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        api_messages = [self._to_anthropic_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [self._translate_tool(t) for t in tools]

        response = self._client.messages.create(**kwargs)
        return self._normalize_response(response)

    def _to_anthropic_message(self, m: LLMMessage) -> dict:
        # If content is already a list (e.g., tool_use blocks for an assistant
        # message we pre-constructed), pass through unchanged.
        if isinstance(m.content, list):
            return {"role": m.role, "content": m.content}
        return {"role": m.role, "content": m.content}

    def _translate_tool(self, t: LLMToolSpec) -> dict:
        return {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }

    def _normalize_response(self, response) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(LLMToolCall(
                    id=block.id, name=block.name, input=dict(block.input),
                ))
        usage = self._normalize_usage(response)
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            raw=response,
        )

    def _normalize_usage(self, response) -> dict:
        u = getattr(response, "usage", None)
        if not u:
            return {}
        return {
            "input_tokens": getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
            # Provider-agnostic alias used by reporting code:
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0),
        }

    # -- tool-call message helpers --------------------------------------- #

    def format_assistant_with_tool_calls(
        self, text: str, tool_calls: list[LLMToolCall]
    ) -> dict:
        """Anthropic expects the assistant's prior turn echoed back as a list
        of content blocks, including any tool_use blocks the model emitted."""
        content: list[dict] = []
        if text.strip():
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })
        return {"role": "assistant", "content": content}

    def format_tool_result(self, tool_call_id: str, output: str) -> dict:
        """Anthropic packs tool results into a user message with a tool_result
        content block."""
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": output,
            }],
        }
