"""OpenAI provider implementation.

Wraps `client.chat.completions.create`. OpenAI's prompt cache is automatic
for prompts >1024 tokens (no explicit cache flag), so the system prompt
is just sent as-is and the discount happens server-side.

Tool use differs from Anthropic in three places:
  1. Tool spec format: `{"type": "function", "function": {...}}`
  2. Tool-call response: `message.tool_calls = [ToolCall(id, function=...)]`
  3. Tool-result message: `{"role": "tool", "tool_call_id": ..., "content": ...}`

All translation lives in this module; the rest of the codebase only sees
the normalized `LLMResponse` / `LLMToolCall` types.
"""

from __future__ import annotations

import json
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


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._check_availability()
        import openai
        self._client = openai.OpenAI(api_key=self._api_key)

    def _check_availability(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. "
                "Set it via `export OPENAI_API_KEY=...` or pass api_key=."
            )
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed. `pip install openai`"
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
        api_messages: list[dict] = [{"role": "system", "content": system}]
        api_messages.extend(self._to_openai_message(m) for m in messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [self._translate_tool(t) for t in tools]
            # Let the model decide whether to call tools or emit text.
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        return self._normalize_response(response)

    def _to_openai_message(self, m: LLMMessage) -> dict:
        # If we built this message via format_assistant_with_tool_calls /
        # format_tool_result it's already a dict — pass through.
        if isinstance(m.content, dict):
            return m.content  # already a complete message
        if isinstance(m.content, list):
            # Caller pre-built an OpenAI-shaped message (assistant + tool_calls)
            # — assume it's well-formed.
            if all(isinstance(x, dict) for x in m.content):
                # Unusual but if someone passes content-blocks for OpenAI,
                # collapse to text.
                texts = [x.get("text", "") for x in m.content if x.get("type") == "text"]
                return {"role": m.role, "content": "".join(texts)}
        return {"role": m.role, "content": m.content}

    def _translate_tool(self, t: LLMToolSpec) -> dict:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }

    def _normalize_response(self, response) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls: list[LLMToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                logger.warning("OpenAI returned non-JSON tool args: %r", tc.function.arguments)
                args = {"_raw": tc.function.arguments}
            tool_calls.append(LLMToolCall(
                id=tc.id, name=tc.function.name, input=args,
            ))
        # Normalize stop reason. OpenAI's finish_reason is "stop" | "length" |
        # "tool_calls" | "content_filter". Map to our terms.
        finish = getattr(choice, "finish_reason", "stop") or "stop"
        stop_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
        }
        stop_reason = stop_map.get(finish, finish)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=self._normalize_usage(response),
            stop_reason=stop_reason,
            raw=response,
        )

    def _normalize_usage(self, response) -> dict:
        u = getattr(response, "usage", None)
        if not u:
            return {}
        # OpenAI's usage shape: prompt_tokens, completion_tokens, total_tokens,
        # plus prompt_tokens_details.cached_tokens when caching kicks in.
        prompt = getattr(u, "prompt_tokens", 0)
        completion = getattr(u, "completion_tokens", 0)
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return {
            # Normalized common keys:
            "input_tokens": prompt,
            "output_tokens": completion,
            "cache_read_tokens": cached,
            # OpenAI-specific raw fields (handy for cost analysis):
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
        }

    # -- tool-call message helpers --------------------------------------- #

    def format_assistant_with_tool_calls(
        self, text: str, tool_calls: list[LLMToolCall]
    ) -> dict:
        """OpenAI's assistant-with-tool-calls message has tool_calls as a
        sibling field, with arguments JSON-stringified."""
        oai_tool_calls = []
        for tc in tool_calls:
            oai_tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.input),
                },
            })
        out: dict = {"role": "assistant", "content": text or None}
        if oai_tool_calls:
            out["tool_calls"] = oai_tool_calls
        return out

    def format_tool_result(self, tool_call_id: str, output: str) -> dict:
        """OpenAI uses role='tool' with the tool_call_id."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": output,
        }
