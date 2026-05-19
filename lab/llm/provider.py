"""LLM provider abstraction.

Defines the common interface every backend (Anthropic, OpenAI, ...) implements,
plus normalized response/tool-call dataclasses so the rest of the codebase
doesn't care which model is in use.

Provider-specific quirks (tool-use schema, prompt caching, response shape)
are isolated inside the concrete provider classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMMessage:
    """A single message in a conversation. Provider classes translate to
    the format their SDK expects."""
    role: str                  # "user" | "assistant"
    content: Any               # str (plain text) or list (content blocks for tool flows)


@dataclass(frozen=True)
class LLMToolCall:
    """Normalized tool-call output. Both providers' tool-use responses
    project into this shape."""
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    """Normalized response. `raw` keeps the original SDK object for debugging."""
    text: str                  # all text content concatenated
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)   # normalized: input_tokens, output_tokens, cache_read_tokens
    stop_reason: str = "end_turn"               # "end_turn" | "tool_use" | "max_tokens" | other
    raw: Any = None


@dataclass(frozen=True)
class LLMToolSpec:
    """Provider-agnostic tool specification. Translated to per-provider schema
    by `LLMProvider.translate_tools`."""
    name: str
    description: str
    input_schema: dict         # JSON Schema dict


class LLMProvider(ABC):
    """Abstract interface for an LLM backend.

    Concrete implementations: AnthropicProvider, OpenAIProvider.
    """

    name: str = "abstract"
    default_model: str = ""

    @abstractmethod
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
        """Synchronous generation. Returns a normalized LLMResponse."""

    @abstractmethod
    def format_tool_result(
        self, tool_call_id: str, output: str
    ) -> dict:
        """Build the provider-specific message representing a tool result."""

    @abstractmethod
    def format_assistant_with_tool_calls(
        self, text: str, tool_calls: list[LLMToolCall]
    ) -> dict:
        """Build the provider-specific assistant-message dict that echoes back
        the model's prior turn (including tool_use blocks) so the next call
        sees a coherent conversation."""

    # Helpers — default implementations provided.

    def is_available(self) -> bool:
        """True if this provider can be constructed (env vars set, SDK installed)."""
        try:
            self._check_availability()
            return True
        except Exception:
            return False

    def _check_availability(self) -> None:
        """Raise if the provider can't be used. Override per-provider."""
        return None
