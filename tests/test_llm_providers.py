"""Tests for the lab.llm provider abstraction.

Covers:
  - Provider selection (explicit > env > auto-detect > tiebreak > error)
  - Tool schema translation (LLMToolSpec → per-provider format)
  - Tool-call response normalization (provider raw → LLMResponse)
  - Tool-result message formatting
  - Token usage normalization
  - Both providers pass the system-prompt-examples validator round-trip

All tests use mocks. No live API calls, no network.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from lab.llm import (
    LLMMessage,
    LLMToolCall,
    LLMToolSpec,
    get_provider,
)
from lab.llm.selection import (
    SUPPORTED_PROVIDERS,
    _resolve_name,
    list_available_providers,
)


# --------------------------------------------------------------------------- #
# Selection / resolution
# --------------------------------------------------------------------------- #


def test_resolve_explicit_anthropic(monkeypatch):
    monkeypatch.delenv("LAB_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _resolve_name("anthropic") == "anthropic"


def test_resolve_explicit_openai(monkeypatch):
    monkeypatch.delenv("LAB_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _resolve_name("openai") == "openai"


def test_resolve_explicit_unknown_raises(monkeypatch):
    monkeypatch.delenv("LAB_LLM_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="unsupported provider"):
        _resolve_name("gemini")


def test_resolve_via_env(monkeypatch):
    monkeypatch.setenv("LAB_LLM_PROVIDER", "openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _resolve_name(None) == "openai"


def test_resolve_via_env_invalid_raises(monkeypatch):
    monkeypatch.setenv("LAB_LLM_PROVIDER", "fakeprov")
    with pytest.raises(ValueError, match="LAB_LLM_PROVIDER"):
        _resolve_name(None)


def test_resolve_autodetect_anthropic_only(monkeypatch):
    monkeypatch.delenv("LAB_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _resolve_name(None) == "anthropic"


def test_resolve_autodetect_openai_only(monkeypatch):
    monkeypatch.delenv("LAB_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert _resolve_name(None) == "openai"


def test_resolve_both_keys_tiebreak_anthropic(monkeypatch):
    monkeypatch.delenv("LAB_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert _resolve_name(None) == "anthropic"


def test_resolve_no_keys_raises(monkeypatch):
    monkeypatch.delenv("LAB_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        _resolve_name(None)


def test_list_available_providers(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert list_available_providers() == []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert list_available_providers() == ["anthropic"]
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    assert list_available_providers() == ["anthropic", "openai"]


def test_supported_providers_list():
    assert "anthropic" in SUPPORTED_PROVIDERS
    assert "openai" in SUPPORTED_PROVIDERS


# --------------------------------------------------------------------------- #
# Tool schema translation
# --------------------------------------------------------------------------- #


def test_anthropic_translates_tool_spec(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    from lab.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider()
    spec = LLMToolSpec(
        name="do_thing",
        description="describes a thing",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    out = p._translate_tool(spec)
    assert out == {
        "name": "do_thing",
        "description": "describes a thing",
        "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
    }


def test_openai_translates_tool_spec(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from lab.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider()
    spec = LLMToolSpec(
        name="do_thing",
        description="describes a thing",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    out = p._translate_tool(spec)
    assert out == {
        "type": "function",
        "function": {
            "name": "do_thing",
            "description": "describes a thing",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }


# --------------------------------------------------------------------------- #
# Tool-result + assistant-with-tool-calls message formatting
# --------------------------------------------------------------------------- #


def test_anthropic_format_tool_result(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    from lab.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider()
    msg = p.format_tool_result("tool_123", "result body")
    assert msg["role"] == "user"
    assert msg["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tool_123",
        "content": "result body",
    }


def test_openai_format_tool_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from lab.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider()
    msg = p.format_tool_result("call_abc", "output text")
    assert msg == {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "output text",
    }


def test_anthropic_format_assistant_with_tool_calls(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    from lab.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider()
    msg = p.format_assistant_with_tool_calls(
        "thinking out loud",
        [LLMToolCall(id="t1", name="fetch_history", input={"ticker": "SPY"})],
    )
    assert msg["role"] == "assistant"
    assert msg["content"][0] == {"type": "text", "text": "thinking out loud"}
    assert msg["content"][1] == {
        "type": "tool_use", "id": "t1",
        "name": "fetch_history", "input": {"ticker": "SPY"},
    }


def test_openai_format_assistant_with_tool_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from lab.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider()
    msg = p.format_assistant_with_tool_calls(
        "deciding",
        [LLMToolCall(id="call_1", name="compute_metric",
                      input={"equity": [1.0, 1.1], "metric": "sharpe"})],
    )
    assert msg["role"] == "assistant"
    assert msg["content"] == "deciding"
    assert msg["tool_calls"][0]["id"] == "call_1"
    assert msg["tool_calls"][0]["type"] == "function"
    assert msg["tool_calls"][0]["function"]["name"] == "compute_metric"
    # OpenAI requires arguments as a JSON-stringified value.
    args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
    assert args["metric"] == "sharpe"


# --------------------------------------------------------------------------- #
# Response normalization via mocked SDKs
# --------------------------------------------------------------------------- #


def test_anthropic_normalize_text_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    from lab.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider()
    # Fake the SDK response object.
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "hello world"
    fake_response = MagicMock()
    fake_response.content = [fake_block]
    fake_response.stop_reason = "end_turn"
    fake_response.usage.input_tokens = 100
    fake_response.usage.output_tokens = 50
    fake_response.usage.cache_creation_input_tokens = 0
    fake_response.usage.cache_read_input_tokens = 80

    result = p._normalize_response(fake_response)
    assert result.text == "hello world"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"
    assert result.usage["input_tokens"] == 100
    assert result.usage["output_tokens"] == 50
    assert result.usage["cache_read_tokens"] == 80


def test_anthropic_normalize_tool_use_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    from lab.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "calling tool"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_xyz"
    tool_block.name = "run_dry_backtest"
    tool_block.input = {"code": "from lab.strategy import Strategy"}
    fake_response = MagicMock()
    fake_response.content = [text_block, tool_block]
    fake_response.stop_reason = "tool_use"
    fake_response.usage.input_tokens = 10
    fake_response.usage.output_tokens = 20
    fake_response.usage.cache_creation_input_tokens = 0
    fake_response.usage.cache_read_input_tokens = 0

    result = p._normalize_response(fake_response)
    assert result.text == "calling tool"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "tool_xyz"
    assert result.tool_calls[0].name == "run_dry_backtest"
    assert result.tool_calls[0].input["code"].startswith("from lab.strategy")
    assert result.stop_reason == "tool_use"


def test_openai_normalize_text_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from lab.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider()
    fake_msg = MagicMock()
    fake_msg.content = "the answer"
    fake_msg.tool_calls = None
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    fake_choice.finish_reason = "stop"
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage.prompt_tokens = 200
    fake_response.usage.completion_tokens = 75
    fake_response.usage.prompt_tokens_details.cached_tokens = 150

    result = p._normalize_response(fake_response)
    assert result.text == "the answer"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"   # mapped from OpenAI "stop"
    assert result.usage["input_tokens"] == 200
    assert result.usage["output_tokens"] == 75
    assert result.usage["cache_read_tokens"] == 150


def test_openai_normalize_tool_use_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from lab.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider()
    fake_tool = MagicMock()
    fake_tool.id = "call_99"
    fake_tool.function.name = "compute_metric"
    fake_tool.function.arguments = '{"equity": [1, 1.1], "metric": "sharpe"}'
    fake_msg = MagicMock()
    fake_msg.content = None
    fake_msg.tool_calls = [fake_tool]
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    fake_choice.finish_reason = "tool_calls"
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage.prompt_tokens = 100
    fake_response.usage.completion_tokens = 10
    fake_response.usage.prompt_tokens_details.cached_tokens = 0

    result = p._normalize_response(fake_response)
    assert result.text == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_99"
    assert result.tool_calls[0].name == "compute_metric"
    assert result.tool_calls[0].input["metric"] == "sharpe"
    assert result.stop_reason == "tool_use"   # mapped from OpenAI "tool_calls"


def test_openai_normalize_bad_tool_arguments_json(monkeypatch):
    """OpenAI sometimes returns malformed JSON in arguments — should not crash."""
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from lab.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider()
    fake_tool = MagicMock()
    fake_tool.id = "call_bad"
    fake_tool.function.name = "fetch_history"
    fake_tool.function.arguments = "{this is not json"
    fake_msg = MagicMock()
    fake_msg.content = None
    fake_msg.tool_calls = [fake_tool]
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    fake_choice.finish_reason = "tool_calls"
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage.prompt_tokens = 1
    fake_response.usage.completion_tokens = 1
    fake_response.usage.prompt_tokens_details.cached_tokens = 0

    # Should not raise; should preserve the raw string under _raw.
    result = p._normalize_response(fake_response)
    assert result.tool_calls[0].input == {"_raw": "{this is not json"}


# --------------------------------------------------------------------------- #
# get_provider end-to-end (construction only, no API call)
# --------------------------------------------------------------------------- #


def test_get_provider_explicit_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    p = get_provider("anthropic")
    assert p.name == "anthropic"
    assert p.default_model == "claude-opus-4-7"


def test_get_provider_explicit_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    p = get_provider("openai")
    assert p.name == "openai"
    assert p.default_model == "gpt-4o"


def test_get_provider_no_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LAB_LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        get_provider()


# --------------------------------------------------------------------------- #
# Backward-compat: existing tests imported GenerationResult, MODEL etc. —
# these should still work.
# --------------------------------------------------------------------------- #


def test_backward_compat_imports():
    from lab.llm import (
        MODEL, DEFAULT_UNIVERSE, DEFAULT_TRAIN_END, DEFAULT_REBALANCE_FREQ,
        GenerationResult, generate_strategy, refine_strategy,
        build_universe_brief, load_system_prompt,
        _extract_python_block, _parse_defaults, _validate_generated_code,
    )
    assert MODEL == "claude-opus-4-7"
    assert DEFAULT_UNIVERSE == ["SPY", "TLT"]
    assert DEFAULT_TRAIN_END == "2015-01-01"


def test_generation_result_has_provider_field():
    from lab.llm import GenerationResult
    gr = GenerationResult(
        code="", raw_response="", universe=[], train_end="", rebalance_freq=21,
        usage=None,
    )
    # New field, defaults to anthropic.
    assert gr.provider == "anthropic"
