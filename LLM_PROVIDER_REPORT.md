# LLM Provider Abstraction — Implementation Report

**Branch:** `strategy-lab`
**Session date:** 2026-05-19
**Outcome:** Anthropic + OpenAI both supported through a single provider interface. All paths (`backtest`, `--refine`, `--agent`, `lab chat`, `lab fork`) work with either backend. 207 tests passing.

---

## What changed

| Before | After |
|---|---|
| `lab/llm.py` (single module, Anthropic-only) | `lab/llm/` package with provider abstraction |
| `MODEL = "claude-opus-4-7"` hardcoded | `default_model` per provider class |
| `_make_client`, `_call_anthropic`, `_extract_usage` private helpers | `LLMProvider.generate()` interface returns normalized `LLMResponse` |
| `TOOL_SCHEMAS` Anthropic-format dicts | `TOOL_SPECS: list[LLMToolSpec]`, translated per-provider |
| CLI `--model` defaulted to claude-opus-4-7 | CLI `--provider {anthropic,openai}` + `--model` defaults to None (provider picks) |
| Agent loop directly called `client.messages.create` | Agent loop calls `provider.generate(..., tools=TOOL_SPECS)`; provider handles tool-call format |

## Files

New:
- `lab/llm/__init__.py` — public API (generate_strategy, refine_strategy, build_universe_brief, GenerationResult, helpers). Imports preserved for backward compat.
- `lab/llm/provider.py` — `LLMProvider` ABC, `LLMMessage`, `LLMResponse`, `LLMToolCall`, `LLMToolSpec` dataclasses.
- `lab/llm/anthropic_provider.py` — Anthropic implementation. Cache_control=ephemeral on system prompt.
- `lab/llm/openai_provider.py` — OpenAI implementation. Default `gpt-4o`. OpenAI's auto prompt caching kicks in for prompts > 1024 tokens.
- `lab/llm/selection.py` — `get_provider()` resolution (explicit > env > auto-detect > tiebreak-anthropic).
- `tests/test_llm_providers.py` — 27 new tests (mocked SDKs).

Modified:
- `lab/agent.py` — agent loop refactored onto the provider interface. `TOOL_SCHEMAS` → `TOOL_SPECS` (list of `LLMToolSpec`). Provider-specific tool message formatting via `format_assistant_with_tool_calls` / `format_tool_result`.
- `lab/cli.py` — `--provider` flag on `backtest` and `fork`. `--model` default changed to None.
- `lab/chat.py` — `SessionState.provider` field. `:provider` REPL command for live switching.
- `pyproject.toml` — `openai>=1.30` added to required deps.

Removed:
- `lab/llm.py` (replaced by the package). All public names re-exported, all 124 existing tests continued to pass at each commit.

## Provider selection priority

```
1. Explicit kwarg: generate_strategy(prompt, provider="openai")
2. Explicit CLI flag: lab backtest "..." --provider openai
3. LAB_LLM_PROVIDER env var
4. Auto-detect from available keys:
   - only ANTHROPIC_API_KEY set → anthropic
   - only OPENAI_API_KEY set    → openai
5. Both keys set → anthropic (preserves original behavior)
6. Neither set → RuntimeError with a clear message
```

## Decisions made unilaterally

| Decision | Choice | Rationale |
|---|---|---|
| OpenAI default model | `gpt-4o` | Best price/quality balance for code generation; user can override via `--model gpt-4o-mini` (cheaper) or `--model gpt-4-turbo` |
| Tiebreak when both keys set | `anthropic` | Preserves original behavior; existing tests that assumed Anthropic don't break |
| Caching strategy | Provider-native | Anthropic uses explicit `cache_control: ephemeral`; OpenAI uses automatic caching for prompts ≥1024 tokens. Both happen invisibly to users. |
| Token usage shape | Common normalized keys + provider-specific extras | `input_tokens`, `output_tokens`, `cache_read_tokens` always present. OpenAI keeps `prompt_tokens` / `completion_tokens` / `cached_tokens` for cost-analysis tooling. |
| OpenAI tool_choice | `"auto"` when tools are passed | Lets the model decide; matches Anthropic's default |
| `--model` default | `None` (provider picks) | Saves users from passing `--model claude-opus-4-7` when they're using OpenAI |
| LLM-touching tests | Mock SDKs entirely | No live API calls in CI. The `RUN_LIVE_TESTS=1` opt-in pattern used elsewhere applies here too. |
| `JSON mode` for OpenAI | Not used | Conflicts with our markdown code-block extractor. Text mode + existing regex works for both providers. |

## What still works the same

- All existing CLI invocations with `ANTHROPIC_API_KEY` set continue to behave identically — the auto-detect path picks Anthropic, defaults preserve `claude-opus-4-7` if you ever pass `--model` explicitly.
- All 124 pre-existing lab tests pass without modification.
- 56 regime_model tests untouched.
- The strategy code Claude/GPT generates runs through the same validator, the same backtester, the same paper executor. The provider only matters at the moment of code generation.

## Honest caveats

1. **OpenAI agent loop is mock-tested only.** The 4 tool-call → response → tool-result message exchanges have been verified by parsing fake `Choice/Message/ToolCall` objects, but a real OpenAI tool-use round-trip hasn't been exercised in CI. First real `lab backtest ... --provider openai --agent` may surface issues I haven't seen.
2. **Quality parity not empirically verified.** I haven't done a side-by-side comparison of code quality between Opus 4.7 and GPT-4o for this specific system prompt. The system prompt was tuned for Claude; if GPT-4o reliably fails on certain few-shot patterns, that needs measurement and possibly per-provider prompt tweaks.
3. **No streaming.** Neither provider streams tokens to the user. Per-turn responses arrive whole.
4. **Anthropic's prompt caching is more aggressive.** With explicit `cache_control: ephemeral`, Anthropic caches per-call deterministically. OpenAI's automatic cache requires prompts ≥1024 tokens and identical prefixes — the system prompt qualifies, but the cache-hit rate may differ in practice.

## How to use both keys productively

If you have credits on both:

```bash
# Cheap exploration on OpenAI (gpt-4o is ~6x cheaper):
export LAB_LLM_PROVIDER=openai
lab backtest "..."
lab fork <id> "..."

# Final iteration on Anthropic where --agent quality matters most:
unset LAB_LLM_PROVIDER
lab backtest "..." --agent --provider anthropic
```

Or use the chat REPL for switching mid-session:

```
>> :provider openai
>> a bunch of cheap exploratory prompts
>> :provider anthropic
>> the one prompt where I want the best generation
>> :exit
```

## Commits

| SHA | Description |
|---|---|
| (first) | `llm: introduce provider abstraction (Anthropic implementation only)` |
| (second) | `llm: add OpenAI provider (gpt-4o default)` |
| (third) | `agent: refactor multi-tool loop onto provider abstraction` |
| (fourth) | `cli/chat: --provider flag, model default per provider` |
| (fifth) | `tests: 27 tests for the LLM provider abstraction` |
| (sixth — this one) | Docs + this report + push |

## Verification checklist

- [x] All 124 pre-existing lab tests pass at every intermediate commit
- [x] 27 new provider tests pass
- [x] 56 regime_model tests still pass
- [x] `lab backtest --help` shows `--provider` flag
- [x] `lab fork --help` shows `--provider` flag
- [x] `lab chat` has `:provider` command + updated help text
- [x] `lab live-trade` still raises NotImplementedError (live trading guard unchanged)
- [x] `OPENAI_API_KEY` env var honored by auto-detect
- [x] `LAB_LLM_PROVIDER` env var honored
- [x] `pyproject.toml` lists openai in required deps
- [x] STRATEGY_LAB.md has new "LLM provider" section
- [x] Pushed to `strategy-lab` branch
