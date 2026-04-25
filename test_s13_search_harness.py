from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agents" / "s13_search_harness.py"


def load_s13_module():
    spec = importlib.util.spec_from_file_location("s13_search_harness_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s13 = load_s13_module()


def test_correct_obvious_typos_preserves_query_intent():
    corrected = s13.correct_obvious_typos("Does Anthropic support strutured ouput?")

    assert "structured output" in corrected
    assert "Anthropic" in corrected


def test_resolve_standalone_question_uses_active_entity():
    state = s13.ConversationState(active_entities=["Anthropic Messages API"])

    standalone = s13.resolve_standalone_question("那它现在支持 structured output 吗？", state)

    assert standalone.startswith("Anthropic Messages API:")
    assert "structured output" in standalone


def test_heuristic_intent_marks_current_support_question_as_search():
    state = s13.ConversationState(active_entities=["Anthropic Messages API"])

    intent = s13.heuristic_intent("那它现在支持 structured output 吗？", state)

    assert intent.need_search is True
    assert intent.time_sensitivity == "high"


def test_heuristic_intent_keeps_local_repo_question_off_web():
    state = s13.ConversationState()

    intent = s13.heuristic_intent("这个仓库的 agents 目录是做什么的？", state)

    assert intent.need_search is False
    assert intent.intent_type == "general_fact"


def test_dedupe_results_removes_duplicate_urls_and_titles():
    results = [
        s13.SearchResult("Same Title", "https://example.com/a?utm_source=x", "one", "example.com", rank=1),
        s13.SearchResult("Same Title", "https://example.com/a", "two", "example.com", rank=2),
        s13.SearchResult("Other", "https://example.org/b", "three", "example.org", rank=3),
    ]

    deduped = s13.dedupe_results(results)

    assert [result.title for result in deduped] == ["Same Title", "Other"]
    assert [result.rank for result in deduped] == [1, 2]


def test_search_with_fake_provider_builds_evidence_pack():
    state = s13.ConversationState(active_entities=["Anthropic Messages API"])
    intent = s13.heuristic_intent("那它现在支持 structured output 吗？", state)
    plan = s13.heuristic_query_plan("那它现在支持 structured output 吗？", state, intent)

    results, notes = s13.search_with_plan(plan, s13.FakeSearchProvider())
    evidence = s13.build_evidence_pack("那它现在支持 structured output 吗？", intent, plan, results, notes)

    assert results
    assert evidence["results"][0]["url"].startswith("https://docs.anthropic.com")
    assert evidence["queries"][0]["query"]


def test_make_search_provider_uses_tavily_when_configured(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)

    provider = s13.make_search_provider()

    assert isinstance(provider, s13.TavilySearchProvider)
    assert provider.api_key == "test-key"


def test_run_turn_records_search_trace_without_network_or_model():
    state = s13.ConversationState(active_entities=["Anthropic Messages API"])

    result = s13.run_turn(
        "那它现在支持 strutured ouput 吗？",
        state,
        provider=s13.FakeSearchProvider(),
        client=None,
    )

    assert result["intent"].need_search is True
    assert "structured output" in result["query_plan"].corrected_question
    assert state.recent_searches
    assert "https://docs.anthropic.com" in result["answer"]
