#!/usr/bin/env python3
# Harness: search -- turn conversation into query plans and grounded answers.
"""
s13_search_harness.py - Search Harness

Search is where the harness first faces the outside information world:

    User turn
       |
       v
    +----------------------+      +------------------+
    | Conversation State   | ---> | Intent Classifier|
    +----------------------+      +---------+--------+
                                          |
                                          v
                                  +---------------+
                                  | Query Rewrite |
                                  +-------+-------+
                                          |
                                          v
                                  +---------------+
                                  | Search Tool   |
                                  +-------+-------+
                                          |
                                          v
                                  +---------------+
                                  | Evidence Pack |
                                  +-------+-------+
                                          |
                                          v
                                  +---------------+
                                  | Final Answer  |
                                  +---------------+

Key insight: "The model can decide and synthesize, but the harness keeps
the search trace, provider boundary, dedupe, and evidence shape reliable."
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
    readline.parse_and_bind("set enable-meta-keybindings on")
except ImportError:
    pass

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - exercised only when deps are absent.
    Anthropic = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only when deps are absent.
    load_dotenv = None  # type: ignore[assignment]


if load_dotenv:
    load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

MODEL = os.getenv("MODEL_ID", "")
DEFAULT_LANGUAGE = os.getenv("SEARCH_ANSWER_LANGUAGE", "zh")
CURRENT_DATE = dt.date.today().isoformat()
MAX_RECENT_TURNS = 12
MAX_RECENT_SEARCHES = 8

_CLIENT = None


@dataclass
class Intent:
    need_search: bool
    intent_type: str
    user_goal: str
    target_entities: list[str] = field(default_factory=list)
    time_sensitivity: str = "none"
    freshness_window: str = "any"
    answer_language: str = DEFAULT_LANGUAGE
    confidence: float = 0.0
    clarifying_question: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intent":
        return cls(
            need_search=bool(data.get("need_search", False)),
            intent_type=str(data.get("intent_type") or "unclear"),
            user_goal=str(data.get("user_goal") or ""),
            target_entities=_string_list(data.get("target_entities")),
            time_sensitivity=str(data.get("time_sensitivity") or "none"),
            freshness_window=str(data.get("freshness_window") or "any"),
            answer_language=str(data.get("answer_language") or DEFAULT_LANGUAGE),
            confidence=_safe_float(data.get("confidence"), default=0.0),
            clarifying_question=_optional_string(data.get("clarifying_question")),
        )


@dataclass
class QuerySpec:
    query: str
    purpose: str = "primary"
    freshness: str | None = None
    domains: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuerySpec":
        return cls(
            query=str(data.get("query") or "").strip(),
            purpose=str(data.get("purpose") or "primary"),
            freshness=_optional_string(data.get("freshness")),
            domains=_string_list(data.get("domains")),
        )


@dataclass
class QueryPlan:
    corrected_question: str
    standalone_question: str
    search_queries: list[QuerySpec] = field(default_factory=list)
    must_include_terms: list[str] = field(default_factory=list)
    must_exclude_terms: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryPlan":
        raw_queries = data.get("search_queries") or []
        queries = []
        if isinstance(raw_queries, list):
            for item in raw_queries:
                if isinstance(item, str):
                    queries.append(QuerySpec(query=item))
                elif isinstance(item, dict):
                    spec = QuerySpec.from_dict(item)
                    if spec.query:
                        queries.append(spec)
        return cls(
            corrected_question=str(data.get("corrected_question") or "").strip(),
            standalone_question=str(data.get("standalone_question") or "").strip(),
            search_queries=queries[:4],
            must_include_terms=_string_list(data.get("must_include_terms")),
            must_exclude_terms=_string_list(data.get("must_exclude_terms")),
        )


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published_at: str | None = None
    rank: int = 0
    query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationState:
    turns: list[dict[str, str]] = field(default_factory=list)
    topic_summary: str = ""
    active_entities: list[str] = field(default_factory=list)
    user_constraints: dict[str, Any] = field(
        default_factory=lambda: {
            "language": DEFAULT_LANGUAGE,
            "region": os.getenv("SEARCH_REGION", "US"),
            "freshness": None,
        }
    )
    recent_searches: list[dict[str, Any]] = field(default_factory=list)

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > MAX_RECENT_TURNS:
            self.turns = self.turns[-MAX_RECENT_TURNS:]

    def update_entities(self, entities: list[str]) -> None:
        for entity in entities:
            clean = entity.strip()
            if clean and clean not in self.active_entities:
                self.active_entities.append(clean)
        if len(self.active_entities) > 12:
            self.active_entities = self.active_entities[-12:]

    def remember_search(self, query_plan: QueryPlan, results: list[SearchResult]) -> None:
        self.recent_searches.append(
            {
                "at": int(time.time()),
                "standalone_question": query_plan.standalone_question,
                "queries": [spec.query for spec in query_plan.search_queries],
                "top_urls": [result.url for result in results[:5]],
            }
        )
        if len(self.recent_searches) > MAX_RECENT_SEARCHES:
            self.recent_searches = self.recent_searches[-MAX_RECENT_SEARCHES:]

    def context_for_model(self) -> dict[str, Any]:
        return {
            "today": CURRENT_DATE,
            "topic_summary": self.topic_summary,
            "active_entities": self.active_entities,
            "user_constraints": self.user_constraints,
            "recent_turns": self.turns[-6:],
            "recent_searches": self.recent_searches[-3:],
        }


class SearchProvider:
    name = "base"

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        freshness: str | None = None,
        domains: list[str] | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError


class FakeSearchProvider(SearchProvider):
    name = "fake"

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        freshness: str | None = None,
        domains: list[str] | None = None,
    ) -> list[SearchResult]:
        source_query = query.lower()
        if "structured output" in source_query or "tool" in source_query:
            results = [
                SearchResult(
                    title="Anthropic API documentation",
                    url="https://docs.anthropic.com/",
                    snippet="Official API documentation for messages, tools, and model capabilities.",
                    source="docs.anthropic.com",
                    rank=1,
                    query=query,
                ),
                SearchResult(
                    title="Anthropic release notes",
                    url="https://docs.anthropic.com/en/release-notes",
                    snippet="Release notes describing recent API and model changes.",
                    source="docs.anthropic.com",
                    rank=2,
                    query=query,
                ),
            ]
        else:
            slug = re.sub(r"[^a-z0-9]+", "-", source_query).strip("-") or "search"
            results = [
                SearchResult(
                    title=f"Example result for {query}",
                    url=f"https://example.com/search/{slug}",
                    snippet="Fake provider result. Configure SEARCH_PROVIDER=tavily for real web search.",
                    source="example.com",
                    rank=1,
                    query=query,
                ),
                SearchResult(
                    title=f"Background notes for {query}",
                    url=f"https://example.org/background/{slug}",
                    snippet="Second fake result used for deterministic tests and offline demos.",
                    source="example.org",
                    rank=2,
                    query=query,
                ),
            ]
        return results[:limit]


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str, endpoint: str | None = None):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.tavily.com/search"

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        freshness: str | None = None,
        domains: list[str] | None = None,
    ) -> list[SearchResult]:
        body: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(limit, 20)),
            "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        time_range = _tavily_time_range(freshness)
        if time_range:
            body["time_range"] = time_range
        if domains:
            body["include_domains"] = [domain.strip() for domain in domains if domain.strip()]
        topic = os.getenv("TAVILY_TOPIC")
        if topic:
            body["topic"] = topic.strip()
        country = os.getenv("TAVILY_COUNTRY")
        if country:
            body["country"] = country.strip()

        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                **({"X-Project-ID": os.getenv("TAVILY_PROJECT", "")} if os.getenv("TAVILY_PROJECT") else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Tavily search failed: {exc}") from exc

        raw_results = payload.get("results", [])
        results = []
        for idx, item in enumerate(raw_results[:limit], start=1):
            item_url = str(item.get("url") or "")
            results.append(
                SearchResult(
                    title=str(item.get("title") or item_url or "Untitled"),
                    url=item_url,
                    snippet=str(item.get("content") or ""),
                    source=_source_from_url(item_url),
                    published_at=_optional_string(item.get("published_date") or item.get("published_at")),
                    rank=idx,
                    query=query,
                )
            )
        return results


def get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if Anthropic is None or not MODEL:
        return None
    _CLIENT = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
    return _CLIENT


def classify_intent(
    user_message: str,
    state: ConversationState,
    *,
    client=None,
) -> Intent:
    prompt = {
        "task": "Classify the user's intent for a search-capable assistant.",
        "today": CURRENT_DATE,
        "conversation_state": state.context_for_model(),
        "user_message": user_message,
        "schema": {
            "need_search": "boolean",
            "intent_type": "current_fact | general_fact | how_to | comparison | recommendation | troubleshooting | local_context | unclear",
            "user_goal": "string",
            "target_entities": ["string"],
            "time_sensitivity": "none | low | high",
            "freshness_window": "day | week | month | year | any",
            "answer_language": "zh | en | ja | other",
            "confidence": "number 0..1",
            "clarifying_question": "string or null",
        },
        "rules": [
            "Return JSON only.",
            "Use need_search=true for latest/current/version/price/news/policy/schedule facts.",
            "Use need_search=false for local repository questions or stable general reasoning.",
            "Ask a clarifying question only when a missing constraint blocks a useful answer.",
        ],
    }
    fallback = heuristic_intent(user_message, state)
    data = _call_model_json(prompt, client=client, max_tokens=1000)
    if not data:
        return fallback
    intent = Intent.from_dict(data)
    if not intent.user_goal:
        intent.user_goal = fallback.user_goal
    return intent


def rewrite_query(
    user_message: str,
    state: ConversationState,
    intent: Intent,
    *,
    client=None,
) -> QueryPlan:
    prompt = {
        "task": "Correct and rewrite the user's message into standalone search queries.",
        "today": CURRENT_DATE,
        "conversation_state": state.context_for_model(),
        "intent": asdict(intent),
        "user_message": user_message,
        "schema": {
            "corrected_question": "string",
            "standalone_question": "string",
            "search_queries": [
                {
                    "query": "string",
                    "purpose": "primary | background | verification",
                    "freshness": "day | week | month | year | any | null",
                    "domains": ["string"],
                }
            ],
            "must_include_terms": ["string"],
            "must_exclude_terms": ["string"],
        },
        "rules": [
            "Return JSON only.",
            "Resolve pronouns and omitted entities from conversation_state.",
            "Keep the user's intent; do not over-optimize away key constraints.",
            "For high freshness, include the current year or recent wording in the query.",
            "Use at most four search queries.",
        ],
    }
    fallback = heuristic_query_plan(user_message, state, intent)
    data = _call_model_json(prompt, client=client, max_tokens=1200)
    if not data:
        return fallback
    plan = QueryPlan.from_dict(data)
    if not plan.corrected_question:
        plan.corrected_question = fallback.corrected_question
    if not plan.standalone_question:
        plan.standalone_question = fallback.standalone_question
    if intent.need_search and not plan.search_queries:
        plan.search_queries = fallback.search_queries
    return plan


def search_with_plan(
    query_plan: QueryPlan,
    provider: SearchProvider,
    *,
    limit_per_query: int = 5,
) -> tuple[list[SearchResult], list[str]]:
    notes = []
    collected: list[SearchResult] = []
    for spec in query_plan.search_queries:
        try:
            results = provider.search(
                spec.query,
                limit=limit_per_query,
                freshness=spec.freshness,
                domains=spec.domains,
            )
        except Exception as exc:
            notes.append(f"Search failed for {spec.query!r}: {exc}")
            continue
        if not results:
            notes.append(f"No results for {spec.query!r}.")
            continue
        for result in results:
            result.query = result.query or spec.query
            collected.append(result)
    normalized = dedupe_results(collected)
    if len(normalized) < len(collected):
        notes.append(f"Removed {len(collected) - len(normalized)} duplicate or excess-domain results.")
    return normalized, notes


def dedupe_results(results: list[SearchResult], *, max_per_domain: int = 2) -> list[SearchResult]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    domain_counts: dict[str, int] = {}
    deduped = []
    for result in results:
        normalized_url = normalize_url(result.url)
        title_key = _normalize_title(result.title)
        source = result.source or _source_from_url(result.url)
        if normalized_url and normalized_url in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if source:
            count = domain_counts.get(source, 0)
            if count >= max_per_domain:
                continue
            domain_counts[source] = count + 1
        if normalized_url:
            seen_urls.add(normalized_url)
        if title_key:
            seen_titles.add(title_key)
        deduped.append(result)
    for idx, result in enumerate(deduped, start=1):
        result.rank = idx
    return deduped


def build_evidence_pack(
    original_question: str,
    intent: Intent,
    query_plan: QueryPlan,
    results: list[SearchResult],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    search_notes = list(notes or [])
    if intent.need_search and not results:
        search_notes.append("No usable search results were collected.")
    return {
        "original_question": original_question,
        "intent": asdict(intent),
        "corrected_question": query_plan.corrected_question,
        "standalone_question": query_plan.standalone_question,
        "queries": [asdict(spec) for spec in query_plan.search_queries],
        "results": [result.to_dict() for result in results],
        "search_notes": search_notes,
    }


def synthesize_answer(
    user_message: str,
    state: ConversationState,
    intent: Intent,
    query_plan: QueryPlan,
    evidence_pack: dict[str, Any],
    *,
    client=None,
) -> str:
    if intent.clarifying_question:
        return intent.clarifying_question

    prompt = {
        "task": "Answer the user using the evidence pack when search was used.",
        "today": CURRENT_DATE,
        "conversation_state": state.context_for_model(),
        "user_message": user_message,
        "intent": asdict(intent),
        "query_plan": {
            "corrected_question": query_plan.corrected_question,
            "standalone_question": query_plan.standalone_question,
        },
        "evidence_pack": evidence_pack,
        "answer_rules": [
            "Answer in the user's language.",
            "Prioritize the user's actual question.",
            "Cite source URLs for factual claims from search results.",
            "Say when evidence is insufficient or conflicting.",
            "Separate direct evidence from inference.",
        ],
    }
    text = _call_model_text(prompt, client=client, max_tokens=1800)
    if text:
        return text
    if intent.need_search:
        return fallback_search_answer(query_plan, evidence_pack)
    return fallback_direct_answer(query_plan)


def run_turn(
    user_message: str,
    state: ConversationState,
    *,
    provider: SearchProvider | None = None,
    client=None,
    debug: bool = False,
) -> dict[str, Any]:
    provider = provider or make_search_provider()
    client = client if client is not None else get_client()

    state.add_turn("user", user_message)
    intent = classify_intent(user_message, state, client=client)
    state.update_entities(intent.target_entities)
    query_plan = rewrite_query(user_message, state, intent, client=client)

    results: list[SearchResult] = []
    notes: list[str] = []
    if intent.need_search and not intent.clarifying_question:
        results, notes = search_with_plan(query_plan, provider)
        state.remember_search(query_plan, results)

    evidence_pack = build_evidence_pack(user_message, intent, query_plan, results, notes)
    answer = synthesize_answer(user_message, state, intent, query_plan, evidence_pack, client=client)
    state.add_turn("assistant", answer)

    if debug:
        print(json.dumps({"intent": asdict(intent), "query_plan": asdict(query_plan)}, ensure_ascii=False, indent=2))

    return {
        "answer": answer,
        "intent": intent,
        "query_plan": query_plan,
        "results": results,
        "evidence_pack": evidence_pack,
    }


def make_search_provider() -> SearchProvider:
    provider = os.getenv("SEARCH_PROVIDER", "fake").strip().lower()
    tavily_key = os.getenv("TAVILY_API_KEY") or os.getenv("SEARCH_API_KEY")
    if provider == "tavily" or (provider == "fake" and tavily_key):
        api_key = tavily_key
        if api_key:
            return TavilySearchProvider(api_key, endpoint=os.getenv("SEARCH_ENDPOINT") or os.getenv("TAVILY_ENDPOINT"))
        print("[search] SEARCH_PROVIDER=tavily but no TAVILY_API_KEY; using fake provider.")
    return FakeSearchProvider()


def heuristic_intent(user_message: str, state: ConversationState) -> Intent:
    text = user_message.strip()
    lowered = text.lower()
    high_freshness_terms = [
        "latest",
        "current",
        "today",
        "yesterday",
        "price",
        "release",
        "version",
        "news",
        "202",
        "最新",
        "现在",
        "今天",
        "昨天",
        "价格",
        "版本",
        "发布",
        "新闻",
        "支持",
    ]
    local_terms = ["this repo", "this repository", "local file", "这个仓库", "这个文件", "本地"]
    asks_recommendation = any(term in lowered for term in ["recommend", "best", "比较", "推荐", "哪一个"])
    need_search = any(term in lowered for term in high_freshness_terms)
    if any(term in lowered for term in local_terms):
        need_search = False
    intent_type = "current_fact" if need_search else "general_fact"
    if asks_recommendation:
        intent_type = "recommendation"
        need_search = True
    entities = _guess_entities(text, state)
    return Intent(
        need_search=need_search,
        intent_type=intent_type,
        user_goal=text,
        target_entities=entities,
        time_sensitivity="high" if need_search else "none",
        freshness_window="month" if need_search else "any",
        answer_language=_guess_language(text, state),
        confidence=0.45,
    )


def heuristic_query_plan(user_message: str, state: ConversationState, intent: Intent) -> QueryPlan:
    corrected = correct_obvious_typos(user_message.strip())
    standalone = resolve_standalone_question(corrected, state)
    queries = []
    if intent.need_search:
        base_query = standalone
        if intent.time_sensitivity == "high" and not re.search(r"\b20\d{2}\b", base_query):
            base_query = f"{base_query} {dt.date.today().year}"
        queries.append(
            QuerySpec(
                query=base_query,
                purpose="primary",
                freshness=None if intent.freshness_window == "any" else intent.freshness_window,
            )
        )
        if intent.intent_type in {"comparison", "recommendation"} and len(intent.target_entities) >= 2:
            queries.append(QuerySpec(query=" ".join(intent.target_entities), purpose="verification"))
    return QueryPlan(
        corrected_question=corrected,
        standalone_question=standalone,
        search_queries=queries[:4],
        must_include_terms=intent.target_entities,
        must_exclude_terms=[],
    )


def correct_obvious_typos(text: str) -> str:
    replacements = {
        "strutured": "structured",
        "stuctured": "structured",
        "ouput": "output",
        "outpt": "output",
        "langchian": "LangChain",
        "langchainn": "LangChain",
        "claud": "Claude",
        "anthropicc": "Anthropic",
        "qurey": "query",
    }
    corrected = text
    for wrong, right in replacements.items():
        corrected = re.sub(rf"\b{re.escape(wrong)}\b", right, corrected, flags=re.IGNORECASE)
    return corrected


def resolve_standalone_question(text: str, state: ConversationState) -> str:
    stripped = text.strip()
    if not state.active_entities:
        return stripped
    entity = state.active_entities[-1]
    pronoun_patterns = [
        r"^(it|that|this|they|them)\b",
        r"^(它|他|她|这个|那个|那它|那这个|这|那)",
    ]
    if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in pronoun_patterns):
        return f"{entity}: {stripped}"
    return stripped


def fallback_search_answer(query_plan: QueryPlan, evidence_pack: dict[str, Any]) -> str:
    results = evidence_pack.get("results") or []
    notes = evidence_pack.get("search_notes") or []
    if not results:
        note = f" 搜索备注：{'；'.join(notes)}" if notes else ""
        return f"我没有拿到足够可靠的搜索结果来回答这个问题。{note}".strip()
    lines = [f"我把问题改写为：{query_plan.standalone_question}", "", "可用来源："]
    for result in results[:5]:
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        lines.append(f"- {title}: {url}")
        if snippet:
            lines.append(f"  {snippet}")
    if notes:
        lines.append("")
        lines.append("搜索备注：" + "；".join(notes))
    return "\n".join(lines)


def fallback_direct_answer(query_plan: QueryPlan) -> str:
    return (
        "这个问题看起来不需要联网搜索。"
        f"我会把它理解为：{query_plan.standalone_question}"
    )


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = (parsed.scheme or "https").lower()
    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    kept_pairs = [
        (key, value)
        for key, value in query_pairs
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ]
    query = urllib.parse.urlencode(sorted(kept_pairs))
    path = parsed.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit((scheme, hostname, path, query, ""))


def _call_model_json(prompt: dict[str, Any], *, client=None, max_tokens: int = 1000) -> dict[str, Any] | None:
    text = _call_model_text(prompt, client=client, max_tokens=max_tokens)
    if not text:
        return None
    return extract_json_object(text)


def _call_model_text(prompt: dict[str, Any], *, client=None, max_tokens: int = 1000) -> str:
    client = client if client is not None else get_client()
    if client is None or not MODEL:
        return ""
    try:
        response = client.messages.create(
            model=MODEL,
            system="You are a careful search agent harness component. Follow the requested output format exactly.",
            messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            max_tokens=max_tokens,
        )
    except Exception:
        return ""
    return _response_text(response)


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _response_text(response: Any) -> str:
    parts = []
    for block in getattr(response, "content", []) or []:
        if hasattr(block, "text"):
            parts.append(str(block.text))
        elif isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _source_from_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _tavily_time_range(freshness: str | None) -> str | None:
    mapping = {
        "day": "day",
        "week": "week",
        "month": "month",
        "year": "year",
    }
    return mapping.get((freshness or "").lower())


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _guess_entities(text: str, state: ConversationState) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b", text):
        entity = match.group(0).strip()
        if len(entity) > 1 and entity.lower() not in {"i", "the"}:
            entities.append(entity)
    quoted = re.findall(r"[`\"']([^`\"']{2,60})[`\"']", text)
    entities.extend(item.strip() for item in quoted)
    if not entities and _looks_pronominal(text) and state.active_entities:
        entities.append(state.active_entities[-1])
    return list(dict.fromkeys(entities))[:6]


def _looks_pronominal(text: str) -> bool:
    return bool(re.search(r"^\s*(it|that|this|它|这个|那个|那它|那这个)\b", text, flags=re.IGNORECASE))


def _guess_language(text: str, state: ConversationState) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    return str(state.user_constraints.get("language") or DEFAULT_LANGUAGE)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    state = ConversationState()
    provider = make_search_provider()
    debug = os.getenv("SEARCH_DEBUG", "").lower() in {"1", "true", "yes"}
    print(f"[search provider: {provider.name}]")
    while True:
        try:
            query = input("\033[36ms13 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        result = run_turn(query, state, provider=provider, debug=debug)
        print(result["answer"])
        print()
