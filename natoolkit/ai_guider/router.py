from __future__ import annotations

import json
from dataclasses import dataclass


INTENTS = frozenset(
    {
        "usage",
        "troubleshooting",
        "algorithm",
        "implementation",
        "architecture",
        "out_of_scope",
    }
)
SOURCE_INTENTS = frozenset(
    {"troubleshooting", "algorithm", "implementation", "architecture"}
)


class RouteError(ValueError):
    pass


@dataclass(frozen=True)
class RouteDecision:
    in_scope: bool
    intent: str
    topics: tuple[str, ...]
    search_terms: tuple[str, ...]
    reason: str = ""

    @property
    def needs_source(self) -> bool:
        return self.in_scope and self.intent in SOURCE_INTENTS


def parse_route(raw: str, allowed_topics: set[str] | frozenset[str]) -> RouteDecision:
    try:
        payload = json.loads(_json_text(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        raise RouteError("The scope router returned invalid JSON.") from exc
    if not isinstance(payload, dict) or type(payload.get("in_scope")) is not bool:
        raise RouteError("The scope router did not return a Boolean in_scope value.")

    intent = payload.get("intent")
    if intent not in INTENTS:
        raise RouteError(f"Unknown router intent: {intent!r}")

    reason = str(payload.get("reason", "")).strip()[:500]
    if not payload["in_scope"] or intent == "out_of_scope":
        return RouteDecision(False, "out_of_scope", (), (), reason)

    topics = _string_list(payload.get("topics"), "topics", limit=3)
    unknown = set(topics) - set(allowed_topics)
    if not topics or unknown:
        detail = ", ".join(sorted(unknown)) if unknown else "no topic"
        raise RouteError(f"The scope router selected an invalid project topic: {detail}.")
    search_terms = _string_list(
        payload.get("search_terms", []), "search_terms", limit=8, required=False
    )
    return RouteDecision(True, intent, tuple(topics), tuple(search_terms), reason)


def _json_text(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return text


def _string_list(
    value, name: str, *, limit: int, required: bool = True
) -> list[str]:
    if not isinstance(value, list):
        raise RouteError(f"The scope router did not return a list for {name}.")
    result: list[str] = []
    for item in value[:limit]:
        if not isinstance(item, str) or not item.strip():
            raise RouteError(f"The scope router returned an invalid {name} entry.")
        result.append(item.strip()[:160])
    if required and not result:
        raise RouteError(f"The scope router returned an empty {name} list.")
    return result
