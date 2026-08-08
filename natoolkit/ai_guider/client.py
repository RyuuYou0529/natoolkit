from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator

from .knowledge import TOPICS, ContextBundle
from .prompts import answer_messages, routing_messages
from .router import RouteDecision, parse_route


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class ClientConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClientConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    @classmethod
    def from_environment(cls) -> ClientConfig:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ClientConfigurationError(
                "DEEPSEEK_API_KEY is not configured. Set it in the environment "
                "and restart AI Guider."
            )
        base_url = os.environ.get(
            "NATOOLKIT_DEEPSEEK_BASE_URL", DEFAULT_BASE_URL
        ).strip()
        model = os.environ.get("NATOOLKIT_DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
        return cls(api_key=api_key, base_url=base_url, model=model)


class DeepSeekClient:
    def __init__(self, config: ClientConfig, sdk_client: Any | None = None) -> None:
        self.config = config
        self._client = sdk_client if sdk_client is not None else _create_sdk_client(config)

    @classmethod
    def from_environment(cls) -> DeepSeekClient:
        return cls(ClientConfig.from_environment())

    def route(
        self, question: str, history: list[tuple[str, str]]
    ) -> RouteDecision:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=routing_messages(question, history),
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        content = response.choices[0].message.content or ""
        return parse_route(content, frozenset(TOPICS))

    def stream_answer(
        self,
        question: str,
        history: list[tuple[str, str]],
        context: ContextBundle,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=answer_messages(question, history, context),
            max_tokens=1800,
            stream=True,
        )
        try:
            for event in stream:
                if not event.choices:
                    continue
                content = event.choices[0].delta.content
                if content:
                    yield content
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


def _create_sdk_client(config: ClientConfig):
    try:
        from openai import OpenAI

        return OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=45.0,
            max_retries=1,
        )
    except ImportError as exc:
        raise ClientConfigurationError(
            "The OpenAI client or its configured proxy transport is unavailable. "
            "Install the project dependencies, including httpx[socks] when a "
            "SOCKS proxy is configured."
        ) from exc
