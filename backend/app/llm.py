import json
from typing import Any

import httpx

from app.config import get_settings


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.openai_api_key)

    def chat(self, messages: list[dict[str, Any]], *, json_mode: bool = False, max_tokens: int | None = None) -> str:
        if not self.enabled:
            raise RuntimeError("Hosted chat is not configured")
        payload: dict[str, Any] = {"model": self.settings.chat_model, "messages": messages, "temperature": 0.2}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        response = httpx.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = httpx.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.chat_model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.1,
            },
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]

    def rewrite_queries(self, question: str) -> list[str]:
        if not self.enabled or not self.settings.enable_query_rewrite:
            return [question]
        try:
            raw = self.chat(
                [
                    {"role": "system", "content": "Rewrite the question as up to two concise document search queries. Return JSON: {\"queries\":[...]}. Preserve key entities and intent."},
                    {"role": "user", "content": question},
                ],
                json_mode=True,
            )
            parsed = json.loads(raw)
            queries = [str(q).strip() for q in parsed.get("queries", []) if str(q).strip()]
            return list(dict.fromkeys([question, *queries]))[:3]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return [question]
