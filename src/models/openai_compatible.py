"""OpenAI-compatible chat/completion client using only the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    text: str
    raw: dict[str, Any]


@dataclass
class OpenAICompatibleClient:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 64
    timeout_seconds: int = 120
    request_params: dict[str, Any] = field(default_factory=dict)

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.request_params)
        response = self._post_json("/chat/completions", payload)
        # Reasoning models (e.g. gpt-5-mini) can return null content when the token
        # budget is consumed by reasoning; coerce to "" so callers don't crash.
        text = response["choices"][0]["message"].get("content") or ""
        return GenerationResult(text=text, raw=response)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach model endpoint {url}: {exc.reason}") from exc

        return json.loads(body)


def client_from_env(
    *,
    base_url_env: str,
    model_env: str,
    api_key_env: str | None,
    temperature: float,
    max_tokens: int,
    request_params: dict[str, Any] | None = None,
) -> OpenAICompatibleClient:
    base_url = require_env(base_url_env)
    model = require_env(model_env)
    api_key = os.environ.get(api_key_env) if api_key_env else None
    return OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        request_params=request_params or {},
    )


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
