"""Provider adapters for schema-constrained JSON calls, including offline tools."""

import json
import os
from pathlib import Path
from typing import Protocol

from dotenv.parser import parse_stream
import httpx

AI_CONFIG_PATH = Path(__file__).resolve().parents[2] / ".env"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class AIUnavailable(RuntimeError):
    """The provider could not produce a usable answer; nothing should be changed."""


def ai_settings() -> dict:
    """Read the backend .env, then overlay process variables without logging either."""
    settings = {}
    try:
        with AI_CONFIG_PATH.open(encoding="utf-8") as stream:
            for binding in parse_stream(stream):
                if binding.error:
                    raise AIUnavailable("The AI assistant is not configured (invalid .env syntax).")
                if binding.key is not None:
                    settings[binding.key] = binding.value or ""
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError):
        raise AIUnavailable("The AI assistant is not configured (cannot read .env).") from None
    settings.update(os.environ)
    return settings


class JSONProvider(Protocol):
    """A provider returns a JSON object or raises a sanitized AIUnavailable."""

    def call_json(self, system: str, user: str, schema: dict, effort: str, timeout: float) -> dict: ...


def _parse_json(content: str) -> dict:
    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        raise AIUnavailable("The AI provider returned invalid JSON.") from None
    if not isinstance(result, dict):
        raise AIUnavailable("The AI provider returned invalid JSON.")
    return result


def _status_error(status: int) -> AIUnavailable:
    if status == 401:
        return AIUnavailable("The AI provider rejected the server's credentials.")
    if status == 403:
        return AIUnavailable("The AI provider rejected the server's credentials or model access.")
    if status == 429:
        return AIUnavailable("The AI provider is rate limiting requests; try again shortly.")
    if status in (400, 404):
        return AIUnavailable(f"The AI provider rejected the request (HTTP {status}); "
                             "check PSI_AI_MODEL and the provider configuration.")
    return AIUnavailable(f"The AI provider returned an error (HTTP {status}).")


class AnthropicProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def call_json(self, system: str, user: str, schema: dict, effort: str, timeout: float) -> dict:
        try:
            import anthropic
        except ImportError:
            raise AIUnavailable("The anthropic package is not installed on the server.") from None
        try:
            client = anthropic.Anthropic(api_key=self.api_key, timeout=timeout, max_retries=1)
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                # Preserve the API's existing refusal fallback, never a demo response.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIStatusError as error:
            raise _status_error(error.status_code) from None
        except anthropic.APIConnectionError:
            raise AIUnavailable("The AI provider could not be reached.") from None
        except Exception:
            raise AIUnavailable("The AI provider request failed.") from None
        try:
            if response.stop_reason != "end_turn":
                raise AIUnavailable("The AI provider did not finish its answer.")
            content = next(block.text for block in response.content if block.type == "text")
        except (AttributeError, StopIteration, TypeError):
            raise AIUnavailable("The AI provider returned an invalid response.") from None
        return _parse_json(content)


class OpenRouterProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def call_json(self, system: str, user: str, schema: dict, effort: str, timeout: float) -> dict:
        # OpenRouter's JSON-schema response format is supported only by some models.
        # Require a compatible upstream rather than silently dropping the schema.
        try:
            response = httpx.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "response", "strict": True, "schema": schema}},
                    "provider": {"require_parameters": True},
                },
                timeout=timeout,
            )
        except httpx.RequestError:
            raise AIUnavailable("The AI provider could not be reached.") from None
        except Exception:
            raise AIUnavailable("The AI provider request failed.") from None
        if not 200 <= response.status_code < 300:
            raise _status_error(response.status_code)
        try:
            payload = response.json()
            if payload.get("error"):
                raise AIUnavailable("The AI provider request failed.")
            choice = payload["choices"][0]
            if choice.get("error") or choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
                raise AIUnavailable("The AI provider did not finish its answer.")
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise AIUnavailable("The AI provider returned an invalid response.") from None
        return _parse_json(content)


PROVIDERS = {
    "anthropic": (AnthropicProvider, "ANTHROPIC_API_KEY"),
    "openrouter": (OpenRouterProvider, "OPENROUTER_API_KEY"),
}


def get_provider() -> JSONProvider:
    """Select an adapter using server-side configuration."""
    settings = ai_settings()
    name = settings.get("PSI_AI_PROVIDER", "").strip().lower()
    model = settings.get("PSI_AI_MODEL", "").strip()
    problems = []
    if name not in PROVIDERS:
        choices = ", ".join(PROVIDERS)
        problems.append(f"PSI_AI_PROVIDER (choose: {choices})")
    if not model:
        problems.append("PSI_AI_MODEL")
    if name in PROVIDERS:
        adapter, key_name = PROVIDERS[name]
        key = settings.get(key_name, "").strip()
        if not key:
            problems.append(key_name)
    else:
        key_names = list(dict.fromkeys(key_name for _, key_name in PROVIDERS.values()))
        if not any(settings.get(key_name, "").strip() for key_name in key_names):
            problems.append(f"the selected provider's API key ({', '.join(key_names)})")
    if problems:
        raise AIUnavailable("The AI assistant is not configured; set " + ", ".join(problems)
                            + " in the server's .env or environment.")
    return adapter(model, key)


def call_json(system: str, user: str, schema: dict, effort: str, timeout: float) -> dict:
    """Call the selected adapter; interpretation logic is independent of the provider."""
    return get_provider().call_json(system, user, schema, effort, timeout)
