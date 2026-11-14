"""One schema-constrained Claude call, shared by the filter assistant and offline tools."""

import json
import os

MODEL = os.environ.get("PSI_AI_MODEL", "claude-opus-5-5")


class AIUnavailable(RuntimeError):
    """The provider could not produce a usable answer; nothing should be changed."""


def call_json(system: str, user: str, schema: dict, effort: str, timeout: float) -> dict:
    """Return the model's JSON answer for `schema`, or raise AIUnavailable."""
    try:
        import anthropic
    except ImportError as error:
        raise AIUnavailable("The anthropic package is not installed on the server.") from error
    try:
        # The key stays server-side: the SDK reads ANTHROPIC_API_KEY from the environment.
        client = anthropic.Anthropic(timeout=timeout, max_retries=1)
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            # If a safety classifier declines, the API retries on its default fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError as error:
        raise AIUnavailable("The AI provider rejected the server's credentials.") from error
    except anthropic.RateLimitError as error:
        raise AIUnavailable("The AI provider is rate limiting requests; try again shortly.") from error
    except anthropic.APIStatusError as error:
        raise AIUnavailable(f"The AI provider returned an error (HTTP {error.status_code}).") from error
    except anthropic.APIConnectionError as error:  # includes timeouts
        raise AIUnavailable("The AI provider could not be reached.") from error
    except Exception as error:  # e.g. no credentials configured at all
        raise AIUnavailable("The AI assistant is not configured (set ANTHROPIC_API_KEY).") from error
    if response.stop_reason != "end_turn":
        raise AIUnavailable(f"The AI provider did not finish its answer ({response.stop_reason}).")
    try:
        return json.loads(next(block.text for block in response.content if block.type == "text"))
    except (StopIteration, ValueError) as error:
        raise AIUnavailable("The AI provider returned invalid JSON.") from error
