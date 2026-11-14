"""B06: translate a Polish/English description into the fixed filter state.

The provider only proposes a filter state. It is validated with the same rules
as manual search before it is returned; nothing the model writes is executed.
"""

import json
import os

from pydantic import ValidationError

from app.services.catalogue import CATEGORY_KEYS, RANGE_KEYS, Filters, check_categories, filter_options

MODEL = os.environ.get("PSI_AI_MODEL", "claude-opus-5-5")

# Rehearsed prompts -> reviewed complete filter states (PSI_AI_MODE=demo only).
DEMO_PROMPTS = {
    "pokaż testy z butanem powyżej 200 kpa i ponad 39% tlenu":
        {"fuel": ["butane"], "o2_pct": {"min": 39}, "pressure_kpa": {"min": 200}},
    "show butane tests above 200 kpa with more than 39% oxygen":
        {"fuel": ["butane"], "o2_pct": {"min": 39}, "pressure_kpa": {"min": 200}},
    "pokaż testy w atmosferze zbliżonej do ziemskiej: 20-22% tlenu, 95-105 kpa":
        {"o2_pct": {"min": 20, "max": 22}, "pressure_kpa": {"min": 95, "max": 105}},
    "show tests in an earth-like atmosphere: 20-22% oxygen, 95-105 kpa":
        {"o2_pct": {"min": 20, "max": 22}, "pressure_kpa": {"min": 95, "max": 105}},
    "pokaż testy przy niskim ciśnieniu, 45-55 kpa":
        {"pressure_kpa": {"min": 45, "max": 55}},
    "show low-pressure tests, 45-55 kpa":
        {"pressure_kpa": {"min": 45, "max": 55}},
}

SYSTEM = """You translate a user's description (Polish or English) of combustion experiments \
into filters for a small catalogue of NASA test cases. You only set filters; you never answer \
science questions.

Supported filters:
- investigation: list of names, only from the valid values below
- fuel: list of fuels, only from the valid values below
- o2_pct: range of oxygen concentration in percentage points (0-100)
- pressure_kpa: range of absolute pressure in kPa

Rules:
- Return the complete new filter state. Start from the current filters and change only what \
the user asks for; keep every unrelated constraint exactly as it is. Use null for a filter \
that is not set, and for one the user explicitly asks to remove.
- A range needs at least one of min/max; use null for an open side.
- Convert units when the user gives numbers: 1 psi = 6.894757293168 kPa, 1 atm = 101.325 kPa, \
1 bar = 100 kPa, an oxygen fraction of 0.21 = 21.
- Never invent a threshold. If the user is vague ("high oxygen", "low pressure") and gives no \
number, leave that filter as it is and add a warning asking for a number.
- If the user gives one approximate value ("about 34% oxygen"), use that value +/-2 for o2_pct \
or +/-10% for pressure_kpa, and add a warning stating the range you used.
- If the user asks for a fuel or investigation that is not in the valid values, or for \
anything these four filters cannot express (gravity, materials, outcomes, ...), do not guess: \
leave the filters as they are and add a warning saying what is not available.
- The known data bounds are only information; a range outside them is allowed.
- Write warnings in the user's language, one short sentence each. No warning when the request \
was applied exactly."""


class AIUnavailable(RuntimeError):
    """The assistant could not produce a usable filter state; manual search still works."""


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def _schema(options: dict) -> dict:
    number = _nullable({"type": "number"})
    bounds = _nullable({
        "type": "object", "properties": {"min": number, "max": number},
        "required": ["min", "max"], "additionalProperties": False,
    })
    properties = {}
    for key in CATEGORY_KEYS:
        item = {"type": "string", "enum": options[key]} if options[key] else {"type": "string"}
        properties[key] = _nullable({"type": "array", "items": item})
    properties |= {key: bounds for key in RANGE_KEYS}
    return {
        "type": "object",
        "properties": {
            "filters": {"type": "object", "properties": properties,
                        "required": list(properties), "additionalProperties": False},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["filters", "warnings"], "additionalProperties": False,
    }


def ask_provider(text: str, current: dict, options: dict) -> dict:
    """One Claude call returning {"filters": ..., "warnings": [...]} as parsed JSON."""
    try:
        import anthropic
    except ImportError as error:
        raise AIUnavailable("The anthropic package is not installed on the server.") from error
    try:
        # The key stays server-side: the SDK reads ANTHROPIC_API_KEY from the environment.
        client = anthropic.Anthropic(timeout=30.0, max_retries=1)
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _schema(options)}},
            # If a safety classifier declines, the API retries on its default fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps({
                "valid_values": {key: options[key] for key in CATEGORY_KEYS},
                "known_data_bounds": {key: options[key] for key in RANGE_KEYS},
                "current_filters": current,
                "user_description": text,
            }, ensure_ascii=False)}],
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


def _validated(raw) -> Filters:
    """Complete filter state from untrusted output, or ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("filters must be an object")
    state = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            value = {side: bound for side, bound in value.items() if bound is not None}
        if value not in (None, [], {}):
            state[key] = value
    filters = Filters.model_validate(state)  # ValidationError is a ValueError
    check_categories(filters)
    return filters


def interpret(text: str, current: Filters) -> tuple[Filters, list[str], str]:
    """(complete replacement filters, warnings, mode). Raises AIUnavailable."""
    mode = "demo" if os.environ.get("PSI_AI_MODE", "live").strip().lower() == "demo" else "live"
    if mode == "demo":
        reviewed = DEMO_PROMPTS.get(" ".join(text.casefold().replace("–", "-").split()).rstrip(".!?"))
        if reviewed is None:
            return current, ["Demo mode: this prompt is not one of the rehearsed examples, "
                             "so the filters were left unchanged."], mode
        try:
            return _validated(reviewed), [], mode
        except ValueError:
            return current, ["Demo mode: the rehearsed filters do not fit the current "
                             "catalogue, so the filters were left unchanged."], mode
    answer = ask_provider(text, current.model_dump(exclude_none=True), filter_options())
    try:
        filters = _validated(answer["filters"])
        warnings = [str(item) for item in answer["warnings"]]
    except (ValueError, KeyError, TypeError) as error:
        raise AIUnavailable("The AI provider proposed filters that are not valid; "
                            "nothing was changed.") from error
    if filters == current and not warnings:
        warnings = ["No supported filter change was recognised in the description."]
    return filters, warnings, mode
