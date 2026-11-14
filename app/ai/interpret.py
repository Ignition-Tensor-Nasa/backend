"""B06: translate a Polish/English description into the fixed filter state.

Each description starts a new search. The provider only proposes a filter state;
it is validated with the same rules as manual search before it is returned.
"""

import json

from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError

from app.ai.provider import AIUnavailable, ai_settings, call_json
from app.services.catalogue import CATEGORY_KEYS, RANGE_KEYS, Filters, check_categories, filter_options

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
- z_st: range of the stoichiometric mixture fraction Zst (0-1, no unit)

Rules:
- Treat each description as a new search. Start with no filters; do not reuse earlier filters. \
Return the complete filter state described by this request, using null for unset filters.
- Set searchable to true if the request specifies at least one supported filter, or explicitly \
asks to show all cases without constraints. Set searchable to false if no supported search can \
be made, and then set every filter to null and explain why in a warning.
- A range needs at least one of min/max; use null for an open side.
- Convert units when the user gives numbers: 1 psi = 6.894757293168 kPa, 1 atm = 101.325 kPa, \
1 bar = 100 kPa, an oxygen fraction of 0.21 = 21.
- Never invent a threshold. If the user is vague ("high oxygen", "low pressure") and gives no \
number, leave that filter unset and add a warning asking for a number.
- If the user gives one approximate value ("about 34% oxygen"), use that value +/-2 for o2_pct \
or +/-10% for pressure_kpa, and add a warning stating the range you used.
- If the user asks for a fuel or investigation that is not in the valid values, or for \
anything these filters cannot express (gravity, materials, outcomes, ...), do not guess: \
leave unsupported filters unset and add a warning saying what is not available.
- The known data bounds are only information; a range outside them is allowed.
- Write every warning in English, regardless of the user's language, one short sentence each. \
Keep catalogue names unchanged. No warning when the request was applied exactly."""


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
            "warnings": {"type": "array", "items": {"type": "string"},
                         "description": "Write all warnings in English, regardless of the user's language."},
            "searchable": {"type": "boolean"},
        },
        "required": ["filters", "warnings", "searchable"], "additionalProperties": False,
    }


def ask_provider(text: str, current: dict, options: dict) -> dict:
    """Ask the selected provider for a complete filter state and warnings."""
    return call_json(SYSTEM, json.dumps({
        "valid_values": {key: options[key] for key in CATEGORY_KEYS},
        "known_data_bounds": {key: options[key] for key in RANGE_KEYS},
        "current_filters": current,
        "user_description": text,
    }, ensure_ascii=False), _schema(options), effort="low", timeout=30.0)


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


def interpret(text: str) -> tuple[Filters, list[str], str, bool]:
    """Return new filters, warnings, mode and whether the prompt is searchable."""
    mode = ai_settings().get("PSI_AI_MODE", "live").strip().lower()
    if mode not in ("live", "demo"):
        raise AIUnavailable("The AI assistant is not configured (set PSI_AI_MODE to live or demo).")
    if mode == "demo":
        reviewed = DEMO_PROMPTS.get(" ".join(text.casefold().replace("–", "-").split()).rstrip(".!?"))
        if reviewed is None:
            return Filters(), ["Demo mode: this prompt is not one of the rehearsed examples."], mode, False
        try:
            return _validated(reviewed), [], mode, True
        except ValueError:
            return Filters(), ["Demo mode: the rehearsed filters do not fit the current catalogue."], mode, False
    options = filter_options()
    answer = ask_provider(text, {}, options)
    try:
        # Structured-output support differs between models; never trust it alone.
        Draft202012Validator(_schema(options)).validate(answer)
        filters = _validated(answer["filters"])
        warnings = answer["warnings"]
        searchable = answer["searchable"]
        if not searchable and filters != Filters():
            raise ValueError("an unsearchable response cannot contain filters")
    except (ValueError, KeyError, TypeError, SchemaValidationError):
        raise AIUnavailable("The AI provider proposed filters that are not valid; "
                            "no search was applied.") from None
    if not searchable and not warnings:
        warnings = ["No supported search could be created from this prompt."]
    return filters, warnings, mode, searchable
