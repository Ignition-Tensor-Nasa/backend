import pytest

from app.ai import interpret as ai
from app.ai import provider as ai_provider

URL = "/api/v1/filters/interpret"


@pytest.fixture(autouse=True)
def live_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(ai_provider, "AI_CONFIG_PATH", tmp_path / ".env")
    monkeypatch.setenv("PSI_AI_MODE", "live")


def provider(answer):
    return lambda text, current, options: ({"searchable": True, **answer} if isinstance(answer, dict) else answer)


def never_called(text, current, options):
    pytest.fail("the provider must not be called")


def test_live_answer_is_validated_and_nulls_are_dropped(client, monkeypatch):
    seen = {}

    def fake(text, current, options):
        seen.update(text=text, current=current, options=options)
        return {"filters": {"investigation": None, "fuel": ["methane"],
                            "o2_pct": {"min": 30, "max": None}, "pressure_kpa": None,
                            "z_st": None},
                "warnings": [], "searchable": True}

    monkeypatch.setattr(ai, "ask_provider", fake)
    body = client.post(URL, json={"text": "  methane above 30% oxygen ",
                                  "current_filters": {"fuel": ["ethylene"]}}).json()
    assert body == {"filters": {"fuel": ["methane"], "o2_pct": {"min": 30.0}},
                    "warnings": [], "mode": "live", "searchable": True}
    assert seen["text"] == "methane above 30% oxygen"
    assert seen["current"] == {}
    assert seen["options"]["fuel"] == ["ethylene", "methane"]
    # The returned state is directly usable by search.
    assert client.post("/api/v1/cases/search", json={"filters": body["filters"]}).json()["total"] == 1


def test_unrecognised_prompt_is_not_a_search(client, monkeypatch):
    monkeypatch.setattr(ai, "ask_provider", provider(
        {"filters": {"investigation": None, "fuel": None, "o2_pct": None,
                     "pressure_kpa": None, "z_st": None}, "warnings": [], "searchable": False}))
    body = client.post(URL, json={"text": "something nice", "current_filters": {"fuel": ["methane"]}}).json()
    assert body["filters"] == {} and body["searchable"] is False
    assert body["warnings"] == ["No supported search could be created from this prompt."]


def test_explicit_all_cases_is_a_valid_unfiltered_search(client, monkeypatch):
    monkeypatch.setattr(ai, "ask_provider", provider(
        {"filters": {"investigation": None, "fuel": None, "o2_pct": None,
                     "pressure_kpa": None, "z_st": None}, "warnings": [], "searchable": True}))
    body = client.post(URL, json={"text": "show all cases", "current_filters": {"fuel": ["methane"]}}).json()
    assert body == {"filters": {}, "warnings": [], "mode": "live", "searchable": True}
    assert client.post("/api/v1/cases/search", json={"filters": body["filters"]}).json()["total"] == 6


def test_each_prompt_starts_fresh_and_zero_matches_are_empty(client, monkeypatch):
    seen = []

    def fake(text, current, options):
        seen.append(current)
        return {"filters": {"investigation": None, "fuel": ["methane"] if text == "first" else None,
                            "o2_pct": None, "pressure_kpa": None if text == "first" else {"min": 5000, "max": None},
                            "z_st": None}, "warnings": [], "searchable": True}

    monkeypatch.setattr(ai, "ask_provider", fake)
    first = client.post(URL, json={"text": "first"}).json()
    assert client.post("/api/v1/cases/search", json={"filters": first["filters"]}).json()["total"] == 3
    second = client.post(URL, json={"text": "second", "current_filters": first["filters"]}).json()
    assert second["filters"] == {"pressure_kpa": {"min": 5000.0}}
    assert client.post("/api/v1/cases/search", json={"filters": second["filters"]}).json() == {"items": [], "total": 0}
    assert seen == [{}, {}]


def test_unsearchable_response_with_filters_is_rejected(client, monkeypatch):
    monkeypatch.setattr(ai, "ask_provider", provider(
        {"filters": {"investigation": None, "fuel": ["methane"], "o2_pct": None,
                     "pressure_kpa": None, "z_st": None}, "warnings": [], "searchable": False}))
    assert client.post(URL, json={"text": "anything"}).status_code == 503


@pytest.mark.parametrize("answer", [
    {"filters": {"fuel": ["kerosene"]}, "warnings": []},           # invented category
    {"filters": {"o2_pct": {"min": 40, "max": 20}}, "warnings": []},  # reversed range
    {"filters": {"o2_pct": {"min": 140}}, "warnings": []},
    {"filters": {"gravity": ["0g"]}, "warnings": []},              # unsupported key
    {"filters": "pl.col('fuel') == 'methane'", "warnings": []},    # not a filter object
    {"warnings": []},
    {"filters": {}, "warnings": None},
    {"filters": {"investigation": None, "fuel": None, "o2_pct": None,
                 "pressure_kpa": None, "z_st": None}, "warnings": "warning"},
    {"filters": {"investigation": None, "fuel": None, "o2_pct": None,
                 "pressure_kpa": None, "z_st": None}, "warnings": [42]},
    {"filters": {"investigation": None, "fuel": None, "o2_pct": None,
                 "pressure_kpa": None, "z_st": None}, "warnings": [], "extra": 1},
    {"filters": {"investigation": None, "fuel": None, "o2_pct": {"min": "21", "max": None},
                 "pressure_kpa": None, "z_st": None}, "warnings": []},
    {"filters": {"investigation": None, "fuel": None, "o2_pct": None,
                 "pressure_kpa": None, "z_st": None}, "warnings": [], "searchable": "false"},
    "not json",
])
def test_invalid_provider_output_is_never_applied(client, monkeypatch, answer):
    monkeypatch.setattr(ai, "ask_provider", provider(answer))
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and "filters" not in response.json()


def test_provider_failure_leaves_manual_search_usable(client, monkeypatch):
    def down(text, current, options):
        raise ai.AIUnavailable("The AI provider could not be reached.")

    monkeypatch.setattr(ai, "ask_provider", down)
    response = client.post(URL, json={"text": "methane"})
    assert response.status_code == 503 and response.json()["detail"].startswith("The AI provider")
    assert client.post("/api/v1/cases/search", json={"filters": {"fuel": ["methane"]}}).status_code == 200


def test_demo_mode_answers_only_rehearsed_prompts(client, monkeypatch):
    monkeypatch.setenv("PSI_AI_MODE", "demo")
    monkeypatch.setattr(ai, "DEMO_PROMPTS", {"show methane at 21% oxygen":
                                             {"fuel": ["methane"], "o2_pct": {"min": 21, "max": 21}}})
    monkeypatch.setattr(ai, "ask_provider", never_called)
    hit = client.post(URL, json={"text": "  Show METHANE   at 21% oxygen. "}).json()
    assert hit == {"filters": {"fuel": ["methane"], "o2_pct": {"min": 21.0, "max": 21.0}},
                   "warnings": [], "mode": "demo", "searchable": True}
    miss = client.post(URL, json={"text": "show ethylene", "current_filters": {"fuel": ["methane"]}}).json()
    assert miss["filters"] == {} and miss["mode"] == "demo" and miss["warnings"]
    assert miss["searchable"] is False


def test_demo_state_that_does_not_fit_the_catalogue_is_not_applied(client, monkeypatch):
    monkeypatch.setenv("PSI_AI_MODE", "demo")
    monkeypatch.setattr(ai, "DEMO_PROMPTS", {"show butane": {"fuel": ["butane"]}})
    body = client.post(URL, json={"text": "show butane"}).json()
    assert body["filters"] == {} and body["warnings"] and body["searchable"] is False


def test_invalid_mode_is_not_silently_treated_as_live(client, monkeypatch):
    monkeypatch.setenv("PSI_AI_MODE", "demoo")
    monkeypatch.setattr(ai, "ask_provider", never_called)
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and "PSI_AI_MODE" in response.json()["detail"]


@pytest.mark.parametrize("payload", [
    {"text": "   "}, {"text": "x" * 4001}, {}, {"text": "ok", "extra": 1},
    {"text": "ok", "current_filters": {"o2_pct": {"min": 50, "max": 20}}},
])
def test_invalid_requests_are_422(client, monkeypatch, payload):
    monkeypatch.setattr(ai, "ask_provider", never_called)
    assert client.post(URL, json=payload).status_code == 422


def test_schema_offers_only_catalogue_values():
    schema = ai._schema({"investigation": ["INV-A"], "fuel": []})
    filters = schema["properties"]["filters"]
    assert filters["properties"]["investigation"]["anyOf"][0]["items"]["enum"] == ["INV-A"]
    assert "enum" not in filters["properties"]["fuel"]["anyOf"][0]["items"]
    assert filters["required"] == ["investigation", "fuel", "o2_pct", "pressure_kpa", "z_st"]
