import pytest

from app.ai import interpret as ai

URL = "/api/v1/filters/interpret"


def provider(answer):
    return lambda text, current, options: answer


def never_called(text, current, options):
    pytest.fail("the provider must not be called")


def test_live_answer_is_validated_and_nulls_are_dropped(client, monkeypatch):
    seen = {}

    def fake(text, current, options):
        seen.update(text=text, current=current, options=options)
        return {"filters": {"investigation": None, "fuel": ["methane"],
                            "o2_pct": {"min": 30, "max": None}, "pressure_kpa": None},
                "warnings": []}

    monkeypatch.setattr(ai, "ask_provider", fake)
    body = client.post(URL, json={"text": "  methane above 30% oxygen ",
                                  "current_filters": {"fuel": ["ethylene"]}}).json()
    assert body == {"filters": {"fuel": ["methane"], "o2_pct": {"min": 30.0}},
                    "warnings": [], "mode": "live"}
    assert seen["text"] == "methane above 30% oxygen"
    assert seen["current"] == {"fuel": ["ethylene"]}
    assert seen["options"]["fuel"] == ["ethylene", "methane"]
    # The returned state is directly usable by search.
    assert client.post("/api/v1/cases/search", json={"filters": body["filters"]}).json()["total"] == 1


def test_unchanged_answer_gets_a_warning(client, monkeypatch):
    monkeypatch.setattr(ai, "ask_provider", provider(
        {"filters": {"investigation": None, "fuel": ["methane"], "o2_pct": None,
                     "pressure_kpa": None}, "warnings": []}))
    body = client.post(URL, json={"text": "something nice", "current_filters": {"fuel": ["methane"]}}).json()
    assert body["filters"] == {"fuel": ["methane"]} and len(body["warnings"]) == 1


@pytest.mark.parametrize("answer", [
    {"filters": {"fuel": ["kerosene"]}, "warnings": []},           # invented category
    {"filters": {"o2_pct": {"min": 40, "max": 20}}, "warnings": []},  # reversed range
    {"filters": {"o2_pct": {"min": 140}}, "warnings": []},
    {"filters": {"gravity": ["0g"]}, "warnings": []},              # unsupported key
    {"filters": "pl.col('fuel') == 'methane'", "warnings": []},    # not a filter object
    {"warnings": []},
    {"filters": {}, "warnings": None},
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
                   "warnings": [], "mode": "demo"}
    miss = client.post(URL, json={"text": "show ethylene", "current_filters": {"fuel": ["methane"]}}).json()
    assert miss["filters"] == {"fuel": ["methane"]} and miss["mode"] == "demo" and miss["warnings"]


def test_demo_state_that_does_not_fit_the_catalogue_is_not_applied(client, monkeypatch):
    monkeypatch.setenv("PSI_AI_MODE", "demo")
    monkeypatch.setattr(ai, "DEMO_PROMPTS", {"show butane": {"fuel": ["butane"]}})
    body = client.post(URL, json={"text": "show butane"}).json()
    assert body["filters"] == {} and body["warnings"]


@pytest.mark.parametrize("payload", [
    {"text": "   "}, {"text": "x" * 4001}, {}, {"text": "ok", "extra": 1},
    {"text": "ok", "current_filters": {"fuel": ["kerosene"]}},
    {"text": "ok", "current_filters": {"o2_pct": {"min": 50, "max": 20}}},
])
def test_invalid_requests_are_422(client, monkeypatch, payload):
    monkeypatch.setattr(ai, "ask_provider", never_called)
    assert client.post(URL, json=payload).status_code == 422


def test_schema_offers_only_catalogue_values():
    schema = ai._schema({"investigation": ["INV-A"], "fuel": [], "o2_pct": None, "pressure_kpa": None})
    filters = schema["properties"]["filters"]
    assert filters["properties"]["investigation"]["anyOf"][0]["items"]["enum"] == ["INV-A"]
    assert "enum" not in filters["properties"]["fuel"]["anyOf"][0]["items"]
    assert filters["required"] == ["investigation", "fuel", "o2_pct", "pressure_kpa"]
