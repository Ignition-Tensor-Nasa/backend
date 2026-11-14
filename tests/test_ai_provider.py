import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.ai import interpret as ai
from app.ai import provider

URL = "/api/v1/filters/interpret"
SECRET = "test-secret-must-not-appear"
ANSWER = {
    "filters": {"investigation": None, "fuel": ["methane"],
                "o2_pct": {"min": 30, "max": None}, "pressure_kpa": None, "z_st": None},
    "warnings": ["A test warning."],
}


@pytest.fixture(autouse=True)
def clean_config(monkeypatch, tmp_path):
    monkeypatch.setattr(provider, "AI_CONFIG_PATH", tmp_path / ".env")
    for name in ("PSI_AI_MODE", "PSI_AI_PROVIDER", "PSI_AI_MODEL", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def configure(monkeypatch, name):
    monkeypatch.setenv("PSI_AI_PROVIDER", name)
    monkeypatch.setenv("PSI_AI_MODEL", "example/model")
    monkeypatch.setenv("ANTHROPIC_API_KEY" if name == "anthropic" else "OPENROUTER_API_KEY", SECRET)


def mock_provider(monkeypatch, name, answer=ANSWER, *, finish=None):
    seen = {}
    if name == "anthropic":
        def constructor(**kwargs):
            seen["client"] = kwargs

            def create(**request):
                seen["request"] = request
                return SimpleNamespace(stop_reason=finish or "end_turn", content=[
                    SimpleNamespace(type="text", text=json.dumps(answer))])

            return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))

        monkeypatch.setattr(anthropic, "Anthropic", constructor)
    else:
        def post(url, **kwargs):
            seen.update(url=url, **kwargs)
            return httpx.Response(200, json={"choices": [{"finish_reason": finish or "stop",
                "message": {"content": json.dumps(answer)}}]})

        monkeypatch.setattr(provider.httpx, "post", post)
    return seen


@pytest.mark.parametrize("name", ["anthropic", "openrouter"])
def test_context_and_schema_reach_selected_provider(client, monkeypatch, name):
    configure(monkeypatch, name)
    seen = mock_provider(monkeypatch, name)
    response = client.post(URL, json={"text": "methane above 30% oxygen",
                                      "current_filters": {"fuel": ["ethylene"], "pressure_kpa": {"min": 50}}})
    assert response.status_code == 200
    assert response.json() == {"filters": {"fuel": ["methane"], "o2_pct": {"min": 30.0}},
                               "warnings": ["A test warning."], "mode": "live"}
    if name == "anthropic":
        request = seen["request"]
        system = request["system"]
        context = json.loads(request["messages"][0]["content"])
        schema = request["output_config"]["format"]["schema"]
        assert request["model"] == "example/model" and request["output_config"]["effort"] == "low"
        assert seen["client"] == {"api_key": SECRET, "timeout": 30.0, "max_retries": 1}
        assert request["fallbacks"] == "default"
    else:
        request = seen["json"]
        system = request["messages"][0]["content"]
        context = json.loads(request["messages"][1]["content"])
        schema = request["response_format"]["json_schema"]["schema"]
        assert seen["url"] == provider.OPENROUTER_URL and seen["timeout"] == 30.0
        assert seen["headers"]["Authorization"] == f"Bearer {SECRET}"
        assert request["model"] == "example/model" and request["provider"] == {"require_parameters": True}
        assert request["response_format"]["json_schema"]["strict"] is True
    assert context == {
        "valid_values": {"investigation": ["INV-A", "INV-B"], "fuel": ["ethylene", "methane"]},
        "known_data_bounds": {"o2_pct": {"min": 21.0, "max": 40.0},
                              "pressure_kpa": {"min": 50.0, "max": 200.0},
                              "z_st": {"min": 0.1, "max": 0.3}},
        "current_filters": {"fuel": ["ethylene"], "pressure_kpa": {"min": 50.0}},
        "user_description": "methane above 30% oxygen",
    }
    for snippet in ("Supported filters:", "z_st", "1 psi = 6.894757293168 kPa", "1 atm = 101.325 kPa",
                    "1 bar = 100 kPa", "0.21 = 21", "warning", "current filters"):
        assert snippet in system
    assert schema == ai._schema({**context["valid_values"], **context["known_data_bounds"]})


@pytest.mark.parametrize("name", ["anthropic", "openrouter"])
@pytest.mark.parametrize("bad_answer", [
    {**ANSWER, "warnings": "not a list"},
    {**ANSWER, "warnings": [1]},
    {**ANSWER, "filters": {**ANSWER["filters"], "fuel": ["invented"]}},
    {**ANSWER, "filters": {**ANSWER["filters"], "o2_pct": {"min": 80, "max": 20}}},
    {**ANSWER, "filters": {**ANSWER["filters"], "z_st": {"min": 2, "max": None}}},
    {**ANSWER, "filters": {**ANSWER["filters"], "o2_pct": {"min": "30", "max": None}}},
    {**ANSWER, "filters": {**ANSWER["filters"], "o2_pct": {"min": True, "max": None}}},
    {**ANSWER, "filters": {**ANSWER["filters"], "pressure_kpa": {"min": float("nan"), "max": None}}},
    {**ANSWER, "filters": {"fuel": ["methane"]}},
    {**ANSWER, "extra": "unexpected"},
])
def test_untrusted_response_is_rejected(client, monkeypatch, name, bad_answer):
    configure(monkeypatch, name)
    mock_provider(monkeypatch, name, bad_answer)
    response = client.post(URL, json={"text": "anything", "current_filters": {"fuel": ["ethylene"]}})
    assert response.status_code == 503 and "filters" not in response.json()


@pytest.mark.parametrize("settings, missing", [
    ({}, "PSI_AI_PROVIDER"),
    ({"PSI_AI_PROVIDER": "anthropic"}, "PSI_AI_MODEL"),
    ({"PSI_AI_PROVIDER": "anthropic", "PSI_AI_MODEL": "example/model"}, "ANTHROPIC_API_KEY"),
    ({"PSI_AI_PROVIDER": "openrouter"}, "PSI_AI_MODEL"),
    ({"PSI_AI_PROVIDER": "openrouter", "PSI_AI_MODEL": "example/model"}, "OPENROUTER_API_KEY"),
    ({"PSI_AI_PROVIDER": "unknown"}, "PSI_AI_PROVIDER"),
    ({"PSI_AI_PROVIDER": "anthropic", "PSI_AI_MODEL": "  "}, "PSI_AI_MODEL"),
])
def test_missing_configuration_does_not_break_manual_search(client, monkeypatch, settings, missing):
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and missing in response.json()["detail"]
    assert client.post("/api/v1/cases/search", json={"filters": {"fuel": ["methane"]}}).status_code == 200


@pytest.mark.parametrize("name", ["anthropic", "openrouter"])
def test_provider_and_model_are_loaded_from_dotenv(client, monkeypatch, name):
    key_name = "ANTHROPIC_API_KEY" if name == "anthropic" else "OPENROUTER_API_KEY"
    provider.AI_CONFIG_PATH.write_text(
        f"# Local settings\nPSI_AI_MODE=live\nPSI_AI_PROVIDER={name}\n"
        f"PSI_AI_MODEL='configured/model'\n{key_name}='{SECRET}'\n", encoding="utf-8")
    seen = mock_provider(monkeypatch, name)
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 200 and response.json()["mode"] == "live"
    request = seen["request"] if name == "anthropic" else seen["json"]
    assert request["model"] == "configured/model"
    if name == "anthropic":
        assert seen["client"]["api_key"] == SECRET
    else:
        assert seen["headers"]["Authorization"] == f"Bearer {SECRET}"


def test_environment_overrides_dotenv(client, monkeypatch):
    provider.AI_CONFIG_PATH.write_text(
        "PSI_AI_MODE=demo\nPSI_AI_PROVIDER=anthropic\nPSI_AI_MODEL=old-model\n"
        "ANTHROPIC_API_KEY=old-key\nOPENROUTER_API_KEY=old-key\n", encoding="utf-8")
    monkeypatch.setenv("PSI_AI_MODE", "live")
    configure(monkeypatch, "openrouter")
    seen = mock_provider(monkeypatch, "openrouter")
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 200 and response.json()["mode"] == "live"
    assert seen["json"]["model"] == "example/model"
    assert seen["headers"]["Authorization"] == f"Bearer {SECRET}"


def test_empty_environment_value_overrides_dotenv(client, monkeypatch):
    provider.AI_CONFIG_PATH.write_text(
        f"PSI_AI_PROVIDER=anthropic\nPSI_AI_MODEL=configured/model\nANTHROPIC_API_KEY={SECRET}\n",
        encoding="utf-8")
    monkeypatch.setenv("PSI_AI_MODEL", "")
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and "PSI_AI_MODEL" in response.json()["detail"]


def test_changes_to_dotenv_apply_to_later_calls(client, monkeypatch):
    path = provider.AI_CONFIG_PATH
    path.write_text(f"PSI_AI_PROVIDER=openrouter\nPSI_AI_MODEL=first\nOPENROUTER_API_KEY={SECRET}\n",
                    encoding="utf-8")
    seen = mock_provider(monkeypatch, "openrouter")
    assert client.post(URL, json={"text": "anything"}).status_code == 200
    assert seen["json"]["model"] == "first"
    path.write_text(f"PSI_AI_PROVIDER=openrouter\nPSI_AI_MODEL=second\nOPENROUTER_API_KEY={SECRET}\n",
                    encoding="utf-8")
    assert client.post(URL, json={"text": "anything"}).status_code == 200
    assert seen["json"]["model"] == "second"


def test_an_additional_adapter_needs_no_interpretation_changes(client, monkeypatch):
    seen = {}

    class ExampleProvider:
        def __init__(self, model, key):
            seen.update(model=model, key=key)

        def call_json(self, system, user, schema, effort, timeout):
            seen.update(system=system, user=json.loads(user), schema=schema)
            return ANSWER

    monkeypatch.setitem(provider.PROVIDERS, "example", (ExampleProvider, "EXAMPLE_API_KEY"))
    monkeypatch.setenv("PSI_AI_PROVIDER", "example")
    monkeypatch.setenv("PSI_AI_MODEL", "example-model")
    monkeypatch.setenv("EXAMPLE_API_KEY", SECRET)
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 200
    assert seen["model"] == "example-model" and seen["key"] == SECRET
    assert seen["user"]["user_description"] == "anything"


def test_dotenv_can_enable_demo_without_credentials(client, monkeypatch):
    provider.AI_CONFIG_PATH.write_text("PSI_AI_MODE=demo\n", encoding="utf-8")
    monkeypatch.setattr(ai, "ask_provider", lambda *args: pytest.fail("provider must not be called"))
    response = client.post(URL, json={"text": "not rehearsed", "current_filters": {"fuel": ["methane"]}})
    assert response.status_code == 200
    assert response.json()["mode"] == "demo" and response.json()["filters"] == {"fuel": ["methane"]}


def test_malformed_dotenv_is_sanitized_and_manual_search_works(client, caplog):
    provider.AI_CONFIG_PATH.write_text(f'PSI_AI_MODEL="{SECRET}\n', encoding="utf-8")
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and "invalid .env syntax" in response.json()["detail"]
    assert SECRET not in response.text and SECRET not in caplog.text
    assert client.post("/api/v1/cases/search", json={"filters": {}}).status_code == 200


@pytest.mark.parametrize("name", ["anthropic", "openrouter"])
def test_unfinished_answer_never_falls_back_to_demo(client, monkeypatch, name):
    configure(monkeypatch, name)
    mock_provider(monkeypatch, name, finish="length")
    response = client.post(URL, json={"text": "show low-pressure tests, 45-55 kpa"})
    assert response.status_code == 503 and "did not finish" in response.json()["detail"]


@pytest.mark.parametrize("status, message", [(401, "credentials"), (429, "rate limiting"), (500, "HTTP 500")])
def test_openrouter_http_error_is_sanitized(client, monkeypatch, caplog, status, message):
    configure(monkeypatch, "openrouter")
    monkeypatch.setattr(provider.httpx, "post", lambda *a, **kw: httpx.Response(status, text=SECRET))
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and message in response.json()["detail"]
    assert SECRET not in response.text and SECRET not in caplog.text
    assert client.post("/api/v1/cases/search", json={"filters": {}}).status_code == 200


@pytest.mark.parametrize("name", ["anthropic", "openrouter"])
def test_provider_exception_is_sanitized(client, monkeypatch, caplog, name):
    configure(monkeypatch, name)

    def fail(*args, **kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(anthropic if name == "anthropic" else provider.httpx,
                        "Anthropic" if name == "anthropic" else "post", fail)
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and "request failed" in response.json()["detail"]
    assert SECRET not in response.text and SECRET not in caplog.text


@pytest.mark.parametrize("status, message", [(401, "credentials"), (429, "rate limiting"), (503, "HTTP 503")])
def test_anthropic_http_error_is_sanitized(client, monkeypatch, caplog, status, message):
    configure(monkeypatch, "anthropic")

    def fail(**kwargs):
        request = httpx.Request("POST", "https://example.test")
        raise anthropic.APIStatusError(SECRET, response=httpx.Response(status, request=request), body=SECRET)

    monkeypatch.setattr(anthropic, "Anthropic", fail)
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and message in response.json()["detail"]
    assert SECRET not in response.text and SECRET not in caplog.text


@pytest.mark.parametrize("name", ["anthropic", "openrouter"])
def test_connection_failure_is_sanitized(client, monkeypatch, name):
    configure(monkeypatch, name)

    def fail(*args, **kwargs):
        request = httpx.Request("POST", "https://example.test")
        if name == "anthropic":
            raise anthropic.APIConnectionError(request=request)
        raise httpx.ConnectError(SECRET, request=request)

    monkeypatch.setattr(anthropic if name == "anthropic" else provider.httpx,
                        "Anthropic" if name == "anthropic" else "post", fail)
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and "could not be reached" in response.json()["detail"]
    assert SECRET not in response.text


@pytest.mark.parametrize("payload", [
    {"error": {"message": SECRET}},
    {"error": {"message": SECRET}, "choices": [{"finish_reason": "stop",
        "message": {"content": json.dumps(ANSWER)}}]},
    {"choices": [{"error": {"message": SECRET}, "finish_reason": "stop",
        "message": {"content": json.dumps(ANSWER)}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"refusal": SECRET, "content": "{}"}}]},
])
def test_openrouter_bad_payload_is_sanitized(client, monkeypatch, payload):
    configure(monkeypatch, "openrouter")
    monkeypatch.setattr(provider.httpx, "post", lambda *a, **kw: httpx.Response(200, json=payload))
    response = client.post(URL, json={"text": "anything"})
    assert response.status_code == 503 and SECRET not in response.text
