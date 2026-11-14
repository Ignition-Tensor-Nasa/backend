from pathlib import Path

import pytest

from app.ai.provider import AIUnavailable
from data_prep import enrich_pdf

PAGES = ["FLEX-9999   Summary\nP = 772.131 mmHg\nYO2 = 0.210\nDodecane\n",
         "Campaign overview: tests were run between 0.5 and 5 atm.\nIgnition at 10:37:21\n"]


def field(name, value, unit, page, quote):
    return {"name": name, "value": value, "unit": unit, "page": page, "quote": quote}


def test_only_values_with_real_page_evidence_survive():
    answer = {"fields": [
        field("pressure", "772.131", "mmHg", 1, "P =  772.131 mmHg"),       # whitespace differs: ok
        field("o2_fraction", "0.170", None, 1, "YO2 = 0.170"),             # invented quote
        field("fuel", "Dodecane", None, 2, "Dodecane"),                    # wrong page
        field("burn_time", "35", "s", 9, "burn time 35 s"),                # page out of range
        field("n2_fraction", "0.79", None, 1, "YO2 = 0.210"),              # quote lacks the value
        field("he_fraction", None, None, None, None),                      # not stated: stays null
    ], "events": [
        {"label": "Ignition", "time_text": "10:37:21", "page": 2, "quote": "Ignition at 10:37:21"},
        {"label": "Extinction", "time_text": "10:38:00", "page": 2, "quote": "Extinction at 10:38:00"},
    ]}
    result = enrich_pdf.checked(answer, PAGES)
    kept = {name: value for name, value in result["fields"].items() if value}
    assert kept == {"pressure": {"value": "772.131", "unit": "mmHg", "page": 1,
                                 "quote": "P =  772.131 mmHg"}}
    assert set(result["fields"]) == set(enrich_pdf.FIELDS)  # every field present, unknown = None
    assert [event["label"] for event in result["events"]] == ["Ignition"]
    assert sorted(item.get("name", item.get("label")) for item in result["rejected"]) == [
        "Extinction", "burn_time", "fuel", "n2_fraction", "o2_fraction"]


def test_conflicts_are_flagged_never_applied():
    fields = {"o2_fraction": {"value": "0.210", "unit": None, "page": 1, "quote": "YO2 = 0.210"},
              "pressure": {"value": "772.131", "unit": "mmHg", "page": 1, "quote": "P = 772.131 mmHg"}}
    curated = {"o2_pct": 17.0, "pressure_kpa": 101.325}
    found = enrich_pdf.conflicts(fields, curated)
    assert len(found) == 1 and found[0].startswith("o2_pct: document says 0.210")
    assert curated == {"o2_pct": 17.0, "pressure_kpa": 101.325}
    assert enrich_pdf.conflicts(fields, {"o2_pct": None, "pressure_kpa": None}) == []  # unknown: no conflict
    assert enrich_pdf.conflicts(fields, None) == []


def test_enrich_sends_whole_pages_and_reports_a_proposal(monkeypatch, tmp_path):
    seen = {}

    def fake(system, user, schema, effort, timeout):
        seen["user"] = user
        return {"fields": [field("o2_fraction", "0.210", None, 1, "YO2 = 0.210")], "events": []}

    monkeypatch.setattr(enrich_pdf, "pdf_pages", lambda path: PAGES)
    monkeypatch.setattr(enrich_pdf, "call_json", fake)
    proposal = enrich_pdf.enrich("FLEX-9999", tmp_path / "x.pdf", {"o2_pct": 21.0, "pressure_kpa": None})
    assert "=== page 2 ===" in seen["user"] and "between 0.5 and 5 atm" in seen["user"]
    assert proposal["status"].startswith("proposed") and proposal["conflicts_with_curated"] == []
    assert proposal["fields"]["o2_fraction"]["page"] == 1 and proposal["fields"]["pressure"] is None


def test_provider_failure_propagates(monkeypatch, tmp_path):
    def down(*args, **kwargs):
        raise AIUnavailable("The AI provider could not be reached.")

    monkeypatch.setattr(enrich_pdf, "pdf_pages", lambda path: PAGES)
    monkeypatch.setattr(enrich_pdf, "call_json", down)
    with pytest.raises(AIUnavailable):
        enrich_pdf.enrich("FLEX-9999", tmp_path / "x.pdf", None)


REAL_PDF = Path(__file__).parents[2] / "data_nasa" / "CFI_PSI-39" / "pdf" / "FLEX-1213_Summary.pdf"


@pytest.mark.skipif(not REAL_PDF.is_file(), reason="no raw NASA data")
def test_real_pdf_pages_are_split_and_evidence_check_works_on_them():
    pages = enrich_pdf.pdf_pages(REAL_PDF)
    assert len(pages) >= 1 and "FLEX-1213" in pages[0]
    assert enrich_pdf._evidence_problem(pages, 1, "YO2 = 0.210", "0.210") is None
    assert enrich_pdf._evidence_problem(pages, 1, "YO2 = 0.170", "0.170") is not None
