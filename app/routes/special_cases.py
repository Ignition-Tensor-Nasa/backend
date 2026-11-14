"""API routes for bespoke SAFFIRE-III (NASA PSI-100) special cases and AI summary."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints
from typing import Annotated

from app.ai.saffire_summary import FocusArea, LanguageCode, summarize_saffire_case
from app.services.special_cases import list_special_cases, resolve_special_case, special_case_detail

router = APIRouter(prefix="/api/v1")


class SpecialCaseSummary(BaseModel):
    case_id: str
    sample_number: int
    title: str
    subtitle: str
    investigation: str
    capsule: str
    fuel: str
    o2_pct: float
    pressure_kpa: float
    recording_duration_s: float
    peak_temp_c: float
    peak_heat_flux_mw_cm2: float
    highlight_badge: str


class SpecialCasesListResponse(BaseModel):
    bespoke_notice: str
    items: list[SpecialCaseSummary]
    total: int


class AiSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus: FocusArea = "overview"
    language: LanguageCode = "pl"
    custom_prompt: Annotated[str, StringConstraints(max_length=1000)] | None = None


class SaffireAiSummaryBody(BaseModel):
    headline: str
    executive_summary: str
    key_findings: list[str]
    spacecraft_telemetry_insight: str
    methodological_caveats: list[str]


class AiSummaryResponse(BaseModel):
    case_id: str
    sample_number: int
    focus: FocusArea
    language: LanguageCode
    mode: Literal["live", "demo"]
    summary: SaffireAiSummaryBody


@router.get("/special-cases", response_model=SpecialCasesListResponse)
def get_special_cases() -> dict:
    return list_special_cases()


@router.get("/special-cases/{case_id}")
def get_special_case_detail(case_id: str) -> dict:
    payload = special_case_detail(case_id)
    if payload is None:
        raise HTTPException(404, f"Unknown special case: {case_id}")
    return payload


@router.post("/special-cases/{case_id}/ai-summary", response_model=AiSummaryResponse)
def post_special_case_ai_summary(case_id: str, request: AiSummaryRequest = AiSummaryRequest()) -> dict:
    resolved = resolve_special_case(case_id)
    if resolved is None:
        raise HTTPException(404, f"Unknown special case: {case_id}")
    canonical, sample_number = resolved
    detail = special_case_detail(canonical)
    assert detail is not None
    sample_payload = detail["samples"].get(str(sample_number))
    if not sample_payload:
        raise HTTPException(404, f"Sample {sample_number} missing for {canonical}")
    summary, mode = summarize_saffire_case(
        sample_payload,
        detail["spacecraft"],
        focus=request.focus,
        custom_prompt=request.custom_prompt,
        language=request.language,
    )
    return {
        "case_id": canonical,
        "sample_number": sample_number,
        "focus": request.focus,
        "language": request.language,
        "mode": mode,
        "summary": summary,
    }
