from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.services.catalogue import (
    Filters, case_detail, case_neighbors, dataset_csv_path, search_cases)

router = APIRouter(prefix="/api/v1")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: Filters | None = None


class CaseSummary(BaseModel):
    case_id: str
    title: str
    investigation: str
    fuel: str | None
    o2_pct: float | None
    pressure_kpa: float | None
    z_st: float | None


class Column(BaseModel):
    key: str
    label: str
    data_type: str
    unit: str | None


class Dataset(BaseModel):
    dataset_id: str
    label: str
    columns: list[Column]
    default_x: str | None
    default_y: list[str]


class Event(BaseModel):
    label: str
    time_s: float
    dataset_id: str  # the dataset whose time_s axis the event refers to
    source: str


class Evidence(BaseModel):
    document: str
    page: int
    quote: str


class ResearchContext(BaseModel):
    tags: list[Literal["prevention", "detection", "suppression", "unknown"]]
    rationale: str
    evidence: list[Evidence]
    disclaimer: str


class Explanation(BaseModel):
    text: str
    # template: fixed sentence pattern over source fields; curated: written by hand;
    # generated: constrained AI text (not used so far).
    origin: Literal["template", "curated", "generated"]


class CaseDetail(CaseSummary):
    source_label: str
    source_url: str | None
    note: str | None
    datasets: list[Dataset]
    events: list[Event]
    research_context: ResearchContext
    explanation: Explanation | None


class SearchResponse(BaseModel):
    items: list[CaseSummary]
    total: int


@router.post("/cases/search")
def search(request: SearchRequest = SearchRequest()) -> SearchResponse:
    try:
        items = search_cases(request.filters or Filters())
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return SearchResponse(items=items, total=len(items))


@router.get("/cases/{case_id}")
def detail(case_id: str) -> CaseDetail:
    case = case_detail(case_id)
    if case is None:
        raise HTTPException(404, f"Unknown case: {case_id}")
    return CaseDetail(**case)


class Neighbor(BaseModel):
    case: CaseSummary
    distance: float


class Neighbors(BaseModel):
    metric: str
    items: list[Neighbor]


@router.get("/cases/{case_id}/neighbors")
def neighbors(case_id: str) -> Neighbors:
    result = case_neighbors(case_id)
    if result is None:
        raise HTTPException(404, f"Unknown case: {case_id}")
    return Neighbors(**result)


@router.get("/datasets/{dataset_id}/csv")
def dataset_csv(dataset_id: str) -> FileResponse:
    path = dataset_csv_path(dataset_id)
    if path is None:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")
    return FileResponse(path, media_type="text/csv; charset=utf-8", filename=path.name)
