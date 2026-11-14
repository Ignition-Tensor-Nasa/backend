from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.routes.cases import CaseSummary, Column, SearchRequest, SearchResponse
from app.routes.filters import FilterOptions
from app.services.catalogue import Filters
from app.services.projects import get_project, list_projects, overlay, scoped_search

router = APIRouter(prefix="/api/v1")


class ProjectSummary(BaseModel):
    project_id: str
    title: str
    mission: str
    summary: str
    measures: str
    source: str
    case_count: int
    fuels: list[str]


class View(BaseModel):
    kind: str            # dataset kind present in this project, e.g. "sensors"
    label: str
    default_y: list[str]
    channels: list[Column]
    cases: int


class ProjectDetail(ProjectSummary):
    filters: FilterOptions
    views: list[View]


class OverlayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    channel: str
    filters: Filters | None = None
    align: Literal["start", "ignition"] = "start"
    max_series: Annotated[int, Field(ge=1, le=200)] = 40
    max_points: Annotated[int, Field(ge=10, le=5000)] = 1500


class OverlaySeries(CaseSummary):
    z_st: float | None = None
    n_source: int
    sampled: bool
    points: list[list[float | None]]


class ChannelInfo(BaseModel):
    key: str
    label: str
    unit: str | None


class AxisInfo(BaseModel):
    label: str
    unit: str | None
    align: str


class OverlayResponse(BaseModel):
    project_id: str
    kind: str
    channel: ChannelInfo
    x: AxisInfo
    matched: int
    series: list[OverlaySeries]
    note: str


@router.get("/projects")
def projects() -> list[ProjectSummary]:
    return [ProjectSummary(**item) for item in list_projects()]


@router.get("/projects/{project_id}")
def project(project_id: str) -> ProjectDetail:
    found = get_project(project_id)
    if found is None:
        raise HTTPException(404, f"Unknown project: {project_id}")
    return ProjectDetail(**found)


@router.post("/projects/{project_id}/cases/search")
def project_search(project_id: str, request: SearchRequest = SearchRequest()) -> SearchResponse:
    try:
        items = scoped_search(project_id, request.filters or Filters())
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if items is None:
        raise HTTPException(404, f"Unknown project: {project_id}")
    return SearchResponse(items=items, total=len(items))


@router.post("/projects/{project_id}/overlay")
def project_overlay(project_id: str, request: OverlayRequest) -> OverlayResponse:
    """Runs of one project on one axis, for time-over-fuel/condition charts."""
    try:
        result = overlay(project_id, request.kind, request.channel, request.filters or Filters(),
                         request.align, request.max_series, request.max_points)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if result is None:
        raise HTTPException(404, f"Unknown project: {project_id}")
    return OverlayResponse(**result)
