from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from app.services.catalogue import Filters, search_cases

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
