from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.ai.interpret import interpret
from app.services.catalogue import Filters, check_categories, filter_options

router = APIRouter(prefix="/api/v1")


class Bounds(BaseModel):
    min: float
    max: float


class FilterOptions(BaseModel):
    investigation: list[str]
    fuel: list[str]
    o2_pct: Bounds | None
    pressure_kpa: Bounds | None
    z_st: Bounds | None


@router.get("/filters")
def filters() -> FilterOptions:
    return FilterOptions(**filter_options())


class InterpretRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    current_filters: Filters | None = None


class InterpretResponse(BaseModel):
    filters: Filters
    warnings: list[str]
    mode: Literal["live", "demo"]


@router.post("/filters/interpret", response_model_exclude_none=True)
def interpret_filters(request: InterpretRequest) -> InterpretResponse:
    current = request.current_filters or Filters()
    try:
        check_categories(current)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    filters, warnings, mode = interpret(request.text, current)
    return InterpretResponse(filters=filters, warnings=warnings, mode=mode)
