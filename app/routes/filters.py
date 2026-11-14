from fastapi import APIRouter
from pydantic import BaseModel

from app.services.catalogue import filter_options

router = APIRouter(prefix="/api/v1")


class Bounds(BaseModel):
    min: float
    max: float


class FilterOptions(BaseModel):
    investigation: list[str]
    fuel: list[str]
    o2_pct: Bounds | None
    pressure_kpa: Bounds | None


@router.get("/filters")
def filters() -> FilterOptions:
    return FilterOptions(**filter_options())
