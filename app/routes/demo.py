from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
PAGE = Path(__file__).resolve().parents[1] / "static" / "demo.html"


@router.get("/demo", include_in_schema=False)
def demo() -> FileResponse:
    """Self-contained explorer page that uses the JSON API; a demo, not the product frontend."""
    return FileResponse(PAGE, media_type="text/html; charset=utf-8")
