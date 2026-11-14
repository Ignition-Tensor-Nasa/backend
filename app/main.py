import os

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes.cases import router as cases_router
from app.routes.hello import router as hello_router

app = FastAPI(title="PSI-Lens API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)
app.include_router(hello_router)
app.include_router(cases_router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    # The default handler echoes the input; a NaN/Infinity there cannot be JSON-encoded (500).
    errors = [{k: v for k, v in e.items() if k != "input"} for e in error.errors()]
    return JSONResponse({"detail": jsonable_encoder(errors)}, status_code=422)
