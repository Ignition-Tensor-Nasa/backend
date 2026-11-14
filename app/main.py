from fastapi import FastAPI

from app.routes.hello import router as hello_router

app = FastAPI(title="Plot Data API")
app.include_router(hello_router)
