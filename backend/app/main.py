"""FastAPI entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db, wait_for_db
from .kernels.hw import ModelError
from .routers import backtests, fits, jobs, series

app = FastAPI(title="补货预测 Holt-Winters 工具", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    wait_for_db()
    init_db()


@app.exception_handler(ModelError)
async def model_error_handler(request, exc: ModelError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.include_router(series.router)
app.include_router(fits.router)
app.include_router(jobs.router)
app.include_router(backtests.router)
