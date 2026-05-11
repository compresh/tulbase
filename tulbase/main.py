"""tulbase — FastAPI application entry point.

Run with:
    tulbase serve
or
    uvicorn tulbase.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tulbase.router import close_client
from tulbase.router import router as proxy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("[tulbase] Starting up...")

    # Lazy-load DeBERTa injection classifier in background (non-blocking).
    # Without the ML model the proxy still runs in regex-only injection mode.
    try:
        from tulbase.injection import start_ml_loading
        start_ml_loading()
    except Exception as e:
        print(f"[tulbase] ML injection model unavailable, regex-only mode: {e}")

    print("[tulbase] Ready at http://0.0.0.0:8000")
    yield

    # --- Shutdown ---
    print("[tulbase] Shutting down...")
    await close_client()


app = FastAPI(
    title="tulbase",
    description="Depth-aware context compression for LLM proxies",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(proxy_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}
