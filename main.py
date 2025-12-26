import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from detection import router as detect_router
from live_stream import router as live_router

app = FastAPI(
    title="Pothole Detection API",
    version="1.0.0"
)

# ==============================
# CREATE OUTPUT DIR IF MISSING
# ==============================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# SERVE OUTPUT FILES
# ==============================
app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)

# ==============================
# ROUTERS
# ==============================
app.include_router(detect_router, prefix="/detect", tags=["Detection"])
app.include_router(live_router, prefix="/live", tags=["Live Stream"])

# ==============================
# HEALTH CHECK
# ==============================
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "service": "Pothole Detection API"
    }

# ==============================
# RAILWAY ENTRYPOINT
# ==============================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
