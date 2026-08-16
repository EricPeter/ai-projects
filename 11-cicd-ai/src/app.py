from fastapi import FastAPI
from .feature_flags import is_enabled

app = FastAPI(title="AI Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/flags")
def flags():
    return {
        "new_model": is_enabled("ENABLE_NEW_MODEL"),
        "streaming": is_enabled("ENABLE_STREAMING", default=True),
        "caching": is_enabled("ENABLE_CACHING", default=True),
    }
