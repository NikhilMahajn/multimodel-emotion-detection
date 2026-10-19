from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.text_routes import router as text_router

app = FastAPI(title="Multimodal Emotion Detection for Mental Health")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(text_router)


@app.get("/health")
def health():
    return {"status": "ok"}